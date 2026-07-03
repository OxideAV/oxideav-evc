//! §7.3.8.4 Main-profile (`sps_admvp_flag == 1`) inter coding-unit
//! mode-gating driver.
//!
//! The Baseline inter path in [`crate::slice_data`] handles the
//! `sps_admvp_flag == 0` toolset directly. The Main profile adds a layered
//! mode-gating tree (spec lines 2878-2911) that this module walks in spec
//! order, threading the per-tool syntax readers already in
//! [`crate::amvr_syntax`], [`crate::mmvd_syntax`] and
//! [`crate::affine_syntax`] into one structured [`InterCuModeDecision`].
//!
//! This module is a **syntax driver** — it consumes exactly the bins the
//! §7.3.8.4 tree prescribes and returns the decoded decision; the
//! downstream §8.5 MV-reconstruction (merge-list assembly, affine CPMV
//! derivation, MMVD offset application) is the consumer's job via the
//! already-landed [`crate::merge`] / [`crate::affine_cand`] /
//! [`crate::inter`] derivations.
//!
//! ## The `sps_admvp_flag == 1` merge branch (spec lines 2879-2911)
//!
//! ```text
//!   if( sps_amvr_flag )
//!       amvr_idx[ x0 ][ y0 ]
//!   else if( sps_admvp_flag == 1 ) {
//!       if( amvr_idx[ x0 ][ y0 ] == 0 )
//!           merge_mode_flag[ x0 ][ y0 ]
//!       if( merge_mode_flag[ x0 ][ y0 ] ) {
//!           if( sps_mmvd_flag )
//!               mmvd_flag[ x0 ][ y0 ]
//!           if( mmvd_flag[ x0 ][ y0 ] ) {                // MMVD group
//!               …mmvd_group_idx / merge_idx / distance / direction…
//!           } else {
//!               if( sps_affine_flag && log2CbW>=3 && log2CbH>=3 )
//!                   affine_flag[ x0 ][ y0 ]
//!               if( affine_flag[ x0 ][ y0 ] )
//!                   affine_merge_idx[ x0 ][ y0 ]
//!               else
//!                   merge_idx[ x0 ][ y0 ]
//!           }
//!       }
//!   }
//! ```
//!
//! `merge_mode_flag` is present only when `amvr_idx == 0`; when absent
//! it is **inferred 0** (§7.4.9.4 semantics: "when not present, inferred
//! equal to 0") — an AMVR-shifted CU is therefore always an explicit-MVD
//! CU, which is exactly why the §8.5.2.4 `sps_admvp_flag == 1` MVP
//! derivation indexes its candidate neighbour by `amvr_idx`. (Round 384
//! corrected the r381 inferred-1 reading.)
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::affine_syntax::AffineSyntaxStats;
use crate::amvr_syntax::InterModeGateStats;
use crate::cabac::CabacEngine;
use crate::cabac_init::MainCtxTable;
use crate::eipd_syntax::EipdCtx;
use crate::inter::{MotionVector, PRED_BI, PRED_L0, PRED_L1};
use crate::mmvd_syntax::{MmvdDecision, MmvdSyntaxStats};

/// SPS / PPS gates that select which §7.3.8.4 inter tools are present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterToolGates {
    /// `sps_amvr_flag` — adaptive motion-vector resolution present.
    pub sps_amvr_flag: bool,
    /// `sps_mmvd_flag` — merge-with-MVD present.
    pub sps_mmvd_flag: bool,
    /// `sps_affine_flag` — affine model based motion compensation present.
    pub sps_affine_flag: bool,
    /// `sps_admvp_flag` — advanced MV prediction (Main-profile merge
    /// list). Selects the cu_skip non-affine fall-through (`merge_idx` vs
    /// the Baseline `mvp_idx` pair).
    pub sps_admvp_flag: bool,
    /// `mmvd_group_enable_flag` — slice-header gate for `mmvd_group_idx`.
    pub mmvd_group_enable_flag: bool,
}

/// The resolved §7.3.8.4 merge-branch sub-decision. Selected once
/// `merge_mode_flag == 1` (or inferred 1) on the `sps_admvp_flag == 1`
/// path. Exactly one of the three variants fires per CU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeBranch {
    /// `mmvd_flag == 1` — merge with motion-vector difference. The carried
    /// [`MmvdDecision`] holds the group/merge/distance/direction indices.
    Mmvd(MmvdDecision),
    /// `affine_flag == 1` (size-gated) — affine merge with
    /// `affine_merge_idx`.
    AffineMerge { affine_merge_idx: u32 },
    /// The regular merge fall-through — `merge_idx` selects the candidate.
    Regular { merge_idx: u32 },
}

/// The resolved §7.3.8.4 inter-CU mode-gating decision for the
/// `sps_admvp_flag == 1` path, up to the point where the merge branch is
/// fully read. `amvr_idx` is the resolution selector (0 = 1/4-pel);
/// `merge_mode_flag` is the resolved (read-or-inferred) merge gate; `merge`
/// is `Some` exactly when `merge_mode_flag == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterCuModeDecision {
    /// `amvr_idx[ x0 ][ y0 ]` ∈ 0..=4 (0 when `sps_amvr_flag == 0`).
    pub amvr_idx: u32,
    /// The resolved `merge_mode_flag[ x0 ][ y0 ]`.
    pub merge_mode_flag: bool,
    /// The merge-branch sub-decision when `merge_mode_flag == 1`, else
    /// `None` (the explicit-AMVP path is handled by a separate driver).
    pub merge: Option<MergeBranch>,
}

/// Aggregate per-element bin counters across the three syntax readers the
/// driver invokes, so a caller can assert an exact bin budget end-to-end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterCuSyntaxStats {
    pub gate: InterModeGateStats,
    pub mmvd: MmvdSyntaxStats,
    pub affine: AffineSyntaxStats,
}

/// §7.3.8.4 line 2820/2903 — the `affine_flag` size gate inside a merge
/// branch is `sps_affine_flag && log2CbWidth >= 3 && log2CbHeight >= 3`.
pub fn affine_flag_present_in_merge(
    sps_affine_flag: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
) -> bool {
    sps_affine_flag && log2_cb_width >= 3 && log2_cb_height >= 3
}

/// Drive the §7.3.8.4 `sps_admvp_flag == 1` mode-gating tree for one
/// non-skip inter CU, up to (and including) the merge branch.
///
/// Spec order:
/// 1. `amvr_idx` if `sps_amvr_flag` (else inferred 0).
/// 2. `merge_mode_flag` if `amvr_idx == 0` (else inferred 1, line 5827).
/// 3. when `merge_mode_flag == 1`, the merge branch:
///    `mmvd_flag` → (MMVD group | affine-merge | regular merge_idx).
///
/// When `merge_mode_flag == 0` the returned decision carries `merge: None`
/// and the caller dispatches to the explicit-AMVP driver. `affine_merge`
/// is only attempted when [`affine_flag_present_in_merge`] holds.
pub fn read_inter_cu_mode(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    gates: InterToolGates,
    log2_cb_width: u32,
    log2_cb_height: u32,
    stats: &mut InterCuSyntaxStats,
) -> Result<InterCuModeDecision> {
    // Step 1 — amvr_idx (else inferred 0).
    let amvr_idx = if gates.sps_amvr_flag {
        crate::amvr_syntax::read_amvr_idx(eng, ctx, &mut stats.gate)?
    } else {
        0
    };

    // Step 2 — merge_mode_flag is read only when amvr_idx == 0; otherwise
    // it is absent and inferred 0 (§7.4.9.4) — an AMVR-shifted CU always
    // carries explicit MVDs (the §8.5.2.4 admvp MVP derivation indexes
    // its neighbour by amvr_idx).
    let merge_mode_flag = if amvr_idx == 0 {
        crate::amvr_syntax::read_merge_mode_flag(eng, ctx, &mut stats.gate)?
    } else {
        false
    };

    if !merge_mode_flag {
        return Ok(InterCuModeDecision {
            amvr_idx,
            merge_mode_flag,
            merge: None,
        });
    }

    // Step 3 — the merge branch (shared with the cu_skip path), here on
    // the sps_admvp_flag == 1 driver so the non-affine fall-through reads
    // merge_idx.
    let branch = read_merge_branch(eng, ctx, gates, true, log2_cb_width, log2_cb_height, stats)?;

    Ok(InterCuModeDecision {
        amvr_idx,
        merge_mode_flag,
        merge: Some(branch),
    })
}

/// The shared §7.3.8.4 merge-branch reader (spec lines 2811-2832 for the
/// cu_skip path; 2886-2910 for the non-skip `merge_mode_flag == 1` path):
///
/// ```text
///   if( sps_mmvd_flag ) mmvd_flag
///   if( mmvd_flag ) { …MMVD group… }
///   else {
///       if( sps_affine_flag && log2W>=3 && log2H>=3 ) affine_flag
///       if( affine_flag ) affine_merge_idx
///       else { sps_admvp_flag ? merge_idx : mvp_idx_l0 (+ mvp_idx_l1 for B) }
///   }
/// ```
///
/// On the `sps_admvp_flag == 1` paths the non-affine fall-through reads
/// `merge_idx`; the `sps_admvp_flag == 0` cu_skip path reads `mvp_idx`
/// instead, handled by [`read_cu_skip_main`] (which does not call this
/// helper for that case). `admvp` is passed through for clarity but this
/// helper only emits the `merge_idx` fall-through (admvp == true).
fn read_merge_branch(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    gates: InterToolGates,
    _admvp: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
    stats: &mut InterCuSyntaxStats,
) -> Result<MergeBranch> {
    let n_cb_w = 1u32 << log2_cb_width;
    let n_cb_h = 1u32 << log2_cb_height;
    // mmvd_flag is present iff sps_mmvd_flag; read_mmvd_group reads it and
    // returns flag=false (inferred 0) when sps_mmvd_flag is set but the
    // decoded bit is 0. When sps_mmvd_flag == 0 the element is absent
    // (inferred 0) and no bin is consumed.
    let mmvd = if gates.sps_mmvd_flag {
        crate::mmvd_syntax::read_mmvd_group(
            eng,
            ctx,
            gates.mmvd_group_enable_flag,
            log2_cb_width,
            log2_cb_height,
            &mut stats.mmvd,
        )?
    } else {
        MmvdDecision::default()
    };

    if mmvd.flag {
        return Ok(MergeBranch::Mmvd(mmvd));
    }

    if affine_flag_present_in_merge(gates.sps_affine_flag, log2_cb_width, log2_cb_height) {
        // affine_flag (FL cMax=1) — under the merge gate, ctxInc 0 (the
        // §9.3.4.2.4 neighbour term applies; here we leave it to the
        // single-context collapse like the sibling readers in the merge
        // path). Read it, then branch on its value.
        let mut affine_stats = AffineSyntaxStats::default();
        let affine_flag = crate::affine_syntax::read_affine_flag(
            eng,
            ctx,
            crate::affine_syntax::AffineFlagNeighbours::default(),
            &mut affine_stats,
        )?;
        if affine_flag {
            let affine_merge_idx =
                crate::affine_syntax::read_affine_merge_idx(eng, ctx, &mut affine_stats)?;
            stats.affine.flag_bins += affine_stats.flag_bins;
            stats.affine.merge_idx_bins += affine_stats.merge_idx_bins;
            return Ok(MergeBranch::AffineMerge { affine_merge_idx });
        }
        stats.affine.flag_bins += affine_stats.flag_bins;
    }

    // Regular merge_idx fall-through (sps_admvp_flag == 1).
    let merge_idx = crate::amvr_syntax::read_merge_idx(eng, ctx, n_cb_w, n_cb_h, &mut stats.gate)?;
    Ok(MergeBranch::Regular { merge_idx })
}

/// The resolved §7.3.8.4 cu_skip merge decision (spec lines 2811-2832).
/// A skip CU is always a merge CU: it carries either an MMVD group, an
/// affine merge, a regular `merge_idx` (admvp), or the Baseline
/// `mvp_idx_l0` (+ `mvp_idx_l1` for B) pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuSkipDecision {
    /// `mmvd_flag == 1` — MMVD merge.
    Mmvd(MmvdDecision),
    /// Affine merge with `affine_merge_idx`.
    AffineMerge { affine_merge_idx: u32 },
    /// Regular merge `merge_idx` (the `sps_admvp_flag == 1` fall-through).
    Merge { merge_idx: u32 },
    /// Baseline `mvp_idx` pair (the `sps_admvp_flag == 0` fall-through):
    /// `mvp_idx_l0` always, `mvp_idx_l1` for B slices (`None` for P).
    MvpIdx { l0: u32, l1: Option<u32> },
}

/// Drive the §7.3.8.4 cu_skip merge tree (spec lines 2811-2832) for one
/// skip CU on the Main profile. A skip CU reads no `amvr_idx` /
/// `merge_mode_flag` — it is implicitly a merge CU:
///
/// ```text
///   if( sps_mmvd_flag ) mmvd_flag
///   if( mmvd_flag ) { …MMVD group… }
///   else {
///       if( sps_affine_flag && log2W>=3 && log2H>=3 ) affine_flag
///       if( affine_flag ) affine_merge_idx
///       else { !sps_admvp_flag ? (mvp_idx_l0 [+ mvp_idx_l1 if B]) : merge_idx }
///   }
/// ```
///
/// `gates.sps_admvp_flag` selects between the `merge_idx` (admvp) and
/// `mvp_idx` (Baseline) fall-throughs; `slice_is_b` gates `mvp_idx_l1`.
pub fn read_cu_skip_main(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    gates: InterToolGates,
    slice_is_b: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
    stats: &mut InterCuSyntaxStats,
) -> Result<CuSkipDecision> {
    let sps_admvp_flag = gates.sps_admvp_flag;
    // mmvd / affine share the merge-branch reading; only the non-affine
    // fall-through differs by sps_admvp_flag.
    let mmvd = if gates.sps_mmvd_flag {
        crate::mmvd_syntax::read_mmvd_group(
            eng,
            ctx,
            gates.mmvd_group_enable_flag,
            log2_cb_width,
            log2_cb_height,
            &mut stats.mmvd,
        )?
    } else {
        MmvdDecision::default()
    };
    if mmvd.flag {
        return Ok(CuSkipDecision::Mmvd(mmvd));
    }

    if affine_flag_present_in_merge(gates.sps_affine_flag, log2_cb_width, log2_cb_height) {
        let mut affine_stats = AffineSyntaxStats::default();
        let affine_flag = crate::affine_syntax::read_affine_flag(
            eng,
            ctx,
            crate::affine_syntax::AffineFlagNeighbours::default(),
            &mut affine_stats,
        )?;
        if affine_flag {
            let affine_merge_idx =
                crate::affine_syntax::read_affine_merge_idx(eng, ctx, &mut affine_stats)?;
            stats.affine.flag_bins += affine_stats.flag_bins;
            stats.affine.merge_idx_bins += affine_stats.merge_idx_bins;
            return Ok(CuSkipDecision::AffineMerge { affine_merge_idx });
        }
        stats.affine.flag_bins += affine_stats.flag_bins;
    }

    if sps_admvp_flag {
        let n_cb_w = 1u32 << log2_cb_width;
        let n_cb_h = 1u32 << log2_cb_height;
        let merge_idx =
            crate::amvr_syntax::read_merge_idx(eng, ctx, n_cb_w, n_cb_h, &mut stats.gate)?;
        Ok(CuSkipDecision::Merge { merge_idx })
    } else {
        // Baseline mvp_idx pair (spec lines 2825-2828). TR cMax = 3,
        // ctxInc 0,1,2 (Table 48); collapses to (0,0) under Baseline
        // sps_cm_init_flag == 0.
        let l0 = read_mvp_idx(eng, ctx, stats)?;
        let l1 = if slice_is_b {
            Some(read_mvp_idx(eng, ctx, stats)?)
        } else {
            None
        };
        Ok(CuSkipDecision::MvpIdx { l0, l1 })
    }
}

/// §7.3.8.4 + §9.3.3 + Table 48 — read `mvp_idx_lX` (TR cMax = 3,
/// ctxInc 0,1,2). The skip-path Baseline predictor selector; under the
/// Baseline `sps_cm_init_flag == 0` collapse all bins are `(0,0)`.
fn read_mvp_idx(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut InterCuSyntaxStats,
) -> Result<u32> {
    let cm_init = ctx.is_cm_init();
    let table = MainCtxTable::MvpIdx.as_usize();
    let mut bins = 0u32;
    let v = eng.decode_tr_regular(3, 0, if cm_init { table } else { 0 }, |bin_idx| {
        bins += 1;
        if cm_init {
            bin_idx.min(2) as usize
        } else {
            0
        }
    })?;
    stats.gate.merge_idx_bins += bins;
    Ok(v)
}

/// One reference list's decoded explicit-AMVP parameters: the reference
/// index and the (already sign-applied) motion-vector difference in
/// 1/4-luma-pel units (pre-AMVR-shift; the caller applies eq. 145 with
/// `amvr_idx`). When the list is inactive for the CU's `inter_pred_idc`
/// the whole entry is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplicitListMv {
    /// `ref_idx_lX[ x0 ][ y0 ]` — 0 when absent (inferred, spec §7.4.x).
    pub ref_idx: u32,
    /// `MvdLX[ x0 ][ y0 ]` (eq. from `abs_mvd` / sign), 0 when the
    /// `bi_pred_idx` gate suppresses this list's MVD.
    pub mvd: MotionVector,
}

/// One reference list's explicit-**affine** parameters (spec lines
/// 2946-2980): the reference index, the eq.-867 predictor selector, the
/// MVD-present flag and the decoded per-control-point MVDs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplicitAffineList {
    /// `ref_idx_lX` (0 when absent — the TR `cMax = 0` inference).
    pub ref_idx: u32,
    /// `affine_mvp_flag_lX` ∈ {0, 1} — selects `cpMvpListLX[ flag ]`.
    pub mvp_flag: u32,
    /// `affine_mvd_flag_lX` — 1 ⇒ the per-vertex MVD group was present.
    pub mvd_flag: bool,
    /// `MvdCpLX[ vertex ]` for `vertex < numCpMv`; zero when
    /// `mvd_flag == 0` (the §7.4 absent inference).
    pub mvd_cp: [MotionVector; 3],
}

/// The resolved explicit-**affine** decision (`affine_flag == 1` inside
/// the `merge_mode_flag == 0` body, spec lines 2941-2980).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExplicitAffineDecision {
    /// `affine_mode_flag` — 0 ⇒ 4-param (`numCpMv = 2`), 1 ⇒ 6-param
    /// (`numCpMv = 3`); `vertexNum = 1 + affine_flag + affine_mode_flag`.
    pub affine_mode_flag: bool,
    /// L0 group (`None` when `inter_pred_idc == PRED_L1`).
    pub l0: Option<ExplicitAffineList>,
    /// L1 group (`None` when `inter_pred_idc == PRED_L0`).
    pub l1: Option<ExplicitAffineList>,
}

impl ExplicitAffineDecision {
    /// `numCpMv = 2 + affine_mode_flag` (eqs. 136/137).
    pub fn num_cp_mv(&self) -> u32 {
        2 + self.affine_mode_flag as u32
    }
}

/// The resolved §7.3.8.4 explicit-AMVP (`merge_mode_flag == 0`) decision
/// on the `sps_admvp_flag == 1` path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExplicitAmvpDecision {
    /// `inter_pred_idc` — PRED_L0 / PRED_L1 / PRED_BI (Table 8). For a
    /// P slice this is forced PRED_L0 (the element is absent).
    pub inter_pred_idc: u32,
    /// `bi_pred_idx` — the per-list MVD-present selector (0 when not
    /// PRED_BI; Table 71 semantics otherwise). Always 0 on the affine
    /// branch (the element is not read there).
    pub bi_pred_idx: u32,
    /// The explicit-affine sub-tree (spec lines 2941-2980): `Some` when
    /// `affine_flag == 1` was decoded; the translational `l0` / `l1`
    /// entries are then `None`.
    pub affine: Option<ExplicitAffineDecision>,
    /// L0 parameters (`None` when `inter_pred_idc == PRED_L1` or on the
    /// affine branch).
    pub l0: Option<ExplicitListMv>,
    /// L1 parameters (`None` when `inter_pred_idc == PRED_L0` or on the
    /// affine branch).
    pub l1: Option<ExplicitListMv>,
}

/// Per-element bin counters for the explicit-AMVP driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplicitAmvpStats {
    pub gate: InterModeGateStats,
    pub ref_idx_bins: u32,
    pub abs_mvd_bins: u32,
    pub mvd_sign_bins: u32,
    /// Affine sub-tree bins (`affine_flag` / `affine_mode_flag` /
    /// `affine_mvp_flag_lX` / `affine_mvd_flag_lX`).
    pub affine: AffineSyntaxStats,
}

/// §7.3.8.4 — read one `abs_mvd` (EG0 bypass) + optional `mvd_sign_flag`
/// (bypass) component, returning the signed value (spec lines 2918-2925).
fn read_signed_mvd_component(eng: &mut CabacEngine, stats: &mut ExplicitAmvpStats) -> Result<i32> {
    let abs = eng.decode_egk_bypass(0)?;
    stats.abs_mvd_bins += 1;
    if abs == 0 {
        return Ok(0);
    }
    let sign = eng.decode_bypass()?;
    stats.mvd_sign_bins += 1;
    Ok(if sign != 0 { -(abs as i32) } else { abs as i32 })
}

/// Read one list's `{ ref_idx?, abs_mvd[0]/sign, abs_mvd[1]/sign }` group
/// per the §7.3.8.4 admvp=1 explicit-AMVP body (spec lines 2989-3020).
///
/// * `ref_idx_lX` is present iff `num_ref_idx_active_minus1 > 0 &&
///   bi_pred_idx == 0` (TR cMax = num_ref_idx_active_minus1; ctxInc 0,1
///   then bypass — collapsed to `(0,0)` under Baseline).
/// * the MVD pair is present iff `bi_pred_idx != mvd_absent_value` (1 for
///   L0, 2 for L1).
fn read_explicit_list(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    num_ref_idx_active_minus1: u32,
    bi_pred_idx: u32,
    mvd_absent_value: u32,
    stats: &mut ExplicitAmvpStats,
) -> Result<ExplicitListMv> {
    let ref_idx = if num_ref_idx_active_minus1 > 0 && bi_pred_idx == 0 {
        let cm_init = ctx.is_cm_init();
        let table = MainCtxTable::RefIdx.as_usize();
        let mut bins = 0u32;
        let v = eng.decode_tr_regular(
            num_ref_idx_active_minus1,
            0,
            if cm_init { table } else { 0 },
            |bin_idx| {
                bins += 1;
                // ref_idx ctxInc is 0,1 for the first two bins, then bypass
                // (Table 9.3.4.2). Under the Baseline collapse all bins are
                // (0,0); the first two regular ctxIdx are 0 and 1.
                if cm_init {
                    bin_idx.min(1) as usize
                } else {
                    0
                }
            },
        )?;
        stats.ref_idx_bins += bins;
        v
    } else {
        0
    };

    let mvd = if bi_pred_idx != mvd_absent_value {
        let x = read_signed_mvd_component(eng, stats)?;
        let y = read_signed_mvd_component(eng, stats)?;
        MotionVector { x, y }
    } else {
        MotionVector::default()
    };

    Ok(ExplicitListMv { ref_idx, mvd })
}

/// §7.3.8.4 — the explicit-affine size/AMVR gate (spec line 2941):
/// `sps_affine_flag && log2CbWidth >= 4 && log2CbHeight >= 4 &&
/// amvr_idx == 0`.
pub fn affine_flag_present_in_explicit(
    sps_affine_flag: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
    amvr_idx: u32,
) -> bool {
    sps_affine_flag && log2_cb_width >= 4 && log2_cb_height >= 4 && amvr_idx == 0
}

/// Read one explicit-affine list group (spec lines 2946-2980):
/// `ref_idx_lX` (TR `cMax = num_ref_idx_active_minus1`, zero bins when 0)
/// → `affine_mvp_flag_lX` → `affine_mvd_flag_lX` → per-vertex
/// `abs_mvd`/sign pairs when the MVD flag is set.
fn read_explicit_affine_list(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    list: usize,
    num_ref_idx_active_minus1: u32,
    vertex_num: u32,
    stats: &mut ExplicitAmvpStats,
) -> Result<ExplicitAffineList> {
    let ref_idx = read_ref_idx(eng, ctx, num_ref_idx_active_minus1, stats)?;
    let flags =
        crate::affine_syntax::read_affine_list_flags_pub(eng, ctx, list, &mut stats.affine)?;
    let mut mvd_cp = [MotionVector::default(); 3];
    if flags.mvd_flag {
        for slot in mvd_cp.iter_mut().take(vertex_num as usize) {
            let x = read_signed_mvd_component(eng, stats)?;
            let y = read_signed_mvd_component(eng, stats)?;
            *slot = MotionVector { x, y };
        }
    }
    Ok(ExplicitAffineList {
        ref_idx,
        mvp_flag: flags.mvp_flag,
        mvd_flag: flags.mvd_flag,
        mvd_cp,
    })
}

/// Read `ref_idx_lX` (TR `cMax = num_ref_idx_active_minus1`, ctxInc 0,1
/// then bypass; collapsed to `(0, 0)` under Baseline). Zero bins when the
/// list holds a single active reference.
fn read_ref_idx(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    num_ref_idx_active_minus1: u32,
    stats: &mut ExplicitAmvpStats,
) -> Result<u32> {
    if num_ref_idx_active_minus1 == 0 {
        return Ok(0);
    }
    let cm_init = ctx.is_cm_init();
    let table = MainCtxTable::RefIdx.as_usize();
    let mut bins = 0u32;
    let v = eng.decode_tr_regular(
        num_ref_idx_active_minus1,
        0,
        if cm_init { table } else { 0 },
        |bin_idx| {
            bins += 1;
            if cm_init {
                bin_idx.min(1) as usize
            } else {
                0
            }
        },
    )?;
    stats.ref_idx_bins += bins;
    Ok(v)
}

/// Drive the §7.3.8.4 `sps_admvp_flag == 1` explicit-AMVP body (spec lines
/// 2912-3025): `inter_pred_idc` (B only), then the explicit-affine
/// sub-tree when its gate passes (`affine_flag` → `affine_mode_flag` →
/// per-list ref_idx + `affine_mvp_flag` + `affine_mvd_flag` + per-vertex
/// MVDs), else `bi_pred_idx` (PRED_BI only) + the per-list ref_idx + MVD
/// groups.
///
/// `slice_is_b` gates the `inter_pred_idc` read (absent ⇒ PRED_L0 for P
/// slices). `gates` / `amvr_idx` drive the spec-line-2941 affine gate.
#[allow(clippy::too_many_arguments)]
pub fn read_explicit_amvp(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    gates: InterToolGates,
    amvr_idx: u32,
    slice_is_b: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
    // `[ num_ref_idx_active_minus1[0], num_ref_idx_active_minus1[1] ]`.
    num_ref_idx_active_minus1: [u32; 2],
    stats: &mut ExplicitAmvpStats,
) -> Result<ExplicitAmvpDecision> {
    let num_ref_idx_active_minus1_l0 = num_ref_idx_active_minus1[0];
    let num_ref_idx_active_minus1_l1 = num_ref_idx_active_minus1[1];
    let n_cb_w = 1u32 << log2_cb_width;
    let n_cb_h = 1u32 << log2_cb_height;
    // inter_pred_idc — present only for B slices (spec line 2913);
    // P slices force PRED_L0.
    let inter_pred_idc = if slice_is_b {
        crate::amvr_syntax::read_inter_pred_idc(
            eng,
            ctx,
            true, // sps_admvp_flag == 1 on this driver's path.
            n_cb_w,
            n_cb_h,
            &mut stats.gate,
        )?
    } else {
        PRED_L0
    };

    // Explicit-affine sub-tree (spec lines 2941-2980).
    if affine_flag_present_in_explicit(
        gates.sps_affine_flag,
        log2_cb_width,
        log2_cb_height,
        amvr_idx,
    ) {
        let affine_flag = crate::affine_syntax::read_affine_flag(
            eng,
            ctx,
            crate::affine_syntax::AffineFlagNeighbours::default(),
            &mut stats.affine,
        )?;
        if affine_flag {
            let affine_mode_flag =
                crate::affine_syntax::read_affine_mode_flag(eng, ctx, &mut stats.affine)?;
            // vertexNum = 1 + affine_flag + affine_mode_flag.
            let vertex_num = 2 + affine_mode_flag as u32;
            let l0 = if inter_pred_idc != PRED_L1 {
                Some(read_explicit_affine_list(
                    eng,
                    ctx,
                    0,
                    num_ref_idx_active_minus1_l0,
                    vertex_num,
                    stats,
                )?)
            } else {
                None
            };
            let l1 = if inter_pred_idc != PRED_L0 {
                Some(read_explicit_affine_list(
                    eng,
                    ctx,
                    1,
                    num_ref_idx_active_minus1_l1,
                    vertex_num,
                    stats,
                )?)
            } else {
                None
            };
            return Ok(ExplicitAmvpDecision {
                inter_pred_idc,
                bi_pred_idx: 0,
                affine: Some(ExplicitAffineDecision {
                    affine_mode_flag,
                    l0,
                    l1,
                }),
                l0: None,
                l1: None,
            });
        }
    }

    // bi_pred_idx — present only when inter_pred_idc == PRED_BI.
    let bi_pred_idx = if inter_pred_idc == PRED_BI {
        crate::amvr_syntax::read_bi_pred_idx(eng, ctx, &mut stats.gate)?
    } else {
        0
    };

    // L0 group present when inter_pred_idc != PRED_L1.
    let l0 = if inter_pred_idc != PRED_L1 {
        Some(read_explicit_list(
            eng,
            ctx,
            num_ref_idx_active_minus1_l0,
            bi_pred_idx,
            1, // L0 MVD absent when bi_pred_idx == 1.
            stats,
        )?)
    } else {
        None
    };

    // L1 group present when inter_pred_idc != PRED_L0.
    let l1 = if inter_pred_idc != PRED_L0 {
        Some(read_explicit_list(
            eng,
            ctx,
            num_ref_idx_active_minus1_l1,
            bi_pred_idx,
            2, // L1 MVD absent when bi_pred_idx == 2.
            stats,
        )?)
    } else {
        None
    };

    Ok(ExplicitAmvpDecision {
        inter_pred_idc,
        bi_pred_idx,
        affine: None,
        l0,
        l1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacEncoder;

    /// Encode an explicit list of regular bins (all on ctx (0,0), matching
    /// the Baseline `sps_cm_init_flag == 0` collapse) and flush, padding
    /// the tail so the range coder has renormalisation headroom.
    fn bins(seq: &[u8]) -> Vec<u8> {
        let mut enc = CabacEncoder::new();
        for &b in seq {
            enc.encode_decision(0, 0, b);
        }
        enc.encode_terminate(true);
        let mut out = enc.finish();
        out.extend_from_slice(&[0xFF; 4]);
        out
    }

    fn gates_all() -> InterToolGates {
        InterToolGates {
            sps_amvr_flag: true,
            sps_mmvd_flag: true,
            sps_affine_flag: true,
            sps_admvp_flag: true,
            mmvd_group_enable_flag: false,
        }
    }

    /// A bin token: regular `(0,0)`-ctx decision or a bypass bin.
    enum Tok {
        Reg(u8),
        Byp(u8),
    }

    /// Encode a mixed regular/bypass token stream and flush.
    fn mixed(toks: &[Tok]) -> Vec<u8> {
        let mut enc = CabacEncoder::new();
        for t in toks {
            match t {
                Tok::Reg(b) => enc.encode_decision(0, 0, *b),
                Tok::Byp(b) => enc.encode_bypass(*b),
            }
        }
        enc.encode_terminate(true);
        let mut out = enc.finish();
        out.extend_from_slice(&[0xFF; 4]);
        out
    }

    /// affine_flag size gate: needs sps_affine_flag and both dims ≥ 3.
    #[test]
    fn affine_gate_dims() {
        assert!(affine_flag_present_in_merge(true, 3, 3));
        assert!(affine_flag_present_in_merge(true, 4, 6));
        assert!(!affine_flag_present_in_merge(true, 2, 4)); // width too small
        assert!(!affine_flag_present_in_merge(true, 4, 2)); // height too small
        assert!(!affine_flag_present_in_merge(false, 4, 4)); // tool off
    }

    /// amvr_idx == 0 + merge_mode_flag == 1 + mmvd_flag == 0 + affine off
    /// (small block) → regular merge fall-through reading merge_idx.
    /// Bin string: amvr "0", merge_mode "1", mmvd "0", merge_idx "0".
    #[test]
    fn merge_branch_regular_small_block() {
        let bs = bins(&[0, 1, 0, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        // 4×4 block (log2 2,2) → affine gate off → regular merge.
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates_all(), 2, 2, &mut stats)
            .unwrap();
        assert_eq!(d.amvr_idx, 0);
        assert!(d.merge_mode_flag);
        assert_eq!(d.merge, Some(MergeBranch::Regular { merge_idx: 0 }));
        assert_eq!(stats.gate.amvr_idx_bins, 1);
        assert_eq!(stats.gate.merge_mode_flag_bins, 1);
        assert_eq!(stats.mmvd.flag_bins, 1);
        assert_eq!(stats.gate.merge_idx_bins, 1);
    }

    /// mmvd_flag == 1 → MMVD branch. Bin string: amvr "0", merge_mode "1",
    /// mmvd_flag "1", then the MMVD index group (merge_idx "0",
    /// distance_idx "0", direction_idx 2 FL bins "1 0"). mmvd_group_idx
    /// absent (block 8×8 but mmvd_group_enable_flag off here). The
    /// direction value is left unasserted-exact, matching the sibling
    /// `mmvd_syntax` tests — the trailing FL bins ride the range coder's
    /// flush, so only `<= 3` is contractual.
    #[test]
    fn merge_branch_mmvd() {
        let bs = bins(&[0, 1, 1, 0, 0, 1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates_all(), 3, 3, &mut stats)
            .unwrap();
        assert_eq!(d.amvr_idx, 0);
        assert!(d.merge_mode_flag);
        match d.merge {
            Some(MergeBranch::Mmvd(m)) => {
                assert!(m.flag);
                assert_eq!(m.merge_idx, 0);
                assert_eq!(m.distance_idx, 0);
                assert!(m.direction_idx <= 3);
            }
            other => panic!("expected MMVD branch, got {other:?}"),
        }
        assert_eq!(stats.mmvd.flag_bins, 1);
        assert_eq!(stats.mmvd.direction_idx_bins, 2);
    }

    /// amvr_idx != 0 → merge_mode_flag absent and inferred 0 (§7.4.9.4
    /// "when not present, inferred equal to 0"): the CU defers to the
    /// explicit-AMVP driver and no merge-branch bin is consumed.
    /// Bin string: amvr "1 0" (value 1) only.
    #[test]
    fn amvr_nonzero_infers_merge_mode_zero() {
        let bs = bins(&[1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates_all(), 2, 2, &mut stats)
            .unwrap();
        assert_eq!(d.amvr_idx, 1);
        assert!(!d.merge_mode_flag);
        // merge_mode_flag was inferred — no bin consumed for it, and no
        // merge branch was read.
        assert_eq!(stats.gate.merge_mode_flag_bins, 0);
        assert_eq!(d.merge, None);
        assert_eq!(stats.mmvd.flag_bins, 0);
    }

    /// merge_mode_flag == 0 → explicit-AMVP path; driver returns
    /// merge: None and consumes no merge-branch bins.
    #[test]
    fn merge_mode_zero_defers_to_amvp() {
        let bs = bins(&[0, 0]); // amvr "0", merge_mode "0".
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates_all(), 3, 3, &mut stats)
            .unwrap();
        assert!(!d.merge_mode_flag);
        assert_eq!(d.merge, None);
        assert_eq!(stats.mmvd.flag_bins, 0);
    }

    /// Affine merge branch: mmvd_flag 0, affine_flag 1 (block ≥ 8×8),
    /// affine_merge_idx 0.
    #[test]
    fn merge_branch_affine() {
        let bs = bins(&[0, 1, 0, 1, 0]); // amvr 0, merge 1, mmvd 0, affine 1, aff_merge_idx 0
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates_all(), 3, 3, &mut stats)
            .unwrap();
        assert_eq!(
            d.merge,
            Some(MergeBranch::AffineMerge {
                affine_merge_idx: 0
            })
        );
        assert_eq!(stats.affine.flag_bins, 1);
        assert_eq!(stats.affine.merge_idx_bins, 1);
    }

    /// sps_amvr_flag == 0 → amvr_idx inferred 0, no bin read; the very
    /// first bin is merge_mode_flag.
    #[test]
    fn amvr_off_infers_zero() {
        let bs = bins(&[1, 0, 0]); // merge_mode 1, mmvd 0, merge_idx 0 (affine off)
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let gates = InterToolGates {
            sps_amvr_flag: false,
            ..gates_all()
        };
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates, 2, 2, &mut stats).unwrap();
        assert_eq!(d.amvr_idx, 0);
        assert_eq!(stats.gate.amvr_idx_bins, 0);
        assert!(d.merge_mode_flag);
    }

    /// Explicit-AMVP P-slice uni-pred: PRED_L0 forced (no inter_pred_idc
    /// bin), num_ref_idx=0 so no ref_idx → L0 present, L1 absent. The MVD
    /// is read as two EG0 bypass components; the test-only encoder's
    /// bypass-tail defer (documented in `cabac`) makes the trailing values
    /// non-contractual, so we assert the structural decisions + the L1
    /// absence + that exactly two abs_mvd components were consumed.
    #[test]
    fn explicit_amvp_p_unipred_structure() {
        let bs = mixed(&[Tok::Byp(0), Tok::Byp(0)]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            InterToolGates::default(),
            0,
            false, // P slice
            3,
            3,
            [0, 0],
            &mut stats,
        )
        .unwrap();
        assert_eq!(d.inter_pred_idc, PRED_L0);
        assert_eq!(d.bi_pred_idx, 0);
        assert!(d.l0.is_some());
        assert_eq!(d.l0.unwrap().ref_idx, 0); // num_ref_idx=0 → no ref_idx bin
        assert_eq!(d.l1, None);
        // inter_pred_idc absent for P → no bin.
        assert_eq!(stats.gate.inter_pred_idc_bins, 0);
        // two abs_mvd EG0 components consumed (one x, one y).
        assert_eq!(stats.abs_mvd_bins, 2);
    }

    /// Explicit-AMVP B-slice bi-pred: inter_pred_idc PRED_BI ("1 1",
    /// cMax 2 large block) then bi_pred_idx "0" (both lists' MVD present).
    /// These are regular-coded so decode exactly; the L0/L1 presence +
    /// the inter_pred_idc / bi_pred_idx bin counts are contractual.
    #[test]
    fn explicit_amvp_b_bipred_both_lists() {
        let toks = [
            Tok::Reg(1), // inter_pred_idc bin 0
            Tok::Reg(1), // inter_pred_idc bin 1 → value 2 = PRED_BI
            Tok::Reg(0), // bi_pred_idx "0" → 0 (both lists' MVD present)
            Tok::Byp(0), // L0 mvd_x abs
            Tok::Byp(0), // L0 mvd_y abs
            Tok::Byp(0), // L1 mvd_x abs
            Tok::Byp(0), // L1 mvd_y abs
        ];
        let bs = mixed(&toks);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            InterToolGates::default(),
            0,
            true, // B slice
            3,
            3,
            [0, 0],
            &mut stats,
        )
        .unwrap();
        assert_eq!(d.inter_pred_idc, PRED_BI);
        assert_eq!(d.bi_pred_idx, 0);
        assert!(d.l0.is_some());
        assert!(d.l1.is_some());
        assert_eq!(stats.gate.inter_pred_idc_bins, 2);
        assert_eq!(stats.gate.bi_pred_idx_bins, 1);
        // four abs_mvd EG0 components (two per list).
        assert_eq!(stats.abs_mvd_bins, 4);
    }

    /// bi_pred_idx == 1 suppresses the L1 MVD (Table 71): only the L0 MVD
    /// pair is read after the regular-coded "1 1 1 0" prefix
    /// (inter_pred_idc PRED_BI, bi_pred_idx 1). The bin budget proves the
    /// suppression — exactly two abs_mvd components, not four.
    #[test]
    fn explicit_amvp_bi_pred_idx_one_suppresses_l1_mvd() {
        let toks = [
            Tok::Reg(1), // inter_pred_idc bin 0
            Tok::Reg(1), // bin 1 → PRED_BI
            Tok::Reg(1), // bi_pred_idx bin 0
            Tok::Reg(0), // bi_pred_idx bin 1 → value 1
            Tok::Byp(0), // L0 mvd_x abs
            Tok::Byp(0), // L0 mvd_y abs
                         // L1 MVD suppressed (bi_pred_idx == 1).
        ];
        let bs = mixed(&toks);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            InterToolGates::default(),
            0,
            true,
            3,
            3,
            [0, 0],
            &mut stats,
        )
        .unwrap();
        assert_eq!(d.inter_pred_idc, PRED_BI);
        assert_eq!(d.bi_pred_idx, 1);
        assert!(d.l1.is_some()); // L1 active (ref present) but MVD suppressed
                                 // Only two abs_mvd components consumed (L0 only) — the L1 MVD pair
                                 // was skipped by the bi_pred_idx == 1 gate.
        assert_eq!(stats.abs_mvd_bins, 2);
    }

    /// ref_idx is present when num_ref_idx_active_minus1 > 0 && bi_pred_idx
    /// == 0. The regular-coded ref_idx bin decodes exactly; here a P-slice
    /// L0 with num_ref_idx_active_minus1=1 reads ref_idx "1" → ref_idx 1
    /// (cMax 1 TR saturates in one bin), then the MVD pair.
    #[test]
    fn explicit_amvp_ref_idx_present() {
        let toks = [
            Tok::Reg(1), // ref_idx_l0 "1" → 1 (cMax 1 saturates)
            Tok::Byp(0), // mvd_x abs
            Tok::Byp(0), // mvd_y abs
        ];
        let bs = mixed(&toks);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            InterToolGates::default(),
            0,
            false,
            3,
            3,
            [1, 0], // num_ref_idx_active_minus1_l0 = 1 → ref_idx present
            &mut stats,
        )
        .unwrap();
        assert_eq!(d.l0.unwrap().ref_idx, 1);
        assert_eq!(stats.ref_idx_bins, 1);
    }

    /// cu_skip admvp regular merge: mmvd 0, affine 0 (small block),
    /// merge_idx 0.
    #[test]
    fn cu_skip_admvp_regular_merge() {
        let bs = bins(&[0, 0]); // mmvd_flag 0, merge_idx 0 (affine off)
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_cu_skip_main(
            &mut eng,
            EipdCtx::new(false),
            gates_all(), // sps_admvp_flag == true
            false,
            2,
            2,
            &mut stats,
        )
        .unwrap();
        assert_eq!(d, CuSkipDecision::Merge { merge_idx: 0 });
        assert_eq!(stats.mmvd.flag_bins, 1);
        assert_eq!(stats.gate.merge_idx_bins, 1);
    }

    /// cu_skip Baseline (sps_admvp_flag == 0) P-slice: mmvd 0, no affine
    /// (small block), mvp_idx_l0 0 only (P → no mvp_idx_l1).
    #[test]
    fn cu_skip_baseline_p_mvp_idx() {
        let bs = bins(&[0, 0]); // mmvd 0, mvp_idx_l0 0
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let gates = InterToolGates {
            sps_affine_flag: false,
            sps_admvp_flag: false, // Baseline mvp_idx fall-through
            ..gates_all()
        };
        let d = read_cu_skip_main(
            &mut eng,
            EipdCtx::new(false),
            gates,
            false, // P slice
            3,
            3,
            &mut stats,
        )
        .unwrap();
        assert_eq!(d, CuSkipDecision::MvpIdx { l0: 0, l1: None });
    }

    /// cu_skip Baseline B-slice reads both mvp_idx_l0 and mvp_idx_l1.
    #[test]
    fn cu_skip_baseline_b_mvp_idx_pair() {
        let bs = bins(&[0, 0, 0]); // mmvd 0, mvp_idx_l0 0, mvp_idx_l1 0
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let gates = InterToolGates {
            sps_affine_flag: false,
            sps_admvp_flag: false,
            ..gates_all()
        };
        let d = read_cu_skip_main(
            &mut eng,
            EipdCtx::new(false),
            gates,
            true, // B slice → mvp_idx_l1 present
            3,
            3,
            &mut stats,
        )
        .unwrap();
        assert_eq!(d, CuSkipDecision::MvpIdx { l0: 0, l1: Some(0) });
    }

    /// cu_skip affine merge: mmvd 0, affine_flag 1 (≥8×8), affine_merge_idx 0.
    #[test]
    fn cu_skip_affine_merge() {
        let bs = bins(&[0, 1, 0]); // mmvd 0, affine 1, aff_merge_idx 0
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_cu_skip_main(
            &mut eng,
            EipdCtx::new(false),
            gates_all(),
            false,
            3,
            3,
            &mut stats,
        )
        .unwrap();
        assert_eq!(
            d,
            CuSkipDecision::AffineMerge {
                affine_merge_idx: 0
            }
        );
        assert_eq!(stats.affine.flag_bins, 1);
    }

    /// cu_skip MMVD: mmvd_flag 1 → MMVD branch (group absent, indices 0).
    #[test]
    fn cu_skip_mmvd() {
        let bs = bins(&[1, 0, 0, 1, 0]); // mmvd 1, merge 0, distance 0, direction "1 0"
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_cu_skip_main(
            &mut eng,
            EipdCtx::new(false),
            gates_all(),
            false,
            3,
            3,
            &mut stats,
        )
        .unwrap();
        match d {
            CuSkipDecision::Mmvd(m) => assert!(m.flag),
            other => panic!("expected MMVD skip, got {other:?}"),
        }
        assert_eq!(stats.mmvd.flag_bins, 1);
    }

    /// Round 384: the explicit-affine sub-tree on a P slice — gate passes
    /// (16×16, amvr 0, sps_affine on), affine_flag 1, affine_mode_flag 0
    /// (4-param, vertexNum 2), no ref_idx (single active), affine_mvp 0,
    /// affine_mvd_flag 1 → two per-vertex MVD pairs (EG0 bypass).
    #[test]
    fn round384_explicit_affine_p_slice_group() {
        let toks = [
            Tok::Reg(1), // affine_flag
            Tok::Reg(0), // affine_mode_flag → 4-param
            Tok::Reg(0), // affine_mvp_flag_l0
            Tok::Reg(1), // affine_mvd_flag_l0
            Tok::Byp(0), // v0 mvd_x abs = 0
            Tok::Byp(0), // v0 mvd_y abs = 0
            Tok::Byp(0), // v1 mvd_x abs = 0
            Tok::Byp(0), // v1 mvd_y abs = 0
        ];
        let bs = mixed(&toks);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let gates = InterToolGates {
            sps_affine_flag: true,
            sps_admvp_flag: true,
            ..Default::default()
        };
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            gates,
            0, // amvr_idx
            false,
            4,
            4,
            [0, 0],
            &mut stats,
        )
        .unwrap();
        let aff = d.affine.expect("affine branch taken");
        assert!(!aff.affine_mode_flag);
        assert_eq!(aff.num_cp_mv(), 2);
        let l0 = aff.l0.expect("L0 active on P");
        assert_eq!(l0.ref_idx, 0);
        assert_eq!(l0.mvp_flag, 0);
        assert!(l0.mvd_flag);
        assert_eq!(aff.l1, None);
        assert_eq!(d.l0, None, "translational lists empty on affine branch");
        assert_eq!(stats.affine.flag_bins, 1);
        assert_eq!(stats.affine.mode_flag_bins, 1);
        assert_eq!(stats.affine.mvp_flag_bins, 1);
        assert_eq!(stats.affine.mvd_flag_bins, 1);
        // vertexNum 2 → 4 abs_mvd components.
        assert_eq!(stats.abs_mvd_bins, 4);
        // bi_pred_idx never read on the affine branch.
        assert_eq!(stats.gate.bi_pred_idx_bins, 0);
    }

    /// Round 384: `affine_mvd_flag == 0` suppresses every per-vertex MVD
    /// (the §7.4 all-zero inference) — the bin budget proves it.
    #[test]
    fn round384_explicit_affine_mvd_flag_zero_suppresses_mvds() {
        let toks = [
            Tok::Reg(1), // affine_flag
            Tok::Reg(1), // affine_mode_flag → 6-param
            Tok::Reg(1), // affine_mvp_flag_l0 = 1
            Tok::Reg(0), // affine_mvd_flag_l0 = 0 → no MVDs
        ];
        let bs = mixed(&toks);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let gates = InterToolGates {
            sps_affine_flag: true,
            sps_admvp_flag: true,
            ..Default::default()
        };
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            gates,
            0,
            false,
            4,
            5,
            [0, 0],
            &mut stats,
        )
        .unwrap();
        let aff = d.affine.expect("affine");
        assert!(aff.affine_mode_flag);
        assert_eq!(aff.num_cp_mv(), 3);
        let l0 = aff.l0.unwrap();
        assert_eq!(l0.mvp_flag, 1);
        assert!(!l0.mvd_flag);
        assert_eq!(l0.mvd_cp, [MotionVector::default(); 3]);
        assert_eq!(stats.abs_mvd_bins, 0);
    }

    /// Round 384: the spec-line-2941 gate — small blocks (log2 < 4),
    /// non-zero amvr_idx, or sps_affine off skip the affine_flag bin and
    /// fall through to the translational body.
    #[test]
    fn round384_explicit_affine_gate() {
        assert!(affine_flag_present_in_explicit(true, 4, 4, 0));
        assert!(!affine_flag_present_in_explicit(true, 3, 4, 0)); // width
        assert!(!affine_flag_present_in_explicit(true, 4, 3, 0)); // height
        assert!(!affine_flag_present_in_explicit(true, 4, 4, 1)); // amvr
        assert!(!affine_flag_present_in_explicit(false, 4, 4, 0)); // tool

        // Gate off (8×8) → the first bin is bi_pred-body territory; on a
        // P slice with no ref_idx the two MVD pairs read immediately.
        let toks = [Tok::Byp(0), Tok::Byp(0)];
        let bs = mixed(&toks);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = ExplicitAmvpStats::default();
        let gates = InterToolGates {
            sps_affine_flag: true,
            sps_admvp_flag: true,
            ..Default::default()
        };
        let d = read_explicit_amvp(
            &mut eng,
            EipdCtx::new(false),
            gates,
            0,
            false,
            3,
            3,
            [0, 0],
            &mut stats,
        )
        .unwrap();
        assert_eq!(d.affine, None);
        assert!(d.l0.is_some());
        assert_eq!(stats.affine.flag_bins, 0);
    }
}
