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
//! `merge_mode_flag` is inferred 1 when `amvr_idx != 0` (spec line 5827):
//! a non-1/4-pel AMVR resolution forces explicit-MVD off, so AMVR-shifted
//! CUs can only be merge CUs. (The explicit-AMVP `merge_mode_flag == 0`
//! sub-tree is a separate driver, landed alongside this one.)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterToolGates {
    /// `sps_amvr_flag` — adaptive motion-vector resolution present.
    pub sps_amvr_flag: bool,
    /// `sps_mmvd_flag` — merge-with-MVD present.
    pub sps_mmvd_flag: bool,
    /// `sps_affine_flag` — affine model based motion compensation present.
    pub sps_affine_flag: bool,
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
    // `merge_idx`'s §9.3.3 cMax depends on the coding-block area in luma
    // samples, recovered from the log2 dimensions.
    let n_cb_w = 1u32 << log2_cb_width;
    let n_cb_h = 1u32 << log2_cb_height;
    // Step 1 — amvr_idx (else inferred 0).
    let amvr_idx = if gates.sps_amvr_flag {
        crate::amvr_syntax::read_amvr_idx(eng, ctx, &mut stats.gate)?
    } else {
        0
    };

    // Step 2 — merge_mode_flag is read only when amvr_idx == 0; otherwise
    // it is inferred 1 (spec line 5827 — a non-1/4-pel AMVR resolution
    // forces a merge CU).
    let merge_mode_flag = if amvr_idx == 0 {
        crate::amvr_syntax::read_merge_mode_flag(eng, ctx, &mut stats.gate)?
    } else {
        true
    };

    if !merge_mode_flag {
        return Ok(InterCuModeDecision {
            amvr_idx,
            merge_mode_flag,
            merge: None,
        });
    }

    // Step 3 — the merge branch.
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

    let branch = if mmvd.flag {
        MergeBranch::Mmvd(mmvd)
    } else if affine_flag_present_in_merge(gates.sps_affine_flag, log2_cb_width, log2_cb_height) {
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
            MergeBranch::AffineMerge { affine_merge_idx }
        } else {
            stats.affine.flag_bins += affine_stats.flag_bins;
            let merge_idx =
                crate::amvr_syntax::read_merge_idx(eng, ctx, n_cb_w, n_cb_h, &mut stats.gate)?;
            MergeBranch::Regular { merge_idx }
        }
    } else {
        // No affine eligibility — regular merge_idx fall-through.
        let merge_idx =
            crate::amvr_syntax::read_merge_idx(eng, ctx, n_cb_w, n_cb_h, &mut stats.gate)?;
        MergeBranch::Regular { merge_idx }
    };

    Ok(InterCuModeDecision {
        amvr_idx,
        merge_mode_flag,
        merge: Some(branch),
    })
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

/// The resolved §7.3.8.4 explicit-AMVP (`merge_mode_flag == 0`,
/// non-affine) decision on the `sps_admvp_flag == 1` path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExplicitAmvpDecision {
    /// `inter_pred_idc` — PRED_L0 / PRED_L1 / PRED_BI (Table 8). For a
    /// P slice this is forced PRED_L0 (the element is absent).
    pub inter_pred_idc: u32,
    /// `bi_pred_idx` — the per-list MVD-present selector (0 when not
    /// PRED_BI; Table 71 semantics otherwise).
    pub bi_pred_idx: u32,
    /// L0 parameters (`None` when `inter_pred_idc == PRED_L1`).
    pub l0: Option<ExplicitListMv>,
    /// L1 parameters (`None` when `inter_pred_idc == PRED_L0`).
    pub l1: Option<ExplicitListMv>,
}

/// Per-element bin counters for the explicit-AMVP driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExplicitAmvpStats {
    pub gate: InterModeGateStats,
    pub ref_idx_bins: u32,
    pub abs_mvd_bins: u32,
    pub mvd_sign_bins: u32,
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

/// Drive the §7.3.8.4 `sps_admvp_flag == 1` explicit-AMVP body (spec lines
/// 2912-3025, non-affine `else` branch): `inter_pred_idc` (B only),
/// `bi_pred_idx` (PRED_BI only), then the per-list ref_idx + MVD groups.
///
/// `slice_is_b` gates the `inter_pred_idc` read (absent ⇒ PRED_L0 for P
/// slices). The caller has already resolved `affine_flag == 0` (the
/// affine sub-tree is handled separately) and supplied
/// `num_ref_idx_active_minus1[0/1]`.
pub fn read_explicit_amvp(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
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

    /// amvr_idx != 0 forces merge_mode_flag inferred 1 (no bin read).
    /// Bin string: amvr "1 0" (value 1), then mmvd_flag "0", affine off
    /// (small block), merge_idx "0".
    #[test]
    fn amvr_nonzero_infers_merge_mode() {
        let bs = bins(&[1, 0, 0, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterCuSyntaxStats::default();
        let d = read_inter_cu_mode(&mut eng, EipdCtx::new(false), gates_all(), 2, 2, &mut stats)
            .unwrap();
        assert_eq!(d.amvr_idx, 1);
        assert!(d.merge_mode_flag);
        // merge_mode_flag was inferred — no bin consumed for it.
        assert_eq!(stats.gate.merge_mode_flag_bins, 0);
        assert_eq!(d.merge, Some(MergeBranch::Regular { merge_idx: 0 }));
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
}
