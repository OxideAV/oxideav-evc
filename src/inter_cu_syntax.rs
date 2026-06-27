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
use crate::eipd_syntax::EipdCtx;
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
}
