//! EVC Main-profile **affine** inter-syntax CABAC reads
//! (`sps_affine_flag == 1`) — the §7.3.8.4 `coding_unit()` affine group +
//! the §9.3.3 binarisations and the §9.3.4.2 ctxInc derivations.
//!
//! This module reads the affine syntax elements a §7.3.8.4 inter
//! `coding_unit()` carries when `affine_flag` is signalled. It is the
//! syntax-layer companion to [`crate::affine`] (the §8.5.3 derivation):
//! the decoded `affine_mode_flag` resolves `numCpMv` / `MotionModelIdc`
//! the geometric core consumes, and `affine_merge_idx` /
//! `affine_mvp_flag` / `affine_mvd_flag` route the merge vs AMVP path.
//!
//! ## Binarisation + contexts (Table 95 / Table 96)
//!
//! | element                | binarisation              | bin contexts                          |
//! |------------------------|---------------------------|---------------------------------------|
//! | `affine_flag`          | FL, cMax = 1              | §9.3.4.2.4 neighbour-derived (Table 96) |
//! | `affine_merge_idx`     | TR, cMax = 5, cRice = 0   | per-bin ctxInc 0,1,2,3,4 (Table 56)   |
//! | `affine_mode_flag`     | FL, cMax = 1              | ctxInc 0 (Table 57)                   |
//! | `affine_mvp_flag_lX`   | FL, cMax = 1              | ctxInc 0 (Table 58)                   |
//! | `affine_mvd_flag_l0`   | FL, cMax = 1              | ctxInc 0 (Table 59)                   |
//! | `affine_mvd_flag_l1`   | FL, cMax = 1              | ctxInc 0 (Table 60)                   |
//!
//! Under `sps_cm_init_flag == 0` (Baseline-style context collapse) every
//! regular bin routes to `(0, 0)`, mirroring [`crate::eipd_syntax`] and
//! [`crate::ats`]. The §9.3.4.2.4 neighbour-derived `affine_flag` ctxInc
//! still applies under `sps_cm_init_flag == 1`.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::cabac::CabacEngine;
use crate::cabac_init::MainCtxTable;
use crate::eipd_syntax::EipdCtx;

/// §9.3.4.2.4 (Table 96) neighbouring-block inputs for the `affine_flag`
/// ctxInc derivation: the three neighbour affine flags and their
/// §6.4.1 availabilities.
///
/// `cond{L,A,R}` is `affine_flag[ xNb{L,A,R} ][ yNb{L,A,R} ]` of the
/// left / above / right neighbour; `avail_{l,a,r}` is the corresponding
/// availability flag. `ctxInc = Min(Σ (cond && avail), numCtx − 1)` with
/// `numCtx = 2` (eq. 1438), so the result is 0 or 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineFlagNeighbours {
    pub cond_l: bool,
    pub avail_l: bool,
    pub cond_a: bool,
    pub avail_a: bool,
    pub cond_r: bool,
    pub avail_r: bool,
}

impl AffineFlagNeighbours {
    /// §9.3.4.2.4 eq. 1438 — `ctxInc = Min(condL&&availL + condA&&availA +
    /// condR&&availR, 1)`.
    pub fn ctx_inc(self) -> usize {
        let sum = (self.cond_l && self.avail_l) as usize
            + (self.cond_a && self.avail_a) as usize
            + (self.cond_r && self.avail_r) as usize;
        sum.min(1)
    }
}

/// The resolved affine inter-mode decision (the §7.3.8.4 outputs the
/// §8.5.3 derivation needs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineDecision {
    /// `affine_flag == 0` — not an affine CU; fall through to the regular
    /// inter path.
    NotAffine,
    /// `merge_mode_flag == 1` affine path: `affine_merge_idx`.
    Merge { merge_idx: u32 },
    /// AMVP affine path: `affine_mode_flag` (0 = 4-param / numCpMv 2,
    /// 1 = 6-param / numCpMv 3) plus the per-list predictor + MVD-present
    /// flags. List entries are `None` when that list is inactive for the
    /// CU's `inter_pred_idc`.
    Amvp {
        affine_mode_flag: bool,
        l0: Option<AffineListFlags>,
        l1: Option<AffineListFlags>,
    },
}

impl AffineDecision {
    /// `numCpMv` = `MotionModelIdc + 1` (eqs. 136/137): merge inherits its
    /// CP count from the candidate (caller-supplied; `None` here), AMVP
    /// derives it from `affine_mode_flag` (2 for 4-param, 3 for 6-param).
    pub fn num_cp_mv(self) -> Option<u32> {
        match self {
            AffineDecision::Amvp {
                affine_mode_flag, ..
            } => Some(2 + affine_mode_flag as u32),
            _ => None,
        }
    }
}

/// The per-list affine AMVP flags: `affine_mvp_flag_lX` (the predictor
/// index, FL cMax = 1) and `affine_mvd_flag_lX` (whether the per-vertex
/// MVDs are present).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineListFlags {
    /// `affine_mvp_flag_lX` ∈ {0, 1} — selects `cpMvpListLX[ flag ]`
    /// (eq. 867).
    pub mvp_flag: u32,
    /// `affine_mvd_flag_lX` — 1 ⇒ the per-vertex `abs_mvd`/sign group is
    /// present; 0 ⇒ all `mvdCpLX[ cpIdx ] = 0` (the §7.4.x inference).
    pub mvd_flag: bool,
}

/// Per-element bin counters (one per syntax element) for exact bin-budget
/// assertions in tests, mirroring [`crate::ats::AtsSyntaxStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineSyntaxStats {
    pub flag_bins: u32,
    pub merge_idx_bins: u32,
    pub mode_flag_bins: u32,
    pub mvp_flag_bins: u32,
    pub mvd_flag_bins: u32,
}

/// `(ctxTable, ctxIdx)` for a single-context Main-profile table, with the
/// Baseline `sps_cm_init_flag == 0` collapse to `(0, 0)`.
fn ctx1(ctx: EipdCtx, table: MainCtxTable, ctx_inc: usize) -> (usize, usize) {
    if ctx.is_cm_init() {
        (table.as_usize(), ctx_inc)
    } else {
        (0, 0)
    }
}

/// §7.3.8.4 + Table 95/96 — read `affine_flag` (FL cMax = 1) with the
/// §9.3.4.2.4 neighbour-derived ctxInc.
pub fn read_affine_flag(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    neighbours: AffineFlagNeighbours,
    stats: &mut AffineSyntaxStats,
) -> Result<bool> {
    let (t, i) = ctx1(ctx, MainCtxTable::AffineFlag, neighbours.ctx_inc());
    let v = eng.decode_decision(t, i)? != 0;
    stats.flag_bins += 1;
    Ok(v)
}

/// §7.3.8.4 + Table 56 — read `affine_merge_idx` (TR cMax = 5,
/// cRiceParam = 0; per-bin ctxInc 0,1,2,3,4).
pub fn read_affine_merge_idx(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut AffineSyntaxStats,
) -> Result<u32> {
    let cm_init = ctx.is_cm_init();
    let table = MainCtxTable::AffineMergeIdx.as_usize();
    // Count the bins consumed so the stats reflect the actual prefix
    // length; decode_tr_regular's ctxInc closure is invoked once per bin.
    let mut bins = 0u32;
    let v = eng.decode_tr_regular(5, 0, if cm_init { table } else { 0 }, |bin_idx| {
        bins += 1;
        if cm_init {
            bin_idx as usize
        } else {
            0
        }
    })?;
    stats.merge_idx_bins += bins;
    Ok(v)
}

/// §7.3.8.4 + Table 57 — read `affine_mode_flag` (FL cMax = 1, ctxInc 0).
pub fn read_affine_mode_flag(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut AffineSyntaxStats,
) -> Result<bool> {
    let (t, i) = ctx1(ctx, MainCtxTable::AffineModeFlag, 0);
    let v = eng.decode_decision(t, i)? != 0;
    stats.mode_flag_bins += 1;
    Ok(v)
}

/// §7.3.8.4 + Table 58 — read `affine_mvp_flag_lX` (FL cMax = 1, ctxInc 0;
/// the same Table 58 context serves both lists).
fn read_affine_mvp_flag(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut AffineSyntaxStats,
) -> Result<u32> {
    let (t, i) = ctx1(ctx, MainCtxTable::AffineMvpFlag, 0);
    let v = eng.decode_decision(t, i)? as u32;
    stats.mvp_flag_bins += 1;
    Ok(v)
}

/// §7.3.8.4 + Table 59/60 — read `affine_mvd_flag_lX` (FL cMax = 1,
/// ctxInc 0; list 0 uses Table 59, list 1 Table 60).
fn read_affine_mvd_flag(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    list: usize,
    stats: &mut AffineSyntaxStats,
) -> Result<bool> {
    let table = if list == 0 {
        MainCtxTable::AffineMvdFlagL0
    } else {
        MainCtxTable::AffineMvdFlagL1
    };
    let (t, i) = ctx1(ctx, table, 0);
    let v = eng.decode_decision(t, i)? != 0;
    stats.mvd_flag_bins += 1;
    Ok(v)
}

/// §7.3.8.4 — read the per-list AMVP affine flag pair
/// (`affine_mvp_flag_lX` then `affine_mvd_flag_lX`) when list X is
/// active. Public entry for the explicit-AMVP driver, which interleaves
/// the `ref_idx_lX` read the spec places before this pair.
pub fn read_affine_list_flags_pub(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    list: usize,
    stats: &mut AffineSyntaxStats,
) -> Result<AffineListFlags> {
    read_affine_list_flags(eng, ctx, list, stats)
}

/// §7.3.8.4 — read the per-list AMVP affine flag pair
/// (`affine_mvp_flag_lX` then `affine_mvd_flag_lX`) when list X is active.
fn read_affine_list_flags(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    list: usize,
    stats: &mut AffineSyntaxStats,
) -> Result<AffineListFlags> {
    let mvp_flag = read_affine_mvp_flag(eng, ctx, stats)?;
    let mvd_flag = read_affine_mvd_flag(eng, ctx, list, stats)?;
    Ok(AffineListFlags { mvp_flag, mvd_flag })
}

/// §7.3.8.4 — read the full affine syntax group of an inter
/// `coding_unit()` once `affine_flag` has been signalled `== 1`.
///
/// `merge` selects the merge vs AMVP branch (the caller has already
/// decoded `merge_mode_flag` / `cu_skip_flag`). For the AMVP branch
/// `(l0_active, l1_active)` come from the CU's `inter_pred_idc` (eqs.
/// after 2953/2969: list X is read when `inter_pred_idc != PRED_L(1−X)`
/// or `== PRED_BI`). The `ref_idx_lX` reads that interleave with these
/// flags in the spec are handled by the caller; this reader consumes only
/// the affine-specific flags.
pub fn read_affine_group(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    merge: bool,
    l0_active: bool,
    l1_active: bool,
    stats: &mut AffineSyntaxStats,
) -> Result<AffineDecision> {
    if merge {
        let merge_idx = read_affine_merge_idx(eng, ctx, stats)?;
        return Ok(AffineDecision::Merge { merge_idx });
    }
    let affine_mode_flag = read_affine_mode_flag(eng, ctx, stats)?;
    let l0 = if l0_active {
        Some(read_affine_list_flags(eng, ctx, 0, stats)?)
    } else {
        None
    };
    let l1 = if l1_active {
        Some(read_affine_list_flags(eng, ctx, 1, stats)?)
    } else {
        None
    };
    Ok(AffineDecision::Amvp {
        affine_mode_flag,
        l0,
        l1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacEncoder;

    fn regular_bins(bins: &[u8]) -> Vec<u8> {
        let mut enc = CabacEncoder::new();
        for &b in bins {
            enc.encode_decision(0, 0, b);
        }
        enc.encode_terminate(true);
        let mut out = enc.finish();
        // The §9.3.4.3 range coder may renormalise past the committed
        // bytes during the final bin reads; the flush guarantees any
        // over-read sees 1-bits, so pad with 0xFF tail bytes so the
        // bitreader never hits a hard end-of-stream for short streams.
        out.extend_from_slice(&[0xFF; 4]);
        out
    }

    /// §9.3.4.2.4 eq. 1438 — ctxInc is Min(Σ available-and-affine, 1).
    #[test]
    fn affine_flag_ctx_inc_saturates_at_one() {
        // No affine neighbours → 0.
        assert_eq!(AffineFlagNeighbours::default().ctx_inc(), 0);
        // One affine + available → 1.
        let one = AffineFlagNeighbours {
            cond_l: true,
            avail_l: true,
            ..Default::default()
        };
        assert_eq!(one.ctx_inc(), 1);
        // Affine but NOT available → still 0 (the && gate).
        let unavail = AffineFlagNeighbours {
            cond_a: true,
            avail_a: false,
            ..Default::default()
        };
        assert_eq!(unavail.ctx_inc(), 0);
        // All three → Min(3, 1) = 1.
        let all = AffineFlagNeighbours {
            cond_l: true,
            avail_l: true,
            cond_a: true,
            avail_a: true,
            cond_r: true,
            avail_r: true,
        };
        assert_eq!(all.ctx_inc(), 1);
    }

    /// affine_mode_flag → numCpMv: 0 (4-param) ⇒ 2, 1 (6-param) ⇒ 3.
    #[test]
    fn amvp_num_cp_mv_from_mode_flag() {
        let four = AffineDecision::Amvp {
            affine_mode_flag: false,
            l0: None,
            l1: None,
        };
        assert_eq!(four.num_cp_mv(), Some(2));
        let six = AffineDecision::Amvp {
            affine_mode_flag: true,
            l0: None,
            l1: None,
        };
        assert_eq!(six.num_cp_mv(), Some(3));
        // Merge / NotAffine carry no AMVP-derived CP count.
        assert_eq!(AffineDecision::Merge { merge_idx: 0 }.num_cp_mv(), None);
        assert_eq!(AffineDecision::NotAffine.num_cp_mv(), None);
    }

    /// affine_flag reads a single bin and reflects it.
    #[test]
    fn read_flag_one_bin() {
        let bs = regular_bins(&[1]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let f = read_affine_flag(
            &mut eng,
            EipdCtx::new(false),
            AffineFlagNeighbours::default(),
            &mut stats,
        )
        .unwrap();
        assert!(f);
        assert_eq!(stats.flag_bins, 1);
    }

    /// affine_merge_idx (TR cMax=5): a leading 0 bin → value 0, one bin.
    #[test]
    fn read_merge_idx_zero() {
        let bs = regular_bins(&[0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let v = read_affine_merge_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 0);
        assert_eq!(stats.merge_idx_bins, 1);
    }

    /// affine_merge_idx (TR cMax=5): `1 1 0` → value 2, three bins.
    #[test]
    fn read_merge_idx_two() {
        let bs = regular_bins(&[1, 1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let v = read_affine_merge_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 2);
        assert_eq!(stats.merge_idx_bins, 3);
    }

    /// affine_merge_idx (TR cMax=5): five `1`s saturate at 5 (no trailing
    /// 0 once the prefix is full).
    #[test]
    fn read_merge_idx_saturates() {
        let bs = regular_bins(&[1, 1, 1, 1, 1]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let v = read_affine_merge_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 5);
        assert_eq!(stats.merge_idx_bins, 5);
    }

    /// Merge branch of the full group: only affine_merge_idx is read.
    #[test]
    fn group_merge_branch() {
        let bs = regular_bins(&[1, 0]); // merge_idx = 1 (TR: "1 0")
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let d = read_affine_group(&mut eng, EipdCtx::new(false), true, true, false, &mut stats)
            .unwrap();
        assert_eq!(d, AffineDecision::Merge { merge_idx: 1 });
        assert_eq!(stats.mode_flag_bins, 0);
        assert_eq!(stats.merge_idx_bins, 2);
    }

    /// AMVP branch, bi-pred: affine_mode_flag then per-list (mvp, mvd)
    /// pairs for both lists.
    #[test]
    fn group_amvp_bipred() {
        // mode_flag=1, L0:(mvp=0, mvd=1), L1:(mvp=1, mvd=0).
        let bs = regular_bins(&[1, 0, 1, 1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let d = read_affine_group(&mut eng, EipdCtx::new(false), false, true, true, &mut stats)
            .unwrap();
        match d {
            AffineDecision::Amvp {
                affine_mode_flag,
                l0,
                l1,
            } => {
                assert!(affine_mode_flag);
                assert_eq!(d.num_cp_mv(), Some(3));
                assert_eq!(
                    l0,
                    Some(AffineListFlags {
                        mvp_flag: 0,
                        mvd_flag: true
                    })
                );
                assert_eq!(
                    l1,
                    Some(AffineListFlags {
                        mvp_flag: 1,
                        mvd_flag: false
                    })
                );
            }
            _ => panic!("expected AMVP"),
        }
        // 1 mode + 2 mvp + 2 mvd bins.
        assert_eq!(stats.mode_flag_bins, 1);
        assert_eq!(stats.mvp_flag_bins, 2);
        assert_eq!(stats.mvd_flag_bins, 2);
    }

    /// AMVP branch, L0-only: no L1 flags consumed.
    #[test]
    fn group_amvp_l0_only() {
        let bs = regular_bins(&[0, 1, 0]); // mode=0, L0:(mvp=1, mvd=0)
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AffineSyntaxStats::default();
        let d = read_affine_group(
            &mut eng,
            EipdCtx::new(false),
            false,
            true,
            false,
            &mut stats,
        )
        .unwrap();
        match d {
            AffineDecision::Amvp {
                affine_mode_flag,
                l0,
                l1,
            } => {
                assert!(!affine_mode_flag);
                assert_eq!(d.num_cp_mv(), Some(2));
                assert_eq!(
                    l0,
                    Some(AffineListFlags {
                        mvp_flag: 1,
                        mvd_flag: false
                    })
                );
                assert_eq!(l1, None);
            }
            _ => panic!("expected AMVP"),
        }
        assert_eq!(stats.mvp_flag_bins, 1);
        assert_eq!(stats.mvd_flag_bins, 1);
    }
}
