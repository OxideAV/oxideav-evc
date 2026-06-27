//! §7.3.8.4 AMVR + inter-mode-gating CABAC syntax.
//!
//! This module reads the §7.3.8.4 non-skip inter-CU mode-gating group
//! (spec lines 2878-2884) that precedes the MMVD / affine / explicit-MVD
//! paths:
//!
//! ```text
//!   if( sps_amvr_flag )
//!       amvr_idx[ x0 ][ y0 ]                          ae(v)
//!   if( slice_type == B && sps_admvp_flag == 0 )
//!       direct_mode_flag[ x0 ][ y0 ]                  ae(v)
//!   else if( sps_admvp_flag == 1 ) {
//!       if( amvr_idx[ x0 ][ y0 ] == 0 )
//!           merge_mode_flag[ x0 ][ y0 ]               ae(v)
//!       …
//!   }
//! ```
//!
//! * `amvr_idx` — TR cMax=4, per-bin ctxInc 0,1,2,3 (Table 67); selects
//!   the adaptive motion-vector resolution. Its derivation (eq. 145 MVD
//!   shift + eqs. 645/646 MVP round) lives in [`crate::inter`]; this
//!   module is the syntax read.
//! * `merge_mode_flag` — FL cMax=1, ctxInc 0 (Table 70); present only on
//!   the `sps_admvp_flag == 1` path when `amvr_idx == 0` (else inferred 1,
//!   spec line 5827).
//! * `direct_mode_flag` — FL cMax=1, ctxInc 0 (Table 68); the
//!   `sps_admvp_flag == 0` B-slice direct-mode gate.
//!
//! Under `sps_cm_init_flag == 0` every regular bin collapses to `(0, 0)`,
//! matching `eipd_syntax` / `ats` / `affine_syntax` / `mmvd_syntax`.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::cabac::CabacEngine;
use crate::cabac_init::MainCtxTable;
use crate::eipd_syntax::EipdCtx;
use crate::inter::{amvr_idx_ctx_inc, merge_idx_c_max, merge_idx_ctx_inc, AMVR_IDX_MAX};

/// Per-element read counters for the §7.3.8.4 inter-mode-gating group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterModeGateStats {
    pub amvr_idx_bins: u32,
    pub merge_mode_flag_bins: u32,
    pub direct_mode_flag_bins: u32,
    pub merge_idx_bins: u32,
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

/// §7.3.8.4 + Table 67 — read `amvr_idx` (TR cMax=4, per-bin ctxInc
/// 0,1,2,3 via [`crate::inter::amvr_idx_ctx_inc`]).
///
/// The caller must have established `sps_amvr_flag == 1`. Returns the
/// resolved index in `0..=4` (`0` = 1/4-pel … `4` = 4-pel) which feeds the
/// §8.5 eq.-145 MVD shift + eqs.-645/646 MVP round in [`crate::inter`].
pub fn read_amvr_idx(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut InterModeGateStats,
) -> Result<u32> {
    let cm_init = ctx.is_cm_init();
    let table = MainCtxTable::AmvrIdx.as_usize();
    let mut bins = 0u32;
    let v = eng.decode_tr_regular(
        AMVR_IDX_MAX,
        0,
        if cm_init { table } else { 0 },
        |bin_idx| {
            bins += 1;
            if cm_init {
                amvr_idx_ctx_inc(bin_idx).unwrap_or(0)
            } else {
                0
            }
        },
    )?;
    stats.amvr_idx_bins += bins;
    Ok(v)
}

/// §7.3.8.4 + Table 70 — read `merge_mode_flag` (FL cMax=1, ctxInc 0).
///
/// Present only on the `sps_admvp_flag == 1` path when `amvr_idx == 0`;
/// the caller applies that presence gate (an absent flag is inferred 1,
/// spec line 5827).
pub fn read_merge_mode_flag(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut InterModeGateStats,
) -> Result<bool> {
    let (t, i) = ctx1(ctx, MainCtxTable::MergeModeFlag, 0);
    let v = eng.decode_decision(t, i)? != 0;
    stats.merge_mode_flag_bins += 1;
    Ok(v)
}

/// §7.3.8.4 + Table 68 — read `direct_mode_flag` (FL cMax=1, ctxInc 0).
///
/// The `sps_admvp_flag == 0` B-slice direct-mode gate; the caller applies
/// the `slice_type == B && sps_admvp_flag == 0` presence condition.
pub fn read_direct_mode_flag(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut InterModeGateStats,
) -> Result<bool> {
    let (t, i) = ctx1(ctx, MainCtxTable::DirectModeFlag, 0);
    let v = eng.decode_decision(t, i)? != 0;
    stats.direct_mode_flag_bins += 1;
    Ok(v)
}

/// §7.3.8.4 + §9.3.3 + Table 49 — read `merge_idx` (TR `cMax =
/// ( nCbW * nCbH <= 32 ) ? 3 : 5`, per-bin ctxInc 0,1,2,3,4 via
/// [`crate::inter::merge_idx_ctx_inc`]).
///
/// This is the regular merging-candidate selector read on the
/// `sps_admvp_flag == 1` non-affine / non-MMVD merge path (spec lines
/// 2830 / 2908) and on the §7.3.8.4 cu_skip non-affine fall-through. The
/// caller supplies the coding-block dimensions in luma samples so the
/// area-dependent `cMax` matches §9.3.3. An absent `merge_idx` is
/// inferred 0 (spec line 5726); the caller applies that presence gate.
pub fn read_merge_idx(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    n_cb_w: u32,
    n_cb_h: u32,
    stats: &mut InterModeGateStats,
) -> Result<u32> {
    let cm_init = ctx.is_cm_init();
    let table = MainCtxTable::MergeIdx.as_usize();
    let c_max = merge_idx_c_max(n_cb_w, n_cb_h);
    let mut bins = 0u32;
    let v = eng.decode_tr_regular(c_max, 0, if cm_init { table } else { 0 }, |bin_idx| {
        bins += 1;
        if cm_init {
            merge_idx_ctx_inc(bin_idx).unwrap_or(0)
        } else {
            0
        }
    })?;
    stats.merge_idx_bins += bins;
    Ok(v)
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
        enc.finish()
    }

    /// Encode a TR-cMax-4 value as the table-0 bin string the cm_init=false
    /// decoder reads, padded with trailing 0s so the range coder has enough
    /// renormalisation headroom for the final prefix bin (a too-short
    /// stream under-flushes and trips the decoder's bit reader).
    fn amvr_bins(value: u32) -> Vec<u8> {
        let mut bins = vec![1u8; value as usize];
        if value < AMVR_IDX_MAX {
            bins.push(0u8);
        }
        // Renormalisation padding.
        bins.extend_from_slice(&[0, 0, 0, 0]);
        regular_bins(&bins)
    }

    /// amvr_idx TR cMax=4, value 0 (1/4-pel): a single leading-0 bin.
    #[test]
    fn amvr_idx_zero_single_bin() {
        let bs = amvr_bins(0);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let v = read_amvr_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 0);
        assert_eq!(stats.amvr_idx_bins, 1);
    }

    /// amvr_idx TR value 2 (integer-pel): "1 1 0" → three bins.
    #[test]
    fn amvr_idx_two() {
        let bs = amvr_bins(2);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let v = read_amvr_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 2);
        assert_eq!(stats.amvr_idx_bins, 3);
    }

    /// amvr_idx TR value 3 (2-pel): "1 1 1 0" → four bins.
    #[test]
    fn amvr_idx_three() {
        let bs = amvr_bins(3);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let v = read_amvr_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 3);
        assert_eq!(stats.amvr_idx_bins, 4);
    }

    /// The resolved amvr_idx feeds the §8.5 eq.-145 MVD shift: idx 2 → MVD
    /// component << 2.
    #[test]
    fn amvr_idx_drives_mvd_shift() {
        let bs = amvr_bins(2);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let idx = read_amvr_idx(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(idx, 2);
        let shifted = crate::inter::amvr_apply_to_mvd(3, idx).unwrap();
        assert_eq!(shifted, 3 << 2);
    }

    /// merge_mode_flag / direct_mode_flag: single FL bin each.
    #[test]
    fn mode_flags_single_bin() {
        let bs = regular_bins(&[1]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let m = read_merge_mode_flag(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert!(m);
        assert_eq!(stats.merge_mode_flag_bins, 1);

        let bs2 = regular_bins(&[0]);
        let mut eng2 = CabacEngine::new(&bs2).unwrap();
        let mut stats2 = InterModeGateStats::default();
        let d = read_direct_mode_flag(&mut eng2, EipdCtx::new(false), &mut stats2).unwrap();
        assert!(!d);
        assert_eq!(stats2.direct_mode_flag_bins, 1);
    }

    /// Encode a TR value as the table-0 bin string the cm_init=false
    /// decoder reads, padded with trailing 0s for renormalisation headroom.
    fn tr_bins(value: u32, c_max: u32) -> Vec<u8> {
        let mut bins = vec![1u8; value as usize];
        if value < c_max {
            bins.push(0u8);
        }
        bins.extend_from_slice(&[0, 0, 0, 0]);
        regular_bins(&bins)
    }

    /// merge_idx cMax depends on the coding-block area (§9.3.3): 32-sample
    /// area → cMax 3; larger → cMax 5.
    #[test]
    fn merge_idx_c_max_area_split() {
        // 4×8 = 32 → cMax 3.
        assert_eq!(crate::inter::merge_idx_c_max(4, 8), 3);
        // 8×8 = 64 → cMax 5.
        assert_eq!(crate::inter::merge_idx_c_max(8, 8), 5);
        // 4×4 = 16 → cMax 3.
        assert_eq!(crate::inter::merge_idx_c_max(4, 4), 3);
    }

    /// merge_idx value 0 on a large block: single leading-0 bin.
    #[test]
    fn merge_idx_zero_large_block() {
        let bs = tr_bins(0, 5);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let v = read_merge_idx(&mut eng, EipdCtx::new(false), 16, 16, &mut stats).unwrap();
        assert_eq!(v, 0);
        assert_eq!(stats.merge_idx_bins, 1);
    }

    /// merge_idx value 4 (cMax 5 branch): "1 1 1 1 0" → five bins.
    #[test]
    fn merge_idx_four_large_block() {
        let bs = tr_bins(4, 5);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let v = read_merge_idx(&mut eng, EipdCtx::new(false), 16, 16, &mut stats).unwrap();
        assert_eq!(v, 4);
        assert_eq!(stats.merge_idx_bins, 5);
    }

    /// On a 32-sample (4×8) block cMax saturates at 3: the all-ones
    /// "1 1 1" string with no terminating 0 reads value 3 in three bins.
    #[test]
    fn merge_idx_small_block_saturates_at_three() {
        let bs = tr_bins(3, 3);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = InterModeGateStats::default();
        let v = read_merge_idx(&mut eng, EipdCtx::new(false), 4, 8, &mut stats).unwrap();
        assert_eq!(v, 3);
        assert_eq!(stats.merge_idx_bins, 3);
    }

    /// Main-profile context routing: under cm_init each flag lands on its
    /// own table at ctxIdx 0; Baseline collapses to (0, 0).
    #[test]
    fn ctx_routing() {
        assert_eq!(
            ctx1(EipdCtx::new(true), MainCtxTable::MergeModeFlag, 0),
            (MainCtxTable::MergeModeFlag.as_usize(), 0)
        );
        assert_eq!(
            ctx1(EipdCtx::new(false), MainCtxTable::MergeModeFlag, 0),
            (0, 0)
        );
    }
}
