//! §7.3.8.4 MMVD (merge mode with motion-vector difference) CABAC syntax.
//!
//! This module is the bitstream-walk companion to the §8.5 MMVD
//! derivation already in [`crate::inter`] (`mmvd_distance` / `mmvd_sign` /
//! `mmvd_offset` and the per-element ctxInc helpers). When
//! `sps_mmvd_flag == 1` an inter merge CU may signal MMVD: a `mmvd_flag`
//! gate followed by an optional `mmvd_group_idx` and the three index
//! elements (`mmvd_merge_idx`, `mmvd_distance_idx`, `mmvd_direction_idx`)
//! that select the base merge candidate and the offset to add to it.
//!
//! The §7.3.8.4 syntax (spec lines 2811-2818) is:
//!
//! ```text
//!   if( sps_mmvd_flag )
//!       mmvd_flag[ x0 ][ y0 ]                              ae(v)
//!   if( mmvd_flag[ x0 ][ y0 ] ) {
//!       if( mmvd_group_enable_flag && ( log2CbWidth + log2CbHeight ) > 5 )
//!           mmvd_group_idx[ x0 ][ y0 ]                     ae(v)
//!       mmvd_merge_idx[ x0 ][ y0 ]                         ae(v)
//!       mmvd_distance_idx[ x0 ][ y0 ]                      ae(v)
//!       mmvd_direction_idx[ x0 ][ y0 ]                     ae(v)
//!   }
//! ```
//!
//! Binarizations (Table 91) + ctxInc (Table 95 / §9.3.4):
//!
//! | element              | binarization        | per-bin ctxInc |
//! | -------------------- | ------------------- | -------------- |
//! | `mmvd_flag`          | FL, cMax = 1        | 0              |
//! | `mmvd_group_idx`     | TR, cMax = 2        | 0, 1 (Table 51)|
//! | `mmvd_merge_idx`     | TR, cMax = 3        | 0, 1, 2 (T.52) |
//! | `mmvd_distance_idx`  | TR, cMax = 7        | 0..6 (Table 53)|
//! | `mmvd_direction_idx` | FL, cMax = 3        | 0, 1 (Table 54)|
//!
//! Under `sps_cm_init_flag == 0` every regular bin collapses to `(0, 0)`
//! (the Baseline-style context collapse shared by `eipd_syntax` / `ats` /
//! `affine_syntax`); the per-bin ctxInc still applies under `== 1`.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::cabac::CabacEngine;
use crate::cabac_init::MainCtxTable;
use crate::eipd_syntax::EipdCtx;
use crate::inter::{
    mmvd_direction_idx_ctx_inc, mmvd_distance_idx_ctx_inc, mmvd_group_idx_ctx_inc,
    mmvd_merge_idx_ctx_inc, MMVD_DISTANCE_IDX_MAX, MMVD_GROUP_IDX_MAX, MMVD_MERGE_IDX_MAX,
};

/// The decoded §7.3.8.4 MMVD syntax group for one CU.
///
/// When `flag` is false MMVD is not used and the index fields are their
/// inferred-0 defaults (spec lines 5648/5654/5660). When `flag` is true
/// the indices feed the [`crate::inter`] derivation:
/// `mmvd_distance_idx` → `MmvdDistance` (Table 9), `mmvd_direction_idx` →
/// `MmvdSign` (Table 10), `mmvd_merge_idx` selects the base candidate, and
/// `mmvd_group_idx` selects the prediction direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MmvdDecision {
    /// `mmvd_flag` — MMVD engaged for this CU.
    pub flag: bool,
    /// `mmvd_group_idx` (0 when absent/inferred).
    pub group_idx: u32,
    /// `mmvd_merge_idx` — base merge candidate selector (0..=3).
    pub merge_idx: u32,
    /// `mmvd_distance_idx` — step-size selector (0..=7).
    pub distance_idx: u32,
    /// `mmvd_direction_idx` — sign/direction selector (0..=3).
    pub direction_idx: u32,
}

/// Per-element read counters for the MMVD syntax group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MmvdSyntaxStats {
    pub flag_bins: u32,
    pub group_idx_bins: u32,
    pub merge_idx_bins: u32,
    pub distance_idx_bins: u32,
    pub direction_idx_bins: u32,
}

/// Spec line 2814 — `mmvd_group_idx` is present iff
/// `mmvd_group_enable_flag && (log2CbWidth + log2CbHeight) > 5`.
pub fn mmvd_group_idx_present(
    mmvd_group_enable_flag: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
) -> bool {
    mmvd_group_enable_flag && (log2_cb_width + log2_cb_height) > 5
}

/// §7.3.8.4 — read the MMVD syntax group and resolve it to an
/// [`MmvdDecision`].
///
/// The caller must have established `sps_mmvd_flag == 1` (the gate that
/// makes `mmvd_flag` present). When `mmvd_flag` resolves to 0 no further
/// bins are read and the inferred-default decision is returned. The
/// `mmvd_group_idx` presence is gated by [`mmvd_group_idx_present`].
pub fn read_mmvd_group(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    mmvd_group_enable_flag: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
    stats: &mut MmvdSyntaxStats,
) -> Result<MmvdDecision> {
    let cm_init = ctx.is_cm_init();

    // mmvd_flag — FL cMax=1, ctxInc 0 (Table 50 / 95).
    let (t, i) = ctx1(ctx, MainCtxTable::MmvdFlag, 0);
    let flag = eng.decode_decision(t, i)? != 0;
    stats.flag_bins += 1;
    if !flag {
        return Ok(MmvdDecision::default());
    }

    // mmvd_group_idx — TR cMax=2, per-bin ctxInc 0,1 (Table 51), gated.
    let group_idx = if mmvd_group_idx_present(mmvd_group_enable_flag, log2_cb_width, log2_cb_height)
    {
        let table = MainCtxTable::MmvdGroupIdx.as_usize();
        let mut bins = 0u32;
        let v = eng.decode_tr_regular(
            MMVD_GROUP_IDX_MAX,
            0,
            if cm_init { table } else { 0 },
            |bin_idx| {
                bins += 1;
                if cm_init {
                    mmvd_group_idx_ctx_inc(bin_idx).unwrap_or(0)
                } else {
                    0
                }
            },
        )?;
        stats.group_idx_bins += bins;
        v
    } else {
        0
    };

    // mmvd_merge_idx — TR cMax=3, per-bin ctxInc 0,1,2 (Table 52).
    let merge_idx = {
        let table = MainCtxTable::MmvdMergeIdx.as_usize();
        let mut bins = 0u32;
        let v = eng.decode_tr_regular(
            MMVD_MERGE_IDX_MAX,
            0,
            if cm_init { table } else { 0 },
            |bin_idx| {
                bins += 1;
                if cm_init {
                    mmvd_merge_idx_ctx_inc(bin_idx).unwrap_or(0)
                } else {
                    0
                }
            },
        )?;
        stats.merge_idx_bins += bins;
        v
    };

    // mmvd_distance_idx — TR cMax=7, per-bin ctxInc 0..6 (Table 53).
    let distance_idx = {
        let table = MainCtxTable::MmvdDistanceIdx.as_usize();
        let mut bins = 0u32;
        let v = eng.decode_tr_regular(
            MMVD_DISTANCE_IDX_MAX,
            0,
            if cm_init { table } else { 0 },
            |bin_idx| {
                bins += 1;
                if cm_init {
                    mmvd_distance_idx_ctx_inc(bin_idx).unwrap_or(0)
                } else {
                    0
                }
            },
        )?;
        stats.distance_idx_bins += bins;
        v
    };

    // mmvd_direction_idx — FL cMax=3 (2 bins), per-bin ctxInc 0,1 (Table 54).
    let direction_idx = {
        let table = MainCtxTable::MmvdDirectionIdx.as_usize();
        let mut value = 0u32;
        for bin_idx in 0..2u32 {
            let ctx_inc = if cm_init {
                mmvd_direction_idx_ctx_inc(bin_idx).unwrap_or(0)
            } else {
                0
            };
            let (tbl, idx) = if cm_init { (table, ctx_inc) } else { (0, 0) };
            let bin = eng.decode_decision(tbl, idx)?;
            value = (value << 1) | bin as u32;
        }
        stats.direction_idx_bins += 2;
        value
    };

    Ok(MmvdDecision {
        flag: true,
        group_idx,
        merge_idx,
        distance_idx,
        direction_idx,
    })
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

    /// mmvd_group_idx presence: needs the enable flag AND log2 sum > 5.
    #[test]
    fn group_idx_presence_predicate() {
        assert!(mmvd_group_idx_present(true, 3, 3)); // 6 > 5
        assert!(!mmvd_group_idx_present(true, 3, 2)); // 5, not > 5
        assert!(!mmvd_group_idx_present(false, 4, 4)); // enable off
    }

    /// mmvd_flag == 0 → MMVD not used, only one bin consumed, defaults.
    #[test]
    fn flag_zero_no_mmvd() {
        let bs = regular_bins(&[0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = MmvdSyntaxStats::default();
        let d = read_mmvd_group(&mut eng, EipdCtx::new(false), true, 4, 4, &mut stats).unwrap();
        assert_eq!(d, MmvdDecision::default());
        assert!(!d.flag);
        assert_eq!(stats.flag_bins, 1);
        assert_eq!(stats.merge_idx_bins, 0);
    }

    /// Full MMVD group with group_idx present (CB 8×8, log2 sum 6 > 5):
    /// flag=1, group_idx="0", merge_idx="0", distance_idx="0",
    /// direction_idx=2 bins. Asserts the bin tallies + resolved values.
    #[test]
    fn full_group_with_group_idx() {
        // flag=1, group=0 (TR "0"), merge=0 (TR "0"), distance=0 (TR "0"),
        // direction 2 FL bins = e.g. "1 0".
        let bs = regular_bins(&[1, 0, 0, 0, 1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = MmvdSyntaxStats::default();
        let d = read_mmvd_group(&mut eng, EipdCtx::new(false), true, 3, 3, &mut stats).unwrap();
        assert!(d.flag);
        assert_eq!(stats.flag_bins, 1);
        // group_idx present → at least one bin read.
        assert!(stats.group_idx_bins >= 1);
        assert!(stats.merge_idx_bins >= 1);
        assert!(stats.distance_idx_bins >= 1);
        assert_eq!(stats.direction_idx_bins, 2);
        // direction_idx is a 2-bit FL value.
        assert!(d.direction_idx <= 3);
    }

    /// group_idx absent (CB 4×4, log2 sum 4 ≤ 5): no group_idx bins read;
    /// group_idx stays 0.
    #[test]
    fn group_idx_absent_small_cb() {
        let bs = regular_bins(&[1, 0, 0, 0, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = MmvdSyntaxStats::default();
        let d = read_mmvd_group(&mut eng, EipdCtx::new(false), true, 2, 2, &mut stats).unwrap();
        assert!(d.flag);
        assert_eq!(stats.group_idx_bins, 0);
        assert_eq!(d.group_idx, 0);
    }

    /// mmvd_distance_idx TR saturates at cMax = 7: seven `1` prefix bins
    /// (no trailing 0) resolve to 7.
    #[test]
    fn distance_idx_saturates_at_7() {
        // flag=1, group absent (small CB), merge=0 ("0"),
        // distance: seven 1s → 7, direction 2 bins.
        let bs = regular_bins(&[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = MmvdSyntaxStats::default();
        let d = read_mmvd_group(&mut eng, EipdCtx::new(false), false, 2, 2, &mut stats).unwrap();
        assert!(d.flag);
        assert_eq!(d.distance_idx, 7);
        assert_eq!(stats.distance_idx_bins, 7); // cMax reached → no trailing 0
    }

    /// distance_idx feeds the existing §8.5 derivation: idx 3 → MmvdDistance
    /// = 1 << 3 = 8 (Table 9). Sanity that the syntax output is the index
    /// the derivation consumes.
    #[test]
    fn decision_drives_distance_derivation() {
        let d = MmvdDecision {
            flag: true,
            group_idx: 0,
            merge_idx: 1,
            distance_idx: 3,
            direction_idx: 0,
        };
        let dist = crate::inter::mmvd_distance(d.distance_idx).unwrap();
        assert_eq!(dist, 8);
        let (sx, sy) = crate::inter::mmvd_sign(d.direction_idx).unwrap();
        // direction 0 → (+1, 0) per Table 10.
        assert_eq!((sx, sy), (1, 0));
    }
}
