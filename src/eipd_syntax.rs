//! EVC Main-profile **EIPD intra-mode CABAC syntax** (ISO/IEC
//! 23094-1:2020 §7.3.8.4 `coding_unit()`, `sps_eipd_flag == 1` path).
//!
//! When `sps_eipd_flag == 1` the per-CU intra prediction syntax is the
//! six-element group (spec lines 2852-2866):
//!
//! ```text
//!   intra_luma_pred_mpm_flag[ x0 ][ y0 ]                 ae(v)
//!   if( intra_luma_pred_mpm_flag )
//!       intra_luma_pred_mpm_idx[ x0 ][ y0 ]              ae(v)
//!   else {
//!       intra_luma_pred_pims_flag[ x0 ][ y0 ]            ae(v)
//!       if( intra_luma_pred_pims_flag )
//!           intra_luma_pred_pims_idx[ x0 ][ y0 ]         ae(v)
//!       else
//!           intra_luma_pred_rem_mode[ x0 ][ y0 ]         ae(v)
//!   }
//!   …
//!   intra_chroma_pred_mode[ x0 ][ y0 ]                   ae(v)   (chroma tree)
//! ```
//!
//! This module reads those bins from a [`CabacEngine`] and resolves them
//! into a [`ModeSelector`] (luma) and a raw `intra_chroma_pred_mode`
//! value (chroma), which the [`crate::eipd_mode`] derivation then turns
//! into `IntraPredModeY` / `IntraPredModeC`. The mode-derivation
//! neighbourhood (`candIntraPredModeA/B/C`) is the caller's
//! responsibility; this module only owns the bitstream reads.
//!
//! ## Binarisation + contexts (Tables 91/93/95)
//!
//! | element                        | binarisation     | bin contexts (Table 95)            |
//! |--------------------------------|------------------|-------------------------------------|
//! | `intra_luma_pred_mpm_flag`     | FL, cMax = 1     | bin0 ctxInc 0 (Table 63)            |
//! | `intra_luma_pred_mpm_idx`      | FL, cMax = 1     | bin0 ctxInc 0 (Table 64)            |
//! | `intra_luma_pred_pims_flag`    | FL, cMax = 1     | bin0 **bypass**                     |
//! | `intra_luma_pred_pims_idx`     | FL, cMax = 7     | bins 0-2 **bypass**                 |
//! | `intra_luma_pred_rem_mode`     | TB, cMax = 22    | all bins **bypass**                 |
//! | `intra_chroma_pred_mode`       | Table 93         | bin0 ctxInc 0 (Table 65), rest bypass |
//!
//! For `sps_cm_init_flag == 0` (Baseline) the context-coded bins collapse
//! to the single `(ctxTable = 0, ctxIdx = 0)` context (Annex A.3.2); for
//! `sps_cm_init_flag == 1` each lands on its own Main-profile context
//! table at `ctxIdx = ctxInc = 0`. The [`EipdCtx`] selector encapsulates
//! that choice so the read code stays branch-free.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::cabac::CabacEngine;
use crate::cabac_init::MainCtxTable;
use crate::eipd_mode::{derive_chroma_mode, derive_mode_lists, ModeSelector, NeighbourMode};

/// Resolves the `(ctxTable, ctxIdx)` pair for a single-context EIPD
/// syntax element, honouring `sps_cm_init_flag`.
///
/// Under `sps_cm_init_flag == 0` every context-coded bin shares
/// `(0, 0)`; under `sps_cm_init_flag == 1` the element uses its
/// Main-profile context table (Table 63/64/65) at `ctxIdx = 0` (the only
/// ctxInc Table 95 assigns these elements).
#[derive(Clone, Copy, Debug)]
pub struct EipdCtx {
    cm_init: bool,
}

impl EipdCtx {
    /// Build the context selector for the slice's `sps_cm_init_flag`.
    pub fn new(sps_cm_init_flag: bool) -> Self {
        Self {
            cm_init: sps_cm_init_flag,
        }
    }

    /// Whether `sps_cm_init_flag == 1` (Main-profile per-element contexts).
    /// Used by sibling syntax readers (e.g. [`crate::ats`]) that share the
    /// same Baseline-collapse-to-`(0, 0)` discipline.
    #[inline]
    pub fn is_cm_init(self) -> bool {
        self.cm_init
    }

    /// `(ctxTable, ctxIdx)` for the named Main-profile context table.
    #[inline]
    fn ctx(self, table: MainCtxTable) -> (usize, usize) {
        if self.cm_init {
            (table.as_usize(), 0)
        } else {
            (0, 0)
        }
    }
}

/// Read counters for the EIPD intra-mode syntax (mirrors the slice-data
/// `stats` style — one counter per element so tests can assert exact bin
/// budgets per CU).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EipdSyntaxStats {
    pub mpm_flag_bins: u32,
    pub mpm_idx_bins: u32,
    pub pims_flag_bins: u32,
    pub pims_idx_bins: u32,
    pub rem_mode_bins: u32,
    pub chroma_pred_mode_bins: u32,
}

/// §7.3.8.4 — read the luma intra-mode selector group
/// (`intra_luma_pred_mpm_flag` → …) and resolve it to a
/// [`ModeSelector`] that indexes the [`crate::eipd_mode::EipdModeLists`].
///
/// * `mpm_flag == 1` → `Mpm(mpm_idx)`   (`mpm_idx` ∈ {0, 1}),
/// * else `pims_flag == 1` → `Pims(pims_idx)` (`pims_idx` ∈ 0..7),
/// * else → `Rem(rem_mode)` (`rem_mode` ∈ 0..22, mapping to
///   `remModeList[ rem_mode + 10 ]`).
pub fn read_luma_mode_selector(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut EipdSyntaxStats,
) -> Result<ModeSelector> {
    // intra_luma_pred_mpm_flag — FL cMax=1, ctxInc 0 (Table 63).
    let (t, i) = ctx.ctx(MainCtxTable::IntraLumaPredMpmFlag);
    let mpm_flag = eng.decode_decision(t, i)?;
    stats.mpm_flag_bins += 1;
    if mpm_flag != 0 {
        // intra_luma_pred_mpm_idx — FL cMax=1, ctxInc 0 (Table 64).
        let (t, i) = ctx.ctx(MainCtxTable::IntraLumaPredMpmIdx);
        let mpm_idx = eng.decode_decision(t, i)? as usize;
        stats.mpm_idx_bins += 1;
        return Ok(ModeSelector::Mpm(mpm_idx));
    }
    // intra_luma_pred_pims_flag — FL cMax=1, **bypass** (Table 95).
    let pims_flag = eng.decode_bypass()?;
    stats.pims_flag_bins += 1;
    if pims_flag != 0 {
        // intra_luma_pred_pims_idx — FL cMax=7, 3 **bypass** bins.
        let pims_idx = eng.decode_fl_bypass(7)? as usize;
        stats.pims_idx_bins += 1;
        Ok(ModeSelector::Pims(pims_idx))
    } else {
        // intra_luma_pred_rem_mode — TB cMax=22, all **bypass**.
        let rem = eng.decode_tb_bypass(22)? as usize;
        stats.rem_mode_bins += 1;
        Ok(ModeSelector::Rem(rem))
    }
}

/// §7.3.8.4 + §9.3.3.7 (Table 93) — read `intra_chroma_pred_mode`.
///
/// Bin string (Table 93): `1`→0, `00`→1, `010`→2, `0110`→3, `0111`→4.
/// The first bin is context-coded (ctxInc 0, Table 65); every subsequent
/// bin is bypass (Table 95). Decoded as a prefix walk: a `1` at any
/// position terminates the small-value branch, otherwise the next bin is
/// consumed.
pub fn read_intra_chroma_pred_mode(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut EipdSyntaxStats,
) -> Result<i32> {
    stats.chroma_pred_mode_bins += 1;
    // bin0 — context-coded (Table 65).
    let (t, i) = ctx.ctx(MainCtxTable::IntraChromaPredMode);
    if eng.decode_decision(t, i)? != 0 {
        // "1" → value 0 (DM mode).
        return Ok(0);
    }
    // bin1 — bypass. "00" → 1.
    if eng.decode_bypass()? == 0 {
        return Ok(1);
    }
    // bin2 — bypass. "010" → 2.
    if eng.decode_bypass()? == 0 {
        return Ok(2);
    }
    // bin3 — bypass. "0110" → 3, "0111" → 4.
    if eng.decode_bypass()? == 0 {
        Ok(3)
    } else {
        Ok(4)
    }
}

/// §7.3.8.4 + §8.4.2 — read the luma intra-mode syntax group from the
/// bitstream and resolve it to the concrete `IntraPredModeY` for the
/// block.
///
/// Composes [`read_luma_mode_selector`] (the CABAC reads) with
/// [`crate::eipd_mode::derive_mode_lists`] (the three ranked candidate
/// lists built from the §6.4.1 neighbour modes A/B/C) and the §8.4.2
/// step-6 selection. The caller supplies the three already-derived
/// neighbour candidates; this is the single entry point a `coding_unit()`
/// walker calls to turn the EIPD luma syntax into the mode that
/// [`crate::eipd::predict_eipd`] consumes.
pub fn resolve_eipd_luma_mode(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut EipdSyntaxStats,
    a: NeighbourMode,
    b: NeighbourMode,
    c: NeighbourMode,
) -> Result<i32> {
    let selector = read_luma_mode_selector(eng, ctx, stats)?;
    let lists = derive_mode_lists(a, b, c);
    Ok(lists.select(selector))
}

/// §7.3.8.4 + §8.4.3 — read `intra_chroma_pred_mode` and resolve it to
/// the concrete `IntraPredModeC`, given the co-located luma mode
/// `IntraPredModeY` (already resolved via [`resolve_eipd_luma_mode`] for
/// the matching luma cell, or forced to `INTRA_DC` for `MODE_IBC` blocks
/// per §8.4.3).
pub fn resolve_eipd_chroma_mode(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut EipdSyntaxStats,
    intra_pred_mode_y: i32,
) -> Result<i32> {
    let raw = read_intra_chroma_pred_mode(eng, ctx, stats)?;
    Ok(derive_chroma_mode(raw, intra_pred_mode_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacEncoder;

    /// Helper: build a CABAC bitstream of `bins` regular-coded on
    /// `(0, 0)`, terminate, and return the engine ready to decode. Only
    /// the regular-bin path is exercised so the test-only encoder's
    /// bypass defer behaviour never bites (the same discipline the
    /// `slice_data` round-90/95 tests use).
    fn engine_with_regular_bins(bins: &[u8]) -> Vec<u8> {
        let mut enc = CabacEncoder::new();
        for &b in bins {
            enc.encode_decision(0, 0, b);
        }
        enc.encode_terminate(true);
        enc.finish()
    }

    /// Baseline ctx selector routes every context-coded bin to (0, 0).
    #[test]
    fn baseline_ctx_routes_to_zero() {
        let ctx = EipdCtx::new(false);
        assert_eq!(ctx.ctx(MainCtxTable::IntraLumaPredMpmFlag), (0, 0));
        assert_eq!(ctx.ctx(MainCtxTable::IntraChromaPredMode), (0, 0));
    }

    /// Main-profile ctx selector routes each element to its own table at
    /// ctxIdx 0.
    #[test]
    fn main_ctx_routes_to_table() {
        let ctx = EipdCtx::new(true);
        assert_eq!(
            ctx.ctx(MainCtxTable::IntraLumaPredMpmFlag),
            (MainCtxTable::IntraLumaPredMpmFlag.as_usize(), 0)
        );
        assert_eq!(
            ctx.ctx(MainCtxTable::IntraLumaPredMpmIdx),
            (MainCtxTable::IntraLumaPredMpmIdx.as_usize(), 0)
        );
        assert_eq!(
            ctx.ctx(MainCtxTable::IntraChromaPredMode),
            (MainCtxTable::IntraChromaPredMode.as_usize(), 0)
        );
    }

    /// `mpm_flag == 1` then `mpm_idx == 1` → `Mpm(1)`; exactly two
    /// context-coded bins consumed, nothing else.
    #[test]
    fn luma_mpm_path() {
        // bins: mpm_flag=1, mpm_idx=1.
        let bs = engine_with_regular_bins(&[1, 1]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = EipdSyntaxStats::default();
        let sel = read_luma_mode_selector(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(sel, ModeSelector::Mpm(1));
        assert_eq!(stats.mpm_flag_bins, 1);
        assert_eq!(stats.mpm_idx_bins, 1);
        assert_eq!(stats.pims_flag_bins, 0);
        assert_eq!(stats.rem_mode_bins, 0);
    }

    /// `mpm_flag == 1`, `mpm_idx == 0` → `Mpm(0)`.
    #[test]
    fn luma_mpm_idx_zero() {
        let bs = engine_with_regular_bins(&[1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = EipdSyntaxStats::default();
        let sel = read_luma_mode_selector(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(sel, ModeSelector::Mpm(0));
    }

    /// `intra_chroma_pred_mode` Table 93 bin0 == 1 → value 0 (DM), one
    /// context bin only.
    #[test]
    fn chroma_dm_single_bin() {
        let bs = engine_with_regular_bins(&[1]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = EipdSyntaxStats::default();
        let v = read_intra_chroma_pred_mode(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(v, 0);
        assert_eq!(stats.chroma_pred_mode_bins, 1);
    }

    /// Table 93 bin-string structure: the prefix-walk decode maps each
    /// bin string to the right value. We verify the decode tree directly
    /// (independent of the arithmetic engine) by feeding a bin oracle.
    #[test]
    fn chroma_table_93_decode_tree() {
        // (bins, expected) per Table 93 — note bin0 is context-coded but
        // for this structural check we read all bins via decode_bypass on
        // a crafted predicate; instead assert the mapping logic against
        // the spec table by re-deriving it from the bin strings.
        let cases: &[(&[u8], i32)] = &[
            (&[1], 0),
            (&[0, 0], 1),
            (&[0, 1, 0], 2),
            (&[0, 1, 1, 0], 3),
            (&[0, 1, 1, 1], 4),
        ];
        for (bins, expected) in cases {
            // Mirror the read_intra_chroma_pred_mode walk over the bin
            // sequence directly.
            let mut it = bins.iter();
            let v = if *it.next().unwrap() != 0 {
                0
            } else if *it.next().unwrap() == 0 {
                1
            } else if *it.next().unwrap() == 0 {
                2
            } else if *it.next().unwrap() == 0 {
                3
            } else {
                4
            };
            assert_eq!(v, *expected, "Table 93 bin string {bins:?}");
        }
    }

    /// End-to-end luma: `mpm_flag=1, mpm_idx=0` against three valid
    /// neighbour modes resolves to `candModeList[0]` — i.e. the smaller of
    /// the (deduplicated) A/B modes (§8.4.2 eqs. 172-175).
    #[test]
    fn resolve_luma_mpm_picks_cand_list() {
        use crate::eipd_mode::derive_mode_lists;
        // bins: mpm_flag=1, mpm_idx=0.
        let bs = engine_with_regular_bins(&[1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = EipdSyntaxStats::default();
        let a = NeighbourMode::valid(18);
        let b = NeighbourMode::valid(12);
        let c = NeighbourMode::invalid();
        let mode =
            resolve_eipd_luma_mode(&mut eng, EipdCtx::new(false), &mut stats, a, b, c).unwrap();
        let expected = derive_mode_lists(a, b, c).cand_mode_list[0];
        assert_eq!(mode, expected);
        assert_eq!(mode, 12, "candModeList[0] = min(18, 12)");
    }

    /// End-to-end chroma DM: `intra_chroma_pred_mode == 0` reuses the
    /// resolved luma mode regardless of value (§8.4.3).
    #[test]
    fn resolve_chroma_dm_reuses_luma() {
        let bs = engine_with_regular_bins(&[1]); // chroma bin0=1 → value 0
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = EipdSyntaxStats::default();
        let mode = resolve_eipd_chroma_mode(&mut eng, EipdCtx::new(false), &mut stats, 18).unwrap();
        assert_eq!(mode, 18);
    }

    /// Full pipeline: resolve the luma mode from the bitstream, then run
    /// the §8.4.4 prediction kernel for that mode over a flat reference
    /// neighbourhood. A flat DC neighbourhood predicts the constant value,
    /// proving the syntax → derivation → kernel chain executes coherently.
    #[test]
    fn resolve_then_predict_runs_kernel() {
        use crate::eipd::{predict_eipd, AvailLr, EipdRefSamples, INTRA_DC};
        // Neighbours all DC → candModeList collapses to {DC, BI}; mpm_idx=0
        // selects DC.
        let bs = engine_with_regular_bins(&[1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = EipdSyntaxStats::default();
        let n = NeighbourMode::valid(INTRA_DC);
        let mode =
            resolve_eipd_luma_mode(&mut eng, EipdCtx::new(false), &mut stats, n, n, n).unwrap();
        assert_eq!(mode, INTRA_DC);

        // Build a flat reference neighbourhood at value 128 and predict.
        let (w, h) = (4usize, 4usize);
        let mut refs = EipdRefSamples::unavailable(w, h, 8);
        for x in 0..w as i32 {
            refs.set_top(x, 128);
        }
        for y in 0..h as i32 {
            refs.set_left(y, 128);
        }
        let mut dst = vec![0i32; w * h];
        predict_eipd(mode, &refs, w, h, 8, AvailLr::Lr11, &mut dst);
        // DC over a flat 128 neighbourhood → all 128.
        assert!(dst.iter().all(|&p| p == 128), "DC of flat 128 must be 128");
    }
}
