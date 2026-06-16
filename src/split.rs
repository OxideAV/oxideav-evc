//! Main-profile coding-tree split geometry — ISO/IEC 23094-1:2020(E).
//!
//! This module implements the pure (CABAC-state-free) layer of the
//! §7.3.8.3 `split_unit()` binary/ternary-tree (BTT) machinery:
//!
//! * the BTT size variables derived in §7.3.2.2 — `MaxCbLog2Size11Ratio`
//!   (eq. 43), `MaxCbLog2Size12Ratio` (eq. 44), `MinCbLog2Size11Ratio`
//!   (eq. 62), `MinCbLog2Size12Ratio` (eq. 63), `MinCbLog2Size14Ratio`
//!   (eq. 64), `MaxCbLog2Size14Ratio` (eq. 65), `MaxTtLog2Size`
//!   (eq. 66), and `MinTtLog2Size` (eq. 67);
//! * the §7.4.8.3 `allowSplitBtVer` / `allowSplitBtHor` /
//!   `allowSplitTtVer` / `allowSplitTtHor` derivations that gate which
//!   `btt_split_*` syntax elements are signalled;
//! * the §7.4.8.3 inference of `btt_split_type` when not present;
//! * the §7.4.8.3 `SplitMode` derivation that turns the parsed
//!   `btt_split_flag` / `btt_split_dir` / `btt_split_type` (plus the
//!   picture-boundary implicit-split rules) into one of the five split
//!   shapes.
//!
//! The pure geometry above consumes no bitstream — the per-syntax-element
//! ctxInc derivations live in [`crate::cabac_init`]. On top of that this
//! module also exposes [`decode_btt_split`], the §7.3.8.3 CABAC-driven
//! split-syntax reader that consumes the `btt_split_flag` /
//! `btt_split_dir` / `btt_split_type` bins from a [`CabacEngine`], applies
//! the §7.3.8.3 presence gating + §7.4.8.3 inference rules, and resolves
//! the final [`SplitMode`]. The geometry helpers stay pure so the whole
//! BTT decision surface remains independently testable; the decoder is the
//! thin glue that ties them to the arithmetic engine.
//!
//! All clause / equation numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::cabac::CabacEngine;
use crate::cabac_init::{ctx_inc_btt_split_flag, MainCtxTable};

/// The five coding-tree split shapes (§7.4.8.3, `SplitMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitMode {
    /// No split — the block is a coding-unit leaf.
    NoSplit,
    /// Binary split, vertical boundary (two half-width children).
    SplitBtVer,
    /// Binary split, horizontal boundary (two half-height children).
    SplitBtHor,
    /// Ternary split, vertical boundaries (¼ / ½ / ¼ width children).
    SplitTtVer,
    /// Ternary split, horizontal boundaries (¼ / ½ / ¼ height children).
    SplitTtHor,
}

/// `predModeConstraintCurrent` (§7.3.8.3) — only the INTER value affects
/// the §7.4.8.3 `allowSplit*` derivation, so this module tracks just the
/// distinction it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredModeConstraint {
    /// `PRED_MODE_NO_CONSTRAINT` or `PRED_MODE_CONSTRAINT_INTRA_IBC`.
    NotInterConstrained,
    /// `PRED_MODE_CONSTRAINT_INTER`.
    Inter,
}

/// The BTT size limits derived once per active SPS (§7.3.2.2). Every
/// field is a log2 size in luma samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BttSizeLimits {
    /// `MaxCbLog2Size11Ratio` (eq. 43) — `CtbLog2SizeY`.
    pub max_cb_log2_size_11_ratio: u32,
    /// `MaxCbLog2Size12Ratio` (eq. 44) — equal to the 1:1 limit.
    pub max_cb_log2_size_12_ratio: u32,
    /// `MaxCbLog2Size14Ratio` (eq. 65).
    pub max_cb_log2_size_14_ratio: u32,
    /// `MinCbLog2Size11Ratio` (eq. 62) — `MinCbLog2SizeY`.
    pub min_cb_log2_size_11_ratio: u32,
    /// `MinCbLog2Size12Ratio` (eq. 63).
    pub min_cb_log2_size_12_ratio: u32,
    /// `MinCbLog2Size14Ratio` (eq. 64).
    pub min_cb_log2_size_14_ratio: u32,
    /// `MaxTtLog2Size` (eq. 66).
    pub max_tt_log2_size: u32,
    /// `MinTtLog2Size` (eq. 67).
    pub min_tt_log2_size: u32,
}

impl BttSizeLimits {
    /// Derive the §7.3.2.2 BTT size limits from the raw SPS syntax
    /// elements. `ctb_log2_size_y` is `CtbLog2SizeY` (eq. 41); the three
    /// `log2_diff_*` and `log2_min_cb_size_minus2` values are read
    /// verbatim from the SPS (present only when `sps_btt_flag == 1`,
    /// otherwise inferred to 0, which this constructor accepts as-is).
    ///
    /// `MaxTbLog2SizeY` is fixed at 6 (eq. 51), used as the clamp ceiling
    /// in eq. 65 and eq. 66.
    pub fn derive(
        ctb_log2_size_y: u32,
        log2_min_cb_size_minus2: u32,
        log2_diff_ctu_max_14_cb_size: u32,
        log2_diff_ctu_max_tt_cb_size: u32,
        log2_diff_min_cb_min_tt_cb_size_minus2: u32,
    ) -> Self {
        const MAX_TB_LOG2_SIZE_Y: u32 = 6; // eq. 51

        // eq. 57: MinCbLog2SizeY.
        let min_cb_log2_size_y = 2 + log2_min_cb_size_minus2;

        // eq. 43 / 44.
        let max_cb_log2_size_11_ratio = ctb_log2_size_y;
        let max_cb_log2_size_12_ratio = max_cb_log2_size_11_ratio;

        // eq. 62 / 63 / 64.
        let min_cb_log2_size_11_ratio = min_cb_log2_size_y;
        let min_cb_log2_size_12_ratio = min_cb_log2_size_11_ratio + 1;
        let min_cb_log2_size_14_ratio = min_cb_log2_size_12_ratio + 1;

        // eq. 65: Min( CtbLog2SizeY − log2_diff_ctu_max_14_cb_size,
        //              MaxTbLog2SizeY ). The subtraction is in the valid
        // syntax range (§7.4.3.1 bounds log2_diff_ctu_max_14_cb_size to
        // CtbLog2SizeY − MinCbLog2Size14Ratio + 1), so it never wraps;
        // saturating_sub is defence in depth against malformed input.
        let max_cb_log2_size_14_ratio = ctb_log2_size_y
            .saturating_sub(log2_diff_ctu_max_14_cb_size)
            .min(MAX_TB_LOG2_SIZE_Y);

        // eq. 66.
        let max_tt_log2_size = ctb_log2_size_y
            .saturating_sub(log2_diff_ctu_max_tt_cb_size)
            .min(MAX_TB_LOG2_SIZE_Y);

        // eq. 67: MinCbLog2SizeY + 2 + log2_diff_min_cb_min_tt_cb_size_minus2.
        let min_tt_log2_size = min_cb_log2_size_y + 2 + log2_diff_min_cb_min_tt_cb_size_minus2;

        Self {
            max_cb_log2_size_11_ratio,
            max_cb_log2_size_12_ratio,
            max_cb_log2_size_14_ratio,
            min_cb_log2_size_11_ratio,
            min_cb_log2_size_12_ratio,
            min_cb_log2_size_14_ratio,
            max_tt_log2_size,
            min_tt_log2_size,
        }
    }
}

/// The four §7.4.8.3 `allowSplit*` decisions for the current block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllowedSplits {
    /// `allowSplitBtVer` — binary vertical split permitted.
    pub bt_ver: bool,
    /// `allowSplitBtHor` — binary horizontal split permitted.
    pub bt_hor: bool,
    /// `allowSplitTtVer` — ternary vertical split permitted.
    pub tt_ver: bool,
    /// `allowSplitTtHor` — ternary horizontal split permitted.
    pub tt_hor: bool,
}

impl AllowedSplits {
    /// `true` when at least one BTT split is permitted — gates whether
    /// `btt_split_flag` is signalled at all (spec line 2650).
    pub fn any(&self) -> bool {
        self.bt_ver || self.bt_hor || self.tt_ver || self.tt_hor
    }
}

/// Shared inner test of the §7.4.8.3 binary-split ratio bound. For a
/// candidate child whose longer side is `log2_longer` and whose
/// width/height log2 difference is `ratio_wh`, returns `true` when the
/// block is **outside** the permitted-ratio window (i.e. the split is
/// disallowed). Mirrors the three `ratioWH == {0,1,2}` cases.
fn bt_ratio_disallows(limits: &BttSizeLimits, log2_longer: u32, ratio_wh: u32) -> bool {
    match ratio_wh {
        0 => {
            log2_longer > limits.max_cb_log2_size_11_ratio
                || log2_longer < limits.min_cb_log2_size_11_ratio
        }
        1 => {
            log2_longer > limits.max_cb_log2_size_12_ratio
                || log2_longer < limits.min_cb_log2_size_12_ratio
        }
        2 => {
            log2_longer > limits.max_cb_log2_size_14_ratio
                || log2_longer < limits.min_cb_log2_size_14_ratio
        }
        // ratioWH >= 3 (e.g. an 8:1 child) is outside every permitted
        // window; the spec's three explicit cases cover 1:1/1:2/1:4, and
        // anything beyond is disallowed.
        _ => true,
    }
}

/// §7.4.8.3 — derive `allowSplitBtVer`, `allowSplitBtHor`,
/// `allowSplitTtVer`, `allowSplitTtHor` for the coding block of size
/// `2^log2_cb_width × 2^log2_cb_height` under `pred_mode_constraint`.
pub fn derive_allowed_splits(
    limits: &BttSizeLimits,
    log2_cb_width: u32,
    log2_cb_height: u32,
    pred_mode_constraint: PredModeConstraint,
) -> AllowedSplits {
    let inter = pred_mode_constraint == PredModeConstraint::Inter;

    // allowSplitBtVer: child halves the width.
    let bt_ver = {
        let log2w_temp = log2_cb_width as i64 - 1;
        if log2w_temp < 2 {
            false
        } else {
            let log2w_temp = log2w_temp as u32;
            let log2_longer = log2w_temp.max(log2_cb_height);
            let ratio_wh = log2w_temp.abs_diff(log2_cb_height);
            let disallowed = bt_ratio_disallows(limits, log2_longer, ratio_wh)
                || (inter && log2w_temp == 2 && log2_cb_height == 2);
            !disallowed
        }
    };

    // allowSplitBtHor: child halves the height.
    let bt_hor = {
        let log2h_temp = log2_cb_height as i64 - 1;
        if log2h_temp < 2 {
            false
        } else {
            let log2h_temp = log2h_temp as u32;
            let log2_longer = log2_cb_width.max(log2h_temp);
            let ratio_wh = log2_cb_width.abs_diff(log2h_temp);
            let disallowed = bt_ratio_disallows(limits, log2_longer, ratio_wh)
                || (inter && log2_cb_width == 2 && log2h_temp == 2);
            !disallowed
        }
    };

    // allowSplitTtVer: ternary on the width axis.
    let tt_ver = {
        let mut disallowed = false;
        if log2_cb_width < log2_cb_height {
            disallowed = true;
        }
        if log2_cb_width > limits.max_tt_log2_size || log2_cb_width < limits.min_tt_log2_size {
            disallowed = true;
        }
        if log2_cb_width == log2_cb_height
            && (log2_cb_width > limits.max_cb_log2_size_14_ratio
                || log2_cb_width < limits.min_cb_log2_size_14_ratio)
        {
            disallowed = true;
        }
        if inter && log2_cb_width == 4 && log2_cb_height == 2 {
            disallowed = true;
        }
        !disallowed
    };

    // allowSplitTtHor: ternary on the height axis.
    let tt_hor = {
        let mut disallowed = false;
        if log2_cb_height < log2_cb_width {
            disallowed = true;
        }
        if log2_cb_height > limits.max_tt_log2_size || log2_cb_height < limits.min_tt_log2_size {
            disallowed = true;
        }
        if log2_cb_width == log2_cb_height
            && (log2_cb_height > limits.max_cb_log2_size_14_ratio
                || log2_cb_height < limits.min_cb_log2_size_14_ratio)
        {
            disallowed = true;
        }
        if inter && log2_cb_width == 2 && log2_cb_height == 4 {
            disallowed = true;
        }
        !disallowed
    };

    AllowedSplits {
        bt_ver,
        bt_hor,
        tt_ver,
        tt_hor,
    }
}

/// §7.4.8.3 — whether `btt_split_dir` is signalled (spec lines
/// 2653–2654): only when both a vertical *and* a horizontal split are
/// possible.
pub fn btt_split_dir_present(allowed: &AllowedSplits) -> bool {
    (allowed.bt_ver || allowed.tt_ver) && (allowed.bt_hor || allowed.tt_hor)
}

/// §7.4.8.3 — inference of `btt_split_dir` when not present (spec line
/// 5431): 1 if a vertical split is possible, else 0.
pub fn infer_btt_split_dir(allowed: &AllowedSplits) -> u32 {
    u32::from(allowed.bt_ver || allowed.tt_ver)
}

/// §7.4.8.3 — whether `btt_split_type` is signalled (spec lines
/// 2656–2657): for the chosen direction, both binary and ternary must be
/// possible.
pub fn btt_split_type_present(allowed: &AllowedSplits, btt_split_dir: u32) -> bool {
    if btt_split_dir != 0 {
        allowed.bt_ver && allowed.tt_ver
    } else {
        allowed.bt_hor && allowed.tt_hor
    }
}

/// §7.4.8.3 — inference of `btt_split_type` when not present (spec lines
/// 5436–5450): 1 (ternary) when the only available split in the chosen
/// direction is ternary, else 0 (binary).
pub fn infer_btt_split_type(allowed: &AllowedSplits, btt_split_dir: u32) -> u32 {
    let ternary = if btt_split_dir == 0 {
        allowed.tt_hor
    } else {
        allowed.tt_ver
    };
    u32::from(ternary)
}

/// §7.4.8.3 — derive `SplitMode` from the parsed (or inferred)
/// `btt_split_flag` / `btt_split_dir` / `btt_split_type`, plus the
/// picture-boundary implicit-split rules.
///
/// `x0`/`y0` and the block size are in luma samples; `pic_width` /
/// `pic_height` are `pic_{width,height}_in_luma_samples`. When
/// `btt_split_flag == 0` and the block straddles a picture edge, the
/// boundary clauses force a binary split toward the in-picture side.
///
/// **Errata note (clean-room disambiguation):** the spec's `SplitMode`
/// table (lines 5459–5469) lists `SPLIT_TT_VER` for the case
/// `btt_split_dir == 0 && btt_split_type == 1`. That contradicts the
/// §7.3.8.3 recursion: a horizontal split (`btt_split_dir == 0`) reduces
/// the *height* (a `log2CbHeight − 1` / `− 2` recursion under
/// `SPLIT_TT_HOR`, lines 2769–2786), whereas `SPLIT_TT_VER` reduces the
/// *width* (lines 2749–2767). The disallowed listing is a typographic
/// duplicate of the `btt_split_dir == 1` branch; we take the
/// recursion-consistent reading `SPLIT_TT_HOR`.
#[allow(clippy::too_many_arguments)]
pub fn derive_split_mode(
    btt_split_flag: bool,
    btt_split_dir: u32,
    btt_split_type: u32,
    allowed: &AllowedSplits,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    pic_width: u32,
    pic_height: u32,
) -> SplitMode {
    if btt_split_flag {
        return match (btt_split_dir, btt_split_type) {
            (0, 0) => SplitMode::SplitBtHor,
            (0, _) => SplitMode::SplitTtHor, // errata-corrected (see doc)
            (_, 0) => SplitMode::SplitBtVer,
            (_, _) => SplitMode::SplitTtVer,
        };
    }

    let right_edge = x0 + (1 << log2_cb_width);
    let bottom_edge = y0 + (1 << log2_cb_height);

    // Boundary: extends past the right edge but fits vertically — split
    // vertically toward the in-picture columns.
    if right_edge > pic_width && bottom_edge <= pic_height {
        return if allowed.bt_ver {
            SplitMode::SplitBtVer
        } else {
            SplitMode::SplitBtHor
        };
    }

    // Boundary: extends past the bottom edge — split horizontally.
    if bottom_edge > pic_height {
        return if allowed.bt_hor {
            SplitMode::SplitBtHor
        } else {
            SplitMode::SplitBtVer
        };
    }

    SplitMode::NoSplit
}

/// Tallies of the BTT split-syntax bins consumed by [`decode_btt_split`].
/// Threaded back so fixture tests can assert the §7.3.8.3 presence gating
/// fired exactly as the spec requires.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BttSplitStats {
    /// `btt_split_flag` regular bins decoded (one per BTT decision point
    /// where at least one split is allowed).
    pub flag_bins: u32,
    /// `btt_split_dir` regular bins decoded (one per split where both a
    /// vertical and a horizontal split were possible).
    pub dir_bins: u32,
    /// `btt_split_type` regular bins decoded (one per split where both a
    /// binary and a ternary split were possible in the chosen direction).
    pub type_bins: u32,
}

/// Resolved outcome of a §7.3.8.3 `split_unit()` BTT split decision: the
/// final [`SplitMode`] plus the raw (parsed-or-inferred) flag values that
/// produced it. Returned by [`decode_btt_split`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BttSplit {
    /// The resolved coding-tree split shape.
    pub mode: SplitMode,
    /// `btt_split_flag` (parsed or inferred to 0 when not present).
    pub flag: bool,
    /// `btt_split_dir` (parsed or inferred per §7.4.8.3).
    pub dir: u32,
    /// `btt_split_type` (parsed or inferred per §7.4.8.3).
    pub split_type: u32,
}

/// §7.3.8.3 — CABAC-driven `split_unit()` BTT split-syntax reader for the
/// `sps_btt_flag == 1` path.
///
/// Given a coding block of size `2^log2_cb_width × 2^log2_cb_height` at
/// luma position (`x0`, `y0`), the precomputed [`AllowedSplits`] (from
/// [`derive_allowed_splits`]), and the picture dimensions, this consumes
/// the 0–3 `btt_split_*` bins the spec syntax (lines 2650–2658) signals,
/// applies the §7.4.8.3 inference for any absent element, and returns the
/// resolved [`SplitMode`] via [`derive_split_mode`].
///
/// Bin/context wiring (§9.3.3 binarization + Table 95 ctxInc):
///
/// * `btt_split_flag` — present only when `allowed.any()`. FL(cMax = 1),
///   context Table 42. ctxInc is 0 under `sps_cm_init_flag == 0`; under
///   `== 1` it is the §9.3.4.2.5 eq. (1440) value
///   `Min(numSmaller, 2) + 3 * ctxSetIdx` ([`ctx_inc_btt_split_flag`]),
///   where `num_smaller` (eq. 1439) is supplied by the caller from the
///   L/A/R neighbour block sizes.
/// * `btt_split_dir` — present only when [`btt_split_dir_present`]. FL(cMax
///   = 1), context Table 43. ctxInc is 0 under `sps_cm_init_flag == 0`;
///   under `== 1` it is `log2CbWidth − log2CbHeight + 2` (Table 95),
///   clamped into the table's 0..=4 range. Inferred via
///   [`infer_btt_split_dir`] when absent.
/// * `btt_split_type` — present only when [`btt_split_type_present`] for
///   the chosen direction. FL(cMax = 1), context Table 44, ctxInc 0
///   unconditionally. Inferred via [`infer_btt_split_type`] when absent.
///
/// `cm_init` is `sps_cm_init_flag`; `num_smaller` is the eq. (1439)
/// neighbour count (ignored when `cm_init` is `false`).
#[allow(clippy::too_many_arguments)]
pub fn decode_btt_split(
    eng: &mut CabacEngine,
    allowed: &AllowedSplits,
    cm_init: bool,
    num_smaller: u32,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    pic_width: u32,
    pic_height: u32,
    stats: &mut BttSplitStats,
) -> Result<BttSplit> {
    let n_cb_w = 1u32 << log2_cb_width;
    let n_cb_h = 1u32 << log2_cb_height;

    // --- btt_split_flag (spec lines 2650-2651) ---
    // Present only when at least one BTT split is allowed; otherwise it is
    // inferred to 0 (spec line 5424).
    let flag = if allowed.any() {
        let ctx_inc = if cm_init {
            ctx_inc_btt_split_flag(num_smaller, n_cb_w, n_cb_h)
        } else {
            0
        };
        let bin = eng.decode_decision(MainCtxTable::BttSplitFlag as usize, ctx_inc)?;
        stats.flag_bins += 1;
        bin != 0
    } else {
        false
    };

    if !flag {
        // No BTT split here — SplitMode comes from the picture-boundary
        // implicit-split rules (or NO_SPLIT) via derive_split_mode.
        let mode = derive_split_mode(
            false,
            0,
            0,
            allowed,
            x0,
            y0,
            log2_cb_width,
            log2_cb_height,
            pic_width,
            pic_height,
        );
        return Ok(BttSplit {
            mode,
            flag: false,
            dir: 0,
            split_type: 0,
        });
    }

    // --- btt_split_dir (spec lines 2653-2655) ---
    let dir = if btt_split_dir_present(allowed) {
        let ctx_inc = if cm_init {
            // Table 95: log2CbWidth − log2CbHeight + 2, clamped to the
            // Table 43 range (5 entries per init_type: indices 0..=4).
            let raw = log2_cb_width as i32 - log2_cb_height as i32 + 2;
            raw.clamp(0, 4) as usize
        } else {
            0
        };
        let bin = eng.decode_decision(MainCtxTable::BttSplitDir as usize, ctx_inc)?;
        stats.dir_bins += 1;
        bin as u32
    } else {
        infer_btt_split_dir(allowed)
    };

    // --- btt_split_type (spec lines 2656-2658) ---
    let split_type = if btt_split_type_present(allowed, dir) {
        // Table 44 / Table 95: single context, ctxInc 0 unconditionally.
        let bin = eng.decode_decision(MainCtxTable::BttSplitType as usize, 0)?;
        stats.type_bins += 1;
        bin as u32
    } else {
        infer_btt_split_type(allowed, dir)
    };

    let mode = derive_split_mode(
        true,
        dir,
        split_type,
        allowed,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        pic_width,
        pic_height,
    );

    Ok(BttSplit {
        mode,
        flag: true,
        dir,
        split_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default Main-profile-ish limits: 64×64 CTU (CtbLog2SizeY = 6),
    /// MinCbLog2SizeY = 2, all diff fields 0.
    fn default_limits() -> BttSizeLimits {
        BttSizeLimits::derive(6, 0, 0, 0, 0)
    }

    #[test]
    fn size_limits_match_spec_equations() {
        let l = default_limits();
        // eq. 43 / 44: both equal CtbLog2SizeY.
        assert_eq!(l.max_cb_log2_size_11_ratio, 6);
        assert_eq!(l.max_cb_log2_size_12_ratio, 6);
        // eq. 62 / 63 / 64: MinCbLog2SizeY, +1, +2.
        assert_eq!(l.min_cb_log2_size_11_ratio, 2);
        assert_eq!(l.min_cb_log2_size_12_ratio, 3);
        assert_eq!(l.min_cb_log2_size_14_ratio, 4);
        // eq. 65: Min(6 - 0, 6) = 6.
        assert_eq!(l.max_cb_log2_size_14_ratio, 6);
        // eq. 66: Min(6 - 0, 6) = 6.
        assert_eq!(l.max_tt_log2_size, 6);
        // eq. 67: 2 + 2 + 0 = 4.
        assert_eq!(l.min_tt_log2_size, 4);
    }

    #[test]
    fn size_limits_honour_diff_fields_and_tb_clamp() {
        // ctb=6, min_cb_minus2=1 (=> MinCbLog2SizeY=3),
        // diff14=1, diffTt=2, diffMinTt_minus2=1.
        let l = BttSizeLimits::derive(6, 1, 1, 2, 1);
        assert_eq!(l.min_cb_log2_size_11_ratio, 3);
        assert_eq!(l.min_cb_log2_size_12_ratio, 4);
        assert_eq!(l.min_cb_log2_size_14_ratio, 5);
        // eq. 65: Min(6-1, 6) = 5.
        assert_eq!(l.max_cb_log2_size_14_ratio, 5);
        // eq. 66: Min(6-2, 6) = 4.
        assert_eq!(l.max_tt_log2_size, 4);
        // eq. 67: 3 + 2 + 1 = 6.
        assert_eq!(l.min_tt_log2_size, 6);
    }

    #[test]
    fn tb_clamp_caps_max_tt_at_six() {
        // A hypothetical 128 CTU (ctb=7) with diff 0 clamps to 6.
        let l = BttSizeLimits::derive(7, 0, 0, 0, 0);
        assert_eq!(l.max_tt_log2_size, 6);
        assert_eq!(l.max_cb_log2_size_14_ratio, 6);
    }

    #[test]
    fn square_32x32_allows_binary_both_axes() {
        // 32×32 block (log2 5×5) under 64 CTU.
        let l = default_limits();
        let a = derive_allowed_splits(&l, 5, 5, PredModeConstraint::NotInterConstrained);
        // BtVer child is 16×32 -> log2w_temp=4, longer=5, ratio=1.
        // 1:2 window is [Min12=3, Max12=6] -> 5 in range -> allowed.
        assert!(a.bt_ver);
        assert!(a.bt_hor);
        // TtVer: log2w==log2h==5, in [Min14=4, Max14=6] -> allowed.
        assert!(a.tt_ver);
        assert!(a.tt_hor);
        assert!(a.any());
    }

    #[test]
    fn min_size_block_disallows_all_splits() {
        // 4×4 block (log2 2×2): BtVer child would be log2w_temp=1 < 2.
        let l = default_limits();
        let a = derive_allowed_splits(&l, 2, 2, PredModeConstraint::NotInterConstrained);
        assert!(!a.bt_ver);
        assert!(!a.bt_hor);
        // TtVer: log2_cb_width=2 < MinTtLog2Size=4 -> disallowed.
        assert!(!a.tt_ver);
        assert!(!a.tt_hor);
        assert!(!a.any());
    }

    #[test]
    fn ternary_needs_min_tt_size() {
        // 8×8 block (log2 3×3): MinTtLog2Size=4, so 3 < 4 disallows TT.
        let l = default_limits();
        let a = derive_allowed_splits(&l, 3, 3, PredModeConstraint::NotInterConstrained);
        assert!(!a.tt_ver);
        assert!(!a.tt_hor);
        // Binary still allowed: 8×8 -> child 4×8, log2w_temp=2, longer=3,
        // ratio=1 -> 1:2 window [3,6] -> 3 in range -> allowed.
        assert!(a.bt_ver);
        assert!(a.bt_hor);
    }

    #[test]
    fn ternary_disallowed_on_non_square_short_axis() {
        // 32×16 (log2 5×4): TtVer needs width>=height (5>=4 ok), width in
        // [MinTt=4, MaxTt=6] -> allowed. TtHor needs height>=width: 4<5
        // -> disallowed.
        let l = default_limits();
        let a = derive_allowed_splits(&l, 5, 4, PredModeConstraint::NotInterConstrained);
        assert!(a.tt_ver);
        assert!(!a.tt_hor);
    }

    #[test]
    fn inter_constraint_blocks_4x4_outcome_blocks() {
        // 8×4 (log2 3×2) under INTER: BtVer child is 4×4
        // (log2w_temp=2, h=2) -> the INTER 2&2 clause disallows it.
        let l = default_limits();
        let inter = derive_allowed_splits(&l, 3, 2, PredModeConstraint::Inter);
        assert!(!inter.bt_ver);
        // Without the constraint it would be allowed (1:2 window).
        let noc = derive_allowed_splits(&l, 3, 2, PredModeConstraint::NotInterConstrained);
        assert!(noc.bt_ver);
    }

    #[test]
    fn inter_constraint_blocks_tt_ver_16x4() {
        // 16×4 (log2 4×2) under INTER -> TtVer 4&2 clause disallows.
        let l = default_limits();
        let inter = derive_allowed_splits(&l, 4, 2, PredModeConstraint::Inter);
        assert!(!inter.tt_ver);
    }

    #[test]
    fn dir_and_type_signalling_predicates() {
        let both = AllowedSplits {
            bt_ver: true,
            bt_hor: true,
            tt_ver: true,
            tt_hor: true,
        };
        assert!(btt_split_dir_present(&both));
        assert!(btt_split_type_present(&both, 0));
        assert!(btt_split_type_present(&both, 1));

        // Only vertical splits possible -> dir not signalled, inferred 1.
        let vonly = AllowedSplits {
            bt_ver: true,
            bt_hor: false,
            tt_ver: true,
            tt_hor: false,
        };
        assert!(!btt_split_dir_present(&vonly));
        assert_eq!(infer_btt_split_dir(&vonly), 1);
        // type signalled for vertical (both bt+tt vertical present).
        assert!(btt_split_type_present(&vonly, 1));

        // Only BT-horizontal possible -> dir not signalled, inferred 0;
        // type not signalled (no TT horizontal), inferred 0 (binary).
        let bthor = AllowedSplits {
            bt_ver: false,
            bt_hor: true,
            tt_ver: false,
            tt_hor: false,
        };
        assert!(!btt_split_dir_present(&bthor));
        assert_eq!(infer_btt_split_dir(&bthor), 0);
        assert!(!btt_split_type_present(&bthor, 0));
        assert_eq!(infer_btt_split_type(&bthor, 0), 0);

        // Only TT-horizontal possible -> type inferred 1 (ternary).
        let tthor = AllowedSplits {
            bt_ver: false,
            bt_hor: false,
            tt_ver: false,
            tt_hor: true,
        };
        assert_eq!(infer_btt_split_type(&tthor, 0), 1);
    }

    #[test]
    fn split_mode_from_flags_covers_four_shapes() {
        let a = AllowedSplits {
            bt_ver: true,
            bt_hor: true,
            tt_ver: true,
            tt_hor: true,
        };
        // dir=0 (horizontal), type=0 -> BT_HOR.
        assert_eq!(
            derive_split_mode(true, 0, 0, &a, 0, 0, 5, 5, 256, 256),
            SplitMode::SplitBtHor
        );
        // dir=0 (horizontal), type=1 -> TT_HOR (errata-corrected).
        assert_eq!(
            derive_split_mode(true, 0, 1, &a, 0, 0, 5, 5, 256, 256),
            SplitMode::SplitTtHor
        );
        // dir=1 (vertical), type=0 -> BT_VER.
        assert_eq!(
            derive_split_mode(true, 1, 0, &a, 0, 0, 5, 5, 256, 256),
            SplitMode::SplitBtVer
        );
        // dir=1 (vertical), type=1 -> TT_VER.
        assert_eq!(
            derive_split_mode(true, 1, 1, &a, 0, 0, 5, 5, 256, 256),
            SplitMode::SplitTtVer
        );
    }

    #[test]
    fn split_mode_no_flag_in_picture_is_no_split() {
        let a = AllowedSplits {
            bt_ver: true,
            bt_hor: true,
            tt_ver: true,
            tt_hor: true,
        };
        assert_eq!(
            derive_split_mode(false, 0, 0, &a, 0, 0, 6, 6, 256, 256),
            SplitMode::NoSplit
        );
    }

    #[test]
    fn split_mode_right_boundary_forces_vertical() {
        // 64×64 block at x0=192 in a 200-wide picture: right edge 256 >
        // 200, bottom edge 64 <= 256 -> vertical boundary split.
        let a = AllowedSplits {
            bt_ver: true,
            bt_hor: true,
            tt_ver: true,
            tt_hor: true,
        };
        assert_eq!(
            derive_split_mode(false, 0, 0, &a, 192, 0, 6, 6, 200, 256),
            SplitMode::SplitBtVer
        );
        // If BtVer disallowed, falls back to BtHor.
        let no_bv = AllowedSplits { bt_ver: false, ..a };
        assert_eq!(
            derive_split_mode(false, 0, 0, &no_bv, 192, 0, 6, 6, 200, 256),
            SplitMode::SplitBtHor
        );
    }

    #[test]
    fn split_mode_bottom_boundary_forces_horizontal() {
        // bottom edge past picture height -> horizontal boundary split.
        let a = AllowedSplits {
            bt_ver: true,
            bt_hor: true,
            tt_ver: true,
            tt_hor: true,
        };
        assert_eq!(
            derive_split_mode(false, 0, 0, &a, 0, 192, 6, 6, 256, 200),
            SplitMode::SplitBtHor
        );
        let no_bh = AllowedSplits { bt_hor: false, ..a };
        assert_eq!(
            derive_split_mode(false, 0, 0, &no_bh, 0, 192, 6, 6, 256, 200),
            SplitMode::SplitBtVer
        );
    }

    // -----------------------------------------------------------------
    // decode_btt_split — §7.3.8.3 CABAC-driven split-syntax reader.
    // Each test encodes the exact bin sequence the spec syntax signals
    // with the symmetric in-test CabacEncoder, then asserts the decoder
    // consumes precisely those bins (presence gating) and resolves the
    // expected SplitMode. cm_init == false, so every ctxInc is 0 and the
    // engine + encoder share the default (256, 0) context slot.
    // -----------------------------------------------------------------

    use crate::cabac::{CabacEncoder, CabacEngine};

    const ALL_ALLOWED: AllowedSplits = AllowedSplits {
        bt_ver: true,
        bt_hor: true,
        tt_ver: true,
        tt_hor: true,
    };

    /// Encode `bins` against `(ctx_table, 0)` slots, terminate, and build a
    /// fresh decode engine over the produced RBSP.
    fn engine_for(bins: &[(usize, u8)]) -> Vec<u8> {
        let mut enc = CabacEncoder::new();
        for &(ctx_table, bin) in bins {
            enc.encode_decision(ctx_table, 0, bin);
        }
        enc.encode_terminate(true);
        enc.finish()
    }

    #[test]
    fn decode_btt_split_binary_horizontal() {
        // 32×32 block, all splits allowed. Encode flag=1, dir=0, type=0 →
        // SPLIT_BT_HOR. All three elements present (both axes + both
        // shapes available).
        let flag_t = MainCtxTable::BttSplitFlag as usize;
        let dir_t = MainCtxTable::BttSplitDir as usize;
        let type_t = MainCtxTable::BttSplitType as usize;
        let rbsp = engine_for(&[(flag_t, 1), (dir_t, 0), (type_t, 0)]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out = decode_btt_split(
            &mut eng,
            &ALL_ALLOWED,
            false,
            0,
            0,
            0,
            5,
            5,
            256,
            256,
            &mut stats,
        )
        .unwrap();
        assert_eq!(out.mode, SplitMode::SplitBtHor);
        assert!(out.flag);
        assert_eq!(out.dir, 0);
        assert_eq!(out.split_type, 0);
        assert_eq!(stats.flag_bins, 1);
        assert_eq!(stats.dir_bins, 1);
        assert_eq!(stats.type_bins, 1);
    }

    #[test]
    fn decode_btt_split_ternary_vertical() {
        // flag=1, dir=1 (vertical), type=1 (ternary) → SPLIT_TT_VER.
        let flag_t = MainCtxTable::BttSplitFlag as usize;
        let dir_t = MainCtxTable::BttSplitDir as usize;
        let type_t = MainCtxTable::BttSplitType as usize;
        let rbsp = engine_for(&[(flag_t, 1), (dir_t, 1), (type_t, 1)]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out = decode_btt_split(
            &mut eng,
            &ALL_ALLOWED,
            false,
            0,
            0,
            0,
            5,
            5,
            256,
            256,
            &mut stats,
        )
        .unwrap();
        assert_eq!(out.mode, SplitMode::SplitTtVer);
        assert_eq!(out.dir, 1);
        assert_eq!(out.split_type, 1);
        assert_eq!(stats.dir_bins, 1);
        assert_eq!(stats.type_bins, 1);
    }

    #[test]
    fn decode_btt_split_type1_horizontal_is_tt_hor() {
        // flag=1, dir=0, type=1 → SPLIT_TT_HOR (errata-corrected branch).
        let flag_t = MainCtxTable::BttSplitFlag as usize;
        let dir_t = MainCtxTable::BttSplitDir as usize;
        let type_t = MainCtxTable::BttSplitType as usize;
        let rbsp = engine_for(&[(flag_t, 1), (dir_t, 0), (type_t, 1)]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out = decode_btt_split(
            &mut eng,
            &ALL_ALLOWED,
            false,
            0,
            0,
            0,
            5,
            5,
            256,
            256,
            &mut stats,
        )
        .unwrap();
        assert_eq!(out.mode, SplitMode::SplitTtHor);
    }

    #[test]
    fn decode_btt_split_flag_zero_in_picture_is_no_split() {
        // flag=0 with the block fully inside the picture → NO_SPLIT, and
        // only the single flag bin is consumed (no dir/type reads).
        let flag_t = MainCtxTable::BttSplitFlag as usize;
        let rbsp = engine_for(&[(flag_t, 0)]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out = decode_btt_split(
            &mut eng,
            &ALL_ALLOWED,
            false,
            0,
            0,
            0,
            5,
            5,
            256,
            256,
            &mut stats,
        )
        .unwrap();
        assert_eq!(out.mode, SplitMode::NoSplit);
        assert!(!out.flag);
        assert_eq!(stats.flag_bins, 1);
        assert_eq!(stats.dir_bins, 0);
        assert_eq!(stats.type_bins, 0);
    }

    #[test]
    fn decode_btt_split_dir_inferred_no_bin() {
        // Only vertical splits possible (both bt+tt vertical) → dir not
        // signalled (inferred 1), but type IS signalled (both vertical
        // shapes). Encode flag=1, type=0 → SPLIT_BT_VER.
        let allowed = AllowedSplits {
            bt_ver: true,
            bt_hor: false,
            tt_ver: true,
            tt_hor: false,
        };
        let flag_t = MainCtxTable::BttSplitFlag as usize;
        let type_t = MainCtxTable::BttSplitType as usize;
        let rbsp = engine_for(&[(flag_t, 1), (type_t, 0)]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out = decode_btt_split(
            &mut eng, &allowed, false, 0, 0, 0, 5, 5, 256, 256, &mut stats,
        )
        .unwrap();
        assert_eq!(out.mode, SplitMode::SplitBtVer);
        assert_eq!(out.dir, 1, "dir inferred to 1 (vertical-only)");
        assert_eq!(stats.dir_bins, 0, "dir not signalled");
        assert_eq!(stats.type_bins, 1, "type signalled (both vertical shapes)");
    }

    #[test]
    fn decode_btt_split_type_inferred_no_bin() {
        // Only BT-horizontal possible → dir inferred 0, type inferred 0
        // (no TT horizontal). Encode flag=1 only → SPLIT_BT_HOR.
        let allowed = AllowedSplits {
            bt_ver: false,
            bt_hor: true,
            tt_ver: false,
            tt_hor: false,
        };
        let flag_t = MainCtxTable::BttSplitFlag as usize;
        let rbsp = engine_for(&[(flag_t, 1)]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out = decode_btt_split(
            &mut eng, &allowed, false, 0, 0, 0, 5, 5, 256, 256, &mut stats,
        )
        .unwrap();
        assert_eq!(out.mode, SplitMode::SplitBtHor);
        assert_eq!(out.dir, 0);
        assert_eq!(out.split_type, 0);
        assert_eq!(stats.dir_bins, 0);
        assert_eq!(stats.type_bins, 0);
    }

    #[test]
    fn decode_btt_split_no_allowed_infers_flag_zero_no_bins() {
        // No split allowed at all → btt_split_flag not present (inferred
        // 0); zero bins consumed before terminate.
        let none = AllowedSplits {
            bt_ver: false,
            bt_hor: false,
            tt_ver: false,
            tt_hor: false,
        };
        let rbsp = engine_for(&[]);
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut stats = BttSplitStats::default();
        let out =
            decode_btt_split(&mut eng, &none, false, 0, 0, 0, 4, 4, 256, 256, &mut stats).unwrap();
        assert_eq!(out.mode, SplitMode::NoSplit);
        assert!(!out.flag);
        assert_eq!(stats.flag_bins, 0);
    }
}
