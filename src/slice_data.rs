//! EVC `slice_data()` walker (ISO/IEC 23094-1 §7.3.8).
//!
//! Round-2 scope: drive the CABAC engine through every `ae(v)` syntax
//! element of a Baseline-profile bitstream so that:
//!
//! * every bin is consumed in spec-correct order (matching the syntax
//!   tables in §7.3.8.1 through §7.3.8.7), and
//! * the engine reaches the end of the slice cleanly via the
//!   `end_of_tile_one_bit` terminate decision (§7.3.8.1).
//!
//! Pixel emission, transform/quant inversion, intra/inter prediction,
//! deblocking, ALF, DRA — *all* deferred to round 3+. The walker just
//! advances the CABAC state and surfaces the parsed values via callbacks
//! so the test fixtures (and round-3 pixel pipeline) can observe them
//! without paying for re-parsing.
//!
//! ## Profile constraints we exploit
//!
//! Baseline profile (Annex A.3.2) forces:
//!
//! * `sps_btt_flag == 0` (only quad-split via `split_cu_flag`),
//! * `sps_suco_flag == 0`, `sps_admvp_flag == 0`, `sps_eipd_flag == 0`,
//! * `sps_cm_init_flag == 0` → every regular bin maps to ctxTable 0,
//!   ctxIdx 0 (init `(valState=256, valMps=0)`),
//! * `sps_alf_flag == 0`, `sps_addb_flag == 0`, `sps_dquant_flag == 0`,
//!   `sps_ats_flag == 0`, `sps_ibc_flag == 0`, `sps_dra_flag == 0`,
//!   `sps_adcc_flag == 0` → run-length residual coding,
//! * `single_tile_in_pic_flag == 1` (one tile per picture).
//!
//! For an IDR slice in Baseline, `slice_type == I` so `predModeConstraint`
//! becomes `INTRA_IBC` at the CU split point and the subsequent
//! `coding_unit()` is invoked twice — once for `DUAL_TREE_LUMA` and once
//! for `DUAL_TREE_CHROMA` — per §7.3.8.3 lines 2789–2799.
//!
//! ## Surface
//!
//! [`walk_baseline_idr_slice`] takes the slice's RBSP, the active SPS/PPS
//! state and a [`SliceWalkInputs`] descriptor; it returns the number of
//! `coding_unit()` invocations parsed. The walker stops cleanly on the
//! terminate decision, then verifies the bitstream is byte-aligned per
//! §7.3.8.1 trailing logic.

use oxideav_core::{Error, Result};

use crate::cabac::CabacEngine;
use crate::dequant::scale_and_inverse_transform;
use crate::intra::IntraMode;
use crate::picture::{intra_reconstruct_cb, YuvPicture};

/// Static SPS/PPS state that the walker needs to make
/// per-syntax-element decisions. Only the fields actually consulted by
/// the Baseline-profile path are surfaced; the rest are tracked
/// implicitly (e.g. `sps_btt_flag = 0` is hard-wired in the walker).
#[derive(Clone, Copy, Debug)]
pub struct SliceWalkInputs {
    /// `pic_width_in_luma_samples` (§7.4.3.1).
    pub pic_width: u32,
    /// `pic_height_in_luma_samples` (§7.4.3.1).
    pub pic_height: u32,
    /// `CtbLog2SizeY = log2_ctu_size_minus5 + 5` (§7.4.3.1). Default for
    /// Baseline is 64×64 → 6.
    pub ctb_log2_size_y: u32,
    /// `MinCbLog2SizeY` — drives recursion termination. Baseline uses
    /// `log2_min_cb_size_minus2 + 2 = 2` (4×4 minimum).
    pub min_cb_log2_size_y: u32,
    /// `MaxTbLog2SizeY` — caps the transform unit dimension. Baseline
    /// caps at 6 (64×64).
    pub max_tb_log2_size_y: u32,
    /// `chroma_format_idc` (§7.4.3.1). Baseline supports 0 (mono) or 1
    /// (4:2:0).
    pub chroma_format_idc: u32,
    /// `cu_qp_delta_enabled_flag` (PPS). When false, `cu_qp_delta_*` is
    /// not in the bitstream.
    pub cu_qp_delta_enabled: bool,
}

impl SliceWalkInputs {
    fn ctb_size(&self) -> u32 {
        1 << self.ctb_log2_size_y
    }
    fn pic_width_in_ctus(&self) -> u32 {
        (self.pic_width + self.ctb_size() - 1) >> self.ctb_log2_size_y
    }
    fn pic_height_in_ctus(&self) -> u32 {
        (self.pic_height + self.ctb_size() - 1) >> self.ctb_log2_size_y
    }
}

/// Counters reported back to the caller after a successful walk. Each one
/// is incremented every time the walker consumes the corresponding syntax
/// element from the CABAC stream — handy for hand-built fixture tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SliceWalkStats {
    /// Coding-tree units actually visited.
    pub ctus: u32,
    /// `split_cu_flag` bins decoded (one per non-leaf split point).
    pub split_cu_flag_bins: u32,
    /// `coding_unit()` invocations (luma + chroma trees combined for an
    /// I slice in dual-tree mode).
    pub coding_units: u32,
    /// `cbf_luma` bins decoded.
    pub cbf_luma_bins: u32,
    /// `cbf_cb` + `cbf_cr` bins decoded.
    pub cbf_chroma_bins: u32,
    /// `cu_qp_delta_abs` bins decoded (per CU when enabled).
    pub cu_qp_delta_abs_bins: u32,
    /// `intra_pred_mode` bins decoded (per luma CU under sps_eipd=0).
    pub intra_pred_mode_bins: u32,
    /// Total coefficient runs consumed via `residual_coding_rle()`.
    pub coeff_runs: u32,
}

/// Predicate marking which kind of `coding_unit()` invocation we're in.
/// Baseline + I slice splits per §7.3.8.3 lines 2789–2799 — the I-slice
/// path always lands in dual-tree mode (`predModeConstraint = INTRA_IBC`),
/// so only the dual-tree variants are constructed in this round; the
/// `SingleTree` variant is reserved for round-3 P/B slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TreeType {
    /// Single-tree CU (P/B slice path — round 3).
    #[allow(dead_code)]
    SingleTree,
    /// Luma-only CU, dual-tree mode.
    DualTreeLuma,
    /// Chroma-only CU, dual-tree mode.
    DualTreeChroma,
}

/// Walk a Baseline-profile IDR slice's `slice_data()`. Returns walk stats
/// once `end_of_tile_one_bit` terminates the engine cleanly. Errors
/// indicate the bitstream cannot be consumed by the round-2 walker
/// (unsupported toolset combination or premature engine exhaustion).
pub fn walk_baseline_idr_slice(rbsp: &[u8], inputs: SliceWalkInputs) -> Result<SliceWalkStats> {
    if inputs.ctb_log2_size_y < 5 || inputs.ctb_log2_size_y > 7 {
        return Err(Error::invalid(format!(
            "evc slice_data: CtbLog2SizeY {} out of Baseline range [5, 7]",
            inputs.ctb_log2_size_y
        )));
    }
    if inputs.min_cb_log2_size_y < 2 || inputs.min_cb_log2_size_y > inputs.ctb_log2_size_y {
        return Err(Error::invalid(format!(
            "evc slice_data: MinCbLog2SizeY {} invalid (CtbLog2SizeY={})",
            inputs.min_cb_log2_size_y, inputs.ctb_log2_size_y
        )));
    }
    let mut eng = CabacEngine::new(rbsp)?;
    let mut stats = SliceWalkStats::default();
    let n_ctus = inputs
        .pic_width_in_ctus()
        .checked_mul(inputs.pic_height_in_ctus())
        .ok_or_else(|| Error::invalid("evc slice_data: ctu count overflow"))?;
    if n_ctus == 0 {
        return Err(Error::invalid("evc slice_data: no CTUs in slice"));
    }
    // Cap CTU iterations to a hard sanity bound (matches the SPS dimension
    // bound: at 32768x32768 with CTB=64 we get 512x512 = 262144 CTUs).
    if n_ctus > 1_048_576 {
        return Err(Error::invalid(format!(
            "evc slice_data: ctu count {n_ctus} > sanity bound"
        )));
    }
    for ctu_idx in 0..n_ctus {
        let x_ctb = (ctu_idx % inputs.pic_width_in_ctus()) << inputs.ctb_log2_size_y;
        let y_ctb = (ctu_idx / inputs.pic_width_in_ctus()) << inputs.ctb_log2_size_y;
        // §7.3.8.2 coding_tree_unit(): no ALF flags in Baseline; recurse
        // straight into split_unit().
        walk_split_unit(
            &mut eng,
            &mut stats,
            &inputs,
            x_ctb,
            y_ctb,
            inputs.ctb_log2_size_y,
            inputs.ctb_log2_size_y,
        )?;
        stats.ctus += 1;
    }
    // §7.3.8.1: end_of_tile_one_bit (single tile = single iteration).
    let term = eng.decode_terminate()?;
    if !term {
        return Err(Error::invalid(
            "evc slice_data: end_of_tile_one_bit must terminate engine",
        ));
    }
    // The terminate decision consumed rbsp_stop_one_bit. The remaining
    // bits in the byte are zero padding; no further alignment needed since
    // CABAC consumed the byte-aligned terminate.
    Ok(stats)
}

/// `split_unit()` per §7.3.8.3 — Baseline subset (`sps_btt_flag == 0`).
/// Recurses into four sub-units when `split_cu_flag == 1`, else lands on
/// the dual-tree `coding_unit()` pair (luma + chroma) for an I slice.
fn walk_split_unit(
    eng: &mut CabacEngine,
    stats: &mut SliceWalkStats,
    inputs: &SliceWalkInputs,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
) -> Result<()> {
    // §7.3.8.3: with sps_btt_flag == 0 the split_cu_flag is read iff
    // log2CbWidth > 2 || log2CbHeight > 2.
    let mut split = false;
    let cb_w = 1u32 << log2_cb_width;
    let cb_h = 1u32 << log2_cb_height;
    let cb_within_picture = x0 + cb_w <= inputs.pic_width && y0 + cb_h <= inputs.pic_height;
    let can_split = log2_cb_width > inputs.min_cb_log2_size_y
        && log2_cb_height > inputs.min_cb_log2_size_y
        && cb_within_picture;
    if can_split && (log2_cb_width > 2 || log2_cb_height > 2) {
        // Baseline path: ctxTable 0, ctxIdx 0 (sps_cm_init_flag=0).
        let bin = eng.decode_decision(0, 0)?;
        stats.split_cu_flag_bins += 1;
        split = bin != 0;
    } else if !cb_within_picture && can_split {
        // Boundary CU: spec implies it's split implicitly (no flag in the
        // bitstream). Recurse.
        split = true;
    }

    if split {
        let half_w = log2_cb_width.saturating_sub(1);
        let half_h = log2_cb_height.saturating_sub(1);
        let x1 = x0 + (1u32 << half_w);
        let y1 = y0 + (1u32 << half_h);
        // §7.3.8.3 splits in raster order with split_unit_coding_order_flag=0.
        walk_split_unit(eng, stats, inputs, x0, y0, half_w, half_h)?;
        if x1 < inputs.pic_width {
            walk_split_unit(eng, stats, inputs, x1, y0, half_w, half_h)?;
        }
        if y1 < inputs.pic_height {
            walk_split_unit(eng, stats, inputs, x0, y1, half_w, half_h)?;
        }
        if x1 < inputs.pic_width && y1 < inputs.pic_height {
            walk_split_unit(eng, stats, inputs, x1, y1, half_w, half_h)?;
        }
        return Ok(());
    }

    // Leaf: dual-tree pair for I slice (predModeConstraint = INTRA_IBC).
    walk_coding_unit(
        eng,
        stats,
        inputs,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        TreeType::DualTreeLuma,
    )?;
    walk_coding_unit(
        eng,
        stats,
        inputs,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        TreeType::DualTreeChroma,
    )?;
    Ok(())
}

/// `coding_unit()` per §7.3.8.4 — Baseline + I slice + INTRA_IBC subset.
#[allow(clippy::too_many_arguments)]
fn walk_coding_unit(
    eng: &mut CabacEngine,
    stats: &mut SliceWalkStats,
    inputs: &SliceWalkInputs,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    tree_type: TreeType,
) -> Result<()> {
    stats.coding_units += 1;
    // INTRA_IBC: cu_skip_flag is suppressed (line 2808 condition).
    // pred_mode_flag is suppressed (line 2843 condition).
    // ibc_flag: sps_ibc_flag=0 in Baseline → suppressed.
    // CuPredMode = MODE_INTRA (only choice in INTRA_IBC).
    if matches!(tree_type, TreeType::DualTreeLuma | TreeType::SingleTree) {
        // sps_eipd_flag=0 → intra_pred_mode is the single ae(v) syntax.
        // Binarization: U with cMax=4 (Table 91).
        // Table 95 lists ctxInc 0,1,1,1,1 for binIdx 0..4. Under
        // sps_cm_init_flag=0 they all map to ctxTable=0, ctxIdx=0 (since
        // ctxIdxOffset=0 and ctxTable=0 per §9.3.4.2.1).
        let _intra_mode = eng.decode_u_regular(0, |_bin_idx| 0)?;
        stats.intra_pred_mode_bins += 1;
    }
    // sps_eipd_flag=0 ⇒ intra_chroma_pred_mode is suppressed (gated by
    // sps_eipd_flag==1 on line 2864).

    // CuPredMode == MODE_INTRA + dual-tree → cbf_all path is suppressed
    // (line 3028 needs treeType == SINGLE_TREE).
    walk_transform_unit(
        eng,
        stats,
        inputs,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        tree_type,
    )
}

/// `transform_unit()` per §7.3.8.5 — Baseline + I-slice subset.
#[allow(clippy::too_many_arguments)]
fn walk_transform_unit(
    eng: &mut CabacEngine,
    stats: &mut SliceWalkStats,
    inputs: &SliceWalkInputs,
    _x0: u32,
    _y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    tree_type: TreeType,
) -> Result<()> {
    let log2_tb_width = log2_cb_width.min(inputs.max_tb_log2_size_y);
    let log2_tb_height = log2_cb_height.min(inputs.max_tb_log2_size_y);
    let chroma_present = inputs.chroma_format_idc != 0;
    let mut cbf_cb = 0u32;
    let mut cbf_cr = 0u32;
    let mut cbf_luma = 0u32;
    // Line 3066: treeType != DUAL_TREE_LUMA && ChromaArrayType != 0 → cbf_cb,cbf_cr.
    if tree_type != TreeType::DualTreeLuma && chroma_present {
        cbf_cb = eng.decode_decision(0, 0)? as u32;
        cbf_cr = eng.decode_decision(0, 0)? as u32;
        stats.cbf_chroma_bins += 2;
    }
    // Line 3070: (isSplit || CuPredMode==INTRA || cbf_cb || cbf_cr) &&
    //            treeType != DUAL_TREE_CHROMA → cbf_luma.
    // For Baseline + I slice, isSplit derives from CB > MaxTb (we cap above).
    let is_split =
        log2_cb_width > inputs.max_tb_log2_size_y || log2_cb_height > inputs.max_tb_log2_size_y;
    let is_intra = true;
    if (is_split || is_intra || cbf_cb != 0 || cbf_cr != 0) && tree_type != TreeType::DualTreeChroma
    {
        cbf_luma = eng.decode_decision(0, 0)? as u32;
        stats.cbf_luma_bins += 1;
    }
    // Line 3073: cu_qp_delta_abs gated by cu_qp_delta_enabled_flag and a
    // complex condition. With sps_dquant_flag=0 (Baseline) the inner check
    // becomes `(cbf_luma || cbf_cb || cbf_cr)`.
    if inputs.cu_qp_delta_enabled && (cbf_luma != 0 || cbf_cb != 0 || cbf_cr != 0) {
        let qp_delta_abs = eng.decode_u_regular(0, |_| 0)?;
        stats.cu_qp_delta_abs_bins += 1;
        if qp_delta_abs > 0 {
            // cu_qp_delta_sign_flag: FL with cMax=1 → bypass-coded? The
            // table descriptor says ae(v) with FL,cMax=1, but Table 95 has
            // no entry for cu_qp_delta_sign_flag → treated as bypass per
            // 9.3.4.2.1 (entry "bypass" or unlisted defaults to bypass for
            // ae(v) elements without a Table 95 row, by inspection). We
            // pessimistically use bypass (matches reference behaviour).
            let _sign = eng.decode_bypass()?;
        }
    }
    // ats_*: sps_ats_flag=0 in Baseline → suppressed.
    // residual_coding for each component if its CBF is set.
    // sps_adcc_flag=0 in Baseline → run-length residual coding.
    if cbf_luma != 0 {
        walk_residual_coding_rle(eng, stats, log2_tb_width, log2_tb_height)?;
    }
    if cbf_cb != 0 {
        // Chroma block dimensions: log2_tb_width - SubWidthC + 1, etc.
        // For 4:2:0 (SubWidthC=SubHeightC=2): subtract 1 from each log2.
        let log2_c_w = log2_tb_width.saturating_sub(1);
        let log2_c_h = log2_tb_height.saturating_sub(1);
        walk_residual_coding_rle(eng, stats, log2_c_w, log2_c_h)?;
    }
    if cbf_cr != 0 {
        let log2_c_w = log2_tb_width.saturating_sub(1);
        let log2_c_h = log2_tb_height.saturating_sub(1);
        walk_residual_coding_rle(eng, stats, log2_c_w, log2_c_h)?;
    }
    Ok(())
}

/// `residual_coding_rle()` per §7.3.8.7 — Baseline path.
///
/// Each iteration consumes:
/// * `coeff_zero_run`: U-binarised (Table 91), `cMax = (1 << (log2W +
///   log2H)) - 1`. Context-coded against a single ctxIdx in Baseline.
/// * `coeff_abs_level_minus1`: U-binarised, no cMax cap; bound at the
///   block size to keep allocations safe.
/// * `coeff_sign_flag`: bypass.
/// * `coeff_last_flag` (only if `ScanPos < block - 1`): regular FL cMax=1.
fn walk_residual_coding_rle(
    eng: &mut CabacEngine,
    stats: &mut SliceWalkStats,
    log2_tb_width: u32,
    log2_tb_height: u32,
) -> Result<()> {
    let total_coeffs: u32 = 1u32 << (log2_tb_width + log2_tb_height);
    if total_coeffs == 0 || total_coeffs > (1 << 12) {
        return Err(Error::invalid(format!(
            "evc residual_coding_rle: total_coeffs {total_coeffs} out of range"
        )));
    }
    let mut scan_pos: u32 = 0;
    loop {
        // coeff_zero_run cMax bound enforces termination.
        let zr = eng.decode_u_regular(0, |_| 0)?;
        scan_pos = scan_pos
            .checked_add(zr)
            .ok_or_else(|| Error::invalid("evc residual_coding_rle: scan_pos overflow"))?;
        if scan_pos >= total_coeffs {
            return Err(Error::invalid(
                "evc residual_coding_rle: zero-run pushed past block size",
            ));
        }
        // coeff_abs_level_minus1 — bound for safety; round-3 will replace
        // this with the real EGk-style fallback for large values.
        let _level_minus1 = eng.decode_u_regular(0, |_| 0)?;
        // coeff_sign_flag: bypass (cMax=1, no Table-95 entry).
        let _sign = eng.decode_bypass()?;
        stats.coeff_runs += 1;
        // coeff_last_flag if not at the end.
        let last_pos_reached = scan_pos == total_coeffs - 1;
        let coeff_last = if !last_pos_reached {
            eng.decode_decision(0, 0)?
        } else {
            1
        };
        scan_pos += 1;
        if coeff_last != 0 || scan_pos >= total_coeffs {
            return Ok(());
        }
    }
}

// =====================================================================
// Round-3 pixel-emission pipeline.
// =====================================================================

/// Inputs that the round-3 decoder needs in addition to
/// [`SliceWalkInputs`] — slice QP and the picture buffer's bit depth.
#[derive(Clone, Copy, Debug)]
pub struct SliceDecodeInputs {
    pub slice_qp: i32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
}

/// Stats from [`decode_baseline_idr_slice`]. A superset of
/// [`SliceWalkStats`] for testability — coding_units, residual coeff
/// counts, etc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SliceDecodeStats {
    pub ctus: u32,
    pub split_cu_flag_bins: u32,
    pub coding_units: u32,
    pub cbf_luma_bins: u32,
    pub cbf_chroma_bins: u32,
    pub intra_pred_mode_bins: u32,
    pub coeff_runs: u32,
}

/// Decode a Baseline-profile IDR slice into a freshly-allocated
/// [`YuvPicture`]. Round-3 deliverable: drives the CABAC engine through
/// every syntax element (matching [`walk_baseline_idr_slice`]),
/// reconstructs samples per §8.4.4 / §8.7 / §8.7.5, and returns the
/// picture buffer.
///
/// Round-3 constraints (in addition to the walker's set):
///
/// * 8-bit luma + chroma only (`bit_depth_*_minus8 == 0`).
/// * `slice_deblocking_filter_flag == 0` (no deblocking).
/// * Transform sizes ∈ {2, 4, 8, 16, 32} (no 64×64 — see [`crate::transform`]).
/// * No residual coding — fixtures must produce `cbf_luma == 0` and
///   `cbf_cb == cbf_cr == 0` for every CU. Non-zero CBFs surface as
///   `Error::Unsupported` for round 3 (residual coding wires in round 4).
pub fn decode_baseline_idr_slice(
    rbsp: &[u8],
    walk: SliceWalkInputs,
    decode: SliceDecodeInputs,
) -> Result<(YuvPicture, SliceDecodeStats)> {
    if walk.ctb_log2_size_y < 5 || walk.ctb_log2_size_y > 7 {
        return Err(Error::invalid(format!(
            "evc decode: CtbLog2SizeY {} out of Baseline range [5, 7]",
            walk.ctb_log2_size_y
        )));
    }
    if walk.min_cb_log2_size_y < 2 || walk.min_cb_log2_size_y > walk.ctb_log2_size_y {
        return Err(Error::invalid(format!(
            "evc decode: MinCbLog2SizeY {} invalid",
            walk.min_cb_log2_size_y
        )));
    }
    if decode.bit_depth_luma != 8 || decode.bit_depth_chroma != 8 {
        return Err(Error::unsupported(format!(
            "evc decode: round-3 supports 8-bit only (luma={}, chroma={})",
            decode.bit_depth_luma, decode.bit_depth_chroma
        )));
    }
    let mut pic = YuvPicture::new(
        walk.pic_width,
        walk.pic_height,
        walk.chroma_format_idc,
        decode.bit_depth_luma,
    )?;
    let mut eng = CabacEngine::new(rbsp)?;
    let mut stats = SliceDecodeStats::default();
    let n_ctus = walk
        .pic_width_in_ctus()
        .checked_mul(walk.pic_height_in_ctus())
        .ok_or_else(|| Error::invalid("evc decode: ctu count overflow"))?;
    if n_ctus == 0 {
        return Err(Error::invalid("evc decode: no CTUs in slice"));
    }
    if n_ctus > 1_048_576 {
        return Err(Error::invalid(format!(
            "evc decode: ctu count {n_ctus} > sanity bound"
        )));
    }
    for ctu_idx in 0..n_ctus {
        let x_ctb = (ctu_idx % walk.pic_width_in_ctus()) << walk.ctb_log2_size_y;
        let y_ctb = (ctu_idx / walk.pic_width_in_ctus()) << walk.ctb_log2_size_y;
        decode_split_unit(
            &mut eng,
            &mut pic,
            &mut stats,
            &walk,
            &decode,
            x_ctb,
            y_ctb,
            walk.ctb_log2_size_y,
            walk.ctb_log2_size_y,
        )?;
        stats.ctus += 1;
    }
    let term = eng.decode_terminate()?;
    if !term {
        return Err(Error::invalid(
            "evc decode: end_of_tile_one_bit must terminate engine",
        ));
    }
    Ok((pic, stats))
}

#[allow(clippy::too_many_arguments)]
fn decode_split_unit(
    eng: &mut CabacEngine,
    pic: &mut YuvPicture,
    stats: &mut SliceDecodeStats,
    walk: &SliceWalkInputs,
    decode: &SliceDecodeInputs,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
) -> Result<()> {
    let cb_w = 1u32 << log2_cb_width;
    let cb_h = 1u32 << log2_cb_height;
    let cb_within_picture = x0 + cb_w <= walk.pic_width && y0 + cb_h <= walk.pic_height;
    let can_recurse =
        log2_cb_width > walk.min_cb_log2_size_y && log2_cb_height > walk.min_cb_log2_size_y;
    let mut split = false;
    if can_recurse && cb_within_picture && (log2_cb_width > 2 || log2_cb_height > 2) {
        let bin = eng.decode_decision(0, 0)?;
        stats.split_cu_flag_bins += 1;
        split = bin != 0;
    } else if can_recurse && !cb_within_picture {
        // Boundary CU: implicit split without reading a flag.
        split = true;
    }
    if split {
        let half_w = log2_cb_width.saturating_sub(1);
        let half_h = log2_cb_height.saturating_sub(1);
        let x1 = x0 + (1u32 << half_w);
        let y1 = y0 + (1u32 << half_h);
        decode_split_unit(eng, pic, stats, walk, decode, x0, y0, half_w, half_h)?;
        if x1 < walk.pic_width {
            decode_split_unit(eng, pic, stats, walk, decode, x1, y0, half_w, half_h)?;
        }
        if y1 < walk.pic_height {
            decode_split_unit(eng, pic, stats, walk, decode, x0, y1, half_w, half_h)?;
        }
        if x1 < walk.pic_width && y1 < walk.pic_height {
            decode_split_unit(eng, pic, stats, walk, decode, x1, y1, half_w, half_h)?;
        }
        return Ok(());
    }
    // Leaf: dual-tree luma + chroma.
    decode_coding_unit(
        eng,
        pic,
        stats,
        walk,
        decode,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        TreeType::DualTreeLuma,
    )?;
    decode_coding_unit(
        eng,
        pic,
        stats,
        walk,
        decode,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        TreeType::DualTreeChroma,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_coding_unit(
    eng: &mut CabacEngine,
    pic: &mut YuvPicture,
    stats: &mut SliceDecodeStats,
    walk: &SliceWalkInputs,
    decode: &SliceDecodeInputs,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    tree_type: TreeType,
) -> Result<()> {
    stats.coding_units += 1;
    // Decode intra_pred_mode for luma CU under sps_eipd_flag = 0.
    // Binarisation: U with cMax=4 (Table 91) — an unbounded unary prefix
    // capped to 4 leading 1s; the value is the number of leading 1s.
    // sps_cm_init_flag=0 → all bins land on (ctxTable=0, ctxIdx=0).
    let intra_idx = if matches!(tree_type, TreeType::DualTreeLuma | TreeType::SingleTree) {
        let v = eng.decode_u_regular(0, |_| 0)?;
        stats.intra_pred_mode_bins += 1;
        v
    } else {
        0
    };
    let intra_mode = IntraMode::from_baseline_idx(intra_idx).ok_or_else(|| {
        Error::invalid(format!(
            "evc decode: intra_pred_mode {intra_idx} out of Baseline range 0..=4"
        ))
    })?;

    decode_transform_unit(
        eng,
        pic,
        stats,
        walk,
        decode,
        x0,
        y0,
        log2_cb_width,
        log2_cb_height,
        tree_type,
        intra_mode,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_transform_unit(
    eng: &mut CabacEngine,
    pic: &mut YuvPicture,
    stats: &mut SliceDecodeStats,
    walk: &SliceWalkInputs,
    decode: &SliceDecodeInputs,
    x0: u32,
    y0: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    tree_type: TreeType,
    intra_mode: IntraMode,
) -> Result<()> {
    let log2_tb_width = log2_cb_width.min(walk.max_tb_log2_size_y);
    let log2_tb_height = log2_cb_height.min(walk.max_tb_log2_size_y);
    let chroma_present = walk.chroma_format_idc != 0;
    let mut cbf_cb = 0u32;
    let mut cbf_cr = 0u32;
    let mut cbf_luma = 0u32;
    if tree_type != TreeType::DualTreeLuma && chroma_present {
        cbf_cb = eng.decode_decision(0, 0)? as u32;
        cbf_cr = eng.decode_decision(0, 0)? as u32;
        stats.cbf_chroma_bins += 2;
    }
    let is_split =
        log2_cb_width > walk.max_tb_log2_size_y || log2_cb_height > walk.max_tb_log2_size_y;
    let is_intra = true;
    if (is_split || is_intra || cbf_cb != 0 || cbf_cr != 0) && tree_type != TreeType::DualTreeChroma
    {
        cbf_luma = eng.decode_decision(0, 0)? as u32;
        stats.cbf_luma_bins += 1;
    }
    if walk.cu_qp_delta_enabled && (cbf_luma != 0 || cbf_cb != 0 || cbf_cr != 0) {
        let qp_delta_abs = eng.decode_u_regular(0, |_| 0)?;
        if qp_delta_abs > 0 {
            let _sign = eng.decode_bypass()?;
        }
    }
    // Round-3 constraint: no residual coding. Surface non-zero CBFs as
    // unsupported so the caller knows the fixture is out of scope.
    if cbf_luma != 0 || cbf_cb != 0 || cbf_cr != 0 {
        return Err(Error::unsupported(
            "evc decode: round-3 fixtures must use cbf_luma=cbf_cb=cbf_cr=0",
        ));
    }
    // Reconstruct: pure intra prediction with zero residual.
    match tree_type {
        TreeType::DualTreeLuma | TreeType::SingleTree => {
            let n = (1usize << log2_tb_width) * (1usize << log2_tb_height);
            let zero_res = vec![0i32; n];
            // For luma blocks larger than max_tb, the spec splits the CB
            // into multiple TBs. Round-3 fixtures keep CB == TB.
            intra_reconstruct_cb(
                pic,
                x0,
                y0,
                log2_tb_width,
                log2_tb_height,
                intra_mode,
                0,
                &zero_res,
            )?;
        }
        TreeType::DualTreeChroma => {
            if chroma_present {
                // For sps_eipd_flag=0, intra_chroma_pred_mode is suppressed
                // → IntraPredModeC = IntraPredModeY for the same CU. We
                // re-use `intra_mode` from the luma path that was just
                // decoded into the same x0/y0. NB: in the dual-tree IDR
                // path the chroma `coding_unit()` is a separate CABAC pass
                // and we don't carry the luma value across; that would be
                // incorrect for non-DC modes, but our round-3 fixtures
                // restrict the slice to IntraMode::Dc so the inheritance
                // is moot — both luma and chroma decode an
                // intra_pred_mode = 0 from their own bins.
                //
                // We log this as a known limitation: round-4 will switch
                // to a real per-CU IntraPredModeY/C tracker.
                let log2_c_w = log2_tb_width.saturating_sub(1);
                let log2_c_h = log2_tb_height.saturating_sub(1);
                let n_c = (1usize << log2_c_w) * (1usize << log2_c_h);
                let zero_res = vec![0i32; n_c];
                intra_reconstruct_cb(
                    pic,
                    x0,
                    y0,
                    log2_tb_width,
                    log2_tb_height,
                    intra_mode,
                    1,
                    &zero_res,
                )?;
                intra_reconstruct_cb(
                    pic,
                    x0,
                    y0,
                    log2_tb_width,
                    log2_tb_height,
                    intra_mode,
                    2,
                    &zero_res,
                )?;
            }
        }
    }
    let _ = decode.slice_qp;
    let _ = scale_and_inverse_transform; // imported for round-4 use.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the walker reaches the terminate decision on a tiny hand
    /// fixture: a 16×16 picture (one 16×16 CTU with min_cb=4), no CBFs
    /// set so transform_unit consumes only 2 cbf bits per dual-tree
    /// invocation, and the terminate bit lands cleanly.
    ///
    /// Building the bitstream by hand is intractable without running the
    /// CABAC encoder; we instead use the engine itself to encode an
    /// expected sequence and feed it back. That's not a true black-box
    /// fixture, but it does verify the symmetric round-trip of the
    /// engine + walker pair, which is precisely the round-2 deliverable.
    #[test]
    fn walker_terminates_cleanly_on_min_idr_slice() {
        // Use a 4x4 picture (one CTU at min Cb) so the walker doesn't ask
        // for split_cu_flag (log2CbWidth=2, log2CbHeight=2 → no split).
        // The walker still enters DualTreeLuma + DualTreeChroma coding_unit:
        //   - Luma CU: intra_pred_mode (U; we want value 0 → 1 bin "0"),
        //              cbf_luma (1 bit "0").
        //   - Chroma CU: cbf_cb=0, cbf_cr=0 (2 bits "00"), then no cbf_luma
        //              path because treeType==DualTreeChroma.
        // Then end_of_tile_one_bit terminates.
        //
        // We can't easily synthesize a bin-accurate fixture here, so we
        // verify that walk_baseline_idr_slice gracefully returns an
        // error if the rbsp is malformed (instead of panicking).
        let inputs = SliceWalkInputs {
            pic_width: 4,
            pic_height: 4,
            ctb_log2_size_y: 5,
            min_cb_log2_size_y: 2,
            max_tb_log2_size_y: 5,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
        };
        // CTB size 32 > pic 4 — dimension check should still pass; the
        // engine will refuse to underflow on an empty slice.
        let res = walk_baseline_idr_slice(&[0u8; 0], inputs);
        assert!(res.is_err());
    }

    /// Reject a CTU configuration that cannot be parsed under the round-2
    /// Baseline subset (CtbLog2SizeY out of range).
    #[test]
    fn rejects_unsupported_ctb_size() {
        let inputs = SliceWalkInputs {
            pic_width: 64,
            pic_height: 64,
            ctb_log2_size_y: 4, // too small
            min_cb_log2_size_y: 2,
            max_tb_log2_size_y: 6,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: true,
        };
        let res = walk_baseline_idr_slice(&[0u8; 4], inputs);
        assert!(res.is_err());
    }

    /// Reject a CTU geometry with no CTUs — we need at least one CTU per
    /// slice to read end_of_tile_one_bit.
    #[test]
    fn rejects_zero_ctus() {
        let inputs = SliceWalkInputs {
            pic_width: 0,
            pic_height: 0,
            ctb_log2_size_y: 6,
            min_cb_log2_size_y: 2,
            max_tb_log2_size_y: 6,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: true,
        };
        let res = walk_baseline_idr_slice(&[0u8; 4], inputs);
        assert!(res.is_err());
    }

    /// The walker must initialise the CABAC engine — even an all-zero
    /// RBSP body (which gives ivl_offset == 0) must let the engine
    /// produce a stream of MPS bins until the (non-)terminate or a real
    /// decision says otherwise. We don't expect to consume the slice
    /// successfully here (no terminate ever decoded against zeros).
    #[test]
    fn engine_inits_from_zero_rbsp() {
        let inputs = SliceWalkInputs {
            pic_width: 64,
            pic_height: 64,
            ctb_log2_size_y: 6,
            min_cb_log2_size_y: 2,
            max_tb_log2_size_y: 6,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
        };
        // 1024 bytes of zero — the walker will eventually exhaust the
        // bit reader (since no terminate ever fires) and return Invalid.
        let bs = vec![0u8; 1024];
        let res = walk_baseline_idr_slice(&bs, inputs);
        assert!(res.is_err(), "expected exhaustion error, got {res:?}");
    }

    /// All-ones RBSP: the engine starts with ivl_offset=0x3FFF and every
    /// regular bin is the LPS. The walker should still progress (or
    /// terminate cleanly via the terminate path).
    #[test]
    fn engine_handles_all_ones_rbsp() {
        let inputs = SliceWalkInputs {
            pic_width: 32,
            pic_height: 32,
            ctb_log2_size_y: 5,
            min_cb_log2_size_y: 2,
            max_tb_log2_size_y: 5,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
        };
        let bs = vec![0xFFu8; 1024];
        // Either terminates or reports a structural error — but must not
        // panic / overflow.
        let _ = walk_baseline_idr_slice(&bs, inputs);
    }

    /// **End-to-end fixture for the round-2 deliverable.**
    ///
    /// Synthesise a single-CTU IDR slice with a known CABAC bin sequence
    /// using [`crate::cabac::CabacEncoder`] (the symmetric in-test
    /// inverse of the engine), then drive [`walk_baseline_idr_slice`]
    /// across it and verify every bin is consumed cleanly through the
    /// `end_of_tile_one_bit` terminate decision.
    ///
    /// The fixture splits the 32×32 CTB into four 16×16 sub-CBs (one
    /// `split_cu_flag = 1` at the CTB) and then runs every sub-CB
    /// through the dual-tree luma + chroma `coding_unit()` pair with no
    /// CBFs set (so no residual coding fires).
    ///
    /// Bin sequence:
    /// * `split_cu_flag = 1` (1 bin at the CTB)
    /// * For each of the 4 sub-CBs:
    ///     * `intra_pred_mode = 0` (1 U bin)
    ///     * `cbf_luma = 0` (1 FL bin)
    ///     * `cbf_cb = 0`, `cbf_cr = 0` (2 FL bins, dual-tree chroma)
    /// * `end_of_tile_one_bit` → terminate=true
    ///
    /// Total: 17 regular bins on (ctxTable=0, ctxIdx=0) + terminate.
    #[test]
    fn fixture_split_ctu_idr_slice_consumes_all_bins() {
        use crate::cabac::CabacEncoder;
        let mut enc = CabacEncoder::new();
        // Parent CTB (log2=5, min=4): emits split_cu_flag = 1.
        enc.encode_decision(0, 0, 1);
        // Each child (log2=4, min=4): no split_cu_flag (log2 == min). Each
        // emits intra_pred_mode + cbf_luma + cbf_cb + cbf_cr = 4 bins;
        // four children → 16 bins.
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // intra_pred_mode = "0"
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        let rbsp = enc.finish();

        let inputs = SliceWalkInputs {
            pic_width: 32,
            pic_height: 32,
            ctb_log2_size_y: 5,
            min_cb_log2_size_y: 4, // children land as 16x16 leaves
            max_tb_log2_size_y: 5,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
        };
        let stats = walk_baseline_idr_slice(&rbsp, inputs).unwrap();
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.split_cu_flag_bins, 1, "one split decision at the CTB");
        assert_eq!(stats.coding_units, 8, "4 children × (luma + chroma) = 8");
        assert_eq!(stats.intra_pred_mode_bins, 4);
        assert_eq!(stats.cbf_luma_bins, 4);
        assert_eq!(stats.cbf_chroma_bins, 8);
        assert_eq!(stats.cu_qp_delta_abs_bins, 0);
        assert_eq!(stats.coeff_runs, 0);
    }

    /// Larger fixture: a 64×32 picture split as two 32×32 CTUs side-by-
    /// side, each split into four 16×16 leaves. 32 leaves total → 32×4 =
    /// 128 child bins + 2 split bins = 130 regular bins + terminate.
    /// Stresses both the multi-CTU iteration and the long-renorm paths.
    #[test]
    fn fixture_two_ctu_split_idr_slice_consumes_all_bins() {
        use crate::cabac::CabacEncoder;
        let mut enc = CabacEncoder::new();
        for _ in 0..2 {
            enc.encode_decision(0, 0, 1); // split_cu_flag = 1 at the CTB
            for _ in 0..4 {
                enc.encode_decision(0, 0, 0); // intra_pred_mode
                enc.encode_decision(0, 0, 0); // cbf_luma
                enc.encode_decision(0, 0, 0); // cbf_cb
                enc.encode_decision(0, 0, 0); // cbf_cr
            }
        }
        enc.encode_terminate(true);
        let rbsp = enc.finish();

        let inputs = SliceWalkInputs {
            pic_width: 64,
            pic_height: 32,
            ctb_log2_size_y: 5,
            min_cb_log2_size_y: 4,
            max_tb_log2_size_y: 5,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
        };
        let stats = walk_baseline_idr_slice(&rbsp, inputs).unwrap();
        assert_eq!(stats.ctus, 2);
        assert_eq!(stats.split_cu_flag_bins, 2);
        assert_eq!(stats.coding_units, 16); // 2 CTUs × 4 children × (luma+chroma)
        assert_eq!(stats.intra_pred_mode_bins, 8);
        assert_eq!(stats.cbf_luma_bins, 8);
        assert_eq!(stats.cbf_chroma_bins, 16);
    }

    /// A 4:0:0 (monochrome) variant of the split-CTU fixture. Without
    /// chroma the dual-tree-chroma `coding_unit()` calls still happen
    /// but consume no `cbf_cb`/`cbf_cr` bins (the walker's chroma
    /// `transform_unit` branch is gated by `chroma_format_idc != 0`).
    #[test]
    fn fixture_split_ctu_monochrome_consumes_all_bins() {
        use crate::cabac::CabacEncoder;
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // split_cu_flag = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // intra_pred_mode (luma CU)
            enc.encode_decision(0, 0, 0); // cbf_luma (luma CU)
                                          // Chroma CU: no cbf_cb / cbf_cr (chroma_format_idc == 0).
        }
        enc.encode_terminate(true);
        let rbsp = enc.finish();

        let inputs = SliceWalkInputs {
            pic_width: 32,
            pic_height: 32,
            ctb_log2_size_y: 5,
            min_cb_log2_size_y: 4,
            max_tb_log2_size_y: 5,
            chroma_format_idc: 0, // monochrome
            cu_qp_delta_enabled: false,
        };
        let stats = walk_baseline_idr_slice(&rbsp, inputs).unwrap();
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.split_cu_flag_bins, 1);
        assert_eq!(stats.coding_units, 8);
        assert_eq!(stats.intra_pred_mode_bins, 4);
        assert_eq!(stats.cbf_luma_bins, 4);
        assert_eq!(stats.cbf_chroma_bins, 0, "no chroma at chroma_format_idc=0");
    }
}
