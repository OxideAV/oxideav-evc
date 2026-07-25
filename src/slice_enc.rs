//! Baseline-profile **intra slice encoder** (round 429): the write-side
//! dual of the §7.3.8 IDR `slice_data()` walker in [`crate::slice_data`].
//!
//! The emitted bin stream mirrors the decoder's read order bin for bin
//! (`sps_cm_init_flag == 0`: every regular bin shares the single
//! `(ctxTable 0, ctxIdx 0)` context, so encoder and decoder context
//! state evolve identically):
//!
//! * per CTU — `split_unit()` quad recursion (§7.3.8.3, `sps_btt_flag
//!   == 0` shape): `split_cu_flag` on recursable in-picture blocks,
//!   implicit splits at picture edges, leaves as the I-slice local dual
//!   tree (luma `coding_unit()` then chroma `coding_unit()`);
//! * luma CU — `intra_pred_mode` (U over the Table-13 5-mode set),
//!   `cbf_luma`, `residual_coding_rle()` (§7.3.8.7);
//! * chroma CU — `cbf_cb`, `cbf_cr`, per-component RLE residuals at the
//!   4:2:0 sub-sampled TB dimensions (prediction is the decoder's
//!   Baseline chroma shape: `INTRA_DC`);
//! * one `end_of_tile_one_bit` terminate after the last CTU (§7.3.8.1).
//!
//! ## Mode / split decisions
//!
//! Rate-distortion over the exact reconstruction the decoder will
//! produce: every candidate runs the *decoder's own* pipeline
//! ([`crate::quant_enc::forward_quantize`] →
//! [`crate::dequant::scale_and_inverse_transform`] → `clip(pred +
//! res)`), so the encoder-side `recon` picture is byte-identical to
//! what `decode_baseline_idr_slice` reconstructs from the emitted
//! stream. The coding tree is chosen bottom-up per `split_unit()`: the
//! leaf option (5-mode search) competes against the quad split of its
//! children under `SSE + λ·bits` with the classic `λ =
//! 0.57·2^((QP−12)/3)` and a bin-count rate estimate.
//!
//! Deblocking is signalled off (`slice_deblocking_filter_flag = 0`) so
//! the decoder's output equals the shared recon exactly.

use oxideav_core::{Error, Result};

use crate::cabac::CabacEncoder;
use crate::dequant::scale_and_inverse_transform;
use crate::intra::{predict, IntraMode, RefSamples};
use crate::picture::{intra_reconstruct_cb_in_tile, YuvPicture};
use crate::quant_enc::forward_quantize;
use crate::slice_data::zigzag_scan;

/// The five Table-13 Baseline intra modes in syntax-index order.
const MODES: [IntraMode; 5] = [
    IntraMode::Dc,
    IntraMode::Hor,
    IntraMode::Ver,
    IntraMode::Ul,
    IntraMode::Ur,
];

/// Geometry constants of the all-zero-toolset SPS the header writer
/// emits (§7.4.3.1 sps_btt_flag == 0 defaults + eq. 51).
const CTB_LOG2: u32 = 6;
const MIN_CB_LOG2: u32 = 2;

/// Per-picture encode statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncStats {
    /// CTUs walked.
    pub ctus: u32,
    /// Leaf coding units (each is a luma + chroma `coding_unit()` pair).
    pub leaves: u32,
    /// Leaves per chosen luma intra mode (Table 13 order DC/HOR/VER/UL/UR).
    pub mode_histogram: [u32; 5],
    /// Leaves whose `cbf_luma` was signalled 1.
    pub cbf_luma_set: u32,
    /// Chroma CBFs signalled 1 (cb + cr).
    pub cbf_chroma_set: u32,
    /// `split_cu_flag` bins emitted.
    pub split_flag_bins: u32,
}

/// One decided leaf: everything the emit pass needs to reproduce the
/// exact bin stream whose decode lands on the already-committed recon.
struct LeafPlan {
    mode_idx: usize,
    levels_y: Vec<i32>,
    cbf_y: bool,
    levels_cb: Vec<i32>,
    cbf_cb: bool,
    levels_cr: Vec<i32>,
    cbf_cr: bool,
}

/// A decided `split_unit()` subtree.
enum Node {
    Split(Vec<(u32, u32, u32, u32, Node)>),
    Leaf(LeafPlan),
}

struct EncCtx<'a> {
    src: &'a YuvPicture,
    recon: YuvPicture,
    qp: i32,
    lambda: f64,
    bit_depth: u32,
    pic_w: u32,
    pic_h: u32,
}

/// Encode one Baseline IDR picture's `slice_data()` payload. Returns
/// the CABAC payload bytes (byte-aligned, ready to append after the
/// byte-aligned slice header), the reconstruction the decoder must
/// reproduce exactly, and the encode statistics.
///
/// Requirements: 4:2:0 source, dimensions multiples of 4 (the §7.4.3.1
/// minimum CB), any bit depth the recon chain supports (8..=16).
pub fn encode_idr_slice_data(
    src: &YuvPicture,
    slice_qp: i32,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    if src.chroma_format_idc != 1 {
        return Err(Error::unsupported(
            "evc encoder: only 4:2:0 (chroma_format_idc == 1) is supported",
        ));
    }
    if src.width % 4 != 0 || src.height % 4 != 0 {
        return Err(Error::unsupported(format!(
            "evc encoder: dimensions {}x{} must be multiples of the 4-sample minimum CB",
            src.width, src.height
        )));
    }
    if !(0..=51).contains(&slice_qp) {
        return Err(Error::invalid(format!(
            "evc encoder: slice_qp {slice_qp} out of range [0, 51]"
        )));
    }
    let recon = YuvPicture::new(src.width, src.height, 1, src.bit_depth)?;
    let mut ctx = EncCtx {
        src,
        recon,
        qp: slice_qp,
        lambda: 0.57 * 2f64.powf((slice_qp as f64 - 12.0) / 3.0),
        bit_depth: src.bit_depth,
        pic_w: src.width,
        pic_h: src.height,
    };
    let mut stats = EncStats::default();

    // Decide pass: raster CTU order, committing the chosen recon as we
    // go (later CUs predict from it exactly like the decoder).
    let ctus_x = src.width.div_ceil(1 << CTB_LOG2);
    let ctus_y = src.height.div_ceil(1 << CTB_LOG2);
    let mut roots = Vec::with_capacity((ctus_x * ctus_y) as usize);
    for cy in 0..ctus_y {
        for cx in 0..ctus_x {
            let (node, _cost) = decide_split_unit(
                &mut ctx,
                &mut stats,
                cx << CTB_LOG2,
                cy << CTB_LOG2,
                CTB_LOG2,
                CTB_LOG2,
            )?;
            roots.push((cx << CTB_LOG2, cy << CTB_LOG2, node));
            stats.ctus += 1;
        }
    }

    // Emit pass: replay the decided tree into the arithmetic coder in
    // the decoder's exact read order.
    let mut enc = CabacEncoder::new();
    for (x0, y0, node) in &roots {
        emit_split_unit(
            &mut enc, &mut stats, &ctx, *x0, *y0, CTB_LOG2, CTB_LOG2, node,
        );
    }
    enc.encode_terminate(true); // §7.3.8.1 end_of_tile_one_bit
    Ok((enc.finish(), ctx.recon, stats))
}

/// Mirror of the decoder's Baseline `resolve_split_unit` presence rules
/// (§7.3.8.3, `sps_btt_flag == 0`): whether this block reads a
/// `split_cu_flag`, may recurse, and lies fully inside the picture.
fn split_geometry(ctx: &EncCtx<'_>, x0: u32, y0: u32, log2_w: u32, log2_h: u32) -> (bool, bool) {
    let within = x0 + (1 << log2_w) <= ctx.pic_w && y0 + (1 << log2_h) <= ctx.pic_h;
    let can_recurse = log2_w > MIN_CB_LOG2 && log2_h > MIN_CB_LOG2;
    (within, can_recurse)
}

/// Decide one `split_unit()`: leaf vs quad split under RD cost,
/// committing the winning reconstruction into `ctx.recon`. Returns the
/// decided subtree and its cost (distortion + λ·bits).
fn decide_split_unit(
    ctx: &mut EncCtx<'_>,
    stats: &mut EncStats,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(Node, f64)> {
    let (within, can_recurse) = split_geometry(ctx, x0, y0, log2_w, log2_h);
    let flag_present = can_recurse && within && (log2_w > 2 || log2_h > 2);

    if can_recurse && !within {
        // Implicit split at the picture edge — no flag, no leaf option.
        let (children, cost) = decide_children(ctx, stats, x0, y0, log2_w, log2_h)?;
        return Ok((Node::Split(children), cost));
    }
    if !flag_present {
        // 4×4 in-picture leaf: nothing signalled, coding_unit() pair.
        let (plan, cost) = decide_leaf(ctx, stats, x0, y0, log2_w, log2_h)?;
        return Ok((Node::Leaf(plan), cost + ctx.lambda));
    }

    // Both options are open. Trial the leaf first, snapshot its recon,
    // rewind, trial the split, then keep the cheaper reconstruction.
    let before = save_region(&ctx.recon, x0, y0, log2_w, log2_h);
    let (leaf_plan, leaf_cost) = decide_leaf(ctx, stats, x0, y0, log2_w, log2_h)?;
    let leaf_cost = leaf_cost + ctx.lambda; // split_cu_flag = 0 bin
    let after_leaf = save_region(&ctx.recon, x0, y0, log2_w, log2_h);
    restore_region(&mut ctx.recon, &before, x0, y0, log2_w, log2_h);

    let (children, split_cost) = decide_children(ctx, stats, x0, y0, log2_w, log2_h)?;
    let split_cost = split_cost + ctx.lambda; // split_cu_flag = 1 bin

    if leaf_cost <= split_cost {
        restore_region(&mut ctx.recon, &after_leaf, x0, y0, log2_w, log2_h);
        Ok((Node::Leaf(leaf_plan), leaf_cost))
    } else {
        Ok((Node::Split(children), split_cost))
    }
}

/// Decide the four (in-picture) quad children in the decoder's child
/// enumeration order (`split::quad_split_children`, SUCO order 0).
#[allow(clippy::type_complexity)]
fn decide_children(
    ctx: &mut EncCtx<'_>,
    stats: &mut EncStats,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(Vec<(u32, u32, u32, u32, Node)>, f64)> {
    let mut out = Vec::with_capacity(4);
    let mut cost = 0f64;
    for ch in crate::split::quad_split_children(x0, y0, log2_w, log2_h, 0, 0, ctx.pic_w, ctx.pic_h)
    {
        let (node, c) = decide_split_unit(
            ctx,
            stats,
            ch.x0,
            ch.y0,
            ch.log2_cb_width,
            ch.log2_cb_height,
        )?;
        cost += c;
        out.push((ch.x0, ch.y0, ch.log2_cb_width, ch.log2_cb_height, node));
    }
    Ok((out, cost))
}

/// Decide one leaf: 5-mode luma search + DC chroma, committing the
/// winning reconstruction (via the decoder's own
/// `intra_reconstruct_cb_in_tile`) and returning the plan + RD cost.
fn decide_leaf(
    ctx: &mut EncCtx<'_>,
    stats: &mut EncStats,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(LeafPlan, f64)> {
    let w = 1usize << log2_w;
    let h = 1usize << log2_h;
    let bd = ctx.bit_depth;
    let max_val = (1i32 << bd) - 1;

    // ---- luma: 5-mode search over the decoder's reference fetch ----
    let refs = ctx.recon.fetch_intra_refs(x0, y0, w, h, 0);
    let src_y = gather_block(&ctx.src.y, ctx.src.y_stride(), x0, y0, w, h);
    let mut best: Option<(usize, Vec<i32>, bool, Vec<i32>)> = None;
    let mut best_cost = f64::INFINITY;
    for (mode_idx, &mode) in MODES.iter().enumerate() {
        let (levels, cbf, res, dist) =
            quantize_block(&refs, mode, &src_y, w, h, ctx.qp, bd, max_val)?;
        let bits = (mode_idx + 1) as f64 + 1.0 + rle_bits_estimate(&levels, cbf);
        let cost = dist + ctx.lambda * bits;
        if cost < best_cost {
            best_cost = cost;
            best = Some((mode_idx, levels, cbf, res));
        }
    }
    let (mode_idx, levels_y, cbf_y, res_y) = best.expect("5 candidates");
    let cost_y = best_cost;
    intra_reconstruct_cb_in_tile(
        &mut ctx.recon,
        x0,
        y0,
        log2_w,
        log2_h,
        MODES[mode_idx],
        0,
        &res_y,
        None,
    )?;

    // ---- chroma: the decoder's Baseline chroma prediction is INTRA_DC
    // on the dual-tree chroma CU; quantize each component's residual ----
    let wc = 1usize << (log2_w - 1);
    let hc = 1usize << (log2_h - 1);
    let mut cost = cost_y;
    let mut chroma = Vec::with_capacity(2);
    for c_idx in 1..=2u32 {
        let plane = if c_idx == 1 { &ctx.src.cb } else { &ctx.src.cr };
        let refs_c = ctx.recon.fetch_intra_refs(x0 >> 1, y0 >> 1, wc, hc, c_idx);
        let src_c = gather_block(plane, ctx.src.c_stride(), x0 >> 1, y0 >> 1, wc, hc);
        let (levels, cbf, res, dist) =
            quantize_block(&refs_c, IntraMode::Dc, &src_c, wc, hc, ctx.qp, bd, max_val)?;
        let bits = 1.0 + rle_bits_estimate(&levels, cbf);
        cost += dist + ctx.lambda * bits;
        intra_reconstruct_cb_in_tile(
            &mut ctx.recon,
            x0,
            y0,
            log2_w,
            log2_h,
            IntraMode::Dc,
            c_idx,
            &res,
            None,
        )?;
        chroma.push((levels, cbf));
    }
    let (levels_cr, cbf_cr) = chroma.pop().expect("cr");
    let (levels_cb, cbf_cb) = chroma.pop().expect("cb");

    stats.leaves += 1;
    stats.mode_histogram[mode_idx] += 1;
    stats.cbf_luma_set += u32::from(cbf_y);
    stats.cbf_chroma_set += u32::from(cbf_cb) + u32::from(cbf_cr);

    Ok((
        LeafPlan {
            mode_idx,
            levels_y,
            cbf_y,
            levels_cb,
            cbf_cb,
            levels_cr,
            cbf_cr,
        },
        cost,
    ))
}

/// Predict + transform + quantize + reconstruct one candidate block
/// through the decoder's pipeline; returns (levels, cbf, recon
/// residual, SSE distortion).
#[allow(clippy::too_many_arguments)]
fn quantize_block(
    refs: &RefSamples,
    mode: IntraMode,
    src: &[i32],
    w: usize,
    h: usize,
    qp: i32,
    bit_depth: u32,
    max_val: i32,
) -> Result<(Vec<i32>, bool, Vec<i32>, f64)> {
    let n = w * h;
    let mut pred = vec![0i32; n];
    predict(mode, refs, w, h, bit_depth, &mut pred);
    let diff: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let mut levels = vec![0i32; n];
    let cbf = forward_quantize(&diff, &mut levels, w, h, qp, bit_depth)?;
    let mut res = vec![0i32; n];
    if cbf {
        scale_and_inverse_transform(&levels, &mut res, w, h, qp, bit_depth)?;
    }
    let mut dist = 0f64;
    for i in 0..n {
        let rec = (pred[i] + res[i]).clamp(0, max_val);
        let d = (rec - src[i]) as f64;
        dist += d * d;
    }
    Ok((levels, cbf, res, dist))
}

/// Crude §7.3.8.7 rate estimate in bins: per non-zero coefficient the
/// zero-run unary, the level unary, the sign and the last flag.
fn rle_bits_estimate(levels: &[i32], cbf: bool) -> f64 {
    if !cbf {
        return 0.0;
    }
    let mut bits = 0f64;
    let mut run = 0f64;
    for &l in levels {
        if l == 0 {
            run += 1.0;
        } else {
            bits += run + 1.0 + (l.unsigned_abs() as f64) + 2.0;
            run = 0.0;
        }
    }
    bits
}

fn gather_block(plane: &[u16], stride: usize, x0: u32, y0: u32, w: usize, h: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(w * h);
    for yy in 0..h {
        let row = (y0 as usize + yy) * stride + x0 as usize;
        out.extend(plane[row..row + w].iter().map(|&v| v as i32));
    }
    out
}

/// Saved recon rectangle (luma + both chroma sub-rects), for the
/// leaf-vs-split trial rewinds.
struct RegionSave {
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
}

fn save_region(pic: &YuvPicture, x0: u32, y0: u32, log2_w: u32, log2_h: u32) -> RegionSave {
    let grab = |plane: &[u16], stride: usize, x: usize, y: usize, w: usize, h: usize| {
        let mut out = Vec::with_capacity(w * h);
        for yy in 0..h {
            let row = (y + yy) * stride + x;
            out.extend_from_slice(&plane[row..(row + w).min(plane.len().max(row))]);
        }
        out
    };
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    let (x, y) = (x0 as usize, y0 as usize);
    let yw = (pic.width as usize).saturating_sub(x).min(w);
    let yh = (pic.height as usize).saturating_sub(y).min(h);
    let cs = pic.c_stride();
    let cwp = pic.width.div_ceil(2) as usize;
    let chp = pic.height.div_ceil(2) as usize;
    let cw = cwp.saturating_sub(x / 2).min(w / 2);
    let ch = chp.saturating_sub(y / 2).min(h / 2);
    RegionSave {
        y: grab(&pic.y, pic.y_stride(), x, y, yw, yh),
        cb: grab(&pic.cb, cs, x / 2, y / 2, cw, ch),
        cr: grab(&pic.cr, cs, x / 2, y / 2, cw, ch),
    }
}

fn restore_region(
    pic: &mut YuvPicture,
    save: &RegionSave,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) {
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    let (x, y) = (x0 as usize, y0 as usize);
    let yw = (pic.width as usize).saturating_sub(x).min(w);
    let yh = (pic.height as usize).saturating_sub(y).min(h);
    let stride = pic.y_stride();
    for yy in 0..yh {
        let row = (y + yy) * stride + x;
        pic.y[row..row + yw].copy_from_slice(&save.y[yy * yw..(yy + 1) * yw]);
    }
    let cs = pic.c_stride();
    let cwp = pic.width.div_ceil(2) as usize;
    let chp = pic.height.div_ceil(2) as usize;
    let cw = cwp.saturating_sub(x / 2).min(w / 2);
    let ch = chp.saturating_sub(y / 2).min(h / 2);
    for yy in 0..ch {
        let row = (y / 2 + yy) * cs + x / 2;
        pic.cb[row..row + cw].copy_from_slice(&save.cb[yy * cw..(yy + 1) * cw]);
        pic.cr[row..row + cw].copy_from_slice(&save.cr[yy * cw..(yy + 1) * cw]);
    }
}

// ---------------------------------------------------------------------
// Emit pass — decoder-exact bin order.
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_split_unit(
    enc: &mut CabacEncoder,
    stats: &mut EncStats,
    ctx: &EncCtx<'_>,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    node: &Node,
) {
    let (within, can_recurse) = split_geometry(ctx, x0, y0, log2_w, log2_h);
    let flag_present = can_recurse && within && (log2_w > 2 || log2_h > 2);
    match node {
        Node::Split(children) => {
            if flag_present {
                enc.encode_decision(0, 0, 1);
                stats.split_flag_bins += 1;
            }
            // (implicit split when !within: no bin, matching the decoder)
            for (cx, cy, clw, clh, child) in children {
                emit_split_unit(enc, stats, ctx, *cx, *cy, *clw, *clh, child);
            }
        }
        Node::Leaf(plan) => {
            if flag_present {
                enc.encode_decision(0, 0, 0);
                stats.split_flag_bins += 1;
            }
            emit_leaf(enc, plan, log2_w, log2_h);
        }
    }
}

fn emit_leaf(enc: &mut CabacEncoder, plan: &LeafPlan, log2_w: u32, log2_h: u32) {
    // Luma coding_unit(): intra_pred_mode (U, all bins on (0,0)) —
    // the decoder reads it via `decode_u_regular` (63-bin compat cap).
    enc.encode_u_regular_capped(plan.mode_idx as u32, 63, 0, |_| 0);
    // transform_unit(), DUAL_TREE_LUMA: cbf_luma only.
    enc.encode_decision(0, 0, u8::from(plan.cbf_y));
    if plan.cbf_y {
        emit_residual_rle(enc, &plan.levels_y, log2_w, log2_h);
    }
    // Chroma coding_unit(): no intra_pred_mode read (DualTreeChroma),
    // transform_unit() reads cbf_cb then cbf_cr then the residuals at
    // the 4:2:0 dimensions.
    enc.encode_decision(0, 0, u8::from(plan.cbf_cb));
    enc.encode_decision(0, 0, u8::from(plan.cbf_cr));
    if plan.cbf_cb {
        emit_residual_rle(enc, &plan.levels_cb, log2_w - 1, log2_h - 1);
    }
    if plan.cbf_cr {
        emit_residual_rle(enc, &plan.levels_cr, log2_w - 1, log2_h - 1);
    }
}

/// §7.3.8.7 `residual_coding_rle()` writer — the exact dual of
/// `decode_residual_coding_rle`: per non-zero coefficient in §6.5.2
/// zig-zag order, `coeff_zero_run` (U, cMax = blockSize − 1),
/// `coeff_abs_level_minus1` (U, cMax 32767), `coeff_sign_flag`
/// (bypass), and `coeff_last_flag` unless the coefficient sits at the
/// final scan position (where the decoder infers 1).
fn emit_residual_rle(enc: &mut CabacEncoder, levels: &[i32], log2_w: u32, log2_h: u32) {
    let blk_w = 1usize << log2_w;
    let blk_h = 1usize << log2_h;
    let total = blk_w * blk_h;
    debug_assert_eq!(levels.len(), total);
    let scan = zigzag_scan(blk_w, blk_h);
    let nz: Vec<(usize, i32)> = scan
        .iter()
        .enumerate()
        .filter_map(|(scan_pos, &blk_pos)| {
            let l = levels[blk_pos];
            (l != 0).then_some((scan_pos, l))
        })
        .collect();
    debug_assert!(!nz.is_empty(), "cbf set with all-zero levels");
    let zero_run_c_max = (total as u32) - 1;
    let mut cursor = 0usize;
    let last = nz.len() - 1;
    for (i, &(scan_pos, level)) in nz.iter().enumerate() {
        let zero_run = (scan_pos - cursor) as u32;
        enc.encode_u_regular_capped(zero_run, zero_run_c_max, 0, |_| 0);
        enc.encode_u_regular_capped(level.unsigned_abs() - 1, 32767, 0, |_| 0);
        enc.encode_bypass(u8::from(level < 0));
        if scan_pos < total - 1 {
            enc.encode_decision(0, 0, u8::from(i == last));
        }
        cursor = scan_pos + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice_data::{
        decode_baseline_idr_slice, CodingTreeGates, SliceDecodeInputs, SliceWalkInputs,
    };

    fn walk_inputs(w: u32, h: u32) -> SliceWalkInputs {
        SliceWalkInputs {
            pic_width: w,
            pic_height: h,
            ctb_log2_size_y: CTB_LOG2,
            min_cb_log2_size_y: MIN_CB_LOG2,
            max_tb_log2_size_y: 6,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
            sps_adcc_flag: false,
            sps_eipd_flag: false,
            sps_dquant_flag: false,
            cu_qp_delta_area: 6,
            sps_ibc_flag: false,
            log2_max_ibc_cand_size: 0,
            slice_alf_enabled_flag: false,
            slice_alf_map_flag: false,
            slice_chroma_alf_enabled_flag: false,
            slice_alf_chroma_map_flag: false,
            slice_chroma2_alf_enabled_flag: false,
            slice_alf_chroma2_map_flag: false,
            tree_gates: CodingTreeGates::default(),
        }
    }

    fn decode_inputs(qp: i32) -> SliceDecodeInputs {
        SliceDecodeInputs {
            slice_qp: qp,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            enable_deblock: false,
            sps_addb_flag: false,
            filter_offset_a: 0,
            filter_offset_b: 0,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            sps_ibc_flag: false,
            log2_max_ibc_cand_size: 0,
            sps_htdf_flag: false,
        }
    }

    /// Deterministic pseudo-natural test frame: smooth gradients plus a
    /// few hard edges and a textured band.
    fn synth_picture(w: u32, h: u32, seed: u32) -> YuvPicture {
        let mut pic = YuvPicture::new(w, h, 1, 8).unwrap();
        let mut s = seed;
        let mut noise = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 24) & 0x0F) as i32 - 8
        };
        for y in 0..h as usize {
            for x in 0..w as usize {
                let mut v = 40 + ((x as i32 * 3 + y as i32 * 2) % 160);
                if x > w as usize / 2 && y < h as usize / 3 {
                    v = 210; // hard bright block
                }
                if y > 2 * h as usize / 3 {
                    v += noise(); // textured band
                }
                pic.y[y * w as usize + x] = v.clamp(0, 255) as u16;
            }
        }
        let cw = w.div_ceil(2) as usize;
        let chh = h.div_ceil(2) as usize;
        for y in 0..chh {
            for x in 0..cw {
                pic.cb[y * cw + x] = (100 + ((x + 2 * y) % 60)) as u16;
                pic.cr[y * cw + x] = (140 + ((2 * x + y) % 50)) as u16;
            }
        }
        pic
    }

    fn psnr(a: &[u16], b: &[u16], max_val: f64) -> f64 {
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (max_val * max_val / mse).log10()
        }
    }

    /// THE core pin: encode → decode with the crate's own §7.3.8 walker
    /// must (a) consume every bin cleanly and (b) reconstruct
    /// byte-exactly the encoder's recon, across a size × QP matrix that
    /// covers CTU-aligned, sub-CTU and implicit-boundary-split shapes.
    #[test]
    fn round_trip_recon_exact_size_qp_matrix() {
        for &(w, h) in &[(64u32, 64u32), (32, 32), (128, 96), (100, 60), (176, 144)] {
            for &qp in &[4i32, 22, 37, 51] {
                let src = synth_picture(w, h, 0xC0FFEE ^ (w * 31 + h * 7 + qp as u32));
                let (payload, enc_recon, stats) = encode_idr_slice_data(&src, qp).expect("encode");
                assert!(stats.ctus >= 1);
                let (dec, dec_stats) =
                    decode_baseline_idr_slice(&payload, walk_inputs(w, h), decode_inputs(qp))
                        .unwrap_or_else(|e| panic!("{w}x{h} qp{qp}: decode failed: {e}"));
                assert_eq!(dec.y, enc_recon.y, "{w}x{h} qp{qp}: luma recon mismatch");
                assert_eq!(dec.cb, enc_recon.cb, "{w}x{h} qp{qp}: cb recon mismatch");
                assert_eq!(dec.cr, enc_recon.cr, "{w}x{h} qp{qp}: cr recon mismatch");
                assert_eq!(dec_stats.ctus, stats.ctus);
            }
        }
    }

    /// Low-QP quality pin: the decoded picture is near-lossless against
    /// the *source* (PSNR ≥ 46 dB at QP 4 on the synthetic frame), and
    /// quality degrades monotonically toward high QP while rate shrinks.
    #[test]
    fn quality_and_rate_track_qp() {
        let (w, h) = (128u32, 96u32);
        let src = synth_picture(w, h, 0xBADC0DE);
        let mut prev_bytes = usize::MAX;
        let mut prev_psnr = f64::INFINITY;
        for &qp in &[4i32, 22, 37, 51] {
            let (payload, recon, _stats) = encode_idr_slice_data(&src, qp).expect("encode");
            let p = psnr(&src.y, &recon.y, 255.0);
            assert!(
                payload.len() <= prev_bytes,
                "rate must not grow with QP (qp {qp}: {} > {prev_bytes})",
                payload.len()
            );
            assert!(
                p <= prev_psnr + 0.01,
                "PSNR must not improve with QP (qp {qp}: {p:.2} > {prev_psnr:.2})"
            );
            if qp == 4 {
                assert!(p >= 46.0, "QP 4 luma PSNR {p:.2} < 46 dB");
            }
            prev_bytes = payload.len();
            prev_psnr = p;
        }
    }

    /// A flat grey frame must cost almost nothing (every CU predicts
    /// perfectly from the mid-level substitution / DC chain) and decode
    /// losslessly.
    #[test]
    fn flat_frame_is_lossless_and_tiny() {
        let mut src = YuvPicture::new(64, 64, 1, 8).unwrap();
        for v in src.y.iter_mut() {
            *v = 128;
        }
        for v in src.cb.iter_mut().chain(src.cr.iter_mut()) {
            *v = 128;
        }
        let (payload, recon, stats) = encode_idr_slice_data(&src, 30).unwrap();
        assert_eq!(recon.y, src.y);
        assert_eq!(recon.cb, src.cb);
        assert_eq!(recon.cr, src.cr);
        assert!(
            payload.len() < 32,
            "flat 64x64 frame should be a handful of bytes, got {}",
            payload.len()
        );
        assert_eq!(stats.cbf_luma_set, 0);
        assert_eq!(stats.cbf_chroma_set, 0);
        let (dec, _) =
            decode_baseline_idr_slice(&payload, walk_inputs(64, 64), decode_inputs(30)).unwrap();
        assert_eq!(dec.y, src.y);
    }

    /// Directional content picks directional modes: a pure vertical
    /// stripe pattern must select INTRA_VER somewhere (and reconstruct
    /// exactly at low QP once the first row is coded).
    #[test]
    fn directional_content_uses_directional_modes() {
        let mut src = YuvPicture::new(64, 64, 1, 8).unwrap();
        for y in 0..64usize {
            for x in 0..64usize {
                src.y[y * 64 + x] = if (x / 4) % 2 == 0 { 60 } else { 200 };
            }
        }
        for v in src.cb.iter_mut().chain(src.cr.iter_mut()) {
            *v = 128;
        }
        let (_payload, recon, stats) = encode_idr_slice_data(&src, 8).unwrap();
        assert!(
            stats.mode_histogram[2] > 0,
            "vertical stripes must engage INTRA_VER: {:?}",
            stats.mode_histogram
        );
        let p = psnr(&src.y, &recon.y, 255.0);
        assert!(p >= 42.0, "striped frame QP8 PSNR {p:.2}");
    }

    /// The RLE writer is the exact dual of the RLE reader for hand-set
    /// level patterns (first/last position, negatives, isolated DC).
    #[test]
    fn residual_rle_writer_reader_duality() {
        use crate::cabac::CabacEngine;
        let patterns: Vec<Vec<i32>> = vec![
            {
                let mut v = vec![0i32; 16];
                v[0] = 5;
                v
            },
            {
                let mut v = vec![0i32; 16];
                v[15] = -3;
                v
            },
            {
                let mut v = vec![0i32; 16];
                v[0] = -1;
                v[5] = 2;
                v[15] = 7;
                v
            },
            (1..=16).map(|i| if i % 3 == 0 { -i } else { i }).collect(),
        ];
        for levels in patterns {
            let mut enc = CabacEncoder::new();
            emit_residual_rle(&mut enc, &levels, 2, 2);
            enc.encode_terminate(true);
            let bytes = enc.finish();
            let mut eng = CabacEngine::new(&bytes).unwrap();
            let sel = crate::cabac_init::CtxSel::new(false, crate::cabac::InitType::I);
            let mut decoded = vec![0i32; 16];
            let mut runs = 0u32;
            crate::slice_data::decode_residual_coding_rle(
                &mut eng,
                sel,
                0,
                &mut decoded,
                &mut runs,
                2,
                2,
            )
            .unwrap();
            assert!(eng.decode_terminate().unwrap());
            assert_eq!(decoded, levels);
        }
    }

    /// Encoder input validation: non-multiple-of-4 dims and bad QPs are
    /// refused up front.
    #[test]
    fn rejects_bad_dims_and_qp() {
        let src = YuvPicture::new(66, 64, 1, 8).unwrap();
        assert!(encode_idr_slice_data(&src, 30).is_err());
        let src = YuvPicture::new(64, 64, 1, 8).unwrap();
        assert!(encode_idr_slice_data(&src, 52).is_err());
        assert!(encode_idr_slice_data(&src, -1).is_err());
    }
}
