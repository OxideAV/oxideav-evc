//! **Low-delay P slice encoder** (round 431): the write-side dual of
//! the §7.3.8 P-slice `slice_data()` walker in
//! [`crate::slice_data::decode_baseline_inter_slice`] under the
//! Baseline toolset (`sps_admvp_flag == 0`, single reference,
//! `num_ref_idx_active_minus1[0] == 0`).
//!
//! ## Mode ladder (per leaf, RD over the decoder's own reconstruction)
//!
//! * **skip** — `cu_skip_flag = 1` + `mvp_idx_l0` (TR cMax 3): the MV
//!   comes straight from the §8.5.2.4 AMVP candidate list (the same
//!   [`crate::slice_data::baseline_amvp_select_with_grid_and_hmvp`]
//!   the decoder runs, over the encoder's decode-order side-info grid
//!   and §8.5.2.7 HMVP list), no MVD, **no residual syntax** (§7.3.8.4:
//!   the `cbf_all` block lives in the non-skip branch);
//! * **explicit inter** — `cu_skip_flag = 0`, `pred_mode_flag = 0`,
//!   `mvp_idx_l0` + the `abs_mvd`/sign pair: quarter-pel motion search
//!   against the decoder's §8.5.4 Baseline interpolation kernels
//!   (full-pel hill climb from the AMVP candidates, then half- and
//!   quarter-pel refinement through
//!   [`crate::inter::interpolate_luma_block`]), residual through the
//!   §8.7-inverted [`crate::quant_enc::forward_quantize`]; the CU
//!   signals `cbf_all` (line 3028) — 0 elides the whole
//!   `transform_unit()`, and with quiet chroma `cbf_luma` is inferred
//!   1 (§7.4.9.5), exactly the decoder's read;
//! * **intra** — `pred_mode_flag = 1` + the Baseline 5-mode search
//!   (single tree: chroma predicts with the luma mode, mirroring the
//!   decoder's `decode_inter_intra_cu`).
//!
//! The coding tree is the same bottom-up quad `split_unit()` RD of the
//! IDR encoder. The decide pass runs in decode order, committing the
//! chosen reconstruction *and* the decoder-visible state (side-info
//! stamps, `cu_skip` cell marks, HMVP updates — reset at each CTU row
//! per §7.3.8.2) so every AMVP list the encoder consults is exactly
//! the list the decoder will build. The emit pass replays the decided
//! tree bin for bin (initType 1 contexts; under `sps_cm_init_flag == 1`
//! the §9.3.4.2.4 neighbour ctxIncs are re-derived over an emit-side
//! grid stamped in the same order).
//!
//! With `deblock` the §8.8.2 post-pass runs over the recon with the
//! stamped side info — on a P picture the inter/cbf edges carry live
//! boundary strengths (unlike the r429 all-intra no-op), so the
//! returned picture is the decoder's filtered output byte for byte.

use oxideav_core::{Error, Result};

use crate::cabac::{CabacEncoder, InitType};
use crate::cabac_init::{CtxSel, MainCtxTable};
use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
use crate::dequant::scale_and_inverse_transform;
use crate::hmvp::{HmvpCandList, HmvpCandidate};
use crate::inter::{
    derive_chroma_mv, interpolate_chroma_block, interpolate_luma_block, MotionVector,
    RefPictureView,
};
use crate::picture::{intra_reconstruct_cb_in_tile, YuvPicture};
use crate::quant_enc::forward_quantize;
use crate::slice_data::{
    baseline_amvp_select_with_grid_and_hmvp, ctx_inc_neighbour_cells, mark_cu_skip_cells,
    SliceWalkInputs,
};
use crate::slice_enc::{
    emit_residual_rle, gather_block, quantize_block, restore_region, rle_bits_estimate,
    save_region, MODES,
};

/// Geometry constants of the encoder SPS (§7.4.3.1 `sps_btt_flag == 0`
/// defaults): 64×64 CTU, 4×4 minimum CB. With `MaxTbLog2SizeY == 6`
/// (eq. 51) no leaf ever TB-splits, so every CU is a single
/// `transform_unit()`.
const CTB_LOG2: u32 = 6;
const MIN_CB_LOG2: u32 = 2;

/// Per-picture P-encode statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PEncStats {
    pub ctus: u32,
    pub leaves: u32,
    /// Leaves coded `cu_skip_flag = 1`.
    pub skip_cus: u32,
    /// Non-skip MODE_INTER leaves (explicit MV).
    pub inter_cus: u32,
    /// Explicit inter leaves whose `cbf_all` was signalled 0.
    pub cbf_all_zero_cus: u32,
    /// MODE_INTRA leaves.
    pub intra_cus: u32,
    /// `split_cu_flag` bins emitted.
    pub split_flag_bins: u32,
}

/// One decided leaf.
enum PLeaf {
    Skip {
        mvp_idx: u32,
        mv: MotionVector,
    },
    Inter {
        mvp_idx: u32,
        mv: MotionVector,
        mvd: MotionVector,
        levels_y: Vec<i32>,
        cbf_y: bool,
        levels_cb: Vec<i32>,
        cbf_cb: bool,
        levels_cr: Vec<i32>,
        cbf_cr: bool,
    },
    Intra {
        mode_idx: usize,
        levels_y: Vec<i32>,
        cbf_y: bool,
        levels_cb: Vec<i32>,
        cbf_cb: bool,
        levels_cr: Vec<i32>,
        cbf_cr: bool,
    },
}

/// A decided `split_unit()` subtree.
enum Node {
    Split(Vec<(u32, u32, u32, u32, Node)>),
    Leaf(PLeaf),
}

struct PCtx<'a> {
    src: &'a YuvPicture,
    recon: YuvPicture,
    refp: &'a YuvPicture,
    qp: i32,
    lambda: f64,
    bit_depth: u32,
    pic_w: u32,
    pic_h: u32,
    /// Decode-order state the AMVP/ctxInc probes read — evolves exactly
    /// like the decoder's.
    side_info: SideInfoGrid,
    hmvp: HmvpCandList,
    /// Minimal walk inputs for the shared §9.3.4.2.4 probe helper.
    walk: SliceWalkInputs,
}

impl PCtx<'_> {
    fn ref_view(&self) -> RefPictureView<'_> {
        RefPictureView {
            y: &self.refp.y,
            cb: &self.refp.cb,
            cr: &self.refp.cr,
            width: self.refp.width,
            height: self.refp.height,
            y_stride: self.refp.y_stride(),
            c_stride: self.refp.c_stride(),
            chroma_format_idc: self.refp.chroma_format_idc,
        }
    }
}

/// Encode one Baseline P picture's `slice_data()` payload against
/// `ref_pic` (the previous picture exactly as the decoder holds it in
/// the DPB — i.e. post-deblock when deblocking is on). Returns the
/// CABAC payload, the reconstruction the decoder reproduces byte for
/// byte (post-§8.8.2 when `deblock`), and the statistics.
pub fn encode_p_slice_data(
    src: &YuvPicture,
    ref_pic: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
) -> Result<(Vec<u8>, YuvPicture, PEncStats)> {
    if src.chroma_format_idc != 1 {
        return Err(Error::unsupported(
            "evc p encoder: only 4:2:0 (chroma_format_idc == 1) is supported",
        ));
    }
    if src.width % 4 != 0 || src.height % 4 != 0 {
        return Err(Error::unsupported(format!(
            "evc p encoder: dimensions {}x{} must be multiples of 4",
            src.width, src.height
        )));
    }
    if !(0..=51).contains(&slice_qp) {
        return Err(Error::invalid(format!(
            "evc p encoder: slice_qp {slice_qp} out of range [0, 51]"
        )));
    }
    if ref_pic.width != src.width
        || ref_pic.height != src.height
        || ref_pic.bit_depth != src.bit_depth
    {
        return Err(Error::invalid(
            "evc p encoder: reference geometry must match the source",
        ));
    }
    let recon = YuvPicture::new(src.width, src.height, 1, src.bit_depth)?;
    let mut ctx = PCtx {
        src,
        recon,
        refp: ref_pic,
        qp: slice_qp,
        lambda: 0.57 * 2f64.powf((slice_qp as f64 - 12.0) / 3.0),
        bit_depth: src.bit_depth,
        pic_w: src.width,
        pic_h: src.height,
        side_info: SideInfoGrid::new(src.width, src.height),
        hmvp: HmvpCandList::new(),
        walk: SliceWalkInputs {
            pic_width: src.width,
            pic_height: src.height,
            ..Default::default()
        },
    };
    let mut stats = PEncStats::default();

    // Decide pass — decode order, committing recon + grid + HMVP.
    let ctus_x = src.width.div_ceil(1 << CTB_LOG2);
    let ctus_y = src.height.div_ceil(1 << CTB_LOG2);
    let mut roots = Vec::with_capacity((ctus_x * ctus_y) as usize);
    for cy in 0..ctus_y {
        for cx in 0..ctus_x {
            // §7.3.8.2 lines 2624-2625: NumHmvpCand resets at the
            // leftmost CTB of every CTU row.
            if cx == 0 {
                ctx.hmvp.reset();
            }
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

    // Emit pass — decoder-exact bin order over a replayed grid (the
    // §9.3.4.2.4 neighbour ctxIncs probe decode-time state).
    let sel = CtxSel::new(cm_init, InitType::Pb);
    let mut enc = CabacEncoder::new();
    if cm_init {
        enc.init_main_profile(InitType::Pb, slice_qp);
    }
    let mut emit_grid = SideInfoGrid::new(ctx.pic_w, ctx.pic_h);
    for (x0, y0, node) in &roots {
        emit_split_unit(
            &mut enc,
            &mut stats,
            &ctx,
            sel,
            &mut emit_grid,
            *x0,
            *y0,
            CTB_LOG2,
            CTB_LOG2,
            node,
        );
    }
    enc.encode_terminate(true); // §7.3.8.1 end_of_tile_one_bit

    if deblock {
        // The decoder's own §8.8.2 post-pass over the stamped grid —
        // inter/cbf edges are live on a P picture.
        let layout = crate::tiles::PicTileLayout::single_tile(ctx.pic_w, ctx.pic_h);
        ctx.side_info.tile_bounds = crate::tiles::TileBounds::for_loop_filters(&layout);
        crate::deblock::deblock_luma(&mut ctx.recon, &ctx.side_info, slice_qp)?;
        crate::deblock::deblock_chroma(&mut ctx.recon, &ctx.side_info, slice_qp, 0, 1)?;
        crate::deblock::deblock_chroma(&mut ctx.recon, &ctx.side_info, slice_qp, 0, 2)?;
    }
    Ok((enc.finish(), ctx.recon, stats))
}

// ---------------------------------------------------------------------
// Decide pass.
// ---------------------------------------------------------------------

fn split_geometry(ctx: &PCtx<'_>, x0: u32, y0: u32, log2_w: u32, log2_h: u32) -> (bool, bool) {
    let within = x0 + (1 << log2_w) <= ctx.pic_w && y0 + (1 << log2_h) <= ctx.pic_h;
    let can_recurse = log2_w > MIN_CB_LOG2 && log2_h > MIN_CB_LOG2;
    (within, can_recurse)
}

fn decide_split_unit(
    ctx: &mut PCtx<'_>,
    stats: &mut PEncStats,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(Node, f64)> {
    let (within, can_recurse) = split_geometry(ctx, x0, y0, log2_w, log2_h);
    let flag_present = can_recurse && within && (log2_w > 2 || log2_h > 2);

    if can_recurse && !within {
        let (children, cost) = decide_children(ctx, stats, x0, y0, log2_w, log2_h)?;
        return Ok((Node::Split(children), cost));
    }
    if !flag_present {
        let (plan, cost) = decide_leaf(ctx, stats, x0, y0, log2_w, log2_h)?;
        return Ok((Node::Leaf(plan), cost + ctx.lambda));
    }

    // Trial the leaf, snapshot, rewind, trial the split, keep the
    // cheaper state (recon + grid + HMVP all roll back together).
    let before_pix = save_region(&ctx.recon, x0, y0, log2_w, log2_h);
    let before_grid = ctx.side_info.clone();
    let before_hmvp = ctx.hmvp.clone();

    let (leaf_plan, leaf_cost) = decide_leaf(ctx, stats, x0, y0, log2_w, log2_h)?;
    let leaf_cost = leaf_cost + ctx.lambda; // split_cu_flag = 0 bin
    let after_leaf_pix = save_region(&ctx.recon, x0, y0, log2_w, log2_h);
    let after_leaf_grid = ctx.side_info.clone();
    let after_leaf_hmvp = ctx.hmvp.clone();

    restore_region(&mut ctx.recon, &before_pix, x0, y0, log2_w, log2_h);
    ctx.side_info = before_grid;
    ctx.hmvp = before_hmvp;

    let (children, split_cost) = decide_children(ctx, stats, x0, y0, log2_w, log2_h)?;
    let split_cost = split_cost + ctx.lambda; // split_cu_flag = 1 bin

    if leaf_cost <= split_cost {
        restore_region(&mut ctx.recon, &after_leaf_pix, x0, y0, log2_w, log2_h);
        ctx.side_info = after_leaf_grid;
        ctx.hmvp = after_leaf_hmvp;
        Ok((Node::Leaf(leaf_plan), leaf_cost))
    } else {
        Ok((Node::Split(children), split_cost))
    }
}

#[allow(clippy::type_complexity)]
fn decide_children(
    ctx: &mut PCtx<'_>,
    stats: &mut PEncStats,
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

/// Motion-compensated L0 prediction for one CU — the exact uni-pred
/// shape of the decoder's `apply_inter_prediction` (Baseline
/// interpolation tables, §8.5.2.6 chroma MV).
fn mc_pred(
    ctx: &PCtx<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    mv: MotionVector,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    let rv = ctx.ref_view();
    let mv16 = mv.quarter_to_sixteenth();
    let mut py = vec![0i32; w * h];
    interpolate_luma_block(rv, x0 as i32, y0 as i32, mv16, w, h, ctx.bit_depth, &mut py)?;
    let (cw, chh) = (w / 2, h / 2);
    let mvc = derive_chroma_mv(mv16, 1);
    let mut pcb = vec![0i32; cw * chh];
    let mut pcr = vec![0i32; cw * chh];
    interpolate_chroma_block(
        rv,
        1,
        (x0 / 2) as i32,
        (y0 / 2) as i32,
        mvc,
        cw,
        chh,
        ctx.bit_depth,
        &mut pcb,
    )?;
    interpolate_chroma_block(
        rv,
        2,
        (x0 / 2) as i32,
        (y0 / 2) as i32,
        mvc,
        cw,
        chh,
        ctx.bit_depth,
        &mut pcr,
    )?;
    Ok((py, pcb, pcr))
}

/// Full-pel SAD with the interpolator's clamped reference fetch —
/// the integer-search metric.
#[allow(clippy::too_many_arguments)]
fn sad_full_pel(
    ctx: &PCtx<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src: &[i32],
    fx: i32,
    fy: i32,
) -> u64 {
    let rw = ctx.refp.width as i32;
    let rh = ctx.refp.height as i32;
    let stride = ctx.refp.y_stride();
    let mut sad = 0u64;
    for j in 0..h {
        let yy = (y0 as i32 + j as i32 + fy).clamp(0, rh - 1) as usize;
        for i in 0..w {
            let xx = (x0 as i32 + i as i32 + fx).clamp(0, rw - 1) as usize;
            let r = ctx.refp.y[yy * stride + xx] as i32;
            sad += (src[j * w + i] - r).unsigned_abs() as u64;
        }
    }
    sad
}

/// Quarter-pel SAD through the decoder's own interpolation kernel.
fn sad_quarter(
    ctx: &PCtx<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src: &[i32],
    mv: MotionVector,
) -> Result<u64> {
    let rv = ctx.ref_view();
    let mut buf = vec![0i32; w * h];
    interpolate_luma_block(
        rv,
        x0 as i32,
        y0 as i32,
        mv.quarter_to_sixteenth(),
        w,
        h,
        ctx.bit_depth,
        &mut buf,
    )?;
    let max_val = (1i32 << ctx.bit_depth) - 1;
    Ok(src
        .iter()
        .zip(buf.iter())
        .map(|(&s, &p)| (s - p.clamp(0, max_val)).unsigned_abs() as u64)
        .sum())
}

/// Quarter-pel motion search: full-pel hill climb seeded from the AMVP
/// candidates (+ zero), then half- and quarter-pel refinement through
/// the §8.5.4 interpolation. Returns the best quarter-pel MV.
fn motion_search(
    ctx: &PCtx<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src: &[i32],
    seeds: &[MotionVector],
) -> Result<MotionVector> {
    // Full-pel stage.
    let mut best_f = (0i32, 0i32);
    let mut best_sad = u64::MAX;
    let consider =
        |fx: i32, fy: i32, best_f: &mut (i32, i32), best_sad: &mut u64, ctx: &PCtx<'_>| {
            // Bound the search so MVs stay sane (±64 full-pel).
            if !(-64..=64).contains(&fx) || !(-64..=64).contains(&fy) {
                return;
            }
            let s = sad_full_pel(ctx, x0, y0, w, h, src, fx, fy);
            if s < *best_sad {
                *best_sad = s;
                *best_f = (fx, fy);
            }
        };
    consider(0, 0, &mut best_f, &mut best_sad, ctx);
    for s in seeds {
        consider(s.x >> 2, s.y >> 2, &mut best_f, &mut best_sad, ctx);
    }
    for _ in 0..32 {
        let (cx, cy) = best_f;
        let before = best_sad;
        for (dx, dy) in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ] {
            consider(cx + dx, cy + dy, &mut best_f, &mut best_sad, ctx);
        }
        if best_sad == before {
            break;
        }
    }
    // Sub-pel stage: half then quarter steps around the best.
    let mut best_mv = MotionVector::quarter_pel(best_f.0 << 2, best_f.1 << 2);
    let mut best_sub = sad_quarter(ctx, x0, y0, w, h, src, best_mv)?;
    for step in [2i32, 1] {
        let center = best_mv;
        for (dx, dy) in [
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (step, -step),
            (-step, step),
            (step, step),
        ] {
            let cand = MotionVector::quarter_pel(center.x + dx, center.y + dy);
            let s = sad_quarter(ctx, x0, y0, w, h, src, cand)?;
            if s < best_sub {
                best_sub = s;
                best_mv = cand;
            }
        }
    }
    Ok(best_mv)
}

/// EG0 bin length (prefix + suffix) of an `abs_mvd` magnitude.
fn eg0_bits(v: u32) -> f64 {
    let mut p = 0u32;
    while (1u64 << (p + 1)) - 1 <= v as u64 {
        p += 1;
    }
    (2 * p + 1) as f64 + if v > 0 { 1.0 } else { 0.0 } // + sign
}

/// TR (cMax 3) bin count of an `mvp_idx`.
fn tr3_bits(v: u32) -> f64 {
    (v + u32::from(v < 3)) as f64
}

/// Quantize one inter residual plane through the decoder's §8.7 chain;
/// returns (levels, cbf, reconstructed residual, SSE vs source).
fn quantize_inter_plane(
    src: &[i32],
    pred: &[i32],
    w: usize,
    h: usize,
    qp: i32,
    bit_depth: u32,
) -> Result<(Vec<i32>, bool, Vec<i32>, f64)> {
    let n = w * h;
    let max_val = (1i32 << bit_depth) - 1;
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

/// SSE of a clipped prediction against the source (skip candidates).
fn sse_pred(src: &[i32], pred: &[i32], max_val: i32) -> f64 {
    src.iter()
        .zip(pred.iter())
        .map(|(&s, &p)| {
            let d = (p.clamp(0, max_val) - s) as f64;
            d * d
        })
        .sum()
}

/// Store `pred + res` (res may be empty ⇒ pure prediction) clipped into
/// the recon plane — the encoder-side mirror of the decoder's
/// `store_block` composition.
#[allow(clippy::too_many_arguments)]
fn store_recon(
    recon: &mut YuvPicture,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    c_idx: u32,
    pred: &[i32],
    res: &[i32],
) {
    let mut combined = pred.to_vec();
    if !res.is_empty() {
        for (a, b) in combined.iter_mut().zip(res.iter()) {
            *a += *b;
        }
    }
    recon.store_block(x0, y0, w, h, c_idx, &combined);
}

/// The four §8.5.2.4 AMVP candidates for this block over the current
/// decode-order state (`ref_idx = 0`, list 0).
fn amvp_candidates(ctx: &PCtx<'_>, x0: u32, y0: u32, w: usize, h: usize) -> [MotionVector; 4] {
    core::array::from_fn(|k| {
        baseline_amvp_select_with_grid_and_hmvp(
            k as u32,
            &ctx.side_info,
            &ctx.hmvp,
            x0 as i32,
            y0 as i32,
            w as i32,
            h as i32,
            0,
            0,
        )
    })
}

/// Commit an inter (skip or explicit) leaf into the decode-order state:
/// recon, side-info stamp, HMVP update, and the skip cell marks — the
/// exact order of the decoder's shared tail.
#[allow(clippy::too_many_arguments)]
fn commit_inter(
    ctx: &mut PCtx<'_>,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    mv: MotionVector,
    cbf_y: bool,
    skip: bool,
    pred: (&[i32], &[i32], &[i32]),
    res: (&[i32], &[i32], &[i32]),
) {
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    ctx.side_info.stamp_block(
        x0,
        y0,
        w as u32,
        h as u32,
        CuSideInfo {
            pred_mode: CuPredMode::Inter,
            cbf_luma: u8::from(cbf_y),
            mv_l0_x: mv.x,
            mv_l0_y: mv.y,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            cu_x0: x0 as u16,
            cu_y0: y0 as u16,
            cu_log2_w: log2_w as u8,
            cu_log2_h: log2_h as u8,
            qp_y: ctx.qp.clamp(0, 51) as u8,
            ..Default::default()
        },
    );
    ctx.hmvp.update(HmvpCandidate {
        mv_l0: mv,
        mv_l1: MotionVector::default(),
        ref_idx_l0: 0,
        ref_idx_l1: -1,
    });
    store_recon(&mut ctx.recon, x0, y0, w, h, 0, pred.0, res.0);
    store_recon(
        &mut ctx.recon,
        x0 / 2,
        y0 / 2,
        w / 2,
        h / 2,
        1,
        pred.1,
        res.1,
    );
    store_recon(
        &mut ctx.recon,
        x0 / 2,
        y0 / 2,
        w / 2,
        h / 2,
        2,
        pred.2,
        res.2,
    );
    if skip {
        mark_cu_skip_cells(&mut ctx.side_info, x0, y0, w as u32, h as u32);
    }
}

fn decide_leaf(
    ctx: &mut PCtx<'_>,
    stats: &mut PEncStats,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(PLeaf, f64)> {
    let w = 1usize << log2_w;
    let h = 1usize << log2_h;
    let (wc, hc) = (w / 2, h / 2);
    let bd = ctx.bit_depth;
    let max_val = (1i32 << bd) - 1;
    let src_y = gather_block(&ctx.src.y, ctx.src.y_stride(), x0, y0, w, h);
    let src_cb = gather_block(&ctx.src.cb, ctx.src.c_stride(), x0 / 2, y0 / 2, wc, hc);
    let src_cr = gather_block(&ctx.src.cr, ctx.src.c_stride(), x0 / 2, y0 / 2, wc, hc);

    let cands = amvp_candidates(ctx, x0, y0, w, h);

    // ---- skip ladder: 4 mvp slots, whole-CU prediction, no residual.
    type SkipChoice = (u32, MotionVector, Vec<i32>, Vec<i32>, Vec<i32>);
    let mut best_skip: Option<SkipChoice> = None;
    let mut best_skip_cost = f64::INFINITY;
    for (k, &mv) in cands.iter().enumerate() {
        let (py, pcb, pcr) = mc_pred(ctx, x0, y0, w, h, mv)?;
        let dist = sse_pred(&src_y, &py, max_val)
            + sse_pred(&src_cb, &pcb, max_val)
            + sse_pred(&src_cr, &pcr, max_val);
        let bits = 1.0 + tr3_bits(k as u32); // cu_skip + mvp_idx
        let cost = dist + ctx.lambda * bits;
        if cost < best_skip_cost {
            best_skip_cost = cost;
            best_skip = Some((k as u32, mv, py, pcb, pcr));
        }
    }

    // ---- explicit inter: ME + residual.
    let me_mv = motion_search(ctx, x0, y0, w, h, &src_y, &cands)?;
    // Choose the mvp minimizing the signalling cost of this MV.
    let (best_mvp_idx, _) = cands
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let mvd = MotionVector::quarter_pel(me_mv.x - p.x, me_mv.y - p.y);
            let bits = tr3_bits(k as u32)
                + eg0_bits(mvd.x.unsigned_abs())
                + eg0_bits(mvd.y.unsigned_abs());
            (k as u32, bits)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("4 candidates");
    let mvp = cands[best_mvp_idx as usize];
    let mvd = MotionVector::quarter_pel(me_mv.x - mvp.x, me_mv.y - mvp.y);
    let (py, pcb, pcr) = mc_pred(ctx, x0, y0, w, h, me_mv)?;
    let (levels_y, cbf_y, res_y, dist_y) = quantize_inter_plane(&src_y, &py, w, h, ctx.qp, bd)?;
    let (levels_cb, cbf_cb, res_cb, dist_cb) =
        quantize_inter_plane(&src_cb, &pcb, wc, hc, ctx.qp, bd)?;
    let (levels_cr, cbf_cr, res_cr, dist_cr) =
        quantize_inter_plane(&src_cr, &pcr, wc, hc, ctx.qp, bd)?;
    // §7.4.9.5: with quiet chroma cbf_luma is inferred 1, so an
    // "explicit CU with luma-only-zero residual" collapses to
    // cbf_all = 0 (drop the chroma residuals too if luma is quiet and
    // they are the only content — cheaper to keep them; the inference
    // only bites when cbf_y == false && !cbf_cb && !cbf_cr, which IS
    // representable as cbf_all = 0).
    let any_cbf = cbf_y || cbf_cb || cbf_cr;
    let mut inter_bits = 1.0 // cu_skip = 0
        + 1.0 // pred_mode_flag = 0
        + tr3_bits(best_mvp_idx)
        + eg0_bits(mvd.x.unsigned_abs())
        + eg0_bits(mvd.y.unsigned_abs())
        + 1.0; // cbf_all
    let mut inter_dist = 0f64;
    if any_cbf {
        inter_bits += 2.0; // cbf_cb + cbf_cr
        if cbf_cb || cbf_cr {
            inter_bits += 1.0; // cbf_luma present
        }
        inter_bits += rle_bits_estimate(&levels_y, cbf_y)
            + rle_bits_estimate(&levels_cb, cbf_cb)
            + rle_bits_estimate(&levels_cr, cbf_cr);
        inter_dist += dist_y + dist_cb + dist_cr;
    } else {
        inter_dist += sse_pred(&src_y, &py, max_val)
            + sse_pred(&src_cb, &pcb, max_val)
            + sse_pred(&src_cr, &pcr, max_val);
    }
    // Representability guard (§7.4.9.5): cbf_all = 1 with quiet chroma
    // infers cbf_luma = 1 — if luma quantized to zero while a chroma
    // plane carries content, that is representable (cbf_luma bin = 0);
    // the impossible shape (all three quiet under cbf_all = 1) is
    // exactly `!any_cbf`, which we signal as cbf_all = 0.
    let inter_cost = inter_dist + ctx.lambda * inter_bits;

    // ---- intra (single tree, chroma follows the luma mode).
    let refs = ctx.recon.fetch_intra_refs(x0, y0, w, h, 0);
    let mut best_intra: Option<(usize, Vec<i32>, bool, Vec<i32>)> = None;
    let mut best_intra_luma_cost = f64::INFINITY;
    for (mode_idx, &mode) in MODES.iter().enumerate() {
        let (levels, cbf, res, dist) =
            quantize_block(&refs, mode, &src_y, w, h, ctx.qp, bd, max_val)?;
        let bits = 2.0 + (mode_idx + 1) as f64 + 1.0 + rle_bits_estimate(&levels, cbf);
        let cost = dist + ctx.lambda * bits;
        if cost < best_intra_luma_cost {
            best_intra_luma_cost = cost;
            best_intra = Some((mode_idx, levels, cbf, res));
        }
    }
    let (i_mode, i_levels_y, i_cbf_y, i_res_y) = best_intra.expect("5 candidates");
    // Chroma with the same mode (decoder: IntraPredModeC = IntraPredModeY).
    let mode = MODES[i_mode];
    let refs_cb = ctx.recon.fetch_intra_refs(x0 / 2, y0 / 2, wc, hc, 1);
    let (i_levels_cb, i_cbf_cb, i_res_cb, i_dist_cb) =
        quantize_block(&refs_cb, mode, &src_cb, wc, hc, ctx.qp, bd, max_val)?;
    let refs_cr = ctx.recon.fetch_intra_refs(x0 / 2, y0 / 2, wc, hc, 2);
    let (i_levels_cr, i_cbf_cr, i_res_cr, i_dist_cr) =
        quantize_block(&refs_cr, mode, &src_cr, wc, hc, ctx.qp, bd, max_val)?;
    let intra_cost = best_intra_luma_cost
        + i_dist_cb
        + i_dist_cr
        + ctx.lambda
            * (2.0
                + rle_bits_estimate(&i_levels_cb, i_cbf_cb)
                + rle_bits_estimate(&i_levels_cr, i_cbf_cr));

    stats.leaves += 1;

    // ---- choose and commit.
    if best_skip_cost <= inter_cost && best_skip_cost <= intra_cost {
        let (k, mv, py, pcb, pcr) = best_skip.expect("skip ladder evaluated");
        commit_inter(
            ctx,
            x0,
            y0,
            log2_w,
            log2_h,
            mv,
            false,
            true,
            (&py, &pcb, &pcr),
            (&[], &[], &[]),
        );
        stats.skip_cus += 1;
        Ok((PLeaf::Skip { mvp_idx: k, mv }, best_skip_cost))
    } else if inter_cost <= intra_cost {
        commit_inter(
            ctx,
            x0,
            y0,
            log2_w,
            log2_h,
            me_mv,
            cbf_y,
            false,
            (&py, &pcb, &pcr),
            (&res_y, &res_cb, &res_cr),
        );
        stats.inter_cus += 1;
        if !any_cbf {
            stats.cbf_all_zero_cus += 1;
        }
        Ok((
            PLeaf::Inter {
                mvp_idx: best_mvp_idx,
                mv: me_mv,
                mvd,
                levels_y,
                cbf_y,
                levels_cb,
                cbf_cb,
                levels_cr,
                cbf_cr,
            },
            inter_cost,
        ))
    } else {
        // Commit the intra reconstruction through the decoder's own
        // kernels (luma then chroma, both with the luma mode), then
        // stamp — the same aggregate the decoder records.
        ctx.side_info.stamp_block(
            x0,
            y0,
            w as u32,
            h as u32,
            CuSideInfo {
                pred_mode: CuPredMode::Intra,
                cbf_luma: u8::from(i_cbf_y),
                cu_x0: x0 as u16,
                cu_y0: y0 as u16,
                cu_log2_w: log2_w as u8,
                cu_log2_h: log2_h as u8,
                intra_luma_mode: i_mode as u8,
                qp_y: ctx.qp.clamp(0, 51) as u8,
                ..Default::default()
            },
        );
        intra_reconstruct_cb_in_tile(
            &mut ctx.recon,
            x0,
            y0,
            log2_w,
            log2_h,
            mode,
            0,
            &i_res_y,
            None,
        )?;
        intra_reconstruct_cb_in_tile(
            &mut ctx.recon,
            x0,
            y0,
            log2_w,
            log2_h,
            mode,
            1,
            &i_res_cb,
            None,
        )?;
        intra_reconstruct_cb_in_tile(
            &mut ctx.recon,
            x0,
            y0,
            log2_w,
            log2_h,
            mode,
            2,
            &i_res_cr,
            None,
        )?;
        stats.intra_cus += 1;
        Ok((
            PLeaf::Intra {
                mode_idx: i_mode,
                levels_y: i_levels_y,
                cbf_y: i_cbf_y,
                levels_cb: i_levels_cb,
                cbf_cb: i_cbf_cb,
                levels_cr: i_levels_cr,
                cbf_cr: i_cbf_cr,
            },
            intra_cost,
        ))
    }
}

// ---------------------------------------------------------------------
// Emit pass — decoder-exact bin order.
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_split_unit(
    enc: &mut CabacEncoder,
    stats: &mut PEncStats,
    ctx: &PCtx<'_>,
    sel: CtxSel,
    grid: &mut SideInfoGrid,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    node: &Node,
) {
    let (within, can_recurse) = split_geometry(ctx, x0, y0, log2_w, log2_h);
    let flag_present = can_recurse && within && (log2_w > 2 || log2_h > 2);
    let (split_t, split_i) = sel.ctx(MainCtxTable::SplitCuFlag, 0);
    match node {
        Node::Split(children) => {
            if flag_present {
                enc.encode_decision(split_t, split_i, 1);
                stats.split_flag_bins += 1;
            }
            for (cx, cy, clw, clh, child) in children {
                emit_split_unit(enc, stats, ctx, sel, grid, *cx, *cy, *clw, *clh, child);
            }
        }
        Node::Leaf(plan) => {
            if flag_present {
                enc.encode_decision(split_t, split_i, 0);
                stats.split_flag_bins += 1;
            }
            emit_leaf(enc, ctx, sel, grid, x0, y0, log2_w, log2_h, plan);
        }
    }
}

/// `cu_skip_flag` / `pred_mode_flag` context — the decoder's Table 47 /
/// Table 61 §9.3.4.2.4 neighbour ctxIncs over the emit-order grid
/// (Baseline collapse: `(0, 0)`).
fn skip_flag_ctx(
    ctx: &PCtx<'_>,
    sel: CtxSel,
    grid: &SideInfoGrid,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> (usize, usize) {
    if sel.cm_init {
        let inc = ctx_inc_neighbour_cells(grid, &ctx.walk, x0, y0, log2_w, log2_h, 2, |c| {
            c.cu_skip != 0
        });
        sel.ctx(MainCtxTable::CuSkipFlag, inc)
    } else {
        // Table 95, sps_cm_init_flag == 0 row: ctxInc 0.
        sel.ctx_shaped(MainCtxTable::CuSkipFlag, 0, 0)
    }
}

fn pred_mode_flag_ctx(
    ctx: &PCtx<'_>,
    sel: CtxSel,
    grid: &SideInfoGrid,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> (usize, usize) {
    if sel.cm_init {
        let inc = ctx_inc_neighbour_cells(grid, &ctx.walk, x0, y0, log2_w, log2_h, 3, |c| {
            matches!(c.pred_mode, CuPredMode::Intra | CuPredMode::Ibc)
        });
        sel.ctx(MainCtxTable::PredModeFlag, inc)
    } else {
        // Table 95, sps_cm_init_flag == 0 row: ctxInc 0.
        sel.ctx_shaped(MainCtxTable::PredModeFlag, 0, 0)
    }
}

/// TR cMax 3 write for `mvp_idx_l0` — the dual of the decoder's
/// `decode_tr_regular(3, 0, …)` with the Table 48 per-bin ctxInc.
fn emit_mvp_idx(enc: &mut CabacEncoder, sel: CtxSel, v: u32) {
    // Table 95: per-bin ctxInc 0,1,2 under both entropy shapes.
    let table = MainCtxTable::MvpIdx;
    let (t, off) = if sel.cm_init {
        (table.as_usize(), table.ctx_idx_offset(sel.init_type))
    } else {
        (0, table.cm0_ctx_idx_offset(sel.init_type))
    };
    let idx = |b: u32| -> usize { off + (b as usize).min(2) };
    for b in 0..v {
        enc.encode_decision(t, idx(b), 1);
    }
    if v < 3 {
        enc.encode_decision(t, idx(v), 0);
    }
}

/// Signed `abs_mvd` + `mvd_sign_flag` write — the dual of the decoder's
/// `decode_signed_mvd` (EG0 bin0 regular on Table 73 under cm_init,
/// all-bypass otherwise; sign bypass when non-zero).
fn emit_signed_mvd(enc: &mut CabacEncoder, sel: CtxSel, v: i32) {
    let abs = v.unsigned_abs();
    // Table 95: bin0 regular under both entropy shapes, rest bypass.
    {
        let (t, i) = sel.ctx(MainCtxTable::AbsMvd, 0);
        enc.encode_eg0_first_regular(t, i, abs);
    }
    if abs != 0 {
        enc.encode_bypass(u8::from(v < 0));
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_leaf(
    enc: &mut CabacEncoder,
    ctx: &PCtx<'_>,
    sel: CtxSel,
    grid: &mut SideInfoGrid,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    plan: &PLeaf,
) {
    let (w, h) = (1u32 << log2_w, 1u32 << log2_h);
    let stamp_inter = |grid: &mut SideInfoGrid, mv: MotionVector, cbf_y: bool| {
        grid.stamp_block(
            x0,
            y0,
            w,
            h,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                cbf_luma: u8::from(cbf_y),
                mv_l0_x: mv.x,
                mv_l0_y: mv.y,
                ref_idx_l0: 0,
                ref_idx_l1: -1,
                cu_x0: x0 as u16,
                cu_y0: y0 as u16,
                cu_log2_w: log2_w as u8,
                cu_log2_h: log2_h as u8,
                qp_y: ctx.qp.clamp(0, 51) as u8,
                ..Default::default()
            },
        );
    };
    match plan {
        PLeaf::Skip { mvp_idx, mv } => {
            let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 1); // cu_skip_flag = 1
            emit_mvp_idx(enc, sel, *mvp_idx);
            // No residual syntax (§7.3.8.4).
            stamp_inter(grid, *mv, false);
            mark_cu_skip_cells(grid, x0, y0, w, h);
        }
        PLeaf::Inter {
            mvp_idx,
            mv,
            mvd,
            levels_y,
            cbf_y,
            levels_cb,
            cbf_cb,
            levels_cr,
            cbf_cr,
        } => {
            let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // cu_skip_flag = 0
            let (t, i) = pred_mode_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // pred_mode_flag = 0 (MODE_INTER)
            emit_mvp_idx(enc, sel, *mvp_idx);
            emit_signed_mvd(enc, sel, mvd.x);
            emit_signed_mvd(enc, sel, mvd.y);
            let any = *cbf_y || *cbf_cb || *cbf_cr;
            let (t, i) = sel.ctx(MainCtxTable::CbfAll, 0);
            enc.encode_decision(t, i, u8::from(any)); // cbf_all
            if any {
                let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
                enc.encode_decision(t, i, u8::from(*cbf_cb));
                let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
                enc.encode_decision(t, i, u8::from(*cbf_cr));
                if *cbf_cb || *cbf_cr {
                    let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
                    enc.encode_decision(t, i, u8::from(*cbf_y));
                } else {
                    debug_assert!(*cbf_y, "quiet chroma infers cbf_luma = 1 (§7.4.9.5)");
                }
                if *cbf_y {
                    emit_residual_rle(enc, sel, 0, levels_y, log2_w, log2_h);
                }
                if *cbf_cb {
                    emit_residual_rle(enc, sel, 1, levels_cb, log2_w - 1, log2_h - 1);
                }
                if *cbf_cr {
                    emit_residual_rle(enc, sel, 2, levels_cr, log2_w - 1, log2_h - 1);
                }
            }
            stamp_inter(grid, *mv, *cbf_y);
        }
        PLeaf::Intra {
            mode_idx,
            levels_y,
            cbf_y,
            levels_cb,
            cbf_cb,
            levels_cr,
            cbf_cr,
        } => {
            let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // cu_skip_flag = 0
            let (t, i) = pred_mode_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 1); // pred_mode_flag = 1 (MODE_INTRA)
                                          // intra_pred_mode — U over Table 62 (bin0 → 0, later → 1).
            {
                let table = MainCtxTable::IntraPredMode;
                let (t, off) = if sel.cm_init {
                    (table.as_usize(), table.ctx_idx_offset(sel.init_type))
                } else {
                    (0, table.cm0_ctx_idx_offset(sel.init_type))
                };
                enc.encode_u_regular_capped(*mode_idx as u32, 63, t, |b| off + (b as usize).min(1));
            }
            // Single-tree intra TU: cbf_cb, cbf_cr, then cbf_luma
            // (always present — MODE_INTRA), then luma/cb/cr residuals.
            let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
            enc.encode_decision(t, i, u8::from(*cbf_cb));
            let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
            enc.encode_decision(t, i, u8::from(*cbf_cr));
            let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
            enc.encode_decision(t, i, u8::from(*cbf_y));
            if *cbf_y {
                emit_residual_rle(enc, sel, 0, levels_y, log2_w, log2_h);
            }
            if *cbf_cb {
                emit_residual_rle(enc, sel, 1, levels_cb, log2_w - 1, log2_h - 1);
            }
            if *cbf_cr {
                emit_residual_rle(enc, sel, 2, levels_cr, log2_w - 1, log2_h - 1);
            }
            grid.stamp_block(
                x0,
                y0,
                w,
                h,
                CuSideInfo {
                    pred_mode: CuPredMode::Intra,
                    cbf_luma: u8::from(*cbf_y),
                    cu_x0: x0 as u16,
                    cu_y0: y0 as u16,
                    cu_log2_w: log2_w as u8,
                    cu_log2_h: log2_h as u8,
                    intra_luma_mode: *mode_idx as u8,
                    qp_y: ctx.qp.clamp(0, 51) as u8,
                    ..Default::default()
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice_data::{
        decode_baseline_inter_slice, CodingTreeGates, InterDecodeInputs, SliceDecodeInputs,
    };

    fn walk_inputs(w: u32, h: u32, cm_init: bool) -> SliceWalkInputs {
        SliceWalkInputs {
            pic_width: w,
            pic_height: h,
            ctb_log2_size_y: CTB_LOG2,
            min_cb_log2_size_y: MIN_CB_LOG2,
            max_tb_log2_size_y: 6,
            chroma_format_idc: 1,
            tree_gates: CodingTreeGates {
                sps_cm_init_flag: cm_init,
                ..CodingTreeGates::default()
            },
            ..Default::default()
        }
    }

    fn ref_view(p: &YuvPicture) -> RefPictureView<'_> {
        RefPictureView {
            y: &p.y,
            cb: &p.cb,
            cr: &p.cr,
            width: p.width,
            height: p.height,
            y_stride: p.y_stride(),
            c_stride: p.c_stride(),
            chroma_format_idc: p.chroma_format_idc,
        }
    }

    /// Deterministic pseudo-natural frame `t` of a moving-scene GOP:
    /// a diagonal gradient plus a bright square translating (3, 2)
    /// pixels per frame and a noise band, so P frames exercise real
    /// motion (skip / non-zero-MV inter / intra refresh all appear).
    fn synth_moving(w: u32, h: u32, t: u32, bit_depth: u32) -> YuvPicture {
        let mut pic = YuvPicture::new(w, h, 1, bit_depth).unwrap();
        let scale = 1u32 << (bit_depth - 8);
        let mut s = 0x1234_5678u32 ^ (t * 0x9E37);
        let mut noise = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 24) & 0x07) as i32 - 4
        };
        let (bx, by) = ((8 + 3 * t) as usize, (6 + 2 * t) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let mut v = 40 + ((x as i32 + 2 * y as i32) % 140);
                if x >= bx && x < bx + 12 && y >= by && y < by + 10 {
                    v = 220;
                }
                if y + 8 >= h as usize {
                    v += noise();
                }
                pic.y[y * w as usize + x] = (v.clamp(0, 255) as u16) * scale as u16;
            }
        }
        let cw = w.div_ceil(2) as usize;
        let chh = h.div_ceil(2) as usize;
        for y in 0..chh {
            for x in 0..cw {
                pic.cb[y * cw + x] = ((110 + (x + y + t as usize) % 40) as u16) * scale as u16;
                pic.cr[y * cw + x] = ((130 + (2 * x + y) % 50) as u16) * scale as u16;
            }
        }
        pic
    }

    fn decode_p(
        payload: &[u8],
        refp: &YuvPicture,
        w: u32,
        h: u32,
        qp: i32,
        deblock: bool,
        cm_init: bool,
    ) -> (YuvPicture, crate::slice_data::InterDecodeStats) {
        let refs = [ref_view(refp)];
        let inputs = InterDecodeInputs {
            walk: walk_inputs(w, h, cm_init),
            decode: SliceDecodeInputs {
                slice_qp: qp,
                bit_depth_luma: refp.bit_depth,
                bit_depth_chroma: refp.bit_depth,
                enable_deblock: deblock,
                ..Default::default()
            },
            slice_is_b: false,
            num_ref_idx_active_minus1_l0: 0,
            num_ref_idx_active_minus1_l1: 0,
            ref_list_l0: &refs,
            ref_list_l1: &[],
            inter_tool_gates: Default::default(),
            pocs: Default::default(),
            col_pic: None,
        };
        decode_baseline_inter_slice(payload, inputs).expect("p slice must decode")
    }

    /// THE core pin: a whole GOP (IDR + 4 P) must run the
    /// encode→decode == recon loop byte-exactly at every frame, on
    /// both entropy shapes and both deblock settings, with each frame
    /// predicting from the previous reconstruction exactly as the
    /// decoder's DPB serves it.
    #[test]
    fn p_gop_round_trip_recon_exact() {
        let (w, h) = (96u32, 64u32);
        for &cm_init in &[false, true] {
            for &deblock in &[false, true] {
                let qp = 30;
                let frame0 = synth_moving(w, h, 0, 8);
                let (_p, mut refr, _s) =
                    crate::slice_enc::encode_idr_slice_data_opts(&frame0, qp, deblock, cm_init)
                        .unwrap();
                for t in 1..=4u32 {
                    let src = synth_moving(w, h, t, 8);
                    let (payload, recon, stats) =
                        encode_p_slice_data(&src, &refr, qp, deblock, cm_init).unwrap();
                    let (dec, dec_stats) = decode_p(&payload, &refr, w, h, qp, deblock, cm_init);
                    assert_eq!(dec.y, recon.y, "t{t} cm{cm_init} db{deblock}: luma");
                    assert_eq!(dec.cb, recon.cb, "t{t} cm{cm_init} db{deblock}: cb");
                    assert_eq!(dec.cr, recon.cr, "t{t} cm{cm_init} db{deblock}: cr");
                    assert_eq!(dec_stats.ctus, stats.ctus);
                    assert_eq!(
                        stats.leaves,
                        stats.skip_cus + stats.inter_cus + stats.intra_cus,
                        "mode ladder accounting"
                    );
                    refr = recon;
                }
            }
        }
    }

    /// A static scene collapses to skip CUs and a P frame costs a tiny
    /// fraction of the intra frame (the whole point of low-delay P).
    #[test]
    fn static_scene_collapses_to_skip() {
        let (w, h) = (64u32, 64u32);
        let frame = synth_moving(w, h, 0, 8);
        let qp = 30;
        let (idr_payload, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&frame, qp, false, true).unwrap();
        let (p_payload, recon, stats) =
            encode_p_slice_data(&frame, &refr, qp, false, true).unwrap();
        assert!(
            stats.skip_cus > 0 && stats.intra_cus == 0,
            "static content must ride the skip ladder: {stats:?}"
        );
        assert!(
            p_payload.len() * 20 < idr_payload.len(),
            "static P frame ({}) must be tiny next to the IDR ({})",
            p_payload.len(),
            idr_payload.len()
        );
        // The recon of a skip-only frame IS the reference.
        assert_eq!(recon.y, refr.y);
        let (dec, _) = decode_p(&p_payload, &refr, w, h, qp, false, true);
        assert_eq!(dec.y, recon.y);
    }

    /// Pure translation is caught by the quarter-pel ME: a scene shifted
    /// by exactly (−3, −2) full pel encodes into non-intra CUs with the
    /// dominant motion, and the decode stays recon-exact.
    #[test]
    fn translation_is_tracked_by_me() {
        let (w, h) = (64u32, 48u32);
        let f0 = synth_moving(w, h, 0, 8);
        let f1 = synth_moving(w, h, 1, 8);
        let qp = 26;
        let (_ip, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        let (payload, recon, stats) = encode_p_slice_data(&f1, &refr, qp, false, true).unwrap();
        assert!(
            stats.skip_cus + stats.inter_cus > stats.intra_cus,
            "a translating scene must be predominantly inter: {stats:?}"
        );
        let (dec, dec_stats) = decode_p(&payload, &refr, w, h, qp, false, true);
        assert_eq!(dec.y, recon.y);
        // The decoder saw real motion syntax.
        assert!(dec_stats.cu_skip_flag_bins > 0);
    }

    /// Deblock-on P frames: the §8.8.2 pass is pixel-effective on inter
    /// edges (unlike the r429 all-intra no-op), and the filtered output
    /// still round-trips byte-exactly.
    #[test]
    fn p_deblock_is_live_and_exact() {
        let (w, h) = (64u32, 64u32);
        let f0 = synth_moving(w, h, 0, 8);
        let f1 = synth_moving(w, h, 2, 8);
        let qp = 40;
        let (_ip, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        let (pay_db, rec_db, _) = encode_p_slice_data(&f1, &refr, qp, true, true).unwrap();
        let (pay_no, rec_no, _) = encode_p_slice_data(&f1, &refr, qp, false, true).unwrap();
        assert_eq!(pay_db, pay_no, "the payload is filter-independent");
        assert_ne!(
            rec_db.y, rec_no.y,
            "§8.8.2 must move samples on a P picture (live inter edges)"
        );
        let (dec_db, dec_stats) = decode_p(&pay_db, &refr, w, h, qp, true, true);
        assert_eq!(dec_db.y, rec_db.y, "filtered decode == filtered recon");
        assert!(dec_stats.deblock_edges > 0);
    }

    /// A scene cut inside the GOP: the P frame's content shares nothing
    /// with the reference, so the ladder falls back to intra CUs — the
    /// single-tree intra-in-P write path (chroma-first cbfs, luma-mode
    /// chroma prediction) must round-trip byte-exactly.
    #[test]
    fn scene_cut_p_frame_goes_intra_and_round_trips() {
        let (w, h) = (64u32, 64u32);
        let f0 = synth_moving(w, h, 0, 8);
        // A completely different scene: inverted checkerboard field.
        let mut f1 = YuvPicture::new(w, h, 1, 8).unwrap();
        for y in 0..h as usize {
            for x in 0..w as usize {
                f1.y[y * w as usize + x] = if (x / 8 + y / 8) % 2 == 0 { 30 } else { 225 };
            }
        }
        for (i, v) in f1.cb.iter_mut().enumerate() {
            *v = (60 + i % 130) as u16;
        }
        for (i, v) in f1.cr.iter_mut().enumerate() {
            *v = (190 - i % 120) as u16;
        }
        let qp = 30;
        let (_ip, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        for &cm_init in &[false, true] {
            let (payload, recon, stats) =
                encode_p_slice_data(&f1, &refr, qp, false, cm_init).unwrap();
            assert!(
                stats.intra_cus > stats.skip_cus,
                "a scene cut must refresh with intra CUs: {stats:?}"
            );
            let (dec, _) = decode_p(&payload, &refr, w, h, qp, false, cm_init);
            assert_eq!(dec.y, recon.y, "cm{cm_init}: luma");
            assert_eq!(dec.cb, recon.cb, "cm{cm_init}: cb");
            assert_eq!(dec.cr, recon.cr, "cm{cm_init}: cr");
        }
    }

    /// 12-bit P frame closes the encoder depth matrix on the inter path.
    #[test]
    fn p_twelve_bit_round_trip_exact() {
        let (w, h) = (32u32, 32u32);
        let f0 = synth_moving(w, h, 0, 12);
        let f1 = synth_moving(w, h, 1, 12);
        let qp = 30;
        let (_ip, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        let (payload, recon, _stats) = encode_p_slice_data(&f1, &refr, qp, false, true).unwrap();
        let (dec, _) = decode_p(&payload, &refr, w, h, qp, false, true);
        assert_eq!(dec.y, recon.y);
        assert_eq!(dec.cb, recon.cb);
        assert_eq!(dec.cr, recon.cr);
        assert!(recon.y.iter().any(|&v| v > 1023), "true 12-bit range");
    }

    /// 10-bit P GOP: the depth-parameterized loop holds byte-exactly.
    #[test]
    fn p_ten_bit_round_trip_exact() {
        let (w, h) = (32u32, 32u32);
        let f0 = synth_moving(w, h, 0, 10);
        let f1 = synth_moving(w, h, 1, 10);
        let qp = 24;
        let (_ip, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        let (payload, recon, _stats) = encode_p_slice_data(&f1, &refr, qp, false, true).unwrap();
        let (dec, _) = decode_p(&payload, &refr, w, h, qp, false, true);
        assert_eq!(dec.y, recon.y);
        assert_eq!(dec.cb, recon.cb);
        assert_eq!(dec.cr, recon.cr);
        assert!(recon.y.iter().any(|&v| v > 255), "true 10-bit range");
    }

    /// Ragged dimensions (multiples of 4, not of the 64-CTU): the
    /// implicit boundary splits ride the P emit path too, across the
    /// QP extremes.
    #[test]
    fn p_boundary_split_dims_and_qp_extremes_round_trip() {
        for &(w, h) in &[(100u32, 60u32), (72, 40)] {
            let f0 = synth_moving(w, h, 0, 8);
            let f1 = synth_moving(w, h, 1, 8);
            for &qp in &[4i32, 51] {
                let (_ip, refr, _s) =
                    crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
                let (payload, recon, stats) =
                    encode_p_slice_data(&f1, &refr, qp, false, true).unwrap();
                assert!(stats.ctus >= 2, "{w}x{h}: multiple CTUs walked");
                let (dec, dec_stats) = decode_p(&payload, &refr, w, h, qp, false, true);
                assert_eq!(dec.y, recon.y, "{w}x{h} qp{qp}: luma");
                assert_eq!(dec.cb, recon.cb, "{w}x{h} qp{qp}: cb");
                assert_eq!(dec.cr, recon.cr, "{w}x{h} qp{qp}: cr");
                assert_eq!(dec_stats.ctus, stats.ctus);
            }
        }
    }

    /// Determinism across the whole P pipeline.
    #[test]
    fn p_encode_is_deterministic() {
        let (w, h) = (48u32, 48u32);
        let f0 = synth_moving(w, h, 0, 8);
        let f1 = synth_moving(w, h, 3, 8);
        let (_ip, refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, 33, false, true).unwrap();
        let (a, _, _) = encode_p_slice_data(&f1, &refr, 33, false, true).unwrap();
        let (b, _, _) = encode_p_slice_data(&f1, &refr, 33, false, true).unwrap();
        assert_eq!(a, b);
    }

    /// QCIF P-GOP characterization (run with `--nocapture` for the
    /// numbers): IDR + 4 moving-scene P frames at QP 30 — every P frame
    /// must round-trip recon-exact and cost well under the IDR, and
    /// PSNR must stay in the same band across the GOP (no drift: the
    /// closed loop predicts from bit-exact reconstructions).
    #[test]
    fn p_gop_qcif_characterization() {
        let (w, h) = (176u32, 144u32);
        let qp = 30;
        let pixels = (w * h) as f64;
        let f0 = synth_moving(w, h, 0, 8);
        let (idr_payload, mut refr, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        let psnr = |src: &YuvPicture, rec: &YuvPicture| -> f64 {
            let mse: f64 = src
                .y
                .iter()
                .zip(rec.y.iter())
                .map(|(&a, &b)| {
                    let d = a as f64 - b as f64;
                    d * d
                })
                .sum::<f64>()
                / pixels;
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };
        let idr_psnr = psnr(&f0, &refr);
        eprintln!(
            "qcif gop qp {qp}: IDR {} bytes  {idr_psnr:5.2} dB",
            idr_payload.len()
        );
        let mut worst_p_psnr = f64::INFINITY;
        for t in 1..=4u32 {
            let src = synth_moving(w, h, t, 8);
            let (payload, recon, stats) =
                encode_p_slice_data(&src, &refr, qp, false, true).unwrap();
            let (dec, _) = decode_p(&payload, &refr, w, h, qp, false, true);
            assert_eq!(dec.y, recon.y, "t{t}: recon-exact");
            let p = psnr(&src, &recon);
            worst_p_psnr = worst_p_psnr.min(p);
            eprintln!(
                "qcif gop qp {qp}: P{t} {:6} bytes  {p:5.2} dB  \
                 (skip {} / inter {} / intra {} of {} leaves)",
                payload.len(),
                stats.skip_cus,
                stats.inter_cus,
                stats.intra_cus,
                stats.leaves
            );
            assert!(
                payload.len() * 2 < idr_payload.len(),
                "t{t}: P frame {} must undercut half the IDR {}",
                payload.len(),
                idr_payload.len()
            );
            refr = recon;
        }
        // The λ trade-off spends P bits sparingly (skip-heavy), so P
        // quality sits a few dB under the IDR — pin a sane envelope
        // rather than parity: no drift collapse, ≥ 45 dB absolute.
        assert!(
            worst_p_psnr >= 45.0 && worst_p_psnr + 10.0 >= idr_psnr,
            "P quality out of band (worst {worst_p_psnr:.2} vs IDR {idr_psnr:.2})"
        );
    }

    /// Input validation.
    #[test]
    fn p_rejects_bad_inputs() {
        let src = YuvPicture::new(64, 64, 1, 8).unwrap();
        let refp = YuvPicture::new(64, 64, 1, 8).unwrap();
        assert!(encode_p_slice_data(&src, &refp, 52, false, false).is_err());
        let small_ref = YuvPicture::new(32, 32, 1, 8).unwrap();
        assert!(encode_p_slice_data(&src, &small_ref, 30, false, false).is_err());
    }
}
