//! **P / B slice encoder** (round 431 low-delay P; round 452 B slices +
//! multi-reference): the write-side dual of the §7.3.8 P/B-slice
//! `slice_data()` walker in
//! [`crate::slice_data::decode_baseline_inter_slice`] under the
//! Baseline toolset (`sps_admvp_flag == 0`).
//!
//! ## Mode ladder (per leaf, RD over the decoder's own reconstruction)
//!
//! * **skip** — `cu_skip_flag = 1` + `mvp_idx_l0` (TR cMax 3; on a B
//!   slice also `mvp_idx_l1`): the MVs come straight from the §8.5.2.4
//!   AMVP candidate lists (the same
//!   [`crate::slice_data::baseline_amvp_select_with_grid_and_hmvp`]
//!   the decoder runs, over the encoder's decode-order side-info grid
//!   + §8.5.2.7 HMVP list) at `refIdx = 0` (§8.5.2.2 eqs. 445/448), a
//!   B skip being the eq. 988 bi-average, no MVD, **no residual
//!   syntax** (§7.3.8.4: the `cbf_all` block lives in the non-skip
//!   branch);
//! * **direct** (B only) — `direct_mode_flag = 1`: the §8.5.2.5
//!   temporal derivation (the decoder's own
//!   [`crate::slice_data::derive_direct_mode_mvs`] over the retained
//!   `RefPicList1[ 0 ]` motion field), bi-predicted per errata #313,
//!   then the `cbf_all`-gated residual;
//! * **explicit inter** — `cu_skip_flag = 0`, `pred_mode_flag = 0`
//!   (`direct_mode_flag = 0` on B), on B the `inter_pred_idc` list
//!   choice (PRED_L0 / PRED_L1 / PRED_BI), and per used list the
//!   `ref_idx_lX` (TR, present when `num_ref_idx_active_minus1[ X ] >
//!   0`), `mvp_idx_lX` and the `abs_mvd`/sign pair: quarter-pel motion
//!   search against every active reference through the decoder's
//!   §8.5.4 Baseline interpolation kernels (full-pel hill climb seeded
//!   from that reference's AMVP candidates, then half-/quarter-pel
//!   refinement), the bi-prediction candidate pairing the best list-0
//!   and list-1 vectors with one list-1 sub-pel re-fit against the
//!   averaged prediction; residual through the §8.7-inverted
//!   [`crate::quant_enc::forward_quantize`]; the CU signals `cbf_all`
//!   (line 3028) — 0 elides the whole `transform_unit()`, and with
//!   quiet chroma `cbf_luma` is inferred 1 (§7.4.9.5);
//! * **intra** — `pred_mode_flag = 1` + the Baseline 5-mode search
//!   (single tree: chroma predicts with the luma mode, mirroring the
//!   decoder's `decode_inter_intra_cu`).
//!
//! The coding tree is the same bottom-up quad `split_unit()` RD of the
//! IDR encoder. The decide pass runs in decode order, committing the
//! chosen reconstruction *and* the decoder-visible state (side-info
//! stamps for both lists, `cu_skip` cell marks, HMVP updates — reset
//! at each CTU row per §7.3.8.2) so every AMVP list the encoder
//! consults is exactly the list the decoder will build. The emit pass
//! replays the decided tree bin for bin (initType 1 contexts; under
//! `sps_cm_init_flag == 1` the §9.3.4.2.4 neighbour ctxIncs are
//! re-derived over an emit-side grid stamped in the same order).
//!
//! With `deblock` the §8.8.2 post-pass runs over the recon with the
//! stamped side info — on a P/B picture the inter/cbf edges carry live
//! boundary strengths, so the returned picture is the decoder's
//! filtered output byte for byte. The stamped motion field is returned
//! too: it is the §8.3.4 collocated field a later B picture's direct
//! mode reads, so the encoder's DPB keeps it beside the picture exactly
//! as the decoder's does.

use oxideav_core::{Error, Result};

use crate::bin_cost::BitCostModel;
use crate::cabac::{BinSink, CabacEncoder, InitType};
use crate::cabac_init::{CtxSel, MainCtxTable};
use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
use crate::dequant::scale_and_inverse_transform;
use crate::hmvp::{HmvpCandList, HmvpCandidate};
use crate::inter::{
    average_bipred, derive_chroma_mv, interpolate_chroma_block, interpolate_luma_block,
    MotionVector, RefPictureView,
};
use crate::picture::{intra_reconstruct_cb_in_tile, YuvPicture};
use crate::rdoq::RdoqInputs;
use crate::slice_data::{
    baseline_amvp_select_with_grid_and_hmvp, ctx_inc_neighbour_cells, derive_direct_mode_mvs,
    mark_cu_skip_cells, ColPicInputs, InterPocs, SliceWalkInputs,
};
use crate::slice_enc::{
    emit_intra_pred_mode, emit_residual_rle, gather_block, quantize_block, quantize_residual,
    restore_region, save_region, MODES,
};

/// Geometry constants of the encoder SPS (§7.4.3.1 `sps_btt_flag == 0`
/// defaults): 64×64 CTU, 4×4 minimum CB. With `MaxTbLog2SizeY == 6`
/// (eq. 51) no leaf ever TB-splits, so every CU is a single
/// `transform_unit()`.
const CTB_LOG2: u32 = 6;
const MIN_CB_LOG2: u32 = 2;

/// Per-picture P/B-encode statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PEncStats {
    pub ctus: u32,
    pub leaves: u32,
    /// Leaves coded `cu_skip_flag = 1`.
    pub skip_cus: u32,
    /// Non-skip MODE_INTER leaves (explicit MV, or direct on B).
    pub inter_cus: u32,
    /// Explicit/direct inter leaves whose `cbf_all` was signalled 0.
    pub cbf_all_zero_cus: u32,
    /// MODE_INTRA leaves.
    pub intra_cus: u32,
    /// `split_cu_flag` bins emitted.
    pub split_flag_bins: u32,
    /// B-slice leaves coded `direct_mode_flag = 1` (subset of
    /// `inter_cus`).
    pub direct_cus: u32,
    /// Explicit leaves signalling `inter_pred_idc = PRED_BI` (subset of
    /// `inter_cus`).
    pub bi_cus: u32,
    /// Explicit leaves signalling `inter_pred_idc = PRED_L1` (subset of
    /// `inter_cus`).
    pub l1_only_cus: u32,
    /// Explicit leaves with a non-zero `ref_idx_lX` on some list.
    pub multi_ref_cus: u32,
}

/// One active reference picture as the slice encoder addresses it —
/// the DPB picture (post-§8.8.2 when deblocking is on) and its
/// `PicOrderCntVal`.
#[derive(Clone, Copy)]
pub struct RefEntry<'a> {
    pub pic: &'a YuvPicture,
    pub poc: i32,
}

/// The §8.3.4 collocated motion field a B slice's direct mode reads:
/// `RefPicList1[ 0 ]`'s stamped side-info grid, its POC and the POCs of
/// its own list 0 (eq. 665 `refPicListTemp[ 0 ]`).
#[derive(Clone, Copy)]
pub struct ColMotion<'a> {
    pub grid: &'a SideInfoGrid,
    pub poc: i32,
    pub ref_pocs_l0: &'a [i32],
}

/// Inputs to [`encode_inter_slice_data`].
#[derive(Clone, Copy)]
pub struct InterEncInputs<'a> {
    /// `RefPicList0` in `ref_idx` order (the §8.3.2.2 list the decoder
    /// rebuilds from its DPB; at least one entry).
    pub refs_l0: &'a [RefEntry<'a>],
    /// `RefPicList1` for a B slice (empty for P).
    pub refs_l1: &'a [RefEntry<'a>],
    /// `slice_type == B`.
    pub slice_is_b: bool,
    /// `PicOrderCntVal` of the picture being coded.
    pub curr_poc: i32,
    /// The collocated motion field (B slices; `None` degrades direct
    /// mode to the eq. 668-671 zero vectors exactly like the decoder).
    pub col: Option<ColMotion<'a>>,
    pub slice_qp: i32,
    pub deblock: bool,
    pub cm_init: bool,
}

/// Output of [`encode_inter_slice_data`].
pub struct InterEncOutput {
    /// The CABAC `slice_data()` payload.
    pub payload: Vec<u8>,
    /// The reconstruction the decoder reproduces byte for byte
    /// (post-§8.8.2 when `deblock`).
    pub recon: YuvPicture,
    /// The stamped per-4×4 motion field (the picture's §8.3.4
    /// collocated field for later slices).
    pub side_info: SideInfoGrid,
    pub stats: PEncStats,
}

/// Per-list explicit prediction of a decided leaf.
#[derive(Clone, Copy)]
struct ListPred {
    ref_idx: u32,
    mvp_idx: u32,
    mv: MotionVector,
    mvd: MotionVector,
}

/// Quantized residual of a decided leaf (levels + cbfs per plane).
#[derive(Clone, Default)]
struct Residual {
    levels_y: Vec<i32>,
    cbf_y: bool,
    levels_cb: Vec<i32>,
    cbf_cb: bool,
    levels_cr: Vec<i32>,
    cbf_cr: bool,
}

impl Residual {
    fn any(&self) -> bool {
        self.cbf_y || self.cbf_cb || self.cbf_cr
    }
}

/// One decided leaf.
enum PLeaf {
    Skip {
        mvp_idx: [u32; 2],
        mv: [MotionVector; 2],
    },
    Direct {
        mv: [MotionVector; 2],
        res: Residual,
    },
    Inter {
        l0: Option<ListPred>,
        l1: Option<ListPred>,
        res: Residual,
    },
    Intra {
        mode_idx: usize,
        res: Residual,
    },
}

/// The best explicit-inter candidate of a leaf under evaluation.
struct Explicit {
    leaf: PLeaf,
    pred: Pred,
    planes: Pred,
    cost: f64,
}

/// A decided `split_unit()` subtree.
enum Node {
    Split(Vec<(u32, u32, u32, u32, Node)>),
    Leaf(PLeaf),
}

struct PCtx<'a> {
    src: &'a YuvPicture,
    recon: YuvPicture,
    refs: [Vec<RefPictureView<'a>>; 2],
    ref_pocs: [Vec<i32>; 2],
    col: Option<ColPicInputs<'a>>,
    slice_is_b: bool,
    curr_poc: i32,
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
    /// The entropy shape the emit pass runs under.
    sel: CtxSel,
}

impl<'a> PCtx<'a> {
    fn ref_view(&self, list: usize, ref_idx: u32) -> RefPictureView<'a> {
        self.refs[list][ref_idx as usize]
    }

    fn n_refs(&self, list: usize) -> u32 {
        self.refs[list].len() as u32
    }

    fn pocs(&self) -> InterPocs<'_> {
        InterPocs {
            curr_poc: self.curr_poc,
            ref_pocs_l0: &self.ref_pocs[0],
            ref_pocs_l1: &self.ref_pocs[1],
        }
    }
}

fn view_of(p: &YuvPicture) -> RefPictureView<'_> {
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

/// Encode one Baseline **P** picture's `slice_data()` payload against
/// `ref_pic` as the single list-0 reference (the round-431 shape:
/// `RefPicList0 = [ ref_pic ]`, POC distance 1). Returns the CABAC
/// payload, the reconstruction the decoder reproduces byte for byte
/// (post-§8.8.2 when `deblock`), and the statistics.
pub fn encode_p_slice_data(
    src: &YuvPicture,
    ref_pic: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
) -> Result<(Vec<u8>, YuvPicture, PEncStats)> {
    let refs = [RefEntry {
        pic: ref_pic,
        poc: 0,
    }];
    let out = encode_inter_slice_data(
        src,
        InterEncInputs {
            refs_l0: &refs,
            refs_l1: &[],
            slice_is_b: false,
            curr_poc: 1,
            col: None,
            slice_qp,
            deblock,
            cm_init,
        },
    )?;
    Ok((out.payload, out.recon, out.stats))
}

/// Encode one Baseline P or B picture's `slice_data()` payload against
/// the given reference lists (each entry exactly as the decoder holds it
/// in the DPB — post-§8.8.2 when deblocking is on).
pub fn encode_inter_slice_data(
    src: &YuvPicture,
    inputs: InterEncInputs<'_>,
) -> Result<InterEncOutput> {
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
    if !(0..=51).contains(&inputs.slice_qp) {
        return Err(Error::invalid(format!(
            "evc p encoder: slice_qp {} out of range [0, 51]",
            inputs.slice_qp
        )));
    }
    if inputs.refs_l0.is_empty() || (inputs.slice_is_b && inputs.refs_l1.is_empty()) {
        return Err(Error::invalid(
            "evc p encoder: every active reference list needs at least one picture",
        ));
    }
    if inputs.refs_l0.len() > 15 || inputs.refs_l1.len() > 15 {
        return Err(Error::invalid(
            "evc p encoder: num_ref_idx_active_minus1 is bounded by 14 (§7.4.5)",
        ));
    }
    for r in inputs.refs_l0.iter().chain(inputs.refs_l1.iter()) {
        if r.pic.width != src.width
            || r.pic.height != src.height
            || r.pic.bit_depth != src.bit_depth
        {
            return Err(Error::invalid(
                "evc p encoder: reference geometry must match the source",
            ));
        }
    }
    let slice_qp = inputs.slice_qp;
    let recon = YuvPicture::new(src.width, src.height, 1, src.bit_depth)?;
    let refs = [
        inputs.refs_l0.iter().map(|r| view_of(r.pic)).collect(),
        inputs.refs_l1.iter().map(|r| view_of(r.pic)).collect(),
    ];
    let ref_pocs = [
        inputs.refs_l0.iter().map(|r| r.poc).collect(),
        inputs.refs_l1.iter().map(|r| r.poc).collect(),
    ];
    let col = inputs.col.map(|c| ColPicInputs {
        grid: c.grid,
        col_poc: c.poc,
        ref_pocs_l0: c.ref_pocs_l0,
        ref_pocs_l1: &[],
    });
    let mut ctx = PCtx {
        src,
        recon,
        refs,
        ref_pocs,
        col,
        slice_is_b: inputs.slice_is_b,
        curr_poc: inputs.curr_poc,
        qp: slice_qp,
        lambda: crate::slice_enc::rd_lambda(slice_qp, src.bit_depth),
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
        sel: CtxSel::new(inputs.cm_init, InitType::Pb),
    };
    let mut stats = PEncStats::default();
    // The decide pass's rate model — the emit pass's context table,
    // advanced by every decided bin in decode order.
    let mut model = BitCostModel::new();
    if inputs.cm_init {
        model.init_main_profile(InitType::Pb, slice_qp);
    }

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
                &mut model,
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
    let cm_init = inputs.cm_init;
    let sel = ctx.sel;
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

    if inputs.deblock {
        // The decoder's own §8.8.2 post-pass over the stamped grid —
        // inter/cbf edges are live on a P/B picture.
        let layout = crate::tiles::PicTileLayout::single_tile(ctx.pic_w, ctx.pic_h);
        ctx.side_info.tile_bounds = crate::tiles::TileBounds::for_loop_filters(&layout);
        crate::deblock::deblock_luma(&mut ctx.recon, &ctx.side_info, slice_qp)?;
        crate::deblock::deblock_chroma(&mut ctx.recon, &ctx.side_info, slice_qp, 0, 1)?;
        crate::deblock::deblock_chroma(&mut ctx.recon, &ctx.side_info, slice_qp, 0, 2)?;
    }
    Ok(InterEncOutput {
        payload: enc.finish(),
        recon: ctx.recon,
        side_info: ctx.side_info,
        stats,
    })
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
    model: &mut BitCostModel,
    stats: &mut PEncStats,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(Node, f64)> {
    let (within, can_recurse) = split_geometry(ctx, x0, y0, log2_w, log2_h);
    let flag_present = can_recurse && within && (log2_w > 2 || log2_h > 2);

    if can_recurse && !within {
        let (children, cost) = decide_children(ctx, model, stats, x0, y0, log2_w, log2_h)?;
        return Ok((Node::Split(children), cost));
    }
    if !flag_present {
        let (plan, cost) = decide_leaf(ctx, model, x0, y0, log2_w, log2_h)?;
        return Ok((Node::Leaf(plan), cost));
    }

    // Trial the leaf (its split_cu_flag = 0 bin committed first, in
    // emit order), snapshot, rewind, trial the split, keep the cheaper
    // state (recon + grid + HMVP + rate model all roll back together).
    let (split_t, split_i) = ctx.sel.ctx(MainCtxTable::SplitCuFlag, 0);
    let before_pix = save_region(&ctx.recon, x0, y0, log2_w, log2_h);
    let before_grid = ctx.side_info.clone();
    let before_hmvp = ctx.hmvp.clone();
    let before_model = model.clone();

    let leaf_flag_bits = model.commit(|m| m.encode_decision(split_t, split_i, 0));
    let (leaf_plan, leaf_cost) = decide_leaf(ctx, model, x0, y0, log2_w, log2_h)?;
    let leaf_cost = leaf_cost + ctx.lambda * leaf_flag_bits;
    let after_leaf_pix = save_region(&ctx.recon, x0, y0, log2_w, log2_h);
    let after_leaf_grid = ctx.side_info.clone();
    let after_leaf_hmvp = ctx.hmvp.clone();
    let after_leaf_model = model.clone();

    restore_region(&mut ctx.recon, &before_pix, x0, y0, log2_w, log2_h);
    ctx.side_info = before_grid;
    ctx.hmvp = before_hmvp;
    *model = before_model;

    let split_flag_bits = model.commit(|m| m.encode_decision(split_t, split_i, 1));
    let (children, split_cost) = decide_children(ctx, model, stats, x0, y0, log2_w, log2_h)?;
    let split_cost = split_cost + ctx.lambda * split_flag_bits;

    if leaf_cost <= split_cost {
        restore_region(&mut ctx.recon, &after_leaf_pix, x0, y0, log2_w, log2_h);
        ctx.side_info = after_leaf_grid;
        ctx.hmvp = after_leaf_hmvp;
        *model = after_leaf_model;
        Ok((Node::Leaf(leaf_plan), leaf_cost))
    } else {
        Ok((Node::Split(children), split_cost))
    }
}

#[allow(clippy::type_complexity)]
fn decide_children(
    ctx: &mut PCtx<'_>,
    model: &mut BitCostModel,
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
            model,
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

/// Motion-compensated prediction planes of one CU: `(Y, Cb, Cr)`.
type Pred = (Vec<i32>, Vec<i32>, Vec<i32>);

/// Uni-directional motion-compensated prediction for one CU from
/// `(list, ref_idx)` — the exact single-list shape of the decoder's
/// `apply_inter_prediction` (Baseline interpolation tables, §8.5.2.6
/// chroma MV).
#[allow(clippy::too_many_arguments)]
fn mc_pred(
    ctx: &PCtx<'_>,
    list: usize,
    ref_idx: u32,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    mv: MotionVector,
) -> Result<Pred> {
    let rv = ctx.ref_view(list, ref_idx);
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

/// eq. 988 bi-average of two prediction triples.
fn bi_average(a: &Pred, b: &Pred) -> Pred {
    let mut y = vec![0i32; a.0.len()];
    let mut cb = vec![0i32; a.1.len()];
    let mut cr = vec![0i32; a.2.len()];
    average_bipred(&a.0, &b.0, &mut y);
    average_bipred(&a.1, &b.1, &mut cb);
    average_bipred(&a.2, &b.2, &mut cr);
    (y, cb, cr)
}

/// Prediction for a `(l0, l1)` pair — uni or the eq. 988 average.
#[allow(clippy::too_many_arguments)]
fn mc_pred_pair(
    ctx: &PCtx<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    l0: Option<(u32, MotionVector)>,
    l1: Option<(u32, MotionVector)>,
) -> Result<Pred> {
    match (l0, l1) {
        (Some((r0, m0)), Some((r1, m1))) => {
            let a = mc_pred(ctx, 0, r0, x0, y0, w, h, m0)?;
            let b = mc_pred(ctx, 1, r1, x0, y0, w, h, m1)?;
            Ok(bi_average(&a, &b))
        }
        (Some((r0, m0)), None) => mc_pred(ctx, 0, r0, x0, y0, w, h, m0),
        (None, Some((r1, m1))) => mc_pred(ctx, 1, r1, x0, y0, w, h, m1),
        (None, None) => Err(Error::invalid("evc p encoder: prediction with no list")),
    }
}

/// Full-pel SAD with the interpolator's clamped reference fetch —
/// the integer-search metric.
#[allow(clippy::too_many_arguments)]
fn sad_full_pel(
    rv: RefPictureView<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src: &[i32],
    fx: i32,
    fy: i32,
) -> u64 {
    let rw = rv.width as i32;
    let rh = rv.height as i32;
    let stride = rv.y_stride;
    let mut sad = 0u64;
    for j in 0..h {
        let yy = (y0 as i32 + j as i32 + fy).clamp(0, rh - 1) as usize;
        for i in 0..w {
            let xx = (x0 as i32 + i as i32 + fx).clamp(0, rw - 1) as usize;
            let r = rv.y[yy * stride + xx] as i32;
            sad += (src[j * w + i] - r).unsigned_abs() as u64;
        }
    }
    sad
}

/// Quarter-pel SAD through the decoder's own interpolation kernel.
#[allow(clippy::too_many_arguments)]
fn sad_quarter(
    rv: RefPictureView<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src: &[i32],
    mv: MotionVector,
    bit_depth: u32,
) -> Result<u64> {
    let mut buf = vec![0i32; w * h];
    interpolate_luma_block(
        rv,
        x0 as i32,
        y0 as i32,
        mv.quarter_to_sixteenth(),
        w,
        h,
        bit_depth,
        &mut buf,
    )?;
    let max_val = (1i32 << bit_depth) - 1;
    Ok(src
        .iter()
        .zip(buf.iter())
        .map(|(&s, &p)| (s - p.clamp(0, max_val)).unsigned_abs() as u64)
        .sum())
}

/// Quarter-pel motion search against one reference picture: full-pel
/// hill climb seeded from the AMVP candidates (+ zero), then half- and
/// quarter-pel refinement through the §8.5.4 interpolation. Returns
/// the best quarter-pel MV.
#[allow(clippy::too_many_arguments)]
fn motion_search(
    rv: RefPictureView<'_>,
    bit_depth: u32,
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
    let consider = |fx: i32, fy: i32, best_f: &mut (i32, i32), best_sad: &mut u64| {
        // Bound the search so MVs stay sane (±64 full-pel).
        if !(-64..=64).contains(&fx) || !(-64..=64).contains(&fy) {
            return;
        }
        let s = sad_full_pel(rv, x0, y0, w, h, src, fx, fy);
        if s < *best_sad {
            *best_sad = s;
            *best_f = (fx, fy);
        }
    };
    consider(0, 0, &mut best_f, &mut best_sad);
    for s in seeds {
        consider(s.x >> 2, s.y >> 2, &mut best_f, &mut best_sad);
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
            consider(cx + dx, cy + dy, &mut best_f, &mut best_sad);
        }
        if best_sad == before {
            break;
        }
    }
    // Sub-pel stage: half then quarter steps around the best.
    let mut best_mv = MotionVector::quarter_pel(best_f.0 << 2, best_f.1 << 2);
    let mut best_sub = sad_quarter(rv, x0, y0, w, h, src, best_mv, bit_depth)?;
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
            let s = sad_quarter(rv, x0, y0, w, h, src, cand, bit_depth)?;
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

/// TR bin count of a `ref_idx_lX` (`cMax = num_ref_idx_active_minus1`;
/// absent — zero bins — when the list holds a single picture).
fn ref_idx_bits(v: u32, n_refs: u32) -> f64 {
    if n_refs <= 1 {
        return 0.0;
    }
    let c_max = n_refs - 1;
    (v + u32::from(v < c_max)) as f64
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
    rdoq: &RdoqInputs<'_>,
) -> Result<(Vec<i32>, bool, Vec<i32>, f64)> {
    let n = w * h;
    let max_val = (1i32 << bit_depth) - 1;
    let diff: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let (levels, cbf) = quantize_residual(&diff, w, h, qp, bit_depth, Some(rdoq))?;
    let mut res = vec![0i32; n];
    if cbf {
        scale_and_inverse_transform(&levels, &mut res, w, h, qp, bit_depth, false)?;
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

/// Quantized residual of an inter prediction against the source, with
/// the distortion and the exact rate of the §7.3.8.5 residual syntax
/// it would emit (`cbf_all`, then with any content `cbf_cb` +
/// `cbf_cr`, the `cbf_luma` bin only when chroma carries something —
/// quiet chroma infers it 1, the all-quiet shape is exactly
/// `cbf_all = 0` — and the RLE strings), costed at the current
/// context state of `model`.
/// Returns `(residual, reconstructed residual planes, distortion, bits)`.
#[allow(clippy::type_complexity)]
fn quantize_inter_residual(
    ctx: &PCtx<'_>,
    model: &mut BitCostModel,
    src: (&[i32], &[i32], &[i32]),
    pred: &Pred,
    w: usize,
    h: usize,
) -> Result<(Residual, Pred, f64, f64)> {
    let bd = ctx.bit_depth;
    let max_val = (1i32 << bd) - 1;
    let (wc, hc) = (w / 2, h / 2);
    // §8.7.1: quantize at qP = Qp′ (eqs. 1050-1052).
    let qp_y = crate::dequant::qp_prime_y(ctx.qp, bd);
    let qp_c = crate::dequant::qp_prime_c(ctx.qp, 0, bd, false);
    let sel = ctx.sel;
    let lambda_c = crate::slice_enc::rd_lambda_at_qp_prime(qp_c);
    let rdoq = |c_idx: u32, table: MainCtxTable| {
        let lambda = if c_idx == 0 { ctx.lambda } else { lambda_c };
        RdoqInputs::new(&*model, lambda, sel, c_idx, table)
    };
    let (levels_y, cbf_y, res_y, dist_y) = quantize_inter_plane(
        src.0,
        &pred.0,
        w,
        h,
        qp_y,
        bd,
        &rdoq(0, MainCtxTable::CbfLuma),
    )?;
    let (levels_cb, cbf_cb, res_cb, dist_cb) = quantize_inter_plane(
        src.1,
        &pred.1,
        wc,
        hc,
        qp_c,
        bd,
        &rdoq(1, MainCtxTable::CbfCb),
    )?;
    let (levels_cr, cbf_cr, res_cr, dist_cr) = quantize_inter_plane(
        src.2,
        &pred.2,
        wc,
        hc,
        qp_c,
        bd,
        &rdoq(2, MainCtxTable::CbfCr),
    )?;
    let res = Residual {
        levels_y,
        cbf_y,
        levels_cb,
        cbf_cb,
        levels_cr,
        cbf_cr,
    };
    let (log2_w, log2_h) = ((w as u32).trailing_zeros(), (h as u32).trailing_zeros());
    let bits = model.measure(|m| emit_inter_residual(m, ctx.sel, &res, log2_w, log2_h));
    let dist = if res.any() {
        dist_y + dist_cb + dist_cr
    } else {
        sse_pred(src.0, &pred.0, max_val)
            + sse_pred(src.1, &pred.1, max_val)
            + sse_pred(src.2, &pred.2, max_val)
    };
    let planes = if res.any() {
        (res_y, res_cb, res_cr)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    Ok((res, planes, dist, bits))
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
/// decode-order state, for `(list, ref_idx)`.
fn amvp_candidates(
    ctx: &PCtx<'_>,
    list: usize,
    ref_idx: u32,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
) -> [MotionVector; 4] {
    core::array::from_fn(|k| {
        baseline_amvp_select_with_grid_and_hmvp(
            k as u32,
            &ctx.side_info,
            &ctx.hmvp,
            x0 as i32,
            y0 as i32,
            w as i32,
            h as i32,
            ref_idx as i8,
            list as u8,
        )
    })
}

/// Motion of a committed inter leaf per list: `(ref_idx, mv)`.
type ListMotion = Option<(u32, MotionVector)>;

#[allow(clippy::too_many_arguments)]
fn inter_side_info(
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    l0: ListMotion,
    l1: ListMotion,
    cbf_y: bool,
    qp: i32,
) -> CuSideInfo {
    CuSideInfo {
        pred_mode: CuPredMode::Inter,
        cbf_luma: u8::from(cbf_y),
        mv_l0_x: l0.map(|(_, m)| m.x).unwrap_or(0),
        mv_l0_y: l0.map(|(_, m)| m.y).unwrap_or(0),
        mv_l1_x: l1.map(|(_, m)| m.x).unwrap_or(0),
        mv_l1_y: l1.map(|(_, m)| m.y).unwrap_or(0),
        ref_idx_l0: l0.map(|(r, _)| r as i8).unwrap_or(-1),
        ref_idx_l1: l1.map(|(r, _)| r as i8).unwrap_or(-1),
        cu_x0: x0 as u16,
        cu_y0: y0 as u16,
        cu_log2_w: log2_w as u8,
        cu_log2_h: log2_h as u8,
        qp_y: qp.clamp(0, 51) as u8,
        ..Default::default()
    }
}

/// Commit an inter (skip / direct / explicit) leaf into the
/// decode-order state: recon, side-info stamp, HMVP update, and the
/// skip cell marks — the exact order of the decoder's shared tail.
#[allow(clippy::too_many_arguments)]
fn commit_inter(
    ctx: &mut PCtx<'_>,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    l0: ListMotion,
    l1: ListMotion,
    cbf_y: bool,
    skip: bool,
    pred: &Pred,
    res: &Pred,
) {
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    ctx.side_info.stamp_block(
        x0,
        y0,
        w as u32,
        h as u32,
        inter_side_info(x0, y0, log2_w, log2_h, l0, l1, cbf_y, ctx.qp),
    );
    ctx.hmvp.update(HmvpCandidate {
        mv_l0: l0.map(|(_, m)| m).unwrap_or_default(),
        mv_l1: l1.map(|(_, m)| m).unwrap_or_default(),
        ref_idx_l0: l0.map(|(r, _)| r as i8).unwrap_or(-1),
        ref_idx_l1: l1.map(|(r, _)| r as i8).unwrap_or(-1),
    });
    store_recon(&mut ctx.recon, x0, y0, w, h, 0, &pred.0, &res.0);
    store_recon(
        &mut ctx.recon,
        x0 / 2,
        y0 / 2,
        w / 2,
        h / 2,
        1,
        &pred.1,
        &res.1,
    );
    store_recon(
        &mut ctx.recon,
        x0 / 2,
        y0 / 2,
        w / 2,
        h / 2,
        2,
        &pred.2,
        &res.2,
    );
    if skip {
        mark_cu_skip_cells(&mut ctx.side_info, x0, y0, w as u32, h as u32);
    }
}

/// Explicit uni-prediction on one list, per active reference: motion
/// search against the reference, the AMVP slot minimizing the MVD
/// cost, and the full residual RD. Returns one `(ListPred, cost)` per
/// `ref_idx`, in list order (`cost = SSE + λ·bits` of the uni CU).
#[allow(clippy::too_many_arguments)]
fn uni_per_ref(
    ctx: &PCtx<'_>,
    model: &mut BitCostModel,
    list: usize,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src: (&[i32], &[i32], &[i32]),
) -> Result<Vec<(ListPred, f64)>> {
    let n_refs = ctx.n_refs(list);
    let mut out = Vec::with_capacity(n_refs as usize);
    for r in 0..n_refs {
        let cands = amvp_candidates(ctx, list, r, x0, y0, w, h);
        let rv = ctx.ref_view(list, r);
        let mv = motion_search(rv, ctx.bit_depth, x0, y0, w, h, src.0, &cands)?;
        let lp = fit_mvp(&cands, r, mv);
        let pred = mc_pred(ctx, list, r, x0, y0, w, h, mv)?;
        let (_res, _planes, dist, res_bits) =
            quantize_inter_residual(ctx, model, src, &pred, w, h)?;
        let bits = ref_idx_bits(r, n_refs) + list_pred_bits(&lp) + res_bits;
        out.push((lp, dist + ctx.lambda * bits));
    }
    Ok(out)
}

/// The cheapest entry of a [`uni_per_ref`] result.
fn best_of(per_ref: &[(ListPred, f64)]) -> ListPred {
    per_ref
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("at least one reference")
        .0
}

/// Choose the AMVP slot minimizing the MVD signalling of `mv`.
fn fit_mvp(cands: &[MotionVector; 4], ref_idx: u32, mv: MotionVector) -> ListPred {
    let (k, _) = cands
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let mvd = MotionVector::quarter_pel(mv.x - p.x, mv.y - p.y);
            let bits = tr3_bits(k as u32)
                + eg0_bits(mvd.x.unsigned_abs())
                + eg0_bits(mvd.y.unsigned_abs());
            (k as u32, bits)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("4 candidates");
    let p = cands[k as usize];
    ListPred {
        ref_idx,
        mvp_idx: k,
        mv,
        mvd: MotionVector::quarter_pel(mv.x - p.x, mv.y - p.y),
    }
}

/// `mvp_idx` + `abs_mvd`/sign bins of one list's explicit motion.
fn list_pred_bits(lp: &ListPred) -> f64 {
    tr3_bits(lp.mvp_idx) + eg0_bits(lp.mvd.x.unsigned_abs()) + eg0_bits(lp.mvd.y.unsigned_abs())
}

/// One sub-pel re-fit of the list-1 vector against the averaged
/// prediction (the list-0 half fixed): half- then quarter-pel
/// neighbours, picking the pair minimizing the luma SSE of the eq. 988
/// average.
#[allow(clippy::too_many_arguments)]
fn refine_bi_l1(
    ctx: &PCtx<'_>,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
    src_y: &[i32],
    pred_l0_y: &[i32],
    r1: u32,
    mv1: MotionVector,
) -> Result<MotionVector> {
    let rv = ctx.ref_view(1, r1);
    let max_val = (1i32 << ctx.bit_depth) - 1;
    let mut buf = vec![0i32; w * h];
    let mut avg = vec![0i32; w * h];
    let mut eval = |mv: MotionVector| -> Result<f64> {
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
        average_bipred(pred_l0_y, &buf, &mut avg);
        Ok(sse_pred(src_y, &avg, max_val))
    };
    let mut best = mv1;
    let mut best_cost = eval(mv1)?;
    for step in [2i32, 1] {
        let center = best;
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
            let c = eval(cand)?;
            if c < best_cost {
                best_cost = c;
                best = cand;
            }
        }
    }
    Ok(best)
}

fn decide_leaf(
    ctx: &mut PCtx<'_>,
    model: &mut BitCostModel,
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
    let is_b = ctx.slice_is_b;
    let sel = ctx.sel;
    let src_y = gather_block(&ctx.src.y, ctx.src.y_stride(), x0, y0, w, h);
    let src_cb = gather_block(&ctx.src.cb, ctx.src.c_stride(), x0 / 2, y0 / 2, wc, hc);
    let src_cr = gather_block(&ctx.src.cr, ctx.src.c_stride(), x0 / 2, y0 / 2, wc, hc);
    let src = (&src_y[..], &src_cb[..], &src_cr[..]);
    let sse_all = |p: &Pred| -> f64 {
        sse_pred(&src_y, &p.0, max_val)
            + sse_pred(&src_cb, &p.1, max_val)
            + sse_pred(&src_cr, &p.2, max_val)
    };
    // Exact rate of a candidate leaf's whole bin string at the current
    // context state (the neighbour-derived ctxIncs probe the committed
    // decode-order grid, exactly as the emit pass will).
    let leaf_bits = |model: &mut BitCostModel, ctx: &PCtx<'_>, leaf: &PLeaf| -> f64 {
        model.measure(|m| {
            emit_leaf_bins(m, ctx, sel, &ctx.side_info, x0, y0, log2_w, log2_h, leaf);
        })
    };

    // ---- skip ladder: mvp slots at refIdx 0 (eqs. 445/448), whole-CU
    // prediction, no residual. A B skip pairs a list-0 and a list-1
    // slot (bi-average).
    let cands0 = amvp_candidates(ctx, 0, 0, x0, y0, w, h);
    let cands1 = if is_b {
        amvp_candidates(ctx, 1, 0, x0, y0, w, h)
    } else {
        [MotionVector::default(); 4]
    };
    let mut best_skip: Option<(PLeaf, Pred)> = None;
    let mut best_skip_cost = f64::INFINITY;
    let preds0: Vec<Pred> = cands0
        .iter()
        .map(|&mv| mc_pred(ctx, 0, 0, x0, y0, w, h, mv))
        .collect::<Result<_>>()?;
    if is_b {
        let preds1: Vec<Pred> = cands1
            .iter()
            .map(|&mv| mc_pred(ctx, 1, 0, x0, y0, w, h, mv))
            .collect::<Result<_>>()?;
        for (k0, p0) in preds0.iter().enumerate() {
            for (k1, p1) in preds1.iter().enumerate() {
                let avg = bi_average(p0, p1);
                let leaf = PLeaf::Skip {
                    mvp_idx: [k0 as u32, k1 as u32],
                    mv: [cands0[k0], cands1[k1]],
                };
                let cost = sse_all(&avg) + ctx.lambda * leaf_bits(model, ctx, &leaf);
                if cost < best_skip_cost {
                    best_skip_cost = cost;
                    best_skip = Some((leaf, avg));
                }
            }
        }
    } else {
        for (k0, p0) in preds0.into_iter().enumerate() {
            let leaf = PLeaf::Skip {
                mvp_idx: [k0 as u32, 0],
                mv: [cands0[k0], MotionVector::default()],
            };
            let cost = sse_all(&p0) + ctx.lambda * leaf_bits(model, ctx, &leaf);
            if cost < best_skip_cost {
                best_skip_cost = cost;
                best_skip = Some((leaf, p0));
            }
        }
    }

    // ---- direct (B): the §8.5.2.5 pair, bi-predicted, cbf_all-gated
    // residual.
    let mut direct: Option<(PLeaf, Pred, Pred, f64)> = None;
    if is_b {
        let (mv0, mv1) =
            derive_direct_mode_mvs(&ctx.pocs(), ctx.col.as_ref(), x0, y0, w as u32, h as u32);
        let pred = mc_pred_pair(ctx, x0, y0, w, h, Some((0, mv0)), Some((0, mv1)))?;
        let (res, planes, dist, _) = quantize_inter_residual(ctx, model, src, &pred, w, h)?;
        let leaf = PLeaf::Direct {
            mv: [mv0, mv1],
            res,
        };
        let cost = dist + ctx.lambda * leaf_bits(model, ctx, &leaf);
        direct = Some((leaf, pred, planes, cost));
    }

    // ---- explicit inter: per-list ME (every reference) + residual;
    // on B also the bi-prediction pairing.
    let mut explicit = {
        let lp0 = best_of(&uni_per_ref(ctx, model, 0, x0, y0, w, h, src)?);
        let pred0 = mc_pred(ctx, 0, lp0.ref_idx, x0, y0, w, h, lp0.mv)?;
        let (res0, planes0, dist0, _) = quantize_inter_residual(ctx, model, src, &pred0, w, h)?;
        let leaf = PLeaf::Inter {
            l0: Some(lp0),
            l1: None,
            res: res0,
        };
        let cost = dist0 + ctx.lambda * leaf_bits(model, ctx, &leaf);
        Explicit {
            leaf,
            pred: pred0,
            planes: planes0,
            cost,
        }
    };
    if is_b {
        let PLeaf::Inter { l0: Some(lp0), .. } = explicit.leaf else {
            unreachable!("list-0 candidate")
        };
        let per_ref1 = uni_per_ref(ctx, model, 1, x0, y0, w, h, src)?;
        let lp1 = best_of(&per_ref1);
        let pred1 = mc_pred(ctx, 1, lp1.ref_idx, x0, y0, w, h, lp1.mv)?;
        let (res1, planes1, dist1, _) = quantize_inter_residual(ctx, model, src, &pred1, w, h)?;
        let leaf1 = PLeaf::Inter {
            l0: None,
            l1: Some(lp1),
            res: res1,
        };
        let cost1 = dist1 + ctx.lambda * leaf_bits(model, ctx, &leaf1);
        if cost1 < explicit.cost {
            explicit = Explicit {
                leaf: leaf1,
                pred: pred1,
                planes: planes1,
                cost: cost1,
            };
        }
        // Bi: list-0 half fixed, list-1 vector re-fit on the average.
        // Two pairings: the best list-1 reference, and — when that
        // is the very picture list 0 already predicts from — the
        // best list-1 reference that is a *different* picture (the
        // averaging gain comes from independent reconstructions).
        let pred_l0 = mc_pred(ctx, 0, lp0.ref_idx, x0, y0, w, h, lp0.mv)?;
        let poc0 = ctx.ref_pocs[0][lp0.ref_idx as usize];
        let mut pairings = vec![lp1];
        if ctx.ref_pocs[1][lp1.ref_idx as usize] == poc0 {
            let other = per_ref1
                .iter()
                .filter(|(p, _)| ctx.ref_pocs[1][p.ref_idx as usize] != poc0)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if let Some((p, _)) = other {
                pairings.push(*p);
            }
        }
        for lp1 in pairings {
            let mv1b = refine_bi_l1(ctx, x0, y0, w, h, &src_y, &pred_l0.0, lp1.ref_idx, lp1.mv)?;
            let cands1b = amvp_candidates(ctx, 1, lp1.ref_idx, x0, y0, w, h);
            let lp1b = fit_mvp(&cands1b, lp1.ref_idx, mv1b);
            let predb = mc_pred_pair(
                ctx,
                x0,
                y0,
                w,
                h,
                Some((lp0.ref_idx, lp0.mv)),
                Some((lp1b.ref_idx, lp1b.mv)),
            )?;
            let (resb, planesb, distb, _) = quantize_inter_residual(ctx, model, src, &predb, w, h)?;
            let leafb = PLeaf::Inter {
                l0: Some(lp0),
                l1: Some(lp1b),
                res: resb,
            };
            let costb = distb + ctx.lambda * leaf_bits(model, ctx, &leafb);
            if costb < explicit.cost {
                explicit = Explicit {
                    leaf: leafb,
                    pred: predb,
                    planes: planesb,
                    cost: costb,
                };
            }
        }
    }
    let Explicit {
        leaf: ex_leaf,
        pred: ex_pred,
        planes: ex_planes,
        cost: inter_cost,
    } = explicit;

    // ---- intra (single tree, chroma follows the luma mode). The luma
    // mode search costs each mode's own syntax (pred-mode bins, the
    // U-coded `intra_pred_mode`, `cbf_luma`, the RLE string); chroma is
    // added on the decided mode.
    let refs = ctx.recon.fetch_intra_refs(x0, y0, w, h, 0);
    let mut best_intra: Option<(usize, Vec<i32>, bool, Vec<i32>)> = None;
    let mut best_intra_luma_cost = f64::INFINITY;
    let qp_y = crate::dequant::qp_prime_y(ctx.qp, bd);
    let qp_c = crate::dequant::qp_prime_c(ctx.qp, 0, bd, false);
    for (mode_idx, &mode) in MODES.iter().enumerate() {
        let (levels, cbf, res, dist) = quantize_block(
            &refs,
            mode,
            &src_y,
            w,
            h,
            qp_y,
            bd,
            max_val,
            Some(&RdoqInputs::new(
                model,
                ctx.lambda,
                sel,
                0,
                MainCtxTable::CbfLuma,
            )),
        )?;
        let bits = model.measure(|m| {
            emit_intra_head(
                m,
                ctx,
                sel,
                &ctx.side_info,
                x0,
                y0,
                log2_w,
                log2_h,
                mode_idx,
            );
            let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
            m.encode_decision(t, i, u8::from(cbf));
            if cbf {
                emit_residual_rle(m, sel, 0, &levels, log2_w, log2_h);
            }
        });
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
    let (i_levels_cb, i_cbf_cb, i_res_cb, i_dist_cb) = quantize_block(
        &refs_cb,
        mode,
        &src_cb,
        wc,
        hc,
        qp_c,
        bd,
        max_val,
        Some(&RdoqInputs::new(
            model,
            crate::slice_enc::rd_lambda_at_qp_prime(qp_c),
            sel,
            1,
            MainCtxTable::CbfCb,
        )),
    )?;
    let refs_cr = ctx.recon.fetch_intra_refs(x0 / 2, y0 / 2, wc, hc, 2);
    let (i_levels_cr, i_cbf_cr, i_res_cr, i_dist_cr) = quantize_block(
        &refs_cr,
        mode,
        &src_cr,
        wc,
        hc,
        qp_c,
        bd,
        max_val,
        Some(&RdoqInputs::new(
            model,
            crate::slice_enc::rd_lambda_at_qp_prime(qp_c),
            sel,
            2,
            MainCtxTable::CbfCr,
        )),
    )?;
    let chroma_bits = model.measure(|m| {
        let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
        m.encode_decision(t, i, u8::from(i_cbf_cb));
        let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
        m.encode_decision(t, i, u8::from(i_cbf_cr));
        if i_cbf_cb {
            emit_residual_rle(m, sel, 1, &i_levels_cb, log2_w - 1, log2_h - 1);
        }
        if i_cbf_cr {
            emit_residual_rle(m, sel, 2, &i_levels_cr, log2_w - 1, log2_h - 1);
        }
    });
    let intra_cost = best_intra_luma_cost + i_dist_cb + i_dist_cr + ctx.lambda * chroma_bits;

    // ---- choose and commit (the statistics are tallied by the emit
    // pass over the *decided* tree — the decide pass trials leaves that
    // a split later discards).
    let direct_cost = direct.as_ref().map_or(f64::INFINITY, |d| d.3);
    let best = [best_skip_cost, direct_cost, inter_cost, intra_cost]
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .expect("4 costs");
    let (leaf, cost) = match best {
        0 => {
            let (leaf, pred) = best_skip.expect("skip ladder evaluated");
            let PLeaf::Skip { mv, .. } = &leaf else {
                unreachable!()
            };
            let l1 = if is_b { Some((0, mv[1])) } else { None };
            commit_inter(
                ctx,
                x0,
                y0,
                log2_w,
                log2_h,
                Some((0, mv[0])),
                l1,
                false,
                true,
                &pred,
                &(Vec::new(), Vec::new(), Vec::new()),
            );
            (leaf, best_skip_cost)
        }
        1 => {
            let (leaf, pred, planes, cost) = direct.expect("direct evaluated");
            let PLeaf::Direct { mv, res } = &leaf else {
                unreachable!()
            };
            commit_inter(
                ctx,
                x0,
                y0,
                log2_w,
                log2_h,
                Some((0, mv[0])),
                Some((0, mv[1])),
                res.cbf_y,
                false,
                &pred,
                &planes,
            );
            (leaf, cost)
        }
        2 => {
            let PLeaf::Inter { l0, l1, res } = &ex_leaf else {
                unreachable!()
            };
            commit_inter(
                ctx,
                x0,
                y0,
                log2_w,
                log2_h,
                l0.map(|p| (p.ref_idx, p.mv)),
                l1.map(|p| (p.ref_idx, p.mv)),
                res.cbf_y,
                false,
                &ex_pred,
                &ex_planes,
            );
            (ex_leaf, inter_cost)
        }
        _ => {
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
            for (c_idx, res) in [(0u32, &i_res_y), (1, &i_res_cb), (2, &i_res_cr)] {
                intra_reconstruct_cb_in_tile(
                    &mut ctx.recon,
                    x0,
                    y0,
                    log2_w,
                    log2_h,
                    mode,
                    c_idx,
                    res,
                    None,
                )?;
            }
            (
                PLeaf::Intra {
                    mode_idx: i_mode,
                    res: Residual {
                        levels_y: i_levels_y,
                        cbf_y: i_cbf_y,
                        levels_cb: i_levels_cb,
                        cbf_cb: i_cbf_cb,
                        levels_cr: i_levels_cr,
                        cbf_cr: i_cbf_cr,
                    },
                },
                intra_cost,
            )
        }
    };
    // Advance the rate model over the decided leaf's exact bin string
    // (the §9.3.4.2.4 probes read the L/A/R neighbours, which this
    // CU's own stamp does not touch — the emit pass sees the same).
    model.commit(|m| emit_leaf_bins(m, ctx, sel, &ctx.side_info, x0, y0, log2_w, log2_h, &leaf));
    Ok((leaf, cost))
}

// ---------------------------------------------------------------------
// Emit pass — decoder-exact bin order.
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_split_unit<S: BinSink>(
    enc: &mut S,
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
            emit_leaf(enc, stats, ctx, sel, grid, x0, y0, log2_w, log2_h, plan);
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

/// TR cMax 3 write for `mvp_idx_lX` — the dual of the decoder's
/// `decode_tr_regular(3, 0, …)` with the Table 48 per-bin ctxInc.
fn emit_mvp_idx<S: BinSink>(enc: &mut S, sel: CtxSel, v: u32) {
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

/// TR cMax 2 write for `inter_pred_idc` — the dual of the decoder's
/// `decode_tr_regular(2, 0, …)` over Table 69 (per-bin ctxInc 0, 1).
fn emit_inter_pred_idc<S: BinSink>(enc: &mut S, sel: CtxSel, v: u32) {
    let table = MainCtxTable::InterPredIdc;
    let (t, off) = if sel.cm_init {
        (table.as_usize(), table.ctx_idx_offset(sel.init_type))
    } else {
        (0, table.cm0_ctx_idx_offset(sel.init_type))
    };
    let idx = |b: u32| -> usize { off + (b as usize).min(1) };
    for b in 0..v {
        enc.encode_decision(t, idx(b), 1);
    }
    if v < 2 {
        enc.encode_decision(t, idx(v), 0);
    }
}

/// TR (`cMax = num_ref_idx_active_minus1`) write for `ref_idx_lX` —
/// the dual of the decoder's `decode_ref_idx_tr`: under
/// `sps_cm_init_flag == 1` bins 0/1 regular on Table 72 (ctxInc 0, 1),
/// later bins bypass; under the Baseline collapse every bin is the
/// `(0, 0)` regular context. Writes nothing when the list holds one
/// picture (the element is absent).
fn emit_ref_idx<S: BinSink>(enc: &mut S, sel: CtxSel, v: u32, n_refs: u32) {
    if n_refs <= 1 {
        return;
    }
    let c_max = n_refs - 1;
    let n_bins = v + u32::from(v < c_max);
    for b in 0..n_bins {
        let bin = u8::from(b < v);
        if !sel.cm_init {
            enc.encode_decision(0, 0, bin);
        } else if b < 2 {
            let table = MainCtxTable::RefIdx;
            let off = table.ctx_idx_offset(InitType::Pb);
            enc.encode_decision(table.as_usize(), off + b as usize, bin);
        } else {
            enc.encode_bypass(bin);
        }
    }
}

/// Signed `abs_mvd` + `mvd_sign_flag` write — the dual of the decoder's
/// `decode_signed_mvd` (EG0 bin0 regular on Table 73 under cm_init,
/// all-bypass otherwise; sign bypass when non-zero).
fn emit_signed_mvd<S: BinSink>(enc: &mut S, sel: CtxSel, v: i32) {
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

/// One list's explicit motion syntax: `ref_idx_lX` (when present),
/// `mvp_idx_lX`, `abs_mvd_lX` + signs.
fn emit_list_pred<S: BinSink>(enc: &mut S, sel: CtxSel, lp: &ListPred, n_refs: u32) {
    emit_ref_idx(enc, sel, lp.ref_idx, n_refs);
    emit_mvp_idx(enc, sel, lp.mvp_idx);
    emit_signed_mvd(enc, sel, lp.mvd.x);
    emit_signed_mvd(enc, sel, lp.mvd.y);
}

/// The inter `transform_unit()` presence + body: `cbf_all`, then
/// `cbf_cb` / `cbf_cr` / (presence-gated) `cbf_luma` and the residuals.
fn emit_inter_residual<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    res: &Residual,
    log2_w: u32,
    log2_h: u32,
) {
    let any = res.any();
    let (t, i) = sel.ctx(MainCtxTable::CbfAll, 0);
    enc.encode_decision(t, i, u8::from(any)); // cbf_all
    if any {
        let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
        enc.encode_decision(t, i, u8::from(res.cbf_cb));
        let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
        enc.encode_decision(t, i, u8::from(res.cbf_cr));
        if res.cbf_cb || res.cbf_cr {
            let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
            enc.encode_decision(t, i, u8::from(res.cbf_y));
        } else {
            debug_assert!(res.cbf_y, "quiet chroma infers cbf_luma = 1 (§7.4.9.5)");
        }
        if res.cbf_y {
            emit_residual_rle(enc, sel, 0, &res.levels_y, log2_w, log2_h);
        }
        if res.cbf_cb {
            emit_residual_rle(enc, sel, 1, &res.levels_cb, log2_w - 1, log2_h - 1);
        }
        if res.cbf_cr {
            emit_residual_rle(enc, sel, 2, &res.levels_cr, log2_w - 1, log2_h - 1);
        }
    }
}

/// The `cu_skip_flag = 0`, `pred_mode_flag = 1` head of an intra CU
/// on a P/B slice (both §9.3.4.2.4 neighbour-context flags) plus its
/// `intra_pred_mode`.
#[allow(clippy::too_many_arguments)]
fn emit_intra_head<S: BinSink>(
    enc: &mut S,
    ctx: &PCtx<'_>,
    sel: CtxSel,
    grid: &SideInfoGrid,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    mode_idx: usize,
) {
    let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
    enc.encode_decision(t, i, 0); // cu_skip_flag = 0
    let (t, i) = pred_mode_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
    enc.encode_decision(t, i, 1); // pred_mode_flag = 1 (MODE_INTRA)
    emit_intra_pred_mode(enc, sel, mode_idx);
}

/// The complete `coding_unit()` bin string of one leaf, probing `grid`
/// for the neighbour-derived contexts — no side effects beyond the
/// sink. Shared by the emit pass (real coder + stamp) and the decide
/// pass (rate model).
#[allow(clippy::too_many_arguments)]
fn emit_leaf_bins<S: BinSink>(
    enc: &mut S,
    ctx: &PCtx<'_>,
    sel: CtxSel,
    grid: &SideInfoGrid,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    plan: &PLeaf,
) {
    let is_b = ctx.slice_is_b;
    match plan {
        PLeaf::Skip { mvp_idx, .. } => {
            let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 1); // cu_skip_flag = 1
            emit_mvp_idx(enc, sel, mvp_idx[0]);
            if is_b {
                emit_mvp_idx(enc, sel, mvp_idx[1]);
            }
            // No residual syntax (§7.3.8.4).
        }
        PLeaf::Direct { res, .. } => {
            let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // cu_skip_flag = 0
            let (t, i) = pred_mode_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // pred_mode_flag = 0 (MODE_INTER)
            let (t, i) = sel.ctx(MainCtxTable::DirectModeFlag, 0);
            enc.encode_decision(t, i, 1); // direct_mode_flag = 1
            emit_inter_residual(enc, sel, res, log2_w, log2_h);
        }
        PLeaf::Inter { l0, l1, res } => {
            let (t, i) = skip_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // cu_skip_flag = 0
            let (t, i) = pred_mode_flag_ctx(ctx, sel, grid, x0, y0, log2_w, log2_h);
            enc.encode_decision(t, i, 0); // pred_mode_flag = 0 (MODE_INTER)
            if is_b {
                let (t, i) = sel.ctx(MainCtxTable::DirectModeFlag, 0);
                enc.encode_decision(t, i, 0); // direct_mode_flag = 0
                let idc = match (l0.is_some(), l1.is_some()) {
                    (true, false) => 0,
                    (false, true) => 1,
                    _ => 2,
                };
                emit_inter_pred_idc(enc, sel, idc);
            }
            if let Some(lp) = l0 {
                emit_list_pred(enc, sel, lp, ctx.n_refs(0));
            }
            if let Some(lp) = l1 {
                emit_list_pred(enc, sel, lp, ctx.n_refs(1));
            }
            emit_inter_residual(enc, sel, res, log2_w, log2_h);
        }
        PLeaf::Intra { mode_idx, res } => {
            emit_intra_head(enc, ctx, sel, grid, x0, y0, log2_w, log2_h, *mode_idx);
            // Single-tree intra TU: cbf_cb, cbf_cr, then cbf_luma
            // (always present — MODE_INTRA), then luma/cb/cr residuals.
            let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
            enc.encode_decision(t, i, u8::from(res.cbf_cb));
            let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
            enc.encode_decision(t, i, u8::from(res.cbf_cr));
            let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
            enc.encode_decision(t, i, u8::from(res.cbf_y));
            if res.cbf_y {
                emit_residual_rle(enc, sel, 0, &res.levels_y, log2_w, log2_h);
            }
            if res.cbf_cb {
                emit_residual_rle(enc, sel, 1, &res.levels_cb, log2_w - 1, log2_h - 1);
            }
            if res.cbf_cr {
                emit_residual_rle(enc, sel, 2, &res.levels_cr, log2_w - 1, log2_h - 1);
            }
        }
    }
}

/// Emit one decided leaf: its bins into the real coder, its statistics,
/// and its side-info stamp into the emit-order grid (after the bins —
/// the §9.3.4.2.4 probes read the neighbours, which the stamp does not
/// touch, but the skip-cell marks must land before the next CU).
#[allow(clippy::too_many_arguments)]
fn emit_leaf<S: BinSink>(
    enc: &mut S,
    stats: &mut PEncStats,
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
    let is_b = ctx.slice_is_b;
    emit_leaf_bins(enc, ctx, sel, grid, x0, y0, log2_w, log2_h, plan);
    let stamp_inter = |grid: &mut SideInfoGrid, l0: ListMotion, l1: ListMotion, cbf_y: bool| {
        grid.stamp_block(
            x0,
            y0,
            w,
            h,
            inter_side_info(x0, y0, log2_w, log2_h, l0, l1, cbf_y, ctx.qp),
        );
    };
    stats.leaves += 1;
    match plan {
        PLeaf::Skip { mv, .. } => {
            stats.skip_cus += 1;
            let l1 = if is_b { Some((0, mv[1])) } else { None };
            stamp_inter(grid, Some((0, mv[0])), l1, false);
            mark_cu_skip_cells(grid, x0, y0, w, h);
        }
        PLeaf::Direct { mv, res } => {
            stats.inter_cus += 1;
            stats.direct_cus += 1;
            if !res.any() {
                stats.cbf_all_zero_cus += 1;
            }
            stamp_inter(grid, Some((0, mv[0])), Some((0, mv[1])), res.cbf_y);
        }
        PLeaf::Inter { l0, l1, res } => {
            stats.inter_cus += 1;
            if !res.any() {
                stats.cbf_all_zero_cus += 1;
            }
            match (l0, l1) {
                (Some(_), Some(_)) => stats.bi_cus += 1,
                (None, Some(_)) => stats.l1_only_cus += 1,
                _ => {}
            }
            if l0.is_some_and(|p| p.ref_idx > 0) || l1.is_some_and(|p| p.ref_idx > 0) {
                stats.multi_ref_cus += 1;
            }
            stamp_inter(
                grid,
                l0.map(|p| (p.ref_idx, p.mv)),
                l1.map(|p| (p.ref_idx, p.mv)),
                res.cbf_y,
            );
        }
        PLeaf::Intra { mode_idx, res } => {
            stats.intra_cus += 1;
            grid.stamp_block(
                x0,
                y0,
                w,
                h,
                CuSideInfo {
                    pred_mode: CuPredMode::Intra,
                    cbf_luma: u8::from(res.cbf_y),
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
        view_of(p)
    }

    /// Deterministic pseudo-natural frame `t` of a moving-scene GOP:
    /// a diagonal gradient plus a bright square translating (3, 2)
    /// pixels per frame and a noise band, so P frames exercise real
    /// motion (skip / non-zero-MV inter / intra refresh all appear).
    pub(crate) fn synth_moving(w: u32, h: u32, t: u32, bit_depth: u32) -> YuvPicture {
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

    #[allow(clippy::too_many_arguments)]
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

    /// Decode a payload against explicit reference lists (the decoder's
    /// own §7.3.8 walker with the lists a DPB would resolve).
    #[allow(clippy::too_many_arguments)]
    fn decode_lists(
        payload: &[u8],
        refs_l0: &[RefEntry<'_>],
        refs_l1: &[RefEntry<'_>],
        slice_is_b: bool,
        curr_poc: i32,
        col: Option<ColMotion<'_>>,
        w: u32,
        h: u32,
        qp: i32,
        deblock: bool,
        cm_init: bool,
    ) -> (YuvPicture, crate::slice_data::InterDecodeStats) {
        let v0: Vec<RefPictureView<'_>> = refs_l0.iter().map(|r| ref_view(r.pic)).collect();
        let v1: Vec<RefPictureView<'_>> = refs_l1.iter().map(|r| ref_view(r.pic)).collect();
        let p0: Vec<i32> = refs_l0.iter().map(|r| r.poc).collect();
        let p1: Vec<i32> = refs_l1.iter().map(|r| r.poc).collect();
        let bd = refs_l0[0].pic.bit_depth;
        let inputs = InterDecodeInputs {
            walk: walk_inputs(w, h, cm_init),
            decode: SliceDecodeInputs {
                slice_qp: qp,
                bit_depth_luma: bd,
                bit_depth_chroma: bd,
                enable_deblock: deblock,
                ..Default::default()
            },
            slice_is_b,
            num_ref_idx_active_minus1_l0: refs_l0.len() as u32 - 1,
            num_ref_idx_active_minus1_l1: if slice_is_b {
                refs_l1.len() as u32 - 1
            } else {
                0
            },
            ref_list_l0: &v0,
            ref_list_l1: &v1,
            inter_tool_gates: Default::default(),
            pocs: InterPocs {
                curr_poc,
                ref_pocs_l0: &p0,
                ref_pocs_l1: &p1,
            },
            col_pic: col.map(|c| ColPicInputs {
                grid: c.grid,
                col_poc: c.poc,
                ref_pocs_l0: c.ref_pocs_l0,
                ref_pocs_l1: &[],
            }),
        };
        decode_baseline_inter_slice(payload, inputs).expect("inter slice must decode")
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
        // Skip carries the frame; the few CUs the RD refines (the
        // IDR's own quantization error re-coded where that beats the
        // skip distortion at λ) stay a small minority.
        assert!(
            stats.skip_cus * 4 >= stats.leaves * 3,
            "skip must carry a static P picture: {stats:?}"
        );
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
        // An empty list, or a B slice without list 1, is refused.
        let bad = encode_inter_slice_data(
            &src,
            InterEncInputs {
                refs_l0: &[],
                refs_l1: &[],
                slice_is_b: false,
                curr_poc: 1,
                col: None,
                slice_qp: 30,
                deblock: false,
                cm_init: true,
            },
        );
        assert!(bad.is_err());
        let refs = [RefEntry { pic: &refp, poc: 0 }];
        let bad_b = encode_inter_slice_data(
            &src,
            InterEncInputs {
                refs_l0: &refs,
                refs_l1: &[],
                slice_is_b: true,
                curr_poc: 1,
                col: None,
                slice_qp: 30,
                deblock: false,
                cm_init: true,
            },
        );
        assert!(bad_b.is_err());
    }

    /// Round 452 — **multi-reference P**: a two-picture `RefPicList0`
    /// (POC 1 then POC 0 — the §8.3.2.2 descending order). The ladder
    /// must address `ref_idx_l0 = 1` where the older picture predicts
    /// better, the `ref_idx` TR bins must replay through the decoder,
    /// and the loop stays recon-exact on both entropy shapes.
    #[test]
    fn multi_ref_p_uses_older_reference_and_round_trips() {
        let (w, h) = (64u32, 64u32);
        let qp = 28;
        // Frame 2 is a copy of frame 0 (the older reference), so the
        // best predictor of most CUs is ref_idx 1.
        let f0 = synth_moving(w, h, 0, 8);
        let f1 = synth_moving(w, h, 5, 8);
        let f2 = synth_moving(w, h, 0, 8);
        for &cm_init in &[false, true] {
            let (_ip, r0, _s) =
                crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, cm_init).unwrap();
            let (_pp, r1, _s) = encode_p_slice_data(&f1, &r0, qp, false, cm_init).unwrap();
            let refs = [RefEntry { pic: &r1, poc: 1 }, RefEntry { pic: &r0, poc: 0 }];
            let out = encode_inter_slice_data(
                &f2,
                InterEncInputs {
                    refs_l0: &refs,
                    refs_l1: &[],
                    slice_is_b: false,
                    curr_poc: 2,
                    col: None,
                    slice_qp: qp,
                    deblock: false,
                    cm_init,
                },
            )
            .unwrap();
            assert!(
                out.stats.multi_ref_cus > 0,
                "cm{cm_init}: the older reference must be addressed: {:?}",
                out.stats
            );
            let (dec, dec_stats) = decode_lists(
                &out.payload,
                &refs,
                &[],
                false,
                2,
                None,
                w,
                h,
                qp,
                false,
                cm_init,
            );
            assert!(
                dec_stats.ref_idx_bins > 0,
                "ref_idx syntax reached the decoder"
            );
            assert_eq!(dec.y, out.recon.y, "cm{cm_init}: luma");
            assert_eq!(dec.cb, out.recon.cb, "cm{cm_init}: cb");
            assert_eq!(dec.cr, out.recon.cr, "cm{cm_init}: cr");
            // The two-reference encode must not lose to the
            // single-reference one on this content.
            let (single, _, _) = encode_p_slice_data(&f2, &r1, qp, false, cm_init).unwrap();
            assert!(
                out.payload.len() < single.len(),
                "cm{cm_init}: two refs {} vs one ref {}",
                out.payload.len(),
                single.len()
            );
        }
    }

    /// Round 452 — **low-delay B**: both lists resolve to the same
    /// descending past (§8.3.2.2 with nothing above the current POC),
    /// the collocated field is the previous P/B picture's grid. Every
    /// B tool must appear (over a low and a high QP — at QP 14 the
    /// per-frame noise stays in the reconstructions and the bi/direct
    /// averaging wins leaves; at QP 51 it quantizes away and the skip
    /// ladder carries the picture) and the loop stays recon-exact on
    /// both entropy shapes and both deblock settings.
    #[test]
    fn low_delay_b_round_trips_with_every_tool() {
        let (w, h) = (96u32, 64u32);
        // Independent per-frame noise on top of the moving scene: the
        // eq. 988 average of two past pictures halves the noise
        // variance, which is what makes explicit bi-prediction win
        // leaves against uni-prediction on a low-delay B picture.
        let synth_noisy = |t: u32| -> YuvPicture {
            let mut pic = synth_moving(w, h, t, 8);
            let mut s = 0x2545_F491u32 ^ t.wrapping_mul(0x9E37_79B9);
            for v in pic.y.iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((s >> 24) % 13) as i32 - 6;
                *v = (*v as i32 + n).clamp(0, 255) as u16;
            }
            pic
        };
        for &cm_init in &[false, true] {
            for &deblock in &[false, true] {
                let mut totals = PEncStats::default();
                for &qp in &[14i32, 51] {
                    let f0 = synth_noisy(0);
                    let (_ip, r0, _s) =
                        crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, deblock, cm_init)
                            .unwrap();
                    let f1 = synth_noisy(1);
                    let (_pp, r1, _s) =
                        encode_p_slice_data(&f1, &r0, qp, deblock, cm_init).unwrap();
                    // The P picture's motion field must come from the same
                    // encoder path — re-run through the general entry to
                    // capture the grid (identical payload).
                    let refs1 = [RefEntry { pic: &r0, poc: 0 }];
                    let p_out = encode_inter_slice_data(
                        &f1,
                        InterEncInputs {
                            refs_l0: &refs1,
                            refs_l1: &[],
                            slice_is_b: false,
                            curr_poc: 1,
                            col: None,
                            slice_qp: qp,
                            deblock,
                            cm_init,
                        },
                    )
                    .unwrap();
                    assert_eq!(p_out.recon.y, r1.y);
                    let mut dpb: Vec<(YuvPicture, SideInfoGrid, i32, Vec<i32>)> = vec![
                        (r0.clone(), SideInfoGrid::new(w, h), 0, vec![]),
                        (r1.clone(), p_out.side_info, 1, vec![0]),
                    ];
                    for t in 2..=4u32 {
                        let src = synth_noisy(t);
                        let curr_poc = t as i32;
                        // §8.3.2.2 low-delay: L0 = L1 = descending POCs.
                        let mut order: Vec<usize> = (0..dpb.len()).collect();
                        order.sort_by_key(|&i| -dpb[i].2);
                        let refs: Vec<RefEntry<'_>> = order
                            .iter()
                            .take(2)
                            .map(|&i| RefEntry {
                                pic: &dpb[i].0,
                                poc: dpb[i].2,
                            })
                            .collect();
                        let col_i = order[0];
                        let col = ColMotion {
                            grid: &dpb[col_i].1,
                            poc: dpb[col_i].2,
                            ref_pocs_l0: &dpb[col_i].3,
                        };
                        let out = encode_inter_slice_data(
                            &src,
                            InterEncInputs {
                                refs_l0: &refs,
                                refs_l1: &refs,
                                slice_is_b: true,
                                curr_poc,
                                col: Some(col),
                                slice_qp: qp,
                                deblock,
                                cm_init,
                            },
                        )
                        .unwrap();
                        let (dec, dec_stats) = decode_lists(
                            &out.payload,
                            &refs,
                            &refs,
                            true,
                            curr_poc,
                            Some(col),
                            w,
                            h,
                            qp,
                            deblock,
                            cm_init,
                        );
                        assert_eq!(dec.y, out.recon.y, "t{t} cm{cm_init} db{deblock}: luma");
                        assert_eq!(dec.cb, out.recon.cb, "t{t} cm{cm_init} db{deblock}: cb");
                        assert_eq!(dec.cr, out.recon.cr, "t{t} cm{cm_init} db{deblock}: cr");
                        assert_eq!(dec_stats.direct_cus, out.stats.direct_cus);
                        assert_eq!(
                            dec_stats.bi_pred_cus > 0,
                            out.stats.bi_cus + out.stats.direct_cus + out.stats.skip_cus > 0
                        );
                        let s = out.stats;
                        totals.skip_cus += s.skip_cus;
                        totals.direct_cus += s.direct_cus;
                        totals.bi_cus += s.bi_cus;
                        totals.l1_only_cus += s.l1_only_cus;
                        totals.multi_ref_cus += s.multi_ref_cus;
                        totals.inter_cus += s.inter_cus;
                        let ref_pocs: Vec<i32> = refs.iter().map(|r| r.poc).collect();
                        dpb.push((out.recon, out.side_info, curr_poc, ref_pocs));
                    }
                }
                assert!(
                    totals.skip_cus > 0,
                    "cm{cm_init} db{deblock}: B skip: {totals:?}"
                );
                assert!(
                    totals.direct_cus > 0,
                    "cm{cm_init} db{deblock}: direct: {totals:?}"
                );
                assert!(
                    totals.bi_cus > 0,
                    "cm{cm_init} db{deblock}: explicit bi: {totals:?}"
                );
            }
        }
    }

    /// Round 452 — the B ladder's syntax duals are exercised one by one
    /// through the decoder's bin tallies: `inter_pred_idc` (PRED_L1 and
    /// PRED_BI), `direct_mode_flag`, the second skip `mvp_idx`, and
    /// `ref_idx` on both lists.
    #[test]
    fn b_syntax_duals_reach_the_decoder() {
        let (w, h) = (64u32, 48u32);
        let qp = 34;
        let f0 = synth_moving(w, h, 0, 8);
        let (_ip, r0, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, qp, false, true).unwrap();
        let f1 = synth_moving(w, h, 2, 8);
        let refs1 = [RefEntry { pic: &r0, poc: 0 }];
        let p1 = encode_inter_slice_data(
            &f1,
            InterEncInputs {
                refs_l0: &refs1,
                refs_l1: &[],
                slice_is_b: false,
                curr_poc: 1,
                col: None,
                slice_qp: qp,
                deblock: false,
                cm_init: true,
            },
        )
        .unwrap();
        // Frame 2 shares content with both references.
        let f2 = synth_moving(w, h, 1, 8);
        let refs = [
            RefEntry {
                pic: &p1.recon,
                poc: 1,
            },
            RefEntry { pic: &r0, poc: 0 },
        ];
        let col = ColMotion {
            grid: &p1.side_info,
            poc: 1,
            ref_pocs_l0: &[0],
        };
        let out = encode_inter_slice_data(
            &f2,
            InterEncInputs {
                refs_l0: &refs,
                refs_l1: &refs,
                slice_is_b: true,
                curr_poc: 2,
                col: Some(col),
                slice_qp: qp,
                deblock: false,
                cm_init: true,
            },
        )
        .unwrap();
        let (dec, st) = decode_lists(
            &out.payload,
            &refs,
            &refs,
            true,
            2,
            Some(col),
            w,
            h,
            qp,
            false,
            true,
        );
        assert_eq!(dec.y, out.recon.y);
        assert_eq!(dec.cb, out.recon.cb);
        assert_eq!(dec.cr, out.recon.cr);
        assert_eq!(
            st.direct_mode_flag_bins, out.stats.inter_cus,
            "one direct flag per non-skip inter CU"
        );
        assert_eq!(st.direct_cus, out.stats.direct_cus);
        assert_eq!(
            st.inter_pred_idc_bins,
            out.stats.inter_cus - out.stats.direct_cus,
            "inter_pred_idc on every explicit CU"
        );
        assert_eq!(
            st.bi_pred_cus,
            out.stats.bi_cus + out.stats.direct_cus + out.stats.skip_cus
        );
        assert!(st.ref_idx_bins > 0);
        assert!(
            st.mvp_idx_bins >= 2 * out.stats.skip_cus,
            "two skip mvp indices"
        );
    }

    /// Determinism of the B pipeline.
    #[test]
    fn b_encode_is_deterministic() {
        let (w, h) = (48u32, 48u32);
        let f0 = synth_moving(w, h, 0, 8);
        let (_ip, r0, _s) =
            crate::slice_enc::encode_idr_slice_data_opts(&f0, 33, false, true).unwrap();
        let f1 = synth_moving(w, h, 3, 8);
        let refs = [RefEntry { pic: &r0, poc: 0 }];
        let run = || {
            encode_inter_slice_data(
                &f1,
                InterEncInputs {
                    refs_l0: &refs,
                    refs_l1: &refs,
                    slice_is_b: true,
                    curr_poc: 1,
                    col: None,
                    slice_qp: 33,
                    deblock: false,
                    cm_init: true,
                },
            )
            .unwrap()
            .payload
        };
        assert_eq!(run(), run());
    }
}
