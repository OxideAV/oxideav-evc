//! **Intra slice encoder** (round 429 bootstrap, round 431 context
//! modelling): the write-side dual of the §7.3.8 IDR `slice_data()`
//! walker in [`crate::slice_data`].
//!
//! The emitted bin stream mirrors the decoder's read order bin for bin.
//! Two entropy shapes, selected by `sps_cm_init_flag`:
//!
//! * `cm_init == false` — the Baseline collapse: every regular bin
//!   shares the single `(ctxTable 0, ctxIdx 0)` context;
//! * `cm_init == true` — the §9.3.2.2 context-model initialization
//!   (Tables 40-90 at the slice QP, initType 0) with the §9.3.4.2.1
//!   `ctxIdx = ctxIdxOffset + ctxInc` selection per syntax element,
//!   exactly the decoder's [`crate::cabac_init::CtxSel`] routing:
//!   `split_cu_flag` (Table 41, ctxInc 0), `intra_pred_mode` (Table 62,
//!   bin0 → 0 / later bins → 1), `cbf_luma`/`cbf_cb`/`cbf_cr`
//!   (Tables 75/76/77, ctxInc 0), and the §7.3.8.7 RLE residual
//!   (Tables 84/85 with the §9.3.4.2.2 eq. 1434/1435 `PrevLevel`-chain
//!   ctxInc, Table 86 `coeff_last_flag` at cIdx-keyed ctxInc).
//!
//! In both shapes encoder and decoder context state evolve identically,
//! so the emit is byte-exact under re-decode:
//!
//! * per CTU — `split_unit()` quad recursion (§7.3.8.3, `sps_btt_flag
//!   == 0` shape): `split_cu_flag` on recursable in-picture blocks,
//!   implicit splits at picture edges, each leaf one SINGLE_TREE
//!   `coding_unit()` (§7.3.8.3 lines 2788-2795 — the in-leaf INTRA_IBC
//!   reassignment binds only the CU-internal presence gates, so no
//!   per-leaf chroma partner CU exists);
//! * per CU — `intra_pred_mode` (U over the Table-13 5-mode set), then
//!   the §7.3.8.5 `transform_unit()`: `cbf_cb`, `cbf_cr`, `cbf_luma`,
//!   and per-component `residual_coding_rle()` (§7.3.8.7) in luma / Cb /
//!   Cr order (chroma at the 4:2:0 sub-sampled TB dimensions; chroma
//!   prediction is the §8.4.3 DM — `IntraPredModeC = IntraPredModeY`
//!   via the inferred-0 `intra_chroma_pred_mode`);
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

use crate::ats::{self, AtsIntra};
use crate::bin_cost::BitCostModel;
use crate::cabac::{BinSink, CabacEncoder, InitType};
use crate::cabac_init::{ctx_inc_coeff_zero_run, CtxSel, MainCtxTable};
use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
use crate::dequant::scale_and_inverse_transform_ats;
use crate::eipd_mode::{derive_chroma_mode, ModeSelector};
use crate::eipd_syntax::EipdCtx;
use crate::intra::{predict, IntraMode, RefSamples};
use crate::intra_enc::{self, IntraSel};
use crate::picture::{intra_reconstruct_cb_eipd_in_tile, intra_reconstruct_cb_in_tile, YuvPicture};
use crate::quant_enc::{
    forward_quantize_typed, forward_transform_fractional_typed, level_unit_sse_weights_typed,
    TransformSpec,
};
use crate::rdoq::{rdoq_rle, RdoqInputs};
use crate::slice_data::zigzag_scan;
use crate::tree_enc::{self, TreeCoder, TreeGeometry, TreeNode, TreeStats};

/// The five Table-13 Baseline intra modes in syntax-index order.
pub(crate) const MODES: [IntraMode; 5] = [
    IntraMode::Dc,
    IntraMode::Hor,
    IntraMode::Ver,
    IntraMode::Ul,
    IntraMode::Ur,
];

/// Geometry constants of the all-zero-toolset SPS the header writer
/// emits (§7.4.3.1 sps_btt_flag == 0 defaults + eq. 51).
const CTB_LOG2: u32 = 6;

/// Per-picture encode statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncStats {
    /// CTUs walked.
    pub ctus: u32,
    /// Leaf coding units (one SINGLE_TREE `coding_unit()` each).
    pub leaves: u32,
    /// Leaves per chosen luma intra mode (Table 13 order DC/HOR/VER/UL/UR)
    /// under `sps_eipd_flag == 0`.
    pub mode_histogram: [u32; 5],
    /// Leaves per chosen `IntraPredModeY` (Table 15, 0..=32) under
    /// `sps_eipd_flag == 1`.
    pub eipd_mode_histogram: [u32; 33],
    /// EIPD leaves whose `intra_chroma_pred_mode != 0` (a non-DM chroma
    /// mode won the chroma RD).
    pub eipd_chroma_non_dm: u32,
    /// Leaves whose `cbf_luma` was signalled 1.
    pub cbf_luma_set: u32,
    /// Chroma CBFs signalled 1 (cb + cr).
    pub cbf_chroma_set: u32,
    /// `split_cu_flag` bins emitted (`sps_btt_flag == 0`).
    pub split_flag_bins: u32,
    /// The `sps_btt_flag == 1` tree syntax (round 458).
    pub tree: TreeStats,
}

impl Default for EncStats {
    fn default() -> Self {
        Self {
            ctus: 0,
            leaves: 0,
            mode_histogram: [0; 5],
            eipd_mode_histogram: [0; 33],
            eipd_chroma_non_dm: 0,
            cbf_luma_set: 0,
            cbf_chroma_set: 0,
            split_flag_bins: 0,
            tree: TreeStats::default(),
        }
    }
}

/// One decided leaf: everything the emit pass needs to reproduce the
/// exact bin stream whose decode lands on the already-committed recon.
struct LeafPlan {
    intra: IntraSel,
    /// The §7.3.8.5 ATS-intra decision of the luma TB (round 458;
    /// `disabled` = plain DCT-II, and nothing signalled unless
    /// `sps_ats_flag` gates it in).
    ats: AtsIntra,
    levels_y: Vec<i32>,
    cbf_y: bool,
    levels_cb: Vec<i32>,
    cbf_cb: bool,
    levels_cr: Vec<i32>,
    cbf_cr: bool,
}

/// A decided `split_unit()` subtree.
type Node = TreeNode<LeafPlan>;

struct EncCtx<'a> {
    src: &'a YuvPicture,
    recon: YuvPicture,
    qp: i32,
    lambda: f64,
    bit_depth: u32,
    pic_w: u32,
    pic_h: u32,
    /// The entropy shape the emit pass will run under — the decide
    /// pass costs every candidate against the same contexts.
    sel: CtxSel,
    /// `sps_eipd_flag`: the 33-mode intra search + MPM syntax.
    eipd: bool,
    /// Decode-order side-info grid — the §8.4.2 neighbour modes the
    /// EIPD lists derive from (and the §8.8.2 deblocking inputs).
    side_info: SideInfoGrid,
    /// The coding-tree shape (`sps_btt_flag`) and its size limits.
    geom: TreeGeometry,
    /// `sps_iqt_flag` — the §8.7 improved quantization / transform chain.
    iqt: bool,
    /// `sps_ats_flag` — the ATS-intra kernel search on every luma TB.
    ats: bool,
    /// `slice_cb_qp_offset == slice_cr_qp_offset` the slice header
    /// carries ([`crate::headers_enc::iqt_chroma_qp_offset`]).
    chroma_qp_offset: i32,
}

/// The SPS tool set an IDR picture is coded under (round 458) — the
/// switches of [`encode_idr_slice_data_cfg`]; the SPS the caller
/// writes must declare the same flags ([`crate::headers_enc`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntraToolset {
    /// `slice_deblocking_filter_flag`.
    pub deblock: bool,
    /// `sps_cm_init_flag`.
    pub cm_init: bool,
    /// `sps_eipd_flag`.
    pub eipd: bool,
    /// `sps_btt_flag`.
    pub btt: bool,
    /// `sps_iqt_flag` (required by `ats`).
    pub iqt: bool,
    /// `sps_ats_flag`.
    pub ats: bool,
    /// `sps_adcc_flag` (requires `cm_init`) — the §7.3.8.8 advanced
    /// residual coding instead of the §7.3.8.7 run-length coding.
    pub adcc: bool,
}

/// The decode-order state a tree trial rewinds: the block's recon,
/// the side-info grid and the rate model.
struct IntraSnap {
    pixels: RegionSave,
    grid: SideInfoGrid,
    model: BitCostModel,
}

impl TreeCoder for EncCtx<'_> {
    type Leaf = LeafPlan;
    type Snap = IntraSnap;

    fn geometry(&self) -> &TreeGeometry {
        &self.geom
    }
    fn sel(&self) -> CtxSel {
        self.sel
    }
    fn lambda(&self) -> f64 {
        self.lambda
    }
    fn grid(&self) -> &SideInfoGrid {
        &self.side_info
    }
    fn snapshot(&self, model: &BitCostModel, x0: u32, y0: u32, lw: u32, lh: u32) -> IntraSnap {
        IntraSnap {
            pixels: save_region(&self.recon, x0, y0, lw, lh),
            grid: self.side_info.clone(),
            model: model.clone(),
        }
    }
    fn restore(
        &mut self,
        model: &mut BitCostModel,
        snap: &IntraSnap,
        x0: u32,
        y0: u32,
        lw: u32,
        lh: u32,
    ) {
        restore_region(&mut self.recon, &snap.pixels, x0, y0, lw, lh);
        self.side_info = snap.grid.clone();
        *model = snap.model.clone();
    }
    fn decide_leaf(
        &mut self,
        model: &mut BitCostModel,
        x0: u32,
        y0: u32,
        lw: u32,
        lh: u32,
    ) -> Result<(LeafPlan, f64)> {
        decide_leaf(self, model, x0, y0, lw, lh)
    }
}

/// [`encode_idr_slice_data_with`] with deblocking off — the historical
/// entry point (Baseline `sps_cm_init_flag == 0` entropy shape).
pub fn encode_idr_slice_data(
    src: &YuvPicture,
    slice_qp: i32,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_slice_data_opts(src, slice_qp, false, false)
}

/// Encode one Baseline IDR picture's `slice_data()` payload. Returns
/// the CABAC payload bytes (byte-aligned, ready to append after the
/// byte-aligned slice header), the reconstruction the decoder must
/// reproduce exactly, and the encode statistics.
///
/// With `deblock` set the caller signals
/// `slice_deblocking_filter_flag = 1` in the slice header and the
/// returned recon is the §8.8.2-filtered picture: the encoder stamps
/// the same per-CU side info the decoder builds (pred mode, cbf,
/// geometry, QpY) and runs the decoder's own `deblock_luma` /
/// `deblock_chroma` post-pass, so the output stays byte-exact.
/// (Prediction during the encode uses un-deblocked samples, exactly
/// like the decoder — deblocking is a whole-picture post-pass.)
/// Note the §8.8.2.3 semantics: intra edges derive `bS = 0` and
/// Table 33 defines `sT` only for `bS ∈ {1, 2, 3}`, so on an all-intra
/// picture the pass is normatively a no-op — this wiring becomes
/// pixel-effective once the P encoder lands.
///
/// Requirements: 4:2:0 source, dimensions multiples of 4 (the §7.4.3.1
/// minimum CB), any bit depth the recon chain supports (8..=16).
pub fn encode_idr_slice_data_with(
    src: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_slice_data_opts(src, slice_qp, deblock, false)
}

/// [`encode_idr_slice_data_with`] plus the `sps_cm_init_flag` entropy
/// selection (round 431): with `cm_init` the emit pass initialises the
/// §9.3.2.2 Main-profile context tables (initType 0 at `slice_qp`) and
/// routes every regular bin through the decoder's §9.3.4.2.1
/// `ctxIdxOffset + ctxInc` selection — the per-syntax-element context
/// modelling that collapses the Baseline single-context rate penalty.
/// The caller must signal `sps_cm_init_flag = 1` in the SPS
/// ([`crate::headers_enc::EncSequenceConfig::cm_init`]).
pub fn encode_idr_slice_data_opts(
    src: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_slice_data_full(src, slice_qp, deblock, cm_init, false)
}

/// [`encode_idr_slice_data_opts`] plus `sps_eipd_flag` (round 455):
/// with `eipd` every leaf runs the 33-mode EIPD search
/// ([`crate::intra_enc`]) and writes the §7.3.8.4 MPM / PIMS /
/// rem-mode luma group plus `intra_chroma_pred_mode`; the caller must
/// signal `sps_eipd_flag = 1` in the SPS.
pub fn encode_idr_slice_data_full(
    src: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
    eipd: bool,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_slice_data_tree(src, slice_qp, deblock, cm_init, eipd, false)
}

/// [`encode_idr_slice_data_full`] plus `sps_btt_flag` (round 458): with
/// `btt` the coding tree is the binary / ternary tree of
/// [`crate::tree_enc`] (the SPS must declare `sps_btt_flag = 1` with the
/// [`crate::tree_enc`] size limits — [`crate::headers_enc`] does);
/// without it the round-429 quad tree.
pub fn encode_idr_slice_data_tree(
    src: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
    eipd: bool,
    btt: bool,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_slice_data_cfg(
        src,
        slice_qp,
        IntraToolset {
            deblock,
            cm_init,
            eipd,
            btt,
            iqt: false,
            ats: false,
            adcc: false,
        },
    )
}

/// The general IDR `slice_data()` encoder over an [`IntraToolset`]
/// (round 458): every earlier entry point is a projection of this one.
/// `ats` requires `iqt` (§7.3.2.1 reads `sps_ats_flag` only under
/// `sps_iqt_flag == 1`).
pub fn encode_idr_slice_data_cfg(
    src: &YuvPicture,
    slice_qp: i32,
    tools: IntraToolset,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    let IntraToolset {
        deblock,
        cm_init,
        eipd,
        btt,
        iqt,
        ats,
        adcc,
    } = tools;
    if ats && !iqt {
        return Err(Error::invalid(
            "evc encoder: sps_ats_flag requires sps_iqt_flag (§7.3.2.1)",
        ));
    }
    if adcc && !cm_init {
        return Err(Error::invalid(
            "evc encoder: sps_adcc_flag requires sps_cm_init_flag (§7.3.2.1)",
        ));
    }
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
    let sel = CtxSel::new(cm_init, InitType::I).with_adcc(adcc);
    let mut ctx = EncCtx {
        src,
        recon,
        qp: slice_qp,
        lambda: rd_lambda(slice_qp, src.bit_depth),
        bit_depth: src.bit_depth,
        pic_w: src.width,
        pic_h: src.height,
        sel,
        eipd,
        side_info: SideInfoGrid::new(src.width, src.height),
        geom: TreeGeometry::encoder(src.width, src.height, btt),
        iqt,
        ats,
        chroma_qp_offset: crate::headers_enc::iqt_chroma_qp_offset(slice_qp, iqt),
    };
    let mut stats = EncStats::default();
    // The decide pass's rate model: the same context table the emit
    // pass starts from, advanced by every decided bin in decode order,
    // so each candidate is costed at the state it would be coded in.
    let mut model = BitCostModel::new();
    if cm_init {
        model.init_main_profile(InitType::I, slice_qp);
    }

    // Decide pass: raster CTU order, committing the chosen recon as we
    // go (later CUs predict from it exactly like the decoder).
    let ctus_x = src.width.div_ceil(1 << CTB_LOG2);
    let ctus_y = src.height.div_ceil(1 << CTB_LOG2);
    let mut roots = Vec::with_capacity((ctus_x * ctus_y) as usize);
    for cy in 0..ctus_y {
        for cx in 0..ctus_x {
            let (node, _cost) = tree_enc::search_split_unit(
                &mut ctx,
                &mut model,
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
    // the decoder's exact read order. Under `cm_init` the encoder's
    // context table starts from the identical §9.3.2.2 init the decoder
    // runs (initType 0 — I slice — at the slice QP), so both context
    // states evolve in lockstep bin for bin.
    let mut enc = CabacEncoder::new();
    if cm_init {
        enc.init_main_profile(InitType::I, slice_qp);
    }
    // The emit-order grid: the §8.4.2 neighbour probes must see exactly
    // the CUs the decoder has already reconstructed at that point.
    let mut emit_grid = SideInfoGrid::new(ctx.pic_w, ctx.pic_h);
    for (x0, y0, node) in &roots {
        let mut tree_stats = stats.tree;
        let mut leaf_fn = |enc: &mut CabacEncoder,
                           grid: &mut SideInfoGrid,
                           x0: u32,
                           y0: u32,
                           lw: u32,
                           lh: u32,
                           plan: &LeafPlan| {
            emit_leaf(
                enc, sel, plan, grid, ctx.pic_w, ctx.pic_h, ctx.ats, x0, y0, lw, lh,
            );
            grid.stamp_block(
                x0,
                y0,
                1u32 << lw,
                1u32 << lh,
                leaf_side_info(plan, ctx.qp, x0, y0, lw, lh),
            );
            stats.leaves += 1;
            stats.cbf_luma_set += u32::from(plan.cbf_y);
            stats.cbf_chroma_set += u32::from(plan.cbf_cb) + u32::from(plan.cbf_cr);
            match plan.intra {
                IntraSel::Baseline(mode_idx) => stats.mode_histogram[mode_idx] += 1,
                IntraSel::Eipd { mode_y, chroma_raw } => {
                    stats.eipd_mode_histogram[mode_y as usize] += 1;
                    stats.eipd_chroma_non_dm += u32::from(chroma_raw != 0);
                }
            }
        };
        tree_enc::emit_tree(
            &mut enc,
            sel,
            &ctx.geom,
            &mut emit_grid,
            *x0,
            *y0,
            CTB_LOG2,
            CTB_LOG2,
            node,
            &mut tree_stats,
            &mut leaf_fn,
        );
        stats.tree = tree_stats;
    }
    stats.split_flag_bins = stats.tree.split_cu_flag_bins;
    enc.encode_terminate(true); // §7.3.8.1 end_of_tile_one_bit

    if deblock {
        // Mirror the decoder's post-reconstruction §8.8.2 pass: stamp
        // the side-info grid exactly as `decode_transform_unit` does
        // for intra luma CUs, arm the single-tile loop-filter bounds,
        // and run the decoder's own deblock kernels on the recon.
        let mut side_info = SideInfoGrid::new(ctx.pic_w, ctx.pic_h);
        let layout = crate::tiles::PicTileLayout::single_tile(ctx.pic_w, ctx.pic_h);
        side_info.tile_bounds = crate::tiles::TileBounds::for_loop_filters(&layout);
        for (x0, y0, node) in &roots {
            stamp_decided(&mut side_info, slice_qp, *x0, *y0, CTB_LOG2, CTB_LOG2, node);
        }
        crate::deblock::deblock_luma(&mut ctx.recon, &side_info, slice_qp)?;
        let off = ctx.chroma_qp_offset;
        crate::deblock::deblock_chroma(&mut ctx.recon, &side_info, slice_qp, off, 1)?;
        crate::deblock::deblock_chroma(&mut ctx.recon, &side_info, slice_qp, off, 2)?;
    }
    Ok((enc.finish(), ctx.recon, stats))
}

/// Stamp the decided tree's leaves into a [`SideInfoGrid`] with the
/// exact `CuSideInfo` the decoder's intra path records — the inputs of
/// the §8.8.2 boundary-strength derivation.
fn stamp_decided(
    side_info: &mut SideInfoGrid,
    slice_qp: i32,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    node: &Node,
) {
    tree_enc::for_each_leaf(x0, y0, log2_w, log2_h, node, &mut |x, y, lw, lh, plan| {
        side_info.stamp_block(
            x,
            y,
            1u32 << lw,
            1u32 << lh,
            leaf_side_info(plan, slice_qp, x, y, lw, lh),
        );
    });
}

/// The side info of one decided intra leaf — what the decoder's
/// `decode_transform_unit` stamps for a MODE_INTRA CU.
fn leaf_side_info(
    plan: &LeafPlan,
    qp: i32,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> CuSideInfo {
    CuSideInfo {
        pred_mode: CuPredMode::Intra,
        cbf_luma: u8::from(plan.cbf_y),
        cu_x0: x0 as u16,
        cu_y0: y0 as u16,
        cu_log2_w: log2_w as u8,
        cu_log2_h: log2_h as u8,
        intra_luma_mode: plan.intra.stamp_value(),
        qp_y: qp.clamp(0, 51) as u8,
        ..Default::default()
    }
}

/// Decide one leaf: 5-mode luma search + §8.4.3 DM chroma, committing the
/// winning reconstruction (via the decoder's own
/// `intra_reconstruct_cb_in_tile`) and the winning bins (into `model`),
/// returning the plan + RD cost. Every candidate's rate is the exact
/// cost of its bin string at the current context state.
fn decide_leaf(
    ctx: &mut EncCtx<'_>,
    model: &mut BitCostModel,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> Result<(LeafPlan, f64)> {
    let w = 1usize << log2_w;
    let h = 1usize << log2_h;
    let bd = ctx.bit_depth;
    let max_val = (1i32 << bd) - 1;
    let sel = ctx.sel;
    let src_y = gather_block(&ctx.src.y, ctx.src.y_stride(), x0, y0, w, h);
    // §8.7.1: quantization runs at qP = Qp′ (eqs. 1050-1052) — the
    // luma bit-depth offset and the chroma ChromaQpTable mapping (the
    // encoder's SPS declares sps_iqt_flag = 0 and zero chroma offsets).
    let qp_y = crate::dequant::qp_prime_y(ctx.qp, bd);
    let qp_c = crate::dequant::qp_prime_c(ctx.qp, ctx.chroma_qp_offset, bd, ctx.iqt);
    let lambda_c = rd_lambda_at_qp_prime(qp_c);
    let spec = TransformSpec::dct(ctx.iqt);
    let ats_present = ctx.ats && log2_w <= 5 && log2_h <= 5;
    let wc = 1usize << (log2_w - 1);
    let hc = 1usize << (log2_h - 1);
    let src_cb = gather_block(&ctx.src.cb, ctx.src.c_stride(), x0 >> 1, y0 >> 1, wc, hc);
    let src_cr = gather_block(&ctx.src.cr, ctx.src.c_stride(), x0 >> 1, y0 >> 1, wc, hc);

    let (plan, cost) = if ctx.eipd {
        // ---- EIPD luma: SAD-ranked candidates + MPMs through the RD ----
        let lists = intra_enc::mode_lists(&ctx.side_info, ctx.pic_w, ctx.pic_h, x0, y0, log2_w);
        let mode_bits = |model: &mut BitCostModel, selector: ModeSelector| -> f64 {
            model.measure(|m| intra_enc::emit_luma_selector(m, sel, selector))
        };
        let cands = intra_enc::luma_candidates(
            &ctx.recon,
            &src_y,
            x0,
            y0,
            log2_w,
            log2_h,
            &lists,
            ctx.lambda,
            |selector| mode_bits(model, selector),
        );
        let mut best: Option<EipdLumaCand> = None;
        let mut best_cost = f64::INFINITY;
        for &mode in &cands {
            let pred = intra_enc::predict(&ctx.recon, x0, y0, log2_w, log2_h, 0, mode);
            let rdoq = RdoqInputs::new(model, ctx.lambda, sel, 0, MainCtxTable::CbfLuma);
            let (levels, cbf, res, dist) =
                quantize_pred(&pred, &src_y, w, h, qp_y, bd, max_val, Some(&rdoq), spec)?;
            let selector = intra_enc::selector_for(&lists, mode);
            let mode_bits = model.measure(|m| intra_enc::emit_luma_selector(m, sel, selector));
            let bits = mode_bits
                + luma_tail_bits(
                    model,
                    sel,
                    ats_present,
                    AtsIntra::disabled(),
                    cbf,
                    &levels,
                    log2_w,
                    log2_h,
                );
            let cost = dist + ctx.lambda * bits;
            if cost < best_cost {
                best_cost = cost;
                best = Some((mode, levels, cbf, res, pred, mode_bits));
            }
        }
        let (mode_y, levels_y, cbf_y, res_y, pred_y, mode_bits) =
            best.expect("at least one candidate");
        let mut luma = LumaChoice {
            levels: levels_y,
            cbf: cbf_y,
            res: res_y,
            cost: best_cost,
            ats: AtsIntra::disabled(),
        };
        if ats_present {
            let q = IntraQuantCtx {
                bit_depth: bd,
                lambda: ctx.lambda,
                sel,
                iqt: ctx.iqt,
            };
            refine_luma_ats(
                &q, model, &pred_y, &src_y, log2_w, log2_h, qp_y, mode_bits, &mut luma,
            )?;
        }
        let LumaChoice {
            levels: levels_y,
            cbf: cbf_y,
            res: res_y,
            cost: best_cost,
            ats: ats_y,
        } = luma;
        intra_reconstruct_cb_eipd_in_tile(
            &mut ctx.recon,
            x0,
            y0,
            log2_w,
            log2_h,
            mode_y,
            0,
            false,
            false,
            &res_y,
            None,
        )?;
        // ---- EIPD chroma: the five intra_chroma_pred_mode values ----
        let mut best_c: Option<intra_enc::ChromaChoice> = None;
        let mut best_c_cost = f64::INFINITY;
        for raw in 0..=4i32 {
            let mode_c = derive_chroma_mode(raw, mode_y);
            let pred_cb = intra_enc::predict(&ctx.recon, x0, y0, log2_w, log2_h, 1, mode_c);
            let pred_cr = intra_enc::predict(&ctx.recon, x0, y0, log2_w, log2_h, 2, mode_c);
            let rdoq_cb = RdoqInputs::new(model, lambda_c, sel, 1, MainCtxTable::CbfCb);
            let (lv_cb, cbf_cb, res_cb, d_cb) = quantize_pred(
                &pred_cb,
                &src_cb,
                wc,
                hc,
                qp_c,
                bd,
                max_val,
                Some(&rdoq_cb),
                spec,
            )?;
            let rdoq_cr = RdoqInputs::new(model, lambda_c, sel, 2, MainCtxTable::CbfCr);
            let (lv_cr, cbf_cr, res_cr, d_cr) = quantize_pred(
                &pred_cr,
                &src_cr,
                wc,
                hc,
                qp_c,
                bd,
                max_val,
                Some(&rdoq_cr),
                spec,
            )?;
            let bits = model.measure(|m| {
                intra_enc::emit_chroma_pred_mode(m, sel, raw);
                let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
                m.encode_decision(t, i, u8::from(cbf_cb));
                let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
                m.encode_decision(t, i, u8::from(cbf_cr));
                if cbf_cb {
                    emit_residual(m, sel, 1, &lv_cb, log2_w - 1, log2_h - 1);
                }
                if cbf_cr {
                    emit_residual(m, sel, 2, &lv_cr, log2_w - 1, log2_h - 1);
                }
            });
            let cost = d_cb + d_cr + ctx.lambda * bits;
            if cost < best_c_cost {
                best_c_cost = cost;
                best_c = Some((raw, mode_c, lv_cb, cbf_cb, res_cb, lv_cr, cbf_cr, res_cr));
            }
        }
        let (raw, mode_c, levels_cb, cbf_cb, res_cb, levels_cr, cbf_cr, res_cr) =
            best_c.expect("five chroma candidates");
        for (c_idx, res) in [(1u32, &res_cb), (2, &res_cr)] {
            intra_reconstruct_cb_eipd_in_tile(
                &mut ctx.recon,
                x0,
                y0,
                log2_w,
                log2_h,
                mode_c,
                c_idx,
                false,
                false,
                res,
                None,
            )?;
        }
        (
            LeafPlan {
                intra: IntraSel::Eipd {
                    mode_y,
                    chroma_raw: raw,
                },
                ats: ats_y,
                levels_y,
                cbf_y,
                levels_cb,
                cbf_cb,
                levels_cr,
                cbf_cr,
            },
            best_cost + best_c_cost,
        )
    } else {
        // ---- Baseline luma: 5-mode search over the decoder's reference fetch ----
        let refs = ctx.recon.fetch_intra_refs(x0, y0, w, h, 0);
        let mut best: Option<BaselineLumaCand> = None;
        let mut best_cost = f64::INFINITY;
        for (mode_idx, &mode) in MODES.iter().enumerate() {
            let rdoq = RdoqInputs::new(model, ctx.lambda, sel, 0, MainCtxTable::CbfLuma);
            let (levels, cbf, res, dist) = quantize_block(
                &refs,
                mode,
                &src_y,
                w,
                h,
                qp_y,
                bd,
                max_val,
                Some(&rdoq),
                spec,
            )?;
            let mode_bits = model.measure(|m| emit_intra_pred_mode(m, sel, mode_idx));
            let bits = mode_bits
                + luma_tail_bits(
                    model,
                    sel,
                    ats_present,
                    AtsIntra::disabled(),
                    cbf,
                    &levels,
                    log2_w,
                    log2_h,
                );
            let cost = dist + ctx.lambda * bits;
            if cost < best_cost {
                best_cost = cost;
                best = Some((mode_idx, levels, cbf, res, mode_bits));
            }
        }
        let (mode_idx, levels_y, cbf_y, res_y, mode_bits) = best.expect("5 candidates");
        let mut luma = LumaChoice {
            levels: levels_y,
            cbf: cbf_y,
            res: res_y,
            cost: best_cost,
            ats: AtsIntra::disabled(),
        };
        if ats_present {
            let mut pred = vec![0i32; w * h];
            predict(MODES[mode_idx], &refs, w, h, bd, &mut pred);
            let q = IntraQuantCtx {
                bit_depth: bd,
                lambda: ctx.lambda,
                sel,
                iqt: ctx.iqt,
            };
            refine_luma_ats(
                &q, model, &pred, &src_y, log2_w, log2_h, qp_y, mode_bits, &mut luma,
            )?;
        }
        let LumaChoice {
            levels: levels_y,
            cbf: cbf_y,
            res: res_y,
            cost: cost_y,
            ats: ats_y,
        } = luma;
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

        // ---- chroma: §8.4.3 DM — the never-present (`sps_eipd_flag == 0`)
        // `intra_chroma_pred_mode` is inferred 0, so IntraPredModeC =
        // IntraPredModeY (the mode just decided for this SINGLE_TREE CU);
        // quantize each component's residual under that prediction ----
        let mode_c = MODES[mode_idx];
        let mut cost = cost_y;
        let mut chroma = Vec::with_capacity(2);
        for c_idx in 1..=2u32 {
            let src_c = if c_idx == 1 { &src_cb } else { &src_cr };
            let refs_c = ctx.recon.fetch_intra_refs(x0 >> 1, y0 >> 1, wc, hc, c_idx);
            let table = if c_idx == 1 {
                MainCtxTable::CbfCb
            } else {
                MainCtxTable::CbfCr
            };
            let rdoq = RdoqInputs::new(model, lambda_c, sel, c_idx, table);
            let (levels, cbf, res, dist) = quantize_block(
                &refs_c,
                mode_c,
                src_c,
                wc,
                hc,
                qp_c,
                bd,
                max_val,
                Some(&rdoq),
                spec,
            )?;
            let bits = model.measure(|m| {
                let (t, i) = sel.ctx(table, 0);
                m.encode_decision(t, i, u8::from(cbf));
                if cbf {
                    emit_residual(m, sel, c_idx, &levels, log2_w - 1, log2_h - 1);
                }
            });
            cost += dist + ctx.lambda * bits;
            intra_reconstruct_cb_in_tile(
                &mut ctx.recon,
                x0,
                y0,
                log2_w,
                log2_h,
                mode_c,
                c_idx,
                &res,
                None,
            )?;
            chroma.push((levels, cbf));
        }
        let (levels_cr, cbf_cr) = chroma.pop().expect("cr");
        let (levels_cb, cbf_cb) = chroma.pop().expect("cb");
        (
            LeafPlan {
                intra: IntraSel::Baseline(mode_idx),
                ats: ats_y,
                levels_y,
                cbf_y,
                levels_cb,
                cbf_cb,
                levels_cr,
                cbf_cr,
            },
            cost,
        )
    };

    // Advance the rate model over the decided leaf's exact bin string
    // (the §8.4.2 probes read the L/A/R neighbours, which this CU's own
    // stamp does not touch — so the pre-stamp grid is the emit grid),
    // then stamp the CU for the CUs that follow.
    model.commit(|m| {
        emit_leaf(
            m,
            sel,
            &plan,
            &ctx.side_info,
            ctx.pic_w,
            ctx.pic_h,
            ctx.ats,
            x0,
            y0,
            log2_w,
            log2_h,
        )
    });
    ctx.side_info.stamp_block(
        x0,
        y0,
        w as u32,
        h as u32,
        leaf_side_info(&plan, ctx.qp, x0, y0, log2_w, log2_h),
    );
    Ok((plan, cost))
}

/// An EIPD luma candidate under evaluation: `(mode, levels, cbf,
/// residual, prediction, mode bits)`.
pub(crate) type EipdLumaCand = (i32, Vec<i32>, bool, Vec<i32>, Vec<i32>, f64);
/// A Baseline luma candidate under evaluation: `(mode index, levels,
/// cbf, residual, mode bits)`.
pub(crate) type BaselineLumaCand = (usize, Vec<i32>, bool, Vec<i32>, f64);

/// The decided luma transform block of an intra leaf under evaluation.
pub(crate) struct LumaChoice {
    pub levels: Vec<i32>,
    pub cbf: bool,
    pub res: Vec<i32>,
    /// `D + λ · R` including the mode syntax.
    pub cost: f64,
    pub ats: AtsIntra,
}

/// The exact rate of an intra luma TB's tail — `cbf_luma`, the
/// §7.3.8.5 ATS-intra group when it is present (`sps_ats_flag`, both
/// sides ≤ 32, `cbf_luma == 1`) and the residual string — at the
/// model's current context state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_tail_bits(
    model: &mut BitCostModel,
    sel: CtxSel,
    ats_present: bool,
    ats: AtsIntra,
    cbf: bool,
    levels: &[i32],
    log2_w: u32,
    log2_h: u32,
) -> f64 {
    model.measure(|m| {
        let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
        m.encode_decision(t, i, u8::from(cbf));
        if cbf {
            if ats_present {
                ats::write_ats_intra(m, EipdCtx::for_slice(sel.cm_init, sel.init_type), ats);
            }
            emit_residual(m, sel, 0, levels, log2_w, log2_h);
        }
    })
}

/// §7.3.8.5 ATS-intra search (round 458): re-quantize the chosen
/// mode's luma residual under each Table-30 kernel pair (DST-VII /
/// DCT-VIII per direction, `sps_iqt_flag` chain) with the RDOQ
/// trellis, and keep whichever of the five transforms minimises
/// `D + λ · R` — `R` including the `ats_cu_intra_flag` /
/// `ats_hor_mode` / `ats_ver_mode` bins at the current context state.
/// `mode_bits` is the (transform-independent) rate of the mode syntax.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_luma_ats(
    ctx: &IntraQuantCtx,
    model: &mut BitCostModel,
    pred: &[i32],
    src: &[i32],
    log2_w: u32,
    log2_h: u32,
    qp_y: i32,
    mode_bits: f64,
    best: &mut LumaChoice,
) -> Result<()> {
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    let max_val = (1i32 << ctx.bit_depth) - 1;
    for choice in ats::ATS_INTRA_CHOICES {
        let spec = TransformSpec {
            tr_type_hor: choice.tr_type_hor,
            tr_type_ver: choice.tr_type_ver,
            sps_iqt_flag: ctx.iqt,
        };
        let rdoq = RdoqInputs::new(model, ctx.lambda, ctx.sel, 0, MainCtxTable::CbfLuma);
        let (levels, cbf, res, dist) = quantize_pred(
            pred,
            src,
            w,
            h,
            qp_y,
            ctx.bit_depth,
            max_val,
            Some(&rdoq),
            spec,
        )?;
        if !cbf {
            // A silent TB signals no kernel; the DCT-II candidate
            // already covers it.
            continue;
        }
        let bits =
            mode_bits + luma_tail_bits(model, ctx.sel, true, choice, cbf, &levels, log2_w, log2_h);
        let cost = dist + ctx.lambda * bits;
        if cost < best.cost {
            *best = LumaChoice {
                levels,
                cbf,
                res,
                cost,
                ats: choice,
            };
        }
    }
    Ok(())
}

/// What [`refine_luma_ats`] needs of a slice encoder's state.
#[derive(Clone, Copy)]
pub(crate) struct IntraQuantCtx {
    pub bit_depth: u32,
    pub lambda: f64,
    pub sel: CtxSel,
    pub iqt: bool,
}

/// Quantize one residual block under `spec`: the RDOQ trellis when
/// `rdoq` is given (levels chosen under `D + λ · R` at the model's
/// context state), else nearest-level rounding. Returns `(levels, cbf)`.
pub(crate) fn quantize_residual(
    diff: &[i32],
    w: usize,
    h: usize,
    qp: i32,
    bit_depth: u32,
    rdoq: Option<&RdoqInputs<'_>>,
    spec: TransformSpec,
) -> Result<(Vec<i32>, bool)> {
    match rdoq {
        Some(inp) => {
            let frac = forward_transform_fractional_typed(diff, w, h, qp, bit_depth, spec)?;
            let weights = level_unit_sse_weights_typed(w, h, qp, bit_depth, spec);
            // The residual syntax the selector carries decides the
            // optimiser: the RLE trellis or the ADCC candidate search.
            let (levels, cbf, _cost) = if inp.sel.adcc {
                crate::adcc::rdoq_adcc(&frac, &weights, w, h, 1, inp)
            } else {
                rdoq_rle(&frac, &weights, w, h, inp)
            };
            Ok((levels, cbf))
        }
        None => {
            let mut levels = vec![0i32; w * h];
            let cbf = forward_quantize_typed(diff, &mut levels, w, h, qp, bit_depth, spec)?;
            Ok((levels, cbf))
        }
    }
}

/// The RD Lagrange multiplier, calibrated to the §8.7 quantizer step
/// this crate's decode chain actually realises: `λ = RD_LAMBDA_SCALE ·
/// Δ²`, where `Δ` is the pixel-domain (orthonormal-basis) step of one
/// `TransCoeffLevel` unit at `Qp′Y` (eq. 1043),
///
/// ```text
/// Δ = levelScale[ Qp′ % 6 ] · 2^( Qp′ / 6 ) / 2^10
/// ```
///
/// — eq. 1059's `levelScale << ( qP / 6 ) >> bdShift` (eq. 1056:
/// `BitDepth + Log2( nTbS ) − 5`) carried through the eq. 1062 kernels
/// (row norm² `nTbS · 64²`) and the eq. 1053/1055 `( 20 − BitDepth ) +
/// 7` renormalisation: the bit-depth and size terms cancel, leaving the
/// step above for every TB shape (rectangular shapes via `rectNorm`,
/// eq. 1058, to within its 181/128 rounding). At 8 bits, Qp′ 32 gives
/// Δ ≈ 1.6 and Qp′ 51 ≈ 14 — this chain's QP scale is 24 finer than
/// the classic `2^(( QP − 4 ) / 6)` step, which is why the historical
/// `0.57 · 2^(( qp − 12 ) / 3)` (right for that classic step) sat ~256×
/// too high here and drove every inter picture into the skip ladder.
/// `RD_LAMBDA_SCALE` is the classic constant re-expressed in Δ²
/// (`0.57 · 2^(( QP − 12 ) / 3) / Δ_classic( QP )²`, QP-independent), then
/// tuned on the crate's corpus (see the CHANGELOG round-455 entry).
///
/// Evaluated without transcendental calls (`2^( Qp′ / 6 )` is an exact
/// power of two here since Δ² only needs `2^( 2 · ⌊Qp′ / 6⌋ )`), so
/// every platform's encoder lands on bit-identical costs — the
/// MD5-pinned stream fixtures rely on it.
pub(crate) fn rd_lambda(qp: i32, bit_depth: u32) -> f64 {
    rd_lambda_at_qp_prime(crate::dequant::qp_prime_y(qp, bit_depth))
}

/// [`rd_lambda`] at an explicit `Qp′` — the chroma planes quantize at
/// `Qp′Cb` / `Qp′Cr` (eqs. 1048/1049, through the ChromaQpTable), so
/// their level decisions trade distortion against bits at the λ of
/// *their* step.
pub(crate) fn rd_lambda_at_qp_prime(qp_prime: i32) -> f64 {
    let ls = f64::from(crate::dequant::LEVEL_SCALE_BASELINE[qp_prime.rem_euclid(6) as usize]);
    let step = ls * 2f64.powi(qp_prime.div_euclid(6)) / 1024.0;
    RD_LAMBDA_SCALE * step * step
}

/// `λ / Δ²` — see [`rd_lambda`].
pub(crate) const RD_LAMBDA_SCALE: f64 = 0.09;

#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_block(
    refs: &RefSamples,
    mode: IntraMode,
    src: &[i32],
    w: usize,
    h: usize,
    qp: i32,
    bit_depth: u32,
    max_val: i32,
    rdoq: Option<&RdoqInputs<'_>>,
    spec: TransformSpec,
) -> Result<(Vec<i32>, bool, Vec<i32>, f64)> {
    let n = w * h;
    let mut pred = vec![0i32; n];
    predict(mode, refs, w, h, bit_depth, &mut pred);
    quantize_pred(&pred, src, w, h, qp, bit_depth, max_val, rdoq, spec)
}

/// Transform + quantize + reconstruct the residual of `src` against an
/// already-computed prediction `pred`; returns (levels, cbf, recon
/// residual, SSE distortion) exactly like [`quantize_block`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_pred(
    pred: &[i32],
    src: &[i32],
    w: usize,
    h: usize,
    qp: i32,
    bit_depth: u32,
    max_val: i32,
    rdoq: Option<&RdoqInputs<'_>>,
    spec: TransformSpec,
) -> Result<(Vec<i32>, bool, Vec<i32>, f64)> {
    let n = w * h;
    let diff: Vec<i32> = src.iter().zip(pred.iter()).map(|(&s, &p)| s - p).collect();
    let (levels, cbf) = quantize_residual(&diff, w, h, qp, bit_depth, rdoq, spec)?;
    let mut res = vec![0i32; n];
    if cbf {
        scale_and_inverse_transform_ats(
            &levels,
            &mut res,
            w,
            h,
            qp,
            bit_depth,
            spec.tr_type_hor,
            spec.tr_type_ver,
            spec.sps_iqt_flag,
        )?;
    }
    let mut dist = 0f64;
    for i in 0..n {
        let rec = (pred[i] + res[i]).clamp(0, max_val);
        let d = (rec - src[i]) as f64;
        dist += d * d;
    }
    Ok((levels, cbf, res, dist))
}

pub(crate) fn gather_block(
    plane: &[u16],
    stride: usize,
    x0: u32,
    y0: u32,
    w: usize,
    h: usize,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(w * h);
    for yy in 0..h {
        let row = (y0 as usize + yy) * stride + x0 as usize;
        out.extend(plane[row..row + w].iter().map(|&v| v as i32));
    }
    out
}

/// Saved recon rectangle (luma + both chroma sub-rects), for the
/// leaf-vs-split trial rewinds.
pub(crate) struct RegionSave {
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
}

pub(crate) fn save_region(
    pic: &YuvPicture,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) -> RegionSave {
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

pub(crate) fn restore_region(
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

/// `intra_pred_mode` — U over Table 62 with the Table 95 ctxInc (bin0
/// → 0, later bins → 1) under `cm_init`; all bins on (0, 0) under the
/// Baseline collapse. The decoder reads it via `decode_u_regular`
/// (63-bin compat cap).
pub(crate) fn emit_intra_pred_mode<S: BinSink>(enc: &mut S, sel: CtxSel, mode_idx: usize) {
    let table = MainCtxTable::IntraPredMode;
    let (t, off) = if sel.cm_init {
        (table.as_usize(), table.ctx_idx_offset(sel.init_type))
    } else {
        (0, table.cm0_ctx_idx_offset(sel.init_type))
    };
    enc.encode_u_regular_capped(mode_idx as u32, 63, t, |bin_idx| {
        off + (bin_idx as usize).min(1)
    });
}

/// One SINGLE_TREE `coding_unit()`: the intra-mode syntax (Baseline
/// `intra_pred_mode`, or the EIPD luma group + `intra_chroma_pred_mode`
/// against the §8.4.2 lists derived from `grid` — the neighbours as the
/// decoder holds them at this point), then the `transform_unit()`.
#[allow(clippy::too_many_arguments)]
fn emit_leaf<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    plan: &LeafPlan,
    grid: &SideInfoGrid,
    pic_w: u32,
    pic_h: u32,
    ats_enabled: bool,
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
) {
    match plan.intra {
        IntraSel::Baseline(mode_idx) => emit_intra_pred_mode(enc, sel, mode_idx),
        IntraSel::Eipd { mode_y, chroma_raw } => {
            let lists = intra_enc::mode_lists(grid, pic_w, pic_h, x0, y0, log2_w);
            intra_enc::emit_luma_selector(enc, sel, intra_enc::selector_for(&lists, mode_y));
            intra_enc::emit_chroma_pred_mode(enc, sel, chroma_raw);
        }
    }
    // transform_unit(), SINGLE_TREE (§7.3.8.5): cbf_cb then cbf_cr
    // (Tables 76/77, ctxInc 0 — the line-3066 chroma-carrying-tree
    // gate), then cbf_luma (Table 75 — always present on a MODE_INTRA
    // CU), then the residuals in luma, Cb, Cr order (chroma at the
    // 4:2:0 sub-sampled dimensions).
    let (t, i) = sel.ctx(MainCtxTable::CbfCb, 0);
    enc.encode_decision(t, i, u8::from(plan.cbf_cb));
    let (t, i) = sel.ctx(MainCtxTable::CbfCr, 0);
    enc.encode_decision(t, i, u8::from(plan.cbf_cr));
    let (t, i) = sel.ctx(MainCtxTable::CbfLuma, 0);
    enc.encode_decision(t, i, u8::from(plan.cbf_y));
    if plan.cbf_y {
        // §7.3.8.5 lines 3080-3087: the ATS-intra group, after the
        // (absent) cu_qp_delta block and before the residuals.
        if ats::ats_intra_flag_present(ats_enabled, log2_w, log2_h, true) {
            ats::write_ats_intra(
                enc,
                EipdCtx::for_slice(sel.cm_init, sel.init_type),
                plan.ats,
            );
        }
        emit_residual(enc, sel, 0, &plan.levels_y, log2_w, log2_h);
    }
    if plan.cbf_cb {
        emit_residual(enc, sel, 1, &plan.levels_cb, log2_w - 1, log2_h - 1);
    }
    if plan.cbf_cr {
        emit_residual(enc, sel, 2, &plan.levels_cr, log2_w - 1, log2_h - 1);
    }
}

/// §7.3.8.6 `residual_coding()` writer (round 458): the §7.3.8.8
/// advanced coding ([`crate::adcc::encode_residual_coding_adv`]) when
/// the selector carries `sps_adcc_flag`, else the §7.3.8.7 run-length
/// coding ([`emit_residual_rle`]). 4:2:0 (`ChromaArrayType == 1`).
#[doc(hidden)]
pub fn emit_residual<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    c_idx: u32,
    levels: &[i32],
    log2_w: u32,
    log2_h: u32,
) {
    if sel.adcc {
        crate::adcc::encode_residual_coding_adv(enc, sel, c_idx, 1, levels, log2_w, log2_h);
    } else {
        emit_residual_rle(enc, sel, c_idx, levels, log2_w, log2_h);
    }
}

/// §7.3.8.7 `residual_coding_rle()` writer — the exact dual of
/// `decode_residual_coding_rle`: per non-zero coefficient in §6.5.2
/// zig-zag order, `coeff_zero_run` (U, cMax = blockSize − 1, Table 84),
/// `coeff_abs_level_minus1` (U, cMax 32767, Table 85), `coeff_sign_flag`
/// (bypass), and `coeff_last_flag` (Table 86) unless the coefficient
/// sits at the final scan position (where the decoder infers 1). Under
/// `cm_init` the run/level bins carry the §9.3.4.2.2 eq. 1434/1435
/// ctxInc driven by `cIdx`, the bin position and the §7.3.8.7
/// `PrevLevel` chain (init 6, then the previous coefficient's absolute
/// level), and `coeff_last_flag` the Table 95 `cIdx == 0 ? 0 : 1`.
#[doc(hidden)]
pub fn emit_residual_rle<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    c_idx: u32,
    levels: &[i32],
    log2_w: u32,
    log2_h: u32,
) {
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
    let mut prev_level = 6u32;
    let last = nz.len() - 1;
    for (i, &(scan_pos, level)) in nz.iter().enumerate() {
        let zero_run = (scan_pos - cursor) as u32;
        let lvl_minus1 = level.unsigned_abs() - 1;
        if sel.cm_init {
            let zr_table = MainCtxTable::CoeffZeroRun;
            let zr_off = zr_table.ctx_idx_offset(sel.init_type);
            enc.encode_u_regular_capped(zero_run, zero_run_c_max, zr_table.as_usize(), |bin_idx| {
                zr_off + ctx_inc_coeff_zero_run(bin_idx, c_idx, prev_level)
            });
            let lv_table = MainCtxTable::CoeffAbsLevelMinus1;
            let lv_off = lv_table.ctx_idx_offset(sel.init_type);
            enc.encode_u_regular_capped(lvl_minus1, 32767, lv_table.as_usize(), |bin_idx| {
                lv_off + ctx_inc_coeff_zero_run(bin_idx, c_idx, prev_level)
            });
        } else {
            // Table 95, sps_cm_init_flag == 0 rows: bin 0 is
            // `cIdx == 0 ? 0 : 2`, later bins `cIdx == 0 ? 1 : 3`, on
            // the shared ctxTable 0 at each element's Table-39 offset.
            let chroma = if c_idx == 0 { 0 } else { 2 };
            let zr_off = MainCtxTable::CoeffZeroRun.cm0_ctx_idx_offset(sel.init_type);
            enc.encode_u_regular_capped(zero_run, zero_run_c_max, 0, |bin_idx| {
                zr_off + chroma + (bin_idx.min(1) as usize)
            });
            let lv_off = MainCtxTable::CoeffAbsLevelMinus1.cm0_ctx_idx_offset(sel.init_type);
            enc.encode_u_regular_capped(lvl_minus1, 32767, 0, |bin_idx| {
                lv_off + chroma + (bin_idx.min(1) as usize)
            });
        }
        enc.encode_bypass(u8::from(level < 0));
        if scan_pos < total - 1 {
            let inc = if c_idx == 0 { 0 } else { 1 };
            let (t, ci) = sel.ctx(MainCtxTable::CoeffLastFlag, inc);
            enc.encode_decision(t, ci, u8::from(i == last));
        }
        // §7.3.8.7: PrevLevel = coeff_abs_level_minus1 + 1.
        prev_level = lvl_minus1 + 1;
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
        walk_inputs_cm(w, h, false)
    }

    fn walk_inputs_cm(w: u32, h: u32, cm_init: bool) -> SliceWalkInputs {
        SliceWalkInputs {
            pic_width: w,
            pic_height: h,
            ctb_log2_size_y: CTB_LOG2,
            min_cb_log2_size_y: 2,
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
            tree_gates: CodingTreeGates {
                sps_cm_init_flag: cm_init,
                ..CodingTreeGates::default()
            },
        }
    }

    /// The `sps_btt_flag == 1` walker gates — what the decoder derives
    /// from the encoder's BTT SPS.
    fn walk_inputs_btt(w: u32, h: u32, cm_init: bool) -> SliceWalkInputs {
        let mut inputs = walk_inputs_cm(w, h, cm_init);
        inputs.tree_gates.sps_btt_flag = true;
        inputs.tree_gates.btt_limits = TreeGeometry::encoder(w, h, true).btt.unwrap();
        inputs
    }

    fn decode_inputs(qp: i32) -> SliceDecodeInputs {
        SliceDecodeInputs {
            slice_qp: qp,
            sps_iqt_flag: false,
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
    /// byte-exactly the encoder's recon, across a size × QP ×
    /// `sps_cm_init_flag` matrix that covers CTU-aligned, sub-CTU and
    /// implicit-boundary-split shapes on both entropy shapes.
    #[test]
    fn round_trip_recon_exact_size_qp_matrix() {
        for &cm_init in &[false, true] {
            for &(w, h) in &[(64u32, 64u32), (32, 32), (128, 96), (100, 60), (176, 144)] {
                for &qp in &[4i32, 22, 37, 51] {
                    let src = synth_picture(w, h, 0xC0FFEE ^ (w * 31 + h * 7 + qp as u32));
                    let (payload, enc_recon, stats) =
                        encode_idr_slice_data_opts(&src, qp, false, cm_init).expect("encode");
                    assert!(stats.ctus >= 1);
                    let (dec, dec_stats) = decode_baseline_idr_slice(
                        &payload,
                        walk_inputs_cm(w, h, cm_init),
                        decode_inputs(qp),
                    )
                    .unwrap_or_else(|e| panic!("{w}x{h} qp{qp} cm{cm_init}: decode failed: {e}"));
                    assert_eq!(dec.y, enc_recon.y, "{w}x{h} qp{qp} cm{cm_init}: luma recon");
                    assert_eq!(dec.cb, enc_recon.cb, "{w}x{h} qp{qp} cm{cm_init}: cb recon");
                    assert_eq!(dec.cr, enc_recon.cr, "{w}x{h} qp{qp} cm{cm_init}: cr recon");
                    assert_eq!(dec_stats.ctus, stats.ctus);
                }
            }
        }
    }

    /// Round 458 — **BTT**: the binary / ternary coding tree round-trips
    /// recon-exactly through the decoder's `sps_btt_flag == 1` walker
    /// (the `btt_split_*` group incl. the eq. 1440 `numSmaller` ctxInc,
    /// rectangular leaves down to 4×8 / 8×4 with 2-wide chroma TBs,
    /// ternary shapes, the implicit boundary binary splits) on both
    /// entropy shapes, Baseline and EIPD intra alike; the tree actually
    /// uses the binary shapes and never emits a `split_cu_flag`.
    #[test]
    fn btt_round_trip_recon_exact_size_qp_matrix() {
        for &cm_init in &[false, true] {
            for &eipd in &[false, true] {
                for &(w, h) in &[(64u32, 64u32), (32, 32), (72, 40), (100, 60), (176, 144)] {
                    for &qp in &[10i32, 30, 44] {
                        let src = synth_picture(w, h, 0xB77 ^ (w * 31 + h * 7 + qp as u32));
                        let (payload, enc_recon, stats) =
                            encode_idr_slice_data_tree(&src, qp, false, cm_init, eipd, true)
                                .expect("encode");
                        assert_eq!(stats.split_flag_bins, 0, "no split_cu_flag under BTT");
                        assert!(stats.tree.bt_splits > 0, "{w}x{h} qp{qp}: binary splits");
                        let mut walk = walk_inputs_btt(w, h, cm_init);
                        walk.sps_eipd_flag = eipd;
                        let (dec, dec_stats) =
                            decode_baseline_idr_slice(&payload, walk, decode_inputs(qp))
                                .unwrap_or_else(|e| {
                                    panic!("{w}x{h} qp{qp} cm{cm_init} eipd{eipd}: {e}")
                                });
                        assert_eq!(dec.y, enc_recon.y, "{w}x{h} qp{qp} cm{cm_init}: luma");
                        assert_eq!(dec.cb, enc_recon.cb, "{w}x{h} qp{qp} cm{cm_init}: cb");
                        assert_eq!(dec.cr, enc_recon.cr, "{w}x{h} qp{qp} cm{cm_init}: cr");
                        assert_eq!(dec_stats.ctus, stats.ctus);
                        assert_eq!(dec_stats.tree.btt.flag_bins, stats.tree.btt_flag_bins);
                        assert_eq!(dec_stats.tree.btt.dir_bins, stats.tree.btt_dir_bins);
                        assert_eq!(dec_stats.tree.btt.type_bins, stats.tree.btt_type_bins);
                        assert_eq!(dec_stats.coding_units, stats.leaves);
                    }
                }
            }
        }
    }

    /// Round 458 — **ADCC**: the §7.3.8.8 residual writer with its
    /// candidate-set RDOQ round-trips recon-exactly through the
    /// decoder's `sps_adcc_flag == 1` walker (both trees, with and
    /// without ATS) and the syntax counts match; the run-length shape is
    /// untouched (`adcc` off keeps the RLE bins).
    #[test]
    fn adcc_round_trip_recon_exact_matrix() {
        for &btt in &[false, true] {
            for &ats in &[false, true] {
                for &(w, h) in &[(64u32, 64u32), (72, 40), (176, 144)] {
                    for &qp in &[8i32, 26, 44] {
                        let src = synth_picture(w, h, 0xADC ^ (w * 31 + h * 7 + qp as u32));
                        let tools = IntraToolset {
                            deblock: false,
                            cm_init: true,
                            eipd: true,
                            btt,
                            iqt: ats,
                            ats,
                            adcc: true,
                        };
                        let (payload, enc_recon, stats) =
                            encode_idr_slice_data_cfg(&src, qp, tools).expect("encode");
                        let mut walk = if btt {
                            walk_inputs_btt(w, h, true)
                        } else {
                            walk_inputs_cm(w, h, true)
                        };
                        walk.sps_eipd_flag = true;
                        walk.sps_adcc_flag = true;
                        walk.tree_gates.sps_ats_flag = ats;
                        let off = crate::headers_enc::iqt_chroma_qp_offset(qp, ats);
                        let dec_in = SliceDecodeInputs {
                            sps_iqt_flag: ats,
                            slice_cb_qp_offset: off,
                            slice_cr_qp_offset: off,
                            ..decode_inputs(qp)
                        };
                        let (dec, dec_stats) = decode_baseline_idr_slice(&payload, walk, dec_in)
                            .unwrap_or_else(|e| panic!("{w}x{h} qp{qp} btt{btt} ats{ats}: {e}"));
                        assert_eq!(dec.y, enc_recon.y, "{w}x{h} qp{qp} btt{btt} ats{ats}: luma");
                        assert_eq!(dec.cb, enc_recon.cb, "{w}x{h} qp{qp} btt{btt} ats{ats}: cb");
                        assert_eq!(dec.cr, enc_recon.cr, "{w}x{h} qp{qp} btt{btt} ats{ats}: cr");
                        assert_eq!(dec_stats.coding_units, stats.leaves);
                        assert!(
                            dec_stats.adcc.blocks > 0,
                            "{w}x{h} qp{qp}: ADCC blocks coded"
                        );
                        assert_eq!(dec_stats.coeff_runs, 0, "no run-length bins under ADCC");
                    }
                }
            }
        }
        let src = synth_picture(64, 64, 2);
        assert!(encode_idr_slice_data_cfg(
            &src,
            30,
            IntraToolset {
                adcc: true,
                ..IntraToolset::default()
            }
        )
        .is_err());
    }

    /// Round 458 — **IQT + ATS-intra**: the improved quantization /
    /// transform chain with the balancing chroma QP offset and the
    /// Table-30 kernel search round-trip recon-exactly through the
    /// decoder's `sps_iqt_flag == 1` / `sps_ats_flag == 1` walker on
    /// both entropy shapes and both trees; the streams actually use the
    /// alternative kernels and the ATS syntax bins match the decoder's
    /// counts.
    #[test]
    fn iqt_ats_round_trip_recon_exact_matrix() {
        for &cm_init in &[false, true] {
            for &btt in &[false, true] {
                for &(w, h) in &[(64u32, 64u32), (72, 40), (176, 144)] {
                    for &qp in &[10i32, 30, 44] {
                        let src = synth_picture(w, h, 0xA75 ^ (w * 31 + h * 7 + qp as u32));
                        let tools = IntraToolset {
                            deblock: qp == 30,
                            cm_init,
                            eipd: true,
                            btt,
                            iqt: true,
                            ats: true,
                            adcc: false,
                        };
                        let (payload, enc_recon, stats) =
                            encode_idr_slice_data_cfg(&src, qp, tools).expect("encode");
                        let mut walk = if btt {
                            walk_inputs_btt(w, h, cm_init)
                        } else {
                            walk_inputs_cm(w, h, cm_init)
                        };
                        walk.sps_eipd_flag = true;
                        walk.tree_gates.sps_ats_flag = true;
                        let off = crate::headers_enc::iqt_chroma_qp_offset(qp, true);
                        let dec_in = SliceDecodeInputs {
                            sps_iqt_flag: true,
                            slice_cb_qp_offset: off,
                            slice_cr_qp_offset: off,
                            enable_deblock: qp == 30,
                            ..decode_inputs(qp)
                        };
                        let (dec, dec_stats) = decode_baseline_idr_slice(&payload, walk, dec_in)
                            .unwrap_or_else(|e| panic!("{w}x{h} qp{qp} cm{cm_init} btt{btt}: {e}"));
                        assert_eq!(
                            dec.y, enc_recon.y,
                            "{w}x{h} qp{qp} cm{cm_init} btt{btt}: luma"
                        );
                        assert_eq!(
                            dec.cb, enc_recon.cb,
                            "{w}x{h} qp{qp} cm{cm_init} btt{btt}: cb"
                        );
                        assert_eq!(
                            dec.cr, enc_recon.cr,
                            "{w}x{h} qp{qp} cm{cm_init} btt{btt}: cr"
                        );
                        assert_eq!(dec_stats.coding_units, stats.leaves);
                        assert!(
                            dec_stats.ats_intra.cu_intra_flag_bins > 0,
                            "{w}x{h} qp{qp}: the ATS-intra flag is coded"
                        );
                        if qp <= 30 {
                            assert!(
                                dec_stats.ats_intra.hor_mode_bins > 0,
                                "{w}x{h} qp{qp}: an alternative kernel is chosen somewhere"
                            );
                        }
                    }
                }
            }
        }
        // ats without iqt is refused (§7.3.2.1).
        let src = synth_picture(64, 64, 1);
        assert!(encode_idr_slice_data_cfg(
            &src,
            30,
            IntraToolset {
                ats: true,
                ..IntraToolset::default()
            }
        )
        .is_err());
    }

    /// Round 455 — **EIPD**: the 33-mode search + MPM / PIMS / rem-mode
    /// syntax + `intra_chroma_pred_mode` round-trip recon-exactly through
    /// the decoder's `sps_eipd_flag == 1` walker on both entropy shapes
    /// across the size × QP matrix; the mode histogram reaches beyond
    /// the five Baseline directions, non-DM chroma modes get picked, and
    /// the EIPD stream never costs more than 5 % over the Baseline-mode
    /// stream at the same QP while beating it at at least one QP.
    #[test]
    fn eipd_round_trip_recon_exact_and_pays_off() {
        let mut wins = 0;
        for &cm_init in &[false, true] {
            for &(w, h) in &[(64u32, 64u32), (32, 32), (128, 96), (100, 60), (176, 144)] {
                for &qp in &[4i32, 22, 37, 51] {
                    let src = synth_picture(w, h, 0xC0FFEE ^ (w * 31 + h * 7 + qp as u32));
                    let (payload, enc_recon, stats) =
                        encode_idr_slice_data_full(&src, qp, false, cm_init, true).expect("encode");
                    let mut walk = walk_inputs_cm(w, h, cm_init);
                    walk.sps_eipd_flag = true;
                    let (dec, dec_stats) =
                        decode_baseline_idr_slice(&payload, walk, decode_inputs(qp))
                            .unwrap_or_else(|e| {
                                panic!("{w}x{h} qp{qp} cm{cm_init}: decode failed: {e}")
                            });
                    assert_eq!(dec.y, enc_recon.y, "{w}x{h} qp{qp} cm{cm_init}: luma recon");
                    assert_eq!(dec.cb, enc_recon.cb, "{w}x{h} qp{qp} cm{cm_init}: cb recon");
                    assert_eq!(dec.cr, enc_recon.cr, "{w}x{h} qp{qp} cm{cm_init}: cr recon");
                    assert_eq!(dec_stats.ctus, stats.ctus);
                    assert_eq!(
                        stats.eipd_mode_histogram.iter().sum::<u32>(),
                        stats.leaves,
                        "every leaf is an EIPD leaf"
                    );
                    assert_eq!(stats.mode_histogram, [0; 5]);
                    if (w, h) == (176, 144) {
                        let directional: u32 = stats.eipd_mode_histogram[5..].iter().sum();
                        assert!(
                            directional > 0,
                            "qp{qp} cm{cm_init}: {:?}",
                            stats.eipd_mode_histogram
                        );
                        let (base, _, _) =
                            encode_idr_slice_data_opts(&src, qp, false, cm_init).expect("encode");
                        assert!(
                            payload.len() * 100 <= base.len() * 105,
                            "qp{qp} cm{cm_init}: eipd {} vs baseline {}",
                            payload.len(),
                            base.len()
                        );
                        wins += usize::from(payload.len() < base.len());
                    }
                }
            }
        }
        assert!(
            wins >= 4,
            "EIPD must beat the Baseline modes on the busy frame: {wins}"
        );
    }

    /// The `sps_cm_init_flag == 1` entropy shape must never lose to the
    /// Baseline single-context collapse on the busy synthetic frame:
    /// strictly fewer payload bytes at every QP (the per-element
    /// context modelling is precisely the rate win the r429 README
    /// promised). Since round 455 the RD decisions are costed against
    /// each shape's own contexts, so the two reconstructions may
    /// legitimately differ — only the rate ordering is pinned.
    #[test]
    fn cm_init_shrinks_payload_at_every_qp() {
        let (w, h) = (128u32, 96u32);
        let src = synth_picture(w, h, 0xFEED_BEEF);
        for &qp in &[4i32, 16, 28, 40, 51] {
            let (p0, _, _) = encode_idr_slice_data_opts(&src, qp, false, false).unwrap();
            let (p1, _, _) = encode_idr_slice_data_opts(&src, qp, false, true).unwrap();
            assert!(
                p1.len() < p0.len(),
                "qp {qp}: cm_init payload {} must beat baseline {}",
                p1.len(),
                p0.len()
            );
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
            // Both entropy shapes, both colour classes: the writer must
            // be the exact dual of the reader — including the cm_init
            // PrevLevel-chain ctxInc walk over Tables 84/85/86.
            for &(cm_init, c_idx) in &[(false, 0u32), (true, 0), (true, 1)] {
                let sel = crate::cabac_init::CtxSel::new(cm_init, crate::cabac::InitType::I);
                let mut enc = CabacEncoder::new();
                if cm_init {
                    enc.init_main_profile(crate::cabac::InitType::I, 27);
                }
                emit_residual(&mut enc, sel, c_idx, &levels, 2, 2);
                enc.encode_terminate(true);
                let bytes = enc.finish();
                let mut eng = CabacEngine::new(&bytes).unwrap();
                if cm_init {
                    crate::cabac_init::init_main_profile_contexts(
                        &mut eng,
                        crate::cabac::InitType::I,
                        27,
                    )
                    .unwrap();
                }
                let mut decoded = vec![0i32; 16];
                let mut runs = 0u32;
                crate::slice_data::decode_residual_coding_rle(
                    &mut eng,
                    sel,
                    c_idx,
                    &mut decoded,
                    &mut runs,
                    2,
                    2,
                )
                .unwrap();
                assert!(eng.decode_terminate().unwrap());
                assert_eq!(decoded, levels, "cm_init {cm_init} c_idx {c_idx}");
            }
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
