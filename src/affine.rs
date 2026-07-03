//! §8.5.3 affine motion-vector derivation — the **geometric core**.
//!
//! This module implements the pure, self-contained subclauses of the
//! §8.5.3 "Derivation process for affine motion vector components and
//! reference indices" that turn a small set of **control point motion
//! vectors** (CPMVs) into a dense per-subblock motion field:
//!
//! * §8.5.3.9 [`affine_model_params`] — the horizontal change `dX`,
//!   vertical change `dY`, and scaled base vector `mvBaseScaled` derived
//!   from the 2- or 3-CPMV set (eqs. 897-906).
//! * §8.5.3.8 [`affine_subblock_size`] — the per-direction subblock size
//!   `(sizeSbX, sizeSbY)`, subblock counts `(numSbX, numSbY)`, and the
//!   `clipMV` flag (Tables 22/23 + the EIF-applicability test, eqs.
//!   879-896).
//! * §8.5.3.7 [`affine_subblock_mvs`] — the dense
//!   `mvLX[ xSbIdx ][ ySbIdx ]` luma array (1/16-pel, eqs. 873-878) plus
//!   the §8.5.2.6 chroma field `mvCLX` (eqs. 676/677).
//! * §8.5.3.1 [`reconstruct_cp_mv`] — the AMVP-path control-point
//!   reconstruction `cpMvLX = (mvpCpLX + mvdCpLX) wrapped to 18 bits`
//!   (eqs. 688-691), and [`affine_center_mv`] — the §8.5.3 center MV
//!   (eqs. 696-701) the §8.5.2.7 HMVP update consumes.
//!
//! The §8.5.3.10 motion-vector rounding the array/center derivations rely
//! on is [`crate::inter::round_motion_vector`] (already present); this
//! module wires it in at `rightShift = 5` (subblock array) and
//! `rightShift = 7` (center).
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).
//!
//! ## Resolution conventions
//!
//! The CPMVs and the derived `dX`/`dY`/`mvBaseScaled` carry **extra
//! fractional bits** relative to the stored 1/4-pel MV: `mvBaseScaled`
//! is `cpMv << 7` (eqs. 905/906), so the intermediate accumulator in
//! eqs. 875/876 lives at 1/4-pel × 2⁷ = 1/512-pel before the
//! §8.5.3.10 `rightShift = 5` reduction brings it to 1/16-pel. The
//! output `mvLX[ xSbIdx ][ ySbIdx ]` is therefore in **1/16-pel** luma
//! accuracy (the §8.5.3.1 output contract), and `mvCLX` is in
//! **1/32-pel** chroma accuracy after the §8.5.2.6 `* 2 / SubWC` map.

use crate::inter::{round_motion_vector, MotionVector};

/// A single affine control point motion vector — one of the 2 (4-param)
/// or 3 (6-param) `cpMvLX[ cpIdx ]`. Components are in 1/4-pel luma
/// accuracy (the stored MV grid), matching [`MotionVector`].
pub type ControlPointMv = MotionVector;

/// The §8.5.3.9 affine model parameters derived from the CPMV set.
///
/// `d_x` / `d_y` are the per-sample horizontal/vertical motion-vector
/// changes; `mv_base_scaled` is `cpMv[0] << 7`. All three are 2-vectors
/// `[component0, component1]` carrying the eq.-905 extra 7 fractional
/// bits (1/512-pel for the base, per-sample slope for `d_x`/`d_y`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineModelParams {
    /// `dX = [ dX[0], dX[1] ]` (eqs. 899/900).
    pub d_x: [i32; 2],
    /// `dY = [ dY[0], dY[1] ]` (eqs. 901-904).
    pub d_y: [i32; 2],
    /// `mvBaseScaled = cpMvLX[0] << 7` (eqs. 905/906).
    pub mv_base_scaled: [i32; 2],
}

/// §8.5.3.9 — derive `dX`, `dY`, and `mvBaseScaled` from the CPMV set.
///
/// `num_cp_mv` is 2 (4-parameter) or 3 (6-parameter). `cp_mv` carries
/// at least `num_cp_mv` entries; only `cp_mv[0..num_cp_mv]` are read.
///
/// `7 − log2CbW` / `7 − log2CbH` are non-negative for every affine CU
/// (affine requires `log2CbWidth >= 3` / `>= 4` and CTU width ≤ 128 ⇒
/// `log2 ≤ 7`), so the left shifts never underflow.
pub fn affine_model_params(
    cb_width: u32,
    cb_height: u32,
    num_cp_mv: u32,
    cp_mv: &[ControlPointMv],
) -> AffineModelParams {
    let log2_cb_w = cb_width.trailing_zeros();
    let log2_cb_h = cb_height.trailing_zeros();
    let sh_w = 7 - log2_cb_w;
    let sh_h = 7 - log2_cb_h;

    // eqs. 899/900.
    let d_x = [
        (cp_mv[1].x - cp_mv[0].x) << sh_w,
        (cp_mv[1].y - cp_mv[0].y) << sh_w,
    ];

    // eqs. 901-904.
    let d_y = if num_cp_mv == 3 {
        [
            (cp_mv[2].x - cp_mv[0].x) << sh_h,
            (cp_mv[2].y - cp_mv[0].y) << sh_h,
        ]
    } else {
        // numCpMv == 2: dY[0] = −dX[1], dY[1] = dX[0].
        [-d_x[1], d_x[0]]
    };

    // eqs. 905/906.
    let mv_base_scaled = [cp_mv[0].x << 7, cp_mv[0].y << 7];

    AffineModelParams {
        d_x,
        d_y,
        mv_base_scaled,
    }
}

/// The §8.5.3.8 subblock-geometry result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineSubblockGeometry {
    /// `sizeSbX` — horizontal subblock size in luma samples.
    pub size_sb_x: u32,
    /// `sizeSbY` — vertical subblock size in luma samples.
    pub size_sb_y: u32,
    /// `numSbX = cbWidth / sizeSbX` (eq. 895).
    pub num_sb_x: u32,
    /// `numSbY = cbHeight / sizeSbY` (eq. 896).
    pub num_sb_y: u32,
    /// `clipMV` — motion-vector clipping type for EIF blocks (eq. 892).
    pub clip_mv: bool,
}

/// Table 22 — `sizeSbXTemp` as a function of `mvWx` (0/1/2/3/4/>4). The
/// `mvWx == 0` entry is `cbWidth`; the caller supplies it.
fn size_sb_temp(mv_w: i32, cb_dim: u32) -> u32 {
    match mv_w {
        0 => cb_dim,
        1 => 32,
        2 => 16,
        3 => 8,
        4 => 8,
        _ => 4, // > 4
    }
}

/// §8.5.3.8 — derive subblock size, subblock counts, and `clipMV`.
///
/// `pred_flag` is `[ predFlagL0, predFlagL1 ]`; `params` carries the
/// §8.5.3.9 outputs **per list** (only the entries whose `pred_flag` is
/// set are read). The `eifCanBeApplied` / `clipMV` accumulation folds
/// over both active lists exactly as eqs. 891/892.
pub fn affine_subblock_size(
    cb_width: u32,
    cb_height: u32,
    pred_flag: [bool; 2],
    params: [AffineModelParams; 2],
) -> AffineSubblockGeometry {
    // sizeSb{X,Y} start at the full block (the per-list Min folds shrink
    // them); eq. 879/880 Min over active lists.
    let mut size_sb_x = cb_width;
    let mut size_sb_y = cb_height;

    let mut eif_can_be_applied = true;
    let mut clip_mv = false;

    const EIF_SUBBLOCK_SIZE: i32 = 4;

    for x in 0..2usize {
        if !pred_flag[x] {
            continue;
        }
        let d_x = params[x].d_x;
        let d_y = params[x].d_y;

        // eqs. 879/880.
        let mv_wx = d_x[0].abs().max(d_x[1].abs());
        let mv_wy = d_y[0].abs().max(d_y[1].abs());

        // Tables 22/23 + eqs. 881/882.
        size_sb_x = size_sb_x.min(size_sb_temp(mv_wx, cb_width));
        size_sb_y = size_sb_y.min(size_sb_temp(mv_wy, cb_height));

        // eqs. 883-890 — the EIF bounding-box arrays. X[3]/Y[3] are the
        // sum of the [1]/[2] entries (eqs. 886/890).
        let x1 = (EIF_SUBBLOCK_SIZE + 1) * (d_x[0] + (1 << 9));
        let x2 = (EIF_SUBBLOCK_SIZE + 1) * d_y[0];
        let xa = [0, x1, x2, x1 + x2];
        let y1 = (EIF_SUBBLOCK_SIZE + 1) * d_x[1];
        let y2 = (EIF_SUBBLOCK_SIZE + 1) * (d_y[1] + (1 << 9));
        let ya = [0, y1, y2, y1 + y2];

        let x_max = *xa.iter().max().unwrap();
        let x_min = *xa.iter().min().unwrap();
        let y_max = *ya.iter().max().unwrap();
        let y_min = *ya.iter().min().unwrap();

        let w = (x_max - x_min + (1 << 9) - 1) >> 9;
        let h = (y_max - y_min + (1 << 9) - 1) >> 9;

        // eq.-after-890: clipMVX = TRUE when (W+2)*(H+2) > 72.
        let clip_mv_x = (w + 2) * (h + 2) > 72;

        // eifCanBeAppliedX test: EIF is disqualified when dY[1] is below
        // −512, OR (otherwise) when the §8.5.3.8 slope-magnitude bound is
        // exceeded. Both spec branches set the flag FALSE, so they fold
        // into one disjunction (the `< −512` short-circuits the second
        // term's evaluation exactly as the spec's `else if`).
        let eif_can_be_applied_x = !(d_y[1] < ((-1) << 9)
            || (d_y[1].max(0) + d_x[1].abs()) * (1 + EIF_SUBBLOCK_SIZE) > (1 << 9));

        // eqs. 891/892 fold.
        eif_can_be_applied &= eif_can_be_applied_x;
        clip_mv |= clip_mv_x;
    }

    // eqs. 893/894: when EIF cannot be applied, floor the subblock size
    // to 8.
    if !eif_can_be_applied {
        size_sb_x = size_sb_x.max(8);
        size_sb_y = size_sb_y.max(8);
    }

    // eqs. 895/896.
    let num_sb_x = cb_width / size_sb_x;
    let num_sb_y = cb_height / size_sb_y;

    AffineSubblockGeometry {
        size_sb_x,
        size_sb_y,
        num_sb_x,
        num_sb_y,
        clip_mv,
    }
}

/// §8.5.2.6 — derive a chroma motion vector from a luma motion vector
/// (eqs. 676/677): `mvCLX = mvLX * 2 / SubWC`. For 4:2:0
/// (`SubWidthC = SubHeightC = 2`) this is the identity; for 4:2:2 the Y
/// axis halves; for 4:4:4 both axes double. The `* 2` lifts the result
/// to the 1/32-pel chroma grid (the §8.5.3.1 output contract).
///
/// Integer division truncates toward zero (the spec's `/`).
pub fn chroma_motion_vector(mv: MotionVector, sub_width_c: i32, sub_height_c: i32) -> MotionVector {
    MotionVector {
        x: (mv.x * 2) / sub_width_c,
        y: (mv.y * 2) / sub_height_c,
    }
}

/// One subblock's derived luma + chroma motion vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SubblockMv {
    /// `mvLX[ xSbIdx ][ ySbIdx ]` in 1/16-pel luma accuracy.
    pub luma: MotionVector,
    /// `mvCLX[ xSbIdx ][ ySbIdx ]` in 1/32-pel chroma accuracy.
    pub chroma: MotionVector,
}

/// The dense per-list affine subblock motion field: a `num_sb_y`-row,
/// `num_sb_x`-column array stored row-major (`[ySbIdx * num_sb_x +
/// xSbIdx]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineMvField {
    pub num_sb_x: u32,
    pub num_sb_y: u32,
    pub size_sb_x: u32,
    pub size_sb_y: u32,
    pub mvs: Vec<SubblockMv>,
}

impl AffineMvField {
    /// Fetch the subblock at grid index `(x_sb_idx, y_sb_idx)`.
    pub fn at(&self, x_sb_idx: u32, y_sb_idx: u32) -> SubblockMv {
        self.mvs[(y_sb_idx * self.num_sb_x + x_sb_idx) as usize]
    }
}

/// §8.5.3.7 — derive the dense subblock motion-vector array from the CPMV
/// set for one prediction list.
///
/// Runs §8.5.3.9 + §8.5.3.8 internally to obtain `dX`/`dY`/`mvBaseScaled`
/// and the subblock geometry, then for each `(xSbIdx, ySbIdx)` evaluates
/// eqs. 873-876 at the subblock centre, applies the §8.5.3.10 rounding
/// (`rightShift = 5`), clips to `[−2¹⁷, 2¹⁷ − 1]` (eqs. 877/878), and
/// derives the §8.5.2.6 chroma vector.
///
/// `sub_width_c` / `sub_height_c` are the chroma sampling factors
/// (`2, 2` for 4:2:0).
pub fn affine_subblock_mvs(
    cb_width: u32,
    cb_height: u32,
    num_cp_mv: u32,
    cp_mv: &[ControlPointMv],
    sub_width_c: i32,
    sub_height_c: i32,
) -> AffineMvField {
    let params = affine_model_params(cb_width, cb_height, num_cp_mv, cp_mv);
    // Single-list geometry: the caller invokes once per active list, so
    // only this list's params drive the size derivation.
    let geom = affine_subblock_size(cb_width, cb_height, [true, false], [params, params]);

    let d_x = params.d_x;
    let d_y = params.d_y;
    let base = params.mv_base_scaled;

    const MV_CLIP: i32 = 1 << 17;

    let mut mvs = Vec::with_capacity((geom.num_sb_x * geom.num_sb_y) as usize);
    for y_sb_idx in 0..geom.num_sb_y {
        for x_sb_idx in 0..geom.num_sb_x {
            // eqs. 873/874 — subblock-centre position.
            let x_pos_sb = (geom.size_sb_x * x_sb_idx + (geom.size_sb_x >> 1)) as i32;
            let y_pos_sb = (geom.size_sb_y * y_sb_idx + (geom.size_sb_y >> 1)) as i32;

            // eqs. 875/876 — accumulate the affine model.
            let mv = MotionVector {
                x: base[0] + d_x[0] * x_pos_sb + d_y[0] * y_pos_sb,
                y: base[1] + d_x[1] * x_pos_sb + d_y[1] * y_pos_sb,
            };

            // eqs. (8.5.3.10) rightShift = 5, leftShift = 0.
            let rounded = round_motion_vector(mv, 5, 0);

            // eqs. 877/878 — clip to 18-bit signed.
            let luma = MotionVector {
                x: rounded.x.clamp(-MV_CLIP, MV_CLIP - 1),
                y: rounded.y.clamp(-MV_CLIP, MV_CLIP - 1),
            };

            // §8.5.2.6 chroma MV.
            let chroma = chroma_motion_vector(luma, sub_width_c, sub_height_c);

            mvs.push(SubblockMv { luma, chroma });
        }
    }

    AffineMvField {
        num_sb_x: geom.num_sb_x,
        num_sb_y: geom.num_sb_y,
        size_sb_x: geom.size_sb_x,
        size_sb_y: geom.size_sb_y,
        mvs,
    }
}

/// §8.5.3.1 step-5 — reconstruct one control-point MV from its predictor
/// and decoded MVD on the AMVP (non-merge) affine path (eqs. 688-691).
///
/// `uLX = (mvpCp + mvdCp + 2¹⁶) % 2¹⁶`; then `cpMv = (uLX >= 2¹⁵) ? uLX −
/// 2¹⁶ : uLX`. This is the signed-16-bit modular wrap of `mvp + mvd`,
/// identical in form to the non-affine §8.5.2 reconstruction
/// ([`MotionVector::wrapping_add`]). Both components are wrapped
/// independently.
pub fn reconstruct_cp_mv(mvp_cp: ControlPointMv, mvd_cp: MotionVector) -> ControlPointMv {
    mvp_cp.wrapping_add(&mvd_cp)
}

/// §8.5.3.1 eqs. 696-701 — the affine **center** motion vector for the
/// §8.5.2.7 HMVP-list update.
///
/// Evaluates the affine model at `(cbWidth >> 1, cbHeight >> 1)` (eqs.
/// 696-699), rounds with `rightShift = 7` (eq. after 699 → §8.5.3.10),
/// then clips to `[−2¹⁵, 2¹⁵ − 1]` (eqs. 700/701). Output is a 1/4-pel
/// luma MV (the HMVP list's accuracy).
pub fn affine_center_mv(
    cb_width: u32,
    cb_height: u32,
    num_cp_mv: u32,
    cp_mv: &[ControlPointMv],
) -> MotionVector {
    let params = affine_model_params(cb_width, cb_height, num_cp_mv, cp_mv);
    let x_pos_sb = (cb_width >> 1) as i32;
    let y_pos_sb = (cb_height >> 1) as i32;

    // eqs. 698/699.
    let mv = MotionVector {
        x: params.mv_base_scaled[0] + params.d_x[0] * x_pos_sb + params.d_y[0] * y_pos_sb,
        y: params.mv_base_scaled[1] + params.d_x[1] * x_pos_sb + params.d_y[1] * y_pos_sb,
    };

    // §8.5.3.10 rightShift = 7, leftShift = 0.
    let rounded = round_motion_vector(mv, 7, 0);

    // eqs. 700/701 — Clip3(−2¹⁵, 2¹⁵ − 1, ·).
    const C: i32 = 1 << 15;
    MotionVector {
        x: rounded.x.clamp(-C, C - 1),
        y: rounded.y.clamp(-C, C - 1),
    }
}

/// The §8.5.3.3 input: the affine-coded neighbour block's stored corner
/// motion vectors (for one prediction list X) plus its geometry and
/// motion model.
///
/// `MvLX` is sampled at three neighbour corners (eqs. 744-753):
///
/// * `mv_tl` ⇐ `MvLX[ xNb ][ yNb ]` (top-left).
/// * `mv_tr` ⇐ `MvLX[ xNb + nNbW − 1 ][ yNb ]` (top-right).
/// * `mv_bl` ⇐ `MvLX[ xNb ][ yNb + nNbH − 1 ]` (bottom-left).
///
/// When the §8.5.3.3 `isCTUboundary` path is taken (the neighbour lies in
/// the CTU row immediately above), the spec instead samples the
/// neighbour's **bottom** edge — `MvLX[ xNb ][ yNb + nNbH − 1 ]` and
/// `MvLX[ xNb + nNbW − 1 ][ yNb + nNbH − 1 ]`. The caller supplies the
/// extra bottom-right corner in `mv_br` so this module can pick the
/// correct row without re-reading the motion grid.
///
/// `motion_model_idc` is the neighbour's `MotionModelIdc[ xNb ][ yNb ]`
/// (1 = 4-parameter, 2 = 6-parameter), selecting whether the genuine
/// `dHorY`/`dVerY` (eqs. 752/753) or the 4-parameter rotation identity
/// (eqs. 754/755) is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct NeighbourAffineSource {
    /// `( xNb, yNb )` — neighbour top-left, picture-relative.
    pub x_nb: i32,
    pub y_nb: i32,
    /// `nNbW` / `nNbH` — neighbour block dimensions (powers of two).
    pub n_nb_w: u32,
    pub n_nb_h: u32,
    /// `MvLX[ xNb ][ yNb ]`.
    pub mv_tl: MotionVector,
    /// `MvLX[ xNb + nNbW − 1 ][ yNb ]`.
    pub mv_tr: MotionVector,
    /// `MvLX[ xNb ][ yNb + nNbH − 1 ]`.
    pub mv_bl: MotionVector,
    /// `MvLX[ xNb + nNbW − 1 ][ yNb + nNbH − 1 ]`.
    pub mv_br: MotionVector,
    /// `MotionModelIdc[ xNb ][ yNb ]` (1 or 2).
    pub motion_model_idc: u32,
}

/// §8.5.3.3 — derive the current block's affine control point motion
/// vectors **inherited** from an affine-coded neighbour.
///
/// Picks the §8.5.3.3 `isCTUboundary` row (eqs. 744-751), derives the
/// model gradients (`mvScaleHor/Ver`, `dHorX`, `dVerX`, and the
/// `dHorY`/`dVerY` model-idc split of eqs. 752-755), projects them onto
/// the current block's `numCpMv` control points (eqs. 756-761), then
/// applies the §8.5.3.10 rounding (`rightShift = 7`) and the eq.-762/763
/// clip. Returns `cpMvLX[ 0..numCpMv ]`.
///
/// `ctb_size_y` is the luma CTB size in samples (for the `isCTUboundary`
/// test). `( x_cb, y_cb )` / `cb_width` / `cb_height` are the current
/// block.
pub fn inherited_cp_mvs(
    x_cb: i32,
    y_cb: i32,
    cb_width: u32,
    cb_height: u32,
    num_cp_mv: u32,
    ctb_size_y: i32,
    src: NeighbourAffineSource,
) -> Vec<ControlPointMv> {
    let n_nb_h = src.n_nb_h as i32;
    // isCTUboundary: the neighbour's bottom edge is on a CTB boundary AND
    // is exactly the current block's top edge.
    let is_ctu_boundary = ((src.y_nb + n_nb_h) % ctb_size_y == 0) && (src.y_nb + n_nb_h == y_cb);

    let log2_nb_w = src.n_nb_w.trailing_zeros();
    let log2_nb_h = src.n_nb_h.trailing_zeros();
    let sh_w = 7 - log2_nb_w;
    let sh_h = 7 - log2_nb_h;

    // eqs. 744-751 — mvScaleHor/Ver, dHorX, dVerX. The isCTUboundary
    // branch samples the neighbour's bottom edge (mv_bl / mv_br); the
    // ordinary branch samples its top edge (mv_tl / mv_tr).
    let (base, right) = if is_ctu_boundary {
        (src.mv_bl, src.mv_br)
    } else {
        (src.mv_tl, src.mv_tr)
    };
    let mv_scale_hor = base.x << 7;
    let mv_scale_ver = base.y << 7;
    let d_hor_x = (right.x - base.x) << sh_w;
    let d_ver_x = (right.y - base.y) << sh_w;

    // eqs. 752-755 — dHorY / dVerY.
    let (d_hor_y, d_ver_y) = if !is_ctu_boundary && src.motion_model_idc == 2 {
        // 6-parameter genuine vertical gradient (eqs. 752/753): bottom-left
        // minus top-left, scaled by 7 − log2NbH.
        (
            (src.mv_bl.x - src.mv_tl.x) << sh_h,
            (src.mv_bl.y - src.mv_tl.y) << sh_h,
        )
    } else {
        // 4-parameter (or CTU-boundary) rotation identity (eqs. 754/755).
        (-d_ver_x, d_hor_x)
    };

    let dx = x_cb - src.x_nb;
    let dy = y_cb - src.y_nb;
    let cb_w = cb_width as i32;
    let cb_h = cb_height as i32;

    // eqs. 756-761 — project onto the current control points.
    let mut cps = Vec::with_capacity(num_cp_mv as usize);
    // cpMvLX[0] (eqs. 756/757).
    cps.push(MotionVector {
        x: mv_scale_hor + d_hor_x * dx + d_hor_y * dy,
        y: mv_scale_ver + d_ver_x * dx + d_ver_y * dy,
    });
    // cpMvLX[1] (eqs. 758/759): + cbWidth in the horizontal projection.
    cps.push(MotionVector {
        x: mv_scale_hor + d_hor_x * (dx + cb_w) + d_hor_y * dy,
        y: mv_scale_ver + d_ver_x * (dx + cb_w) + d_ver_y * dy,
    });
    if num_cp_mv == 3 {
        // cpMvLX[2] (eqs. 760/761): + cbHeight in the vertical projection.
        cps.push(MotionVector {
            x: mv_scale_hor + d_hor_x * dx + d_hor_y * (dy + cb_h),
            y: mv_scale_ver + d_ver_x * dx + d_ver_y * (dy + cb_h),
        });
    }

    // eqs. 762/763 — §8.5.3.10 rightShift=7 rounding, then Clip3.
    const C: i32 = 1 << 15;
    for cp in &mut cps {
        let r = round_motion_vector(*cp, 7, 0);
        cp.x = r.x.clamp(-C, C - 1);
        cp.y = r.y.clamp(-C, C - 1);
    }
    cps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(x: i32, y: i32) -> ControlPointMv {
        MotionVector::quarter_pel(x, y)
    }

    /// §8.5.3.9 — a translational (all-equal) CPMV set yields zero slope
    /// and a base equal to `cpMv[0] << 7`. The 4-param dY identity
    /// (dY[0] = −dX[1], dY[1] = dX[0]) collapses to zero too.
    #[test]
    fn model_params_translational_is_zero_slope() {
        let cps = [cp(40, -24), cp(40, -24)];
        let p = affine_model_params(16, 16, 2, &cps);
        assert_eq!(p.d_x, [0, 0]);
        assert_eq!(p.d_y, [0, 0]);
        assert_eq!(p.mv_base_scaled, [40 << 7, -24 << 7]);
    }

    /// §8.5.3.9 — 4-parameter slope and the dY rotation identity.
    #[test]
    fn model_params_4param_slope() {
        // cbWidth 16 → log2 = 4 → shift = 3. cpMv[1].x − cpMv[0].x = 8.
        let cps = [cp(0, 0), cp(8, 4)];
        let p = affine_model_params(16, 16, 2, &cps);
        assert_eq!(p.d_x, [8 << 3, 4 << 3]); // [64, 32]
                                             // dY[0] = −dX[1] = −32, dY[1] = dX[0] = 64.
        assert_eq!(p.d_y, [-32, 64]);
        assert_eq!(p.mv_base_scaled, [0, 0]);
    }

    /// §8.5.3.9 — 6-parameter dY uses cpMv[2] and the height shift.
    #[test]
    fn model_params_6param_uses_third_cp() {
        // cbHeight 8 → log2 = 3 → shift = 4. cpMv[2] − cpMv[0] = (0, 16).
        let cps = [cp(0, 0), cp(16, 0), cp(0, 16)];
        let p = affine_model_params(16, 8, 3, &cps);
        // dX: width 16 → shift 3. (16−0)<<3 = 128, (0−0)<<3 = 0.
        assert_eq!(p.d_x, [128, 0]);
        // dY: (0−0)<<4 = 0, (16−0)<<4 = 256.
        assert_eq!(p.d_y, [0, 256]);
    }

    /// §8.5.3.8 — a translational model (zero slope) keeps the full block
    /// as a single subblock: Table-22/23 `mvW == 0` ⇒ size = cb dim.
    #[test]
    fn subblock_size_translational_single_block() {
        let p = affine_model_params(16, 16, 2, &[cp(10, 10), cp(10, 10)]);
        let g = affine_subblock_size(16, 16, [true, false], [p, p]);
        assert_eq!((g.size_sb_x, g.size_sb_y), (16, 16));
        assert_eq!((g.num_sb_x, g.num_sb_y), (1, 1));
    }

    /// §8.5.3.8 — a steep slope shrinks the subblock toward the 4-sample
    /// floor (or 8 when EIF cannot be applied).
    #[test]
    fn subblock_size_steep_slope_shrinks() {
        // Large CPMV delta → large dX → mvWx > 4 → Table-22 size 4.
        let cps = [cp(0, 0), cp(2000, 0)];
        let p = affine_model_params(16, 16, 2, &cps);
        let g = affine_subblock_size(16, 16, [true, false], [p, p]);
        // size shrinks well below the full 16; numSb grows accordingly.
        assert!(g.size_sb_x <= 16 && g.size_sb_y <= 16);
        assert_eq!(g.num_sb_x, 16 / g.size_sb_x);
        assert_eq!(g.num_sb_y, 16 / g.size_sb_y);
    }

    /// §8.5.3.7 — a translational CPMV set produces a uniform field equal
    /// to the (1/16-pel) base vector across every subblock.
    #[test]
    fn subblock_mvs_translational_uniform() {
        let cps = [cp(12, -8), cp(12, -8)];
        let field = affine_subblock_mvs(16, 16, 2, &cps, 2, 2);
        // Single subblock (zero slope). Base = 12<<7 = 1536; centre adds
        // zero slope; round rightShift=5 → 1536 → (1536 + 16 − 1)>>5 = 48.
        // 48 == 12<<2 == 12 in 1/16-pel. Likewise y: −8<<2 = −32.
        assert_eq!(field.num_sb_x * field.num_sb_y, 1);
        let sb = field.at(0, 0);
        assert_eq!(sb.luma, MotionVector::quarter_pel(48, -32));
        // 4:2:0 chroma: mvC = mv * 2 / 2 = mv.
        assert_eq!(sb.chroma, MotionVector::quarter_pel(48, -32));
    }

    /// §8.5.3.7 — a non-translational field varies across subblocks and
    /// every entry stays inside the 18-bit clip.
    #[test]
    fn subblock_mvs_vary_and_clip() {
        let cps = [cp(0, 0), cp(64, 0), cp(0, 64)];
        let field = affine_subblock_mvs(16, 16, 3, &cps, 2, 2);
        assert!(field.num_sb_x >= 1 && field.num_sb_y >= 1);
        let first = field.at(0, 0);
        let last = field.at(field.num_sb_x - 1, field.num_sb_y - 1);
        // The field is not uniform for a rotational/zoom model.
        assert_ne!(first.luma, last.luma);
        for sb in &field.mvs {
            assert!(sb.luma.x >= -(1 << 17) && sb.luma.x < (1 << 17));
            assert!(sb.luma.y >= -(1 << 17) && sb.luma.y < (1 << 17));
        }
    }

    /// §8.5.2.6 chroma MV map across the three subsampling modes.
    #[test]
    fn chroma_mv_subsampling() {
        let mv = MotionVector::quarter_pel(20, -12);
        // 4:2:0: *2/2 = identity.
        assert_eq!(
            chroma_motion_vector(mv, 2, 2),
            MotionVector::quarter_pel(20, -12)
        );
        // 4:2:2: x *2/2, y *2/1 doubles y.
        assert_eq!(
            chroma_motion_vector(mv, 2, 1),
            MotionVector::quarter_pel(20, -24)
        );
        // 4:4:4: both double.
        assert_eq!(
            chroma_motion_vector(mv, 1, 1),
            MotionVector::quarter_pel(40, -24)
        );
    }

    /// §8.5.3.1 eqs. 688-691 — CP MV reconstruction is the 16-bit modular
    /// wrap of mvp + mvd.
    #[test]
    fn cp_mv_reconstruction_wraps() {
        let mvp = cp(100, -50);
        let mvd = cp(20, 10);
        assert_eq!(reconstruct_cp_mv(mvp, mvd), cp(120, -40));
        // Wrap at the signed-16-bit boundary.
        let big = reconstruct_cp_mv(cp(32760, 0), cp(20, 0));
        // 32760 + 20 = 32780 = 0x800C → −32756.
        assert_eq!(big.x, -32756);
    }

    /// §8.5.3.1 eqs. 696-701 — the affine center MV of a translational
    /// model equals the (1/4-pel) CPMV; rightShift = 7 brings the
    /// `cpMv << 7` base back to 1/4-pel.
    #[test]
    fn center_mv_translational_recovers_cpmv() {
        let cps = [cp(36, -20), cp(36, -20)];
        let c = affine_center_mv(16, 16, 2, &cps);
        assert_eq!(c, MotionVector::quarter_pel(36, -20));
    }

    /// §8.5.3.1 — the center MV stays inside the eq.-700/701 clip even for
    /// a large model.
    #[test]
    fn center_mv_clips_to_15_bit() {
        let cps = [cp(20000, 20000), cp(30000, -30000)];
        let c = affine_center_mv(32, 32, 2, &cps);
        assert!(c.x >= -(1 << 15) && c.x < (1 << 15));
        assert!(c.y >= -(1 << 15) && c.y < (1 << 15));
    }

    /// §8.5.3.3 — a purely translational neighbour (all four corner MVs
    /// equal) inherits as the same translational CPMV at every control
    /// point: zero gradients ⇒ cpMv = mvScale >> 7 = the corner MV.
    #[test]
    fn inherited_translational_neighbour() {
        let t = cp(24, -16);
        let src = NeighbourAffineSource {
            x_nb: 0,
            y_nb: 0,
            n_nb_w: 16,
            n_nb_h: 16,
            mv_tl: t,
            mv_tr: t,
            mv_bl: t,
            mv_br: t,
            motion_model_idc: 1,
        };
        // Current block to the right of the neighbour, same row (not a
        // CTU boundary), 4-param.
        let cps = inherited_cp_mvs(16, 0, 16, 16, 2, 64, src);
        assert_eq!(cps.len(), 2);
        for c in cps {
            assert_eq!(c, cp(24, -16));
        }
    }

    /// §8.5.3.3 — a 6-parameter neighbour with distinct corners produces
    /// distinct projected control points (genuine dHorY/dVerY path), and
    /// every output stays inside the eq.-762/763 clip.
    #[test]
    fn inherited_6param_distinct_cps() {
        let src = NeighbourAffineSource {
            x_nb: 0,
            y_nb: 0,
            n_nb_w: 16,
            n_nb_h: 16,
            mv_tl: cp(0, 0),
            mv_tr: cp(16, 0),
            mv_bl: cp(0, 16),
            mv_br: cp(16, 16),
            motion_model_idc: 2,
        };
        let cps = inherited_cp_mvs(16, 16, 16, 16, 3, 64, src);
        assert_eq!(cps.len(), 3);
        // Not all equal — the model has genuine gradients.
        assert!(!(cps[0] == cps[1] && cps[1] == cps[2]));
        for c in &cps {
            assert!(c.x >= -(1 << 15) && c.x < (1 << 15));
            assert!(c.y >= -(1 << 15) && c.y < (1 << 15));
        }
    }

    /// §8.5.3.3 — the CTU-boundary path samples the neighbour's bottom
    /// edge (mv_bl / mv_br) rather than its top edge. With a neighbour
    /// whose top edge differs from its bottom edge, the inherited base
    /// follows the bottom-left corner.
    #[test]
    fn inherited_ctu_boundary_uses_bottom_edge() {
        // Neighbour occupies y in [0,16); current block at y = 16, so the
        // neighbour's bottom edge (yNb + nNbH = 16) == yCb and lands on a
        // 16-aligned CTB boundary (ctb_size_y = 16).
        let src = NeighbourAffineSource {
            x_nb: 0,
            y_nb: 0,
            n_nb_w: 16,
            n_nb_h: 16,
            mv_tl: cp(100, 100), // top edge — must NOT be the base
            mv_tr: cp(100, 100),
            mv_bl: cp(8, -8), // bottom edge — the CTU-boundary base
            mv_br: cp(8, -8),
            motion_model_idc: 2,
        };
        let cps = inherited_cp_mvs(0, 16, 16, 16, 2, 16, src);
        // Bottom edge is translational (bl == br) → zero horizontal
        // gradient → cpMv tracks the bottom-left corner, not the top.
        for c in cps {
            assert_eq!(c, cp(8, -8));
        }
    }
}
