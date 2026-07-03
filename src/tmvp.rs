//! §8.5.2.3.3–§8.5.2.3.5 temporal (collocated) merge-candidate derivation.
//!
//! This module derives the **temporal motion-vector predictor (TMVP)**
//! merge candidate from the motion field of the collocated picture
//! (`ColPic`). It is the §8.5.2.3.3 step-2 sub-derivation of the
//! [`crate::merge`] ADMVP merge-candidate-list assembly: the
//! [`build_merge_cand_list`](crate::merge::build_merge_cand_list)
//! `temporal: Option<MergeCand>` slot is exactly the output of this
//! module.
//!
//! The derivation is three nested clauses:
//!
//! * §8.5.2.3.4 [`tmvp_collocated_mv`] — given a single collocated
//!   sample's stored motion (`predFlagL{0,1}Col`, `mvL{0,1}Col`,
//!   `refIdxL{0,1}Col`) plus the four POC distances, produce the scaled
//!   `mvpLXCol` (eqs. 503/504) and the joint `availableFlagCol` code
//!   (0 / 1 / 2 / 3 per the eq.-after-505 table).
//! * §8.5.2.3.5 [`constrain_scaled_mv`] — clip the scaled vector against
//!   the padded picture boundary (eqs. 506-511).
//! * §8.5.2.3.3 [`tmvp_merge_candidate`] — try the **central**,
//!   **bottom**, then **side** collocated positions in order (eqs.
//!   485-500), 8×8-grid-quantised, stopping at the first that yields a
//!   non-zero `availableFlagCol`, and emit the resulting [`MergeCand`].
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).
//!
//! ## Purity & the collocated-lookup contract
//!
//! Like [`crate::merge::spatial_merge_candidates`], this module is a
//! pure function parameterised on a caller-supplied closure that
//! resolves an 8×8-grid-aligned collocated luma location to the
//! [`CollocatedMv`] stored there in `ColPic`. The decoder's
//! reference-picture motion store wires the closure in; the derivation
//! itself never touches the DPB.

use crate::inter::{MergeCand, MotionVector};
use crate::neighbour::AvailLr;

/// The motion state stored at one collocated luma sample of `ColPic`:
/// the §8.5.2.3.4 `predFlagLXCol` / `mvLXCol` / `refIdxLXCol` arrays
/// indexed at `( xColCb, yColCb )`.
///
/// `refIdxLXCol == -1` (or `pred_flag_lX == false`) marks list X
/// "invalid" — the §8.5.2.3.4 "If refIdxLXCol[…] is invalid" branch.
/// When both lists are invalid the position contributes nothing and
/// `availableFlagCol` is 0.
///
/// The stored MV is `MvDmvrL{0,1}` of `ColPic` in 1/4-pel accuracy (the
/// §8.5.2.3.4 input arrays explicitly read `MvDmvr`, the
/// decoder-side-motion-refined field; for the Baseline / no-DMVR path
/// it equals the plain `MvLX`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollocatedMv {
    /// `predFlagL0Col[ xColCb ][ yColCb ]`.
    pub pred_flag_l0: bool,
    /// `predFlagL1Col[ xColCb ][ yColCb ]`.
    pub pred_flag_l1: bool,
    /// `refIdxL0Col[ xColCb ][ yColCb ]` (–1 ⇒ invalid).
    pub ref_idx_l0: i32,
    /// `refIdxL1Col[ xColCb ][ yColCb ]` (–1 ⇒ invalid).
    pub ref_idx_l1: i32,
    /// `mvL0Col[ xColCb ][ yColCb ]` (1/4-pel).
    pub mv_l0: MotionVector,
    /// `mvL1Col[ xColCb ][ yColCb ]` (1/4-pel).
    pub mv_l1: MotionVector,
}

/// The POC-distance context the §8.5.2.3.4 scaling (eqs. 501-504) needs.
///
/// All four are signed `DiffPicOrderCnt` outputs: positive when the
/// second argument precedes the first in output order.
///
/// * `curr_poc_diff_l0 = DiffPicOrderCnt( currPic, RefPicList0[ 0 ] )`
///   (eq. 501, `refIdxLX == 0`).
/// * `curr_poc_diff_l1 = DiffPicOrderCnt( currPic, RefPicList1[ 0 ] )`.
/// * `col_poc_diff_l0 = DiffPicOrderCnt( ColPic, refPicOfColPic[ 0 ] )`
///   (eq. 502).
/// * `col_poc_diff_l1 = DiffPicOrderCnt( ColPic, refPicOfColPic[ 1 ] )`.
///
/// `refIdxLXCol` (always 0, per the §8.5.2.3.3 `refIdxLXCol = 0`
/// assignment) selects which `RefPicListX[ 0 ]` the current-picture
/// distance refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PocContext {
    pub curr_poc_diff_l0: i32,
    pub curr_poc_diff_l1: i32,
    pub col_poc_diff_l0: i32,
    pub col_poc_diff_l1: i32,
}

/// §8.6.2 eq. 165 — `DiffPicOrderCnt( picA, picB ) = PicOrderCnt( picA )
/// − PicOrderCnt( picB )`. The fundamental signed picture-order-count
/// distance every §8.5.2.3.4 / §8.5.2.3.9 scaling derivation is built on.
#[inline]
pub fn diff_pic_order_cnt(poc_a: i32, poc_b: i32) -> i32 {
    poc_a - poc_b
}

/// The per-picture POC inputs the §8.5.2.3.3 caller resolves once per CU
/// before any collocated lookup, plus the per-collocated-cell referenced
/// POCs the §8.5.2.3.4 `colPocDiff` (eq. 502) needs.
///
/// This is the decoder-side wiring contract for the §8.5.2.3.3 POC
/// distances: the current picture's POC, the POCs of `RefPicList0[ 0 ]`
/// and `RefPicList1[ 0 ]` (the `refIdxLX == 0` references eq. 501 scales
/// against), and `ColPic`'s own POC. The `refPicOfColPic[ X ]` POCs —
/// which depend on the **per-cell** stored reference — are supplied at
/// [`PocInputs::derive`] time, not here, because they vary per collocated
/// sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PocInputs {
    /// `PicOrderCnt( currPic )`.
    pub curr_poc: i32,
    /// `PicOrderCnt( RefPicList0[ 0 ] )` (the eq. 501 `refIdxL0 == 0`
    /// reference).
    pub ref_l0_poc: i32,
    /// `PicOrderCnt( RefPicList1[ 0 ] )`. Ignored on a P-slice (no L1).
    pub ref_l1_poc: i32,
    /// `PicOrderCnt( ColPic )` — the collocated picture's own POC.
    pub col_pic_poc: i32,
}

impl PocInputs {
    /// §8.5.2.3.3 eq. 501 — the current-picture POC distances, resolved
    /// once per CU (independent of the collocated cell):
    /// `currPocDiffLX = DiffPicOrderCnt( currPic, RefPicListX[ 0 ] )`.
    #[inline]
    pub fn curr_poc_diff_l0(&self) -> i32 {
        diff_pic_order_cnt(self.curr_poc, self.ref_l0_poc)
    }

    /// `currPocDiffL1 = DiffPicOrderCnt( currPic, RefPicList1[ 0 ] )`.
    #[inline]
    pub fn curr_poc_diff_l1(&self) -> i32 {
        diff_pic_order_cnt(self.curr_poc, self.ref_l1_poc)
    }

    /// Build the full [`PocContext`] for one collocated cell, given the
    /// POCs of `refPicOfColPic[ 0 ]` / `refPicOfColPic[ 1 ]` — the
    /// pictures `ColPic`'s stored L0 / L1 motion at this cell referenced
    /// (§8.5.2.3.4: `refPicOfColPic[ X ]` is the picture with reference
    /// index `refIdxCol[ X ]` in `ColPic`'s list `listCol[ X ]`).
    ///
    /// `colPocDiffLX = DiffPicOrderCnt( ColPic, refPicOfColPic[ X ] )`
    /// (eq. 502). The current-picture distances are the per-CU eq.-501
    /// values; together they form the eq.-503 `distScaleFactorLX` ratio.
    #[inline]
    pub fn derive(&self, col_ref_l0_poc: i32, col_ref_l1_poc: i32) -> PocContext {
        PocContext {
            curr_poc_diff_l0: self.curr_poc_diff_l0(),
            curr_poc_diff_l1: self.curr_poc_diff_l1(),
            col_poc_diff_l0: diff_pic_order_cnt(self.col_pic_poc, col_ref_l0_poc),
            col_poc_diff_l1: diff_pic_order_cnt(self.col_pic_poc, col_ref_l1_poc),
        }
    }
}

/// The padded-picture boundary parameters for the §8.5.2.3.5 clip.
///
/// `picPaddingSize` is the fixed `144` (eq. 506/507); the caller supplies
/// the picture's luma dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PicBounds {
    pub pic_width_in_luma_samples: i32,
    pub pic_height_in_luma_samples: i32,
}

/// §8.5.2.3.4 — scale one collocated component by `distScaleFactor`
/// (eq. 504): `Clip3(−32768, 32767, Sign(p) * ((Abs(p) + 16) >> 5))`
/// where `p = distScaleFactor * mvComponent`.
///
/// The `+16` / `>> 5` is the round-half-up reduction of the 5-bit
/// fractional `distScaleFactor`. The `Sign(p) * ((Abs(p) + …) >> …)`
/// form rounds the magnitude away from zero, matching the spec's
/// explicit `Sign`/`Abs` decomposition (a plain arithmetic shift would
/// round negative products toward −∞ instead).
fn scale_component(dist_scale_factor: i32, mv_component: i32) -> i32 {
    let p = (dist_scale_factor as i64) * (mv_component as i64);
    let mag = (p.abs() + 16) >> 5;
    let signed = if p < 0 { -mag } else { mag };
    signed.clamp(-32768, 32767) as i32
}

/// §8.5.2.3.4 eq. 503 — `distScaleFactorLX = (currPocDiffLX << 5) /
/// colPocDiffLX`. Integer division truncates toward zero (the spec's
/// `/` on integers). `col_poc_diff` is guaranteed non-zero by the
/// caller (the eq.-503 branch is gated on `colPocDiffLX != 0`).
fn dist_scale_factor(curr_poc_diff: i32, col_poc_diff: i32) -> i32 {
    ((curr_poc_diff as i64) << 5).wrapping_div(col_poc_diff as i64) as i32
}

/// §8.5.2.3.5 — constrained-scaled-motion clip (eqs. 506-511).
///
/// Clips a single scaled `mvLXCol` so that `( xCb, yCb ) + mvLXCol`
/// stays inside the padded reference grid `[−picPaddingSize, padded{W,H}]`.
/// `picPaddingSize` is the fixed `144`.
pub fn constrain_scaled_mv(
    mv: MotionVector,
    x_cb: i32,
    y_cb: i32,
    bounds: PicBounds,
) -> MotionVector {
    const PIC_PADDING_SIZE: i32 = 144;
    let padded_width = bounds.pic_width_in_luma_samples + PIC_PADDING_SIZE;
    let padded_height = bounds.pic_height_in_luma_samples + PIC_PADDING_SIZE;

    let mut x = mv.x;
    let mut y = mv.y;

    // eq. 508 / 509: clamp the lower (negative-overrun) boundary first.
    if x_cb + x < -PIC_PADDING_SIZE {
        x = -(x_cb + PIC_PADDING_SIZE);
    }
    if y_cb + y < -PIC_PADDING_SIZE {
        y = -(y_cb + PIC_PADDING_SIZE);
    }
    // eq. 510 / 511: clamp the upper (positive-overrun) boundary.
    if x_cb + x > padded_width {
        x = padded_width - x_cb;
    }
    if y_cb + y > padded_height {
        y = padded_height - y_cb;
    }
    MotionVector { x, y }
}

/// The joint §8.5.2.3.4 `availableFlagCol` code: 0 (neither list), 1 (L0
/// only), 2 (L1 only), 3 (both). Names mirror the eq.-after-505 table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvailableFlagCol {
    None,
    L0Only,
    L1Only,
    Both,
}

impl AvailableFlagCol {
    fn from_pair(l0: bool, l1: bool) -> Self {
        match (l0, l1) {
            (true, true) => AvailableFlagCol::Both,
            (true, false) => AvailableFlagCol::L0Only,
            (false, true) => AvailableFlagCol::L1Only,
            (false, false) => AvailableFlagCol::None,
        }
    }

    /// Numeric code (0/1/2/3) the spec uses for the §8.5.2.3.3 gates.
    pub fn code(self) -> u8 {
        match self {
            AvailableFlagCol::None => 0,
            AvailableFlagCol::L0Only => 1,
            AvailableFlagCol::L1Only => 2,
            AvailableFlagCol::Both => 3,
        }
    }
}

/// The §8.5.2.3.4 output: the scaled `mvpLXCol` for both lists plus the
/// joint availability code. `mvp_lX` only carries meaning when the
/// `available` code includes list X.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollocatedResult {
    pub available: AvailableFlagCol,
    pub mvp_l0: MotionVector,
    pub mvp_l1: MotionVector,
}

/// §8.5.2.3.4 — derive `mvpLXCol` / `availableFlagCol` from a single
/// collocated sample's stored motion.
///
/// This implements the `temporal_mvp_assigned_flag == 0` path (the
/// Baseline / ordinary case): the collocated reference list `colX` is
/// chosen by which of `predFlagL{0,1}Col` is set (L1-only → take L1,
/// L0-only → take L0, both → take both). The
/// `temporal_mvp_assigned_flag == 1` path (explicit `col_pic_list_idx` /
/// `col_source_mvp_list_idx` selection) is a Main-profile slice-header
/// option not yet threaded; callers on that path pass the already
/// list-selected motion in `col.mv_l0` / `col.pred_flag_l0` and leave
/// `temporal_mvp_assigned_flag == false` — equivalent for the single
/// active list.
///
/// `bounds` + `( x_cb, y_cb )` drive the §8.5.2.3.5 boundary clip
/// applied to each scaled vector before output.
pub fn tmvp_collocated_mv(
    col: CollocatedMv,
    poc: PocContext,
    x_cb: i32,
    y_cb: i32,
    bounds: PicBounds,
) -> CollocatedResult {
    // The §8.5.2.3.4 "If refIdxLXCol is invalid" gate: a list is invalid
    // when its predFlag is 0 or its refIdx is −1.
    let l0_valid = col.pred_flag_l0 && col.ref_idx_l0 != -1;
    let l1_valid = col.pred_flag_l1 && col.ref_idx_l1 != -1;

    // §8.5.2.3.4 "Otherwise" — choose mvCol[ colX ] per the predFlag
    // pattern. The spec's three cases (L1-only, L0-only, both) collapse
    // to "per active list, take that list's stored mv".
    let derive = |valid: bool,
                  mv: MotionVector,
                  curr_poc_diff: i32,
                  col_poc_diff: i32|
     -> Option<MotionVector> {
        // eq.-after-504: invalid refIdxCol[X] or colPocDiff == 0 ⇒
        // availableFlagLXCol = 0, mvpLXCol = 0 (eq. 505).
        if !valid || col_poc_diff == 0 {
            return None;
        }
        // eq. 503/504: scale mvCol[X] by the POC ratio, then clip.
        let sf = dist_scale_factor(curr_poc_diff, col_poc_diff);
        let scaled = MotionVector {
            x: scale_component(sf, mv.x),
            y: scale_component(sf, mv.y),
        };
        Some(constrain_scaled_mv(scaled, x_cb, y_cb, bounds))
    };

    let mvp_l0 = derive(
        l0_valid,
        col.mv_l0,
        poc.curr_poc_diff_l0,
        poc.col_poc_diff_l0,
    );
    let mvp_l1 = derive(
        l1_valid,
        col.mv_l1,
        poc.curr_poc_diff_l1,
        poc.col_poc_diff_l1,
    );

    CollocatedResult {
        available: AvailableFlagCol::from_pair(mvp_l0.is_some(), mvp_l1.is_some()),
        mvp_l0: mvp_l0.unwrap_or_default(),
        mvp_l1: mvp_l1.unwrap_or_default(),
    }
}

/// Convert a §8.5.2.3.4 result into the §8.5.2.3.3 merge candidate,
/// applying the eqs. 487/488 (493/494, 499/500) small-block demotion:
/// when both lists are available **and** `nCbW + nCbH ≤ 12`, list 1 is
/// dropped (`predFlagL1Col = 0`, `refIdxL1Col = −1`).
///
/// Returns `None` when the result's availability code is 0 (no
/// candidate produced). `refIdxLXCol` is always 0 (the §8.5.2.3.3
/// `refIdxLXCol = 0` assignment).
fn result_to_merge_cand(res: CollocatedResult, n_cb_w: i32, n_cb_h: i32) -> Option<MergeCand> {
    let mut l0 = matches!(
        res.available,
        AvailableFlagCol::L0Only | AvailableFlagCol::Both
    );
    let mut l1 = matches!(
        res.available,
        AvailableFlagCol::L1Only | AvailableFlagCol::Both
    );
    if !l0 && !l1 {
        return None;
    }
    // eqs. 487/488 (and the bottom/side parallels 493/494, 499/500):
    // small-block bi-pred demotion. "availableFlagCol == 3" ≡ both lists.
    if l0 && l1 && (n_cb_w + n_cb_h) <= 12 {
        l1 = false;
    }
    let _ = &mut l0;
    Some(MergeCand {
        pred_flag_l0: l0,
        pred_flag_l1: l1,
        ref_idx_l0: if l0 { 0 } else { -1 },
        ref_idx_l1: if l1 { 0 } else { -1 },
        mv_l0: if l0 {
            res.mvp_l0
        } else {
            MotionVector::default()
        },
        mv_l1: if l1 {
            res.mvp_l1
        } else {
            MotionVector::default()
        },
    })
}

/// Quantise a luma location to the 8×8 collocated-motion grid:
/// `( ( v >> 3 ) << 3 )`. The §8.5.2.3.3 central / bottom / side
/// positions are all snapped to this grid before the §8.5.2.3.4 lookup.
fn grid8(v: i32) -> i32 {
    (v >> 3) << 3
}

/// §8.5.2.3.3 — derive the temporal (collocated) merge candidate.
///
/// Tries the three collocated positions in order — **central** (eqs.
/// 485/486), **bottom** (eqs. 489-492), then **side** (eqs. 495-498) —
/// each snapped to the 8×8 grid and looked up via `col_at`, stopping at
/// the first position whose §8.5.2.3.4 derivation yields a non-zero
/// `availableFlagCol`. Returns the resulting [`MergeCand`], or `None`
/// when every position is unavailable.
///
/// Position fallbacks are gated exactly as the spec:
///
/// * **bottom** is consulted only when the central position was
///   unavailable, `yColBot < pic_height_in_luma_samples`, and the bottom
///   sample lies in the same CTB row as the CU (`yCb >> CtbLog2SizeY ==
///   yColBot >> CtbLog2SizeY`).
/// * **side** is consulted only when both earlier positions failed,
///   `xColSide > 0`, and `xColSide < pic_width_in_luma_samples`.
///
/// The §8.5.2.3.3 redundancy trim (the per-step §8.5.2.3.10 invocation)
/// is **not** applied here — it is the responsibility of
/// [`build_merge_cand_list`](crate::merge::build_merge_cand_list)'s
/// step-2, which already runs `merge_cand_redundancy_check` after
/// appending this candidate.
///
/// `col_at` resolves an 8×8-grid-aligned collocated luma location
/// `( xColCb, yColCb )` to the [`CollocatedMv`] stored there in `ColPic`.
#[allow(clippy::too_many_arguments)]
pub fn tmvp_merge_candidate<F>(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    n_cb_h: i32,
    avail_lr: AvailLr,
    poc: PocContext,
    bounds: PicBounds,
    ctb_log2_size_y: u32,
    mut col_at: F,
) -> Option<MergeCand>
where
    F: FnMut(i32, i32) -> CollocatedMv,
{
    let try_position = |x_col_cb: i32, y_col_cb: i32, col_at: &mut F| -> Option<MergeCand> {
        let col = col_at(x_col_cb, y_col_cb);
        let res = tmvp_collocated_mv(col, poc, x_cb, y_cb, bounds);
        result_to_merge_cand(res, n_cb_w, n_cb_h)
    };

    // Step 1: central collocated position (eqs. 485/486).
    let x_col_ctr = x_cb + (n_cb_w >> 1);
    let y_col_ctr = y_cb + (n_cb_h >> 1);
    if let Some(cand) = try_position(grid8(x_col_ctr), grid8(y_col_ctr), &mut col_at) {
        return Some(cand);
    }

    // Step 2: bottom collocated position (eqs. 489-492), gated on
    // same-CTB-row + in-picture-height.
    let (x_col_bot, y_col_bot) = if avail_lr == AvailLr::Lr01 {
        (x_cb, y_cb + n_cb_h) // eqs. 489/490
    } else {
        (x_cb + n_cb_w - 1, y_cb + n_cb_h) // eqs. 491/492
    };
    if y_col_bot < bounds.pic_height_in_luma_samples
        && (y_cb >> ctb_log2_size_y) == (y_col_bot >> ctb_log2_size_y)
    {
        if let Some(cand) = try_position(grid8(x_col_bot), grid8(y_col_bot), &mut col_at) {
            return Some(cand);
        }
    }

    // Step 3: side collocated position (eqs. 495-498), gated on
    // in-picture-width (strictly inside the left + right edges).
    let (x_col_side, y_col_side) = if avail_lr == AvailLr::Lr01 {
        (x_cb - 1, y_cb + n_cb_h - 1) // eqs. 495/496
    } else {
        (x_cb + n_cb_w, y_cb + n_cb_h - 1) // eqs. 497/498
    };
    if x_col_side > 0 && x_col_side < bounds.pic_width_in_luma_samples {
        if let Some(cand) = try_position(grid8(x_col_side), grid8(y_col_side), &mut col_at) {
            return Some(cand);
        }
    }

    None
}

/// A collocated cell extended with the POCs of the pictures its stored L0
/// / L1 motion referenced (`refPicOfColPic[ 0 ]` / `refPicOfColPic[ 1 ]`).
///
/// The plain [`CollocatedMv`] carries the cell's `refIdxLXCol`, but the
/// §8.5.2.3.4 `colPocDiffLX` (eq. 502) needs the *POC* of the picture that
/// reference index resolved to inside `ColPic`'s own reference list —
/// information the decoder's DPB knows but a bare `refIdx` does not. This
/// type pairs the motion with those resolved POCs so [`PocContext::derive`]
/// can compute `colPocDiff` per cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CollocatedCell {
    /// The cell's stored motion (`predFlagLXCol` / `mvLXCol` / `refIdxLXCol`).
    pub mv: CollocatedMv,
    /// `PicOrderCnt( refPicOfColPic[ 0 ] )` — the picture `ColPic`'s L0
    /// motion at this cell referenced. Ignored when L0 is invalid.
    pub col_ref_l0_poc: i32,
    /// `PicOrderCnt( refPicOfColPic[ 1 ] )`.
    pub col_ref_l1_poc: i32,
}

/// §8.5.2.3.3 — derive the temporal (collocated) merge candidate with the
/// POC distances resolved **per collocated cell** from [`PocInputs`].
///
/// This is the per-CU TMVP entry point the decoder drives: rather than a
/// single pre-computed [`PocContext`] (which bakes in one fixed
/// `colPocDiff`), it takes the per-CU [`PocInputs`] and a `col_at` closure
/// returning a [`CollocatedCell`] that carries each cell's
/// `refPicOfColPic` POCs. For every consulted position the eq.-502
/// `colPocDiff` is recomputed from that cell's stored references via
/// [`PocInputs::derive`], so a `ColPic` whose collocated blocks reference
/// different pictures is scaled correctly.
///
/// The position fallback order, gating, grid-quantisation and small-block
/// demotion are identical to [`tmvp_merge_candidate`]; only the POC
/// context is per-cell.
#[allow(clippy::too_many_arguments)]
pub fn tmvp_merge_candidate_with_poc<F>(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    n_cb_h: i32,
    avail_lr: AvailLr,
    poc_inputs: PocInputs,
    bounds: PicBounds,
    ctb_log2_size_y: u32,
    mut col_at: F,
) -> Option<MergeCand>
where
    F: FnMut(i32, i32) -> CollocatedCell,
{
    let mut try_position = |x_col_cb: i32, y_col_cb: i32| -> Option<MergeCand> {
        let cell = col_at(x_col_cb, y_col_cb);
        let poc = poc_inputs.derive(cell.col_ref_l0_poc, cell.col_ref_l1_poc);
        let res = tmvp_collocated_mv(cell.mv, poc, x_cb, y_cb, bounds);
        result_to_merge_cand(res, n_cb_w, n_cb_h)
    };

    // Step 1: central collocated position (eqs. 485/486).
    let x_col_ctr = x_cb + (n_cb_w >> 1);
    let y_col_ctr = y_cb + (n_cb_h >> 1);
    if let Some(cand) = try_position(grid8(x_col_ctr), grid8(y_col_ctr)) {
        return Some(cand);
    }

    // Step 2: bottom collocated position (eqs. 489-492).
    let (x_col_bot, y_col_bot) = if avail_lr == AvailLr::Lr01 {
        (x_cb, y_cb + n_cb_h)
    } else {
        (x_cb + n_cb_w - 1, y_cb + n_cb_h)
    };
    if y_col_bot < bounds.pic_height_in_luma_samples
        && (y_cb >> ctb_log2_size_y) == (y_col_bot >> ctb_log2_size_y)
    {
        if let Some(cand) = try_position(grid8(x_col_bot), grid8(y_col_bot)) {
            return Some(cand);
        }
    }

    // Step 3: side collocated position (eqs. 495-498).
    let (x_col_side, y_col_side) = if avail_lr == AvailLr::Lr01 {
        (x_cb - 1, y_cb + n_cb_h - 1)
    } else {
        (x_cb + n_cb_w, y_cb + n_cb_h - 1)
    };
    if x_col_side > 0 && x_col_side < bounds.pic_width_in_luma_samples {
        if let Some(cand) = try_position(grid8(x_col_side), grid8(y_col_side)) {
            return Some(cand);
        }
    }

    None
}

/// Adapt a decoded picture's [`SideInfoGrid`](crate::deblock::SideInfoGrid)
/// — the per-4×4-cell motion field stamped during reconstruction — into
/// the [`CollocatedMv`] a [`tmvp_merge_candidate`] `col_at` closure
/// returns, for use as the `ColPic` motion store.
///
/// The §8.5.2.3.4 input arrays (`predFlagLXCol`, `mvLXCol`,
/// `refIdxLXCol`) map onto a [`CuSideInfo`](crate::deblock::CuSideInfo)
/// cell as:
///
/// * `predFlagLXCol` ⇐ the cell was coded `Inter` **and** `refIdxLX != −1`
///   (the grid stores `−1` as the "list unavailable" sentinel; an intra /
///   IBC cell carries no inter motion so both lists read invalid).
/// * `refIdxLXCol` ⇐ `ref_idx_lX` (already `−1` when unavailable).
/// * `mvLXCol` ⇐ `(mv_lX_x, mv_lX_y)` in 1/4-pel.
///
/// The lookup snaps the requested luma `( x_col_cb, y_col_cb )` to the
/// grid's 4×4 cell (`>> 2`); the §8.5.2.3.3 caller has already snapped
/// the position to the coarser 8×8 collocated grid, so the extra `>> 2`
/// only selects the covering cell. Out-of-grid locations return the
/// default (invalid) cell, which the §8.5.2.3.4 invalid-refIdx escape
/// turns into `availableFlagCol = 0`.
pub fn collocated_mv_from_side_info(
    col_grid: &crate::deblock::SideInfoGrid,
    x_col_cb: i32,
    y_col_cb: i32,
) -> CollocatedMv {
    use crate::deblock::CuPredMode;

    if x_col_cb < 0 || y_col_cb < 0 {
        return CollocatedMv::default();
    }
    let cell = col_grid.at((x_col_cb >> 2) as usize, (y_col_cb >> 2) as usize);
    // Only an inter-coded cell carries usable collocated motion; intra /
    // IBC cells leave both lists invalid (refIdx == −1 sentinel).
    if cell.pred_mode != CuPredMode::Inter {
        return CollocatedMv::default();
    }
    let l0_valid = cell.ref_idx_l0 != -1;
    let l1_valid = cell.ref_idx_l1 != -1;
    // §8.5.1 NOTE — the collocated readers consume the *refined* motion
    // `refMvLX = ( mvLX << 2 ) + delta` (eqs. 396/397/400; delta is 0 on
    // every non-DMVR cell), rounded from the 1/16-pel refined vector back
    // onto the grid's 1/4-pel unit via the §8.5.3.10 rule.
    let refined = |mv_x: i32, mv_y: i32, dx: i32, dy: i32| -> MotionVector {
        if dx == 0 && dy == 0 {
            return MotionVector::quarter_pel(mv_x, mv_y);
        }
        crate::inter::round_motion_vector(
            MotionVector {
                x: (mv_x << 2) + dx,
                y: (mv_y << 2) + dy,
            },
            2,
            0,
        )
    };
    CollocatedMv {
        pred_flag_l0: l0_valid,
        pred_flag_l1: l1_valid,
        ref_idx_l0: cell.ref_idx_l0 as i32,
        ref_idx_l1: cell.ref_idx_l1 as i32,
        mv_l0: refined(
            cell.mv_l0_x,
            cell.mv_l0_y,
            cell.ref_mv_delta_l0_x,
            cell.ref_mv_delta_l0_y,
        ),
        mv_l1: refined(
            cell.mv_l1_x,
            cell.mv_l1_y,
            cell.ref_mv_delta_l1_x,
            cell.ref_mv_delta_l1_y,
        ),
    }
}

/// Adapt a decoded picture's [`SideInfoGrid`](crate::deblock::SideInfoGrid)
/// into the [`CollocatedCell`] the per-cell [`tmvp_merge_candidate_with_poc`]
/// consumes — the `ColPic` motion store **plus** the resolved
/// `refPicOfColPic` POCs.
///
/// This extends [`collocated_mv_from_side_info`] with the §8.5.2.3.4
/// eq.-502 POC resolution: `col_ref_poc(list_x, ref_idx)` maps the cell's
/// stored `RefIdxLX` to the POC of the picture that index resolved to in
/// `ColPic`'s own reference list `list_x` (the DPB knows this; the bare
/// grid cell does not). For an invalid / unavailable list the lookup is
/// skipped and the corresponding `col_ref_*_poc` stays `0` (it is unused
/// because the §8.5.2.3.4 invalid-refIdx escape fires first).
pub fn collocated_cell_from_side_info<P>(
    col_grid: &crate::deblock::SideInfoGrid,
    x_col_cb: i32,
    y_col_cb: i32,
    mut col_ref_poc: P,
) -> CollocatedCell
where
    P: FnMut(u8, i32) -> i32,
{
    let mv = collocated_mv_from_side_info(col_grid, x_col_cb, y_col_cb);
    let col_ref_l0_poc = if mv.pred_flag_l0 {
        col_ref_poc(0, mv.ref_idx_l0)
    } else {
        0
    };
    let col_ref_l1_poc = if mv.pred_flag_l1 {
        col_ref_poc(1, mv.ref_idx_l1)
    } else {
        0
    };
    CollocatedCell {
        mv,
        col_ref_l0_poc,
        col_ref_l1_poc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> PicBounds {
        PicBounds {
            pic_width_in_luma_samples: 1920,
            pic_height_in_luma_samples: 1080,
        }
    }

    /// Same-distance scaling (currPocDiff == colPocDiff) is the identity:
    /// distScaleFactor = (d << 5) / d = 32, and (32 * mv + 16) >> 5 == mv.
    #[test]
    fn equal_poc_distance_is_identity() {
        let col = CollocatedMv {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: MotionVector::quarter_pel(40, -24),
            mv_l1: MotionVector::default(),
        };
        let poc = PocContext {
            curr_poc_diff_l0: 4,
            curr_poc_diff_l1: 4,
            col_poc_diff_l0: 4,
            col_poc_diff_l1: 4,
        };
        let res = tmvp_collocated_mv(col, poc, 100, 100, bounds());
        assert_eq!(res.available, AvailableFlagCol::L0Only);
        assert_eq!(res.mvp_l0, MotionVector::quarter_pel(40, -24));
    }

    /// Double-distance scaling: distScaleFactor = (8 << 5)/4 = 64, so
    /// (64 * mv + 16) >> 5 doubles the magnitude (round-half-up).
    #[test]
    fn double_poc_distance_doubles_mv() {
        let col = CollocatedMv {
            pred_flag_l0: true,
            ref_idx_l0: 0,
            mv_l0: MotionVector::quarter_pel(10, 7),
            ..Default::default()
        };
        let poc = PocContext {
            curr_poc_diff_l0: 8,
            curr_poc_diff_l1: 0,
            col_poc_diff_l0: 4,
            col_poc_diff_l1: 0,
        };
        let res = tmvp_collocated_mv(col, poc, 0, 0, bounds());
        // sf = 64. x: (64*10+16)>>5 = 656>>5 = 20. y: (64*7+16)>>5 = 464>>5 = 14.
        assert_eq!(res.mvp_l0, MotionVector::quarter_pel(20, 14));
    }

    /// Negative product rounds away from zero (Sign/Abs decomposition),
    /// not toward −∞.
    #[test]
    fn negative_scaling_rounds_away_from_zero() {
        // p = sf*mv = -17. Sign/Abs: (|−17|+16)>>5 = 33>>5 = 1, signed → -1.
        // A plain arithmetic shift `-17 >> 5` would give -1 here too, but
        // diverges at p = -33: Sign/Abs → (49)>>5 = 1 → -1; `-33>>5` = -2.
        assert_eq!(scale_component(1, -17), -1);
        assert_eq!(scale_component(1, -33), -1);
        assert_eq!(scale_component(1, 33), 1);
    }

    /// Zero colPocDiff ⇒ unavailable (eq.-after-504 / eq. 505).
    #[test]
    fn zero_col_poc_diff_unavailable() {
        let col = CollocatedMv {
            pred_flag_l0: true,
            ref_idx_l0: 0,
            mv_l0: MotionVector::quarter_pel(8, 8),
            ..Default::default()
        };
        let poc = PocContext {
            curr_poc_diff_l0: 4,
            curr_poc_diff_l1: 0,
            col_poc_diff_l0: 0,
            col_poc_diff_l1: 0,
        };
        let res = tmvp_collocated_mv(col, poc, 0, 0, bounds());
        assert_eq!(res.available, AvailableFlagCol::None);
    }

    /// Invalid refIdx on a list ⇒ that list contributes nothing.
    #[test]
    fn invalid_ref_idx_drops_list() {
        let col = CollocatedMv {
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_idx_l0: -1, // invalid
            ref_idx_l1: 0,
            mv_l0: MotionVector::quarter_pel(99, 99),
            mv_l1: MotionVector::quarter_pel(4, 4),
        };
        let poc = PocContext {
            curr_poc_diff_l0: 2,
            curr_poc_diff_l1: 2,
            col_poc_diff_l0: 2,
            col_poc_diff_l1: 2,
        };
        let res = tmvp_collocated_mv(col, poc, 0, 0, bounds());
        assert_eq!(res.available, AvailableFlagCol::L1Only);
        assert_eq!(res.mvp_l1, MotionVector::quarter_pel(4, 4));
    }

    /// Small-block bi-pred demotion (eqs. 487/488): nCbW+nCbH ≤ 12 drops L1.
    #[test]
    fn small_block_demotes_bipred() {
        let res = CollocatedResult {
            available: AvailableFlagCol::Both,
            mvp_l0: MotionVector::quarter_pel(4, 0),
            mvp_l1: MotionVector::quarter_pel(0, 8),
        };
        let cand = result_to_merge_cand(res, 4, 8).unwrap(); // 4+8=12 ≤ 12
        assert!(cand.pred_flag_l0 && !cand.pred_flag_l1);
        assert_eq!(cand.ref_idx_l1, -1);
        // Larger block keeps both.
        let cand2 = result_to_merge_cand(res, 8, 8).unwrap();
        assert!(cand2.pred_flag_l0 && cand2.pred_flag_l1);
    }

    /// §8.5.2.3.5 boundary clip: a far-positive MV is clamped to the
    /// padded right/bottom edge.
    #[test]
    fn boundary_clip_clamps_positive_overrun() {
        let b = bounds(); // 1920x1080, padding 144 → padded 2064 x 1224.
                          // CU at (2000, 1200), MV +200 in x would land at 2200 > 2064.
        let mv = MotionVector::quarter_pel(200, 200);
        let clipped = constrain_scaled_mv(mv, 2000, 1200, b);
        assert_eq!(clipped.x, 2064 - 2000); // = 64
        assert_eq!(clipped.y, 1224 - 1200); // = 24
    }

    /// §8.5.2.3.5 boundary clip: a far-negative MV is clamped to the
    /// padded left/top edge.
    #[test]
    fn boundary_clip_clamps_negative_overrun() {
        let b = bounds();
        // CU at (10, 10), MV -200 → 10-200 = -190 < -144.
        let mv = MotionVector::quarter_pel(-200, -200);
        let clipped = constrain_scaled_mv(mv, 10, 10, b);
        assert_eq!(clipped.x, -(10 + 144)); // = -154
        assert_eq!(clipped.y, -(10 + 144));
    }

    /// §8.5.2.3.3 central position succeeds first — bottom/side never
    /// consulted.
    #[test]
    fn central_position_short_circuits() {
        let poc = PocContext {
            curr_poc_diff_l0: 2,
            curr_poc_diff_l1: 0,
            col_poc_diff_l0: 2,
            col_poc_diff_l1: 0,
        };
        // 16x16 CU at (32, 32). Center = (40,40) → grid8 = (40,40).
        let mut consulted = Vec::new();
        let cand = tmvp_merge_candidate(32, 32, 16, 16, AvailLr::Lr00, poc, bounds(), 6, |x, y| {
            consulted.push((x, y));
            if (x, y) == (40, 40) {
                CollocatedMv {
                    pred_flag_l0: true,
                    ref_idx_l0: 0,
                    mv_l0: MotionVector::quarter_pel(12, -4),
                    ..Default::default()
                }
            } else {
                CollocatedMv::default()
            }
        })
        .unwrap();
        assert_eq!(cand.mv_l0, MotionVector::quarter_pel(12, -4));
        assert_eq!(consulted, vec![(40, 40)]); // only the center.
    }

    /// §8.5.2.3.3 fallback: central unavailable → bottom consulted.
    #[test]
    fn falls_back_to_bottom_then_side() {
        let poc = PocContext {
            curr_poc_diff_l0: 2,
            curr_poc_diff_l1: 0,
            col_poc_diff_l0: 2,
            col_poc_diff_l1: 0,
        };
        // 16x16 CU at (0,0), LR_00. Center=(8,8). Bottom=(15,16)→grid(8,16).
        let mut consulted = Vec::new();
        let cand = tmvp_merge_candidate(0, 0, 16, 16, AvailLr::Lr00, poc, bounds(), 6, |x, y| {
            consulted.push((x, y));
            if (x, y) == (8, 16) {
                CollocatedMv {
                    pred_flag_l0: true,
                    ref_idx_l0: 0,
                    mv_l0: MotionVector::quarter_pel(5, 5),
                    ..Default::default()
                }
            } else {
                CollocatedMv::default() // center unavailable.
            }
        })
        .unwrap();
        assert_eq!(cand.mv_l0, MotionVector::quarter_pel(5, 5));
        assert_eq!(consulted, vec![(8, 8), (8, 16)]);
    }

    /// All three positions unavailable ⇒ no candidate.
    #[test]
    fn all_positions_unavailable_yields_none() {
        let poc = PocContext {
            curr_poc_diff_l0: 2,
            curr_poc_diff_l1: 2,
            col_poc_diff_l0: 2,
            col_poc_diff_l1: 2,
        };
        let cand = tmvp_merge_candidate(0, 0, 16, 16, AvailLr::Lr00, poc, bounds(), 6, |_, _| {
            CollocatedMv::default()
        });
        assert!(cand.is_none());
    }

    /// Bottom position skipped when it crosses into the next CTB row.
    #[test]
    fn bottom_skipped_across_ctb_row() {
        let poc = PocContext {
            curr_poc_diff_l0: 2,
            curr_poc_diff_l1: 0,
            col_poc_diff_l0: 2,
            col_poc_diff_l1: 0,
        };
        // CtbLog2SizeY = 5 (32-sample CTB). CU 16x16 at (0,16): yColBot =
        // 16+16 = 32. yCb>>5 = 0, yColBot>>5 = 1 → different CTB row → skip.
        // Side: xColSide = 16, in-width → consulted.
        let mut consulted = Vec::new();
        let cand = tmvp_merge_candidate(0, 16, 16, 16, AvailLr::Lr00, poc, bounds(), 5, |x, y| {
            consulted.push((x, y));
            if (x, y) == (16, 24) {
                // side = (16, 16+16-1=31) → grid (16, 24)
                CollocatedMv {
                    pred_flag_l0: true,
                    ref_idx_l0: 0,
                    mv_l0: MotionVector::quarter_pel(7, 7),
                    ..Default::default()
                }
            } else {
                CollocatedMv::default()
            }
        })
        .unwrap();
        assert_eq!(cand.mv_l0, MotionVector::quarter_pel(7, 7));
        // center (8, 24) then side (16, 24) — bottom (8, 32) skipped.
        assert_eq!(consulted, vec![(8, 24), (16, 24)]);
    }

    /// The SideInfoGrid bridge maps an inter-coded cell's motion onto a
    /// valid `CollocatedMv`.
    #[test]
    fn side_info_bridge_inter_cell() {
        use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
        let mut grid = SideInfoGrid::new(64, 64);
        grid.stamp_block(
            16,
            16,
            16,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                cbf_luma: 0,
                mv_l0_x: 12,
                mv_l0_y: -4,
                mv_l1_x: 0,
                mv_l1_y: 0,
                ref_idx_l0: 0,
                ref_idx_l1: -1,
                ..Default::default()
            },
        );
        // Anywhere inside the stamped 16x16 block resolves to it.
        let col = collocated_mv_from_side_info(&grid, 24, 24);
        assert!(col.pred_flag_l0 && !col.pred_flag_l1);
        assert_eq!(col.ref_idx_l0, 0);
        assert_eq!(col.ref_idx_l1, -1);
        assert_eq!(col.mv_l0, MotionVector::quarter_pel(12, -4));
    }

    /// Intra / IBC and out-of-grid cells read invalid (both lists off).
    #[test]
    fn side_info_bridge_non_inter_and_oob() {
        use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
        let mut grid = SideInfoGrid::new(64, 64);
        grid.stamp_block(
            0,
            0,
            16,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Intra,
                mv_l0_x: 99,
                ref_idx_l0: 0,
                ..Default::default()
            },
        );
        // Intra cell → invalid.
        let intra = collocated_mv_from_side_info(&grid, 4, 4);
        assert!(!intra.pred_flag_l0 && !intra.pred_flag_l1);
        // Out-of-grid (negative + past the right edge) → default invalid.
        assert_eq!(
            collocated_mv_from_side_info(&grid, -8, 0),
            CollocatedMv::default()
        );
        assert_eq!(
            collocated_mv_from_side_info(&grid, 4096, 0),
            CollocatedMv::default()
        );
    }

    /// End-to-end: bridge a SideInfoGrid ColPic into a full
    /// §8.5.2.3.3 derivation and recover the scaled candidate.
    #[test]
    fn side_info_bridge_drives_full_tmvp() {
        use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
        let mut col_grid = SideInfoGrid::new(128, 128);
        // Stamp an inter cell covering the central collocated position of
        // a 16x16 CU at (32,32): center = (40,40), grid8 = (40,40).
        col_grid.stamp_block(
            40,
            40,
            8,
            8,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                mv_l0_x: 10,
                mv_l0_y: 6,
                ref_idx_l0: 0,
                ref_idx_l1: -1,
                ..Default::default()
            },
        );
        let poc = PocContext {
            curr_poc_diff_l0: 8, // sf = (8<<5)/4 = 64 → doubles
            curr_poc_diff_l1: 0,
            col_poc_diff_l0: 4,
            col_poc_diff_l1: 0,
        };
        let cand = tmvp_merge_candidate(32, 32, 16, 16, AvailLr::Lr00, poc, bounds(), 6, |x, y| {
            collocated_mv_from_side_info(&col_grid, x, y)
        })
        .unwrap();
        // sf = 64 → (64*10+16)>>5 = 20, (64*6+16)>>5 = 12.
        assert!(cand.pred_flag_l0 && !cand.pred_flag_l1);
        assert_eq!(cand.mv_l0, MotionVector::quarter_pel(20, 12));
    }

    // --- §8.5.2.3.3 per-cell POC-distance wiring ---

    /// eq. 165 `DiffPicOrderCnt` is plain signed subtraction.
    #[test]
    fn diff_pic_order_cnt_is_subtraction() {
        assert_eq!(diff_pic_order_cnt(8, 4), 4);
        assert_eq!(diff_pic_order_cnt(4, 8), -4);
        assert_eq!(diff_pic_order_cnt(5, 5), 0);
    }

    /// `PocInputs::derive` builds the four eq.-501/502 distances:
    /// curr distances from RefPicListX[0], col distances from the per-cell
    /// refPicOfColPic POCs.
    #[test]
    fn poc_inputs_derive_eqs_501_502() {
        let inputs = PocInputs {
            curr_poc: 16,
            ref_l0_poc: 12, // currPocDiffL0 = 4
            ref_l1_poc: 20, // currPocDiffL1 = -4
            col_pic_poc: 8,
        };
        // refPicOfColPic[0] POC 4 → colPocDiffL0 = 8-4 = 4;
        // refPicOfColPic[1] POC 10 → colPocDiffL1 = 8-10 = -2.
        let poc = inputs.derive(4, 10);
        assert_eq!(poc.curr_poc_diff_l0, 4);
        assert_eq!(poc.curr_poc_diff_l1, -4);
        assert_eq!(poc.col_poc_diff_l0, 4);
        assert_eq!(poc.col_poc_diff_l1, -2);
    }

    /// The per-cell entry point scales each consulted cell by *its own*
    /// colPocDiff: a central cell that referenced a 2×-distant picture is
    /// scaled by the eq.-503 ratio derived from that cell's POCs.
    #[test]
    fn per_cell_poc_scales_central() {
        // currPocDiffL0 = 16-12 = 4. Central cell referenced a picture at
        // POC 6 → colPocDiffL0 = 8-6 = 2 → sf = (4<<5)/2 = 64 → doubles.
        let inputs = PocInputs {
            curr_poc: 16,
            ref_l0_poc: 12,
            ref_l1_poc: 0,
            col_pic_poc: 8,
        };
        let cand = tmvp_merge_candidate_with_poc(
            32,
            32,
            16,
            16,
            AvailLr::Lr00,
            inputs,
            bounds(),
            6,
            |x, y| {
                if (x, y) == (40, 40) {
                    CollocatedCell {
                        mv: CollocatedMv {
                            pred_flag_l0: true,
                            ref_idx_l0: 0,
                            mv_l0: MotionVector::quarter_pel(10, 6),
                            ..Default::default()
                        },
                        col_ref_l0_poc: 6,
                        col_ref_l1_poc: 0,
                    }
                } else {
                    CollocatedCell::default()
                }
            },
        )
        .unwrap();
        // sf = 64 → (64*10+16)>>5 = 20, (64*6+16)>>5 = 12.
        assert_eq!(cand.mv_l0, MotionVector::quarter_pel(20, 12));
    }

    /// A per-cell colPocDiff of 0 (cell referenced ColPic's own POC) makes
    /// that position unavailable; the fallback chain advances.
    #[test]
    fn per_cell_zero_col_poc_diff_falls_through() {
        let inputs = PocInputs {
            curr_poc: 16,
            ref_l0_poc: 12,
            ref_l1_poc: 0,
            col_pic_poc: 8,
        };
        // Central cell: col_ref_l0_poc == col_pic_poc → colPocDiff 0 →
        // unavailable. Bottom cell (8,16) is valid.
        let mut consulted = Vec::new();
        let cand = tmvp_merge_candidate_with_poc(
            0,
            0,
            16,
            16,
            AvailLr::Lr00,
            inputs,
            bounds(),
            6,
            |x, y| {
                consulted.push((x, y));
                if (x, y) == (8, 8) {
                    CollocatedCell {
                        mv: CollocatedMv {
                            pred_flag_l0: true,
                            ref_idx_l0: 0,
                            mv_l0: MotionVector::quarter_pel(9, 9),
                            ..Default::default()
                        },
                        col_ref_l0_poc: 8, // == col_pic_poc → colPocDiff 0
                        col_ref_l1_poc: 0,
                    }
                } else if (x, y) == (8, 16) {
                    CollocatedCell {
                        mv: CollocatedMv {
                            pred_flag_l0: true,
                            ref_idx_l0: 0,
                            mv_l0: MotionVector::quarter_pel(4, 4),
                            ..Default::default()
                        },
                        col_ref_l0_poc: 4, // colPocDiff = 8-4 = 4 = currPocDiff → identity
                        col_ref_l1_poc: 0,
                    }
                } else {
                    CollocatedCell::default()
                }
            },
        )
        .unwrap();
        assert_eq!(cand.mv_l0, MotionVector::quarter_pel(4, 4));
        assert_eq!(consulted, vec![(8, 8), (8, 16)]);
    }

    /// The cell bridge resolves refPicOfColPic POCs through the supplied
    /// ColPic reference-list lookup, only for valid lists.
    #[test]
    fn collocated_cell_bridge_resolves_ref_poc() {
        use crate::deblock::{CuPredMode, CuSideInfo, SideInfoGrid};
        let mut grid = SideInfoGrid::new(64, 64);
        grid.stamp_block(
            16,
            16,
            16,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                mv_l0_x: 12,
                mv_l0_y: -4,
                ref_idx_l0: 1,
                ref_idx_l1: -1,
                ..Default::default()
            },
        );
        // ColPic's L0 ref-list: refIdx 1 → POC 30.
        let cell = collocated_cell_from_side_info(&grid, 24, 24, |list, ref_idx| {
            assert_eq!((list, ref_idx), (0, 1)); // only L0 queried
            30
        });
        assert!(cell.mv.pred_flag_l0 && !cell.mv.pred_flag_l1);
        assert_eq!(cell.col_ref_l0_poc, 30);
        assert_eq!(cell.col_ref_l1_poc, 0); // L1 invalid → not queried
    }
}
