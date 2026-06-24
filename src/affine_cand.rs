//! §8.5.3.2 / §8.5.3.4 / §8.5.3.5 / §8.5.3.6 affine candidate-list assembly.
//!
//! This module completes the affine motion-derivation surface begun in
//! [`crate::affine`] (the §8.5.3.1/.3/.7-.10 geometric + inheritance core).
//! Where [`crate::affine`] turns one *known* control-point motion-vector
//! (CPMV) set into a dense subblock field, this module builds the two
//! **candidate lists** the affine syntax indexes into:
//!
//! * §8.5.3.2 [`build_affine_merge_cand_list`] — the affine **merge**
//!   candidate list `affineMergeCandList` (up to 5 entries): the
//!   inherited (model-based) neighbour candidates ordered by the §6.4.2
//!   `availLR` branch, then the §8.5.3.4 constructed (corner-combined)
//!   candidates Const1..Const6, then the §8.5.3.2-step-9 zero-CPMV tail.
//!   [`select_affine_merge_candidate`] is the step-after-9 bridge that
//!   projects a decoded `affine_merge_idx` into the concrete
//!   `cpMvLX`/`refIdxLX`/`predFlagLX`/`numCpMv` the §8.5.3.7 subblock
//!   derivation consumes.
//! * §8.5.3.4 [`constructed_merge_candidates`] — the six corner-combined
//!   merge candidates Const1..Const6 (eqs. 764-835), built from the four
//!   §8.5.3.4 "corner" CPMVs (top-left, top-right, bottom-left,
//!   bottom-right) with the eqs. 807/813/819/828/829 corner-completion
//!   arithmetic.
//! * §8.5.3.5 [`build_affine_mvp_cand_list`] — the affine **predictor**
//!   (AMVP) candidate list `cpMvpListLX` (exactly 2 entries): the
//!   inherited A / B / C neighbour predictors (refIdx-matched), then the
//!   §8.5.3.6 constructed predictor, then the per-corner translational
//!   fill, then the zero-CPMV tail (eqs. 836-867).
//! * §8.5.3.6 [`constructed_mvp_candidate`] — the single constructed
//!   predictor candidate (eqs. 868-872) from the four refIdx-matched
//!   corner MVs.
//!
//! The neighbour **geometry** is exposed separately so the decoder's
//! MV-store wiring can resolve each named neighbour without re-deriving
//! the eqs. 708-717 / 836-842 sample locations:
//! [`affine_merge_nb_positions`] + [`affine_merge_inherited_order`] +
//! [`AffineNbName`] for the merge scan, and [`affine_mvp_nb_positions`]
//! for the predictor scan.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).
//!
//! ## Purity & the neighbour-lookup contract
//!
//! Like [`crate::merge`] and [`crate::tmvp`], every derivation here is a
//! pure function over a **caller-supplied neighbour-lookup closure**
//! rather than a concrete motion grid. The merge path reads affine-coded
//! neighbours through [`AffineNeighbour`] (their stored corner MVs +
//! geometry + `MotionModelIdc`); the constructed paths read translational
//! corner MVs through [`CornerMv`]. The decoder's MV / affine-model store
//! wires these in; this module never touches the grid directly, keeping
//! every step unit-testable against synthetic neighbours.
//!
//! The §8.5.3.4 corner-2/3 **temporal** (collocated) fallback — the
//! `availLR != LR_10/LR_11` branch of corner 2 and the
//! `availLR != LR_01/LR_11` branch of corner 3, which fall back to a
//! §8.5.2.3.4 collocated MV (eqs. 776-797) — is threaded through the same
//! optional [`CornerMv`] slots: the caller resolves the collocated MV (via
//! [`crate::tmvp`]) and supplies it as the corner's [`CornerMv`], so this
//! module's corner logic is uniform over spatial and temporal sources.

use crate::affine::{inherited_cp_mvs, ControlPointMv, NeighbourAffineSource};
use crate::inter::MotionVector;
use crate::neighbour::AvailLr;

/// `Clip3(−2¹⁵, 2¹⁵ − 1, ·)` — the eqs. 808/809/814/815/820/821/834/835/872
/// 16-bit CPMV clip applied to constructed-candidate components.
#[inline]
fn clip_cpmv_component(v: i32) -> i32 {
    const C: i32 = 1 << 15;
    v.clamp(-C, C - 1)
}

/// One prediction list's affine candidate: the 2- or 3-CPMV set plus its
/// reference index and utilization flag, for a single list X.
///
/// `pred_flag == false` marks an absent list (`predFlagLX = 0`,
/// `refIdxLX = −1`); the `cp_mv` entries are then ignored. A full affine
/// candidate is a pair of these (`l0`, `l1`) — see [`AffineCand`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AffineListMv {
    /// `predFlagLX` — list X is used by this candidate.
    pub pred_flag: bool,
    /// `refIdxLX` (meaningful iff `pred_flag`; `−1` when absent).
    pub ref_idx: i32,
    /// `cpMvLX[ 0..numCpMv ]` — control-point MVs, 1/4-pel luma. Only the
    /// first `num_cp_mv` entries (from the owning [`AffineCand`]) are
    /// meaningful.
    pub cp_mv: [ControlPointMv; 3],
}

impl AffineListMv {
    /// An absent list (`predFlag = 0`, `refIdx = −1`, zero CPMVs).
    #[inline]
    fn absent() -> Self {
        Self {
            pred_flag: false,
            ref_idx: -1,
            cp_mv: [MotionVector::default(); 3],
        }
    }
}

/// A full affine candidate — both prediction lists plus the model order.
///
/// `motion_model_idc` is 1 (4-parameter, `numCpMv = 2`) or 2 (6-parameter,
/// `numCpMv = 3`); `num_cp_mv()` returns `motion_model_idc + 1`. This is
/// the entry type stored in both `affineMergeCandList` (§8.5.3.2) and
/// returned by [`select_affine_merge_candidate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineCand {
    /// `MotionModelIdc` for this candidate: 1 (4-param) or 2 (6-param).
    pub motion_model_idc: u32,
    /// List-0 CPMV set.
    pub l0: AffineListMv,
    /// List-1 CPMV set.
    pub l1: AffineListMv,
}

impl AffineCand {
    /// `numCpMv = motionModelIdc + 1` (eq. 740).
    #[inline]
    pub fn num_cp_mv(&self) -> u32 {
        self.motion_model_idc + 1
    }

    /// The §8.5.3.2-step-9 zero-CPMV merge candidate `zeroCandm` (eqs.
    /// 723-733): L0 used with `refIdx = 0` + zero CPMVs; L1 used only for
    /// a B-slice; `motionModelIdc = 1`.
    fn zero(slice_is_b: bool) -> Self {
        Self {
            motion_model_idc: 1,
            l0: AffineListMv {
                pred_flag: true,
                ref_idx: 0,
                cp_mv: [MotionVector::default(); 3],
            },
            l1: AffineListMv {
                pred_flag: slice_is_b,
                ref_idx: if slice_is_b { 0 } else { -1 },
                cp_mv: [MotionVector::default(); 3],
            },
        }
    }
}

/// The motion state of one §8.5.3.4 / §8.5.3.6 **corner** — a single
/// translational neighbour (or collocated) sample's stored motion.
///
/// Corners feed the constructed-candidate arithmetic. `available == false`
/// marks an absent corner (the corner-availability loop never selected a
/// neighbour); the remaining fields are then ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CornerMv {
    /// The corner was filled (a §6.4.3-available neighbour matched, or the
    /// collocated fallback produced a vector).
    pub available: bool,
    /// `PredFlagL0` / `RefIdxL0` / `MvL0` at the selected corner sample.
    pub pred_flag_l0: bool,
    pub ref_idx_l0: i32,
    pub mv_l0: MotionVector,
    /// `PredFlagL1` / `RefIdxL1` / `MvL1`.
    pub pred_flag_l1: bool,
    pub ref_idx_l1: i32,
    pub mv_l1: MotionVector,
}

/// The four §8.5.3.4 corner CPMVs (top-left, top-right, bottom-left,
/// bottom-right), already resolved by the caller against the §8.5.3.4
/// spatial selection order (B2→B3→A2; B0→B1→C2; A0/A1; C0/C1) and the
/// corner-2/3 temporal fallback.
///
/// Indices match the spec's `cpMvLXCorner[ 0..3 ]`:
/// `[0] = top-left, [1] = top-right, [2] = bottom-left, [3] = bottom-right`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AffineCorners {
    pub corner: [CornerMv; 4],
}

/// One inherited-affine neighbour for §8.5.3.2 / §8.5.3.5: an
/// affine-coded neighbour block whose corner MVs project (via §8.5.3.3
/// [`inherited_cp_mvs`]) onto the current block's control points.
///
/// `available_flag` is the §8.5.3.2-step-3 `availableFlagBLK` (the
/// neighbour is §6.4.3-available **and** `MotionModelIdc > 0`). The two
/// per-list sources carry the neighbour's stored corner MVs + geometry
/// (the §8.5.3.3 [`NeighbourAffineSource`] input) plus its
/// `PredFlagLX` / `RefIdxLX`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineNeighbour {
    /// §8.5.3.2-step-3 `availableFlagBLK`.
    pub available_flag: bool,
    /// Neighbour `MotionModelIdc` (1 or 2) ⇒ current `numCpMv` (eq.: the
    /// step-5 `numCpMv = MotionModelIdc + 1`).
    pub motion_model_idc: u32,
    /// `PredFlagL0` / `RefIdxL0` at the neighbour.
    pub pred_flag_l0: bool,
    pub ref_idx_l0: i32,
    /// `PredFlagL1` / `RefIdxL1`.
    pub pred_flag_l1: bool,
    pub ref_idx_l1: i32,
    /// The §8.5.3.3 list-0 corner-MV source (geometry + 4 corner MVs).
    pub src_l0: NeighbourAffineSource,
    /// The §8.5.3.3 list-1 corner-MV source.
    pub src_l1: NeighbourAffineSource,
}

impl AffineNeighbour {
    /// Project this inherited neighbour into an [`AffineCand`] for the
    /// current block via §8.5.3.3 [`inherited_cp_mvs`] per active list.
    ///
    /// `numCpMv` is the neighbour's `motion_model_idc + 1` (§8.5.3.2
    /// step-5). The current block's CPMVs are derived independently for
    /// each list X with `PredFlagLX == 1`.
    fn as_affine_cand(
        &self,
        x_cb: i32,
        y_cb: i32,
        cb_width: u32,
        cb_height: u32,
        ctb_size_y: i32,
    ) -> AffineCand {
        let num_cp_mv = self.motion_model_idc + 1;
        let project = |used: bool, ref_idx: i32, src: NeighbourAffineSource| -> AffineListMv {
            if !used {
                return AffineListMv::absent();
            }
            let cps = inherited_cp_mvs(x_cb, y_cb, cb_width, cb_height, num_cp_mv, ctb_size_y, src);
            let mut cp_mv = [MotionVector::default(); 3];
            for (i, c) in cps.iter().enumerate().take(3) {
                cp_mv[i] = *c;
            }
            AffineListMv {
                pred_flag: true,
                ref_idx,
                cp_mv,
            }
        };
        AffineCand {
            motion_model_idc: self.motion_model_idc,
            l0: project(self.pred_flag_l0, self.ref_idx_l0, self.src_l0),
            l1: project(self.pred_flag_l1, self.ref_idx_l1, self.src_l1),
        }
    }
}

/// The five §8.5.3.2 inherited-neighbour slots, in the order the
/// step-7 list construction visits them.
///
/// The §8.5.3.2 step-3/-7 neighbour set depends on `availLR`:
/// `LR_01` visits `C1, B3, B2, C0, B0`; every other `availLR` visits
/// `A1, B1, B0, A0, B2`. The caller resolves each slot (running §6.4.3
/// availability + the `MotionModelIdc > 0` gate + the step-4 pruning)
/// and fills the array **in visiting order**, leaving
/// `available_flag == false` for absent neighbours.
pub type InheritedNeighbours = [AffineNeighbour; 5];

/// The ten §8.5.3.2 neighbour sample locations (eqs. 708-717), keyed by
/// the spec's `A0..C2` names. The merge inherited-neighbour scan and the
/// §8.5.3.4 corner scan both index into this set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineMergeNbPositions {
    pub a0: (i32, i32),
    pub a1: (i32, i32),
    pub a2: (i32, i32),
    pub b0: (i32, i32),
    pub b1: (i32, i32),
    pub b2: (i32, i32),
    pub b3: (i32, i32),
    pub c0: (i32, i32),
    pub c1: (i32, i32),
    pub c2: (i32, i32),
}

/// §8.5.3.2 step-1 (eqs. 708-717) — the ten affine-merge neighbour sample
/// locations for a luma block at `(x_cb, y_cb)` of size `cb_width` ×
/// `cb_height`.
pub fn affine_merge_nb_positions(
    x_cb: i32,
    y_cb: i32,
    cb_width: u32,
    cb_height: u32,
) -> AffineMergeNbPositions {
    let w = cb_width as i32;
    let h = cb_height as i32;
    AffineMergeNbPositions {
        a0: (x_cb - 1, y_cb + h),     // eq. 708
        a1: (x_cb - 1, y_cb + h - 1), // eq. 709
        a2: (x_cb - 1, y_cb),         // eq. 710
        b0: (x_cb + w, y_cb - 1),     // eq. 711
        b1: (x_cb + w - 1, y_cb - 1), // eq. 712
        b2: (x_cb - 1, y_cb - 1),     // eq. 713
        b3: (x_cb, y_cb - 1),         // eq. 714
        c0: (x_cb + w, y_cb + h),     // eq. 715
        c1: (x_cb + w, y_cb + h - 1), // eq. 716
        c2: (x_cb + w, y_cb),         // eq. 717
    }
}

/// §8.5.3.2 step-3/-7 — the inherited-neighbour visiting order, as named
/// tokens, selected by `availLR` (eqs. 720/721).
///
/// `LR_01` visits `[C1, B3, B2, C0, B0]`; every other `availLR` visits
/// `[A1, B1, B0, A0, B2]`. The decoder maps each token through
/// [`affine_merge_nb_positions`] to the sample location, resolves the
/// §6.4.3 availability + `MotionModelIdc > 0` gate, and fills the
/// [`InheritedNeighbours`] array in this order.
pub fn affine_merge_inherited_order(avail_lr: AvailLr) -> [AffineNbName; 5] {
    use AffineNbName::*;
    if avail_lr == AvailLr::Lr01 {
        [C1, B3, B2, C0, B0]
    } else {
        [A1, B1, B0, A0, B2]
    }
}

/// A §8.5.3.2 neighbour-position name (`A0..C2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineNbName {
    A0,
    A1,
    A2,
    B0,
    B1,
    B2,
    B3,
    C0,
    C1,
    C2,
}

impl AffineNbName {
    /// Resolve this name to its sample location in `pos`.
    pub fn location(self, pos: &AffineMergeNbPositions) -> (i32, i32) {
        match self {
            AffineNbName::A0 => pos.a0,
            AffineNbName::A1 => pos.a1,
            AffineNbName::A2 => pos.a2,
            AffineNbName::B0 => pos.b0,
            AffineNbName::B1 => pos.b1,
            AffineNbName::B2 => pos.b2,
            AffineNbName::B3 => pos.b3,
            AffineNbName::C0 => pos.c0,
            AffineNbName::C1 => pos.c1,
            AffineNbName::C2 => pos.c2,
        }
    }
}

/// The seven §8.5.3.5 affine-MVP neighbour sample locations (eqs.
/// 836-842): the A-group `(A0, A1)`, the B-group `(B0, B1, B2)`, and the
/// C-group `(C0, C1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineMvpNbPositions {
    pub a0: (i32, i32),
    pub a1: (i32, i32),
    pub b0: (i32, i32),
    pub b1: (i32, i32),
    pub b2: (i32, i32),
    pub c0: (i32, i32),
    pub c1: (i32, i32),
}

/// §8.5.3.5 step-3 (eqs. 836-842) — the seven affine-MVP neighbour sample
/// locations for a luma block at `(x_cb, y_cb)` of size `cb_width` ×
/// `cb_height`.
pub fn affine_mvp_nb_positions(
    x_cb: i32,
    y_cb: i32,
    cb_width: u32,
    cb_height: u32,
) -> AffineMvpNbPositions {
    let w = cb_width as i32;
    let h = cb_height as i32;
    AffineMvpNbPositions {
        a0: (x_cb - 1, y_cb + h),     // eq. 836
        a1: (x_cb - 1, y_cb + h - 1), // eq. 837
        b0: (x_cb + w, y_cb - 1),     // eq. 838
        b1: (x_cb + w - 1, y_cb - 1), // eq. 839
        b2: (x_cb - 1, y_cb - 1),     // eq. 840
        c0: (x_cb + w, y_cb + h),     // eq. 841
        c1: (x_cb + w, y_cb + h - 1), // eq. 842
    }
}

/// §8.5.3.4 — derive the six constructed affine merge candidates
/// `ConstK` with `K = 1..6` from the four resolved corner CPMVs.
///
/// The corner CPMVs are pre-resolved by the caller (the §8.5.3.4 spatial
/// scan B2→B3→A2 for corner 0, B0→B1→C2 for corner 1, A0/A1 + temporal
/// for corner 2, C0/C1 + temporal for corner 3). This function performs
/// the per-`Const` arithmetic of eqs. 798-835:
///
/// * Const1 (eqs. 798-802) — corners {0,1,2}, 6-param, no completion.
/// * Const2 (eqs. 803-809) — corners {0,1,3}, 6-param, eq.-807 corner-2
///   completion `cp3 + cp0 − cp1`.
/// * Const3 (eqs. 810-816) — corners {0,2,3}, 6-param, eq.-813 corner-1
///   completion `cp3 + cp0 − cp2`.
/// * Const4 (eqs. 817-823) — corners {1,2,3}, 6-param, eq.-819 corner-0
///   completion `cp1 + cp2 − cp3`.
/// * Const5 (eqs. 824-827) — corners {0,1}, 4-param.
/// * Const6 (eqs. 828-835) — corners {0,2}, 4-param, eqs. 828/829
///   corner-1 derivation from the affine model.
///
/// Per `Const`, a list X is "available" only when every required corner
/// has `predFlagLXCorner == 1` and all the required corners share the
/// same `refIdxLXCorner`. The whole candidate is available iff list 0 or
/// list 1 is available (the §8.5.3.4 `availableFlagConstK` derivation).
///
/// Returns the six candidates in order; an absent candidate is `None`.
pub fn constructed_merge_candidates(
    corners: &AffineCorners,
    cb_width: u32,
    cb_height: u32,
) -> [Option<AffineCand>; 6] {
    // Resolve, per list X, the K-th const's CPMVs given which corner
    // indices it combines and a completion that fills the missing corner.
    // `combine` returns the per-list (pred_flag, ref_idx, cp_mv[..n]).
    let log2_w = cb_width.trailing_zeros() as i32;
    let log2_h = cb_height.trailing_zeros() as i32;

    // Helper: read corner c's per-list (pred_flag, ref_idx, mv).
    let read = |c: usize, list0: bool| -> (bool, i32, MotionVector) {
        let cm = corners.corner[c];
        if list0 {
            (cm.pred_flag_l0, cm.ref_idx_l0, cm.mv_l0)
        } else {
            (cm.pred_flag_l1, cm.ref_idx_l1, cm.mv_l1)
        }
    };

    // Generic 3-corner const (Const1..4): combine corner indices `a,b,d`
    // for the availability test (pred + matching refIdx of the *first*
    // listed corner), then place the three resulting CPMVs at the
    // model-order positions with `completion` supplying any derived CPMV.
    //
    // `place` maps the resolved corner MVs into cpMv[0..3] for the
    // candidate (handling the eq.-807/813/819 corner completion).
    let three_corner = |req: [usize; 3],
                        place: &dyn Fn([MotionVector; 3]) -> [MotionVector; 3]|
     -> Option<AffineCand> {
        let per_list = |list0: bool| -> Option<AffineListMv> {
            let (p0, r0, m0) = read(req[0], list0);
            let (p1, r1, m1) = read(req[1], list0);
            let (p2, r2, m2) = read(req[2], list0);
            if !(p0 && p1 && p2 && r0 == r1 && r0 == r2) {
                return None;
            }
            let placed = place([m0, m1, m2]);
            Some(AffineListMv {
                pred_flag: true,
                ref_idx: r0,
                cp_mv: placed,
            })
        };
        let l0 = per_list(true);
        let l1 = per_list(false);
        if l0.is_none() && l1.is_none() {
            return None;
        }
        Some(AffineCand {
            motion_model_idc: 2,
            l0: l0.unwrap_or_else(AffineListMv::absent),
            l1: l1.unwrap_or_else(AffineListMv::absent),
        })
    };

    // Const1: corners {0,1,2}, placed directly (eqs. 800-802).
    let const1 = three_corner([0, 1, 2], &|m| [m[0], m[1], m[2]]);

    // Const2: corners {0,1,3}. cp[2] = clip(cp3 + cp0 − cp1) (eq. 807).
    let const2 = three_corner([0, 1, 3], &|m| {
        let c2 = MotionVector {
            x: clip_cpmv_component(m[2].x + m[0].x - m[1].x),
            y: clip_cpmv_component(m[2].y + m[0].y - m[1].y),
        };
        [m[0], m[1], c2]
    });

    // Const3: corners {0,2,3}. cp[1] = clip(cp3 + cp0 − cp2) (eq. 813);
    // cp[2] = cp2.  m = [cp0, cp2, cp3].
    let const3 = three_corner([0, 2, 3], &|m| {
        let c1 = MotionVector {
            x: clip_cpmv_component(m[2].x + m[0].x - m[1].x),
            y: clip_cpmv_component(m[2].y + m[0].y - m[1].y),
        };
        [m[0], c1, m[1]]
    });

    // Const4: corners {1,2,3}. cp[0] = clip(cp1 + cp2 − cp3) (eq. 819);
    // cp[1] = cp1; cp[2] = cp2.  m = [cp1, cp2, cp3].
    let const4 = three_corner([1, 2, 3], &|m| {
        let c0 = MotionVector {
            x: clip_cpmv_component(m[0].x + m[1].x - m[2].x),
            y: clip_cpmv_component(m[0].y + m[1].y - m[2].y),
        };
        [c0, m[0], m[1]]
    });

    // Const5: corners {0,1}, 4-param (eqs. 824-827).
    let const5 = two_corner_const(corners, [0, 1], None, log2_w, log2_h);

    // Const6: corners {0,2}, 4-param. cp[1] derived from the model
    // (eqs. 828/829) — pass the derivation marker.
    let const6 = two_corner_const(corners, [0, 2], Some(()), log2_w, log2_h);

    [const1, const2, const3, const4, const5, const6]
}

/// Const5 / Const6 helper (4-parameter, two corners).
///
/// `req = [0,1]` → Const5: cp[0]=corner0, cp[1]=corner1 directly.
/// `req = [0,2]` with `derive_cp1 = Some` → Const6: cp[0]=corner0,
/// cp[1] derived from the affine model (eqs. 828/829) then §8.5.3.10
/// `rightShift = 7` rounded and clipped (eqs. 834/835).
fn two_corner_const(
    corners: &AffineCorners,
    req: [usize; 2],
    derive_cp1: Option<()>,
    log2_w: i32,
    log2_h: i32,
) -> Option<AffineCand> {
    let read = |c: usize, list0: bool| -> (bool, i32, MotionVector) {
        let cm = corners.corner[c];
        if list0 {
            (cm.pred_flag_l0, cm.ref_idx_l0, cm.mv_l0)
        } else {
            (cm.pred_flag_l1, cm.ref_idx_l1, cm.mv_l1)
        }
    };
    let per_list = |list0: bool| -> Option<AffineListMv> {
        let (p0, r0, m0) = read(req[0], list0);
        let (p1, r1, m1) = read(req[1], list0);
        if !(p0 && p1 && r0 == r1) {
            return None;
        }
        let cp1 = if derive_cp1.is_some() {
            // eqs. 828/829 — derive top-right from top-left (m0) and
            // bottom-left (m1) of a 4-param model.
            // shift = 7 + Log2(cbWidth) − Log2(cbHeight / cbWidth)
            //       = 7 + log2_w − (log2_h − log2_w) = 7 + 2*log2_w − log2_h.
            let shift = 7 + 2 * log2_w - log2_h;
            let cp1x_pre = (m0.x << 7) + ((m1.y - m0.y) << shift);
            let cp1y_pre = (m0.y << 7) - ((m1.x - m0.x) << shift);
            // §8.5.3.10 rightShift = 7, leftShift = 0.
            let rounded = crate::inter::round_motion_vector(
                MotionVector {
                    x: cp1x_pre,
                    y: cp1y_pre,
                },
                7,
                0,
            );
            MotionVector {
                x: clip_cpmv_component(rounded.x),
                y: clip_cpmv_component(rounded.y),
            }
        } else {
            m1
        };
        Some(AffineListMv {
            pred_flag: true,
            ref_idx: r0,
            cp_mv: [m0, cp1, MotionVector::default()],
        })
    };
    let l0 = per_list(true);
    let l1 = per_list(false);
    if l0.is_none() && l1.is_none() {
        return None;
    }
    Some(AffineCand {
        motion_model_idc: 1,
        l0: l0.unwrap_or_else(AffineListMv::absent),
        l1: l1.unwrap_or_else(AffineListMv::absent),
    })
}

/// §8.5.3.2 — assemble `affineMergeCandList` (≤ 5 entries).
///
/// Steps:
/// * 7 — the up-to-5 inherited (model-based) neighbour candidates, in the
///   `availLR`-determined order (the `available_flag == true` slots of
///   `inherited`), capped at 5.
/// * 7 (continued) — the §8.5.3.4 constructed candidates Const1..Const6,
///   appended while the list has fewer than 5 entries.
/// * 9 — zero-CPMV `zeroCandm` (eqs. 723-733) repeated until the list has
///   exactly 5 entries.
///
/// `inherited` is already in visiting order (see [`InheritedNeighbours`]);
/// `corners` feeds [`constructed_merge_candidates`]. `slice_is_b` selects
/// the zero-candidate L1 utilization. `(x_cb, y_cb)` / `cb_width` /
/// `cb_height` / `ctb_size_y` drive the §8.5.3.3 inherited projection.
#[allow(clippy::too_many_arguments)]
pub fn build_affine_merge_cand_list(
    inherited: &InheritedNeighbours,
    corners: &AffineCorners,
    slice_is_b: bool,
    x_cb: i32,
    y_cb: i32,
    cb_width: u32,
    cb_height: u32,
    ctb_size_y: i32,
) -> Vec<AffineCand> {
    let mut list: Vec<AffineCand> = Vec::with_capacity(5);

    // Step 7 — inherited (model-based) candidates.
    for nb in inherited.iter() {
        if list.len() >= 5 {
            break;
        }
        if nb.available_flag {
            list.push(nb.as_affine_cand(x_cb, y_cb, cb_width, cb_height, ctb_size_y));
        }
    }

    // Step 7 (continued) — constructed candidates Const1..Const6.
    let consts = constructed_merge_candidates(corners, cb_width, cb_height);
    for c in consts.into_iter().flatten() {
        if list.len() >= 5 {
            break;
        }
        list.push(c);
    }

    // Step 9 — zero-CPMV tail.
    while list.len() < 5 {
        list.push(AffineCand::zero(slice_is_b));
    }

    list
}

/// §8.5.3.2 step-after-9 — project a decoded `affine_merge_idx` into the
/// selected candidate.
///
/// Returns the `affineMergeCandList[ affine_merge_idx ]` entry; the caller
/// reads its `num_cp_mv()` + per-list CPMVs/refIdx/predFlag for the
/// §8.5.3.7 subblock-MV derivation (eqs. 735-741). `merge_idx` is bounded
/// by 4 (the TR-coded `affine_merge_idx`, cMax = 4); an out-of-range index
/// returns `None`.
pub fn select_affine_merge_candidate(list: &[AffineCand], merge_idx: usize) -> Option<AffineCand> {
    list.get(merge_idx).copied()
}

/// §8.5.3.6 — derive the single constructed affine MVP candidate.
///
/// The four corners are pre-resolved by the caller against the §8.5.3.6
/// selection order (B2→B3→A2 for corner 0; B0→B1→C2 for corner 1; A1→A0
/// for corner 2; C1→C0 for corner 3) **and** the refIdx match: a corner
/// is `available` only when `PredFlagLX == 1` and `RefIdxLX == refIdxLX`
/// for the current list X (so the caller has already filtered on refIdx
/// and the per-corner availability fields here are the post-match flags).
///
/// Returns `(availableConsFlag, cpMv[0..3])` for the single list:
/// * If corners 0,1,2 are all available → `cpMv = [c0, c1, c2]`,
///   availableConsFlag = 1.
/// * Else if corners 0,1,3 available (corner 2 absent) → corner 2 is
///   completed `clip(c3 + c0 − c1)` (eq. 872), availableConsFlag = 1.
/// * Else if corners 0,1 available and `MotionModelIdc[xCb][yCb] == 1`
///   (4-param current block) → availableConsFlag = 1, `cpMv = [c0, c1]`.
/// * Else availableConsFlag = 0.
///
/// `corner_mv` carries the four refIdx-matched MVs; `corner_avail` their
/// post-match availability. `cur_motion_model_idc` is the current block's
/// `MotionModelIdc` (1 or 2).
pub fn constructed_mvp_candidate(
    corner_avail: [bool; 4],
    corner_mv: [MotionVector; 4],
    cur_motion_model_idc: u32,
) -> (bool, [MotionVector; 3]) {
    let [a0, a1, a2, a3] = corner_avail;
    let [c0, c1, c2, c3] = corner_mv;

    if a0 && a1 && a2 {
        (true, [c0, c1, c2])
    } else if a0 && a1 && !a2 && a3 {
        // eq. 872 — complete corner 2 from corner 3 + corner 0 − corner 1.
        let derived = MotionVector {
            x: clip_cpmv_component(c3.x + c0.x - c1.x),
            y: clip_cpmv_component(c3.y + c0.y - c1.y),
        };
        (true, [c0, c1, derived])
    } else if a0 && a1 && cur_motion_model_idc == 1 {
        (true, [c0, c1, MotionVector::default()])
    } else {
        (false, [MotionVector::default(); 3])
    }
}

/// One inherited MVP-neighbour source for §8.5.3.5: an affine-coded
/// neighbour whose CPMVs project (§8.5.3.3) onto the current block, but
/// only after the §8.5.3.5 refIdx-match gate (`PredFlagLX == 1` and
/// `RefIdxLX == refIdxLX`). The caller supplies the post-gate
/// availability + the §8.5.3.3 source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineMvpNeighbour {
    /// The neighbour passed the §6.4.3 availability + `MotionModelIdc > 0`
    /// + `PredFlagLX == 1` + `RefIdxLX == refIdxLX` gate.
    pub matched: bool,
    /// The §8.5.3.3 corner-MV source.
    pub src: NeighbourAffineSource,
}

impl AffineMvpNeighbour {
    /// An absent (unmatched) neighbour with a placeholder source. Used by
    /// callers to pre-fill the group arrays before resolving each slot.
    pub fn absent() -> Self {
        Self {
            matched: false,
            src: NeighbourAffineSource {
                x_nb: 0,
                y_nb: 0,
                n_nb_w: 1,
                n_nb_h: 1,
                mv_tl: MotionVector::default(),
                mv_tr: MotionVector::default(),
                mv_bl: MotionVector::default(),
                mv_br: MotionVector::default(),
                motion_model_idc: 1,
            },
        }
    }
}

/// The §8.5.3.5 inherited-MVP neighbour groups A / B / C. Each group is
/// scanned in order and the **first** matched neighbour contributes one
/// CPMV-predictor list entry (`availableFlagA/B/C`).
///
/// * `group_a` — `(A0, A1)` (eqs. 836/837), scanned A0→A1.
/// * `group_b` — `(B0, B1, B2)` (eqs. 838-840), scanned B0→B1→B2.
/// * `group_c` — `(C0, C1)` (eqs. 841/842), scanned C0→C1, only consulted
///   when fewer than 2 candidates so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineMvpNeighbours {
    pub group_a: [AffineMvpNeighbour; 2],
    pub group_b: [AffineMvpNeighbour; 3],
    pub group_c: [AffineMvpNeighbour; 2],
}

/// §8.5.3.5 — assemble the affine MVP candidate list `cpMvpListLX` for one
/// list X (exactly 2 entries).
///
/// Ordered steps (eqs. 836-866):
/// * 4 — first matched group-A neighbour → inherited CPMV predictor.
/// * 5 — first matched group-B neighbour → inherited CPMV predictor.
/// * 6 — first matched group-C neighbour (only while `< 2`).
/// * 7 — the §8.5.3.6 constructed predictor (only while `< 2`; appended
///   only when `numCpMvpCand == 0`).
/// * 8 — the per-corner translational fill (cpIdx 2→0, with the cpIdx-2
///   → cpIdx-3 redirect when corner 2 unavailable), each replicated
///   across all control points (eqs. 859-861).
/// * 9 — zero-CPMV fill to 2.
///
/// `num_cp_mv` is the current block's `MotionModelIdc + 1`. Each inherited
/// predictor is a full `numCpMv`-CPMV set; the constructed/corner/zero
/// fills replicate a single MV across all control points (the eqs.
/// 855-866 `cpMvpListLX[..][cpIdx] = mv` for all cpIdx pattern, which the
/// §8.5.3.7 derivation reads as a translational model).
///
/// `constructed` is `(availableConsFlag, cpMv[0..3])` from
/// [`constructed_mvp_candidate`]; `corner_avail` / `corner_mv` are the
/// four refIdx-matched corners for the step-8 translational fill.
#[allow(clippy::too_many_arguments)]
pub fn build_affine_mvp_cand_list(
    neigh: &AffineMvpNeighbours,
    constructed: (bool, [MotionVector; 3]),
    corner_avail: [bool; 4],
    corner_mv: [MotionVector; 4],
    num_cp_mv: u32,
    x_cb: i32,
    y_cb: i32,
    cb_width: u32,
    cb_height: u32,
    ctb_size_y: i32,
) -> [[ControlPointMv; 3]; 2] {
    let mut list: Vec<[ControlPointMv; 3]> = Vec::with_capacity(2);

    let project = |src: NeighbourAffineSource| -> [ControlPointMv; 3] {
        let cps = inherited_cp_mvs(x_cb, y_cb, cb_width, cb_height, num_cp_mv, ctb_size_y, src);
        let mut out = [MotionVector::default(); 3];
        for (i, c) in cps.iter().enumerate().take(3) {
            out[i] = *c;
        }
        out
    };

    // Step 4 — first matched group-A neighbour.
    if let Some(nb) = neigh.group_a.iter().find(|n| n.matched) {
        if list.len() < 2 {
            list.push(project(nb.src));
        }
    }

    // Step 5 — first matched group-B neighbour.
    if let Some(nb) = neigh.group_b.iter().find(|n| n.matched) {
        if list.len() < 2 {
            list.push(project(nb.src));
        }
    }

    // Step 6 — first matched group-C neighbour (only while < 2).
    if list.len() < 2 {
        if let Some(nb) = neigh.group_c.iter().find(|n| n.matched) {
            list.push(project(nb.src));
        }
    }

    // Step 7 — constructed predictor (only while < 2 AND numCpMvpCand == 0).
    if list.len() < 2 && list.is_empty() && constructed.0 {
        list.push(constructed.1);
    }

    // Step 8 — per-corner translational fill, cpIdx = 2..0 (with the
    // cpIdx-2 → cpIdx-3 redirect when corner 2 unavailable).
    for cp_idx_raw in (0..=2).rev() {
        if list.len() >= 2 {
            break;
        }
        // eq.-before-859: if cpIdx == 2 and availableFlag[2] == 0, use 3.
        let cp_idx = if cp_idx_raw == 2 && !corner_avail[2] {
            3
        } else {
            cp_idx_raw
        };
        if corner_avail[cp_idx] {
            let mv = corner_mv[cp_idx];
            // eqs. 859-861 — replicate across all control points.
            list.push([mv, mv, mv]);
        }
    }

    // Step 9 — zero-CPMV fill to 2.
    while list.len() < 2 {
        list.push([MotionVector::default(); 3]);
    }

    [list[0], list[1]]
}

/// §8.5.3.5 eq. 867 + §8.5.3.1 eqs. 688-691 — the AMVP affine
/// CPMV-reconstruction bridge.
///
/// Selects the predictor `cpMvpListLX[ affine_mvp_flag_lX ]` (eq. 867)
/// from the [`build_affine_mvp_cand_list`] output, then reconstructs each
/// control point's final motion vector by adding the decoded MVD per
/// §8.5.3.1 (the 16-bit modular wrap of `mvpCp + mvdCp`, eqs. 688-691,
/// via [`crate::affine::reconstruct_cp_mv`]).
///
/// `mvp_list` is the two-entry list for list X; `mvp_flag` is the decoded
/// `affine_mvp_flag_lX` ∈ {0, 1}; `mvd_cp[ cpIdx ]` are the per-vertex
/// decoded MVDs (all-zero when `affine_mvd_flag_lX == 0`, the §7.4 absent
/// inference). `num_cp_mv` (2 or 3) bounds which control points are
/// reconstructed; the unused tail entry is left at the predictor's value
/// (never read by the §8.5.3.7 derivation, which honours `numCpMv`).
pub fn reconstruct_affine_amvp_cp_mvs(
    mvp_list: &[[ControlPointMv; 3]; 2],
    mvp_flag: u32,
    mvd_cp: &[MotionVector; 3],
    num_cp_mv: u32,
) -> [ControlPointMv; 3] {
    let mvp = mvp_list[(mvp_flag & 1) as usize];
    let mut out = mvp;
    for (cp_idx, slot) in out.iter_mut().enumerate().take(num_cp_mv as usize) {
        *slot = crate::affine::reconstruct_cp_mv(mvp[cp_idx], mvd_cp[cp_idx]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    fn corner(avail: bool, p0: bool, r0: i32, m0: MotionVector) -> CornerMv {
        CornerMv {
            available: avail,
            pred_flag_l0: p0,
            ref_idx_l0: r0,
            mv_l0: m0,
            pred_flag_l1: false,
            ref_idx_l1: -1,
            mv_l1: MotionVector::default(),
        }
    }

    /// Const1 (corners 0,1,2 all L0-available, matching refIdx) is placed
    /// directly with the three corner MVs (eqs. 800-802), 6-param.
    #[test]
    fn const1_direct_placement() {
        let mut corners = AffineCorners::default();
        corners.corner[0] = corner(true, true, 0, mv(10, 20));
        corners.corner[1] = corner(true, true, 0, mv(12, 22));
        corners.corner[2] = corner(true, true, 0, mv(8, 24));
        let consts = constructed_merge_candidates(&corners, 16, 16);
        let c1 = consts[0].expect("Const1 available");
        assert_eq!(c1.motion_model_idc, 2);
        assert!(c1.l0.pred_flag);
        assert_eq!(c1.l0.ref_idx, 0);
        assert_eq!(c1.l0.cp_mv[0], mv(10, 20));
        assert_eq!(c1.l0.cp_mv[1], mv(12, 22));
        assert_eq!(c1.l0.cp_mv[2], mv(8, 24));
    }

    /// Const1 requires matching refIdx across corners 0,1,2; a mismatch
    /// makes the list-0 candidate unavailable, and since L1 is absent the
    /// whole Const1 is `None`.
    #[test]
    fn const1_refidx_mismatch_unavailable() {
        let mut corners = AffineCorners::default();
        corners.corner[0] = corner(true, true, 0, mv(10, 20));
        corners.corner[1] = corner(true, true, 1, mv(12, 22)); // refIdx 1
        corners.corner[2] = corner(true, true, 0, mv(8, 24));
        let consts = constructed_merge_candidates(&corners, 16, 16);
        assert!(consts[0].is_none());
    }

    /// Const2 completes corner 2 as clip(c3 + c0 − c1) (eq. 807) from
    /// corners {0,1,3}.
    #[test]
    fn const2_corner2_completion() {
        let mut corners = AffineCorners::default();
        corners.corner[0] = corner(true, true, 0, mv(10, 20));
        corners.corner[1] = corner(true, true, 0, mv(12, 22));
        // corner 2 absent; corner 3 present.
        corners.corner[3] = corner(true, true, 0, mv(40, 50));
        let consts = constructed_merge_candidates(&corners, 16, 16);
        let c2 = consts[1].expect("Const2 available");
        // cp[2] = c3 + c0 − c1 = (40+10−12, 50+20−22) = (38, 48).
        assert_eq!(c2.l0.cp_mv[0], mv(10, 20));
        assert_eq!(c2.l0.cp_mv[1], mv(12, 22));
        assert_eq!(c2.l0.cp_mv[2], mv(38, 48));
    }

    /// Const4 completes corner 0 as clip(c1 + c2 − c3) (eq. 819) from
    /// corners {1,2,3}.
    #[test]
    fn const4_corner0_completion() {
        let mut corners = AffineCorners::default();
        corners.corner[1] = corner(true, true, 0, mv(12, 22));
        corners.corner[2] = corner(true, true, 0, mv(8, 24));
        corners.corner[3] = corner(true, true, 0, mv(40, 50));
        let consts = constructed_merge_candidates(&corners, 16, 16);
        let c4 = consts[3].expect("Const4 available");
        // cp[0] = c1 + c2 − c3 = (12+8−40, 22+24−50) = (−20, −4).
        assert_eq!(c4.l0.cp_mv[0], mv(-20, -4));
        assert_eq!(c4.l0.cp_mv[1], mv(12, 22));
        assert_eq!(c4.l0.cp_mv[2], mv(8, 24));
    }

    /// Const5 (corners 0,1) is a 4-param candidate placed directly.
    #[test]
    fn const5_4param() {
        let mut corners = AffineCorners::default();
        corners.corner[0] = corner(true, true, 0, mv(10, 20));
        corners.corner[1] = corner(true, true, 0, mv(12, 22));
        let consts = constructed_merge_candidates(&corners, 16, 16);
        let c5 = consts[4].expect("Const5 available");
        assert_eq!(c5.motion_model_idc, 1);
        assert_eq!(c5.l0.cp_mv[0], mv(10, 20));
        assert_eq!(c5.l0.cp_mv[1], mv(12, 22));
    }

    /// Const6 (corners 0,2) derives cp[1] from the 4-param affine model
    /// (eqs. 828/829). For a square block (cbW == cbH) the shift is
    /// 7 + 2·log2W − log2H = 7 + log2W, and rounding by 7 brings it back.
    #[test]
    fn const6_model_derivation_square() {
        let mut corners = AffineCorners::default();
        corners.corner[0] = corner(true, true, 0, mv(10, 20));
        corners.corner[2] = corner(true, true, 0, mv(14, 26));
        let consts = constructed_merge_candidates(&corners, 16, 16);
        let c6 = consts[5].expect("Const6 available");
        assert_eq!(c6.motion_model_idc, 1);
        assert_eq!(c6.l0.cp_mv[0], mv(10, 20));
        // shift = 7 + 2*4 − 4 = 11.
        // cp1x_pre = (10<<7) + ((26−20)<<11) = 1280 + 12288 = 13568
        // round(13568, 7) = (13568 + 64 − 1) >> 7 = 13631 >> 7 = 106
        // cp1y_pre = (20<<7) − ((14−10)<<11) = 2560 − 8192 = −5632
        // round(−5632,7) = (−5632 + 64 − 0) >> 7 = −5568 >> 7 = −44
        assert_eq!(c6.l0.cp_mv[1], mv(106, -44));
    }

    /// The merge list is filled to exactly 5 entries: with no inherited
    /// and no constructed candidates the tail is all zero-CPMV
    /// candidates (eqs. 723-733).
    #[test]
    fn merge_list_zero_fill_p_slice() {
        let inherited: InheritedNeighbours = [AffineNeighbour {
            available_flag: false,
            motion_model_idc: 1,
            pred_flag_l0: false,
            ref_idx_l0: -1,
            pred_flag_l1: false,
            ref_idx_l1: -1,
            src_l0: dummy_src(),
            src_l1: dummy_src(),
        }; 5];
        let corners = AffineCorners::default();
        let list = build_affine_merge_cand_list(&inherited, &corners, false, 0, 0, 16, 16, 128);
        assert_eq!(list.len(), 5);
        for cand in &list {
            assert_eq!(cand.motion_model_idc, 1);
            assert!(cand.l0.pred_flag);
            assert_eq!(cand.l0.ref_idx, 0);
            // P-slice: L1 unused.
            assert!(!cand.l1.pred_flag);
            assert_eq!(cand.l1.ref_idx, -1);
        }
    }

    /// B-slice zero candidates use L1 (refIdx 0, predFlag 1).
    #[test]
    fn merge_list_zero_fill_b_slice() {
        let inherited: InheritedNeighbours = [AffineNeighbour {
            available_flag: false,
            motion_model_idc: 1,
            pred_flag_l0: false,
            ref_idx_l0: -1,
            pred_flag_l1: false,
            ref_idx_l1: -1,
            src_l0: dummy_src(),
            src_l1: dummy_src(),
        }; 5];
        let corners = AffineCorners::default();
        let list = build_affine_merge_cand_list(&inherited, &corners, true, 0, 0, 16, 16, 128);
        assert_eq!(list.len(), 5);
        assert!(list[0].l1.pred_flag);
        assert_eq!(list[0].l1.ref_idx, 0);
    }

    /// Constructed candidates follow the inherited ones in the list, and
    /// the list is capped at 5.
    #[test]
    fn merge_list_constructed_after_inherited() {
        // Two available inherited neighbours.
        let nb = AffineNeighbour {
            available_flag: true,
            motion_model_idc: 1,
            pred_flag_l0: true,
            ref_idx_l0: 0,
            pred_flag_l1: false,
            ref_idx_l1: -1,
            src_l0: NeighbourAffineSource {
                x_nb: 0,
                y_nb: 0,
                n_nb_w: 16,
                n_nb_h: 16,
                mv_tl: mv(4, 4),
                mv_tr: mv(4, 4),
                mv_bl: mv(4, 4),
                mv_br: mv(4, 4),
                motion_model_idc: 1,
            },
            src_l1: dummy_src(),
        };
        let absent = AffineNeighbour {
            available_flag: false,
            ..nb
        };
        let inherited: InheritedNeighbours = [nb, nb, absent, absent, absent];
        // A full Const5 available (corners 0,1 L0).
        let mut corners = AffineCorners::default();
        corners.corner[0] = corner(true, true, 0, mv(10, 20));
        corners.corner[1] = corner(true, true, 0, mv(12, 22));
        let list = build_affine_merge_cand_list(&inherited, &corners, false, 32, 32, 16, 16, 128);
        assert_eq!(list.len(), 5);
        // First two are inherited (translational, mv 4,4 since neighbour is
        // a pure-translation affine field).
        assert_eq!(list[0].l0.cp_mv[0], mv(4, 4));
        // Index 2 onward includes the constructed Const5 then zero fill.
        // select bridge returns the indexed entry.
        let sel = select_affine_merge_candidate(&list, 2).unwrap();
        assert_eq!(sel.l0.cp_mv[0], mv(10, 20));
    }

    /// §8.5.3.6 constructed predictor: corners 0,1,2 available → direct
    /// 3-CPMV.
    #[test]
    fn mvp_constructed_three_corners() {
        let (avail, cp) = constructed_mvp_candidate(
            [true, true, true, false],
            [mv(1, 1), mv(2, 2), mv(3, 3), mv(0, 0)],
            2,
        );
        assert!(avail);
        assert_eq!(cp, [mv(1, 1), mv(2, 2), mv(3, 3)]);
    }

    /// §8.5.3.6: corners 0,1,3 (corner 2 absent) → corner 2 completed
    /// clip(c3 + c0 − c1) (eq. 872).
    #[test]
    fn mvp_constructed_corner2_completion() {
        let (avail, cp) = constructed_mvp_candidate(
            [true, true, false, true],
            [mv(10, 10), mv(20, 20), mv(0, 0), mv(40, 40)],
            2,
        );
        assert!(avail);
        // c2 = 40 + 10 − 20 = 30.
        assert_eq!(cp[2], mv(30, 30));
    }

    /// §8.5.3.6: corners 0,1 with current 4-param model → available,
    /// 2-CPMV.
    #[test]
    fn mvp_constructed_two_corner_4param() {
        let (avail, cp) = constructed_mvp_candidate(
            [true, true, false, false],
            [mv(5, 5), mv(6, 6), mv(0, 0), mv(0, 0)],
            1,
        );
        assert!(avail);
        assert_eq!(cp[0], mv(5, 5));
        assert_eq!(cp[1], mv(6, 6));
    }

    /// §8.5.3.6: corners 0,1 with current 6-param model and no corner
    /// 2/3 → unavailable (the eq.-after-872 final else).
    #[test]
    fn mvp_constructed_unavailable_6param() {
        let (avail, _) = constructed_mvp_candidate(
            [true, true, false, false],
            [mv(5, 5), mv(6, 6), mv(0, 0), mv(0, 0)],
            2,
        );
        assert!(!avail);
    }

    /// §8.5.3.5 MVP list always returns exactly 2 entries; with no matched
    /// neighbours, no constructed candidate, and no corners, both are
    /// zero-CPMV (eqs. 863-866).
    #[test]
    fn mvp_list_zero_fill() {
        let neigh = AffineMvpNeighbours {
            group_a: [AffineMvpNeighbour::absent(); 2],
            group_b: [AffineMvpNeighbour::absent(); 3],
            group_c: [AffineMvpNeighbour::absent(); 2],
        };
        let list = build_affine_mvp_cand_list(
            &neigh,
            (false, [MotionVector::default(); 3]),
            [false; 4],
            [MotionVector::default(); 4],
            2,
            0,
            0,
            16,
            16,
            128,
        );
        assert_eq!(list[0], [MotionVector::default(); 3]);
        assert_eq!(list[1], [MotionVector::default(); 3]);
    }

    /// §8.5.3.5 step-8 corner fill: with no neighbours/constructed but a
    /// corner-0 available, the first list entry replicates corner 0
    /// across all control points (eqs. 859-861).
    #[test]
    fn mvp_list_corner_fill() {
        let neigh = AffineMvpNeighbours {
            group_a: [AffineMvpNeighbour::absent(); 2],
            group_b: [AffineMvpNeighbour::absent(); 3],
            group_c: [AffineMvpNeighbour::absent(); 2],
        };
        // corner 0 available; corner 2 not (so cpIdx 2 redirects to 3,
        // also absent), corner 1 available too.
        let list = build_affine_mvp_cand_list(
            &neigh,
            (false, [MotionVector::default(); 3]),
            [true, true, false, false],
            [mv(7, 8), mv(9, 10), mv(0, 0), mv(0, 0)],
            2,
            0,
            0,
            16,
            16,
            128,
        );
        // Step 8 visits cpIdx 2 (redirected to 3, absent), then 1, then 0.
        // First filled: corner 1 → [9,10]×3; second: corner 0 → [7,8]×3.
        assert_eq!(list[0], [mv(9, 10), mv(9, 10), mv(9, 10)]);
        assert_eq!(list[1], [mv(7, 8), mv(7, 8), mv(7, 8)]);
    }

    /// §8.5.3.5 eq. 867 — the AMVP bridge selects `cpMvpListLX[mvp_flag]`
    /// and adds the per-CP MVD (§8.5.3.1). 6-param: all 3 CPs reconstructed.
    #[test]
    fn amvp_reconstruct_selects_and_adds_mvd() {
        let mvp_list = [
            [mv(100, 200), mv(110, 210), mv(120, 220)],
            [mv(5, 5), mv(6, 6), mv(7, 7)],
        ];
        let mvd = [mv(1, -2), mv(3, -4), mv(5, -6)];
        // mvp_flag = 0 → first entry; add MVDs.
        let cp = reconstruct_affine_amvp_cp_mvs(&mvp_list, 0, &mvd, 3);
        assert_eq!(cp[0], mv(101, 198));
        assert_eq!(cp[1], mv(113, 206));
        assert_eq!(cp[2], mv(125, 214));
    }

    /// AMVP bridge honours `mvp_flag == 1` (selects the second predictor)
    /// and `num_cp_mv == 2` (only 2 CPs reconstructed; the 3rd stays at the
    /// predictor value, unread downstream).
    #[test]
    fn amvp_reconstruct_flag1_4param() {
        let mvp_list = [
            [mv(100, 200), mv(110, 210), mv(120, 220)],
            [mv(5, 5), mv(6, 6), mv(7, 7)],
        ];
        let mvd = [mv(10, 10), mv(20, 20), mv(0, 0)];
        let cp = reconstruct_affine_amvp_cp_mvs(&mvp_list, 1, &mvd, 2);
        assert_eq!(cp[0], mv(15, 15));
        assert_eq!(cp[1], mv(26, 26));
        // 3rd CP untouched (predictor value).
        assert_eq!(cp[2], mv(7, 7));
    }

    /// §8.5.3.2 step-1 positions (eqs. 708-717) at a 16×16 block placed at
    /// (32, 48).
    #[test]
    fn merge_nb_positions_eqs_708_717() {
        let p = affine_merge_nb_positions(32, 48, 16, 16);
        assert_eq!(p.a0, (31, 64)); // (xCb−1, yCb+H)
        assert_eq!(p.a1, (31, 63)); // (xCb−1, yCb+H−1)
        assert_eq!(p.a2, (31, 48)); // (xCb−1, yCb)
        assert_eq!(p.b0, (48, 47)); // (xCb+W, yCb−1)
        assert_eq!(p.b1, (47, 47)); // (xCb+W−1, yCb−1)
        assert_eq!(p.b2, (31, 47)); // (xCb−1, yCb−1)
        assert_eq!(p.b3, (32, 47)); // (xCb, yCb−1)
        assert_eq!(p.c0, (48, 64)); // (xCb+W, yCb+H)
        assert_eq!(p.c1, (48, 63)); // (xCb+W, yCb+H−1)
        assert_eq!(p.c2, (48, 48)); // (xCb+W, yCb)
    }

    /// §8.5.3.2 step-7 visiting order: LR_01 vs everything else (eqs.
    /// 720/721).
    #[test]
    fn merge_inherited_order_by_avail_lr() {
        use AffineNbName::*;
        assert_eq!(
            affine_merge_inherited_order(AvailLr::Lr01),
            [C1, B3, B2, C0, B0]
        );
        assert_eq!(
            affine_merge_inherited_order(AvailLr::Lr10),
            [A1, B1, B0, A0, B2]
        );
        assert_eq!(
            affine_merge_inherited_order(AvailLr::Lr00),
            [A1, B1, B0, A0, B2]
        );
        assert_eq!(
            affine_merge_inherited_order(AvailLr::Lr11),
            [A1, B1, B0, A0, B2]
        );
    }

    /// `AffineNbName::location` resolves a token through the position set.
    #[test]
    fn nb_name_location_resolution() {
        let p = affine_merge_nb_positions(0, 0, 16, 16);
        assert_eq!(AffineNbName::C1.location(&p), p.c1);
        assert_eq!(AffineNbName::A0.location(&p), p.a0);
    }

    /// §8.5.3.5 step-3 positions (eqs. 836-842) at a 16×16 block at (32,48).
    #[test]
    fn mvp_nb_positions_eqs_836_842() {
        let p = affine_mvp_nb_positions(32, 48, 16, 16);
        assert_eq!(p.a0, (31, 64));
        assert_eq!(p.a1, (31, 63));
        assert_eq!(p.b0, (48, 47));
        assert_eq!(p.b1, (47, 47));
        assert_eq!(p.b2, (31, 47));
        assert_eq!(p.c0, (48, 64));
        assert_eq!(p.c1, (48, 63));
    }

    fn dummy_src() -> NeighbourAffineSource {
        NeighbourAffineSource {
            x_nb: 0,
            y_nb: 0,
            n_nb_w: 16,
            n_nb_h: 16,
            mv_tl: MotionVector::default(),
            mv_tr: MotionVector::default(),
            mv_bl: MotionVector::default(),
            mv_br: MotionVector::default(),
            motion_model_idc: 1,
        }
    }
}
