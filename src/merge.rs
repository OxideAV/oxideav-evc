//! §8.5.2.3 ADMVP merge-mode motion-vector / reference-index derivation.
//!
//! This module implements the **advanced direct/merge MV prediction
//! (ADMVP)** merge-candidate-list construction that the Main-profile
//! inter path uses when `sps_admvp_flag == 1`. The Baseline toolset
//! (`sps_admvp_flag == 0`) collapses to the much simpler §8.5.2.4 AMVP
//! path already implemented in [`crate::inter`]; this module is the
//! Main-profile generalisation.
//!
//! The §8.5.2.3.1 general process assembles `mergeCandList` from up to
//! five sub-derivations, invoked in order:
//!
//! 1. §8.5.2.3.2 — **spatial** merge candidates (A1, B1, B0, A0, B2),
//!    with the neighbour positions chosen by the §6.4.2 `availLR`
//!    left/right-availability code. Implemented here in
//!    [`spatial_merge_candidates`].
//! 2. §8.5.2.3.3 — **temporal** (collocated) merge candidate. Deferred:
//!    needs the §8.5.2.3.4 collocated-picture motion field, a
//!    DPB-level state the inter path does not yet thread. The general
//!    assembly accepts a caller-supplied optional temporal candidate so
//!    the wiring slots in without restructuring.
//! 3. §8.5.2.3.6 — **history-based** (HMVP) merge candidates. Deferred:
//!    threads `HmvpCandList`/`NumHmvpCand` from the [`crate::hmvp`]
//!    module; the assembly takes a slice of pre-derived HMVP merge
//!    candidates.
//! 4. §8.5.2.3.7 — **combined bi-predictive** candidates. Implemented
//!    here in [`combined_bipred_candidates`] (B-slice only, Table 21
//!    `l0CandIdx`/`l1CandIdx` pairing).
//! 5. §8.5.2.3.8 — **zero motion vector** candidates. Implemented here
//!    in [`zero_mv_candidates`].
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).
//!
//! ## Purity & the neighbour-lookup contract
//!
//! Like [`crate::inter::build_amvp_list_baseline`], the spatial
//! derivation is parameterised on a caller-supplied neighbour lookup
//! closure rather than a concrete MV grid, keeping the derivation a pure
//! function that the decoder's MV-store wires in. The closure resolves a
//! luma sample location `(xN, yN)` to a [`NeighbourMv`] describing the
//! §6.4.3 motion-vector-candidate availability plus the stored
//! `MvLX`/`RefIdxLX`/`PredFlagLX` at that position.

use crate::inter::{merge_cand_redundancy_check, MergeCand, MotionVector};
use crate::neighbour::AvailLr;
use oxideav_core::error::Result;

/// The per-position motion state a §8.5.2.3.2 spatial-neighbour lookup
/// returns for a luma location `(xN, yN)`: the §6.4.3 availability flag
/// plus the stored `MvLX` / `RefIdxLX` / `PredFlagLX` for both lists.
///
/// When `available` is `false` the remaining fields are ignored — the
/// spec sets `mvLXN = 0`, `refIdxLXN = −1`, `predFlagLXN = 0` for an
/// unavailable neighbour (the "If availableN is equal to FALSE" branch
/// of eqs. 460-462 / 465-467 / 470-472 / 475-477 / 480-482).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighbourMv {
    /// §6.4.3 `availableN` — the neighbour is in-picture, in the same
    /// tile/slice, and coded as a *motion* block (not intra / not IBC).
    pub available: bool,
    /// `PredFlagL0[ xN ][ yN ]`.
    pub pred_flag_l0: bool,
    /// `PredFlagL1[ xN ][ yN ]`.
    pub pred_flag_l1: bool,
    /// `RefIdxL0[ xN ][ yN ]` (meaningful iff `pred_flag_l0`).
    pub ref_idx_l0: i32,
    /// `RefIdxL1[ xN ][ yN ]` (meaningful iff `pred_flag_l1`).
    pub ref_idx_l1: i32,
    /// `MvL0[ xN ][ yN ]`.
    pub mv_l0: MotionVector,
    /// `MvL1[ xN ][ yN ]`.
    pub mv_l1: MotionVector,
}

impl NeighbourMv {
    /// Project a §6.4.3-available neighbour into a [`MergeCand`],
    /// applying the eqs. 463/464 (and 468/469, 473/474, 478/479,
    /// 483/484) bi-pred-to-uni demotion: when both `refIdxL0N` and
    /// `refIdxL1N` are valid (≠ −1) **and** `nCbW + nCbH ≤ 12`, list 1
    /// is dropped (`refIdxL1N = −1`, `predFlagL1N = 0`).
    ///
    /// Returns `None` when the neighbour is unavailable (the spec's
    /// "both components 0, refIdx −1, predFlag 0" sentinel never enters
    /// `mergeCandList`; the append is gated on `availableN == TRUE`).
    fn to_merge_cand(self, small_block: bool) -> Option<MergeCand> {
        if !self.available {
            return None;
        }
        let l0_valid = self.pred_flag_l0 && self.ref_idx_l0 != -1;
        let mut l1_valid = self.pred_flag_l1 && self.ref_idx_l1 != -1;
        // eqs. 463/464 (and the parallel pairs): bi-pred demotion for
        // small blocks. "neither of refIdxL0N and refIdxL1N is equal to
        // −1" ≡ both lists valid.
        if l0_valid && l1_valid && small_block {
            l1_valid = false;
        }
        Some(MergeCand {
            pred_flag_l0: l0_valid,
            pred_flag_l1: l1_valid,
            ref_idx_l0: if l0_valid { self.ref_idx_l0 } else { -1 },
            ref_idx_l1: if l1_valid { self.ref_idx_l1 } else { -1 },
            mv_l0: if l0_valid {
                self.mv_l0
            } else {
                MotionVector::default()
            },
            mv_l1: if l1_valid {
                self.mv_l1
            } else {
                MotionVector::default()
            },
        })
    }
}

/// The five §8.5.2.3.2 spatial-neighbour luma sample locations
/// (A1, B1, B0, A0, B2), already resolved against the `availLR` code.
///
/// Returned by [`spatial_neighbour_positions`] so the position geometry
/// is unit-testable in isolation from the MV lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialNeighbourPositions {
    pub a1: (i32, i32),
    pub b1: (i32, i32),
    pub b0: (i32, i32),
    pub a0: (i32, i32),
    pub b2: (i32, i32),
}

/// §8.5.2.3.2 — resolve the five spatial-neighbour luma sample
/// locations for a CU at `(x_cb, y_cb)` of size `n_cb_w × n_cb_h`,
/// branching on the §6.4.2 `availLR` left/right-availability code.
///
/// The three `availLR` branches (LR_11; LR_01; LR_10-or-LR_00) place the
/// neighbours differently — EVC mirrors the candidate geometry when only
/// the right neighbour column is available (LR_01) and uses a
/// double-sided fan when both are (LR_11).
pub fn spatial_neighbour_positions(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    n_cb_h: i32,
    avail_lr: AvailLr,
) -> SpatialNeighbourPositions {
    match avail_lr {
        // LR_11: both left and right neighbours available.
        AvailLr::Lr11 => SpatialNeighbourPositions {
            a1: (x_cb - 1, y_cb + n_cb_h - 1),
            b1: (x_cb + n_cb_w, y_cb + n_cb_h - 1),
            b0: (x_cb, y_cb - 1),
            a0: (x_cb + n_cb_w, y_cb - 1),
            b2: (x_cb - 1, y_cb - 1),
        },
        // LR_01: only the right neighbour column available — mirrored.
        AvailLr::Lr01 => SpatialNeighbourPositions {
            a1: (x_cb + n_cb_w, y_cb + n_cb_h - 1),
            b1: (x_cb, y_cb - 1),
            b0: (x_cb - 1, y_cb - 1),
            a0: (x_cb + n_cb_w, y_cb + n_cb_h),
            b2: (x_cb + n_cb_w, y_cb - 1),
        },
        // LR_10 or LR_00: the canonical (left-biased) fan.
        AvailLr::Lr10 | AvailLr::Lr00 => SpatialNeighbourPositions {
            a1: (x_cb - 1, y_cb + n_cb_h - 1),
            b1: (x_cb + n_cb_w - 1, y_cb - 1),
            b0: (x_cb + n_cb_w, y_cb - 1),
            a0: (x_cb - 1, y_cb + n_cb_h),
            b2: (x_cb - 1, y_cb - 1),
        },
    }
}

/// §8.5.2.3.2 — append the spatial merge candidates (A1, B1, B0, A0,
/// B2) to `merge_cand_list`, returning the updated `numCurrMergeCand`.
///
/// `merge_cand_list` is the working buffer (its `.len()` is the slice's
/// physical capacity, ≥ `m_l_size`); `num_curr_merge_cand` is the count
/// already populated by any earlier sub-derivation (0 on the first
/// call). `m_l_size` is the §8.5.2.3.1 `mLSize` (4 for ≤32-sample CUs,
/// else 6). `mv_at` is the §6.4.3 neighbour lookup.
///
/// The spec's per-neighbour structure (eqs. 460-484):
///
/// * **A1** and **B1** and **B0** are always probed.
/// * **A0** and **B2** are probed only when
///   `numCurrMergeCand < mLSize − 1`.
/// * Each available neighbour is appended; **B1, B0, A0, B2** also run
///   the §8.5.2.3.10 redundancy trim (A1 does not — it is the first
///   spatial candidate so there is nothing to dedup against, matching
///   the spec which omits the "numCurrMergeCand greater than 1" trim
///   clause for A1).
///
/// `small_block` is the eqs. 463/464 predicate `nCbW + nCbH ≤ 12`.
#[allow(clippy::too_many_arguments)]
pub fn spatial_merge_candidates<F>(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    n_cb_h: i32,
    avail_lr: AvailLr,
    m_l_size: usize,
    merge_cand_list: &mut [MergeCand],
    num_curr_merge_cand: usize,
    mut mv_at: F,
) -> Result<usize>
where
    F: FnMut(i32, i32) -> NeighbourMv,
{
    let pos = spatial_neighbour_positions(x_cb, y_cb, n_cb_w, n_cb_h, avail_lr);
    let small_block = (n_cb_w + n_cb_h) <= 12;
    let mut n = num_curr_merge_cand;

    // A1 — always probed; appended with no trim (first spatial cand).
    if let Some(cand) = mv_at(pos.a1.0, pos.a1.1).to_merge_cand(small_block) {
        if n < merge_cand_list.len() {
            merge_cand_list[n] = cand;
            n += 1;
        }
    }

    // B1 — probed; appended then trimmed when numCurrMergeCand > 1.
    if let Some(cand) = mv_at(pos.b1.0, pos.b1.1).to_merge_cand(small_block) {
        if n < merge_cand_list.len() {
            merge_cand_list[n] = cand;
            n += 1;
            n = merge_cand_redundancy_check(merge_cand_list, n)?;
        }
    }

    // B0 — probed; appended then trimmed.
    if let Some(cand) = mv_at(pos.b0.0, pos.b0.1).to_merge_cand(small_block) {
        if n < merge_cand_list.len() {
            merge_cand_list[n] = cand;
            n += 1;
            n = merge_cand_redundancy_check(merge_cand_list, n)?;
        }
    }

    // A0 — probed only when numCurrMergeCand < mLSize − 1.
    if n + 1 < m_l_size {
        if let Some(cand) = mv_at(pos.a0.0, pos.a0.1).to_merge_cand(small_block) {
            if n < merge_cand_list.len() {
                merge_cand_list[n] = cand;
                n += 1;
                n = merge_cand_redundancy_check(merge_cand_list, n)?;
            }
        }
    }

    // B2 — probed only when numCurrMergeCand < mLSize − 1.
    if n + 1 < m_l_size {
        if let Some(cand) = mv_at(pos.b2.0, pos.b2.1).to_merge_cand(small_block) {
            if n < merge_cand_list.len() {
                merge_cand_list[n] = cand;
                n += 1;
                n = merge_cand_redundancy_check(merge_cand_list, n)?;
            }
        }
    }

    Ok(n)
}

/// Table 21 — `(l0CandIdx, l1CandIdx)` for the §8.5.2.3.7 combined
/// bi-predictive candidate `combIdx` (rows 0..=19). Out-of-range
/// `comb_idx` panics in debug; the §8.5.2.3.7 loop never advances past
/// `numInputMergeCand * (numInputMergeCand − 1) ≤ 20`.
const TABLE_21_COMB: [(usize, usize); 20] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
];

/// §8.5.2.3.7 — append combined bi-predictive merge candidates.
///
/// Caller gates on the §8.5.2.3.1 step-4 predicate (`slice_type == B`,
/// `nCbW > 4 && nCbH > 4`, `numCurrMergeCand < mLSize`). The loop pairs
/// the list-0 motion of `mergeCandList[l0CandIdx]` with the list-1
/// motion of `mergeCandList[l1CandIdx]` per Table 21; a combined
/// candidate is emitted only when both source halves carry a valid
/// reference (`refIdxL0l0Cand` and `refIdxL1l1Cand` both ≠ −1, i.e. the
/// source candidate actually predicts from that list).
///
/// Returns the updated `numCurrMergeCand`.
pub fn combined_bipred_candidates(
    m_l_size: usize,
    merge_cand_list: &mut [MergeCand],
    num_curr_merge_cand: usize,
) -> usize {
    // Spec pre-test: "When numCurrMergeCand is greater than 1 and less
    // than mLSize".
    if !(num_curr_merge_cand > 1 && num_curr_merge_cand < m_l_size) {
        return num_curr_merge_cand;
    }
    let num_input = num_curr_merge_cand;
    let mut n = num_curr_merge_cand;
    let mut comb_idx = 0usize;
    let comb_limit = num_input * (num_input - 1);
    loop {
        let (l0, l1) = TABLE_21_COMB[comb_idx];
        // Source halves must index into the input portion of the list.
        if l0 < num_input && l1 < num_input {
            let src0 = merge_cand_list[l0];
            let src1 = merge_cand_list[l1];
            let l0_valid = src0.pred_flag_l0 && src0.ref_idx_l0 != -1;
            let l1_valid = src1.pred_flag_l1 && src1.ref_idx_l1 != -1;
            if l0_valid && l1_valid && n < merge_cand_list.len() {
                merge_cand_list[n] = MergeCand {
                    pred_flag_l0: true,
                    pred_flag_l1: true,
                    ref_idx_l0: src0.ref_idx_l0,
                    ref_idx_l1: src1.ref_idx_l1,
                    mv_l0: src0.mv_l0,
                    mv_l1: src1.mv_l1,
                };
                n += 1;
            }
        }
        comb_idx += 1;
        // Step 5: stop at the comb-pair exhaustion or when the list is
        // full.
        if comb_idx >= comb_limit || n >= m_l_size {
            break;
        }
    }
    n
}

/// §8.5.2.3.8 — fill the remaining `mergeCandList` slots with zero
/// motion-vector candidates until `numCurrMergeCand == mLSize`.
///
/// `bi_pred_allowed` is the spec's predicate: `FALSE` for P slices or
/// for B slices with `nCbW + nCbH ≤ 12`, else `TRUE`. A zero candidate
/// is list-0 uni (refIdx 0) when bi-pred is disallowed, or bi (refIdx 0
/// on both lists) when allowed.
///
/// Returns the updated `numCurrMergeCand` (== `mLSize` on a normal
/// return, capped at the buffer length).
pub fn zero_mv_candidates(
    m_l_size: usize,
    bi_pred_allowed: bool,
    merge_cand_list: &mut [MergeCand],
    num_curr_merge_cand: usize,
) -> usize {
    let mut n = num_curr_merge_cand;
    while n < m_l_size && n < merge_cand_list.len() {
        merge_cand_list[n] = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: bi_pred_allowed,
            ref_idx_l0: 0,
            ref_idx_l1: if bi_pred_allowed { 0 } else { -1 },
            mv_l0: MotionVector::default(),
            mv_l1: MotionVector::default(),
        };
        n += 1;
    }
    n
}

/// Slice-type discriminator for the §8.5.2.3.1 assembly: the merge
/// process needs only the P-vs-B distinction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeSliceType {
    P,
    B,
}

/// The concrete per-CU motion result of the §8.5.2.3.1 step-6 selection:
/// the `mvLX[0][0]` / `refIdxLX` / `predFlagLX[0][0]` that the inter
/// sample-prediction process (§8.5.4) consumes.
///
/// This is the bridge from the merge-candidate-list derivation to motion
/// compensation — given a decoded `merge_idx` (or `mmvd_merge_idx`), it
/// projects the selected list entry into the motion the MC path reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergedMotion {
    pub pred_flag_l0: bool,
    pub pred_flag_l1: bool,
    pub ref_idx_l0: i32,
    pub ref_idx_l1: i32,
    pub mv_l0: MotionVector,
    pub mv_l1: MotionVector,
}

/// §8.5.2.3.1 step 6 — select candidate `N = mergeCandList[ merge_idx ]`
/// and project it into the per-CU motion (eqs. 450-453).
///
/// `merge_cand_list` is the populated buffer; `num_curr_merge_cand` is
/// its filled length (the output of [`build_merge_cand_list`]); and
/// `merge_idx` is the decoded `merge_idx[ xCb ][ yCb ]` (or
/// `mmvd_merge_idx` for the step-7 MMVD path, which selects identically
/// before the §8.5.2.3.9 offset is applied).
///
/// Per eq. 453, `predFlagLX` is set only when `refIdxLX` is a valid
/// reference picture (≠ −1) — a candidate's stale inactive-list `refIdx`
/// never lights up `predFlagLX`. Returns `None` when `merge_idx` is out
/// of the filled range (a bitstream that signalled a `merge_idx` beyond
/// the derived list length, which the caller surfaces as a decode
/// error).
pub fn select_merge_candidate(
    merge_cand_list: &[MergeCand],
    num_curr_merge_cand: usize,
    merge_idx: usize,
) -> Option<MergedMotion> {
    if merge_idx >= num_curr_merge_cand || merge_idx >= merge_cand_list.len() {
        return None;
    }
    let n = merge_cand_list[merge_idx];
    let l0 = n.pred_flag_l0 && n.ref_idx_l0 != -1;
    let l1 = n.pred_flag_l1 && n.ref_idx_l1 != -1;
    Some(MergedMotion {
        pred_flag_l0: l0,
        pred_flag_l1: l1,
        ref_idx_l0: if l0 { n.ref_idx_l0 } else { -1 },
        ref_idx_l1: if l1 { n.ref_idx_l1 } else { -1 },
        mv_l0: if l0 { n.mv_l0 } else { MotionVector::default() },
        mv_l1: if l1 { n.mv_l1 } else { MotionVector::default() },
    })
}

/// §8.5.2.3.1 — the general merge-candidate-list assembly.
///
/// Runs the five ordered sub-derivations and returns the populated
/// `mergeCandList` (truncated to `numCurrMergeCand` entries) ready for
/// the step-6 `N = mergeCandList[ merge_idx ]` selection.
///
/// * `temporal` — the optional §8.5.2.3.3 collocated candidate
///   (`None` when the collocated motion field is unavailable or the
///   tool path does not yet thread it). Appended (with the §8.5.2.3.10
///   trim) immediately after the spatial candidates, exactly where the
///   spec's step 2 places it.
/// * `hmvp` — the §8.5.2.3.6 history-based candidates, already derived
///   by the [`crate::hmvp`] module and passed in order. Appended (each
///   with a trim) while `numCurrMergeCand < mLSize`, matching the
///   step-3 gate (`sps_hmvp_flag == 1 && NumHmvpCand > 2`); the caller
///   passes an empty slice when HMVP is off.
///
/// The spatial sub-derivation's `mv_at` neighbour lookup is supplied by
/// the caller (the decoder's MV store). The combined-bipred step runs
/// only for B slices with both dimensions > 4 (step-4 gate); the zero
/// step always fills the tail.
#[allow(clippy::too_many_arguments)]
pub fn build_merge_cand_list<F>(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    n_cb_h: i32,
    avail_lr: AvailLr,
    slice_type: MergeSliceType,
    mv_at: F,
    temporal: Option<MergeCand>,
    hmvp: &[MergeCand],
    out: &mut [MergeCand],
) -> Result<usize>
where
    F: FnMut(i32, i32) -> NeighbourMv,
{
    // §8.5.2.3.1: mLSize = (nCbW * nCbH <= 32) ? 4 : 6.
    let m_l_size = if (n_cb_w * n_cb_h) <= 32 { 4 } else { 6 };
    debug_assert!(out.len() >= m_l_size, "merge buffer smaller than mLSize");

    // Step 1: spatial.
    let mut n = spatial_merge_candidates(
        x_cb, y_cb, n_cb_w, n_cb_h, avail_lr, m_l_size, out, 0, mv_at,
    )?;

    // Step 2: temporal (collocated).
    if let Some(cand) = temporal {
        if n < out.len() {
            out[n] = cand;
            n += 1;
            n = merge_cand_redundancy_check(out, n)?;
        }
    }

    // Step 3: history-based (HMVP), gated on numCurrMergeCand < mLSize.
    for &cand in hmvp {
        if n >= m_l_size {
            break;
        }
        if n < out.len() {
            out[n] = cand;
            n += 1;
            n = merge_cand_redundancy_check(out, n)?;
        }
    }

    // Step 4: combined bi-predictive (B slices, both dims > 4).
    if slice_type == MergeSliceType::B && n_cb_w > 4 && n_cb_h > 4 && n < m_l_size {
        n = combined_bipred_candidates(m_l_size, out, n);
    }

    // Step 5: zero motion vector fill.
    let bi_pred_allowed = match slice_type {
        MergeSliceType::P => false,
        MergeSliceType::B => (n_cb_w + n_cb_h) > 12,
    };
    n = zero_mv_candidates(m_l_size, bi_pred_allowed, out, n);

    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avail(
        pred_l0: bool,
        ref0: i32,
        mv0: (i32, i32),
        pred_l1: bool,
        ref1: i32,
        mv1: (i32, i32),
    ) -> NeighbourMv {
        NeighbourMv {
            available: true,
            pred_flag_l0: pred_l0,
            pred_flag_l1: pred_l1,
            ref_idx_l0: ref0,
            ref_idx_l1: ref1,
            mv_l0: MotionVector::quarter_pel(mv0.0, mv0.1),
            mv_l1: MotionVector::quarter_pel(mv1.0, mv1.1),
        }
    }

    fn unavail() -> NeighbourMv {
        NeighbourMv::default()
    }

    #[test]
    fn spatial_positions_lr00_left_biased_fan() {
        let p = spatial_neighbour_positions(16, 32, 8, 8, AvailLr::Lr00);
        assert_eq!(p.a1, (15, 39)); // (xCb-1, yCb+nCbH-1)
        assert_eq!(p.b1, (23, 31)); // (xCb+nCbW-1, yCb-1)
        assert_eq!(p.b0, (24, 31)); // (xCb+nCbW, yCb-1)
        assert_eq!(p.a0, (15, 40)); // (xCb-1, yCb+nCbH)
        assert_eq!(p.b2, (15, 31)); // (xCb-1, yCb-1)
    }

    #[test]
    fn spatial_positions_lr01_mirrored() {
        let p = spatial_neighbour_positions(16, 32, 8, 8, AvailLr::Lr01);
        assert_eq!(p.a1, (24, 39)); // (xCb+nCbW, yCb+nCbH-1)
        assert_eq!(p.b1, (16, 31)); // (xCb, yCb-1)
        assert_eq!(p.b0, (15, 31)); // (xCb-1, yCb-1)
        assert_eq!(p.a0, (24, 40)); // (xCb+nCbW, yCb+nCbH)
        assert_eq!(p.b2, (24, 31)); // (xCb+nCbW, yCb-1)
    }

    #[test]
    fn spatial_positions_lr11_double_sided() {
        let p = spatial_neighbour_positions(16, 32, 8, 8, AvailLr::Lr11);
        assert_eq!(p.a1, (15, 39));
        assert_eq!(p.b1, (24, 39));
        assert_eq!(p.b0, (16, 31));
        assert_eq!(p.a0, (24, 31));
        assert_eq!(p.b2, (15, 31));
    }

    #[test]
    fn spatial_collects_distinct_neighbours() {
        // 16x16 CU (mLSize = 6), LR_00. A1, B1 distinct MVs.
        let mut list = [MergeCand::default(); 8];
        let n = spatial_merge_candidates(0, 0, 16, 16, AvailLr::Lr00, 6, &mut list, 0, |x, y| {
            // A1 at (-1,15) — gets (10,0); B1 at (15,-1) — gets (20,0).
            if (x, y) == (-1, 15) {
                avail(true, 0, (10, 0), false, -1, (0, 0))
            } else if (x, y) == (15, -1) {
                avail(true, 0, (20, 0), false, -1, (0, 0))
            } else {
                unavail()
            }
        })
        .unwrap();
        assert_eq!(n, 2);
        assert_eq!(list[0].mv_l0, MotionVector::quarter_pel(10, 0));
        assert_eq!(list[1].mv_l0, MotionVector::quarter_pel(20, 0));
    }

    #[test]
    fn spatial_redundancy_trims_duplicate_b1() {
        // A1 and B1 carry identical motion → B1 trimmed by §8.5.2.3.10.
        let mut list = [MergeCand::default(); 8];
        let n = spatial_merge_candidates(0, 0, 16, 16, AvailLr::Lr00, 6, &mut list, 0, |x, y| {
            if (x, y) == (-1, 15) || (x, y) == (15, -1) {
                avail(true, 0, (10, 0), false, -1, (0, 0))
            } else {
                unavail()
            }
        })
        .unwrap();
        assert_eq!(n, 1); // B1 deduped away.
    }

    #[test]
    fn small_block_demotes_bipred_to_l0() {
        // nCbW + nCbH = 8 ≤ 12 → eqs. 463/464 drop list 1.
        let nb = avail(true, 0, (4, 4), true, 1, (8, 8));
        let cand = nb.to_merge_cand(true).unwrap();
        assert!(cand.pred_flag_l0);
        assert!(!cand.pred_flag_l1);
        assert_eq!(cand.ref_idx_l1, -1);
        // Large block keeps both.
        let cand2 = nb.to_merge_cand(false).unwrap();
        assert!(cand2.pred_flag_l0 && cand2.pred_flag_l1);
    }

    #[test]
    fn a0_b2_gated_on_mlsize_minus_one() {
        // mLSize = 4 (small CU 4x8 = 32 samples). After A1,B1,B0 fill 3
        // distinct entries, numCurrMergeCand=3 == mLSize-1 → A0,B2 skipped.
        let mut list = [MergeCand::default(); 8];
        let n = spatial_merge_candidates(0, 0, 4, 8, AvailLr::Lr00, 4, &mut list, 0, |x, y| {
            let p = spatial_neighbour_positions(0, 0, 4, 8, AvailLr::Lr00);
            if (x, y) == p.a1 {
                avail(true, 0, (1, 0), false, -1, (0, 0))
            } else if (x, y) == p.b1 {
                avail(true, 0, (2, 0), false, -1, (0, 0))
            } else if (x, y) == p.b0 {
                avail(true, 0, (3, 0), false, -1, (0, 0))
            } else {
                // A0/B2 would be (9,0)/(10,0) but must not be consulted.
                avail(true, 0, (99, 0), false, -1, (0, 0))
            }
        })
        .unwrap();
        assert_eq!(n, 3);
        // No (99,0) entry made it in.
        assert!(list[..3].iter().all(|c| c.mv_l0.x != 99));
    }

    #[test]
    fn zero_fill_p_slice_is_uni_l0() {
        let mut list = [MergeCand::default(); 8];
        let n = zero_mv_candidates(4, false, &mut list, 0);
        assert_eq!(n, 4);
        for c in &list[..4] {
            assert!(c.pred_flag_l0 && !c.pred_flag_l1);
            assert_eq!(c.ref_idx_l0, 0);
            assert_eq!(c.ref_idx_l1, -1);
            assert_eq!(c.mv_l0, MotionVector::default());
        }
    }

    #[test]
    fn zero_fill_b_slice_is_bipred() {
        let mut list = [MergeCand::default(); 8];
        let n = zero_mv_candidates(6, true, &mut list, 2);
        assert_eq!(n, 6);
        for c in &list[2..6] {
            assert!(c.pred_flag_l0 && c.pred_flag_l1);
            assert_eq!(c.ref_idx_l0, 0);
            assert_eq!(c.ref_idx_l1, 0);
        }
    }

    #[test]
    fn combined_bipred_pairs_l0_and_l1_halves() {
        // Two uni candidates: idx0 = L0-only, idx1 = L1-only.
        let mut list = [MergeCand::default(); 8];
        list[0] = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: MotionVector::quarter_pel(4, 0),
            mv_l1: MotionVector::default(),
        };
        list[1] = MergeCand {
            pred_flag_l0: false,
            pred_flag_l1: true,
            ref_idx_l0: -1,
            ref_idx_l1: 1,
            mv_l0: MotionVector::default(),
            mv_l1: MotionVector::quarter_pel(0, 8),
        };
        // combIdx 0 -> (l0=0, l1=1): src0.L0 valid, src1.L1 valid -> combine.
        let n = combined_bipred_candidates(6, &mut list, 2);
        assert_eq!(n, 3);
        let comb = list[2];
        assert!(comb.pred_flag_l0 && comb.pred_flag_l1);
        assert_eq!(comb.ref_idx_l0, 0);
        assert_eq!(comb.ref_idx_l1, 1);
        assert_eq!(comb.mv_l0, MotionVector::quarter_pel(4, 0));
        assert_eq!(comb.mv_l1, MotionVector::quarter_pel(0, 8));
    }

    #[test]
    fn combined_bipred_skips_when_below_two_inputs() {
        let mut list = [MergeCand::default(); 8];
        assert_eq!(combined_bipred_candidates(6, &mut list, 1), 1);
    }

    #[test]
    fn general_assembly_p_slice_spatial_then_zero_fill() {
        // 16x16 P-slice CU, one available spatial neighbour, no temporal,
        // no HMVP. Expect [spatial, zero, zero, zero, zero, zero] = 6.
        let mut out = [MergeCand::default(); 6];
        let n = build_merge_cand_list(
            0,
            0,
            16,
            16,
            AvailLr::Lr00,
            MergeSliceType::P,
            |x, y| {
                if (x, y) == (-1, 15) {
                    avail(true, 0, (12, 4), false, -1, (0, 0))
                } else {
                    unavail()
                }
            },
            None,
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(n, 6);
        assert_eq!(out[0].mv_l0, MotionVector::quarter_pel(12, 4));
        // Tail is uni-L0 zero (P slice).
        for c in &out[1..6] {
            assert!(c.pred_flag_l0 && !c.pred_flag_l1);
            assert_eq!(c.mv_l0, MotionVector::default());
        }
    }

    #[test]
    fn general_assembly_threads_temporal_and_hmvp() {
        let mut out = [MergeCand::default(); 6];
        let temporal = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: MotionVector::quarter_pel(7, 7),
            mv_l1: MotionVector::default(),
        };
        let hmvp = [MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: MotionVector::quarter_pel(9, 9),
            mv_l1: MotionVector::default(),
        }];
        let n = build_merge_cand_list(
            0,
            0,
            16,
            16,
            AvailLr::Lr00,
            MergeSliceType::B,
            |_, _| unavail(),
            Some(temporal),
            &hmvp,
            &mut out,
        )
        .unwrap();
        assert_eq!(n, 6);
        // No spatial → temporal at index 0, hmvp at index 1.
        assert_eq!(out[0].mv_l0, MotionVector::quarter_pel(7, 7));
        assert_eq!(out[1].mv_l0, MotionVector::quarter_pel(9, 9));
    }

    #[test]
    fn general_assembly_mlsize_4_for_small_cu() {
        // 4x8 = 32 samples → mLSize = 4 → exactly 4 candidates out.
        let mut out = [MergeCand::default(); 6];
        let n = build_merge_cand_list(
            0,
            0,
            4,
            8,
            AvailLr::Lr00,
            MergeSliceType::P,
            |_, _| unavail(),
            None,
            &[],
            &mut out,
        )
        .unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn select_projects_chosen_candidate() {
        let mut out = [MergeCand::default(); 6];
        let n = build_merge_cand_list(
            0,
            0,
            16,
            16,
            AvailLr::Lr00,
            MergeSliceType::P,
            |x, y| {
                if (x, y) == (-1, 15) {
                    avail(true, 0, (12, 4), false, -1, (0, 0))
                } else {
                    unavail()
                }
            },
            None,
            &[],
            &mut out,
        )
        .unwrap();
        // merge_idx 0 → the spatial candidate.
        let m = select_merge_candidate(&out, n, 0).unwrap();
        assert!(m.pred_flag_l0 && !m.pred_flag_l1);
        assert_eq!(m.mv_l0, MotionVector::quarter_pel(12, 4));
        assert_eq!(m.ref_idx_l0, 0);
        // merge_idx 1 → a zero candidate (P-slice uni-L0).
        let z = select_merge_candidate(&out, n, 1).unwrap();
        assert!(z.pred_flag_l0 && !z.pred_flag_l1);
        assert_eq!(z.mv_l0, MotionVector::default());
    }

    #[test]
    fn select_rejects_out_of_range_idx() {
        let list = [MergeCand::default(); 4];
        assert!(select_merge_candidate(&list, 4, 4).is_none());
        assert!(select_merge_candidate(&list, 2, 3).is_none());
    }

    #[test]
    fn select_does_not_light_predflag_for_invalid_refidx() {
        // A candidate with predFlagL1 set but refIdxL1 == -1 must not
        // project to an active L1 (eq. 453 valid-ref gate).
        let mut list = [MergeCand::default(); 4];
        list[0] = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: MotionVector::quarter_pel(2, 2),
            mv_l1: MotionVector::quarter_pel(9, 9),
        };
        let m = select_merge_candidate(&list, 1, 0).unwrap();
        assert!(m.pred_flag_l0 && !m.pred_flag_l1);
        assert_eq!(m.ref_idx_l1, -1);
        assert_eq!(m.mv_l1, MotionVector::default());
    }
}
