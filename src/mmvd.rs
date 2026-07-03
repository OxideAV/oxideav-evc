//! §8.5.2.3.9 — derivation process for the MMVD motion vector.
//!
//! The §8.5.2.3.1 step-7 bridge: given the base merge candidate selected
//! by `mmvd_merge_idx` (its `mvLX[0][0]` / `refIdxLX` / `predFlagLX`),
//! the decoded `mmvd_group_idx`, and the eqs.-133/134 axis-aligned
//! `MmvdOffset`, produce the final MMVD motion `mMvLX` with the modified
//! reference indices and prediction-list utilization flags.
//!
//! The process has two stages:
//!
//! 1. **Group retargeting** (eqs. 533-590, `mmvd_group_idx ∈ {1, 2}`):
//!    the base candidate's list utilization is rewritten — a bi-pred base
//!    drops one list (L1 for group 1, L0 for group 2), a uni-pred B base
//!    derives the other list by POC-ratio scaling (group 1 keeps both,
//!    group 2 keeps only the derived list), and a P base retargets
//!    `refIdxL0` (with either a fixed ±3 x-nudge when the target equals
//!    the source, or a POC-ratio rescale when it differs).
//! 2. **Offset scaling** (eqs. 591-612): the `MmvdOffset` is assigned to
//!    the list whose `|currPocDiff|` is smaller and POC-ratio-scaled onto
//!    the other (eqs. 599-606), negated on L1 when the two POC distances
//!    straddle the current picture (eqs. 607-610); a uni-pred candidate
//!    takes the offset on its active list only (eqs. 611/612).
//!
//! All truth from ISO/IEC 23094-1:2020(E); every equation number below
//! cites that document.

use oxideav_core::{Error, Result};

use crate::inter::MotionVector;
use crate::merge::MergedMotion;
use crate::tmvp::diff_pic_order_cnt;

/// The POC context §8.5.2.3.9 draws its `DiffPicOrderCnt` inputs from:
/// the current picture's POC and the POCs of the active `RefPicList0` /
/// `RefPicList1` entries (indexed by `refIdxLX`).
#[derive(Clone, Copy, Debug)]
pub struct MmvdPocs<'a> {
    /// `PicOrderCnt( currPic )`.
    pub curr_poc: i32,
    /// `PicOrderCnt( RefPicList0[ i ] )` for each active L0 index.
    pub ref_pocs_l0: &'a [i32],
    /// `PicOrderCnt( RefPicList1[ i ] )` for each active L1 index. Empty
    /// on a P slice.
    pub ref_pocs_l1: &'a [i32],
}

impl MmvdPocs<'_> {
    /// `PicOrderCnt( RefPicListX[ ref_idx ] )`, bounds-checked.
    fn ref_poc(&self, list_x: u8, ref_idx: i32) -> Result<i32> {
        let list = if list_x == 0 {
            self.ref_pocs_l0
        } else {
            self.ref_pocs_l1
        };
        if ref_idx < 0 {
            return Err(Error::invalid(
                "evc mmvd: negative refIdx in POC-distance derivation (§8.5.2.3.9)",
            ));
        }
        list.get(ref_idx as usize).copied().ok_or_else(|| {
            Error::invalid(format!(
                "evc mmvd: refIdxL{list_x} = {ref_idx} outside the {}-entry reference POC list \
                 (§8.5.2.3.9)",
                list.len()
            ))
        })
    }

    /// `currPocDiffLX = DiffPicOrderCnt( currPic, RefPicListX[ refIdx ] )`
    /// (eqs. 540/541/549/550/557/558/569/570/578/579/586/587/591/592).
    fn curr_poc_diff(&self, list_x: u8, ref_idx: i32) -> Result<i32> {
        Ok(diff_pic_order_cnt(
            self.curr_poc,
            self.ref_poc(list_x, ref_idx)?,
        ))
    }
}

/// eq.-542-style `distScaleFactor = ( num << 5 ) / den`. The spec's `/`
/// on integers truncates toward zero; `den == 0` cannot occur in a
/// conforming stream (a reference picture never shares the current
/// picture's POC) and is surfaced as a decode error.
fn dist_scale_factor(num: i32, den: i32) -> Result<i32> {
    if den == 0 {
        return Err(Error::invalid(
            "evc mmvd: zero POC distance in distScaleFactor derivation (§8.5.2.3.9)",
        ));
    }
    Ok((((num as i64) << 5).wrapping_div(den as i64)) as i32)
}

/// eqs. 543/544 (and the parallel 552/553, 560/561, 572/573, 581/582,
/// 589/590) — scale one MV component by `distScaleFactor` with the
/// explicit `Sign`/`Abs` decomposition (round-half-away-from-zero) and
/// the `Clip3( −32768, 32767, … )`.
fn scale_mv_component(dist_scale_factor: i32, comp: i32) -> i32 {
    let p = (dist_scale_factor as i64) * (comp as i64);
    let mag = (p.abs() + 16) >> 5;
    let signed = if p < 0 { -mag } else { mag };
    signed.clamp(-32768, 32767) as i32
}

/// Scale a whole MV by `distScaleFactor` (both components via
/// [`scale_mv_component`]).
fn scale_mv(dist_scale_factor: i32, mv: MotionVector) -> MotionVector {
    MotionVector {
        x: scale_mv_component(dist_scale_factor, mv.x),
        y: scale_mv_component(dist_scale_factor, mv.y),
    }
}

/// eqs. 600/601/605/606 — the *offset* scaling differs from the motion
/// scaling: the spec writes the literal `( distScaleFactor * d + 16 ) >> 5`
/// with no `Sign`/`Abs` decomposition (`distScaleFactor` is
/// non-negative here, built from `Abs( currPocDiff )` ratios), so a
/// negative product rounds toward −∞ under the arithmetic shift.
fn scale_offset_component(dist_scale_factor: i32, comp: i32) -> i32 {
    let p = (dist_scale_factor as i64) * (comp as i64) + 16;
    (p >> 5).clamp(-32768, 32767) as i32
}

fn scale_offset(dist_scale_factor: i32, off: MotionVector) -> MotionVector {
    MotionVector {
        x: scale_offset_component(dist_scale_factor, off.x),
        y: scale_offset_component(dist_scale_factor, off.y),
    }
}

/// The eq.-538/555/567/584 secondary-reference test: prefer `refIdx = 1`
/// when the list holds a second active entry whose POC mirrors the other
/// list's reference across the current picture —
/// `DiffPicOrderCnt( RefPicListX[ 1 ], currPic ) ==
///  DiffPicOrderCnt( currPic, RefPicListY[ refIdxY ] )`.
fn mirrored_ref_idx(
    pocs: &MmvdPocs<'_>,
    num_ref_idx_active_x: u32,
    list_x: u8,
    other_list: u8,
    other_ref_idx: i32,
) -> Result<i32> {
    if num_ref_idx_active_x > 1 {
        let cand_poc = pocs.ref_poc(list_x, 1)?;
        let other_poc = pocs.ref_poc(other_list, other_ref_idx)?;
        if diff_pic_order_cnt(cand_poc, pocs.curr_poc)
            == diff_pic_order_cnt(pocs.curr_poc, other_poc)
        {
            return Ok(1);
        }
    }
    Ok(0)
}

/// §8.5.2.3.9 — derive the MMVD motion vector.
///
/// * `base` — the step-7 base candidate
///   `mergeCandList[ mmvd_merge_idx ]` (eqs. 454-457 assignments).
/// * `mmvd_group_idx` — the decoded (or inferred-0) group selector.
/// * `offset` — the eqs.-133/134 `MmvdOffset` from
///   [`crate::inter::mmvd_offset`].
/// * `slice_is_b` — `slice_type == B` (selects the uni-pred B-slice
///   group-retarget branches vs the P-slice `targetRefIdxL0` branches).
/// * `num_ref_idx_active` — `NumRefIdxActive[ 0 / 1 ]`.
/// * `pocs` — the POC context for every `DiffPicOrderCnt` in the clause.
///
/// Returns the modified motion (`mMvLX` + updated `refIdxLX` /
/// `predFlagLX`).
pub fn mmvd_motion_vector(
    base: MergedMotion,
    mmvd_group_idx: u32,
    offset: MotionVector,
    slice_is_b: bool,
    num_ref_idx_active: [u32; 2],
    pocs: &MmvdPocs<'_>,
) -> Result<MergedMotion> {
    // eqs. 531/532 — start from the base candidate's motion.
    let mut m = base;

    if mmvd_group_idx == 1 {
        if m.pred_flag_l0 && m.pred_flag_l1 {
            // eqs. 533-536 — drop L1, keep the L0 motion.
            m.ref_idx_l1 = -1;
            m.pred_flag_l1 = false;
            m.mv_l1 = MotionVector::default();
        } else if slice_is_b && m.pred_flag_l0 {
            // eqs. 537-544 — extend the L0-only candidate to bi-pred by
            // POC-scaling mMvL0 onto a derived L1 reference.
            m.pred_flag_l1 = true;
            m.ref_idx_l1 = mirrored_ref_idx(pocs, num_ref_idx_active[1], 1, 0, m.ref_idx_l0)?;
            let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 540
            let curr_poc_diff_l1 = pocs.curr_poc_diff(1, m.ref_idx_l1)?; // eq. 541
            let dsf = dist_scale_factor(curr_poc_diff_l1, curr_poc_diff_l0)?; // eq. 542
            m.mv_l1 = scale_mv(dsf, m.mv_l0); // eqs. 543/544
        } else if !slice_is_b {
            // eqs. 545-553 — P slice: retarget refIdxL0.
            let target = if num_ref_idx_active[0] == 1 {
                m.ref_idx_l0 // eq. 545
            } else {
                // eq. 546 — `!refIdxL0` (C-style logical not).
                i32::from(m.ref_idx_l0 == 0)
            };
            if target == m.ref_idx_l0 {
                // eqs. 547/548 — same reference: fixed +3 x-nudge.
                m.mv_l0.x = m.mv_l0.x.wrapping_add(3);
            } else {
                let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 549
                let curr_poc_diff_target = pocs.curr_poc_diff(0, target)?; // eq. 550
                let dsf = dist_scale_factor(curr_poc_diff_target, curr_poc_diff_l0)?; // eq. 551
                m.mv_l0 = scale_mv(dsf, m.mv_l0); // eqs. 552/553
            }
            m.ref_idx_l0 = target;
        } else if m.pred_flag_l1 {
            // eqs. 554-561 — extend the L1-only candidate to bi-pred by
            // POC-scaling mMvL1 onto a derived L0 reference.
            m.pred_flag_l0 = true;
            m.ref_idx_l0 = mirrored_ref_idx(pocs, num_ref_idx_active[0], 0, 1, m.ref_idx_l1)?;
            let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 557
            let curr_poc_diff_l1 = pocs.curr_poc_diff(1, m.ref_idx_l1)?; // eq. 558
            let dsf = dist_scale_factor(curr_poc_diff_l0, curr_poc_diff_l1)?; // eq. 559
            m.mv_l0 = scale_mv(dsf, m.mv_l1); // eqs. 560/561
        }
    } else if mmvd_group_idx == 2 {
        if m.pred_flag_l0 && m.pred_flag_l1 {
            // eqs. 562-565 — drop L0, keep the L1 motion.
            m.ref_idx_l0 = -1;
            m.pred_flag_l0 = false;
            m.mv_l0 = MotionVector::default();
        } else if slice_is_b && m.pred_flag_l0 {
            // eqs. 566-573 — derive L1 by POC-scaling mMvL0, then drop
            // L0 (the group-2 mirror of the group-1 extend keeps only
            // the derived list).
            m.pred_flag_l1 = true;
            m.ref_idx_l1 = mirrored_ref_idx(pocs, num_ref_idx_active[1], 1, 0, m.ref_idx_l0)?;
            let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 569
            let curr_poc_diff_l1 = pocs.curr_poc_diff(1, m.ref_idx_l1)?; // eq. 570
            let dsf = dist_scale_factor(curr_poc_diff_l1, curr_poc_diff_l0)?; // eq. 571
            m.mv_l1 = scale_mv(dsf, m.mv_l0); // eqs. 572/573
                                              // "refIdxL0 is set equal to −1, and predFlagL0, mMvL0 are 0."
            m.ref_idx_l0 = -1;
            m.pred_flag_l0 = false;
            m.mv_l0 = MotionVector::default();
        } else if !slice_is_b {
            // eqs. 574-582 — P slice: retarget refIdxL0 (group-2 flavour).
            let target = if num_ref_idx_active[0] < 3 {
                m.ref_idx_l0 // eq. 574
            } else if m.ref_idx_l0 < 2 {
                2 // eq. 575
            } else {
                1
            };
            if target == m.ref_idx_l0 {
                // eqs. 576/577 — same reference: fixed −3 x-nudge.
                m.mv_l0.x = m.mv_l0.x.wrapping_sub(3);
            } else {
                let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 578
                let curr_poc_diff_target = pocs.curr_poc_diff(0, target)?; // eq. 579
                let dsf = dist_scale_factor(curr_poc_diff_target, curr_poc_diff_l0)?; // eq. 580
                m.mv_l0 = scale_mv(dsf, m.mv_l0); // eqs. 581/582
            }
            m.ref_idx_l0 = target;
        } else if m.pred_flag_l1 {
            // eqs. 583-590 — derive L0 by POC-scaling mMvL1, then drop L1.
            m.pred_flag_l0 = true;
            m.ref_idx_l0 = mirrored_ref_idx(pocs, num_ref_idx_active[0], 0, 1, m.ref_idx_l1)?;
            let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 586
            let curr_poc_diff_l1 = pocs.curr_poc_diff(1, m.ref_idx_l1)?; // eq. 587
            let dsf = dist_scale_factor(curr_poc_diff_l0, curr_poc_diff_l1)?; // eq. 588
            m.mv_l0 = scale_mv(dsf, m.mv_l1); // eqs. 589/590
                                              // "refIdxL1 is set equal to −1, and predFlagL1, mMvL1 are 0."
            m.ref_idx_l1 = -1;
            m.pred_flag_l1 = false;
            m.mv_l1 = MotionVector::default();
        }
    }

    // Stage 2 — the per-list MVD (offset) derivation.
    let (mvd_l0, mvd_l1) = if m.pred_flag_l0 && m.pred_flag_l1 {
        let curr_poc_diff_l0 = pocs.curr_poc_diff(0, m.ref_idx_l0)?; // eq. 591
        let curr_poc_diff_l1 = pocs.curr_poc_diff(1, m.ref_idx_l1)?; // eq. 592
        let (d0, mut d1) = if curr_poc_diff_l0.abs() == curr_poc_diff_l1.abs() {
            // eqs. 593-596 — equal distances: both lists take the offset.
            (offset, offset)
        } else if curr_poc_diff_l0.abs() > curr_poc_diff_l1.abs() {
            // eqs. 597-601 — L1 nearer: offset on L1, scaled onto L0.
            let dsf = dist_scale_factor(curr_poc_diff_l1.abs(), curr_poc_diff_l0.abs())?; // eq. 599
            (scale_offset(dsf, offset), offset)
        } else {
            // eqs. 602-606 — L0 nearer: offset on L0, scaled onto L1.
            let dsf = dist_scale_factor(curr_poc_diff_l0.abs(), curr_poc_diff_l1.abs())?; // eq. 604
            (offset, scale_offset(dsf, offset))
        };
        // eqs. 607-610 — opposite-side references: negate the L1 delta.
        if (curr_poc_diff_l0 as i64) * (curr_poc_diff_l1 as i64) < 0 {
            d1 = MotionVector { x: -d1.x, y: -d1.y };
        }
        (d0, d1)
    } else {
        // eqs. 611/612 — uni-pred: the active list takes the offset.
        (
            if m.pred_flag_l0 {
                offset
            } else {
                MotionVector::default()
            },
            if m.pred_flag_l1 {
                offset
            } else {
                MotionVector::default()
            },
        )
    };

    // eqs. 613-616 — apply the deltas (16-bit modular per §8.5.2).
    if m.pred_flag_l0 {
        m.mv_l0 = m.mv_l0.wrapping_add(&mvd_l0);
    }
    if m.pred_flag_l1 {
        m.mv_l1 = m.mv_l1.wrapping_add(&mvd_l1);
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    fn bi_base() -> MergedMotion {
        MergedMotion {
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            mv_l0: mv(40, -24),
            mv_l1: mv(-8, 12),
        }
    }

    fn l0_base() -> MergedMotion {
        MergedMotion {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            mv_l0: mv(40, -24),
            mv_l1: MotionVector::default(),
        }
    }

    /// Symmetric B GOP: curr POC 2, L0 ref POC 0 (diff +2), L1 ref POC 4
    /// (diff −2). Equal |distances|, opposite sides.
    fn sym_pocs() -> MmvdPocs<'static> {
        MmvdPocs {
            curr_poc: 2,
            ref_pocs_l0: &[0, -2],
            ref_pocs_l1: &[4, 6],
        }
    }

    /// group 0, bi-pred, |currPocDiffL0| == |currPocDiffL1|, opposite
    /// sides: both lists take the offset, L1 negated (eqs. 593-596 +
    /// 607-610).
    #[test]
    fn group0_bipred_equal_distance_mirrors_offset() {
        let m = mmvd_motion_vector(bi_base(), 0, mv(4, 0), true, [1, 1], &sym_pocs()).unwrap();
        assert_eq!(m.mv_l0, mv(44, -24)); // +offset
        assert_eq!(m.mv_l1, mv(-12, 12)); // −offset (opposite side)
        assert!(m.pred_flag_l0 && m.pred_flag_l1);
    }

    /// group 0, bi-pred, same-side references (both past): no negation.
    #[test]
    fn group0_bipred_same_side_no_negation() {
        let pocs = MmvdPocs {
            curr_poc: 4,
            ref_pocs_l0: &[2], // diff +2
            ref_pocs_l1: &[0], // diff +4
        };
        // |L0| = 2 < |L1| = 4 → offset on L0, scaled ×2 onto L1
        // (dsf = (2 << 5) / 4 = 16 → (16·4 + 16) >> 5 = 2 … wait: the
        // *smaller* distance scales the *larger* list DOWN? eq. 604 dsf =
        // (|L0| << 5)/|L1| = 16; d1 = (16·4+16)>>5 = 2). Same sign.
        let m = mmvd_motion_vector(bi_base(), 0, mv(4, 0), true, [1, 1], &pocs).unwrap();
        assert_eq!(m.mv_l0, mv(44, -24)); // full offset on the nearer L0
        assert_eq!(m.mv_l1, mv(-6, 12)); // +2 scaled offset, not negated
    }

    /// group 0, bi-pred, |L0| > |L1|: offset rides L1, scaled onto L0
    /// (eqs. 597-601).
    #[test]
    fn group0_bipred_l1_nearer_scales_l0() {
        let pocs = MmvdPocs {
            curr_poc: 4,
            ref_pocs_l0: &[0], // diff +4
            ref_pocs_l1: &[2], // diff +2
        };
        let m = mmvd_motion_vector(bi_base(), 0, mv(8, 0), true, [1, 1], &pocs).unwrap();
        // dsf = (2 << 5) / 4 = 16 → d0 = (16·8 + 16) >> 5 = 4.
        assert_eq!(m.mv_l0, mv(44, -24));
        assert_eq!(m.mv_l1, mv(0, 12));
    }

    /// group 0, uni-pred L0: offset applies to L0 only (eqs. 611/612).
    #[test]
    fn group0_unipred_offset_on_active_list() {
        let m = mmvd_motion_vector(l0_base(), 0, mv(0, -2), false, [1, 0], &sym_pocs()).unwrap();
        assert_eq!(m.mv_l0, mv(40, -26));
        assert!(!m.pred_flag_l1);
    }

    /// group 1, bi-pred base: L1 dropped (eqs. 533-536), offset on L0.
    #[test]
    fn group1_bipred_drops_l1() {
        let m = mmvd_motion_vector(bi_base(), 1, mv(4, 0), true, [1, 1], &sym_pocs()).unwrap();
        assert!(m.pred_flag_l0);
        assert!(!m.pred_flag_l1);
        assert_eq!(m.ref_idx_l1, -1);
        assert_eq!(m.mv_l0, mv(44, -24));
        assert_eq!(m.mv_l1, MotionVector::default());
    }

    /// group 1, L0-only B base: L1 derived by POC scaling (eqs. 537-544)
    /// → bi-pred; then stage-2 mirrors the offset.
    #[test]
    fn group1_l0_only_b_extends_to_bipred() {
        let m = mmvd_motion_vector(l0_base(), 1, mv(4, 0), true, [1, 1], &sym_pocs()).unwrap();
        assert!(m.pred_flag_l0 && m.pred_flag_l1);
        assert_eq!(m.ref_idx_l1, 0);
        // dsf = (−2 << 5) / 2 = −32 → mMvL1 = −mvL0 (round-away): (−40, 24).
        // Stage 2: equal |diff|, opposite side → L0 +off, L1 −off.
        assert_eq!(m.mv_l0, mv(44, -24));
        assert_eq!(m.mv_l1, mv(-44, 24));
    }

    /// group 1, L0-only B base with a mirrored second L1 entry: eq. 538
    /// picks refIdxL1 = 1 when DiffPicOrderCnt(RefPicList1[1], curr) ==
    /// DiffPicOrderCnt(curr, RefPicList0[refIdxL0]).
    #[test]
    fn group1_mirrored_secondary_ref_selected() {
        let pocs = MmvdPocs {
            curr_poc: 4,
            ref_pocs_l0: &[2],    // diff(curr, L0[0]) = +2
            ref_pocs_l1: &[8, 6], // diff(L1[1], curr) = +2 → mirrored
        };
        let m = mmvd_motion_vector(l0_base(), 1, mv(4, 0), true, [1, 2], &pocs).unwrap();
        assert_eq!(m.ref_idx_l1, 1);
    }

    /// group 1, P slice, single active reference: targetRefIdx ==
    /// refIdx → the fixed +3 x-nudge (eqs. 545-548).
    #[test]
    fn group1_p_single_ref_nudges_plus3() {
        let pocs = MmvdPocs {
            curr_poc: 2,
            ref_pocs_l0: &[0],
            ref_pocs_l1: &[],
        };
        let m = mmvd_motion_vector(l0_base(), 1, mv(4, 0), false, [1, 0], &pocs).unwrap();
        assert_eq!(m.ref_idx_l0, 0);
        // +3 nudge then +4 offset.
        assert_eq!(m.mv_l0, mv(47, -24));
        assert!(!m.pred_flag_l1);
    }

    /// group 1, P slice, two active references: target = !refIdx → the
    /// POC rescale (eqs. 549-553). curr 4, L0[0] POC 2 (diff 2), L0[1]
    /// POC 0 (diff 4) → dsf = (4 << 5)/2 = 64 → MV doubled.
    #[test]
    fn group1_p_two_refs_retargets_and_scales() {
        let pocs = MmvdPocs {
            curr_poc: 4,
            ref_pocs_l0: &[2, 0],
            ref_pocs_l1: &[],
        };
        let m = mmvd_motion_vector(l0_base(), 1, mv(0, 0), false, [2, 0], &pocs).unwrap();
        assert_eq!(m.ref_idx_l0, 1);
        assert_eq!(m.mv_l0, mv(80, -48));
    }

    /// group 2, bi-pred base: L0 dropped (eqs. 562-565), offset on L1.
    #[test]
    fn group2_bipred_drops_l0() {
        let m = mmvd_motion_vector(bi_base(), 2, mv(4, 0), true, [1, 1], &sym_pocs()).unwrap();
        assert!(!m.pred_flag_l0);
        assert!(m.pred_flag_l1);
        assert_eq!(m.ref_idx_l0, -1);
        assert_eq!(m.mv_l0, MotionVector::default());
        assert_eq!(m.mv_l1, mv(-4, 12));
    }

    /// group 2, L0-only B base: L1 derived by scaling then L0 dropped —
    /// the CU becomes L1-only (eqs. 566-573 + the trailing drop).
    #[test]
    fn group2_l0_only_b_switches_to_l1() {
        let m = mmvd_motion_vector(l0_base(), 2, mv(4, 0), true, [1, 1], &sym_pocs()).unwrap();
        assert!(!m.pred_flag_l0);
        assert!(m.pred_flag_l1);
        assert_eq!(m.ref_idx_l0, -1);
        // dsf = (−2 << 5)/2 = −32 → mMvL1 = (−40, 24); then uni offset +4.
        assert_eq!(m.mv_l1, mv(-36, 24));
        assert_eq!(m.mv_l0, MotionVector::default());
    }

    /// group 2, P slice, NumRefIdxActive[0] < 3: target == refIdx → the
    /// fixed −3 x-nudge (eqs. 574-577).
    #[test]
    fn group2_p_few_refs_nudges_minus3() {
        let pocs = MmvdPocs {
            curr_poc: 2,
            ref_pocs_l0: &[0, -2],
            ref_pocs_l1: &[],
        };
        let m = mmvd_motion_vector(l0_base(), 2, mv(4, 0), false, [2, 0], &pocs).unwrap();
        assert_eq!(m.ref_idx_l0, 0);
        assert_eq!(m.mv_l0, mv(41, -24)); // −3 nudge, +4 offset
    }

    /// group 2, P slice, NumRefIdxActive[0] >= 3: refIdx 0 → target 2
    /// with the POC rescale (eqs. 574-582).
    #[test]
    fn group2_p_many_refs_retargets_to_two() {
        let pocs = MmvdPocs {
            curr_poc: 6,
            ref_pocs_l0: &[4, 2, 0],
            ref_pocs_l1: &[],
        };
        let m = mmvd_motion_vector(l0_base(), 2, mv(0, 0), false, [3, 0], &pocs).unwrap();
        assert_eq!(m.ref_idx_l0, 2);
        // dsf = (6 << 5)/2 = 96 → ×3.
        assert_eq!(m.mv_l0, mv(120, -72));
    }

    /// group 2, L1-only B base: L0 derived then L1 dropped — L0-only.
    #[test]
    fn group2_l1_only_b_switches_to_l0() {
        let base = MergedMotion {
            pred_flag_l0: false,
            pred_flag_l1: true,
            ref_idx_l0: -1,
            ref_idx_l1: 0,
            mv_l0: MotionVector::default(),
            mv_l1: mv(-8, 12),
        };
        let m = mmvd_motion_vector(base, 2, mv(4, 0), true, [1, 1], &sym_pocs()).unwrap();
        assert!(m.pred_flag_l0);
        assert!(!m.pred_flag_l1);
        assert_eq!(m.ref_idx_l1, -1);
        // dsf = (2 << 5)/−2 = −32 → mMvL0 = (8, −12); then +4 offset.
        assert_eq!(m.mv_l0, mv(12, -12));
    }

    /// Out-of-range refIdx in the POC lookup is a decode error, not a
    /// panic.
    #[test]
    fn out_of_range_ref_idx_errors() {
        let base = MergedMotion {
            ref_idx_l0: 5,
            ..l0_base()
        };
        // Bi-pred stage-2 lookup path needs both flags; force group 1
        // L0-only extension which resolves L0's POC first.
        assert!(mmvd_motion_vector(base, 1, mv(4, 0), true, [1, 1], &sym_pocs()).is_err());
    }

    /// Zero POC distance in a scale denominator is a decode error.
    #[test]
    fn zero_poc_distance_errors() {
        let pocs = MmvdPocs {
            curr_poc: 2,
            ref_pocs_l0: &[2], // diff 0 — non-conforming
            ref_pocs_l1: &[0],
        };
        assert!(mmvd_motion_vector(l0_base(), 1, mv(4, 0), true, [1, 1], &pocs).is_err());
    }
}
