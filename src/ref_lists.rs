//! §8.3 slice-level reference bookkeeping for `sps_rpl_flag == 0`
//! streams — the picture order count of §8.3.1 (eqs. 155-163), the
//! §8.3.3.2 reference marking (eqs. 169/170) and the §8.3.2.2 reference
//! picture list construction (eqs. 167/168).
//!
//! These are pure functions over `(PicOrderCntVal, TemporalId)` pairs so
//! the **decoder** (its DPB) and the **encoder** (its mirror DPB) run
//! the identical derivation: whatever list the encoder addressed with a
//! `ref_idx` is, by construction, the list the decoder rebuilds.

/// One DPB picture as the §8.3 processes see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefPicInfo {
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// `TemporalId` (`nuh_temporal_id` of the picture's NAL units).
    pub temporal_id: u8,
}

/// §8.3.1, `sps_pocs_flag == 0` (eqs. 155-163): derive
/// `(PicOrderCntVal, DocOffset)` for a non-IDR picture.
///
/// * `prev_tid0_poc` — `PicOrderCntVal` of `prevTid0Pic` (the previous
///   `TemporalId == 0` picture in decoding order);
/// * `prev_doc_offset` — `DocOffset` of the previous picture in decoding
///   order (an IDR sets it to −1, eq. 156);
/// * `temporal_id` — the current picture's `TemporalId`;
/// * `sub_gop_length` — `SubGopLength` (eq. 72).
///
/// An IDR picture takes `(0, −1)` (eqs. 155/156) without calling this.
pub fn derive_poc_pocs_flag0(
    prev_tid0_poc: i32,
    prev_doc_offset: i32,
    temporal_id: u8,
    sub_gop_length: i32,
) -> (i32, i32) {
    let sub_gop_length = sub_gop_length.max(1);
    if temporal_id == 0 {
        // eqs. 157/158.
        return (prev_tid0_poc + sub_gop_length, 0);
    }
    // eq. 159.
    let mut doc_offset = (prev_doc_offset + 1) % sub_gop_length;
    let mut prev_poc = prev_tid0_poc;
    let mut expected_tid: i32;
    if doc_offset == 0 {
        // eqs. 160/161.
        prev_poc += sub_gop_length;
        expected_tid = 0;
    } else {
        // eq. 162.
        expected_tid = 1 + floor_log2(doc_offset as u32) as i32;
    }
    // eq. 163.
    let tid = temporal_id as i32;
    let mut guard = 0;
    while tid != expected_tid {
        doc_offset = (doc_offset + 1) % sub_gop_length;
        if doc_offset == 0 {
            expected_tid = 0;
        } else {
            expected_tid = 1 + floor_log2(doc_offset as u32) as i32;
        }
        guard += 1;
        if guard > 4 * sub_gop_length + 8 {
            // A TemporalId the sub-GOP shape never produces: the
            // bitstream is non-conforming; fall back to the tid-0 rule
            // rather than spin.
            return (prev_tid0_poc + sub_gop_length, 0);
        }
    }
    // PocOffset = SubGopLength * ( ( 2 * DocOffset + 1 ) / 2^TemporalId − 2 ).
    // The quotient is exact here: SubGopLength = 2^log2_sub_gop_length
    // and TemporalId <= log2_sub_gop_length in a conforming sub-GOP, so
    // `SubGopLength * (2 * DocOffset + 1)` is divisible by 2^TemporalId.
    // Evaluating it as a truncating integer division *before* the
    // multiplication (the §5 default reading) would map the tid-1
    // picture of a 4-picture sub-GOP to the anchor − 4 instead of the
    // anchor − 2, so the multiplication is applied first.
    let poc_offset = ((sub_gop_length * (2 * doc_offset + 1)) >> tid) - 2 * sub_gop_length;
    (prev_poc + poc_offset, doc_offset)
}

fn floor_log2(v: u32) -> u32 {
    31 - v.max(1).leading_zeros()
}

/// §8.3.3.2 reference picture marking (`sps_rpl_flag == 0`), invoked
/// for a current picture with `TemporalId == 0` before its list
/// construction. Returns, index-parallel to `dpb`, whether each picture
/// stays "used for short-term reference" (`true`) or becomes "unused for
/// reference" (`false`). Pictures with `poc >= curr_poc` are outside the
/// eq. 169/170 walk and keep their marking.
pub fn mark_references_rpl_flag0(
    dpb: &[RefPicInfo],
    curr_poc: i32,
    log2_sub_gop_length: u32,
    ref_pic_gap_length: i32,
    max_num_tid0_ref_pics: u32,
) -> Vec<bool> {
    let mut keep = vec![true; dpb.len()];
    let Some(min_poc) = dpb.iter().map(|p| p.poc).min() else {
        return keep;
    };
    let gap = ref_pic_gap_length.max(1);
    let mut idx = 0u32;
    let mut j = curr_poc - 1;
    while j >= min_poc {
        if let Some(k) = dpb.iter().position(|p| p.poc == j) {
            let pic = dpb[k];
            let retain = if log2_sub_gop_length > 0 {
                // eq. 169.
                pic.temporal_id == 0 && idx < max_num_tid0_ref_pics
            } else {
                // eq. 170.
                (pic.poc == curr_poc - 1 || pic.poc % gap == 0) && idx < max_num_tid0_ref_pics
            };
            if retain {
                idx += 1;
            }
            keep[k] = retain;
        }
        j -= 1;
    }
    keep
}

/// §8.3.2.2.2 — fill `list` with lower-POC pictures from `curr_poc`
/// downwards (eq. 167). Returns `nextIdx`.
fn fill_lower(
    list: &mut Vec<i32>,
    refs: &[RefPicInfo],
    curr_poc: i32,
    curr_tid: u8,
    num_active: usize,
) {
    let mut next_tid = curr_tid.saturating_sub(1);
    let Some(min_poc) = refs.iter().map(|p| p.poc).min() else {
        return;
    };
    let mut j = curr_poc;
    while j >= min_poc && list.len() < num_active {
        if let Some(p) = refs
            .iter()
            .find(|p| p.poc == j && p.temporal_id <= next_tid)
        {
            list.push(p.poc);
            next_tid = p.temporal_id.saturating_sub(1);
        }
        j -= 1;
    }
}

/// §8.3.2.2.3 — fill `list` with higher-POC pictures from `curr_poc`
/// upwards (eq. 168).
fn fill_higher(
    list: &mut Vec<i32>,
    refs: &[RefPicInfo],
    curr_poc: i32,
    curr_tid: u8,
    num_active: usize,
) {
    let mut next_tid = curr_tid.saturating_sub(1);
    let Some(max_poc) = refs.iter().map(|p| p.poc).max() else {
        return;
    };
    let mut j = curr_poc;
    while j <= max_poc && list.len() < num_active {
        if let Some(p) = refs
            .iter()
            .find(|p| p.poc == j && p.temporal_id <= next_tid)
        {
            list.push(p.poc);
            next_tid = p.temporal_id.saturating_sub(1);
        }
        j += 1;
    }
}

/// §8.3.2.2.1 — construct `RefPicList[ 0 ]` (and `RefPicList[ 1 ]` for a
/// B slice) over the pictures marked "used for reference". Each list
/// holds at most `num_active[i]` POCs; when fewer pictures qualify the
/// list is shorter and, per step 3, `NumRefIdxActive[ i ]` shrinks to
/// its length.
pub fn construct_ref_pic_lists_rpl_flag0(
    refs: &[RefPicInfo],
    curr_poc: i32,
    curr_tid: u8,
    num_active: [usize; 2],
    slice_is_b: bool,
) -> [Vec<i32>; 2] {
    let mut l0 = Vec::with_capacity(num_active[0]);
    fill_lower(&mut l0, refs, curr_poc, curr_tid, num_active[0]);
    if l0.len() < num_active[0] {
        fill_higher(&mut l0, refs, curr_poc, curr_tid, num_active[0]);
    }
    let mut l1 = Vec::new();
    if slice_is_b {
        l1.reserve(num_active[1]);
        fill_higher(&mut l1, refs, curr_poc, curr_tid, num_active[1]);
        if l1.len() < num_active[1] {
            fill_lower(&mut l1, refs, curr_poc, curr_tid, num_active[1]);
        }
    }
    [l0, l1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tid0(pocs: &[i32]) -> Vec<RefPicInfo> {
        pocs.iter()
            .map(|&poc| RefPicInfo {
                poc,
                temporal_id: 0,
            })
            .collect()
    }

    /// Low-delay coding order (SubGopLength 1, all TemporalId 0): POC
    /// advances by one per picture (eq. 157), DocOffset 0.
    #[test]
    fn poc_low_delay_counts_up() {
        assert_eq!(derive_poc_pocs_flag0(0, -1, 0, 1), (1, 0));
        assert_eq!(derive_poc_pocs_flag0(7, 0, 0, 1), (8, 0));
    }

    /// Hierarchical sub-GOP of 4 (eqs. 159-163): decode order after
    /// the IDR is tid 0 (POC 4), tid 1 (POC 2), tid 2 (POC 1), tid 2
    /// (POC 3), then the next tid-0 anchor (POC 8).
    #[test]
    fn poc_hierarchical_sub_gop_4() {
        assert_eq!(derive_poc_pocs_flag0(0, -1, 0, 4), (4, 0));
        // prevTid0Pic = POC 4 for the whole sub-GOP.
        assert_eq!(derive_poc_pocs_flag0(4, 0, 1, 4), (2, 1));
        assert_eq!(derive_poc_pocs_flag0(4, 1, 2, 4), (1, 2));
        assert_eq!(derive_poc_pocs_flag0(4, 2, 2, 4), (3, 3));
        assert_eq!(derive_poc_pocs_flag0(4, 3, 0, 4), (8, 0));
        // Sub-GOP of 8, tid 3 leaves: DocOffset 4 → 1 + floor(log2 4) = 3.
        assert_eq!(derive_poc_pocs_flag0(8, 3, 3, 8), (1, 4));
        assert_eq!(derive_poc_pocs_flag0(8, 4, 3, 8), (3, 5));
        assert_eq!(derive_poc_pocs_flag0(8, 5, 3, 8), (5, 6));
        assert_eq!(derive_poc_pocs_flag0(8, 6, 3, 8), (7, 7));
    }

    /// eq. 170 (`log2_sub_gop_length == 0`, gap 1): the
    /// `max_num_tid0_ref_pics` most recent pictures stay marked, older
    /// ones drop; 0 drops everything.
    #[test]
    fn marking_gap_one_keeps_most_recent() {
        let dpb = tid0(&[0, 1, 2, 3, 4]);
        assert_eq!(
            mark_references_rpl_flag0(&dpb, 5, 0, 1, 2),
            vec![false, false, false, true, true]
        );
        assert_eq!(mark_references_rpl_flag0(&dpb, 5, 0, 1, 0), vec![false; 5]);
        assert_eq!(mark_references_rpl_flag0(&dpb, 5, 0, 1, 5), vec![true; 5]);
    }

    /// eq. 170 with `RefPicGapLength == 4`: besides POC − 1, only
    /// multiples of the gap qualify.
    #[test]
    fn marking_gap_four() {
        let dpb = tid0(&[0, 1, 2, 3, 4, 5, 6]);
        // curr 7: POC 6 (curr − 1) then 4, 0 qualify; 5, 3, 2, 1 do not.
        assert_eq!(
            mark_references_rpl_flag0(&dpb, 7, 0, 4, 3),
            vec![true, false, false, false, true, false, true]
        );
    }

    /// eq. 169 (`log2_sub_gop_length > 0`): only TemporalId-0 pictures
    /// survive, up to the cap.
    #[test]
    fn marking_hierarchical_keeps_tid0_only() {
        let dpb = vec![
            RefPicInfo {
                poc: 0,
                temporal_id: 0,
            },
            RefPicInfo {
                poc: 2,
                temporal_id: 1,
            },
            RefPicInfo {
                poc: 4,
                temporal_id: 0,
            },
        ];
        assert_eq!(
            mark_references_rpl_flag0(&dpb, 8, 2, 1, 1),
            vec![false, false, true]
        );
        assert_eq!(
            mark_references_rpl_flag0(&dpb, 8, 2, 1, 2),
            vec![true, false, true]
        );
    }

    /// §8.3.2.2: a low-delay P picture lists the references in
    /// descending POC; the list shrinks when fewer pictures qualify.
    #[test]
    fn low_delay_lists_descend() {
        let refs = tid0(&[0, 1, 2]);
        let [l0, l1] = construct_ref_pic_lists_rpl_flag0(&refs, 3, 0, [2, 0], false);
        assert_eq!(l0, vec![2, 1]);
        assert!(l1.is_empty());
        let [l0, _] = construct_ref_pic_lists_rpl_flag0(&refs, 3, 0, [5, 0], false);
        assert_eq!(l0, vec![2, 1, 0], "step 3: NumRefIdxActive shrinks");
    }

    /// A low-delay B picture (nothing above the current POC): L1 falls
    /// through the higher-POC fill to the same descending set as L0.
    #[test]
    fn low_delay_b_lists_coincide() {
        let refs = tid0(&[0, 1, 2]);
        let [l0, l1] = construct_ref_pic_lists_rpl_flag0(&refs, 3, 0, [2, 2], true);
        assert_eq!(l0, vec![2, 1]);
        assert_eq!(l1, vec![2, 1]);
    }

    /// Hierarchical B: the higher-POC anchor leads L1 and the lower-POC
    /// anchor leads L0; the TemporalId gate skips same-or-higher-layer
    /// pictures.
    #[test]
    fn hierarchical_b_lists() {
        let refs = vec![
            RefPicInfo {
                poc: 0,
                temporal_id: 0,
            },
            RefPicInfo {
                poc: 4,
                temporal_id: 0,
            },
            RefPicInfo {
                poc: 2,
                temporal_id: 1,
            },
        ];
        // POC 1, tid 2: nextTemporalId = 1. Lower fill: POC 0 (tid 0);
        // the list still wants a second entry, so the higher fill
        // (nextTemporalId reset to 1) appends POC 2 (tid 1). L1 leads
        // with the higher fill: 2 then (nextTemporalId 0) 4.
        let [l0, l1] = construct_ref_pic_lists_rpl_flag0(&refs, 1, 2, [2, 2], true);
        assert_eq!(l0, vec![0, 2]);
        assert_eq!(l1, vec![2, 4]);
        // POC 2, tid 1: nextTemporalId = 0 → only tid-0 pictures.
        let [l0, l1] = construct_ref_pic_lists_rpl_flag0(&refs, 2, 1, [2, 2], true);
        assert_eq!(l0, vec![0, 4]);
        assert_eq!(l1, vec![4, 0]);
    }
}
