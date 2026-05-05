//! History-based motion vector prediction (ISO/IEC 23094-1 §8.5.2.7,
//! §8.5.2.4.4).
//!
//! HMVP maintains a per-CTU-row LRU table of 23 entries; each entry holds
//! the bipredictive motion data of the previous inter CU (`mvL0`, `mvL1`,
//! `refIdxL0`, `refIdxL1`). After every inter CU, the entry list is updated
//! per §8.5.2.7:
//!
//! * If a duplicate of the new candidate already lives in the list,
//!   nothing happens (the spec wording elides explicit redundancy
//!   removal but the §8.5.2.3.6 history-merge derivation only walks the
//!   tail, so a left-shift at duplicate insertion is implicit).
//! * If the table holds 23 entries already (`NumHmvpCand == 23`), the
//!   list shifts left by one (entries `1..23` move to `0..22`) and the
//!   new candidate lands at slot 22. Otherwise the new candidate appends
//!   at slot `NumHmvpCand` and the count increments.
//!
//! The list resets to empty at the start of every CTU row (per §7.3.8.2:
//! `if (xCtb == xFirstCtb) NumHmvpCand = 0`).
//!
//! [`HmvpCandList::derive_default_mv`] implements §8.5.2.4.4: when a CU's
//! AMVP neighbour candidates are all unavailable, the decoder walks the
//! tail of the HMVP list looking for an entry whose ref-idx matches the
//! current CU's; the first match wins, with a fallback to the most
//! recent valid entry.

use crate::inter::MotionVector;

/// Maximum number of HMVP candidate entries per the spec (§8.5.2.7).
pub const MAX_HMVP_CAND: usize = 23;

/// One HMVP candidate. Mirrors the per-CU motion data the §8.5.2.7 update
/// process consumes. Reference indices are signed 8-bit so `-1` (the spec
/// "invalid" sentinel) is representable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HmvpCandidate {
    pub mv_l0: MotionVector,
    pub mv_l1: MotionVector,
    pub ref_idx_l0: i8,
    pub ref_idx_l1: i8,
}

impl HmvpCandidate {
    /// Quick "any-list-valid" test mirroring §8.5.2.7's update gate:
    /// "If slice_type is equal to P and refIdxL0 is valid or if slice_type
    /// is equal to B and either refIdxL0 or refIdxL1 is valid".
    pub fn any_ref_valid(&self) -> bool {
        self.ref_idx_l0 >= 0 || self.ref_idx_l1 >= 0
    }
}

/// HMVP candidate list. The current entry count is tracked separately
/// from the backing storage so we can keep the buffer pre-sized at
/// [`MAX_HMVP_CAND`] (matching the spec's `NumHmvpCand .. 22` zero-fill).
#[derive(Clone, Debug)]
pub struct HmvpCandList {
    /// Backing array with `MAX_HMVP_CAND` slots. Slots `0..num` carry the
    /// live candidates in least-recent → most-recent order; slots
    /// `num..MAX_HMVP_CAND` are zero-filled with `ref_idx_l0/l1 = -1`.
    cands: [HmvpCandidate; MAX_HMVP_CAND],
    /// Number of valid entries (`NumHmvpCand` in spec notation), in
    /// `0..=MAX_HMVP_CAND`.
    num: usize,
}

impl Default for HmvpCandList {
    fn default() -> Self {
        Self::new()
    }
}

impl HmvpCandList {
    /// Fresh list with `NumHmvpCand = 0`. All slots are zero-MV with
    /// `refIdx = -1` (the spec's empty-list state).
    pub const fn new() -> Self {
        Self {
            cands: [HmvpCandidate {
                mv_l0: MotionVector { x: 0, y: 0 },
                mv_l1: MotionVector { x: 0, y: 0 },
                ref_idx_l0: -1,
                ref_idx_l1: -1,
            }; MAX_HMVP_CAND],
            num: 0,
        }
    }

    /// `NumHmvpCand`.
    pub fn len(&self) -> usize {
        self.num
    }

    pub fn is_empty(&self) -> bool {
        self.num == 0
    }

    /// Reset to empty. Called at the start of every CTU row per §7.3.8.2:
    /// `if (xCtb == xFirstCtb) NumHmvpCand = 0`.
    pub fn reset(&mut self) {
        self.num = 0;
        // Keep the backing storage as zero-MV / refIdx=-1 so out-of-range
        // reads still see the spec's "invalid" sentinel.
        for slot in self.cands.iter_mut() {
            *slot = HmvpCandidate {
                mv_l0: MotionVector::default(),
                mv_l1: MotionVector::default(),
                ref_idx_l0: -1,
                ref_idx_l1: -1,
            };
        }
    }

    /// Borrow the i-th live candidate. `None` for out-of-range indices.
    pub fn get(&self, i: usize) -> Option<HmvpCandidate> {
        if i < self.num {
            Some(self.cands[i])
        } else {
            None
        }
    }

    /// All live candidates in chronological (oldest-first) order.
    pub fn iter(&self) -> impl Iterator<Item = HmvpCandidate> + '_ {
        self.cands.iter().copied().take(self.num)
    }

    /// §8.5.2.7 update step. For P slices, only `mv_l0 / ref_idx_l0` are
    /// meaningful (and `ref_idx_l1` should be `-1`). For B slices, either
    /// list slot can be valid. The candidate is appended at the current
    /// `NumHmvpCand` index, with a left-shift fallback when the list is
    /// already full.
    ///
    /// No-ops if the candidate has no valid ref index (matching the spec
    /// gate "If slice_type is P and refIdxL0 is valid, or B and either …").
    pub fn update(&mut self, cand: HmvpCandidate) {
        if !cand.any_ref_valid() {
            return;
        }
        if self.num == MAX_HMVP_CAND {
            // Full: shift entries `1..23` left to `0..22`, drop slot 0.
            for i in 1..MAX_HMVP_CAND {
                self.cands[i - 1] = self.cands[i];
            }
            self.cands[MAX_HMVP_CAND - 1] = cand;
            // num stays at MAX_HMVP_CAND.
        } else {
            self.cands[self.num] = cand;
            self.num += 1;
        }
    }

    /// §8.5.2.4.4 derivation process for HMVP-based MV prediction.
    ///
    /// Inputs:
    ///   * `cur_ref_idx_lx` — the current CU's reference index on list X.
    ///   * `list_x` — 0 or 1 (selects the per-candidate `ref_idx_l*` and
    ///     `mv_l*` field to compare / return).
    ///
    /// Returns `Some((mv, ref_idx))` for the matched HMVP candidate, or
    /// `None` if no candidate satisfies the spec walk (the caller then
    /// falls back to a zero MV at refidx 0).
    ///
    /// The spec walks at most `Min(4, NumHmvpCand)` tail entries looking
    /// for a ref-idx match; on no match, it walks the same tail again
    /// looking for the first valid ref-idx. We keep the same ordering.
    pub fn derive_default_mv(&self, cur_ref_idx_lx: i8, list_x: u8) -> Option<(MotionVector, i8)> {
        if self.num == 0 {
            return None;
        }
        let max_check = self.num.min(4);
        // Step 1: look for an exact ref_idx match in the tail (most recent
        // first per `NumHmvpCand - hMvpIdx` indexing).
        for h_mvp_idx in 1..=max_check {
            let cand = self.cands[self.num - h_mvp_idx];
            let cand_ref = if list_x == 0 {
                cand.ref_idx_l0
            } else {
                cand.ref_idx_l1
            };
            if cand_ref == cur_ref_idx_lx && cand_ref >= 0 {
                let mv = if list_x == 0 { cand.mv_l0 } else { cand.mv_l1 };
                return Some((mv, cand_ref));
            }
        }
        // Step 2: fall back to the most recent valid candidate on list X.
        for h_mvp_idx in 1..=max_check {
            let cand = self.cands[self.num - h_mvp_idx];
            let cand_ref = if list_x == 0 {
                cand.ref_idx_l0
            } else {
                cand.ref_idx_l1
            };
            if cand_ref >= 0 {
                let mv = if list_x == 0 { cand.mv_l0 } else { cand.mv_l1 };
                return Some((mv, cand_ref));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cand_l0(x: i32, y: i32, ref_idx: i8) -> HmvpCandidate {
        HmvpCandidate {
            mv_l0: MotionVector::quarter_pel(x, y),
            mv_l1: MotionVector::default(),
            ref_idx_l0: ref_idx,
            ref_idx_l1: -1,
        }
    }

    #[test]
    fn fresh_list_is_empty() {
        let h = HmvpCandList::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert!(h.get(0).is_none());
    }

    #[test]
    fn invalid_candidate_is_dropped() {
        let mut h = HmvpCandList::new();
        let dud = HmvpCandidate {
            mv_l0: MotionVector::quarter_pel(8, 8),
            mv_l1: MotionVector::default(),
            ref_idx_l0: -1,
            ref_idx_l1: -1,
        };
        h.update(dud);
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn appends_until_full() {
        let mut h = HmvpCandList::new();
        for i in 0..MAX_HMVP_CAND {
            h.update(make_cand_l0(i as i32 * 4, 0, 0));
            assert_eq!(h.len(), i + 1);
        }
        // The first candidate sits at slot 0 (oldest).
        assert_eq!(h.get(0).unwrap().mv_l0, MotionVector::quarter_pel(0, 0));
        assert_eq!(
            h.get(MAX_HMVP_CAND - 1).unwrap().mv_l0,
            MotionVector::quarter_pel((MAX_HMVP_CAND as i32 - 1) * 4, 0)
        );
    }

    #[test]
    fn full_list_shifts_left() {
        let mut h = HmvpCandList::new();
        for i in 0..MAX_HMVP_CAND {
            h.update(make_cand_l0(i as i32 * 4, 0, 0));
        }
        assert_eq!(h.len(), MAX_HMVP_CAND);
        // One more push: the oldest candidate (the i=0 one with mv=(0,0))
        // gets evicted; everything shifts left; the new candidate lands at
        // slot 22.
        h.update(make_cand_l0(999, 0, 0));
        assert_eq!(h.len(), MAX_HMVP_CAND);
        assert_eq!(h.get(0).unwrap().mv_l0, MotionVector::quarter_pel(4, 0));
        assert_eq!(
            h.get(MAX_HMVP_CAND - 1).unwrap().mv_l0,
            MotionVector::quarter_pel(999, 0)
        );
    }

    #[test]
    fn reset_clears() {
        let mut h = HmvpCandList::new();
        h.update(make_cand_l0(8, 8, 0));
        assert_eq!(h.len(), 1);
        h.reset();
        assert_eq!(h.len(), 0);
        assert!(h.get(0).is_none());
    }

    #[test]
    fn derive_default_returns_none_on_empty() {
        let h = HmvpCandList::new();
        assert!(h.derive_default_mv(0, 0).is_none());
    }

    #[test]
    fn derive_default_finds_ref_idx_match_in_tail() {
        let mut h = HmvpCandList::new();
        // 6 candidates; only the last 4 are walked. Place a refIdx=2 in
        // the second-to-last slot so step 1 picks it.
        h.update(make_cand_l0(4, 0, 0));
        h.update(make_cand_l0(8, 0, 0));
        h.update(make_cand_l0(12, 0, 1));
        h.update(make_cand_l0(16, 0, 2)); // matches our query
        h.update(make_cand_l0(20, 0, 1));
        h.update(make_cand_l0(24, 0, 1)); // most recent
                                          // Query refIdxLX = 2 → find slot with mv=16,0.
        let (mv, ridx) = h.derive_default_mv(2, 0).unwrap();
        assert_eq!(mv, MotionVector::quarter_pel(16, 0));
        assert_eq!(ridx, 2);
    }

    #[test]
    fn derive_default_falls_back_to_most_recent_when_no_match() {
        let mut h = HmvpCandList::new();
        h.update(make_cand_l0(4, 0, 0));
        h.update(make_cand_l0(8, 0, 0));
        h.update(make_cand_l0(12, 0, 1));
        // Most recent (last appended) is mv=12,0 refIdx=1.
        // Query refIdxLX = 5 → no match → fall back to most recent valid.
        let (mv, ridx) = h.derive_default_mv(5, 0).unwrap();
        assert_eq!(mv, MotionVector::quarter_pel(12, 0));
        assert_eq!(ridx, 1);
    }

    #[test]
    fn derive_default_only_walks_last_four() {
        let mut h = HmvpCandList::new();
        // 6 candidates; the matching refIdx=9 sits 5 entries back, so the
        // §8.5.2.4.4 walk (limited to last 4) must NOT find it. Fallback
        // returns the most-recent valid candidate (refIdx=1).
        h.update(make_cand_l0(4, 0, 9)); // out of walk range
        h.update(make_cand_l0(8, 0, 0));
        h.update(make_cand_l0(12, 0, 1));
        h.update(make_cand_l0(16, 0, 2));
        h.update(make_cand_l0(20, 0, 3));
        h.update(make_cand_l0(24, 0, 1));
        let (mv, ridx) = h.derive_default_mv(9, 0).unwrap();
        assert_ne!(mv, MotionVector::quarter_pel(4, 0));
        assert_eq!(ridx, 1);
        assert_eq!(mv, MotionVector::quarter_pel(24, 0));
    }

    #[test]
    fn derive_default_l1_uses_l1_fields() {
        let mut h = HmvpCandList::new();
        let mut c = make_cand_l0(0, 0, 0);
        c.mv_l1 = MotionVector::quarter_pel(40, 40);
        c.ref_idx_l1 = 0;
        h.update(c);
        let (mv, ridx) = h.derive_default_mv(0, 1).unwrap();
        assert_eq!(mv, MotionVector::quarter_pel(40, 40));
        assert_eq!(ridx, 0);
    }
}
