//! Reference picture list parsing (ISO/IEC 23094-1 §7.3.7 / §7.4.8).
//!
//! Implements the `ref_pic_list_struct(listIdx, rplsIdx, ltrpFlag)` syntax
//! structure that the SPS uses to enumerate candidate reference picture
//! lists, and that a slice header may also include directly to override
//! per-picture. The structure carries:
//!
//! * `num_strp_entries` — short-term reference picture entries,
//! * `num_ltrp_entries` (when `ltrpFlag == 1`) — long-term entries,
//! * For each STRP entry: a `delta_poc_st` magnitude plus a sign flag (the
//!   sign flag is omitted when `delta_poc_st == 0`),
//! * For each LTRP entry: a fixed-length `poc_lsb_lt` of
//!   `log2_max_pic_order_cnt_lsb_minus4 + 4` bits.
//!
//! The parser surfaces a [`RefPicListStruct`] with the per-entry data so
//! callers (the SPS parser, the slice-header parser, the DPB walker) can
//! resolve `DeltaPocSt[ listIdx ][ rplsIdx ][ i ]` per eq. 124 and the
//! `FullPocLsbLt[…]` value per eq. 84 without re-reading the bitstream.
//!
//! Round-8 deliverable: no LTRP-only fixtures yet exercise the long-term
//! branch end-to-end through the decoder, but the parser handles both
//! sub-paths so that any RPL-bearing SPS / slice header round-trips.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Single entry inside a `ref_pic_list_struct()`. STRP entries carry the
/// signed `delta_poc_st`; LTRP entries carry the unsigned `poc_lsb_lt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefPicListEntry {
    /// Short-term reference picture entry.
    Strp {
        /// `delta_poc_st[ i ]` decoded as `ue(v)` (the magnitude). Per
        /// §7.4.8 eq. 124, the signed `DeltaPocSt[ i ]` value is recovered
        /// by combining this with [`Self::Strp::sign`].
        delta_poc_st: u32,
        /// `strp_entry_sign_flag[ i ]`. `true` ⇒ positive (>= 0); `false`
        /// ⇒ negative. Inferred to `true` (positive) when not present
        /// (i.e. when `delta_poc_st == 0`).
        sign: bool,
    },
    /// Long-term reference picture entry.
    Ltrp {
        /// `poc_lsb_lt[ i ]` — fixed-length value of
        /// `log2_max_pic_order_cnt_lsb_minus4 + 4` bits.
        poc_lsb_lt: u32,
    },
}

impl RefPicListEntry {
    /// Compute the signed `DeltaPocSt[ i ]` value per §7.4.8 eq. 124.
    /// Returns `None` for LTRP entries (the spec leaves DeltaPocSt
    /// undefined for LTRP).
    pub fn signed_delta_poc(self) -> Option<i32> {
        match self {
            Self::Strp { delta_poc_st, sign } => {
                let mag = delta_poc_st as i32;
                Some(if sign { mag } else { -mag })
            }
            Self::Ltrp { .. } => None,
        }
    }

    pub fn is_strp(self) -> bool {
        matches!(self, Self::Strp { .. })
    }

    pub fn is_ltrp(self) -> bool {
        matches!(self, Self::Ltrp { .. })
    }
}

/// One parsed `ref_pic_list_struct( listIdx, rplsIdx, ltrpFlag )`. The
/// caller knows `listIdx` and `rplsIdx` from context (SPS list index, or
/// `num_ref_pic_lists_in_sps[i]` for slice-header-direct).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefPicListStruct {
    pub num_strp_entries: u32,
    pub num_ltrp_entries: u32,
    pub entries: Vec<RefPicListEntry>,
}

impl RefPicListStruct {
    /// `NumEntriesInList[ listIdx ][ rplsIdx ]` per §7.4.8 eq. 123.
    pub fn num_entries_in_list(&self) -> u32 {
        self.num_strp_entries + self.num_ltrp_entries
    }
}

/// Hard sanity bound on entry counts to prevent oversized allocations on
/// hostile bitstreams. The spec caps each at
/// `sps_max_dec_pic_buffering_minus1`, which itself is bounded by the
/// level table — 64 is a comfortable upper bound.
pub const MAX_REF_PIC_LIST_ENTRIES: u32 = 64;

/// Parse one `ref_pic_list_struct()`. The caller supplies the
/// `long_term_ref_pics_flag` (`ltrpFlag` in spec notation) and the
/// `log2_max_pic_order_cnt_lsb` value used to size `poc_lsb_lt`.
pub fn parse_ref_pic_list_struct(
    br: &mut BitReader,
    long_term_ref_pics_flag: bool,
    log2_max_pic_order_cnt_lsb: u32,
) -> Result<RefPicListStruct> {
    let num_strp_entries = br.ue()?;
    if num_strp_entries > MAX_REF_PIC_LIST_ENTRIES {
        return Err(Error::invalid(format!(
            "evc rpl: num_strp_entries {num_strp_entries} > {MAX_REF_PIC_LIST_ENTRIES}"
        )));
    }
    let num_ltrp_entries = if long_term_ref_pics_flag {
        let n = br.ue()?;
        if n > MAX_REF_PIC_LIST_ENTRIES {
            return Err(Error::invalid(format!(
                "evc rpl: num_ltrp_entries {n} > {MAX_REF_PIC_LIST_ENTRIES}"
            )));
        }
        n
    } else {
        0
    };
    let total = num_strp_entries
        .checked_add(num_ltrp_entries)
        .ok_or_else(|| Error::invalid("evc rpl: total entry count overflow"))?;
    if total > MAX_REF_PIC_LIST_ENTRIES {
        return Err(Error::invalid(format!(
            "evc rpl: total entries {total} > {MAX_REF_PIC_LIST_ENTRIES}"
        )));
    }
    if log2_max_pic_order_cnt_lsb > 31 {
        return Err(Error::invalid(format!(
            "evc rpl: log2_max_pic_order_cnt_lsb {log2_max_pic_order_cnt_lsb} > 31"
        )));
    }
    let mut entries = Vec::with_capacity(total as usize);
    for _ in 0..total {
        let lt_ref_pic_flag = if num_ltrp_entries > 0 {
            br.u1()? != 0
        } else {
            // §7.4.8: when not present, inferred to 0 (STRP).
            false
        };
        if !lt_ref_pic_flag {
            let delta_poc_st = br.ue()?;
            let sign = if delta_poc_st > 0 {
                // strp_entry_sign_flag: u(1) — 1 = positive, 0 = negative.
                br.u1()? != 0
            } else {
                // When not present, inferred to 1 (positive) per §7.4.8.
                true
            };
            entries.push(RefPicListEntry::Strp { delta_poc_st, sign });
        } else {
            let poc_lsb_lt = br.u(log2_max_pic_order_cnt_lsb)?;
            entries.push(RefPicListEntry::Ltrp { poc_lsb_lt });
        }
    }
    // Spec invariant: the number of LTRP entries equals num_ltrp_entries.
    let observed_lt = entries.iter().filter(|e| e.is_ltrp()).count() as u32;
    if observed_lt != num_ltrp_entries {
        return Err(Error::invalid(format!(
            "evc rpl: lt_ref_pic_flag count {observed_lt} != num_ltrp_entries {num_ltrp_entries}"
        )));
    }
    Ok(RefPicListStruct {
        num_strp_entries,
        num_ltrp_entries,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    fn roundtrip_strp_only() {
        let mut e = BitEmitter::new();
        e.ue(2); // num_strp_entries
                 // entry 0: delta_poc_st = 1, sign = 1 (positive)
        e.ue(1);
        e.u(1, 1);
        // entry 1: delta_poc_st = 3, sign = 0 (negative)
        e.ue(3);
        e.u(1, 0);
        e.finish_with_trailing_bits();
        let bytes = e.into_bytes();
        let mut br = BitReader::new(&bytes);
        let rpl = parse_ref_pic_list_struct(&mut br, false, 4).unwrap();
        assert_eq!(rpl.num_strp_entries, 2);
        assert_eq!(rpl.num_ltrp_entries, 0);
        assert_eq!(rpl.num_entries_in_list(), 2);
        assert_eq!(rpl.entries[0].signed_delta_poc(), Some(1));
        assert_eq!(rpl.entries[1].signed_delta_poc(), Some(-3));
    }

    #[test]
    fn parses_strp_only() {
        roundtrip_strp_only();
    }

    #[test]
    fn parses_strp_zero_delta_no_sign_bit() {
        // delta_poc_st = 0 → strp_entry_sign_flag is omitted; entry must
        // still be valid and resolve to a non-negative 0 delta.
        let mut e = BitEmitter::new();
        e.ue(1); // num_strp_entries
        e.ue(0); // delta_poc_st = 0 → no sign bit
        e.finish_with_trailing_bits();
        let bytes = e.into_bytes();
        let mut br = BitReader::new(&bytes);
        let rpl = parse_ref_pic_list_struct(&mut br, false, 4).unwrap();
        assert_eq!(rpl.num_entries_in_list(), 1);
        assert_eq!(rpl.entries[0].signed_delta_poc(), Some(0));
    }

    #[test]
    fn parses_mixed_strp_and_ltrp() {
        // 1 STRP + 1 LTRP, log2_max_poc_lsb = 8 (so poc_lsb_lt is 8 bits).
        let mut e = BitEmitter::new();
        e.ue(1); // num_strp_entries
        e.ue(1); // num_ltrp_entries
                 // entry 0: lt_ref_pic_flag = 0 (STRP), delta = 2, sign = 1
        e.u(1, 0);
        e.ue(2);
        e.u(1, 1);
        // entry 1: lt_ref_pic_flag = 1 (LTRP), poc_lsb_lt = 0xAB (8 bits)
        e.u(1, 1);
        e.u(8, 0xAB);
        e.finish_with_trailing_bits();
        let bytes = e.into_bytes();
        let mut br = BitReader::new(&bytes);
        let rpl = parse_ref_pic_list_struct(&mut br, true, 8).unwrap();
        assert_eq!(rpl.num_strp_entries, 1);
        assert_eq!(rpl.num_ltrp_entries, 1);
        assert!(matches!(
            rpl.entries[0],
            RefPicListEntry::Strp {
                delta_poc_st: 2,
                sign: true
            }
        ));
        assert!(matches!(
            rpl.entries[1],
            RefPicListEntry::Ltrp { poc_lsb_lt: 0xAB }
        ));
        assert_eq!(rpl.entries[0].signed_delta_poc(), Some(2));
        assert!(rpl.entries[1].signed_delta_poc().is_none());
    }

    #[test]
    fn rejects_oversized_strp_count() {
        let mut e = BitEmitter::new();
        e.ue(MAX_REF_PIC_LIST_ENTRIES + 1);
        e.finish_with_trailing_bits();
        let bytes = e.into_bytes();
        let mut br = BitReader::new(&bytes);
        let err = parse_ref_pic_list_struct(&mut br, false, 4).unwrap_err();
        assert!(format!("{err}").contains("num_strp_entries"));
    }

    #[test]
    fn rejects_oversized_ltrp_count() {
        let mut e = BitEmitter::new();
        e.ue(0); // num_strp_entries
        e.ue(MAX_REF_PIC_LIST_ENTRIES + 1);
        e.finish_with_trailing_bits();
        let bytes = e.into_bytes();
        let mut br = BitReader::new(&bytes);
        let err = parse_ref_pic_list_struct(&mut br, true, 4).unwrap_err();
        assert!(format!("{err}").contains("num_ltrp_entries"));
    }

    #[test]
    fn empty_list_parses_cleanly() {
        let mut e = BitEmitter::new();
        e.ue(0); // num_strp_entries
        e.finish_with_trailing_bits();
        let bytes = e.into_bytes();
        let mut br = BitReader::new(&bytes);
        let rpl = parse_ref_pic_list_struct(&mut br, false, 4).unwrap();
        assert_eq!(rpl.num_entries_in_list(), 0);
    }
}
