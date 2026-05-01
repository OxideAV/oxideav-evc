//! EVC NAL unit framing (ISO/IEC 23094-1 §7.3.1, §7.4.2, Annex B).
//!
//! ## Framing
//!
//! Per ISO/IEC 23094-1 Annex B, the **raw bitstream file storage format** is
//! length-prefixed: each NAL unit is preceded by a 4-byte big-endian
//! `nal_unit_length` field (`u(32)`). EVC does **not** define an
//! Annex-B-style start-code (`0x000001`) framing or emulation-prevention
//! bytes — the spec relies on length-prefix to delimit NAL units.
//!
//! For tooling friendliness we still expose [`iter_annex_b`] to recognise
//! `0x000001` / `0x00000001` start codes (some test vectors and ad-hoc
//! transports use them); but the canonical path is
//! [`iter_length_prefixed`].
//!
//! ## NAL header (§7.3.1.2 / §7.4.2.2)
//!
//! 2 bytes, MSB-first:
//!
//! ```text
//!   forbidden_zero_bit       f(1)   — must be 0
//!   nal_unit_type_plus1      u(6)   — NalUnitType = nal_unit_type_plus1 - 1
//!   nuh_temporal_id          u(3)
//!   nuh_reserved_zero_5bits  u(5)   — must be 0 in this spec version
//!   nuh_extension_flag       u(1)   — must be 0 in this spec version
//! ```
//!
//! ## NAL unit types (Table 4)
//!
//! | code | name              | class   |
//! |------|-------------------|---------|
//! | 0    | NONIDR_NUT        | VCL     |
//! | 1    | IDR_NUT           | VCL     |
//! | 2-23 | RSV_VCL_NUT*      | VCL     |
//! | 24   | SPS_NUT           | non-VCL |
//! | 25   | PPS_NUT           | non-VCL |
//! | 26   | APS_NUT           | non-VCL |
//! | 27   | FD_NUT            | non-VCL |
//! | 28   | SEI_NUT           | non-VCL |
//! | 29-55| RSV_NONVCL*       | non-VCL |
//! | 56-62| UNSPEC_NUT*       | non-VCL |

use oxideav_core::{Error, Result};

/// EVC NAL unit type codes (§7.4.2.2, Table 4). 6-bit field values stored
/// post-`-1` (i.e. equal to `nal_unit_type_plus1 - 1`).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NalUnitType {
    /// 0 — coded slice of a non-IDR picture.
    NonIdr,
    /// 1 — coded slice of an IDR picture.
    Idr,
    /// 2..=23 — reserved VCL.
    RsvVcl(u8),
    /// 24 — sequence parameter set.
    Sps,
    /// 25 — picture parameter set.
    Pps,
    /// 26 — adaptation parameter set.
    Aps,
    /// 27 — filler data.
    Fd,
    /// 28 — supplemental enhancement information.
    Sei,
    /// 29..=55 — reserved non-VCL.
    RsvNonVcl(u8),
    /// 56..=62 — unspecified non-VCL.
    Unspec(u8),
}

impl NalUnitType {
    /// Decode a raw 6-bit `NalUnitType` value (`nal_unit_type_plus1 - 1`).
    pub fn from_u8(v: u8) -> Self {
        use NalUnitType::*;
        match v {
            0 => NonIdr,
            1 => Idr,
            2..=23 => RsvVcl(v),
            24 => Sps,
            25 => Pps,
            26 => Aps,
            27 => Fd,
            28 => Sei,
            29..=55 => RsvNonVcl(v),
            56..=62 => Unspec(v),
            // 63 would mean nal_unit_type_plus1 == 64, which can never be
            // encoded in the 6-bit field; we still surface it as Unspec to
            // avoid panicking on hand-built fixtures.
            _ => Unspec(v),
        }
    }

    pub fn as_u8(self) -> u8 {
        use NalUnitType::*;
        match self {
            NonIdr => 0,
            Idr => 1,
            RsvVcl(v) => v,
            Sps => 24,
            Pps => 25,
            Aps => 26,
            Fd => 27,
            Sei => 28,
            RsvNonVcl(v) => v,
            Unspec(v) => v,
        }
    }

    /// VCL classification (§7.4.2.2 Table 4): NalUnitType in 0..=23.
    pub fn is_vcl(self) -> bool {
        matches!(self.as_u8(), 0..=23)
    }

    /// Whether this is a parameter-set NAL (SPS / PPS / APS).
    pub fn is_parameter_set(self) -> bool {
        matches!(self, NalUnitType::Sps | NalUnitType::Pps | NalUnitType::Aps)
    }

    /// IDR random-access NAL (NalUnitType == 1).
    pub fn is_idr(self) -> bool {
        matches!(self, NalUnitType::Idr)
    }
}

/// Parsed 2-byte EVC NAL header (§7.3.1.2).
#[derive(Clone, Copy, Debug)]
pub struct NalHeader {
    pub nal_unit_type: NalUnitType,
    pub nuh_temporal_id: u8,
    pub nuh_reserved_zero_5bits: u8,
    pub nuh_extension_flag: bool,
}

impl NalHeader {
    /// Parse the 2-byte NAL header. Caller must pass at least 2 bytes.
    ///
    /// The strict-decoder behaviour for §7.4.2.2 reserved bits:
    /// * `forbidden_zero_bit` non-zero → hard error (corrupt or non-EVC).
    /// * `nuh_reserved_zero_5bits` or `nuh_extension_flag` non-zero →
    ///   ignored (spec says decoders shall discard the NAL); we still
    ///   surface the parsed values so the caller can decide.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 2 {
            return Err(Error::invalid("evc: NAL header < 2 bytes"));
        }
        let b0 = bytes[0];
        let b1 = bytes[1];
        // forbidden_zero_bit (§7.4.2.2)
        if b0 & 0x80 != 0 {
            return Err(Error::invalid(
                "evc: NAL forbidden_zero_bit must be 0 (corrupt or non-EVC)",
            ));
        }
        // 16-bit header MSB-first layout (§7.3.1.2):
        //   bit 15        : forbidden_zero_bit (1)
        //   bits 14..9    : nal_unit_type_plus1 (6)
        //   bits 8..6     : nuh_temporal_id (3)
        //   bits 5..1     : nuh_reserved_zero_5bits (5)
        //   bit 0         : nuh_extension_flag (1)
        let combined = ((b0 as u16) << 8) | b1 as u16;
        let nal_unit_type_plus1 = ((combined >> 9) & 0x3F) as u8;
        if nal_unit_type_plus1 == 0 {
            return Err(Error::invalid(
                "evc: nal_unit_type_plus1 must not be 0 (§7.4.2.2)",
            ));
        }
        let nal_unit_type_raw = nal_unit_type_plus1 - 1;
        let tid = ((combined >> 6) & 0x07) as u8;
        let reserved = ((combined >> 1) & 0x1F) as u8;
        let ext = (combined & 0x1) != 0;

        Ok(Self {
            nal_unit_type: NalUnitType::from_u8(nal_unit_type_raw),
            nuh_temporal_id: tid,
            nuh_reserved_zero_5bits: reserved,
            nuh_extension_flag: ext,
        })
    }
}

/// One NAL unit located in a buffer (zero-copy slice).
#[derive(Clone, Copy, Debug)]
pub struct NalRef<'a> {
    pub header: NalHeader,
    /// Body bytes including the 2-byte NAL header.
    pub raw: &'a [u8],
}

impl<'a> NalRef<'a> {
    /// Bytes after the 2-byte NAL header (i.e. the RBSP).
    pub fn rbsp(&self) -> &'a [u8] {
        &self.raw[2..]
    }
}

/// Iterate length-prefixed EVC NAL units (Annex B raw bitstream format —
/// `nal_unit_length` is `u(32)`).
pub fn iter_length_prefixed(data: &[u8]) -> Result<Vec<NalRef<'_>>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;
        if len < 2 {
            return Err(Error::invalid(format!(
                "evc: length-prefixed NAL too short (len={len})"
            )));
        }
        if i + len > data.len() {
            return Err(Error::invalid(format!(
                "evc: length-prefixed NAL out of bounds (len={len}, remaining={})",
                data.len() - i
            )));
        }
        let raw = &data[i..i + len];
        let header = NalHeader::parse(raw)?;
        out.push(NalRef { header, raw });
        i += len;
    }
    if i != data.len() {
        return Err(Error::invalid(format!(
            "evc: trailing {} bytes after length-prefixed NAL stream",
            data.len() - i
        )));
    }
    Ok(out)
}

/// Tolerant Annex B start-code scanner. EVC does not normatively use
/// Annex-B-style start codes, but ad-hoc transports occasionally do —
/// this iterator accepts either `0x000001` or `0x00000001` prefixes.
pub fn iter_annex_b(data: &[u8]) -> AnnexBIter<'_> {
    AnnexBIter { data, pos: 0 }
}

pub struct AnnexBIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = NalRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (sc_off, sc_len) = find_start_code(self.data, self.pos)?;
        let body_start = sc_off + sc_len;
        let body_end = match find_start_code(self.data, body_start) {
            Some((next_off, _)) => trim_trailing_zeros(&self.data[..next_off], body_start),
            None => trim_trailing_zeros(self.data, body_start),
        };
        self.pos = body_end;
        if body_end <= body_start || body_end - body_start < 2 {
            return self.next();
        }
        let raw = &self.data[body_start..body_end];
        let header = NalHeader::parse(raw).ok()?;
        Some(NalRef { header, raw })
    }
}

fn trim_trailing_zeros(slice: &[u8], min_end: usize) -> usize {
    let mut end = slice.len();
    while end > min_end && slice[end - 1] == 0 {
        end -= 1;
    }
    end
}

/// Search forward from `from` for the start of the next start-code prefix
/// (`0x000001` or `0x00000001`). Returns `(offset_of_first_zero, prefix_len)`.
pub fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            let mut j = i + 2;
            while j < data.len() && data[j] == 0 {
                j += 1;
            }
            if j < data.len() && data[j] == 0x01 {
                let prefix_len = j - i + 1;
                return Some((i, prefix_len));
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2-byte EVC NAL header for tests.
    /// Layout (MSB-first across the 16-bit word):
    ///   F(1) | NUT+1(6) | TID(3) | RES(5) | EXT(1)
    pub(crate) fn mk_header(nut_raw: u8, tid: u8, reserved: u8, ext: bool) -> [u8; 2] {
        let nut_plus1 = nut_raw + 1;
        let mut w: u16 = 0;
        w |= (nut_plus1 as u16 & 0x3F) << 9; // skip F (top bit stays 0)
        w |= (tid as u16 & 0x07) << 6;
        w |= (reserved as u16 & 0x1F) << 1;
        w |= if ext { 1 } else { 0 };
        [(w >> 8) as u8, (w & 0xFF) as u8]
    }

    #[test]
    fn parse_header_sps() {
        // SPS = type 24, tid 0
        let bytes = mk_header(24, 0, 0, false);
        let h = NalHeader::parse(&bytes).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Sps);
        assert_eq!(h.nuh_temporal_id, 0);
        assert_eq!(h.nuh_reserved_zero_5bits, 0);
        assert!(!h.nuh_extension_flag);
    }

    #[test]
    fn parse_header_idr() {
        let bytes = mk_header(1, 0, 0, false);
        let h = NalHeader::parse(&bytes).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Idr);
        assert!(h.nal_unit_type.is_vcl());
        assert!(h.nal_unit_type.is_idr());
    }

    #[test]
    fn parse_header_pps_aps_sei_fd() {
        let h = NalHeader::parse(&mk_header(25, 0, 0, false)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Pps);
        let h = NalHeader::parse(&mk_header(26, 0, 0, false)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Aps);
        let h = NalHeader::parse(&mk_header(27, 0, 0, false)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Fd);
        let h = NalHeader::parse(&mk_header(28, 0, 0, false)).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Sei);
    }

    #[test]
    fn classification() {
        assert!(NalUnitType::NonIdr.is_vcl());
        assert!(NalUnitType::Idr.is_vcl());
        assert!(!NalUnitType::Sps.is_vcl());
        assert!(NalUnitType::Sps.is_parameter_set());
        assert!(NalUnitType::Pps.is_parameter_set());
        assert!(NalUnitType::Aps.is_parameter_set());
        assert!(!NalUnitType::Sei.is_parameter_set());
    }

    #[test]
    fn forbidden_bit_rejected() {
        // F bit set
        let bytes = [0x80, 0x00];
        assert!(NalHeader::parse(&bytes).is_err());
    }

    #[test]
    fn zero_nut_plus1_rejected() {
        // nal_unit_type_plus1 == 0 (whole NUT field is zero)
        let bytes = [0x00, 0x00];
        assert!(NalHeader::parse(&bytes).is_err());
    }

    #[test]
    fn temporal_id_round_trip() {
        let bytes = mk_header(24, 5, 0, false);
        let h = NalHeader::parse(&bytes).unwrap();
        assert_eq!(h.nuh_temporal_id, 5);
    }

    #[test]
    fn length_prefixed_two_nals() {
        let sps = mk_header(24, 0, 0, false);
        let pps = mk_header(25, 0, 0, false);
        // First NAL: header(2) + 1 byte body, length=3
        // Second NAL: header(2) + 0 body, length=2
        let data = [
            0, 0, 0, 3, sps[0], sps[1], 0xAA, // sps len=3
            0, 0, 0, 2, pps[0], pps[1],
        ];
        let nals = iter_length_prefixed(&data).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Pps);
        assert_eq!(nals[0].rbsp(), &[0xAA]);
        assert!(nals[1].rbsp().is_empty());
    }

    #[test]
    fn length_prefixed_overflow_rejected() {
        // Length 0xFF says 255 bytes but we only have 2.
        let data = [0u8, 0, 0, 0xFF, 0x40, 0x00];
        assert!(iter_length_prefixed(&data).is_err());
    }

    #[test]
    fn length_prefixed_too_short_rejected() {
        // length=1 is below 2-byte header minimum.
        let data = [0u8, 0, 0, 1, 0x40];
        assert!(iter_length_prefixed(&data).is_err());
    }

    #[test]
    fn annex_b_iter_finds_two_nals() {
        let sps = mk_header(24, 0, 0, false);
        let idr = mk_header(1, 0, 0, false);
        let data = [
            0, 0, 0, 1, sps[0], sps[1], 0xAA, //
            0, 0, 1, idr[0], idr[1], 0xBB, 0xCC,
        ];
        let nals: Vec<_> = iter_annex_b(&data).collect();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Idr);
    }
}
