//! MSB-first bit writer plus 0-th-order Exp-Golomb helpers for EVC RBSPs
//! — the write-side dual of [`crate::bitreader`].
//!
//! Per ISO/IEC 23094-1 §7.2 / §9.2, EVC syntax elements are coded
//! MSB-first within each byte and the `ue(v)` / `se(v)` codes are 0-th
//! order Exp-Golomb. EVC's Annex B framing is length-prefixed
//! (`nal_unit_length` is `u(32)`), so there is **no** start-code
//! emulation prevention: the writer emits RBSP bytes verbatim into the
//! NAL body.

/// MSB-first bit writer producing an RBSP byte vector.
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Number of valid bits in the partially-filled last byte (0..8).
    /// When 0 the writer is byte-aligned and `bytes` is complete.
    partial_bits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of bits written so far.
    pub fn bit_position(&self) -> u64 {
        if self.partial_bits == 0 {
            self.bytes.len() as u64 * 8
        } else {
            (self.bytes.len() as u64 - 1) * 8 + self.partial_bits as u64
        }
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.partial_bits == 0
    }

    /// Append a single bit.
    pub fn put_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        if self.partial_bits == 0 {
            self.bytes.push(0);
        }
        let last = self.bytes.last_mut().expect("just pushed");
        *last |= (bit & 1) << (7 - self.partial_bits);
        self.partial_bits = (self.partial_bits + 1) % 8;
    }

    /// Append the `n` low bits of `value`, MSB-first (`u(n)` — §7.2).
    pub fn u(&mut self, n: u32, value: u32) {
        debug_assert!(n <= 32);
        for k in (0..n).rev() {
            self.put_bit(((value >> k) & 1) as u8);
        }
    }

    /// Append one flag bit (`u(1)`).
    pub fn u1(&mut self, value: bool) {
        self.put_bit(u8::from(value));
    }

    /// Append a 0-th order unsigned Exp-Golomb code (`ue(v)` — §9.2).
    pub fn ue(&mut self, value: u32) {
        // codeNum = value; codeword = [zeros × '0'] '1' [suffix].
        let v1 = (value as u64) + 1;
        let bits = 64 - v1.leading_zeros();
        let zeros = bits - 1;
        for _ in 0..zeros {
            self.put_bit(0);
        }
        self.put_bit(1);
        for k in (0..zeros).rev() {
            self.put_bit(((v1 >> k) & 1) as u8);
        }
    }

    /// Append a 0-th order signed Exp-Golomb code (`se(v)` — §9.2.2):
    /// `0 → 0, 1 → 1, −1 → 2, 2 → 3, −2 → 4, …` (the inverse of
    /// [`crate::bitreader::BitReader::se`]).
    pub fn se(&mut self, value: i32) {
        let code = if value > 0 {
            (value as u32) * 2 - 1
        } else {
            (value.unsigned_abs()) * 2
        };
        self.ue(code);
    }

    /// Pad with zero bits up to the next byte boundary.
    pub fn align_to_byte_zero(&mut self) {
        while self.partial_bits != 0 {
            self.put_bit(0);
        }
    }

    /// §7.3.2.10 `rbsp_trailing_bits( )`: one `rbsp_stop_one_bit` (1)
    /// followed by zero `rbsp_alignment_zero_bit`s to the byte boundary.
    pub fn rbsp_trailing_bits(&mut self) {
        self.put_bit(1);
        self.align_to_byte_zero();
    }

    /// Consume the writer, returning the RBSP bytes. Panics in debug
    /// builds if the writer is not byte-aligned (callers must close with
    /// [`Self::rbsp_trailing_bits`] or [`Self::align_to_byte_zero`]).
    pub fn into_bytes(self) -> Vec<u8> {
        debug_assert!(self.partial_bits == 0, "bitwriter: unaligned finish");
        self.bytes
    }

    /// Borrowed view of the bytes emitted so far (final partial byte
    /// included, zero-padded).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    /// u(n) fields written MSB-first read back identically.
    #[test]
    fn u_round_trips() {
        let mut bw = BitWriter::new();
        bw.u(1, 1);
        bw.u(2, 0b01);
        bw.u(5, 0b1_0001);
        bw.u(8, 0b0101_0101);
        bw.align_to_byte_zero();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.u(1).unwrap(), 1);
        assert_eq!(br.u(2).unwrap(), 0b01);
        assert_eq!(br.u(5).unwrap(), 0b1_0001);
        assert_eq!(br.u(8).unwrap(), 0b0101_0101);
    }

    /// ue(v) round-trips across the whole small range plus wide values.
    #[test]
    fn ue_round_trips() {
        let vals: Vec<u32> = (0..64)
            .chain([255, 256, 1023, 65535, 1 << 20, u32::MAX - 1])
            .collect();
        let mut bw = BitWriter::new();
        for &v in &vals {
            bw.ue(v);
        }
        bw.align_to_byte_zero();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for &v in vals.iter().take(64) {
            assert_eq!(br.ue().unwrap(), v, "ue({v})");
        }
        // The >32-leading-zero wide values exceed the reader's ue()
        // guard range only above 2^32-2; every listed value must parse.
        for &v in &vals[64..] {
            assert_eq!(br.ue().unwrap(), v, "ue({v})");
        }
    }

    /// se(v) round-trips over the signed range, matching the reader's
    /// §9.2.2 mapping exactly.
    #[test]
    fn se_round_trips() {
        let vals: Vec<i32> = (-40..=40).chain([1000, -1000, 32767, -32768]).collect();
        let mut bw = BitWriter::new();
        for &v in &vals {
            bw.se(v);
        }
        bw.align_to_byte_zero();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        for &v in &vals {
            assert_eq!(br.se().unwrap(), v, "se({v})");
        }
    }

    /// rbsp_trailing_bits emits the stop bit + zero padding; the reader
    /// sees the payload then aligns cleanly.
    #[test]
    fn trailing_bits_shape() {
        let mut bw = BitWriter::new();
        bw.u(3, 0b101);
        bw.rbsp_trailing_bits();
        let bytes = bw.into_bytes();
        assert_eq!(bytes, vec![0b1011_0000]);
    }

    /// bit_position tracks partial bytes.
    #[test]
    fn bit_position_tracks() {
        let mut bw = BitWriter::new();
        assert_eq!(bw.bit_position(), 0);
        bw.u(3, 0);
        assert_eq!(bw.bit_position(), 3);
        assert!(!bw.is_byte_aligned());
        bw.u(5, 0);
        assert_eq!(bw.bit_position(), 8);
        assert!(bw.is_byte_aligned());
    }

    /// Interleaved ue/se/u sequences round-trip — the shape used by the
    /// §7.3.4 slice-header writer.
    #[test]
    fn mixed_sequence_round_trips() {
        let mut bw = BitWriter::new();
        bw.ue(0); // slice_pps_id
        bw.ue(2); // slice_type
        bw.u1(false); // no_output_of_prior_pics_flag
        bw.u1(false); // slice_deblocking_filter_flag
        bw.u(6, 22); // slice_qp
        bw.se(0); // slice_cb_qp_offset
        bw.se(-3); // slice_cr_qp_offset
        bw.align_to_byte_zero();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.ue().unwrap(), 0);
        assert_eq!(br.ue().unwrap(), 2);
        assert_eq!(br.u1().unwrap(), 0);
        assert_eq!(br.u1().unwrap(), 0);
        assert_eq!(br.u(6).unwrap(), 22);
        assert_eq!(br.se().unwrap(), 0);
        assert_eq!(br.se().unwrap(), -3);
    }
}
