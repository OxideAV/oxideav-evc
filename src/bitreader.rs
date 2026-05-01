//! MSB-first bit reader plus 0-th-order Exp-Golomb helpers for EVC RBSPs.
//!
//! Per ISO/IEC 23094-1 §7.2 / §9.2, EVC syntax elements are coded MSB-first
//! within each byte. The `ue(v)` / `se(v)` codes are 0-th order Exp-Golomb,
//! identical to the AVC / HEVC mapping.
//!
//! EVC uses a length-prefixed NAL framing model (Annex B `nal_unit_length` is
//! `u(32)`), so the EVC syntax does **not** specify start-code emulation
//! prevention bytes — RBSPs are read directly out of the NAL body. This
//! reader therefore operates on the raw NAL payload after the 2-byte NAL
//! header.

use oxideav_core::{Error, Result};

/// MSB-first bit reader over a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    acc: u64,
    bits_in_acc: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Current read offset in bits, measured from the start of the input.
    pub fn bit_position(&self) -> u64 {
        self.byte_pos as u64 * 8 - self.bits_in_acc as u64
    }

    pub fn bits_remaining(&self) -> u64 {
        (self.data.len() as u64 - self.byte_pos as u64) * 8 + self.bits_in_acc as u64
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc % 8 == 0
    }

    /// Discard bits up to the next byte boundary (§7.2 byte_aligned()).
    pub fn align_to_byte(&mut self) {
        let drop = self.bits_in_acc % 8;
        self.acc <<= drop;
        self.bits_in_acc -= drop;
    }

    fn refill(&mut self) {
        while self.bits_in_acc <= 56 && self.byte_pos < self.data.len() {
            self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.bits_in_acc);
            self.bits_in_acc += 8;
            self.byte_pos += 1;
        }
    }

    /// Read `n` bits (0..=32) as an unsigned integer.
    pub fn u(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_acc < n {
            self.refill();
            if self.bits_in_acc < n {
                return Err(Error::invalid("evc bitreader: out of bits"));
            }
        }
        let v = (self.acc >> (64 - n)) as u32;
        self.acc <<= n;
        self.bits_in_acc -= n;
        Ok(v)
    }

    /// Read a single bit.
    pub fn u1(&mut self) -> Result<u32> {
        self.u(1)
    }

    /// Read `n` bits (0..=64).
    pub fn u_long(&mut self, n: u32) -> Result<u64> {
        debug_assert!(n <= 64);
        if n <= 32 {
            return Ok(self.u(n)? as u64);
        }
        let hi = self.u(n - 32)? as u64;
        let lo = self.u(32)? as u64;
        Ok((hi << 32) | lo)
    }

    /// Skip `n` bits.
    pub fn skip(&mut self, mut n: u32) -> Result<()> {
        while n > 32 {
            self.u(32)?;
            n -= 32;
        }
        if n > 0 {
            self.u(n)?;
        }
        Ok(())
    }

    /// Read a 0-th order unsigned Exp-Golomb code, `ue(v)` (§9.2).
    pub fn ue(&mut self) -> Result<u32> {
        let mut zeros: u32 = 0;
        while self.u1()? == 0 {
            zeros += 1;
            if zeros > 32 {
                return Err(Error::invalid("evc ue(v): too many leading zeros"));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let suffix = self.u(zeros)?;
        Ok((1u32 << zeros) - 1 + suffix)
    }

    /// Read a 0-th order signed Exp-Golomb code, `se(v)` (§9.2.2).
    pub fn se(&mut self) -> Result<i32> {
        let k = self.ue()?;
        // 0 -> 0, 1 -> 1, 2 -> -1, 3 -> 2, 4 -> -2, ...
        let val = ((k + 1) >> 1) as i32;
        if k & 1 == 1 {
            Ok(val)
        } else {
            Ok(-val)
        }
    }

    /// Read a k-th order unsigned Exp-Golomb code, `uek(v)` (§9.2 with k > 0).
    pub fn uek(&mut self, k: u32) -> Result<u32> {
        // EG-k: prefix is unary (zero count = M), suffix length M+k bits.
        let mut zeros: u32 = 0;
        while self.u1()? == 0 {
            zeros += 1;
            if zeros > 32 {
                return Err(Error::invalid("evc uek(v): too many leading zeros"));
            }
        }
        let total_suffix = zeros + k;
        let suffix = if total_suffix > 0 {
            self.u(total_suffix)?
        } else {
            0
        };
        // value = ((1<<zeros) - 1) * (1<<k) + suffix
        let base = ((1u32 << zeros).wrapping_sub(1)) << k;
        Ok(base + suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_msb_first() {
        let data = [0b1011_0001u8, 0b0101_0101];
        let mut br = BitReader::new(&data);
        assert_eq!(br.u(1).unwrap(), 1);
        assert_eq!(br.u(2).unwrap(), 0b01);
        assert_eq!(br.u(5).unwrap(), 0b1_0001);
        assert_eq!(br.u(8).unwrap(), 0b0101_0101);
    }

    #[test]
    fn ue_sequence() {
        // bits: 1 (=0), 010 (=1), 011 (=2), 00100 (=3)
        // concatenated: 1 010 011 00100 = 1010_0110_0100
        let data = [0b1010_0110, 0b0100_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.ue().unwrap(), 0);
        assert_eq!(br.ue().unwrap(), 1);
        assert_eq!(br.ue().unwrap(), 2);
        assert_eq!(br.ue().unwrap(), 3);
    }

    #[test]
    fn se_sequence() {
        // se: 0 -> 0, 1 -> 1, 2 -> -1, 3 -> 2
        let data = [0b1010_0110, 0b0100_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.se().unwrap(), 0);
        assert_eq!(br.se().unwrap(), 1);
        assert_eq!(br.se().unwrap(), -1);
        assert_eq!(br.se().unwrap(), 2);
    }

    #[test]
    fn uek_zero_order_matches_ue() {
        // uek with k=0 must equal ue.
        let data = [0b1010_0110, 0b0100_0000];
        let mut br_ue = BitReader::new(&data);
        let mut br_uek = BitReader::new(&data);
        for _ in 0..4 {
            assert_eq!(br_ue.ue().unwrap(), br_uek.uek(0).unwrap());
        }
    }

    #[test]
    fn uek_first_order() {
        // EG-1 codewords: 0 -> "10", 1 -> "11", 2 -> "0100", 3 -> "0101",
        // 4 -> "0110", 5 -> "0111".
        // bits: 10 11 0100 0101 = 1011_0100_0101
        let data = [0b1011_0100, 0b0101_0000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.uek(1).unwrap(), 0);
        assert_eq!(br.uek(1).unwrap(), 1);
        assert_eq!(br.uek(1).unwrap(), 2);
        assert_eq!(br.uek(1).unwrap(), 3);
    }

    #[test]
    fn align_to_byte() {
        let data = [0b1111_0000u8, 0xAA];
        let mut br = BitReader::new(&data);
        assert_eq!(br.u(3).unwrap(), 0b111);
        br.align_to_byte();
        assert!(br.is_byte_aligned());
        assert_eq!(br.u(8).unwrap(), 0xAA);
    }

    #[test]
    fn underflow_is_error() {
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        assert!(br.u(16).is_err());
    }
}
