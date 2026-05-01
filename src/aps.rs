//! EVC adaptation parameter set parser (ISO/IEC 23094-1 §7.3.2.3).
//!
//! The APS body is one of:
//!
//! * **ALF APS** (`aps_params_type == 0`) — `alf_data()` per §7.3.5;
//! * **DRA APS** (`aps_params_type == 1`) — `dra_data()` per §7.3.6.
//!
//! Round-1 surfaces only the APS header (id + type) and the trailing
//! `aps_extension_flag`. The internal `alf_data()` / `dra_data()` payloads
//! are captured as raw byte slices for round-2 to consume; their bit-level
//! parsers live in dedicated modules once the in-loop filter pass lands.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

pub const APS_PARAMS_TYPE_ALF: u8 = 0;
pub const APS_PARAMS_TYPE_DRA: u8 = 1;

#[derive(Clone, Debug)]
pub struct Aps {
    pub adaptation_parameter_set_id: u8,
    pub aps_params_type: u8,
    /// Raw payload bytes of the inner alf_data() / dra_data() — round-2
    /// will replace this with structured fields.
    pub payload_raw: Vec<u8>,
    pub aps_extension_flag: bool,
}

impl Aps {
    pub fn is_alf(&self) -> bool {
        self.aps_params_type == APS_PARAMS_TYPE_ALF
    }

    pub fn is_dra(&self) -> bool {
        self.aps_params_type == APS_PARAMS_TYPE_DRA
    }
}

pub fn parse(rbsp: &[u8]) -> Result<Aps> {
    if rbsp.is_empty() {
        return Err(Error::invalid("evc aps: empty RBSP"));
    }
    let mut br = BitReader::new(rbsp);
    let adaptation_parameter_set_id = br.u(5)? as u8;
    let aps_params_type = br.u(3)? as u8;
    if aps_params_type > APS_PARAMS_TYPE_DRA {
        return Err(Error::invalid(format!(
            "evc aps: unknown aps_params_type {aps_params_type} (round-1 supports 0=ALF, 1=DRA)"
        )));
    }

    // The payload (alf_data() or dra_data()) is bit-aligned, of unknown
    // length until parsed. Round-1 captures the remaining bits up to (but
    // not including) the rbsp_trailing_bits() / aps_extension_flag tail by
    // copying raw bytes after the 1-byte header. Bit-level interpretation
    // happens in round-2.
    //
    // We can't surface aps_extension_flag without finding the end of the
    // payload, so we conservatively read all bits up to the last byte and
    // expose the payload bytes (still bit-shifted by 8 from the start of
    // the RBSP).
    let payload_raw = rbsp[1..].to_vec();

    // Consume all remaining bits so we can find the trailing bits / extension
    // flag at the very end. We only attempt this if there is enough RBSP for
    // the trailing-bit byte; otherwise we return Unsupported aps_extension.
    let total_bits = (rbsp.len() as u64) * 8;
    let consumed = br.bit_position();
    let remaining = total_bits - consumed;
    let aps_extension_flag = if remaining >= 1 {
        // The very last "1" bit of the RBSP is rbsp_stop_one_bit; we look
        // for it from the end to determine the SODB length, then read the
        // single u(1) `aps_extension_flag` immediately preceding it.
        find_aps_extension_flag(rbsp).unwrap_or(false)
    } else {
        false
    };

    Ok(Aps {
        adaptation_parameter_set_id,
        aps_params_type,
        payload_raw,
        aps_extension_flag,
    })
}

/// Walk the RBSP tail to find the `rbsp_stop_one_bit` and read the bit
/// immediately before it as `aps_extension_flag`.
fn find_aps_extension_flag(rbsp: &[u8]) -> Option<bool> {
    // Scan from the end for the first non-zero byte.
    let last_nz = rbsp.iter().rposition(|&b| b != 0)?;
    let last_byte = rbsp[last_nz];
    // Find the lowest set bit — that's rbsp_stop_one_bit.
    let trailing_zeros = last_byte.trailing_zeros() as i32;
    // aps_extension_flag is the bit immediately above rbsp_stop_one_bit.
    let stop_bit_idx = trailing_zeros; // bit position from LSB (0 = LSB)
    let ext_bit_idx = stop_bit_idx + 1;
    if ext_bit_idx >= 8 {
        // Crosses a byte boundary — pull from the previous byte's MSB.
        if last_nz == 0 {
            return None;
        }
        let prev = rbsp[last_nz - 1];
        return Some((prev & 1) != 0);
    }
    Some((last_byte >> ext_bit_idx) & 1 != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    fn emit_minimal_aps(id: u32, params_type: u32, ext: u32) -> Vec<u8> {
        let mut e = BitEmitter::new();
        e.u(5, id); // adaptation_parameter_set_id
        e.u(3, params_type); // aps_params_type
                             // alf_data() / dra_data() body — we emit a tiny stub: a single 0 bit
                             // so the round-1 capture has something to record.
        e.u(1, 0);
        e.u(1, ext); // aps_extension_flag
                     // No aps_extension_data_flag loop (since ext=0 case).
        e.finish_with_trailing_bits();
        e.into_bytes()
    }

    #[test]
    fn parse_alf_aps() {
        let rbsp = emit_minimal_aps(7, APS_PARAMS_TYPE_ALF as u32, 0);
        let aps = parse(&rbsp).unwrap();
        assert_eq!(aps.adaptation_parameter_set_id, 7);
        assert_eq!(aps.aps_params_type, APS_PARAMS_TYPE_ALF);
        assert!(aps.is_alf());
        assert!(!aps.is_dra());
        assert!(!aps.aps_extension_flag);
    }

    #[test]
    fn parse_dra_aps() {
        let rbsp = emit_minimal_aps(3, APS_PARAMS_TYPE_DRA as u32, 0);
        let aps = parse(&rbsp).unwrap();
        assert_eq!(aps.aps_params_type, APS_PARAMS_TYPE_DRA);
        assert!(aps.is_dra());
    }

    #[test]
    fn rejects_unknown_aps_type() {
        // params_type = 7 is reserved.
        let mut e = BitEmitter::new();
        e.u(5, 0);
        e.u(3, 7);
        e.u(1, 0);
        e.u(1, 0);
        e.finish_with_trailing_bits();
        assert!(parse(&e.into_bytes()).is_err());
    }

    #[test]
    fn empty_rbsp_rejected() {
        assert!(parse(&[]).is_err());
    }
}
