//! Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
//! decoder foundation.
//!
//! This crate currently implements the **parser foundation** only:
//!
//! * [`bitreader`] — MSB-first bit reader with `u(n)` / `ue(v)` / `se(v)` /
//!   `uek(v)` helpers (§9.2).
//! * [`nal`] — 2-byte NAL header (§7.3.1.2 / §7.4.2.2) plus length-prefixed
//!   (Annex B raw bitstream) framing and a tolerant Annex-B-style start-code
//!   scanner.
//! * [`sps`] — `seq_parameter_set_rbsp()` (§7.3.2.1) parser, with
//!   sanity-bounded picture dimensions (`MAX_DIMENSION = 32768`).
//! * [`pps`] — `pic_parameter_set_rbsp()` (§7.3.2.2) parser.
//! * [`aps`] — `adaptation_parameter_set_rbsp()` (§7.3.2.3) header parser
//!   (ALF / DRA payload bytes captured raw for round-2).
//! * [`slice_header`] — `slice_header()` (§7.3.4) round-1 subset (PPS id,
//!   slice type, POC LSB, QP, deblocking offsets).
//! * [`decoder`] — registry factory; the registered decoder returns
//!   `Error::Unsupported("EVC pixel decode not yet implemented")` since
//!   the per-CU decoder lands in round 2.
//!
//! All section / clause numbers refer to **ISO/IEC 23094-1:2020(E)** at
//! `docs/video/evc/ISO_IEC_23094-1-EVC-2020.pdf`. No third-party EVC
//! decoder source (MPEG-5 reference, `xeve`, `xevd`, …) was consulted —
//! every parser is spec-only.

pub mod aps;
pub mod bitreader;
pub mod decoder;
pub mod nal;
pub mod pps;
pub mod slice_header;
pub mod sps;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry};

/// Public codec id string. Matches the aggregator feature name `evc`.
pub const CODEC_ID_STR: &str = "evc";

/// Summary info recoverable from a bare EVC SPS — the public return type
/// of [`probe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvcFileInfo {
    pub width: u32,
    pub height: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub chroma_format_idc: u32,
}

/// Probe a buffer for an EVC bitstream and return summary info from the
/// first parseable SPS. Accepts either Annex B raw-bitstream framing
/// (`u(32)` length prefix, the canonical case per ISO/IEC 23094-1 Annex B)
/// or the tolerant `0x000001` / `0x00000001` start-code scanner.
///
/// Returns `None` when no SPS is found; bubbles up `Some(_)` for the
/// first SPS that parses cleanly.
pub fn probe(input: &[u8]) -> Option<EvcFileInfo> {
    // Try length-prefixed first — that's what Annex B specifies.
    if let Ok(nals) = nal::iter_length_prefixed(input) {
        for nal_ref in nals {
            if let Some(info) = info_from_nal(&nal_ref) {
                return Some(info);
            }
        }
    }
    // Fall back to the tolerant Annex-B-style scanner for ad-hoc files.
    for nal_ref in nal::iter_annex_b(input) {
        if let Some(info) = info_from_nal(&nal_ref) {
            return Some(info);
        }
    }
    None
}

fn info_from_nal(nal_ref: &nal::NalRef<'_>) -> Option<EvcFileInfo> {
    if nal_ref.header.nal_unit_type != nal::NalUnitType::Sps {
        return None;
    }
    let sps = sps::parse(nal_ref.rbsp()).ok()?;
    Some(EvcFileInfo {
        width: sps.pic_width_in_luma_samples,
        height: sps.pic_height_in_luma_samples,
        profile_idc: sps.profile_idc,
        level_idc: sps.level_idc,
        bit_depth_luma: sps.bit_depth_y(),
        bit_depth_chroma: sps.bit_depth_c(),
        chroma_format_idc: sps.chroma_format_idc,
    })
}

/// Register the EVC implementation (currently parser-only) with a codec
/// registry. The registered decoder factory returns an unsupported-error
/// decoder per the round-1 deliverable.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("evc_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(sps::MAX_DIMENSION, sps::MAX_DIMENSION);
    // ISOBMFF sample-description FourCCs registered for EVC by
    // ISO/IEC 14496-15 (clauses 12 / 13):
    //   `evc1` — track-stored EVC
    //   `evcC` — EVCDecoderConfigurationRecord box code (atom name, kept
    //   for completeness so callers carrying a 4-cc atom can locate us).
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([
                CodecTag::fourcc(b"evc1"),
                CodecTag::fourcc(b"evcC"),
                CodecTag::fourcc(b"EVC1"),
            ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    /// Build a single-NAL length-prefixed EVC bitstream containing a
    /// minimal 320x240 4:2:0 8-bit SPS, suitable for exercising `probe`.
    fn build_minimal_sps_stream() -> Vec<u8> {
        // SPS body
        let mut body = BitEmitter::new();
        body.ue(0); // sps_seq_parameter_set_id
        body.u(8, 1); // profile_idc
        body.u(8, 30); // level_idc
        body.u(32, 0); // toolset_idc_h
        body.u(32, 0); // toolset_idc_l
        body.ue(1); // chroma_format_idc
        body.ue(320); // pic_width_in_luma_samples
        body.ue(240); // pic_height_in_luma_samples
        body.ue(0); // bit_depth_luma_minus8
        body.ue(0); // bit_depth_chroma_minus8
        for _ in 0..13 {
            body.u(1, 0);
        }
        body.ue(1); // log2_sub_gop_length
        body.ue(1); // max_num_tid0_ref_pics
        body.u(1, 0); // picture_cropping_flag
        body.u(1, 0); // chroma_qp_table_present_flag
        body.u(1, 0); // vui_parameters_present_flag
        body.finish_with_trailing_bits();
        let sps_rbsp = body.into_bytes();

        // 2-byte NAL header for SPS (NUT=24)
        let nut_plus1: u16 = 24 + 1;
        let mut hdr_word: u16 = 0;
        hdr_word |= (nut_plus1 & 0x3F) << 9;
        // tid 0, reserved 0, ext 0 — leaves the lower bits zero
        let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];

        // Build the length-prefixed envelope: [u32 len BE][NAL]
        let nal_len = (hdr.len() + sps_rbsp.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&nal_len.to_be_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&sps_rbsp);
        out
    }

    #[test]
    fn probe_minimal_sps_stream() {
        let bs = build_minimal_sps_stream();
        let info = probe(&bs).expect("probe must recover SPS dimensions");
        assert_eq!(info.width, 320);
        assert_eq!(info.height, 240);
        assert_eq!(info.bit_depth_luma, 8);
        assert_eq!(info.bit_depth_chroma, 8);
        assert_eq!(info.chroma_format_idc, 1);
        assert_eq!(info.profile_idc, 1);
        assert_eq!(info.level_idc, 30);
    }

    #[test]
    fn probe_returns_none_on_empty() {
        assert!(probe(&[]).is_none());
    }

    #[test]
    fn probe_returns_none_on_pps_only() {
        // PPS NAL with empty body — should not satisfy probe.
        let nut_plus1: u16 = 25 + 1;
        let mut hdr_word: u16 = 0;
        hdr_word |= (nut_plus1 & 0x3F) << 9;
        let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps id
        pps_body.ue(0); // sps id
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.u(1, 0);
        pps_body.u(1, 1); // single_tile_in_pic_flag
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.finish_with_trailing_bits();
        let body = pps_body.into_bytes();
        let nal_len = (hdr.len() + body.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&nal_len.to_be_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&body);
        assert!(probe(&out).is_none());
    }

    #[test]
    fn register_creates_factory() {
        let mut reg = CodecRegistry::default();
        register(&mut reg);
        // The registry must know the "evc" codec id and have a decoder
        // implementation registered.
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn make_decoder_returns_unsupported_on_packet() {
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), Vec::new());
        let err = dec.send_packet(&pkt).unwrap_err();
        assert!(format!("{err}").contains("EVC pixel decode not yet implemented"));
    }
}
