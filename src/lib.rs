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
//! * [`cabac`] — CABAC parsing process (§9.3): arithmetic decoding engine
//!   (regular + bypass + terminate) plus binarization helpers (FL, U, TR,
//!   EGk). Round-2 ships with the `sps_cm_init_flag == 0` Baseline-profile
//!   context table (every variable initialised to `(256, 0)`); the
//!   `sps_cm_init_flag == 1` per-table init lands in round 3 alongside
//!   pixel emission.
//! * [`slice_data`] — `slice_data()` walker (§7.3.8) for Baseline-profile
//!   IDR slices. Drives the CABAC engine through every `ae(v)` syntax
//!   element so an end-to-end IDR slice's bins are consumed cleanly via
//!   the `end_of_tile_one_bit` terminate path. Pixel emission is round 3.
//! * [`decoder`] — registry factory; the registered decoder returns
//!   `Error::Unsupported("EVC pixel decode not yet implemented")` since
//!   the per-CU pixel pipeline lands in round 3.
//!
//! All section / clause numbers refer to **ISO/IEC 23094-1:2020(E)** at
//! `docs/video/evc/ISO_IEC_23094-1-EVC-2020.pdf`. No third-party EVC
//! decoder source (MPEG-5 reference, `xeve`, `xevd`, …) was consulted —
//! every parser is spec-only.

pub mod aps;
pub mod bitreader;
pub mod cabac;
pub mod decoder;
pub mod nal;
pub mod pps;
pub mod slice_data;
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

/// Walk an IDR slice's `slice_data()` end-to-end given the active SPS and
/// PPS. The slice's RBSP is split into the slice header + the
/// (byte-aligned) slice-data payload; this helper invokes the slice
/// header parser then drives [`slice_data::walk_baseline_idr_slice`]
/// across the rest of the RBSP. Returns the [`slice_data::SliceWalkStats`]
/// once the engine terminates cleanly.
///
/// **Round-2 scope**: Baseline-profile IDR slices only. Errors out on any
/// SPS toolset combination not yet supported by the walker (see
/// [`slice_data::walk_baseline_idr_slice`] for the constraint set).
pub fn walk_idr_slice(
    sps: &sps::Sps,
    pps: &pps::Pps,
    slice_nal_rbsp: &[u8],
) -> oxideav_core::Result<slice_data::SliceWalkStats> {
    use oxideav_core::Error;
    // Build the slice-parse context from SPS + PPS.
    let ctx = slice_header::SliceParseContext {
        single_tile_in_pic_flag: pps.single_tile_in_pic_flag,
        arbitrary_slice_present_flag: pps.arbitrary_slice_present_flag,
        tile_id_len_minus1: pps.tile_id_len_minus1,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        sps_pocs_flag: sps.sps_pocs_flag,
        sps_rpl_flag: sps.sps_rpl_flag,
        sps_alf_flag: sps.sps_alf_flag,
        sps_mmvd_flag: sps.sps_mmvd_flag,
        sps_admvp_flag: sps.sps_admvp_flag,
        sps_addb_flag: sps.sps_addb_flag,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        chroma_array_type: sps.chroma_array_type(),
    };
    // Round-2 walker requires the Baseline profile constraint set (Annex
    // A.3.2). Refuse anything else cleanly.
    if sps.sps_btt_flag
        || sps.sps_suco_flag
        || sps.sps_admvp_flag
        || sps.sps_eipd_flag
        || sps.sps_alf_flag
        || sps.sps_addb_flag
        || sps.sps_dquant_flag
        || sps.sps_ats_flag
        || sps.sps_ibc_flag
        || sps.sps_dra_flag
        || sps.sps_adcc_flag
        || sps.sps_cm_init_flag
        || sps.sps_amvr_flag
        || sps.sps_mmvd_flag
        || sps.sps_affine_flag
        || sps.sps_dmvr_flag
        || sps.sps_hmvp_flag
    {
        return Err(Error::unsupported(
            "evc walk_idr_slice: round-2 only supports Baseline-profile toolset",
        ));
    }
    if !pps.single_tile_in_pic_flag {
        return Err(Error::unsupported(
            "evc walk_idr_slice: round-2 requires single_tile_in_pic_flag == 1",
        ));
    }
    // Parse the slice header to find where slice_data() begins. We use a
    // bit-counting BitReader to determine how many bits the header
    // consumed.
    let mut hdr_br = crate::bitreader::BitReader::new(slice_nal_rbsp);
    parse_slice_header_consume(&mut hdr_br, nal::NalUnitType::Idr, &ctx)?;
    // Align to the next byte boundary (slice_data() starts byte-aligned).
    hdr_br.align_to_byte();
    let consumed_bits = hdr_br.bit_position();
    if consumed_bits % 8 != 0 {
        return Err(Error::invalid(
            "evc walk_idr_slice: slice header not byte-aligned after parse",
        ));
    }
    let consumed_bytes = (consumed_bits / 8) as usize;
    if consumed_bytes >= slice_nal_rbsp.len() {
        return Err(Error::invalid(
            "evc walk_idr_slice: no slice_data bytes after header",
        ));
    }
    let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
    // Build SliceWalkInputs from SPS / PPS.
    let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
    let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
    let max_tb_log2_size_y = ctb_log2_size_y.min(6);
    let inputs = slice_data::SliceWalkInputs {
        pic_width: sps.pic_width_in_luma_samples,
        pic_height: sps.pic_height_in_luma_samples,
        ctb_log2_size_y,
        min_cb_log2_size_y,
        max_tb_log2_size_y,
        chroma_format_idc: sps.chroma_format_idc,
        cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
    };
    slice_data::walk_baseline_idr_slice(slice_data_bytes, inputs)
}

/// Internal helper that re-runs the slice header parse on a *borrowed*
/// BitReader so the caller can recover the header bit position. Mirrors
/// [`slice_header::parse`] but takes a mutable reference. Round-3 may
/// fold this into the public `slice_header::parse` once the active-PS
/// tracker lands.
fn parse_slice_header_consume(
    br: &mut crate::bitreader::BitReader,
    _nal_unit_type: nal::NalUnitType,
    _ctx: &slice_header::SliceParseContext,
) -> oxideav_core::Result<()> {
    use oxideav_core::Error;
    let _slice_pps_id = br.ue()?;
    // Baseline + IDR + single_tile_in_pic_flag = no tile fields.
    let slice_type = br.ue()?;
    if slice_type != 2 {
        return Err(Error::invalid(format!(
            "evc walk_idr_slice: IDR slice_type must be 2 (got {slice_type})"
        )));
    }
    let _no_output = br.u1()?;
    // sps_mmvd_flag, sps_alf_flag = 0 in Baseline → skip.
    // IDR + sps_pocs_flag is irrelevant (POC LSB not in IDR header).
    // sps_rpl_flag = 0 → no RPL fields.
    // not P/B → no ref_idx fields.
    let _slice_deblocking_filter_flag = br.u1()?;
    // sps_addb_flag = 0 → no alpha/beta offsets.
    let slice_qp = br.u(6)?;
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc walk_idr_slice: slice_qp {slice_qp} > 51"
        )));
    }
    let _slice_cb_qp_offset = br.se()?;
    let _slice_cr_qp_offset = br.se()?;
    Ok(())
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

    /// End-to-end Baseline-IDR walk: build an SPS + PPS in-memory,
    /// hand-craft a slice header for an IDR slice, append a CABAC-encoded
    /// `slice_data()` payload, then run [`walk_idr_slice`] across it
    /// and verify every bin is consumed cleanly through the
    /// `end_of_tile_one_bit` terminate decision.
    ///
    /// This is the round-2 deliverable's "real (small) EVC IDR slice has
    /// its bins consumed cleanly" milestone. The slice covers a 64×64
    /// picture (a single 64×64 CTU explicitly split via `split_cu_flag = 1`
    /// into four 32×32 leaves; each leaf then declines to split via
    /// `split_cu_flag = 0` since the walker permits more recursion under
    /// the Baseline default `MinCbLog2SizeY = 2`). Each leaf goes through
    /// the dual-tree luma + chroma `coding_unit()` pair with no CBFs set.
    ///
    /// Bin sequence (21 regular bins + terminate):
    /// * 1 × `split_cu_flag = 1` (CTB)
    /// * 4 × `split_cu_flag = 0` (each 32×32 child)
    /// * 4 × (intra_pred_mode + cbf_luma + cbf_cb + cbf_cr) = 16
    #[test]
    fn end_to_end_walk_baseline_idr_slice() {
        use crate::cabac::CabacEncoder;
        // Build a Baseline-profile SPS for 64×64 4:2:0 8-bit. Default
        // toolset (sps_btt = 0) leaves the parser at CtbLog2SizeY = 6
        // (64×64 CTU) and MinCbLog2SizeY = 2 (4×4 minimum CB).
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc = Baseline (0)
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc = 1 (4:2:0)
        sps_body.ue(64); // pic_width
        sps_body.ue(64); // pic_height
        sps_body.ue(0); // bit_depth_luma_minus8
        sps_body.ue(0); // bit_depth_chroma_minus8
        for _ in 0..13 {
            sps_body.u(1, 0);
        }
        sps_body.ue(1); // log2_sub_gop_length
        sps_body.ue(1); // max_num_tid0_ref_pics
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        let sps_rbsp = sps_body.into_bytes();
        let sps = sps::parse(&sps_rbsp).expect("SPS parses");

        // Baseline PPS: cu_qp_delta_enabled = 0 so transform_unit doesn't
        // emit cu_qp_delta_abs.
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps_id
        pps_body.ue(0); // sps_id
        pps_body.ue(0); // num_ref_idx_default_active_minus1[0]
        pps_body.ue(0); // num_ref_idx_default_active_minus1[1]
        pps_body.ue(0); // additional_lt_poc_lsb_len
        pps_body.u(1, 0); // rpl1_idx_present_flag
        pps_body.u(1, 1); // single_tile_in_pic_flag
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0); // explicit_tile_id_flag
        pps_body.u(1, 0); // pic_dra_enabled_flag
        pps_body.u(1, 0); // arbitrary_slice_present_flag
        pps_body.u(1, 0); // constrained_intra_pred_flag
        pps_body.u(1, 0); // cu_qp_delta_enabled_flag = 0
        pps_body.finish_with_trailing_bits();
        let pps_rbsp = pps_body.into_bytes();
        let pps = pps::parse(&pps_rbsp).expect("PPS parses");

        // Slice header (Baseline IDR with single_tile_in_pic_flag = 1).
        let mut hdr = BitEmitter::new();
        hdr.ue(0); // slice_pps_id
        hdr.ue(2); // slice_type = I
        hdr.u(1, 0); // no_output_of_prior_pics_flag
        hdr.u(1, 0); // slice_deblocking_filter_flag
        hdr.u(6, 22); // slice_qp = 22
        hdr.ue(0); // slice_cb_qp_offset (se: 0)
        hdr.ue(0); // slice_cr_qp_offset
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();

        // CABAC-encoded slice_data:
        //   1× split_cu_flag = 1 at the CTB
        //   4× split_cu_flag = 0 at each 32×32 child
        //   4× (intra_pred_mode + cbf_luma + cbf_cb + cbf_cr)
        //   then terminate(true).
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // CTB split = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split = 0
            enc.encode_decision(0, 0, 0); // intra_pred_mode
            enc.encode_decision(0, 0, 0); // cbf_luma
            enc.encode_decision(0, 0, 0); // cbf_cb
            enc.encode_decision(0, 0, 0); // cbf_cr
        }
        enc.encode_terminate(true);
        let slice_data_bytes = enc.finish();
        slice_rbsp.extend_from_slice(&slice_data_bytes);

        let stats = walk_idr_slice(&sps, &pps, &slice_rbsp).expect("walk succeeds");
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.split_cu_flag_bins, 5, "1 CTB split + 4 child splits");
        assert_eq!(stats.coding_units, 8, "4 leaves × (luma + chroma)");
        assert_eq!(stats.intra_pred_mode_bins, 4);
        assert_eq!(stats.cbf_luma_bins, 4);
        assert_eq!(stats.cbf_chroma_bins, 8);
    }

    /// Confirm the integration helper rejects a non-Baseline SPS up front
    /// rather than handing the walker a bitstream it cannot parse.
    #[test]
    fn walk_idr_slice_rejects_main_profile_sps() {
        // SPS with sps_btt_flag = 1 — outside Baseline.
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0);
        sps_body.u(8, 1); // profile_idc = 1 (Main)
        sps_body.u(8, 41);
        sps_body.u(32, 1); // toolset_idc_h non-zero
        sps_body.u(32, 1);
        sps_body.ue(1);
        sps_body.ue(1920);
        sps_body.ue(1080);
        sps_body.ue(2);
        sps_body.ue(2);
        sps_body.u(1, 1); // sps_btt_flag = 1
        sps_body.ue(1);
        sps_body.ue(0);
        sps_body.ue(2);
        sps_body.ue(1);
        sps_body.ue(0);
        sps_body.u(1, 0); // sps_suco
        sps_body.u(1, 0); // sps_admvp
        sps_body.u(1, 0); // sps_eipd
        sps_body.u(1, 0); // sps_cm_init
        sps_body.u(1, 0); // sps_iqt
        sps_body.u(1, 0); // sps_addb
        sps_body.u(1, 0); // sps_alf
        sps_body.u(1, 0); // sps_htdf
        sps_body.u(1, 0); // sps_rpl
        sps_body.u(1, 0); // sps_pocs
        sps_body.u(1, 0); // sps_dquant
        sps_body.u(1, 0); // sps_dra
        sps_body.ue(1);
        sps_body.ue(1);
        sps_body.u(1, 0);
        sps_body.u(1, 0);
        sps_body.u(1, 0);
        sps_body.finish_with_trailing_bits();
        let sps_rbsp = sps_body.into_bytes();
        let sps = sps::parse(&sps_rbsp).expect("SPS parses");

        // Minimal PPS.
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.u(1, 0);
        pps_body.u(1, 1);
        pps_body.ue(0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.finish_with_trailing_bits();
        let pps_rbsp = pps_body.into_bytes();
        let pps = pps::parse(&pps_rbsp).unwrap();

        let res = walk_idr_slice(&sps, &pps, &[0u8; 16]);
        assert!(res.is_err());
        let err_text = format!("{}", res.unwrap_err());
        assert!(
            err_text.contains("Baseline-profile toolset"),
            "got: {err_text}"
        );
    }
}
