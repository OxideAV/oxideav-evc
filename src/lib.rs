//! Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1).
//!
//! Round-5 status: a working **Baseline-profile IDR + P + B** decoder
//! with residual coding (RLE + dequant + IDCT for nTbS up to 64),
//! deblocking (§8.8.2 luma path), and a single reference per list. The
//! crate decomposes into:
//!
//! * [`bitreader`] — MSB-first bit reader (§9.2 helpers).
//! * [`nal`] — 2-byte NAL header (§7.3.1.2) + length-prefixed framing.
//! * [`sps`] / [`pps`] / [`aps`] — parameter-set parsers (§7.3.2.x).
//! * [`slice_header`] — `slice_header()` parse (§7.3.4).
//! * [`cabac`] — full CABAC parsing process (§9.3): arithmetic decoding
//!   engine (regular + bypass + terminate) plus the FL / U / TR / EGk
//!   binarization helpers. `sps_cm_init_flag == 0` initialisation only.
//! * [`slice_data`] — `slice_data()` walker plus the round-3 IDR pixel
//!   pipeline ([`slice_data::decode_baseline_idr_slice`]) **and** the
//!   round-4 inter pipeline ([`slice_data::decode_baseline_inter_slice`]).
//! * [`intra`] — intra prediction (§8.4.4) for the Baseline 5-mode set
//!   (DC, HOR, VER, UL, UR; `sps_eipd_flag == 0` path).
//! * [`inter`] — round-4 inter prediction (§8.5): MV resolution + 8-tap
//!   luma + 4-tap chroma sub-pel interpolation (Tables 25 / 27 — Baseline
//!   subset only) + AMVP candidate construction + default-weighted bipred.
//! * [`transform`] — inverse DCT-II for nTbS ∈ {2, 4, 8, 16, 32, 64}
//!   (eq. 1062-1076). The 64-point matrix is built from the closed-form
//!   `M[m][n] = round(64·√2·cos(π·m·(2n+1)/128))` (m≥1, M[0][n]=64),
//!   verified against every printed entry of eq. 1072 / 1074.
//! * [`dequant`] — scaling + transform + final renorm (§8.7.2 / §8.7.3 /
//!   §8.7.4) for the `sps_iqt_flag == 0` Baseline path.
//! * [`picture`] — yuv420p 8-bit picture buffer + per-CU intra
//!   reconstruct glue (§8.7.5).
//! * [`decoder`] — registered decoder factory returning a working
//!   `Decoder` for Baseline IDR + P/B bitstreams (8-bit 4:2:0, no
//!   residuals, single reference).
//!
//! Round-5 deliberate omissions (pending follow-up rounds):
//!
//! * 10-bit support,
//! * advanced deblocking (`sps_addb_flag = 1` — round-5 supports the
//!   `sps_addb_flag = 0` baseline filter only; addb is a Main-profile
//!   feature),
//! * multi-reference + reference list reordering,
//! * Main-profile syntax branches (BTT / SUCO / ADMVP / EIPD / IBC /
//!   ATS / ADCC / ALF / DRA / cm_init / AMVR / MMVD / affine / DMVR / HMVP).
//!
//! All section / clause numbers refer to **ISO/IEC 23094-1:2020(E)** at
//! `docs/video/evc/ISO_IEC_23094-1-EVC-2020.pdf`. No third-party EVC
//! decoder source (MPEG-5 reference, `xeve`, `xevd`, …) was consulted —
//! every module is spec-only.

pub mod aps;
pub mod bitreader;
pub mod cabac;
pub mod deblock;
pub mod decoder;
pub mod dequant;
pub mod inter;
pub mod intra;
pub mod nal;
pub mod picture;
pub mod pps;
pub mod slice_data;
pub mod slice_header;
pub mod sps;
pub mod transform;

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

/// **Round-3** end-to-end decode of a Baseline-profile IDR slice.
///
/// Mirrors [`walk_idr_slice`] but invokes the pixel-emission pipeline:
/// returns a freshly-reconstructed [`picture::YuvPicture`] populated by
/// per-CU intra prediction and the spec's picture-construction step
/// (§8.7.5). Round-3 fixtures must use `cbf_luma == cbf_cb == cbf_cr ==
/// 0` for every CU; non-zero CBFs trigger `Error::Unsupported` (round-4
/// scope).
pub fn decode_idr_slice(
    sps: &sps::Sps,
    pps: &pps::Pps,
    slice_nal_rbsp: &[u8],
) -> oxideav_core::Result<(picture::YuvPicture, slice_data::SliceDecodeStats)> {
    use oxideav_core::Error;
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
    {
        return Err(Error::unsupported(
            "evc decode_idr_slice: round-3 only supports Baseline-profile toolset",
        ));
    }
    if !pps.single_tile_in_pic_flag {
        return Err(Error::unsupported(
            "evc decode_idr_slice: round-3 requires single_tile_in_pic_flag == 1",
        ));
    }
    let mut hdr_br = crate::bitreader::BitReader::new(slice_nal_rbsp);
    let slice_qp = parse_slice_header_for_decode(&mut hdr_br, sps, pps)?;
    hdr_br.align_to_byte();
    let consumed_bits = hdr_br.bit_position();
    if consumed_bits % 8 != 0 {
        return Err(Error::invalid(
            "evc decode_idr_slice: slice header not byte-aligned after parse",
        ));
    }
    let consumed_bytes = (consumed_bits / 8) as usize;
    if consumed_bytes >= slice_nal_rbsp.len() {
        return Err(Error::invalid(
            "evc decode_idr_slice: no slice_data bytes after header",
        ));
    }
    let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
    let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
    let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
    let max_tb_log2_size_y = ctb_log2_size_y.min(5); // round-3: cap at 32x32
    let walk = slice_data::SliceWalkInputs {
        pic_width: sps.pic_width_in_luma_samples,
        pic_height: sps.pic_height_in_luma_samples,
        ctb_log2_size_y,
        min_cb_log2_size_y,
        max_tb_log2_size_y,
        chroma_format_idc: sps.chroma_format_idc,
        cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
    };
    let decode = slice_data::SliceDecodeInputs {
        slice_qp,
        bit_depth_luma: sps.bit_depth_y(),
        bit_depth_chroma: sps.bit_depth_c(),
        enable_deblock: false, // round-3 fixtures keep deblock off
    };
    slice_data::decode_baseline_idr_slice(slice_data_bytes, walk, decode)
}

/// Helper for [`decode_idr_slice`]: parse the Baseline IDR slice header
/// and recover `slice_qp` for downstream dequant. Round-3 supports the
/// minimal set: `slice_pps_id`, `slice_type` (must be 2), trailing
/// flags, `slice_qp ∈ 0..=51`, `slice_cb_qp_offset`, `slice_cr_qp_offset`.
fn parse_slice_header_for_decode(
    br: &mut crate::bitreader::BitReader,
    _sps: &sps::Sps,
    _pps: &pps::Pps,
) -> oxideav_core::Result<i32> {
    use oxideav_core::Error;
    let _slice_pps_id = br.ue()?;
    let slice_type = br.ue()?;
    if slice_type != 2 {
        return Err(Error::invalid(format!(
            "evc decode_idr_slice: IDR slice_type must be 2 (got {slice_type})"
        )));
    }
    let _no_output = br.u1()?;
    let _slice_deblocking_filter_flag = br.u1()?;
    let slice_qp = br.u(6)?;
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc decode_idr_slice: slice_qp {slice_qp} > 51"
        )));
    }
    let _slice_cb_qp_offset = br.se()?;
    let _slice_cr_qp_offset = br.se()?;
    Ok(slice_qp as i32)
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
    fn make_decoder_handles_empty_packet() {
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), Vec::new());
        // Empty packet → no NALs → ok, no frame emitted.
        dec.send_packet(&pkt).unwrap();
        let err = dec.receive_frame().unwrap_err();
        // Iterating with no input must surface NeedMore (not Eof).
        assert!(matches!(err, oxideav_core::Error::NeedMore));
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

    /// Build a minimal Baseline SPS for an `(width × height)` 4:2:0 8-bit
    /// picture. CTB size defaults to 64×64 (`log2_ctu_size_minus5 = 0` →
    /// `CtbLog2SizeY = 5` since `sps_btt_flag = 0` keeps the spec's
    /// default of `1` per §7.4.3.1, putting CTB at 32×32 — round-3
    /// fixtures keep below 64×64 to dodge the unimplemented `nTbS = 64`
    /// transform path).
    fn build_baseline_sps_rbsp(width: u32, height: u32) -> Vec<u8> {
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc Baseline
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc 4:2:0
        sps_body.ue(width);
        sps_body.ue(height);
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
        sps_body.into_bytes()
    }

    /// Build a minimal Baseline PPS for `cu_qp_delta_enabled_flag = 0`.
    fn build_baseline_pps_rbsp() -> Vec<u8> {
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps_id
        pps_body.ue(0); // sps_id
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.u(1, 0); // rpl1_idx_present_flag
        pps_body.u(1, 1); // single_tile_in_pic_flag
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0); // explicit_tile_id_flag
        pps_body.u(1, 0); // pic_dra_enabled_flag
        pps_body.u(1, 0); // arbitrary_slice_present_flag
        pps_body.u(1, 0); // constrained_intra_pred_flag
        pps_body.u(1, 0); // cu_qp_delta_enabled_flag = 0
        pps_body.finish_with_trailing_bits();
        pps_body.into_bytes()
    }

    /// **Round-3 end-to-end pixel decode.** Build a Baseline IDR slice
    /// covering a 32×32 picture (one 32×32 CTU split into four 16×16
    /// leaves; each leaf carries `intra_pred_mode = 0` (= INTRA_DC) and
    /// `cbf_luma = cbf_cb = cbf_cr = 0`). Decode through
    /// [`decode_idr_slice`] and verify the reconstructed Y plane is
    /// uniformly 128 (the bit-depth substitution value for "no
    /// neighbours" is 128, DC-prediction of all-128 references is 128,
    /// and a zero residual leaves it at 128).
    #[test]
    fn round3_end_to_end_decode_grey_idr() {
        use crate::cabac::CabacEncoder;
        // 64×64 picture (1 CTU at the default CTB log2 = 6) — keeps the
        // walker on the simple "CTB == picture" path, avoiding implicit
        // boundary splits.
        let sps = sps::parse(&build_baseline_sps_rbsp(64, 64)).unwrap();
        let pps = pps::parse(&build_baseline_pps_rbsp()).unwrap();
        // Slice header for an IDR with deblocking off, slice_qp = 22.
        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2); // I slice
        hdr.u(1, 0);
        hdr.u(1, 0); // slice_deblocking_filter_flag = 0
        hdr.u(6, 22);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();
        // Encode CABAC: one CTB split (32 → four 16x16 leaves), each
        // leaf intra_pred_mode = 0 (one "0" bin), cbf_luma = cbf_cb =
        // cbf_cr = 0 (no residual_coding fires).
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // CTB split_cu_flag = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split_cu_flag = 0 (leaf)
                                          // dual-tree luma CU: intra_pred_mode + cbf_luma
            enc.encode_decision(0, 0, 0); // intra_pred_mode_idx = 0 (one "0" bin)
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
                                          // dual-tree chroma CU: cbf_cb + cbf_cr (no intra_pred_mode_idx
                                          // for chroma in sps_eipd_flag=0 dual-tree path)
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        slice_rbsp.extend_from_slice(&enc.finish());

        let (pic, stats) = decode_idr_slice(&sps, &pps, &slice_rbsp).unwrap();
        // Picture geometry checks.
        assert_eq!(pic.width, 64);
        assert_eq!(pic.height, 64);
        assert_eq!(pic.y.len(), 64 * 64);
        assert_eq!(pic.cb.len(), 32 * 32);
        assert_eq!(pic.cr.len(), 32 * 32);
        // Stats: 1 CTU, 5 split_cu_flag bins (1 CTB + 4 children), 8
        // coding_units (luma+chroma per leaf), 4 intra_pred_mode bins,
        // 4 cbf_luma bins, 8 cbf_chroma bins.
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.split_cu_flag_bins, 5);
        assert_eq!(stats.coding_units, 8);
        assert_eq!(stats.intra_pred_mode_bins, 4);
        assert_eq!(stats.cbf_luma_bins, 4);
        assert_eq!(stats.cbf_chroma_bins, 8);
        // Pixel content: every Y/Cb/Cr sample should be 128 (the IDR-with-
        // no-neighbours INTRA_DC prediction value at bit-depth 8). Since
        // the picture buffer was pre-filled with 128 and the prediction
        // computes 128 from all-128 references with zero residual, the
        // result is uniform 128.
        let mismatched_y = pic.y.iter().filter(|&&v| v != 128).count();
        assert_eq!(
            mismatched_y, 0,
            "Y plane should be uniform 128 (got {mismatched_y} non-128 samples)"
        );
        let mismatched_cb = pic.cb.iter().filter(|&&v| v != 128).count();
        let mismatched_cr = pic.cr.iter().filter(|&&v| v != 128).count();
        assert_eq!(mismatched_cb, 0);
        assert_eq!(mismatched_cr, 0);
        // PSNR check vs hand-computed reference (uniform 128): PSNR is
        // infinite at MSE = 0 — we only assert MSE = 0.
        let mse: f64 = pic
            .y
            .iter()
            .map(|&v| (v as f64 - 128.0).powi(2))
            .sum::<f64>()
            / pic.y.len() as f64;
        assert_eq!(mse, 0.0);
    }

    /// End-to-end pixel decode through `make_decoder` (the registered
    /// codec factory). Wraps the Baseline IDR slice from above into a
    /// length-prefixed NAL stream containing SPS + PPS + IDR, sends it
    /// to the registered decoder, and pulls a `Frame::Video` out of
    /// `receive_frame`. Verifies the pixel plane shape and content.
    #[test]
    fn round3_make_decoder_decodes_idr_to_grey_frame() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};

        let sps_rbsp = build_baseline_sps_rbsp(64, 64);
        let pps_rbsp = build_baseline_pps_rbsp();

        // Slice header (IDR, slice_qp = 22, deblocking off).
        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2);
        hdr.u(1, 0);
        hdr.u(1, 0);
        hdr.u(6, 22);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut idr_rbsp = hdr.into_bytes();
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // CTB split = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split = 0
            enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&enc.finish());

        // Build NAL headers + length-prefixed envelope.
        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp)); // SPS NUT = 24
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp)); // PPS NUT = 25
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp)); // IDR NUT = 1

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let video = match frame {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };
        assert_eq!(video.planes.len(), 3);
        assert_eq!(video.planes[0].stride, 64);
        assert_eq!(video.planes[0].data.len(), 64 * 64);
        assert!(video.planes[0].data.iter().all(|&v| v == 128));
        assert_eq!(video.planes[1].stride, 32);
        assert_eq!(video.planes[1].data.len(), 32 * 32);
        assert!(video.planes[1].data.iter().all(|&v| v == 128));
        assert_eq!(video.planes[2].data.len(), 32 * 32);
    }

    /// **Round-4 end-to-end fixture through `make_decoder`.** Push an
    /// SPS + PPS + IDR + P-slice sequence into the registered EVC
    /// decoder. The IDR is a uniform grey 32×32 picture; the P-slice is
    /// a single 32×32 leaf with `cu_skip_flag = 1` and `mvp_idx_l0 = 3`
    /// (zero MV). Both frames must come out of `receive_frame`, with the
    /// P frame being a verbatim copy of the IDR (since zero MV + zero
    /// residual ≡ identity).
    #[test]
    fn round4_make_decoder_decodes_idr_plus_p_to_two_frames() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let sps_rbsp = build_baseline_sps_rbsp(32, 32);
        let pps_rbsp = build_baseline_pps_rbsp();

        // IDR: single 32×32 leaf at log2 = 5 (no split).
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0);
        idr_hdr.ue(2); // I slice
        idr_hdr.u(1, 0); // no_output
        idr_hdr.u(1, 0); // slice_deblocking_filter_flag
        idr_hdr.u(6, 22); // slice_qp
        idr_hdr.ue(0);
        idr_hdr.ue(0);
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        let mut idr_enc = CabacEncoder::new();
        // 32x32 single leaf at log2 = 5 (no split needed since
        // log2 == ctb_log2_size_y → no split_cu_flag emitted in walker
        // path? Actually log2 > min so split is emitted).
        // With ctb_log2 = 5 and min_cb_log2 = 4, the CTU is 32×32, and
        // we want a single 32x32 leaf → split_cu_flag = 0.
        idr_enc.encode_decision(0, 0, 0); // split_cu_flag = 0 at CTB
                                          // dual-tree luma CU (32x32):
        idr_enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
        idr_enc.encode_decision(0, 0, 0); // cbf_luma
                                          // dual-tree chroma CU:
        idr_enc.encode_decision(0, 0, 0); // cbf_cb
        idr_enc.encode_decision(0, 0, 0); // cbf_cr
        idr_enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&idr_enc.finish());

        // P slice header: slice_type = 1 (P), no override, deblock off.
        let mut p_hdr = BitEmitter::new();
        p_hdr.ue(0); // slice_pps_id
        p_hdr.ue(1); // slice_type = P
                     // Not IDR + sps_pocs_flag = 0 → no POC LSB.
                     // Not IDR + slice_type=P → ref-idx + admvp branch.
        p_hdr.u(1, 0); // num_ref_idx_active_override_flag = 0
                       // sps_admvp_flag = 0 → skip temporal_mvp.
        p_hdr.u(1, 0); // slice_deblocking_filter_flag
        p_hdr.u(6, 22); // slice_qp
        p_hdr.ue(0);
        p_hdr.ue(0);
        while p_hdr.bit_position() % 8 != 0 {
            p_hdr.u(1, 0);
        }
        let mut p_rbsp = p_hdr.into_bytes();
        let mut p_enc = CabacEncoder::new();
        p_enc.encode_decision(0, 0, 0); // split_cu_flag = 0 at CTB
                                        // Single-tree inter CU:
        p_enc.encode_decision(0, 0, 1); // cu_skip_flag = 1
        for _ in 0..3 {
            p_enc.encode_decision(0, 0, 1); // mvp_idx_l0 = 3 prefix (3 ones)
        }
        p_enc.encode_decision(0, 0, 0); // cbf_luma
        p_enc.encode_decision(0, 0, 0); // cbf_cb
        p_enc.encode_decision(0, 0, 0); // cbf_cr
        p_enc.encode_terminate(true);
        p_rbsp.extend_from_slice(&p_enc.finish());

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp)); // SPS
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp)); // PPS
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp)); // IDR
        bs.extend_from_slice(&nal_envelope(0, &p_rbsp)); // NonIDR

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        // Pull the two frames.
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        let v0 = match f0 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v1 = match f1 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        // Both should be uniform 128 (the IDR is grey, the P slice is a
        // zero-MV, zero-residual copy of the IDR → still grey).
        assert!(v0.planes[0].data.iter().all(|&v| v == 128));
        assert!(v1.planes[0].data.iter().all(|&v| v == 128));
        assert_eq!(
            v0.planes[0].data, v1.planes[0].data,
            "P frame must equal IDR"
        );
        assert_eq!(v0.planes[1].data, v1.planes[1].data);
        assert_eq!(v0.planes[2].data, v1.planes[2].data);
        // PSNR Y = ∞ (MSE=0). The acceptance bar is ≥ 30 dB.
        let mse: f64 = v0.planes[0]
            .data
            .iter()
            .zip(v1.planes[0].data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / v0.planes[0].data.len() as f64;
        assert_eq!(mse, 0.0, "PSNR must be infinite for identical frames");
    }
}
