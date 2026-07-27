//! EVC parameter-set / slice-header **writers** (ISO/IEC 23094-1 §7.3)
//! for the Baseline-profile intra encoder — the field-for-field duals of
//! the [`crate::sps`] / [`crate::pps`] / [`crate::slice_header`] parsers.
//!
//! Scope (round 429 encoder bootstrap): the Baseline all-intra toolset —
//! every SPS tool flag zero (`sps_btt_flag == 0` infers the §7.4.3.1
//! defaults `CtbLog2SizeY = 6`, `MinCbLog2SizeY = 2`), 4:2:0, one tile,
//! IDR slices only. Every writer is pinned by a parse-back fixture
//! against the crate's own parser, which is the same code path the
//! registered decoder runs.

use oxideav_core::{Error, Result};

use crate::bitwriter::BitWriter;
use crate::nal::NalUnitType;

/// The encoder-side sequence configuration. Mirrors the subset of
/// [`crate::sps::Sps`] fields the Baseline intra encoder emits.
#[derive(Clone, Copy, Debug)]
pub struct EncSequenceConfig {
    pub width: u32,
    pub height: u32,
    /// `level_idc` to declare (30 = level 3.0 covers the CIF-class
    /// fixture sizes; callers with bigger frames should raise it).
    pub level_idc: u8,
    /// `BitDepthY == BitDepthC` (§7.4.3.1) — 8..=16; the writer emits
    /// `bit_depth_luma_minus8` / `bit_depth_chroma_minus8` from it.
    pub bit_depth: u32,
    /// `sps_cm_init_flag` (§7.4.3.1) — the §9.3.2.2 context-model
    /// initialization the round-431 encoder drives for its rate win.
    /// Annex A.3.2 requires `sps_cm_init_flag == 0` in the Baseline
    /// profile, so setting this switches the SPS to `profile_idc = 1`
    /// (Main) with `toolset_idc_h` carrying exactly the Table A.6
    /// binIdx-14 cm_init bit (`0x4000`; every other tool stays barred
    /// by the zero upper bound, and `toolset_idc_l = 0` keeps the
    /// lower bound permissive).
    pub cm_init: bool,
}

/// Write the §7.3.2.1 SPS RBSP for the intra encoder configuration:
/// with `cm_init == false`, `profile_idc = 0` (Baseline) and every tool
/// flag 0; with `cm_init == true`, `profile_idc = 1` (Main, per the
/// Annex A.3.2 `sps_cm_init_flag == 0 only` Baseline constraint) with
/// `sps_cm_init_flag = 1` as the single enabled tool (Table A.6
/// `toolset_idc_h = 0x4000`) and the §7.3.2.1 conditional
/// `sps_adcc_flag` written 0 (the crate's §7.3.8.7 RLE residual layer).
///
/// With `sps_btt_flag == 0` the parser applies the §7.4.3.1 defaults
/// `log2_ctu_size_minus5 = 1` (64×64 CTU) and `log2_min_cb_size_minus2
/// = 0` (4×4 minimum CB) — exactly the geometry `slice_enc` walks.
pub fn write_sps_rbsp(cfg: &EncSequenceConfig) -> Result<Vec<u8>> {
    if cfg.width == 0 || cfg.height == 0 {
        return Err(Error::invalid("evc enc sps: zero dimensions"));
    }
    if cfg.width > crate::sps::MAX_DIMENSION || cfg.height > crate::sps::MAX_DIMENSION {
        return Err(Error::invalid("evc enc sps: dimensions exceed bound"));
    }
    if !(8..=16).contains(&cfg.bit_depth) {
        return Err(Error::invalid(format!(
            "evc enc sps: bit depth {} outside 8..=16",
            cfg.bit_depth
        )));
    }
    let mut w = BitWriter::new();
    w.ue(0); // sps_seq_parameter_set_id
    w.u(8, u32::from(cfg.cm_init)); // profile_idc: 0 Baseline / 1 Main (A.3.2/A.3.3)
    w.u(8, cfg.level_idc as u32); // level_idc
    w.u(32, if cfg.cm_init { 0x4000 } else { 0 }); // toolset_idc_h (Table A.6 binIdx 14)
    w.u(32, 0); // toolset_idc_l
    w.ue(1); // chroma_format_idc = 1 (4:2:0)
    w.ue(cfg.width); // pic_width_in_luma_samples
    w.ue(cfg.height); // pic_height_in_luma_samples
    w.ue(cfg.bit_depth - 8); // bit_depth_luma_minus8
    w.ue(cfg.bit_depth - 8); // bit_depth_chroma_minus8 (decoder requires ==)
    w.u1(false); // sps_btt_flag → CtbLog2SizeY=6, MinCbLog2SizeY=2 defaults
    w.u1(false); // sps_suco_flag
    w.u1(false); // sps_admvp_flag
    w.u1(false); // sps_eipd_flag
    w.u1(cfg.cm_init); // sps_cm_init_flag
    if cfg.cm_init {
        w.u1(false); // sps_adcc_flag (§7.3.2.1: present when cm_init)
    }
    w.u1(false); // sps_iqt_flag
    w.u1(false); // sps_addb_flag
    w.u1(false); // sps_alf_flag
    w.u1(false); // sps_htdf_flag
    w.u1(false); // sps_rpl_flag
    w.u1(false); // sps_pocs_flag
    w.u1(false); // sps_dquant_flag
    w.u1(false); // sps_dra_flag
                 // !sps_pocs_flag || !sps_rpl_flag → log2_sub_gop_length; == 0 →
                 // log2_ref_pic_gap_length (all-intra stream: sub-GOP length 1).
    w.ue(0); // log2_sub_gop_length = 0
    w.ue(0); // log2_ref_pic_gap_length
             // !sps_rpl_flag → max_num_tid0_ref_pics
    w.ue(1);
    w.u1(false); // picture_cropping_flag
    w.u1(false); // chroma_qp_table_present_flag (chroma_format_idc != 0)
    w.u1(false); // vui_parameters_present_flag
    w.rbsp_trailing_bits();
    Ok(w.into_bytes())
}

/// Write the §7.3.2.2 PPS RBSP: single tile, no DRA, no per-CU QP delta.
pub fn write_pps_rbsp() -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    w.ue(0); // pps_pic_parameter_set_id
    w.ue(0); // pps_seq_parameter_set_id
    w.ue(0); // num_ref_idx_default_active_minus1[0]
    w.ue(0); // num_ref_idx_default_active_minus1[1]
    w.ue(0); // additional_lt_poc_lsb_len
    w.u1(false); // rpl1_idx_present_flag
    w.u1(true); // single_tile_in_pic_flag
    w.ue(0); // tile_id_len_minus1
    w.u1(false); // explicit_tile_id_flag
    w.u1(false); // pic_dra_enabled_flag
    w.u1(false); // arbitrary_slice_present_flag
    w.u1(false); // constrained_intra_pred_flag
    w.u1(false); // cu_qp_delta_enabled_flag
    w.rbsp_trailing_bits();
    Ok(w.into_bytes())
}

/// Write the §7.3.4 slice header for a Baseline IDR slice (single tile,
/// no ALF/ADDB/RPL fields under the all-zero SPS toolset), then pad to
/// the byte boundary — `slice_data()` starts byte-aligned (§7.4.5).
/// Returns the header bytes; the caller appends the CABAC payload.
/// `deblock` sets `slice_deblocking_filter_flag` (with the Baseline
/// `sps_addb_flag == 0` no alpha/beta offsets follow).
pub fn write_idr_slice_header(slice_qp: u32, deblock: bool) -> Result<Vec<u8>> {
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc enc slice header: slice_qp {slice_qp} > 51"
        )));
    }
    let mut w = BitWriter::new();
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I (§7.4.5: IDR must be 2)
    w.u1(false); // no_output_of_prior_pics_flag
    w.u1(deblock); // slice_deblocking_filter_flag
    w.u(6, slice_qp); // slice_qp
    w.se(0); // slice_cb_qp_offset
    w.se(0); // slice_cr_qp_offset
    w.align_to_byte_zero(); // byte_alignment() before slice_data()
    Ok(w.into_bytes())
}

/// Wrap an RBSP in the 2-byte §7.3.1.2 NAL header plus the Annex B
/// 4-byte big-endian `nal_unit_length` prefix, appending to `out`.
pub fn append_length_prefixed_nal(out: &mut Vec<u8>, nut: NalUnitType, rbsp: &[u8]) {
    let nut_plus1 = (nut.as_u8() as u16) + 1;
    // forbidden_zero_bit(1) | nal_unit_type_plus1(6) | nuh_temporal_id(3)
    // | nuh_reserved_zero_5bits(5) | nuh_extension_flag(1), MSB-first.
    let hdr_word: u16 = (nut_plus1 & 0x3F) << 9;
    let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
    let len = (2 + rbsp.len()) as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(rbsp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal;

    /// SPS writer → SPS parser: every consumed field lands where the
    /// encoder put it, and the §7.4.3.1 sps_btt_flag == 0 defaults give
    /// the 64×64 CTU / 4×4 min-CB geometry.
    #[test]
    fn sps_parse_back() {
        let cfg = EncSequenceConfig {
            width: 176,
            height: 144,
            level_idc: 30,
            bit_depth: 8,
            cm_init: false,
        };
        let rbsp = write_sps_rbsp(&cfg).unwrap();
        let sps = crate::sps::parse(&rbsp).expect("own SPS must parse");
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.profile_idc, 0, "Baseline profile_idc");
        assert_eq!(sps.level_idc, 30);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 176);
        assert_eq!(sps.pic_height_in_luma_samples, 144);
        assert_eq!(sps.bit_depth_y(), 8);
        assert_eq!(sps.bit_depth_c(), 8);
        assert!(!sps.sps_btt_flag);
        assert_eq!(sps.log2_ctu_size_minus5, 1, "§7.4.3.1 default CTU 64");
        assert_eq!(sps.log2_min_cb_size_minus2, 0, "§7.4.3.1 default min CB 4");
        for (name, flag) in [
            ("suco", sps.sps_suco_flag),
            ("admvp", sps.sps_admvp_flag),
            ("eipd", sps.sps_eipd_flag),
            ("cm_init", sps.sps_cm_init_flag),
            ("iqt", sps.sps_iqt_flag),
            ("ats", sps.sps_ats_flag),
            ("addb", sps.sps_addb_flag),
            ("alf", sps.sps_alf_flag),
            ("htdf", sps.sps_htdf_flag),
            ("rpl", sps.sps_rpl_flag),
            ("pocs", sps.sps_pocs_flag),
            ("dquant", sps.sps_dquant_flag),
            ("dra", sps.sps_dra_flag),
            ("ibc", sps.sps_ibc_flag),
        ] {
            assert!(!flag, "sps_{name}_flag must be 0 in the Baseline SPS");
        }
        assert!(!sps.picture_cropping_flag);
        assert!(!sps.vui_parameters_present_flag);
    }

    /// 10-bit SPS parse-back: the bit-depth fields land where the
    /// encoder put them; out-of-range depths are refused.
    #[test]
    fn sps_parse_back_10bit() {
        let cfg = EncSequenceConfig {
            width: 64,
            height: 64,
            level_idc: 30,
            bit_depth: 10,
            cm_init: false,
        };
        let sps = crate::sps::parse(&write_sps_rbsp(&cfg).unwrap()).unwrap();
        assert_eq!(sps.bit_depth_y(), 10);
        assert_eq!(sps.bit_depth_c(), 10);
        let bad = EncSequenceConfig {
            bit_depth: 17,
            ..cfg
        };
        assert!(write_sps_rbsp(&bad).is_err());
    }

    /// PPS writer → PPS parser.
    #[test]
    fn pps_parse_back() {
        let rbsp = write_pps_rbsp().unwrap();
        let pps = crate::pps::parse(&rbsp).expect("own PPS must parse");
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert!(pps.single_tile_in_pic_flag);
        assert!(!pps.cu_qp_delta_enabled_flag);
        assert!(!pps.pic_dra_enabled_flag);
        assert!(!pps.arbitrary_slice_present_flag);
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.rpl1_idx_present_flag);
    }

    /// Slice-header writer → the full §7.3.4 parser (the registered
    /// decoder's path), driven with the exact SPS/PPS context the other
    /// two writers produce. The header must land byte-aligned.
    #[test]
    fn idr_slice_header_parse_back() {
        let cfg = EncSequenceConfig {
            width: 64,
            height: 64,
            level_idc: 30,
            bit_depth: 8,
            cm_init: false,
        };
        let sps = crate::sps::parse(&write_sps_rbsp(&cfg).unwrap()).unwrap();
        let pps = crate::pps::parse(&write_pps_rbsp().unwrap()).unwrap();
        let ctx = crate::slice_header::SliceParseContext {
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
            num_ref_pic_lists_in_sps_l0: sps.num_ref_pic_lists_in_sps_l0,
            num_ref_pic_lists_in_sps_l1: sps.num_ref_pic_lists_in_sps_l1,
            rpl1_idx_present_flag: pps.rpl1_idx_present_flag,
            long_term_ref_pics_flag: sps.long_term_ref_pics_flag,
            additional_lt_poc_lsb_len: pps.additional_lt_poc_lsb_len,
        };
        let hdr_bytes = write_idr_slice_header(27, false).unwrap();
        let mut br = crate::bitreader::BitReader::new(&hdr_bytes);
        let hdr = crate::slice_header::parse_consume(&mut br, nal::NalUnitType::Idr, &ctx).unwrap();
        assert_eq!(hdr.slice_pic_parameter_set_id, 0);
        assert_eq!(hdr.slice_qp, 27);
        assert_eq!(hdr.slice_cb_qp_offset, 0);
        assert_eq!(hdr.slice_cr_qp_offset, 0);
        assert!(!hdr.slice_deblocking_filter_flag);
        br.align_to_byte();
        assert_eq!(
            br.bit_position() as usize,
            hdr_bytes.len() * 8,
            "header must occupy the whole padded byte span"
        );
    }

    /// Out-of-range slice QP is refused.
    #[test]
    fn slice_header_rejects_bad_qp() {
        assert!(write_idr_slice_header(52, false).is_err());
        // The deblock flag round-trips (its parse is covered by the
        // registered-decoder e2e; here just shape-check the bit).
        let with = write_idr_slice_header(20, true).unwrap();
        let without = write_idr_slice_header(20, false).unwrap();
        assert_ne!(with, without);
    }

    /// NAL wrapping matches the crate's own length-prefixed iterator and
    /// `probe` recovers the SPS summary.
    #[test]
    fn nal_wrap_and_probe() {
        let cfg = EncSequenceConfig {
            width: 320,
            height: 240,
            level_idc: 30,
            bit_depth: 8,
            cm_init: false,
        };
        let mut bs = Vec::new();
        append_length_prefixed_nal(
            &mut bs,
            nal::NalUnitType::Sps,
            &write_sps_rbsp(&cfg).unwrap(),
        );
        append_length_prefixed_nal(&mut bs, nal::NalUnitType::Pps, &write_pps_rbsp().unwrap());
        let nals = nal::iter_length_prefixed(&bs).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].header.nal_unit_type, nal::NalUnitType::Sps);
        assert_eq!(nals[1].header.nal_unit_type, nal::NalUnitType::Pps);
        assert_eq!(nals[0].header.nuh_temporal_id, 0);
        let info = crate::probe(&bs).expect("probe must find the SPS");
        assert_eq!(info.width, 320);
        assert_eq!(info.height, 240);
        assert_eq!(info.profile_idc, 0);
        assert_eq!(info.chroma_format_idc, 1);
    }
}
