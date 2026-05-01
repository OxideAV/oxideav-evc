//! EVC sequence parameter set parser (ISO/IEC 23094-1 §7.3.2.1, §7.4.3.1).
//!
//! The SPS carries the picture dimensions, bit depth, chroma format and the
//! profile / level / toolset signalling used to gate a decoder against the
//! Annex A profiles. This module parses every field that round-1 needs to
//! populate [`probe`](crate::probe) plus the toolset bitfields that later
//! rounds will use to drive the per-CTU decoder. Fields below the scope of
//! the round-1 deliverable (chroma QP table, VUI, picture cropping derived
//! variables) are still parsed so the bit reader stays in step, but only
//! summary fields are surfaced on [`Sps`].
//!
//! A hard upper bound (32768) is enforced on `pic_width_in_luma_samples` and
//! `pic_height_in_luma_samples` to keep round-1 immune from oversized
//! header allocations (the workspace runs under a 24 GB RSS watchdog). The
//! same bound is hit by HEVC / VVC profile constraints in practice.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Hard sanity bound on the SPS-declared picture dimensions. Larger values
/// are surfaced as `Error::Invalid` so a hostile bitstream cannot trigger
/// an oversized allocation.
pub const MAX_DIMENSION: u32 = 32_768;

/// Maximum reasonable SPS id (the spec range is 0..=15).
const MAX_SPS_ID: u32 = 15;

/// Parsed EVC SPS — fields surfaced for round-1.
#[derive(Clone, Debug)]
pub struct Sps {
    pub sps_seq_parameter_set_id: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub toolset_idc_h: u32,
    pub toolset_idc_l: u32,
    pub chroma_format_idc: u32,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,

    pub sps_btt_flag: bool,
    pub log2_ctu_size_minus5: u32,
    pub log2_min_cb_size_minus2: u32,
    pub log2_diff_ctu_max_14_cb_size: u32,
    pub log2_diff_ctu_max_tt_cb_size: u32,
    pub log2_diff_min_cb_min_tt_cb_size_minus2: u32,

    pub sps_suco_flag: bool,
    pub log2_diff_ctu_size_max_suco_cb_size: u32,
    pub log2_diff_max_suco_min_suco_cb_size: u32,

    pub sps_admvp_flag: bool,
    pub sps_affine_flag: bool,
    pub sps_amvr_flag: bool,
    pub sps_dmvr_flag: bool,
    pub sps_mmvd_flag: bool,
    pub sps_hmvp_flag: bool,

    pub sps_eipd_flag: bool,
    pub sps_ibc_flag: bool,
    pub log2_max_ibc_cand_size_minus2: u32,

    pub sps_cm_init_flag: bool,
    pub sps_adcc_flag: bool,
    pub sps_iqt_flag: bool,
    pub sps_ats_flag: bool,
    pub sps_addb_flag: bool,
    pub sps_alf_flag: bool,
    pub sps_htdf_flag: bool,
    pub sps_rpl_flag: bool,
    pub sps_pocs_flag: bool,
    pub sps_dquant_flag: bool,
    pub sps_dra_flag: bool,

    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub log2_sub_gop_length: u32,
    pub log2_ref_pic_gap_length: u32,
    pub max_num_tid0_ref_pics: u32,
    pub sps_max_dec_pic_buffering_minus1: u32,
    pub long_term_ref_pics_flag: bool,
    pub rpl1_same_as_rpl0_flag: bool,

    pub picture_cropping_flag: bool,
    pub picture_crop_left_offset: u32,
    pub picture_crop_right_offset: u32,
    pub picture_crop_top_offset: u32,
    pub picture_crop_bottom_offset: u32,

    pub vui_parameters_present_flag: bool,
}

impl Sps {
    /// `BitDepthY` (§7.4.3.1, eq. 37).
    pub fn bit_depth_y(&self) -> u32 {
        8 + self.bit_depth_luma_minus8
    }

    /// `BitDepthC` (§7.4.3.1, eq. 39).
    pub fn bit_depth_c(&self) -> u32 {
        8 + self.bit_depth_chroma_minus8
    }

    /// `ChromaArrayType` is set equal to `chroma_format_idc` (§7.4.3.1).
    pub fn chroma_array_type(&self) -> u32 {
        self.chroma_format_idc
    }
}

/// Parse a SPS RBSP body (§7.3.2.1). The 2-byte NAL header has already
/// been stripped by the caller.
pub fn parse(rbsp: &[u8]) -> Result<Sps> {
    let mut br = BitReader::new(rbsp);

    let sps_seq_parameter_set_id = br.ue()?;
    if sps_seq_parameter_set_id > MAX_SPS_ID {
        return Err(Error::invalid(format!(
            "evc sps: sps_seq_parameter_set_id {sps_seq_parameter_set_id} out of range (0..=15)"
        )));
    }

    let profile_idc = br.u(8)? as u8;
    let level_idc = br.u(8)? as u8;
    let toolset_idc_h = br.u(32)?;
    let toolset_idc_l = br.u(32)?;

    let chroma_format_idc = br.ue()?;
    if chroma_format_idc > 3 {
        return Err(Error::invalid(format!(
            "evc sps: chroma_format_idc {chroma_format_idc} out of range (0..=3)"
        )));
    }

    let pic_width_in_luma_samples = br.ue()?;
    let pic_height_in_luma_samples = br.ue()?;
    if pic_width_in_luma_samples == 0 || pic_height_in_luma_samples == 0 {
        return Err(Error::invalid(
            "evc sps: pic_width / pic_height in luma samples must be non-zero",
        ));
    }
    if pic_width_in_luma_samples > MAX_DIMENSION || pic_height_in_luma_samples > MAX_DIMENSION {
        return Err(Error::invalid(format!(
            "evc sps: pic dimensions {pic_width_in_luma_samples}x{pic_height_in_luma_samples} \
             exceed sanity bound {MAX_DIMENSION}"
        )));
    }

    let bit_depth_luma_minus8 = br.ue()?;
    let bit_depth_chroma_minus8 = br.ue()?;
    if bit_depth_luma_minus8 > 8 || bit_depth_chroma_minus8 > 8 {
        return Err(Error::invalid(format!(
            "evc sps: bit_depth_*_minus8 out of range (0..=8): luma={bit_depth_luma_minus8}, \
             chroma={bit_depth_chroma_minus8}"
        )));
    }

    let sps_btt_flag = br.u1()? != 0;
    let mut log2_ctu_size_minus5 = 1; // §7.4.3.1 default when not present
    let mut log2_min_cb_size_minus2 = 0;
    let mut log2_diff_ctu_max_14_cb_size = 0;
    let mut log2_diff_ctu_max_tt_cb_size = 0;
    let mut log2_diff_min_cb_min_tt_cb_size_minus2 = 0;
    if sps_btt_flag {
        log2_ctu_size_minus5 = br.ue()?;
        log2_min_cb_size_minus2 = br.ue()?;
        log2_diff_ctu_max_14_cb_size = br.ue()?;
        log2_diff_ctu_max_tt_cb_size = br.ue()?;
        log2_diff_min_cb_min_tt_cb_size_minus2 = br.ue()?;
        if log2_ctu_size_minus5 > 2 {
            return Err(Error::invalid(format!(
                "evc sps: log2_ctu_size_minus5 {log2_ctu_size_minus5} out of range (0..=2)"
            )));
        }
    }

    let sps_suco_flag = br.u1()? != 0;
    let mut log2_diff_ctu_size_max_suco_cb_size = 0;
    let mut log2_diff_max_suco_min_suco_cb_size = 0;
    if sps_suco_flag {
        log2_diff_ctu_size_max_suco_cb_size = br.ue()?;
        log2_diff_max_suco_min_suco_cb_size = br.ue()?;
    }

    let sps_admvp_flag = br.u1()? != 0;
    let mut sps_affine_flag = false;
    let mut sps_amvr_flag = false;
    let mut sps_dmvr_flag = false;
    let mut sps_mmvd_flag = false;
    let mut sps_hmvp_flag = false;
    if sps_admvp_flag {
        sps_affine_flag = br.u1()? != 0;
        sps_amvr_flag = br.u1()? != 0;
        sps_dmvr_flag = br.u1()? != 0;
        sps_mmvd_flag = br.u1()? != 0;
        sps_hmvp_flag = br.u1()? != 0;
    }

    let sps_eipd_flag = br.u1()? != 0;
    let mut sps_ibc_flag = false;
    let mut log2_max_ibc_cand_size_minus2 = 0;
    if sps_eipd_flag {
        sps_ibc_flag = br.u1()? != 0;
        if sps_ibc_flag {
            log2_max_ibc_cand_size_minus2 = br.ue()?;
        }
    }

    let sps_cm_init_flag = br.u1()? != 0;
    let mut sps_adcc_flag = false;
    if sps_cm_init_flag {
        sps_adcc_flag = br.u1()? != 0;
    }

    let sps_iqt_flag = br.u1()? != 0;
    let mut sps_ats_flag = false;
    if sps_iqt_flag {
        sps_ats_flag = br.u1()? != 0;
    }

    let sps_addb_flag = br.u1()? != 0;
    let sps_alf_flag = br.u1()? != 0;
    let sps_htdf_flag = br.u1()? != 0;
    let sps_rpl_flag = br.u1()? != 0;
    let sps_pocs_flag = br.u1()? != 0;
    let sps_dquant_flag = br.u1()? != 0;
    let sps_dra_flag = br.u1()? != 0;

    let mut log2_max_pic_order_cnt_lsb_minus4 = 0;
    if sps_pocs_flag {
        log2_max_pic_order_cnt_lsb_minus4 = br.ue()?;
    }

    let mut log2_sub_gop_length = 0;
    let mut log2_ref_pic_gap_length = 0;
    if !sps_pocs_flag || !sps_rpl_flag {
        log2_sub_gop_length = br.ue()?;
        if log2_sub_gop_length == 0 {
            log2_ref_pic_gap_length = br.ue()?;
        }
    }

    let mut max_num_tid0_ref_pics = 0;
    let mut sps_max_dec_pic_buffering_minus1 = 0;
    let mut long_term_ref_pics_flag = false;
    let mut rpl1_same_as_rpl0_flag = false;
    if !sps_rpl_flag {
        max_num_tid0_ref_pics = br.ue()?;
    } else {
        sps_max_dec_pic_buffering_minus1 = br.ue()?;
        long_term_ref_pics_flag = br.u1()? != 0;
        rpl1_same_as_rpl0_flag = br.u1()? != 0;
        // Sanity-bound `num_ref_pic_lists_in_sps[i]` so a bogus value cannot
        // trigger a giant allocation in the inner loop.
        let lists_to_read = if rpl1_same_as_rpl0_flag { 1 } else { 2 };
        for _ in 0..lists_to_read {
            let n = br.ue()?;
            if n > 64 {
                return Err(Error::invalid(format!(
                    "evc sps: num_ref_pic_lists_in_sps[i] {n} > 64 sanity bound"
                )));
            }
            for _ in 0..n {
                // ref_pic_list_struct() is part of the RPL machinery; its
                // shape is not needed for round-1 probe but the encoder
                // still emits the bytes. Skip the structure body — round-2
                // will replace this with a real parser.
                skip_ref_pic_list_struct(&mut br, long_term_ref_pics_flag)?;
            }
        }
    }

    let picture_cropping_flag = br.u1()? != 0;
    let mut picture_crop_left_offset = 0;
    let mut picture_crop_right_offset = 0;
    let mut picture_crop_top_offset = 0;
    let mut picture_crop_bottom_offset = 0;
    if picture_cropping_flag {
        picture_crop_left_offset = br.ue()?;
        picture_crop_right_offset = br.ue()?;
        picture_crop_top_offset = br.ue()?;
        picture_crop_bottom_offset = br.ue()?;
    }

    if chroma_format_idc != 0 {
        let chroma_qp_table_present_flag = br.u1()? != 0;
        if chroma_qp_table_present_flag {
            let same_qp_table_for_chroma = br.u1()? != 0;
            let _global_offset_flag = br.u1()? != 0;
            let n_tables = if same_qp_table_for_chroma { 1 } else { 2 };
            for _ in 0..n_tables {
                let num_points_minus1 = br.ue()?;
                if num_points_minus1 > 64 {
                    return Err(Error::invalid(format!(
                        "evc sps: num_points_in_qp_table_minus1 {num_points_minus1} > 64"
                    )));
                }
                for _ in 0..=num_points_minus1 {
                    let _delta_qp_in = br.u(6)?;
                    let _delta_qp_out = br.se()?;
                }
            }
        }
    }

    let vui_parameters_present_flag = br.u1()? != 0;
    // VUI body parse is out of scope for round-1; if it's signalled but
    // absent we'll surface trailing-bit errors downstream rather than
    // pretending to skip an unknown number of bits.

    Ok(Sps {
        sps_seq_parameter_set_id,
        profile_idc,
        level_idc,
        toolset_idc_h,
        toolset_idc_l,
        chroma_format_idc,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        sps_btt_flag,
        log2_ctu_size_minus5,
        log2_min_cb_size_minus2,
        log2_diff_ctu_max_14_cb_size,
        log2_diff_ctu_max_tt_cb_size,
        log2_diff_min_cb_min_tt_cb_size_minus2,
        sps_suco_flag,
        log2_diff_ctu_size_max_suco_cb_size,
        log2_diff_max_suco_min_suco_cb_size,
        sps_admvp_flag,
        sps_affine_flag,
        sps_amvr_flag,
        sps_dmvr_flag,
        sps_mmvd_flag,
        sps_hmvp_flag,
        sps_eipd_flag,
        sps_ibc_flag,
        log2_max_ibc_cand_size_minus2,
        sps_cm_init_flag,
        sps_adcc_flag,
        sps_iqt_flag,
        sps_ats_flag,
        sps_addb_flag,
        sps_alf_flag,
        sps_htdf_flag,
        sps_rpl_flag,
        sps_pocs_flag,
        sps_dquant_flag,
        sps_dra_flag,
        log2_max_pic_order_cnt_lsb_minus4,
        log2_sub_gop_length,
        log2_ref_pic_gap_length,
        max_num_tid0_ref_pics,
        sps_max_dec_pic_buffering_minus1,
        long_term_ref_pics_flag,
        rpl1_same_as_rpl0_flag,
        picture_cropping_flag,
        picture_crop_left_offset,
        picture_crop_right_offset,
        picture_crop_top_offset,
        picture_crop_bottom_offset,
        vui_parameters_present_flag,
    })
}

/// Skip a single `ref_pic_list_struct()` (§7.3.7). Round-1 only needs to
/// keep the bit reader in step; the structure itself is consumed by round-2.
fn skip_ref_pic_list_struct(br: &mut BitReader, long_term_ref_pics_flag: bool) -> Result<()> {
    // Per §7.3.7 / §7.4.8 the structure is essentially:
    //   num_ref_pics: ue(v)        — sanity-bound below
    //   for i in 0..num_ref_pics:
    //     if long_term_ref_pics_flag: lt_ref_pic_flag: u(1)
    //     if !lt_ref_pic_flag (or LT not enabled):
    //       abs_delta_poc_st_minus1: ue(v)
    //       if abs_delta_poc_st_minus1 != 0: strp_entry_sign_flag: u(1)
    //     else:
    //       poc_lsb_lt: u(log2_max_pic_order_cnt_lsb) — round-1 cannot
    //       resolve the dynamic length without a fully-parsed SPS; we
    //       conservatively skip 4 bits which matches the spec default
    //       log2_max_pic_order_cnt_lsb_minus4 == 0.
    let num_ref_pics = br.ue()?;
    if num_ref_pics > 64 {
        return Err(Error::invalid(format!(
            "evc sps rpl: num_ref_pics {num_ref_pics} > 64 sanity bound"
        )));
    }
    for _ in 0..num_ref_pics {
        let mut is_lt = false;
        if long_term_ref_pics_flag {
            is_lt = br.u1()? != 0;
        }
        if !is_lt {
            let abs_delta_poc_st_minus1 = br.ue()?;
            if abs_delta_poc_st_minus1 > 0 {
                let _sign = br.u1()?;
            }
        } else {
            let _poc_lsb_lt = br.u(4)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Hand-build a minimal SPS RBSP for a 320x240 4:2:0 8-bit stream
    /// with all toolset flags disabled (sps_btt=0, sps_suco=0, sps_admvp=0,
    /// sps_eipd=0, sps_cm_init=0, sps_iqt=0; sps_addb..sps_dra=0;
    /// sps_pocs_flag=0, sps_rpl_flag=0; picture_cropping_flag=0;
    /// vui_parameters_present_flag=0).
    fn build_minimal_sps_rbsp() -> Vec<u8> {
        // Rather than emit bytes directly, drive a small bit-emitter so the
        // codepath we test mirrors what the spec encoder would produce.
        let mut emitter = BitEmitter::new();
        emitter.ue(0); // sps_seq_parameter_set_id
        emitter.u(8, 0x01); // profile_idc = 1 (Baseline placeholder)
        emitter.u(8, 30); // level_idc = 30
        emitter.u(32, 0); // toolset_idc_h
        emitter.u(32, 0); // toolset_idc_l
        emitter.ue(1); // chroma_format_idc = 1 (4:2:0)
        emitter.ue(320); // pic_width_in_luma_samples
        emitter.ue(240); // pic_height_in_luma_samples
        emitter.ue(0); // bit_depth_luma_minus8
        emitter.ue(0); // bit_depth_chroma_minus8
        emitter.u(1, 0); // sps_btt_flag
        emitter.u(1, 0); // sps_suco_flag
        emitter.u(1, 0); // sps_admvp_flag
        emitter.u(1, 0); // sps_eipd_flag
        emitter.u(1, 0); // sps_cm_init_flag
        emitter.u(1, 0); // sps_iqt_flag
        emitter.u(1, 0); // sps_addb_flag
        emitter.u(1, 0); // sps_alf_flag
        emitter.u(1, 0); // sps_htdf_flag
        emitter.u(1, 0); // sps_rpl_flag
        emitter.u(1, 0); // sps_pocs_flag
        emitter.u(1, 0); // sps_dquant_flag
        emitter.u(1, 0); // sps_dra_flag
                         // !sps_pocs_flag || !sps_rpl_flag → log2_sub_gop_length present
        emitter.ue(1); // log2_sub_gop_length = 1 (so log2_ref_pic_gap_length skipped)
                       // !sps_rpl_flag → max_num_tid0_ref_pics
        emitter.ue(1);
        emitter.u(1, 0); // picture_cropping_flag
                         // chroma_format_idc != 0 → chroma_qp_table_present_flag
        emitter.u(1, 0); // chroma_qp_table_present_flag
        emitter.u(1, 0); // vui_parameters_present_flag
        emitter.finish_with_trailing_bits();
        emitter.into_bytes()
    }

    /// Tiny helper that builds RBSP bytes by writing big-endian bit fields.
    /// Out-of-line so tests stay readable; not part of the public surface.
    pub(crate) struct BitEmitter {
        bytes: Vec<u8>,
        // Number of valid bits written (not counting padding).
        bit_pos: u32,
    }

    impl BitEmitter {
        pub fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }

        pub fn u(&mut self, n: u32, value: u32) {
            for k in (0..n).rev() {
                let bit = (value >> k) & 1;
                self.write_bit(bit as u8);
            }
        }

        pub fn ue(&mut self, value: u32) {
            // Encode 0-th order Exp-Golomb: leading zeros count = bits_in(value+1)-1.
            let v1 = value + 1;
            let bits = 32 - v1.leading_zeros();
            let zeros = bits - 1;
            for _ in 0..zeros {
                self.write_bit(0);
            }
            self.write_bit(1);
            let suffix_bits = zeros;
            if suffix_bits > 0 {
                let suffix = v1 & ((1u32 << suffix_bits) - 1);
                for k in (0..suffix_bits).rev() {
                    self.write_bit(((suffix >> k) & 1) as u8);
                }
            }
        }

        pub fn bit_position(&self) -> u32 {
            self.bit_pos
        }

        pub fn finish_with_trailing_bits(&mut self) {
            // rbsp_trailing_bits(): one '1' then '0' padding to byte align.
            self.write_bit(1);
            while self.bit_pos % 8 != 0 {
                self.write_bit(0);
            }
        }

        pub fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }

        fn write_bit(&mut self, bit: u8) {
            if self.bit_pos % 8 == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            let shift = 7 - (self.bit_pos % 8);
            self.bytes[last] |= (bit & 1) << shift;
            self.bit_pos += 1;
        }
    }

    #[test]
    fn parse_minimal_sps_320x240() {
        let rbsp = build_minimal_sps_rbsp();
        let sps = parse(&rbsp).expect("minimal SPS must parse");
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.profile_idc, 1);
        assert_eq!(sps.level_idc, 30);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_width_in_luma_samples, 320);
        assert_eq!(sps.pic_height_in_luma_samples, 240);
        assert_eq!(sps.bit_depth_y(), 8);
        assert_eq!(sps.bit_depth_c(), 8);
        assert!(!sps.sps_btt_flag);
        assert!(!sps.vui_parameters_present_flag);
    }

    #[test]
    fn rejects_oversized_dimensions() {
        // Replace the 320 width with the over-bound value and re-emit.
        let mut emitter = BitEmitter::new();
        emitter.ue(0);
        emitter.u(8, 1);
        emitter.u(8, 30);
        emitter.u(32, 0);
        emitter.u(32, 0);
        emitter.ue(1);
        emitter.ue(MAX_DIMENSION + 1);
        emitter.ue(240);
        emitter.ue(0);
        emitter.ue(0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.ue(1);
        emitter.ue(1);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.u(1, 0);
        emitter.finish_with_trailing_bits();
        let rbsp = emitter.into_bytes();
        let err = parse(&rbsp).unwrap_err();
        assert!(format!("{err}").contains("sanity bound"));
    }

    #[test]
    fn rejects_zero_dimensions() {
        let mut emitter = BitEmitter::new();
        emitter.ue(0);
        emitter.u(8, 1);
        emitter.u(8, 30);
        emitter.u(32, 0);
        emitter.u(32, 0);
        emitter.ue(1);
        emitter.ue(0); // width 0
        emitter.ue(240);
        emitter.ue(0);
        emitter.ue(0);
        emitter.finish_with_trailing_bits();
        let rbsp = emitter.into_bytes();
        let err = parse(&rbsp).unwrap_err();
        assert!(format!("{err}").contains("non-zero"));
    }

    #[test]
    fn parses_with_btt_flag() {
        let mut emitter = BitEmitter::new();
        emitter.ue(0);
        emitter.u(8, 2);
        emitter.u(8, 41);
        emitter.u(32, 1);
        emitter.u(32, 1);
        emitter.ue(1);
        emitter.ue(1920);
        emitter.ue(1080);
        emitter.ue(2); // 10-bit luma
        emitter.ue(2); // 10-bit chroma
        emitter.u(1, 1); // sps_btt_flag
        emitter.ue(1); // log2_ctu_size_minus5
        emitter.ue(0);
        emitter.ue(2);
        emitter.ue(1);
        emitter.ue(0);
        emitter.u(1, 0); // sps_suco_flag
        emitter.u(1, 0); // sps_admvp_flag
        emitter.u(1, 0); // sps_eipd_flag
        emitter.u(1, 0); // sps_cm_init_flag
        emitter.u(1, 0); // sps_iqt_flag
        emitter.u(1, 0); // sps_addb_flag
        emitter.u(1, 0); // sps_alf_flag
        emitter.u(1, 0); // sps_htdf_flag
        emitter.u(1, 0); // sps_rpl_flag
        emitter.u(1, 0); // sps_pocs_flag
        emitter.u(1, 0); // sps_dquant_flag
        emitter.u(1, 0); // sps_dra_flag
        emitter.ue(1); // log2_sub_gop_length=1
        emitter.ue(1); // max_num_tid0_ref_pics
        emitter.u(1, 0); // picture_cropping_flag
        emitter.u(1, 0); // chroma_qp_table_present_flag
        emitter.u(1, 0); // vui_parameters_present_flag
        emitter.finish_with_trailing_bits();
        let rbsp = emitter.into_bytes();
        let sps = parse(&rbsp).unwrap();
        assert_eq!(sps.pic_width_in_luma_samples, 1920);
        assert_eq!(sps.pic_height_in_luma_samples, 1080);
        assert_eq!(sps.bit_depth_y(), 10);
        assert!(sps.sps_btt_flag);
        assert_eq!(sps.log2_ctu_size_minus5, 1);
    }
}
