//! EVC slice header parser (ISO/IEC 23094-1 §7.3.4 / §7.4.5).
//!
//! The slice header depends on a previously-parsed PPS for tile geometry
//! and on the corresponding SPS for several gated flags. Round-1 surfaces
//! only the subset of fields that the workspace needs to drive a probe and
//! to validate hand-built fixtures: PPS id, slice type, POC LSB (when
//! present), QP and the chroma offsets, and the deblocking enable flag.
//!
//! The caller passes a [`SliceParseContext`] carrying the relevant SPS /
//! PPS toggles. To keep this round self-contained we do not yet wire the
//! parser into the central registry.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::nal::NalUnitType;

/// Slice type values from §7.4.5 Table 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    /// `slice_type == 0`
    B,
    /// `slice_type == 1`
    P,
    /// `slice_type == 2`
    I,
}

impl SliceType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(SliceType::B),
            1 => Ok(SliceType::P),
            2 => Ok(SliceType::I),
            other => Err(Error::invalid(format!(
                "evc slice: unknown slice_type {other} (Table 8: 0=B, 1=P, 2=I)"
            ))),
        }
    }
}

/// Subset of SPS / PPS state needed to parse a slice header without the
/// full parameter-set machinery wired up yet. Round-2 will replace this
/// with a proper "active parameter sets" tracker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SliceParseContext {
    pub single_tile_in_pic_flag: bool,
    pub arbitrary_slice_present_flag: bool,
    pub tile_id_len_minus1: u32,
    pub num_tile_columns_minus1: u32,
    pub num_tile_rows_minus1: u32,
    pub sps_pocs_flag: bool,
    pub sps_rpl_flag: bool,
    pub sps_alf_flag: bool,
    pub sps_mmvd_flag: bool,
    pub sps_admvp_flag: bool,
    pub sps_addb_flag: bool,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    pub chroma_array_type: u32,
}

#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub slice_pic_parameter_set_id: u32,
    pub single_tile_in_slice_flag: bool,
    pub first_tile_id: u32,
    pub last_tile_id: u32,
    pub arbitrary_slice_flag: bool,
    pub num_remaining_tiles_in_slice_minus1: u32,
    pub slice_type: SliceType,
    pub no_output_of_prior_pics_flag: bool,
    pub mmvd_group_enable_flag: bool,
    pub slice_alf_enabled_flag: bool,
    pub slice_pic_order_cnt_lsb: u32,
    pub num_ref_idx_active_override_flag: bool,
    pub num_ref_idx_active_minus1: [u32; 2],
    pub temporal_mvp_assigned_flag: bool,
    pub slice_deblocking_filter_flag: bool,
    pub slice_alpha_offset: i32,
    pub slice_beta_offset: i32,
    pub slice_qp: u32,
    pub slice_cb_qp_offset: i32,
    pub slice_cr_qp_offset: i32,
}

pub fn parse(
    rbsp: &[u8],
    nal_unit_type: NalUnitType,
    ctx: &SliceParseContext,
) -> Result<SliceHeader> {
    let mut br = BitReader::new(rbsp);
    let slice_pic_parameter_set_id = br.ue()?;
    if slice_pic_parameter_set_id > 63 {
        return Err(Error::invalid(format!(
            "evc slice: slice_pic_parameter_set_id {slice_pic_parameter_set_id} > 63"
        )));
    }

    let mut single_tile_in_slice_flag = true;
    let mut first_tile_id = 0;
    let mut last_tile_id = 0;
    let mut arbitrary_slice_flag = false;
    let mut num_remaining_tiles_in_slice_minus1 = 0;
    let id_bits = ctx.tile_id_len_minus1 + 1;
    if !ctx.single_tile_in_pic_flag {
        single_tile_in_slice_flag = br.u1()? != 0;
        first_tile_id = br.u(id_bits)?;
    }
    if !single_tile_in_slice_flag {
        if ctx.arbitrary_slice_present_flag {
            arbitrary_slice_flag = br.u1()? != 0;
        }
        if !arbitrary_slice_flag {
            last_tile_id = br.u(id_bits)?;
        } else {
            num_remaining_tiles_in_slice_minus1 = br.ue()?;
            // Skip the delta tile-id loop: round-1 has no need to materialise
            // it, but we still consume the bits so trailing fields parse.
            for _ in 0..num_remaining_tiles_in_slice_minus1 {
                let _delta = br.ue()?;
            }
        }
    }

    let slice_type_raw = br.ue()?;
    let slice_type = SliceType::from_u32(slice_type_raw)?;

    let mut no_output_of_prior_pics_flag = false;
    if matches!(nal_unit_type, NalUnitType::Idr) {
        no_output_of_prior_pics_flag = br.u1()? != 0;
        // §7.4.5: when nal_unit_type == IDR_NUT, slice_type shall equal 2 (I).
        if slice_type != SliceType::I {
            return Err(Error::invalid(
                "evc slice: IDR slice must have slice_type == I (§7.4.5)",
            ));
        }
    }

    let mut mmvd_group_enable_flag = false;
    if ctx.sps_mmvd_flag && (slice_type == SliceType::B || slice_type == SliceType::P) {
        mmvd_group_enable_flag = br.u1()? != 0;
    }

    let mut slice_alf_enabled_flag = false;
    if ctx.sps_alf_flag {
        slice_alf_enabled_flag = br.u1()? != 0;
        // We intentionally skip the rest of the ALF APS-id selection —
        // round-2 will route this through the APS parser. Surfacing the
        // enable flag is enough for round-1 probe.
        if slice_alf_enabled_flag {
            // slice_alf_luma_aps_id: u(5) + slice_alf_map_flag: u(1) +
            // slice_alf_chroma_idc: u(2) — when chroma is present.
            let _luma_id = br.u(5)?;
            let _map_flag = br.u1()?;
            let chroma_idc = br.u(2)?;
            if (ctx.chroma_array_type == 1 || ctx.chroma_array_type == 2) && chroma_idc > 0 {
                let _chroma_id = br.u(5)?;
            }
            // ChromaArrayType == 3 branch would add chroma2 fields; not
            // needed for the round-1 fixtures we drive.
        }
    }

    let mut slice_pic_order_cnt_lsb = 0;
    if !matches!(nal_unit_type, NalUnitType::Idr) && ctx.sps_pocs_flag {
        let bits = ctx.log2_max_pic_order_cnt_lsb_minus4 + 4;
        slice_pic_order_cnt_lsb = br.u(bits)?;
    }
    // The full ref-pic-list-struct path (sps_rpl_flag) is intentionally
    // out of scope for round-1; we won't read those bits, so the header
    // parsing stops being meaningful past this point if RPL is enabled.
    // Round-1 fixtures keep sps_rpl_flag = 0.
    if !matches!(nal_unit_type, NalUnitType::Idr) && ctx.sps_rpl_flag {
        return Err(Error::unsupported(
            "evc slice: sps_rpl_flag path not yet implemented (round-2 deliverable)",
        ));
    }

    let mut num_ref_idx_active_override_flag = false;
    let mut num_ref_idx_active_minus1 = [0u32; 2];
    let mut temporal_mvp_assigned_flag = false;
    if !matches!(nal_unit_type, NalUnitType::Idr)
        && (slice_type == SliceType::P || slice_type == SliceType::B)
    {
        num_ref_idx_active_override_flag = br.u1()? != 0;
        if num_ref_idx_active_override_flag {
            let n = if slice_type == SliceType::B { 2 } else { 1 };
            for slot in num_ref_idx_active_minus1.iter_mut().take(n) {
                *slot = br.ue()?;
            }
        }
        if ctx.sps_admvp_flag {
            temporal_mvp_assigned_flag = br.u1()? != 0;
            if temporal_mvp_assigned_flag {
                if slice_type == SliceType::B {
                    let _col_pic_list_idx = br.u1()?;
                    let _col_source_mvp_list_idx = br.u1()?;
                }
                let _col_pic_ref_idx = br.ue()?;
            }
        }
    }

    let slice_deblocking_filter_flag = br.u1()? != 0;
    let mut slice_alpha_offset = 0;
    let mut slice_beta_offset = 0;
    if slice_deblocking_filter_flag && ctx.sps_addb_flag {
        slice_alpha_offset = br.se()?;
        slice_beta_offset = br.se()?;
        if !(-12..=12).contains(&slice_alpha_offset) || !(-12..=12).contains(&slice_beta_offset) {
            return Err(Error::invalid(
                "evc slice: deblocking offset out of range [-12, +12]",
            ));
        }
    }

    let slice_qp = br.u(6)?;
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc slice: slice_qp {slice_qp} > 51"
        )));
    }
    let slice_cb_qp_offset = br.se()?;
    let slice_cr_qp_offset = br.se()?;
    if !(-12..=12).contains(&slice_cb_qp_offset) || !(-12..=12).contains(&slice_cr_qp_offset) {
        return Err(Error::invalid(
            "evc slice: slice_cb/cr_qp_offset out of range [-12, +12]",
        ));
    }

    Ok(SliceHeader {
        slice_pic_parameter_set_id,
        single_tile_in_slice_flag,
        first_tile_id,
        last_tile_id,
        arbitrary_slice_flag,
        num_remaining_tiles_in_slice_minus1,
        slice_type,
        no_output_of_prior_pics_flag,
        mmvd_group_enable_flag,
        slice_alf_enabled_flag,
        slice_pic_order_cnt_lsb,
        num_ref_idx_active_override_flag,
        num_ref_idx_active_minus1,
        temporal_mvp_assigned_flag,
        slice_deblocking_filter_flag,
        slice_alpha_offset,
        slice_beta_offset,
        slice_qp,
        slice_cb_qp_offset,
        slice_cr_qp_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    fn idr_ctx() -> SliceParseContext {
        SliceParseContext {
            single_tile_in_pic_flag: true,
            chroma_array_type: 1,
            ..Default::default()
        }
    }

    #[test]
    fn parse_idr_minimal() {
        let mut e = BitEmitter::new();
        e.ue(0); // slice_pic_parameter_set_id
        e.ue(2); // slice_type = I
        e.u(1, 1); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0); // slice_cb_qp_offset (se: 0)
        e.ue(0); // slice_cr_qp_offset
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &idr_ctx()).unwrap();
        assert_eq!(sh.slice_pic_parameter_set_id, 0);
        assert_eq!(sh.slice_type, SliceType::I);
        assert!(sh.no_output_of_prior_pics_flag);
        assert!(sh.slice_deblocking_filter_flag);
        assert_eq!(sh.slice_qp, 26);
    }

    #[test]
    fn idr_must_be_intra() {
        let mut e = BitEmitter::new();
        e.ue(0);
        e.ue(0); // slice_type = B (illegal for IDR)
        e.u(1, 0);
        e.u(1, 1);
        e.u(6, 26);
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let err = parse(&rbsp, NalUnitType::Idr, &idr_ctx()).unwrap_err();
        assert!(format!("{err}").contains("IDR slice"));
    }

    #[test]
    fn parse_p_slice_with_pocs() {
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_pocs_flag: true,
            log2_max_pic_order_cnt_lsb_minus4: 4, // 8-bit POC LSB
            chroma_array_type: 1,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.ue(1); // slice_type = P
                 // not IDR → no no_output_of_prior_pics_flag
                 // sps_mmvd_flag = false → no mmvd_group_enable_flag
                 // sps_alf_flag = false → no slice_alf_enabled_flag
        e.u(8, 0xAB); // slice_pic_order_cnt_lsb (8 bits)
                      // not IDR + slice_type == P → ref-idx + admvp branch
        e.u(1, 0); // num_ref_idx_active_override_flag
                   // sps_admvp_flag = false → skip
        e.u(1, 0); // slice_deblocking_filter_flag
        e.u(6, 22); // slice_qp
        e.ue(0); // cb offset
        e.ue(0); // cr offset
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::NonIdr, &ctx).unwrap();
        assert_eq!(sh.slice_type, SliceType::P);
        assert_eq!(sh.slice_pic_order_cnt_lsb, 0xAB);
        assert_eq!(sh.slice_qp, 22);
        assert!(!sh.slice_deblocking_filter_flag);
    }

    #[test]
    fn rejects_qp_over_51() {
        let mut e = BitEmitter::new();
        e.ue(0);
        e.ue(2);
        e.u(1, 0);
        e.u(1, 1);
        e.u(6, 60); // illegal QP
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        assert!(parse(&rbsp, NalUnitType::Idr, &idr_ctx()).is_err());
    }
}
