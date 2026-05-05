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
use crate::rpl::{parse_ref_pic_list_struct, RefPicListStruct};

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
    /// `num_ref_pic_lists_in_sps[ 0 ]` from the active SPS. When non-zero
    /// the slice may signal `ref_pic_list_sps_flag[ 0 ]` to pick an entry
    /// from this list instead of carrying its own RPL inline.
    pub num_ref_pic_lists_in_sps_l0: u32,
    /// `num_ref_pic_lists_in_sps[ 1 ]` from the active SPS.
    pub num_ref_pic_lists_in_sps_l1: u32,
    /// PPS-resident `rpl1_idx_present_flag`. When `0`, list 1's
    /// `ref_pic_list_sps_flag` / `ref_pic_list_idx` are inferred from
    /// list 0's per §7.4.5.
    pub rpl1_idx_present_flag: bool,
    /// `long_term_ref_pics_flag` from the active SPS — gates LTRP entries
    /// inside any inline `ref_pic_list_struct()`.
    pub long_term_ref_pics_flag: bool,
    /// `additional_lt_poc_lsb_len` from the PPS — sizes
    /// `additional_poc_lsb_val` per §7.4.5.
    pub additional_lt_poc_lsb_len: u32,
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
    /// `ref_pic_list_sps_flag[ i ]` — `true` when the slice picks an SPS
    /// candidate; `false` when the slice carries its own RPL inline.
    pub ref_pic_list_sps_flag: [bool; 2],
    /// `ref_pic_list_idx[ i ]` — index into the SPS's
    /// `ref_pic_list_struct(i, *, ltrpFlag)` array. Only meaningful when
    /// `ref_pic_list_sps_flag[ i ] == true`.
    pub ref_pic_list_idx: [u32; 2],
    /// Inline `ref_pic_list_struct( i, num_ref_pic_lists_in_sps[i],
    /// long_term_ref_pics_flag )` carried by the slice header itself.
    /// `None` when `ref_pic_list_sps_flag[ i ] == true`. Round-8 doesn't
    /// yet route this through the DPB but the parser surfaces it for
    /// future rounds.
    pub slice_rpl: [Option<RefPicListStruct>; 2],
    /// `SliceRplsIdx[ i ]` per §7.4.5 eq. 83.
    pub slice_rpls_idx: [u32; 2],
}

pub fn parse(
    rbsp: &[u8],
    nal_unit_type: NalUnitType,
    ctx: &SliceParseContext,
) -> Result<SliceHeader> {
    let mut br = BitReader::new(rbsp);
    parse_from_bitreader(&mut br, nal_unit_type, ctx)
}

/// Parse the slice header off an existing [`BitReader`], leaving the
/// reader positioned just past the last `se(v)` field. Useful for callers
/// that need to recover the byte offset of `slice_data()` (which is
/// byte-aligned right after the header — see §7.3.4 / §7.4.5).
pub fn parse_consume(
    br: &mut BitReader,
    nal_unit_type: NalUnitType,
    ctx: &SliceParseContext,
) -> Result<SliceHeader> {
    parse_from_bitreader(br, nal_unit_type, ctx)
}

fn parse_from_bitreader(
    br: &mut BitReader,
    nal_unit_type: NalUnitType,
    ctx: &SliceParseContext,
) -> Result<SliceHeader> {
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
    // §7.3.4 sps_rpl_flag branch — the slice may either point at an SPS
    // RPL candidate or carry its own `ref_pic_list_struct()` inline. The
    // round-8 deliverable fully consumes both paths so subsequent inter
    // pictures can be decoded against the previous reference frame.
    let mut ref_pic_list_sps_flag = [false; 2];
    let mut ref_pic_list_idx = [0u32; 2];
    let mut slice_rpl: [Option<RefPicListStruct>; 2] = [None, None];
    let mut slice_rpls_idx = [0u32; 2];
    if !matches!(nal_unit_type, NalUnitType::Idr) && ctx.sps_rpl_flag {
        let log2_max_poc_lsb = ctx.log2_max_pic_order_cnt_lsb_minus4 + 4;
        for i in 0..2 {
            let n_in_sps = if i == 0 {
                ctx.num_ref_pic_lists_in_sps_l0
            } else {
                ctx.num_ref_pic_lists_in_sps_l1
            };
            // Per §7.3.4: ref_pic_list_sps_flag[i] is signalled only if
            // num_ref_pic_lists_in_sps[i] > 0 and (i==0 or rpl1_idx_present).
            let signal_sps_flag = n_in_sps > 0 && (i == 0 || ctx.rpl1_idx_present_flag);
            if signal_sps_flag {
                ref_pic_list_sps_flag[i] = br.u1()? != 0;
            } else if i == 1 && !ctx.rpl1_idx_present_flag && n_in_sps > 0 {
                // §7.4.5: when not present and rpl1_idx_present_flag = 0,
                // ref_pic_list_sps_flag[1] is inferred to ref_pic_list_sps_flag[0].
                ref_pic_list_sps_flag[1] = ref_pic_list_sps_flag[0];
            }
            if ref_pic_list_sps_flag[i] {
                let signal_idx = n_in_sps > 1 && (i == 0 || ctx.rpl1_idx_present_flag);
                if signal_idx {
                    let bits = ceil_log2(n_in_sps);
                    ref_pic_list_idx[i] = br.u(bits)?;
                    if ref_pic_list_idx[i] >= n_in_sps {
                        return Err(Error::invalid(format!(
                            "evc slice: ref_pic_list_idx[{i}] {} out of range (0..{n_in_sps})",
                            ref_pic_list_idx[i]
                        )));
                    }
                } else if i == 1 && !ctx.rpl1_idx_present_flag {
                    // §7.4.5: ref_pic_list_idx[1] inferred to ref_pic_list_idx[0]
                    // when not present and rpl1_idx_present_flag = 0.
                    ref_pic_list_idx[1] = ref_pic_list_idx[0];
                }
            } else {
                // Slice carries its own `ref_pic_list_struct()` inline.
                slice_rpl[i] = Some(parse_ref_pic_list_struct(
                    br,
                    ctx.long_term_ref_pics_flag,
                    log2_max_poc_lsb,
                )?);
            }
            // §7.4.5 eq. 83: SliceRplsIdx[i].
            slice_rpls_idx[i] = if ref_pic_list_sps_flag[i] {
                ref_pic_list_idx[i]
            } else {
                n_in_sps
            };
            // additional_poc_lsb fields per LTRP entry. We need to know
            // num_ltrp_entries[i][SliceRplsIdx[i]] — for inline RPL we
            // have it directly; for SPS-resident RPL the caller has to
            // resolve it from the SPS, which the parser doesn't carry
            // here. We conservatively only walk the additional_poc_lsb
            // loop for the inline case (the SPS-RPL case requires the
            // caller to invoke a separate post-parse step, but most real
            // bitstreams use the inline path on slice headers anyway).
            if let Some(ref rpl) = slice_rpl[i] {
                for _ in 0..rpl.num_ltrp_entries {
                    let additional_poc_lsb_present_flag = br.u1()? != 0;
                    if additional_poc_lsb_present_flag {
                        let _additional_poc_lsb_val = br.u(ctx.additional_lt_poc_lsb_len)?;
                    }
                }
            }
        }
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
        ref_pic_list_sps_flag,
        ref_pic_list_idx,
        slice_rpl,
        slice_rpls_idx,
    })
}

/// `Ceil(Log2(n))` per §9.2 / Annex B helper. Defined as `0` for `n <= 1`.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
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
    fn parse_p_slice_with_inline_rpl() {
        // sps_rpl_flag = 1 with num_ref_pic_lists_in_sps[i] == 0 → slice
        // header signals neither ref_pic_list_sps_flag nor ref_pic_list_idx
        // and is forced down the inline ref_pic_list_struct() path.
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_pocs_flag: true,
            sps_rpl_flag: true,
            log2_max_pic_order_cnt_lsb_minus4: 4, // 8-bit POC LSB
            chroma_array_type: 1,
            num_ref_pic_lists_in_sps_l0: 0,
            num_ref_pic_lists_in_sps_l1: 0,
            rpl1_idx_present_flag: false,
            long_term_ref_pics_flag: false,
            additional_lt_poc_lsb_len: 0,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.ue(1); // slice_type = P
                 // not IDR + sps_pocs_flag = 1 → POC LSB (8 bits)
        e.u(8, 0x42);
        // sps_rpl_flag = 1 path: num_ref_pic_lists_in_sps[i] = 0 →
        // ref_pic_list_sps_flag NOT signalled (stays at 0) → inline RPL.
        // Inline RPL for L0: 1 STRP entry, delta=1, sign=1 (positive).
        e.ue(1); // num_strp_entries (long_term=0 so no num_ltrp)
        e.ue(1); // delta_poc_st = 1
        e.u(1, 1); // sign positive
                   // Inline RPL for L1: same shape, 1 STRP, delta=1, sign=0 (negative).
        e.ue(1);
        e.ue(1);
        e.u(1, 0);
        // P slice ref_idx + admvp branch:
        e.u(1, 0); // num_ref_idx_active_override_flag
                   // sps_admvp_flag = 0 → no temporal_mvp.
        e.u(1, 0); // slice_deblocking_filter_flag
        e.u(6, 22); // slice_qp
        e.ue(0); // cb offset
        e.ue(0); // cr offset
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::NonIdr, &ctx).unwrap();
        assert_eq!(sh.slice_type, SliceType::P);
        assert_eq!(sh.slice_pic_order_cnt_lsb, 0x42);
        // Both lists: ref_pic_list_sps_flag = false (inline path).
        assert!(!sh.ref_pic_list_sps_flag[0]);
        assert!(!sh.ref_pic_list_sps_flag[1]);
        // Inline RPLs were parsed.
        let rpl_l0 = sh.slice_rpl[0].as_ref().expect("L0 RPL inline");
        assert_eq!(rpl_l0.num_strp_entries, 1);
        assert_eq!(rpl_l0.entries[0].signed_delta_poc(), Some(1));
        let rpl_l1 = sh.slice_rpl[1].as_ref().expect("L1 RPL inline");
        assert_eq!(rpl_l1.entries[0].signed_delta_poc(), Some(-1));
        // SliceRplsIdx[i] = num_ref_pic_lists_in_sps[i] when inline.
        assert_eq!(sh.slice_rpls_idx[0], 0);
        assert_eq!(sh.slice_rpls_idx[1], 0);
    }

    #[test]
    fn parse_p_slice_with_sps_rpl_pointer() {
        // sps_rpl_flag = 1, num_ref_pic_lists_in_sps[0] = 4 → the slice
        // signals ref_pic_list_sps_flag[0] = 1 and ref_pic_list_idx[0]
        // (Ceil(Log2(4)) = 2 bits). rpl1_idx_present_flag = 0 inherits
        // list 1 from list 0.
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_pocs_flag: true,
            sps_rpl_flag: true,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            chroma_array_type: 1,
            num_ref_pic_lists_in_sps_l0: 4,
            num_ref_pic_lists_in_sps_l1: 4,
            rpl1_idx_present_flag: false,
            long_term_ref_pics_flag: false,
            additional_lt_poc_lsb_len: 0,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.ue(1); // P
        e.u(8, 0xAB); // POC LSB
                      // ref_pic_list_sps_flag[0] = 1
        e.u(1, 1);
        // ref_pic_list_idx[0] = 2 → 2 bits "10"
        e.u(2, 2);
        // rpl1_idx_present_flag = 0 + sps_flag[0] = 1 → ref_pic_list_sps_flag[1]
        // is inferred from sps_flag[0] (no extra bit) — but ONLY when not
        // signalled. Per §7.3.4 i==1 path: signal_sps_flag is gated on
        // (i==0 || rpl1_idx_present), so for i=1 with rpl1_idx_present=0
        // there is no explicit bit for sps_flag[1].
        // Same for ref_pic_list_idx[1] — not signalled, inferred to
        // ref_pic_list_idx[0] = 2.
        // Then ref_idx + admvp:
        e.u(1, 0); // num_ref_idx_active_override_flag
        e.u(1, 0); // slice_deblocking_filter_flag
        e.u(6, 22);
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::NonIdr, &ctx).unwrap();
        assert!(sh.ref_pic_list_sps_flag[0]);
        assert!(sh.ref_pic_list_sps_flag[1]);
        assert_eq!(sh.ref_pic_list_idx[0], 2);
        assert_eq!(sh.ref_pic_list_idx[1], 2);
        assert_eq!(sh.slice_rpls_idx[0], 2);
        assert_eq!(sh.slice_rpls_idx[1], 2);
        assert!(sh.slice_rpl[0].is_none());
        assert!(sh.slice_rpl[1].is_none());
    }

    #[test]
    fn ceil_log2_works() {
        assert_eq!(super::ceil_log2(0), 0);
        assert_eq!(super::ceil_log2(1), 0);
        assert_eq!(super::ceil_log2(2), 1);
        assert_eq!(super::ceil_log2(3), 2);
        assert_eq!(super::ceil_log2(4), 2);
        assert_eq!(super::ceil_log2(5), 3);
        assert_eq!(super::ceil_log2(8), 3);
        assert_eq!(super::ceil_log2(64), 6);
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
