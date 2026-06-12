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
use crate::pps::TileIndexMaps;
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
    /// `delta_tile_id_minus1[ i ]` (§7.4.5) — the per-tile ID deltas of
    /// an `arbitrary_slice_flag == 1` slice. Length `NumTilesInSlice − 1`
    /// (= `num_remaining_tiles_in_slice_minus1 + 1` by eq. 80); empty for
    /// rectangular slices. Feeds the eq. (81)/(82) `SliceTileIdx[ ]`
    /// derivation.
    pub delta_tile_id_minus1: Vec<u32>,
    pub slice_type: SliceType,
    pub no_output_of_prior_pics_flag: bool,
    pub mmvd_group_enable_flag: bool,
    pub slice_alf_enabled_flag: bool,
    /// `slice_alf_map_flag` (§7.4.5). When `slice_alf_enabled_flag` is 0,
    /// inferred 0. Gates the per-CTU luma `alf_ctb_flag` in
    /// `coding_tree_unit()` (§7.3.8.2 line 2626).
    pub slice_alf_map_flag: bool,
    /// `slice_alf_chroma_idc` (§7.4.5). 0 → no chroma ALF; 1 → Cb; 2 →
    /// Cr; 3 → both. Inferred 0 when not present.
    pub slice_alf_chroma_idc: u32,
    /// `sliceChromaAlfEnabledFlag` (§7.4.5 derived): set when
    /// `slice_alf_chroma_idc ∈ {1, 3}`.
    pub slice_chroma_alf_enabled_flag: bool,
    /// `sliceChroma2AlfEnabledFlag` (§7.4.5 derived): set when
    /// `slice_alf_chroma_idc ∈ {2, 3}`.
    pub slice_chroma2_alf_enabled_flag: bool,
    /// `slice_alf_chroma_map_flag` (§7.4.5). Only signalled when
    /// `ChromaArrayType == 3 && sliceChromaAlfEnabledFlag`; inferred 0
    /// otherwise (so 0 for every Baseline 4:2:0 slice). Gates
    /// `alf_ctb_chroma_flag` in `coding_tree_unit()` (line 2628).
    pub slice_alf_chroma_map_flag: bool,
    /// `slice_alf_chroma2_map_flag` (§7.4.5). Only signalled when
    /// `ChromaArrayType == 3 && sliceChroma2AlfEnabledFlag`; inferred 0
    /// otherwise. Gates `alf_ctb_chroma2_flag` (line 2630).
    pub slice_alf_chroma2_map_flag: bool,
    /// `slice_alf_luma_aps_id` (§7.3.4, 5-bit), present only when
    /// `sps_alf_flag && slice_alf_enabled_flag`. Selects the ALF APS
    /// (`adaptation_parameter_set_id` 0..=31) the §8.9 luma path consults.
    /// Round 126: surfaces the spec-mandated APS handle so the decoder can
    /// route slice-by-slice to the right cached APS instead of relying on
    /// the most-recently-parsed one.
    pub slice_alf_luma_aps_id: Option<u8>,
    /// `slice_alf_chroma_aps_id` (§7.3.4, 5-bit), present only when the
    /// ChromaArrayType / `slice_alf_chroma_idc` gating signals one. Used
    /// to route the Cb plane's ALF (and the joint-component Cb branch of
    /// `slice_alf_chroma_idc == 3` on ChromaArrayType 1..2).
    pub slice_alf_chroma_aps_id: Option<u8>,
    /// `slice_alf_chroma2_aps_id` (§7.3.4, 5-bit), present only on
    /// ChromaArrayType == 3 when `slice_alf_chroma_idc ∈ {2, 3}`. Routes
    /// the Cr plane's ALF when the slice signals it independently of Cb.
    pub slice_alf_chroma2_aps_id: Option<u8>,
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
    let mut last_tile_id = None;
    let mut arbitrary_slice_flag = false;
    let mut num_remaining_tiles_in_slice_minus1 = 0;
    let mut delta_tile_id_minus1 = Vec::new();
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
            last_tile_id = Some(br.u(id_bits)?);
        } else {
            num_remaining_tiles_in_slice_minus1 = br.ue()?;
            // §7.4.5: num_remaining_tiles_in_slice_minus1 shall be in the
            // range 0 to NumTilesInPic − 1, inclusive.
            let num_tiles_in_pic =
                (ctx.num_tile_columns_minus1 + 1) * (ctx.num_tile_rows_minus1 + 1);
            if num_remaining_tiles_in_slice_minus1 > num_tiles_in_pic.saturating_sub(1) {
                return Err(Error::invalid(format!(
                    "evc slice: num_remaining_tiles_in_slice_minus1 \
                     {num_remaining_tiles_in_slice_minus1} > NumTilesInPic - 1 \
                     ({num_tiles_in_pic} tiles, §7.4.5)"
                )));
            }
            // §7.3.4: the loop runs i < NumTilesInSlice − 1, with
            // NumTilesInSlice = num_remaining_tiles_in_slice_minus1 + 2
            // (eq. 80) — i.e. minus1 + 1 entries.
            let num_deltas = num_remaining_tiles_in_slice_minus1 + 1;
            delta_tile_id_minus1.reserve_exact(num_deltas as usize);
            for _ in 0..num_deltas {
                delta_tile_id_minus1.push(br.ue()?);
            }
        }
    }
    // §7.4.5: when not present, last_tile_id is inferred equal to
    // first_tile_id (single-tile slices and arbitrary slices).
    let last_tile_id = last_tile_id.unwrap_or(first_tile_id);

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

    // §7.3.4 ALF slice-header block, gated on `sps_alf_flag`. The
    // round-107 parse surfaces the per-CTU-map control fields
    // (`slice_alf_map_flag`, `slice_alf_chroma_idc` + its derived
    // enable flags, and the `ChromaArrayType == 3`-only chroma map
    // flags) so the `coding_tree_unit()` walker can decode
    // `alf_ctb_flag` / `alf_ctb_chroma_flag` / `alf_ctb_chroma2_flag`
    // per §7.3.8.2 lines 2626-2631.
    let mut slice_alf_enabled_flag = false;
    let mut slice_alf_map_flag = false;
    let mut slice_alf_chroma_idc = 0u32;
    let mut slice_alf_chroma_map_flag = false;
    let mut slice_alf_chroma2_map_flag = false;
    let mut slice_alf_luma_aps_id: Option<u8> = None;
    let mut slice_alf_chroma_aps_id: Option<u8> = None;
    let mut slice_alf_chroma2_aps_id: Option<u8> = None;
    if ctx.sps_alf_flag {
        slice_alf_enabled_flag = br.u1()? != 0;
        if slice_alf_enabled_flag {
            slice_alf_luma_aps_id = Some(br.u(5)? as u8);
            slice_alf_map_flag = br.u1()? != 0;
            slice_alf_chroma_idc = br.u(2)?;
            if (ctx.chroma_array_type == 1 || ctx.chroma_array_type == 2)
                && slice_alf_chroma_idc > 0
            {
                slice_alf_chroma_aps_id = Some(br.u(5)? as u8);
            }
        }
        // §7.4.5: when ChromaArrayType == 0, slice_alf_chroma_idc shall
        // be 0 (bitstream conformance). For ChromaArrayType == 3 the
        // chroma idc may be re-signalled when ALF is luma-disabled, and
        // the per-component chroma map flags + APS ids are present.
        if ctx.chroma_array_type == 3 {
            if !slice_alf_enabled_flag {
                slice_alf_chroma_idc = br.u(2)?;
            }
            // §7.4.5 derived: sliceChromaAlfEnabledFlag set for idc 1/3.
            if slice_alf_chroma_idc == 1 || slice_alf_chroma_idc == 3 {
                slice_alf_chroma_aps_id = Some(br.u(5)? as u8);
                slice_alf_chroma_map_flag = br.u1()? != 0;
            }
            // §7.4.5 derived: sliceChroma2AlfEnabledFlag set for idc 2/3.
            if slice_alf_chroma_idc == 2 || slice_alf_chroma_idc == 3 {
                slice_alf_chroma2_aps_id = Some(br.u(5)? as u8);
                slice_alf_chroma2_map_flag = br.u1()? != 0;
            }
        }
    }
    // §7.4.5 derived variables (independent of ChromaArrayType).
    let slice_chroma_alf_enabled_flag = slice_alf_chroma_idc == 1 || slice_alf_chroma_idc == 3;
    let slice_chroma2_alf_enabled_flag = slice_alf_chroma_idc == 2 || slice_alf_chroma_idc == 3;

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
        delta_tile_id_minus1,
        slice_type,
        no_output_of_prior_pics_flag,
        mmvd_group_enable_flag,
        slice_alf_enabled_flag,
        slice_alf_map_flag,
        slice_alf_chroma_idc,
        slice_chroma_alf_enabled_flag,
        slice_chroma2_alf_enabled_flag,
        slice_alf_chroma_map_flag,
        slice_alf_chroma2_map_flag,
        slice_alf_luma_aps_id,
        slice_alf_chroma_aps_id,
        slice_alf_chroma2_aps_id,
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

/// §7.4.5 eq. (78) outputs — the per-slice tile-grid dimensions of an
/// `arbitrary_slice_flag == 0` (rectangular) slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceTileDims {
    /// `numTileRowsInSlice`.
    pub num_tile_rows_in_slice: u32,
    /// `numTileColumnsInSlice`.
    pub num_tile_columns_in_slice: u32,
    /// `NumTilesInSlice = numTileRowsInSlice * numTileColumnsInSlice`.
    pub num_tiles_in_slice: u32,
}

/// The `TileIdToIdx[ tileId ]` lookup with the §7.4.5 well-formedness
/// expectation surfaced as an error: every slice-signalled tile ID must
/// name a tile of the picture.
fn tile_idx_for_id_checked(maps: &TileIndexMaps, tile_id: u32, role: &str) -> Result<u32> {
    maps.tile_idx_for_id(tile_id).ok_or_else(|| {
        Error::invalid(format!(
            "evc slice: {role} {tile_id} names no tile in the picture (§7.4.5 TileIdToIdx)"
        ))
    })
}

/// §7.4.5 eq. (78) — `numTileRowsInSlice` / `numTileColumnsInSlice` /
/// `NumTilesInSlice` for an `arbitrary_slice_flag == 0` slice.
///
/// ```text
/// firstTileIdx = TileIdToIdx[ first_tile_id ]
/// firstTileColumnIdx = firstTileIdx % ( num_tile_columns_minus1 + 1 )
/// lastTileIdx = TileIdToIdx[ last_tile_id ]
/// lastTileColumnIdx = lastTileIdx % ( num_tile_columns_minus1 + 1 )
/// deltaTileIdx = lastTileIdx − firstTileIdx
/// if( lastTileIdx < firstTileIdx ) {
///     if( firstTileColumnIdx > lastTileColumnIdx )
///         deltaTileIdx += NumTilesInPic + num_tile_columns_minus1 + 1
///     else
///         deltaTileIdx += NumTilesInPic                                (78)
/// } else if( firstTileColumnIdx > lastTileColumnIdx )
///     deltaTileIdx += num_tile_columns_minus1 + 1
/// numTileRowsInSlice = ( deltaTileIdx / ( num_tile_columns_minus1 + 1 ) ) + 1
/// numTileColumnsInSlice = ( deltaTileIdx % ( num_tile_columns_minus1 + 1 ) ) + 1
/// NumTilesInSlice = numTileRowsInSlice * numTileColumnsInSlice
/// ```
///
/// The two `+=` arms wrap the slice's tile rectangle around the bottom
/// (`lastTileIdx < firstTileIdx`) and/or right (`firstTileColumnIdx >
/// lastTileColumnIdx`) picture edges. `deltaTileIdx` is carried in `i64`
/// because it is transiently negative on the row-wrap path.
pub fn compute_slice_tile_dims(
    first_tile_id: u32,
    last_tile_id: u32,
    tile_index_maps: &TileIndexMaps,
    num_tile_columns_minus1: u32,
    num_tiles_in_pic: u32,
) -> Result<SliceTileDims> {
    let cols = i64::from(num_tile_columns_minus1) + 1;
    let first_tile_idx = i64::from(tile_idx_for_id_checked(
        tile_index_maps,
        first_tile_id,
        "first_tile_id",
    )?);
    let first_tile_column_idx = first_tile_idx % cols;
    let last_tile_idx = i64::from(tile_idx_for_id_checked(
        tile_index_maps,
        last_tile_id,
        "last_tile_id",
    )?);
    let last_tile_column_idx = last_tile_idx % cols;

    let mut delta_tile_idx = last_tile_idx - first_tile_idx;
    if last_tile_idx < first_tile_idx {
        if first_tile_column_idx > last_tile_column_idx {
            delta_tile_idx += i64::from(num_tiles_in_pic) + cols;
        } else {
            delta_tile_idx += i64::from(num_tiles_in_pic);
        }
    } else if first_tile_column_idx > last_tile_column_idx {
        delta_tile_idx += cols;
    }

    let num_tile_rows_in_slice = (delta_tile_idx / cols + 1) as u32;
    let num_tile_columns_in_slice = (delta_tile_idx % cols + 1) as u32;
    Ok(SliceTileDims {
        num_tile_rows_in_slice,
        num_tile_columns_in_slice,
        num_tiles_in_slice: num_tile_rows_in_slice * num_tile_columns_in_slice,
    })
}

/// §7.4.5 eq. (79) — `SliceTileIdx[ i ]`, the tile index of the i-th
/// tile of an `arbitrary_slice_flag == 0` slice, in slice-tile order.
///
/// ```text
/// tileIdx = TileIdToIdx[ first_tile_id ]
/// for( j = 0, cIdx = 0; j < numTileRowsInSlice; j++, tileIdx += num_tile_columns_minus1 + 1 ) {
///     tileIdx = tileIdx % NumTilesInPic
///     for( i = 0, currTileIdx = tileIdx; i < numTileColumnsInSlice; i++, currTileIdx++, cIdx++ ) {
///         if( currTileIdx / ( num_tile_columns_minus1 + 1 ) >
///                            tileIdx / ( num_tile_columns_minus1 + 1 ) )      (79)
///             SliceTileIdx[ cIdx ] = currTileIdx − ( num_tile_columns_minus1 + 1 )
///         else
///             SliceTileIdx[ cIdx ] = currTileIdx
///     }
/// }
/// ```
///
/// The `tileIdx % NumTilesInPic` at each row-loop head realises the
/// bottom-edge wrap; the inner-row branch (`currTileIdx` crossing into
/// the next tile row) realises the right-edge wrap by stepping back one
/// full row of tiles. The subtraction never underflows: the branch is
/// only taken when `currTileIdx` is at least one full row past zero.
pub fn compute_slice_tile_indices(
    first_tile_id: u32,
    tile_index_maps: &TileIndexMaps,
    num_tile_columns_minus1: u32,
    num_tiles_in_pic: u32,
    dims: &SliceTileDims,
) -> Result<Vec<u32>> {
    if num_tiles_in_pic == 0 {
        return Err(Error::invalid(
            "evc slice: NumTilesInPic == 0 in SliceTileIdx derivation (§7.4.5 eq. 79)",
        ));
    }
    let cols = num_tile_columns_minus1 + 1;
    let mut tile_idx = tile_idx_for_id_checked(tile_index_maps, first_tile_id, "first_tile_id")?;
    let mut slice_tile_idx = Vec::with_capacity(dims.num_tiles_in_slice as usize);
    for _j in 0..dims.num_tile_rows_in_slice {
        tile_idx %= num_tiles_in_pic;
        // eq. (79) inner loop: currTileIdx runs from tileIdx for
        // numTileColumnsInSlice steps.
        for i in 0..dims.num_tile_columns_in_slice {
            let curr_tile_idx = tile_idx + i;
            if curr_tile_idx / cols > tile_idx / cols {
                slice_tile_idx.push(curr_tile_idx - cols);
            } else {
                slice_tile_idx.push(curr_tile_idx);
            }
        }
        tile_idx += cols;
    }
    Ok(slice_tile_idx)
}

/// §7.4.5 eq. (80) — `NumTilesInSlice` for an `arbitrary_slice_flag == 1`
/// slice: `num_remaining_tiles_in_slice_minus1 + 2`.
pub fn compute_num_tiles_in_slice_arbitrary(num_remaining_tiles_in_slice_minus1: u32) -> u32 {
    num_remaining_tiles_in_slice_minus1.saturating_add(2)
}

/// §7.4.5 eq. (81) / (82) — `SliceTileIdx[ i ]` for an
/// `arbitrary_slice_flag == 1` slice.
///
/// ```text
/// sliceTileId[ i ] = i > 0 ? ( sliceTileId[ i − 1 ] + delta_tile_id_minus1[ i − 1 ] + 1 )
///                          : first_tile_id                                    (81)
/// SliceTileIdx[ i ] = TileIdToIdx[ sliceTileId[ i ] ]                         (82)
/// ```
///
/// `i` ranges over `0 ..= NumTilesInSlice − 1`, so the output carries
/// `delta_tile_id_minus1.len() + 1` entries. (The printed eq. (82) reads
/// "liceTileIdx" — a dropped leading character of `SliceTileIdx`, the
/// variable eq. (81)'s prose sentence introduces.) The running tile ID
/// uses saturating adds; a malformed delta that pushes the ID past every
/// picture tile surfaces as the eq. (82) lookup error.
pub fn compute_slice_tile_indices_arbitrary(
    first_tile_id: u32,
    delta_tile_id_minus1: &[u32],
    tile_index_maps: &TileIndexMaps,
) -> Result<Vec<u32>> {
    let mut slice_tile_id = first_tile_id;
    let mut slice_tile_idx = Vec::with_capacity(delta_tile_id_minus1.len() + 1);
    slice_tile_idx.push(tile_idx_for_id_checked(
        tile_index_maps,
        slice_tile_id,
        "first_tile_id",
    )?);
    for &delta in delta_tile_id_minus1 {
        slice_tile_id = slice_tile_id.saturating_add(delta).saturating_add(1);
        slice_tile_idx.push(tile_idx_for_id_checked(
            tile_index_maps,
            slice_tile_id,
            "sliceTileId",
        )?);
    }
    Ok(slice_tile_idx)
}

impl SliceHeader {
    /// `NumTilesInSlice` (§7.4.5) — eq. (78) product for rectangular
    /// slices (covering the single-tile case via the `last_tile_id =
    /// first_tile_id` inference), eq. (80) for arbitrary slices.
    pub fn num_tiles_in_slice(
        &self,
        tile_index_maps: &TileIndexMaps,
        num_tile_columns_minus1: u32,
        num_tiles_in_pic: u32,
    ) -> Result<u32> {
        if self.arbitrary_slice_flag {
            Ok(compute_num_tiles_in_slice_arbitrary(
                self.num_remaining_tiles_in_slice_minus1,
            ))
        } else {
            Ok(compute_slice_tile_dims(
                self.first_tile_id,
                self.last_tile_id,
                tile_index_maps,
                num_tile_columns_minus1,
                num_tiles_in_pic,
            )?
            .num_tiles_in_slice)
        }
    }

    /// `SliceTileIdx[ i ]` (§7.4.5) — eq. (78)+(79) for rectangular
    /// slices, eq. (81)+(82) for arbitrary slices. `tile_index_maps` is
    /// the active PPS's §6.5.1 eq. (32) output
    /// ([`crate::pps::Pps::tile_index_maps`]); `num_tiles_in_pic` is
    /// `NumTilesInPic` ([`crate::pps::Pps::num_tiles_in_pic`]).
    pub fn slice_tile_indices(
        &self,
        tile_index_maps: &TileIndexMaps,
        num_tile_columns_minus1: u32,
        num_tiles_in_pic: u32,
    ) -> Result<Vec<u32>> {
        if self.arbitrary_slice_flag {
            compute_slice_tile_indices_arbitrary(
                self.first_tile_id,
                &self.delta_tile_id_minus1,
                tile_index_maps,
            )
        } else {
            let dims = compute_slice_tile_dims(
                self.first_tile_id,
                self.last_tile_id,
                tile_index_maps,
                num_tile_columns_minus1,
                num_tiles_in_pic,
            )?;
            compute_slice_tile_indices(
                self.first_tile_id,
                tile_index_maps,
                num_tile_columns_minus1,
                num_tiles_in_pic,
                &dims,
            )
        }
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

    /// Round 107 — §7.3.4 ALF slice-header block surfaces the per-CTU
    /// map controls. A Baseline (ChromaArrayType == 1) IDR slice with
    /// `slice_alf_enabled_flag = 1`, `slice_alf_map_flag = 1`,
    /// `slice_alf_chroma_idc = 3` (both Cb and Cr): the parser consumes
    /// the luma APS id + map flag + chroma idc + the (idc > 0) chroma
    /// APS id, surfaces the derived enable flags, and leaves the chroma
    /// map flags inferred 0 (those are ChromaArrayType == 3-only).
    #[test]
    fn round107_alf_map_fields_baseline_chroma1() {
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_alf_flag: true,
            chroma_array_type: 1,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.ue(2); // slice_type = I
        e.u(1, 0); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_alf_enabled_flag
        e.u(5, 7); // slice_alf_luma_aps_id
        e.u(1, 1); // slice_alf_map_flag
        e.u(2, 3); // slice_alf_chroma_idc = 3 (both)
        e.u(5, 4); // slice_alf_chroma_aps_id (idc > 0, ChromaArrayType 1)
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0); // cb offset
        e.ue(0); // cr offset
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &ctx).unwrap();
        assert!(sh.slice_alf_enabled_flag);
        assert!(sh.slice_alf_map_flag);
        assert_eq!(sh.slice_alf_chroma_idc, 3);
        assert!(sh.slice_chroma_alf_enabled_flag);
        assert!(sh.slice_chroma2_alf_enabled_flag);
        // ChromaArrayType != 3 ⇒ chroma map flags inferred 0.
        assert!(!sh.slice_alf_chroma_map_flag);
        assert!(!sh.slice_alf_chroma2_map_flag);
        // Round 126: APS ids are surfaced. Luma id is the u(5) field
        // following slice_alf_enabled_flag; chroma id is the joint id
        // signalled when ChromaArrayType ∈ {1, 2} and idc > 0. ChromaArray-
        // Type ≠ 3 so the second chroma APS id stays unset.
        assert_eq!(sh.slice_alf_luma_aps_id, Some(7));
        assert_eq!(sh.slice_alf_chroma_aps_id, Some(4));
        assert!(sh.slice_alf_chroma2_aps_id.is_none());
        // Trailing fields still parse, proving bit alignment held.
        assert!(sh.slice_deblocking_filter_flag);
        assert_eq!(sh.slice_qp, 26);
    }

    /// Round 126 — ChromaArrayType == 3 with `slice_alf_chroma_idc = 3`
    /// signals BOTH a Cb APS id and a Cr APS id (independent of the luma
    /// id), and BOTH per-component chroma map flags. The parser surfaces
    /// all three APS ids so the decoder can route to three distinct ALF
    /// APS slots on a 4:4:4 slice.
    #[test]
    fn round126_alf_aps_ids_chroma444_three_apsids() {
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_alf_flag: true,
            chroma_array_type: 3,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.ue(2); // slice_type = I
        e.u(1, 0); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_alf_enabled_flag
        e.u(5, 11); // slice_alf_luma_aps_id
        e.u(1, 0); // slice_alf_map_flag
        e.u(2, 3); // slice_alf_chroma_idc = 3 (both Cb and Cr)
                   // §7.4.5: on ChromaArrayType != 3 the chroma APS id is
                   // signalled inside the enabled-flag block. On ChromaArrayType
                   // == 3 it is signalled per-component below instead, so the
                   // enabled-flag-block chroma id read is SKIPPED in this path.
                   //
                   // The ChromaArrayType == 3 fork re-reads (in this case
                   // re-skips) chroma_idc, then reads chroma map flags + APS
                   // ids per component.
        e.u(5, 13); // slice_alf_chroma_aps_id  (idc ∈ {1, 3} → Cb path)
        e.u(1, 1); // slice_alf_chroma_map_flag
        e.u(5, 17); // slice_alf_chroma2_aps_id (idc ∈ {2, 3} → Cr path)
        e.u(1, 0); // slice_alf_chroma2_map_flag
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &ctx).unwrap();
        assert_eq!(sh.slice_alf_luma_aps_id, Some(11));
        assert_eq!(sh.slice_alf_chroma_aps_id, Some(13));
        assert_eq!(sh.slice_alf_chroma2_aps_id, Some(17));
        assert!(sh.slice_alf_chroma_map_flag);
        assert!(!sh.slice_alf_chroma2_map_flag);
        // Trailing fields still align.
        assert!(sh.slice_deblocking_filter_flag);
        assert_eq!(sh.slice_qp, 26);
    }

    /// Round 126 — When `slice_alf_enabled_flag = 0` (and ChromaArrayType
    /// != 3) no APS id is signalled and every `slice_alf_*_aps_id`
    /// surfaces as `None`. This is the path every non-ALF Baseline IDR
    /// takes today.
    #[test]
    fn round126_alf_aps_ids_unset_when_disabled() {
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_alf_flag: true,
            chroma_array_type: 1,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.ue(2); // I
        e.u(1, 0); // no_output_of_prior_pics_flag
        e.u(1, 0); // slice_alf_enabled_flag = 0
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 28);
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let sh = parse(&e.into_bytes(), NalUnitType::Idr, &ctx).unwrap();
        assert!(!sh.slice_alf_enabled_flag);
        assert!(sh.slice_alf_luma_aps_id.is_none());
        assert!(sh.slice_alf_chroma_aps_id.is_none());
        assert!(sh.slice_alf_chroma2_aps_id.is_none());
    }

    /// `slice_alf_chroma_idc = 0` derives both chroma enable flags off,
    /// and the chroma APS id is not in the bitstream (so `slice_qp`
    /// still lands at the right offset).
    #[test]
    fn round107_alf_map_chroma_idc_zero() {
        let ctx = SliceParseContext {
            single_tile_in_pic_flag: true,
            sps_alf_flag: true,
            chroma_array_type: 1,
            ..Default::default()
        };
        let mut e = BitEmitter::new();
        e.ue(0);
        e.ue(2); // I
        e.u(1, 0); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_alf_enabled_flag
        e.u(5, 0); // luma aps id
        e.u(1, 0); // slice_alf_map_flag = 0
        e.u(2, 0); // slice_alf_chroma_idc = 0 → no chroma aps id
        e.u(1, 0); // slice_deblocking_filter_flag
        e.u(6, 30); // slice_qp
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &ctx).unwrap();
        assert!(sh.slice_alf_enabled_flag);
        assert!(!sh.slice_alf_map_flag);
        assert_eq!(sh.slice_alf_chroma_idc, 0);
        assert!(!sh.slice_chroma_alf_enabled_flag);
        assert!(!sh.slice_chroma2_alf_enabled_flag);
        assert_eq!(sh.slice_qp, 30);
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

    // ------------------------------------------------------------------
    // Round 281 — §7.4.5 eq. (78)-(82) slice-tile resolution over the
    // §6.5.1 eq. (32) TileIdToIdx map.
    // ------------------------------------------------------------------

    /// An implicit-tile-ID picture (`explicit_tile_id_flag == 0`): the
    /// eq. (30) implicit branch makes `TileId[ ] == tileIdx`, so the
    /// eq. (32) set is the identity over `0 .. num_tiles`.
    fn implicit_id_maps(num_tiles: u32) -> TileIndexMaps {
        TileIndexMaps {
            tile_id_to_idx: (0..num_tiles).map(|k| (k, k)).collect(),
            // FirstCtbAddrTs is irrelevant to eq. (78)-(82); a 1-CTB-per-
            // tile placeholder keeps the struct well-formed.
            first_ctb_addr_ts: (0..num_tiles).collect(),
        }
    }

    #[test]
    fn round281_eq78_single_tile_slice_is_one_by_one() {
        // last_tile_id inferred equal to first_tile_id → deltaTileIdx = 0.
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(2, 2, &maps, 2, 6).unwrap();
        assert_eq!(dims.num_tile_rows_in_slice, 1);
        assert_eq!(dims.num_tile_columns_in_slice, 1);
        assert_eq!(dims.num_tiles_in_slice, 1);
    }

    #[test]
    fn round281_eq78_full_picture_2x2() {
        let maps = implicit_id_maps(4);
        let dims = compute_slice_tile_dims(0, 3, &maps, 1, 4).unwrap();
        assert_eq!(dims.num_tile_rows_in_slice, 2);
        assert_eq!(dims.num_tile_columns_in_slice, 2);
        assert_eq!(dims.num_tiles_in_slice, 4);
    }

    #[test]
    fn round281_eq78_sub_rectangle_3x2() {
        // 3-column × 2-row grid, slice from tileIdx 1 to tileIdx 5:
        // deltaTileIdx = 4 (no wrap arm taken) → 2 rows × 2 columns.
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(1, 5, &maps, 2, 6).unwrap();
        assert_eq!(dims.num_tile_rows_in_slice, 2);
        assert_eq!(dims.num_tile_columns_in_slice, 2);
        assert_eq!(dims.num_tiles_in_slice, 4);
    }

    #[test]
    fn round281_eq78_row_wrap_adds_num_tiles_in_pic() {
        // lastTileIdx (1) < firstTileIdx (4), equal column indices:
        // deltaTileIdx = 1 − 4 + 6 = 3 → 2 rows × 1 column. Pins the
        // bottom-edge wrap arm (the transiently-negative i64 path).
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(4, 1, &maps, 2, 6).unwrap();
        assert_eq!(dims.num_tile_rows_in_slice, 2);
        assert_eq!(dims.num_tile_columns_in_slice, 1);
        assert_eq!(dims.num_tiles_in_slice, 2);
    }

    #[test]
    fn round281_eq78_column_wrap_adds_one_tile_row() {
        // firstTileColumnIdx (1) > lastTileColumnIdx (0) with
        // lastTileIdx ≥ firstTileIdx: deltaTileIdx = 2 + 3 = 5 →
        // 2 rows × 3 columns. Pins the right-edge wrap arm.
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(1, 3, &maps, 2, 6).unwrap();
        assert_eq!(dims.num_tile_rows_in_slice, 2);
        assert_eq!(dims.num_tile_columns_in_slice, 3);
        assert_eq!(dims.num_tiles_in_slice, 6);
    }

    #[test]
    fn round281_eq78_unknown_tile_id_errors() {
        let maps = implicit_id_maps(4);
        assert!(compute_slice_tile_dims(9, 0, &maps, 1, 4).is_err());
        assert!(compute_slice_tile_dims(0, 9, &maps, 1, 4).is_err());
    }

    #[test]
    fn round281_eq79_full_picture_2x2_identity() {
        let maps = implicit_id_maps(4);
        let dims = compute_slice_tile_dims(0, 3, &maps, 1, 4).unwrap();
        let idx = compute_slice_tile_indices(0, &maps, 1, 4, &dims).unwrap();
        assert_eq!(idx, vec![0, 1, 2, 3]);
    }

    #[test]
    fn round281_eq79_sub_rectangle_3x2() {
        // 2×2 slice rectangle anchored at tileIdx 1 in a 3-column grid:
        // row 0 covers tiles 1, 2; row 1 covers tiles 4, 5.
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(1, 5, &maps, 2, 6).unwrap();
        let idx = compute_slice_tile_indices(1, &maps, 2, 6, &dims).unwrap();
        assert_eq!(idx, vec![1, 2, 4, 5]);
    }

    #[test]
    fn round281_eq79_column_wrap_steps_back_one_row() {
        // The right-edge wrap of round281_eq78_column_wrap_adds_one_tile_row:
        // when currTileIdx crosses into the next tile row, eq. (79)
        // steps it back by one full row (− 3), wrapping the slice's
        // columns around the right picture edge.
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(1, 3, &maps, 2, 6).unwrap();
        let idx = compute_slice_tile_indices(1, &maps, 2, 6, &dims).unwrap();
        assert_eq!(idx, vec![1, 2, 0, 4, 5, 3]);
    }

    #[test]
    fn round281_eq79_row_wrap_mods_into_picture() {
        // The bottom-edge wrap of round281_eq78_row_wrap…: the second
        // row's tileIdx (4 + 3 = 7) is reduced mod NumTilesInPic to 1.
        let maps = implicit_id_maps(6);
        let dims = compute_slice_tile_dims(4, 1, &maps, 2, 6).unwrap();
        let idx = compute_slice_tile_indices(4, &maps, 2, 6, &dims).unwrap();
        assert_eq!(idx, vec![4, 1]);
    }

    #[test]
    fn round281_eq79_count_matches_eq78_and_values_are_in_pic() {
        // Sweep every (first, last) pair on the 3×2 implicit grid: the
        // eq. (79) list always carries NumTilesInSlice tile indices
        // inside the picture, starting at firstTileIdx. Distinctness is
        // additionally pinned for every pair whose eq. (78) rectangle
        // fits the tile grid (a double-edge wrap pair like first = 2,
        // last = 0 derives 3 slice rows in the 2-row picture — a
        // non-conformant slice whose walk necessarily revisits tiles).
        let maps = implicit_id_maps(6);
        for first in 0..6u32 {
            for last in 0..6u32 {
                let dims = compute_slice_tile_dims(first, last, &maps, 2, 6).unwrap();
                let idx = compute_slice_tile_indices(first, &maps, 2, 6, &dims).unwrap();
                assert_eq!(idx.len() as u32, dims.num_tiles_in_slice);
                assert_eq!(idx[0], first, "slice walk starts at firstTileIdx");
                assert!(idx.iter().all(|&t| t < 6), "tile indices stay in-pic");
                if dims.num_tile_rows_in_slice <= 2 && dims.num_tile_columns_in_slice <= 3 {
                    let mut seen = idx.clone();
                    seen.sort_unstable();
                    seen.dedup();
                    assert_eq!(seen.len(), idx.len(), "tile indices are distinct");
                }
            }
        }
    }

    #[test]
    fn round281_eq80_adds_two() {
        assert_eq!(compute_num_tiles_in_slice_arbitrary(0), 2);
        assert_eq!(compute_num_tiles_in_slice_arbitrary(3), 5);
    }

    #[test]
    fn round281_eq81_eq82_explicit_ids_resolve_in_order() {
        // Explicit tile IDs 7 / 9 / 12 at tile indices 0 / 1 / 2.
        // first_tile_id = 7, deltas [1, 2] → sliceTileId 7, 9, 12 →
        // SliceTileIdx [0, 1, 2].
        let maps = TileIndexMaps {
            tile_id_to_idx: vec![(7, 0), (9, 1), (12, 2)],
            first_ctb_addr_ts: vec![0, 4, 8],
        };
        let idx = compute_slice_tile_indices_arbitrary(7, &[1, 2], &maps).unwrap();
        assert_eq!(idx, vec![0, 1, 2]);
    }

    #[test]
    fn round281_eq82_unknown_slice_tile_id_errors() {
        let maps = implicit_id_maps(4);
        // first OK, but 0 + (5 + 1) = 6 names no tile.
        assert!(compute_slice_tile_indices_arbitrary(0, &[5], &maps).is_err());
        // unknown first_tile_id.
        assert!(compute_slice_tile_indices_arbitrary(9, &[], &maps).is_err());
    }

    fn tiled_ctx() -> SliceParseContext {
        SliceParseContext {
            single_tile_in_pic_flag: false,
            arbitrary_slice_present_flag: true,
            tile_id_len_minus1: 3, // 4-bit tile IDs
            num_tile_columns_minus1: 2,
            num_tile_rows_minus1: 1, // 3×2 = 6 tiles
            chroma_array_type: 1,
            ..Default::default()
        }
    }

    #[test]
    fn round281_parser_arbitrary_slice_reads_num_tiles_minus_one_deltas() {
        // §7.3.4: the delta loop runs i < NumTilesInSlice − 1 =
        // num_remaining_tiles_in_slice_minus1 + 1 times (eq. 80). With
        // minus1 = 1 the header carries TWO deltas; the parser must
        // consume both for the trailing fields to line up.
        let mut e = BitEmitter::new();
        e.ue(0); // slice_pic_parameter_set_id
        e.u(1, 0); // single_tile_in_slice_flag = 0
        e.u(4, 0); // first_tile_id = 0
        e.u(1, 1); // arbitrary_slice_flag = 1
        e.ue(1); // num_remaining_tiles_in_slice_minus1 = 1
        e.ue(1); // delta_tile_id_minus1[0] = 1
        e.ue(0); // delta_tile_id_minus1[1] = 0
        e.ue(2); // slice_type = I
        e.u(1, 1); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0); // slice_cb_qp_offset (se: 0)
        e.ue(0); // slice_cr_qp_offset
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &tiled_ctx()).unwrap();
        assert!(sh.arbitrary_slice_flag);
        assert_eq!(sh.num_remaining_tiles_in_slice_minus1, 1);
        assert_eq!(sh.delta_tile_id_minus1, vec![1, 0]);
        // Trailing fields stayed aligned — the old minus1-count read
        // would have shifted everything after the delta loop.
        assert_eq!(sh.slice_type, SliceType::I);
        assert_eq!(sh.slice_qp, 26);
        // last_tile_id not present → inferred equal to first_tile_id.
        assert_eq!(sh.last_tile_id, sh.first_tile_id);
    }

    #[test]
    fn round281_parser_rejects_num_remaining_tiles_over_pic() {
        // §7.4.5: minus1 shall be ≤ NumTilesInPic − 1 (= 5 here).
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.u(1, 0); // single_tile_in_slice_flag = 0
        e.u(4, 0); // first_tile_id
        e.u(1, 1); // arbitrary_slice_flag = 1
        e.ue(6); // num_remaining_tiles_in_slice_minus1 = 6 > 5
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        assert!(parse(&rbsp, NalUnitType::Idr, &tiled_ctx()).is_err());
    }

    #[test]
    fn round281_parser_infers_last_tile_id_for_single_tile_slice() {
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.u(1, 1); // single_tile_in_slice_flag = 1
        e.u(4, 3); // first_tile_id = 3
        e.ue(2); // slice_type = I
        e.u(1, 1); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0); // cb offset
        e.ue(0); // cr offset
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &tiled_ctx()).unwrap();
        assert!(sh.single_tile_in_slice_flag);
        assert_eq!(sh.first_tile_id, 3);
        assert_eq!(sh.last_tile_id, 3, "§7.4.5 inference");
        assert!(sh.delta_tile_id_minus1.is_empty());
    }

    #[test]
    fn round281_dispatch_rectangular_matches_free_functions() {
        // A parsed rectangular multi-tile slice: first_tile_id 1,
        // last_tile_id 5 on the 3×2 implicit grid.
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.u(1, 0); // single_tile_in_slice_flag = 0
        e.u(4, 1); // first_tile_id = 1
        e.u(1, 0); // arbitrary_slice_flag = 0
        e.u(4, 5); // last_tile_id = 5
        e.ue(2); // slice_type = I
        e.u(1, 1); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &tiled_ctx()).unwrap();
        let maps = implicit_id_maps(6);
        assert_eq!(sh.num_tiles_in_slice(&maps, 2, 6).unwrap(), 4);
        assert_eq!(
            sh.slice_tile_indices(&maps, 2, 6).unwrap(),
            vec![1, 2, 4, 5]
        );
    }

    #[test]
    fn round281_dispatch_arbitrary_matches_free_functions() {
        let maps = implicit_id_maps(6);
        let mut e = BitEmitter::new();
        e.ue(0); // pps id
        e.u(1, 0); // single_tile_in_slice_flag = 0
        e.u(4, 0); // first_tile_id = 0
        e.u(1, 1); // arbitrary_slice_flag = 1
        e.ue(1); // num_remaining_tiles_in_slice_minus1 = 1 → 3 tiles
        e.ue(1); // delta[0] = 1 → sliceTileId 2
        e.ue(2); // delta[1] = 2 → sliceTileId 5
        e.ue(2); // slice_type = I
        e.u(1, 1); // no_output_of_prior_pics_flag
        e.u(1, 1); // slice_deblocking_filter_flag
        e.u(6, 26); // slice_qp
        e.ue(0);
        e.ue(0);
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let sh = parse(&rbsp, NalUnitType::Idr, &tiled_ctx()).unwrap();
        assert_eq!(
            sh.num_tiles_in_slice(&maps, 2, 6).unwrap(),
            compute_num_tiles_in_slice_arbitrary(1)
        );
        assert_eq!(sh.slice_tile_indices(&maps, 2, 6).unwrap(), vec![0, 2, 5]);
    }

    #[test]
    fn round281_eq79_zero_tiles_in_pic_errors() {
        let maps = implicit_id_maps(1);
        let dims = SliceTileDims {
            num_tile_rows_in_slice: 1,
            num_tile_columns_in_slice: 1,
            num_tiles_in_slice: 1,
        };
        assert!(compute_slice_tile_indices(0, &maps, 0, 0, &dims).is_err());
    }
}
