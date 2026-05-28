//! EVC Dynamic Range Adjustment (ISO/IEC 23094-1 §8.10 / §7.3.6).
//!
//! DRA is an in-loop post-processing step that applies a piecewise-linear
//! luminance re-mapping to decoded luma samples, and a derived offset to
//! chroma samples. The mapping table is signalled via the APS
//! (`aps_params_type == 1`) as a set of 8 segment boundaries with per-segment
//! scale and offset pairs.
//!
//! ## APS payload (§7.3.6)
//!
//! ```text
//! dra_data() {
//!   dra_descriptor_present_flag         u(1)
//!   if (dra_descriptor_present_flag) {
//!     dra_num_ranges_minus1             u(4)   // 0..15 → 1..16 ranges
//!     for each range i (0..=dra_num_ranges_minus1) {
//!       dra_range_l[i]                  u(10)  // luma boundary, 10 bits
//!     }
//!     dra_scale[0]                      u(11)  // Q8.3 fixed-point
//!     for i in 1..=num_ranges_minus1 {
//!       dra_scale_delta[i]              i(12)  // signed 12-bit delta
//!     }
//!     dra_luma_inverted_flag            u(1)
//!     for i in 0..num_ranges_minus1 {
//!       dra_chroma_qp_scale[i]          u(8)   // Q4.4
//!       dra_chroma_qp_offset[i]         i(8)   // signed 8-bit
//!     }
//!   }
//! }
//! ```
//!
//! ## Luma mapping table (§8.10.2)
//!
//! The decoder builds a 256-entry lookup table `DraLumaMap[0..255]` from the
//! segment boundaries and scale factors. Within segment `i`:
//!
//! ```text
//! DraLumaMap[v] = DraBaseOut[i] + Round(DraScale[i] * (v − DraRangeL[i]))
//! ```
//!
//! where `DraBaseOut[0] = DraRangeL[0]` and subsequent base outputs are
//! computed by accumulating per-segment lengths scaled by `DraScale[i]`.
//! Values are clipped to `[0, 1023]` (for 10-bit; `[0, 255]` for 8-bit).
//!
//! ## Chroma offset (§8.10.3)
//!
//! Chroma samples are mapped via `Clip3(0, (1<<bd)-1, s + DraChromaOffset)`
//! where `DraChromaOffset` is a per-picture integer derived from the DRA
//! chroma parameters at the segment that contains the average luma value.
//!
//! ## Round-11 scope
//!
//! * Full APS `dra_data()` payload parsing.
//! * 256-entry luma LUT construction from the scale/boundary parameters.
//! * Per-sample luma remapping applied to the full Y plane.
//! * Chroma is offset by the DRA chroma offset derived for segment 0
//!   (full per-segment chroma derivation is deferred).
//!
//! All clause numbers refer to **ISO/IEC 23094-1:2020(E)**.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::picture::YuvPicture;

/// Maximum number of DRA segments supported by the spec (§7.3.6).
/// The APS field `dra_num_ranges_minus1` is 4 bits → 0..=15.
pub const DRA_MAX_RANGES: usize = 16;

/// Parsed `dra_data()` payload from a DRA APS.
#[derive(Clone, Debug)]
pub struct DraData {
    /// Whether the DRA descriptor was signalled in this APS.
    pub descriptor_present: bool,
    /// Number of luma segments (1..=16). Valid when `descriptor_present`.
    pub num_ranges: usize,
    /// Luma segment lower boundaries (10-bit values, 0..=1023).
    /// Only `dra_range_l[0..num_ranges]` are valid.
    pub range_l: [u16; DRA_MAX_RANGES],
    /// Per-segment DRA scale values (Q8.3 fixed-point: integer value
    /// represents `scale / 8.0`). Derived from `dra_scale[0]` and
    /// successive `dra_scale_delta[i]` additions.
    pub scale: [i16; DRA_MAX_RANGES],
    /// When true, the luma mapping is the inverse function (§8.10.2.2).
    pub luma_inverted: bool,
    /// Per-segment chroma QP scale (Q4.4, ignored beyond segment count).
    pub chroma_qp_scale: [u8; DRA_MAX_RANGES],
    /// Per-segment chroma QP offset (signed 8-bit).
    pub chroma_qp_offset: [i8; DRA_MAX_RANGES],
}

impl Default for DraData {
    fn default() -> Self {
        Self {
            descriptor_present: false,
            num_ranges: 1,
            range_l: [0u16; DRA_MAX_RANGES],
            scale: [8i16; DRA_MAX_RANGES], // Q8.3 → scale = 1.0
            luma_inverted: false,
            chroma_qp_scale: [16u8; DRA_MAX_RANGES], // Q4.4 → scale = 1.0
            chroma_qp_offset: [0i8; DRA_MAX_RANGES],
        }
    }
}

/// Parse a `dra_data()` payload from the given byte slice.
///
/// The caller passes the DRA APS payload bytes starting immediately after the
/// 1-byte APS header (`adaptation_parameter_set_id` + `aps_params_type`).
pub fn parse_dra_data(payload: &[u8]) -> Result<DraData> {
    if payload.is_empty() {
        return Err(Error::invalid("evc dra: empty DRA APS payload"));
    }
    let mut br = BitReader::new(payload);
    let mut data = DraData::default();

    let descriptor_present = br.u1()? != 0;
    data.descriptor_present = descriptor_present;
    if !descriptor_present {
        return Ok(data);
    }

    let num_ranges_minus1 = br.u(4)? as usize;
    let num_ranges = num_ranges_minus1 + 1;
    data.num_ranges = num_ranges;

    // dra_range_l[i] for i in 0..=num_ranges_minus1.
    for i in 0..num_ranges {
        data.range_l[i] = br.u(10)? as u16;
    }

    // dra_scale[0]: 11-bit unsigned Q8.3 (represents scale × 8).
    let scale0 = br.u(11)? as i16;
    data.scale[0] = scale0;
    // dra_scale_delta[i] for i in 1..=num_ranges_minus1: signed 12-bit.
    for i in 1..num_ranges {
        let delta_raw = br.u(12)?;
        // Sign-extend 12-bit value.
        let delta = if delta_raw & 0x800 != 0 {
            (delta_raw as i32 | !0xFFF) as i16
        } else {
            delta_raw as i16
        };
        data.scale[i] = data.scale[i - 1].saturating_add(delta);
    }

    data.luma_inverted = br.u1()? != 0;

    // Chroma parameters: one pair per segment (0..num_ranges_minus1).
    // Note: the spec iterates i in 0..num_ranges_minus1 (excluding last).
    for i in 0..num_ranges.saturating_sub(1) {
        data.chroma_qp_scale[i] = br.u(8)? as u8;
        let off_raw = br.u(8)?;
        data.chroma_qp_offset[i] = if off_raw & 0x80 != 0 {
            (off_raw as i32 | !0xFF) as i8
        } else {
            off_raw as i8
        };
    }

    Ok(data)
}

/// Build a 256-entry 8-bit luma lookup table from the DRA parameters.
///
/// Implements §8.10.2 for 8-bit (the 10-bit variant scales by `1023/255`
/// but the 8-bit decoder just clips to `[0, 255]`).
///
/// The piecewise-linear mapping works as follows:
///
/// 1. For each segment `i`, the input range is `[range_l[i], range_l[i+1])`.
///    (The last segment covers `[range_l[num_ranges-1], 255]`.)
/// 2. The output base for segment 0 is `range_l[0]`.
/// 3. Subsequent output bases accumulate: `base_out[i+1] = base_out[i] +
///    scale[i] * (range_l[i+1] − range_l[i]) / 8`.
/// 4. Within a segment, the output for input value `v` is:
///    `clip(base_out[i] + round(scale[i] / 8 * (v − range_l[i])))`.
///
/// The `scale` field stores `scale × 8` (Q8.3), so division by 8 uses
/// integer arithmetic with rounding.
pub fn build_luma_lut(dra: &DraData, bit_depth: u32) -> [u8; 256] {
    let max_val = ((1u32 << bit_depth) - 1) as i32;
    let mut lut = [0u8; 256];

    if !dra.descriptor_present || dra.num_ranges == 0 {
        // Identity mapping.
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i.min(255) as u8;
        }
        return lut;
    }

    // Build output bases for each segment.
    let mut base_out = [0i32; DRA_MAX_RANGES];
    base_out[0] = dra.range_l[0] as i32;
    for i in 0..dra.num_ranges.saturating_sub(1) {
        let width = (dra.range_l[i + 1] as i32) - (dra.range_l[i] as i32);
        // Accumulated output: base + scale/8 * width. Round to nearest.
        let delta = (dra.scale[i] as i32 * width + 4) >> 3;
        base_out[i + 1] = base_out[i] + delta;
    }

    // Fill the LUT.
    for v in 0u32..256 {
        let vin = v as i32;
        // Find which segment v belongs to.
        let seg = find_segment(dra, vin);
        let seg_start = dra.range_l[seg] as i32;
        let base = base_out[seg];
        let scale_q3 = dra.scale[seg] as i32;
        // round(scale/8 * (v - seg_start)) = (scale * (v-seg_start) + 4) >> 3.
        let mapped = base + ((scale_q3 * (vin - seg_start) + 4) >> 3);
        let clipped = mapped.clamp(0, max_val) as u8;
        lut[v as usize] = clipped;
    }

    lut
}

/// Find the segment index for luma value `v` in the DRA range table.
/// The last segment absorbs all values ≥ `range_l[num_ranges-1]`.
fn find_segment(dra: &DraData, v: i32) -> usize {
    for i in (0..dra.num_ranges).rev() {
        if v >= dra.range_l[i] as i32 {
            return i;
        }
    }
    0
}

/// §8.9.5 — Identification of the range index of piecewise function.
///
/// Verbatim transcription of eq. 1383 (ISO/IEC 23094-1:2020(E) page 305):
///
/// ```text
/// rangeFound = 0
/// for( rangeIdx = 0; rangeIdx < numRanges; rangeIdx+ + )
///     if( inputSample < rangesArray[ rangeIdx + 1 ] )
///     {
///          rangeFound = 1                                   (1383)
///          break
///     }
/// rangeIdx = ( rangeFound = = 1 ) ? rangeIdx : numRanges – 1
/// rangeIdx = Min( rangeIdx, numRanges – 1 )
/// ```
///
/// `ranges_array` holds `num_ranges + 1` boundaries — entries 0..num_ranges
/// inclusive — so `ranges_array[rangeIdx + 1]` is always in-bounds for
/// `rangeIdx ∈ [0, numRanges − 1]`. Callers that have only `num_ranges`
/// boundaries (e.g. the round-11 `range_l[]` table) should pass a synthetic
/// upper sentinel as the final entry (e.g. `1 << bit_depth`) so the
/// last-segment fall-through matches the spec's behaviour.
///
/// Returns a value in `[0, num_ranges − 1]`. Defined as a saturating no-op
/// when `num_ranges == 0` (returns 0 — there is no valid range, so the
/// caller is expected to short-circuit before calling).
pub fn find_range_idx(input_sample: i32, ranges_array: &[i32], num_ranges: usize) -> usize {
    if num_ranges == 0 {
        return 0;
    }
    debug_assert!(
        ranges_array.len() > num_ranges,
        "ranges_array must have num_ranges + 1 boundaries (got {} for num_ranges {num_ranges})",
        ranges_array.len()
    );
    for range_idx in 0..num_ranges {
        if input_sample < ranges_array[range_idx + 1] {
            return range_idx.min(num_ranges - 1);
        }
    }
    num_ranges - 1
}

/// Build the §8.9.5-ready `rangesArray` of `num_ranges + 1` boundaries from
/// a [`DraData`] for a given luma bit depth.
///
/// The round-11 [`DraData::range_l`] table stores `num_ranges` lower
/// boundaries (one per segment); §8.9.5's `rangesArray` is one entry longer
/// so that the loop's `inputSample < rangesArray[rangeIdx + 1]` test is
/// in-bounds for `rangeIdx = numRanges − 1`. The synthesised top boundary
/// is `1 << bit_depth` (sample space upper bound) so the last segment
/// absorbs every value ≥ `range_l[num_ranges − 1]`.
pub fn build_ranges_array(dra: &DraData, bit_depth: u32) -> Vec<i32> {
    let mut out = Vec::with_capacity(dra.num_ranges + 1);
    for i in 0..dra.num_ranges {
        out.push(dra.range_l[i] as i32);
    }
    out.push(1i32 << bit_depth);
    out
}

/// Derive the chroma offset for the given DRA segment (§8.10.3).
///
/// The spec derives the chroma offset from the segment that contains the
/// average picture luma. For round-11 we use segment 0's parameters as a
/// conservative approximation.
pub fn chroma_offset_for_segment(dra: &DraData, seg: usize) -> i32 {
    if !dra.descriptor_present || dra.num_ranges == 0 {
        return 0;
    }
    let seg = seg.min(dra.num_ranges.saturating_sub(1));
    dra.chroma_qp_offset[seg] as i32
}

/// Apply the DRA luma LUT to every Y sample of the picture in-place,
/// and apply the per-segment chroma offset to every Cb + Cr sample.
///
/// Per §8.9.2, the chroma DRA process takes the **decoded** (pre-DRA) luma
/// sample at `decPictureL[ x * SubWidthC, y * SubHeightC ]` as one of its
/// inputs (along with the chroma sample). The chroma sample's segment
/// index is therefore derived from the co-located luma sample's value
/// **before** the luma DRA mapping has run, so we must snapshot the
/// pre-DRA luma plane before rewriting it in-place. This is the spec
/// behaviour described by §8.9.4 calling §8.9.6 with `lumaSample =
/// decPictureL[ ... ]`.
///
/// The chroma offset itself is the round-11 simplification of §8.9.6: each
/// chroma sample is offset by the parsed `dra_chroma_qp_offset` of the
/// segment that the co-located luma sample falls into (looked up via the
/// spec-faithful §8.9.5 [`find_range_idx`] helper rather than the
/// round-11 segment-0 uniform shift). The full §8.9.6 `chromaScale`
/// derivation (eq. 1384-1385 with `OutOffsetsC` / `OutScalesC` /
/// `OutRangesC` from §8.9.7 + §8.9.8) is now unblocked by the round-151
/// [`parse_dra_syntax`] + [`derive_dra_state`] pair (below) — a
/// follow-up round will retire this round-11 routine in favour of the
/// spec-faithful state.
pub fn apply_dra(pic: &mut YuvPicture, dra: &DraData, bit_depth_luma: u32, bit_depth_chroma: u32) {
    if !dra.descriptor_present {
        return;
    }

    // Snapshot the pre-DRA luma plane: §8.9.4 inputs require decPictureL
    // (the decoded, pre-mapping luma) as the segment-index source.
    let pre_dra_luma: Option<Vec<u8>> = if pic.chroma_format_idc != 0 {
        Some(pic.y.clone())
    } else {
        None
    };

    let lut = build_luma_lut(dra, bit_depth_luma);
    for v in pic.y.iter_mut() {
        *v = lut[*v as usize];
    }

    if pic.chroma_format_idc != 0 {
        let pre_y = pre_dra_luma.expect("snapshot taken when chroma is present");
        let max_c = ((1u32 << bit_depth_chroma) - 1) as i32;
        let ranges = build_ranges_array(dra, bit_depth_luma);
        let num_ranges = dra.num_ranges;
        let (sub_w, sub_h) = chroma_subsampling(pic.chroma_format_idc);
        let (cw, ch) = chroma_plane_dims(pic.width, pic.height, pic.chroma_format_idc);

        apply_chroma_plane_offset(
            &mut pic.cb,
            &pre_y,
            pic.width as usize,
            pic.height as usize,
            cw,
            ch,
            sub_w,
            sub_h,
            dra,
            &ranges,
            num_ranges,
            max_c,
        );
        apply_chroma_plane_offset(
            &mut pic.cr,
            &pre_y,
            pic.width as usize,
            pic.height as usize,
            cw,
            ch,
            sub_w,
            sub_h,
            dra,
            &ranges,
            num_ranges,
            max_c,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_chroma_plane_offset(
    plane: &mut [u8],
    pre_y: &[u8],
    luma_w: usize,
    luma_h: usize,
    chroma_w: usize,
    chroma_h: usize,
    sub_w: usize,
    sub_h: usize,
    dra: &DraData,
    ranges: &[i32],
    num_ranges: usize,
    max_c: i32,
) {
    debug_assert_eq!(plane.len(), chroma_w * chroma_h);
    for y in 0..chroma_h {
        let luma_y = (y * sub_h).min(luma_h.saturating_sub(1));
        let row_off_luma = luma_y * luma_w;
        let row_off_chroma = y * chroma_w;
        for x in 0..chroma_w {
            let luma_x = (x * sub_w).min(luma_w.saturating_sub(1));
            let luma_sample = pre_y[row_off_luma + luma_x] as i32;
            let seg = find_range_idx(luma_sample, ranges, num_ranges);
            let c_off = dra.chroma_qp_offset[seg] as i32;
            let v = plane[row_off_chroma + x] as i32;
            plane[row_off_chroma + x] = (v + c_off).clamp(0, max_c) as u8;
        }
    }
}

fn chroma_subsampling(chroma_format_idc: u32) -> (usize, usize) {
    match chroma_format_idc {
        1 => (2, 2), // 4:2:0
        2 => (2, 1), // 4:2:2
        3 => (1, 1), // 4:4:4
        _ => (1, 1),
    }
}

fn chroma_plane_dims(width: u32, height: u32, chroma_format_idc: u32) -> (usize, usize) {
    let (sw, sh) = chroma_subsampling(chroma_format_idc);
    let cw = (width as usize).div_ceil(sw);
    let ch = (height as usize).div_ceil(sh);
    (cw, ch)
}

// =====================================================================
// Round 151 — §7.3.6-faithful `dra_data()` parser + §7.4.7 derived state.
// =====================================================================
//
// The round-11 [`DraData`] / [`parse_dra_data`] above predate any reading of
// §7.3.6 and don't match the spec wire format (they made up a
// `dra_descriptor_present_flag` + `dra_range_l[]` / `dra_chroma_qp_offset[]`
// shape that no real EVC DRA APS payload follows). They stay in tree
// untouched for now because [`apply_dra`] + the round-148 §8.9.5 per-sample
// chroma-offset path consume them, but a follow-up round (priority #3) will
// retire them in favour of the spec-faithful pair introduced here:
//
//   * [`DraSyntax`] — every raw bit `dra_data()` writes per §7.3.6 (table
//     on page 42 of ISO/IEC 23094-1:2020(E)):
//       dra_descriptor1                u(4)
//       dra_descriptor2                u(4)
//       dra_number_ranges_minus1       ue(v)
//       dra_equal_ranges_flag          u(1)
//       dra_global_offset              u(10)
//       if (dra_equal_ranges_flag)
//           dra_delta_range[0]         u(10)
//       else
//           for j in 0..=num_ranges_minus1
//               dra_delta_range[j]     u(10)
//       for j in 0..=num_ranges_minus1
//           dra_scale_value[j]         u(numBitsDraScale)
//       dra_cb_scale_value             u(numBitsDraScale)
//       dra_cr_scale_value             u(numBitsDraScale)
//       dra_table_idx                  ue(v)
//
//   * [`DraDerived`] — every per-APS derived variable §7.4.7 mandates from
//     those bits, including `numBitsDraScale` (eq. 111), `InDraRange[]`
//     (eq. 112-114), `DraJoinedScaleFlag` (the `dra_table_idx == 58`
//     branch), `numOutRangesL`, `OutRangesL[]` (eq. 115-116 then re-shifted
//     per eq. 122), and the §8.9.3 luma-mapping inputs `InvLumaScales[]`
//     (eq. 117-119) and `DraOffsets[]` (eq. 120-121).
//
// All numeric ranges + bitstream-conformance constraints in §7.4.7 are
// enforced at parse time (descriptor1 ∈ [0, 15], descriptor2 ∈ [0, 15],
// numBitsDraScale > 0, dra_number_ranges_minus1 ∈ [0, 31], InDraRange[j]
// in [0, (1 << BitDepthY) − 1], dra_scale_value[j] ≠ 0 and
// < (4 << dra_descriptor2), dra_table_idx ∈ [0, 58]); a violating
// bitstream is rejected with [`Error::Invalid`] instead of being silently
// truncated.

/// Number of DRA ranges supported by §7.3.6 — `dra_number_ranges_minus1`
/// is in `[0, 31]` so the maximum is 32 segments.
pub const DRA_MAX_RANGES_V2: usize = 32;

/// Spec-faithful `dra_data()` payload (§7.3.6) — every raw bit the parser
/// reads, before §7.4.7 derivation. Companion [`DraDerived`] holds the
/// derived state.
#[derive(Clone, Debug)]
pub struct DraSyntax {
    /// `dra_descriptor1` — integer-part precision of the DRA scale values.
    /// Spec restricts this to 4 in the current version (§7.4.7).
    pub dra_descriptor1: u8,
    /// `dra_descriptor2` — fractional-part precision of the DRA scale
    /// values, also reused as the post-shift right-shift in eq. 121/122.
    /// Spec restricts this to 9 in the current version (§7.4.7).
    pub dra_descriptor2: u8,
    /// `dra_number_ranges_minus1` — number of DRA ranges minus 1, in
    /// `[0, 31]`.
    pub dra_number_ranges_minus1: u8,
    /// `dra_equal_ranges_flag` — when 1, all ranges share the size
    /// `dra_delta_range[0]`; otherwise each range gets its own delta.
    pub dra_equal_ranges_flag: bool,
    /// `dra_global_offset` — `InDraRange[0]` (pre-shift by
    /// `Max(0, BitDepthY − 10)`), in `[1, Min(1023, (1 << BitDepthY) − 1)]`.
    pub dra_global_offset: u16,
    /// `dra_delta_range[j]` — per-range deltas (only the first
    /// `dra_number_ranges_minus1 + 1` are valid; only entry 0 is read
    /// when `dra_equal_ranges_flag` is 1).
    pub dra_delta_range: [u16; DRA_MAX_RANGES_V2],
    /// `dra_scale_value[j]` — per-range luma scale, `numBitsDraScale` bits
    /// wide, in `[1, (4 << dra_descriptor2) − 1]`.
    pub dra_scale_value: [u16; DRA_MAX_RANGES_V2],
    /// `dra_cb_scale_value` — chroma Cb scale (`numBitsDraScale` bits).
    pub dra_cb_scale_value: u16,
    /// `dra_cr_scale_value` — chroma Cr scale (`numBitsDraScale` bits).
    pub dra_cr_scale_value: u16,
    /// `dra_table_idx` — `[0, 58]`; index 58 selects the dual-scale chroma
    /// path (`DraJoinedScaleFlag = 0`), other values select a chroma QP
    /// table entry from `ChromaQpTable` (`DraJoinedScaleFlag = 1`).
    pub dra_table_idx: u8,
}

/// `numBitsDraScale + 1`-bit max width capped at `u16`. The spec restricts
/// `dra_descriptor1 = 4` + `dra_descriptor2 = 9` → 13 bits; the +1 leaves
/// room for any future widening.
const NUM_BITS_DRA_SCALE_MAX: u32 = 16;

/// §7.4.7 derived per-APS state — every variable the spec defines from the
/// parsed [`DraSyntax`] bits, for downstream §8.9.3 (luma mapping) and
/// §8.9.6 / §8.9.7 / §8.9.8 (chroma mapping) consumption.
#[derive(Clone, Debug)]
pub struct DraDerived {
    /// `numBitsDraScale = dra_descriptor1 + dra_descriptor2` (eq. 111).
    pub num_bits_dra_scale: u32,
    /// Number of luma ranges = `dra_number_ranges_minus1 + 1`. Also equal
    /// to `numOutRangesL` (§7.4.7 page 86).
    pub num_ranges: usize,
    /// `DraJoinedScaleFlag` — 0 when `dra_table_idx == 58`, 1 otherwise.
    /// Controls the §8.9.8 (dual-scale) vs §8.9.7 (table-driven) chroma
    /// scale derivation path.
    pub joined_scale_flag: bool,
    /// `InDraRange[j]` for `j` in `0..=num_ranges` (so `num_ranges + 1`
    /// entries). Derived per eq. 112-114; serves as the §8.9.5
    /// `rangesArray` input on the luma side. Wider type than the
    /// 10-bit `dra_global_offset` because the eq. 112 pre-shift by
    /// `Max(0, BitDepthY − 10)` can grow it to (1 << BitDepthY) − 1.
    pub in_dra_range: [u32; DRA_MAX_RANGES_V2 + 1],
    /// `OutRangesL[apsId][i]` for `i` in `0..=num_ranges` — output-side
    /// range boundaries. `OutRangesL[0] = 0`; subsequent entries derive
    /// from eq. 115-116 then re-shift per eq. 122.
    pub out_ranges_l: [i64; DRA_MAX_RANGES_V2 + 1],
    /// `InvLumaScales[apsId][i]` for `i` in `1..=dra_number_ranges_minus1`
    /// per eq. 118-119. Entries 0 and `num_ranges - 0` (the upper tail)
    /// are left at zero — the spec's §8.9.3 invocation uses
    /// `rangeIdx ∈ [0, numRanges − 1]` and the spec-derivation loop only
    /// fills `[1, num_ranges_minus1]`. Round-151 follow-up: confirm with
    /// docs whether entry 0 is meant to be implicitly filled (the §8.9.3
    /// mapping reads it for `rangeIdx == 0`).
    pub inv_luma_scales: [i64; DRA_MAX_RANGES_V2],
    /// `DraOffsets[apsId][i]` for `i` in `1..=dra_number_ranges_minus1`
    /// per eq. 120-121.
    pub dra_offsets: [i64; DRA_MAX_RANGES_V2],
}

impl DraDerived {
    /// Zero-initialised derived state for a single-range identity DRA.
    fn empty() -> Self {
        Self {
            num_bits_dra_scale: 0,
            num_ranges: 0,
            joined_scale_flag: false,
            in_dra_range: [0u32; DRA_MAX_RANGES_V2 + 1],
            out_ranges_l: [0i64; DRA_MAX_RANGES_V2 + 1],
            inv_luma_scales: [0i64; DRA_MAX_RANGES_V2],
            dra_offsets: [0i64; DRA_MAX_RANGES_V2],
        }
    }
}

/// Parse a §7.3.6 `dra_data()` payload and derive every §7.4.7 variable
/// from it.
///
/// * `payload` — APS RBSP after the 1-byte `aps_params_type` /
///   `adaptation_parameter_set_id` header (matches the existing
///   [`parse_dra_data`] contract).
/// * `bit_depth_y` — `BitDepthY` from the active SPS, required for the
///   `Max(0, BitDepthY − 10)` pre-shift in eq. 112 + the bitstream
///   conformance check on `InDraRange[j]`.
///
/// On success, returns `(DraSyntax, DraDerived)` — the syntax half lets a
/// re-encoder reproduce the bitstream verbatim; the derived half is the
/// shape §8.9.3 / §8.9.6 / §8.9.7 / §8.9.8 need.
pub fn parse_dra_syntax(payload: &[u8], bit_depth_y: u32) -> Result<(DraSyntax, DraDerived)> {
    if payload.is_empty() {
        return Err(Error::invalid("evc dra: empty DRA APS payload"));
    }
    if !(8..=16).contains(&bit_depth_y) {
        return Err(Error::invalid(
            "evc dra: bit_depth_y must be in [8, 16] (per SPS §7.4.3.1)",
        ));
    }
    let mut br = BitReader::new(payload);

    // -- §7.3.6 dra_data() ------------------------------------------------
    let dra_descriptor1 = br.u(4)? as u8;
    let dra_descriptor2 = br.u(4)? as u8;

    // §7.4.7 / eq. 111: numBitsDraScale = dra_descriptor1 + dra_descriptor2.
    // Bitstream conformance: must be > 0.
    let num_bits_dra_scale = dra_descriptor1 as u32 + dra_descriptor2 as u32;
    if num_bits_dra_scale == 0 {
        return Err(Error::invalid(
            "evc dra: numBitsDraScale (= dra_descriptor1 + dra_descriptor2) must be > 0",
        ));
    }
    if num_bits_dra_scale > NUM_BITS_DRA_SCALE_MAX {
        return Err(Error::invalid(
            "evc dra: numBitsDraScale exceeds 16 bits (parser cap)",
        ));
    }

    let dra_number_ranges_minus1_u32 = br.ue()?;
    if dra_number_ranges_minus1_u32 > 31 {
        return Err(Error::invalid(
            "evc dra: dra_number_ranges_minus1 out of range [0, 31]",
        ));
    }
    let dra_number_ranges_minus1 = dra_number_ranges_minus1_u32 as u8;
    let num_ranges = dra_number_ranges_minus1 as usize + 1;

    let dra_equal_ranges_flag = br.u1()? != 0;

    let dra_global_offset_u32 = br.u(10)?;
    // §7.4.7: dra_global_offset ∈ [1, Min(1023, (1 << BitDepthY) − 1)].
    let upper = (1u32 << bit_depth_y).saturating_sub(1).min(1023);
    if dra_global_offset_u32 == 0 || dra_global_offset_u32 > upper {
        return Err(Error::invalid(
            "evc dra: dra_global_offset out of range [1, Min(1023, (1<<BitDepthY) − 1)]",
        ));
    }
    let dra_global_offset = dra_global_offset_u32 as u16;

    // dra_delta_range[]: 1 entry when equal_ranges_flag, num_ranges entries
    // otherwise (j in 0..=dra_number_ranges_minus1).
    let mut dra_delta_range = [0u16; DRA_MAX_RANGES_V2];
    if dra_equal_ranges_flag {
        let v = br.u(10)?;
        if v == 0 || v > upper {
            return Err(Error::invalid(
                "evc dra: dra_delta_range[0] out of range [1, Min(1023, (1<<BitDepthY) − 1)]",
            ));
        }
        dra_delta_range[0] = v as u16;
    } else {
        for slot in dra_delta_range.iter_mut().take(num_ranges) {
            let v = br.u(10)?;
            if v == 0 || v > upper {
                return Err(Error::invalid("evc dra: dra_delta_range[j] out of range"));
            }
            *slot = v as u16;
        }
    }

    // dra_scale_value[j] for j in 0..=dra_number_ranges_minus1.
    let scale_upper = if dra_descriptor2 >= 30 {
        u32::MAX // safety against overflow on weird future descriptor2
    } else {
        4u32 << dra_descriptor2
    };
    let mut dra_scale_value = [0u16; DRA_MAX_RANGES_V2];
    for slot in dra_scale_value.iter_mut().take(num_ranges) {
        let v = br.u(num_bits_dra_scale)?;
        if v == 0 || v >= scale_upper {
            return Err(Error::invalid(
                "evc dra: dra_scale_value[j] must be in [1, (4 << dra_descriptor2) − 1]",
            ));
        }
        *slot = v as u16;
    }

    let dra_cb_scale_value_u32 = br.u(num_bits_dra_scale)?;
    if dra_cb_scale_value_u32 == 0 || dra_cb_scale_value_u32 >= scale_upper {
        return Err(Error::invalid(
            "evc dra: dra_cb_scale_value must be in [1, (4 << dra_descriptor2) − 1]",
        ));
    }
    let dra_cb_scale_value = dra_cb_scale_value_u32 as u16;

    let dra_cr_scale_value_u32 = br.u(num_bits_dra_scale)?;
    if dra_cr_scale_value_u32 == 0 || dra_cr_scale_value_u32 >= scale_upper {
        return Err(Error::invalid(
            "evc dra: dra_cr_scale_value must be in [1, (4 << dra_descriptor2) − 1]",
        ));
    }
    let dra_cr_scale_value = dra_cr_scale_value_u32 as u16;

    let dra_table_idx_u32 = br.ue()?;
    if dra_table_idx_u32 > 58 {
        return Err(Error::invalid(
            "evc dra: dra_table_idx out of range [0, 58]",
        ));
    }
    let dra_table_idx = dra_table_idx_u32 as u8;

    let syntax = DraSyntax {
        dra_descriptor1,
        dra_descriptor2,
        dra_number_ranges_minus1,
        dra_equal_ranges_flag,
        dra_global_offset,
        dra_delta_range,
        dra_scale_value,
        dra_cb_scale_value,
        dra_cr_scale_value,
        dra_table_idx,
    };

    // -- §7.4.7 derivation ----------------------------------------------
    let derived = derive_dra_state(&syntax, bit_depth_y)?;
    Ok((syntax, derived))
}

/// Apply the §7.4.7 derivation rules (eq. 111-122 + the `DraJoinedScaleFlag`
/// branch) to a parsed [`DraSyntax`].
///
/// Surfaced as a separate function so a re-encoder can derive state from
/// a hand-constructed [`DraSyntax`] without round-tripping through the
/// byte parser.
pub fn derive_dra_state(syntax: &DraSyntax, bit_depth_y: u32) -> Result<DraDerived> {
    if !(8..=16).contains(&bit_depth_y) {
        return Err(Error::invalid(
            "evc dra: bit_depth_y must be in [8, 16] (per SPS §7.4.3.1)",
        ));
    }
    let mut d = DraDerived::empty();
    d.num_bits_dra_scale = syntax.dra_descriptor1 as u32 + syntax.dra_descriptor2 as u32;
    d.num_ranges = syntax.dra_number_ranges_minus1 as usize + 1;
    // §7.4.7 page 85: dra_table_idx == 58 ⇒ DraJoinedScaleFlag = 0, else 1.
    d.joined_scale_flag = syntax.dra_table_idx != 58;

    // eq. 112: InDraRange[0] = dra_global_offset << Max(0, BitDepthY − 10).
    let shift = bit_depth_y.saturating_sub(10);
    let max_in_range = (1u32 << bit_depth_y).saturating_sub(1);
    let in0 = (syntax.dra_global_offset as u32) << shift;
    if in0 > max_in_range {
        return Err(Error::invalid(
            "evc dra: InDraRange[0] exceeds (1 << BitDepthY) − 1",
        ));
    }
    d.in_dra_range[0] = in0;

    // eq. 113-114: for j in 1..=num_ranges_minus1+1.
    for j in 1..=d.num_ranges {
        let delta_range = if syntax.dra_equal_ranges_flag {
            syntax.dra_delta_range[0]
        } else {
            // dra_delta_range[j − 1]
            syntax.dra_delta_range[j - 1]
        };
        let prev = d.in_dra_range[j - 1];
        let add = (delta_range as u32) << shift;
        let next = prev
            .checked_add(add)
            .ok_or_else(|| Error::invalid("evc dra: InDraRange[j] arithmetic overflowed u32"))?;
        if next > max_in_range {
            return Err(Error::invalid(
                "evc dra: InDraRange[j] exceeds (1 << BitDepthY) − 1",
            ));
        }
        d.in_dra_range[j] = next;
    }

    // §7.4.7 page 86: OutRangesL[0] = 0; for i in 0..numOutRangesL,
    // outDelta = dra_scale_value[i − 1] * (InDraRange[i] − InDraRange[i − 1])
    // (eq. 115); OutRangesL[i] = OutRangesL[i − 1] + outDelta (eq. 116).
    //
    // The spec text says "for i in the range of 0 to numOutRangesL,
    // inclusive" but eq. 115 references `dra_scale_value[i − 1]` and
    // `InDraRange[i] − InDraRange[i − 1]`, so the recursion clearly starts
    // at i = 1 (with the i = 0 entry pinned at 0 by the previous sentence).
    // We iterate i in 1..=num_ranges so we end up with `num_ranges + 1`
    // entries, matching `numOutRangesL + 1` boundaries the §8.9.5 helper
    // wants.
    d.out_ranges_l[0] = 0;
    for i in 1..=d.num_ranges {
        let dscale = syntax.dra_scale_value[i - 1] as i64;
        let drange = d.in_dra_range[i] as i64 - d.in_dra_range[i - 1] as i64;
        let out_delta = dscale * drange;
        d.out_ranges_l[i] = d.out_ranges_l[i - 1] + out_delta;
    }

    // eq. 117-121: InvLumaScales[i] + DraOffsets[i] for
    // i in 1..=dra_number_ranges_minus1.
    //
    // invScalePrec is the spec constant 18 (eq. 117). The eq. 121 right
    // shift uses dra_descriptor2 — undefined for dra_descriptor2 == 0
    // (1 << (dra_descriptor2 − 1) underflows). §7.4.7 restricts
    // dra_descriptor2 to 9 in the current spec version and numBitsDraScale
    // must be > 0; we still guard against dra_descriptor2 == 0 here in case
    // a future descriptor1 = 15 / descriptor2 = 0 payload appears, by
    // skipping the eq. 121 normalisation (it's a no-op when the shift is 0).
    const INV_SCALE_PREC: u32 = 18;
    let d2 = syntax.dra_descriptor2 as u32;
    for i in 1..d.num_ranges {
        let dsv = syntax.dra_scale_value[i] as i64;
        // eq. 118: invScale = ((1 << 18) + (dsv >> 1)) / dsv.
        let inv_scale = ((1i64 << INV_SCALE_PREC) + (dsv >> 1)) / dsv;
        d.inv_luma_scales[i] = inv_scale;

        // eq. 120: diffVal = OutRangesL[i + 1] * invScale.
        // We have num_ranges + 1 entries in out_ranges_l so i + 1 is in
        // bounds for i ∈ [1, num_ranges − 1].
        let diff_val = d.out_ranges_l[i + 1] * inv_scale;

        // eq. 121: DraOffsets[i] = ((InDraRange[i+1] << invScalePrec)
        //                            − diffVal + (1 << (d2 − 1))) >> d2.
        let term1 = (d.in_dra_range[i + 1] as i64) << INV_SCALE_PREC;
        let dra_off = if d2 == 0 {
            term1 - diff_val
        } else {
            (term1 - diff_val + (1i64 << (d2 - 1))) >> d2
        };
        d.dra_offsets[i] = dra_off;
    }

    // eq. 122: OutRangesL[i] = (OutRangesL[i] + (1 << (d2 − 1))) >> d2.
    // Applied after eq. 120 so the diffVal multiplication used the
    // pre-normalisation value (spec page 86 lists the eq. 122 step *after*
    // the InvLumaScales / DraOffsets block).
    if d2 != 0 {
        for i in 0..=d.num_ranges {
            d.out_ranges_l[i] = (d.out_ranges_l[i] + (1i64 << (d2 - 1))) >> d2;
        }
    }

    Ok(derived_ok(d))
}

#[inline]
fn derived_ok(d: DraDerived) -> DraDerived {
    d
}

// =====================================================================
// Round 174 — §8.9.3 luma inverse mapping (eq. 1374-1376).
// =====================================================================
//
// §8.9.3 takes the §7.4.7-derived state (`InvLumaScales`, `DraOffsets`,
// `OutRangesL`, `numOutRangesL`) and applies, per luma sample:
//
//   rangeIdx     = §8.9.5(lumaSample, OutRangesL, numOutRangesL)
//   incrValue    = InvLumaScales[apsId][rangeIdx] * lumaSample        (1374)
//   mappedSample = (DraOffsets[apsId][rangeIdx] + incrValue
//                   + (1 << 8)) >> 9                                   (1375)
//   invLumaSample = Clip1Y(mappedSample)                              (1376)
//
// where `Clip1Y(v) = Clip3(0, (1 << BitDepthY) − 1, v)`.
//
// ## Documented §7.4.7 docs gap on `InvLumaScales[0]` / `DraOffsets[0]`
//
// §7.4.7 spec text on page 86 defines `InvLumaScales[apsId][i]` and
// `DraOffsets[apsId][i]` "for i in the range of 1 to
// dra_number_ranges_minus1, inclusive". With `numOutRangesL =
// dra_number_ranges_minus1 + 1` that loop fills indices
// `[1, numOutRangesL − 1]`, leaving index 0 explicitly undefined.
//
// But §8.9.3 indexes `InvLumaScales[apsId][rangeIdx]` and
// `DraOffsets[apsId][rangeIdx]` with `rangeIdx ∈ [0, numOutRangesL − 1]`
// — including index 0, the lowest-luma segment. Per the literal spec
// (i.e. taking `InvLumaScales[0] = DraOffsets[0] = 0` as the default
// from "unwritten" semantics), eq. 1375 collapses to
// `(0 + 0 + 256) >> 9 = 0` for every sample whose value falls in the
// lowest segment, so every dark sample is forced to 0 — clearly
// degenerate behaviour and inconsistent with the spec's identity-DRA
// case (`dra_scale_value[i] = 1 << dra_descriptor2` for all i) which
// must reproduce input verbatim.
//
// The most plausible reconciliation is that the §7.4.7 loop bounds are
// an off-by-one — "for i in the range of 0 to dra_number_ranges_minus1,
// inclusive" — which matches the per-range one-scale-per-range data
// flow (the parser reads `dra_scale_value[j]` for
// `j ∈ [0, dra_number_ranges_minus1]`, and §8.9.5 returns
// `rangeIdx ∈ [0, numOutRangesL − 1]`). Under that reading
// `InvLumaScales[0]` and `DraOffsets[0]` are computed identically to
// every other entry via eq. 118/120/121, just substituting i = 0.
//
// **Round 174 takes no stance.** Both interpretations are made
// available so a future round can wire whichever the docs collaborator
// resolves to. The default `derive_dra_state` (round 151) leaves index
// 0 at zero matching the literal spec text; the new
// [`fill_inv_luma_scales_range_zero`] applies the off-by-one
// reconciliation in-place on a `DraDerived` for callers that need a
// non-degenerate §8.9.3.

/// Fill `InvLumaScales[0]` and `DraOffsets[0]` on a `DraDerived` using the
/// off-by-one reconciliation of §7.4.7 (extending eq. 118/120/121 to
/// `i = 0`).
///
/// **This is an interpretation, not a spec reading.** §7.4.7 page 86
/// literally restricts the InvLumaScales / DraOffsets derivation to
/// `i ∈ [1, dra_number_ranges_minus1]`. §8.9.3 needs
/// `InvLumaScales[0]` / `DraOffsets[0]` to map samples in the lowest
/// segment. This helper applies the same eq. 118 / 120 / 121 formulas to
/// `i = 0`, treating `dra_scale_value[0]` as the scale for range 0 — the
/// natural symmetric extension. See the module-level docs gap above.
///
/// * `derived` must already have been derived via [`derive_dra_state`]
///   from `syntax`.
/// * Read-only on `syntax`. Mutates `derived.inv_luma_scales[0]` and
///   `derived.dra_offsets[0]`; leaves every other entry alone.
/// * Returns an error if the eq. 118 division would divide by zero
///   (spec already forbids `dra_scale_value[j] == 0`, so this is a
///   defence-in-depth check).
pub fn fill_inv_luma_scales_range_zero(derived: &mut DraDerived, syntax: &DraSyntax) -> Result<()> {
    if derived.num_ranges == 0 {
        return Ok(());
    }
    let dsv0 = syntax.dra_scale_value[0] as i64;
    if dsv0 == 0 {
        return Err(Error::invalid(
            "evc dra: dra_scale_value[0] == 0 (forbidden by §7.4.7); cannot derive InvLumaScales[0]",
        ));
    }
    const INV_SCALE_PREC: u32 = 18;
    let d2 = syntax.dra_descriptor2 as u32;

    // eq. 118 at i = 0.
    let inv_scale = ((1i64 << INV_SCALE_PREC) + (dsv0 >> 1)) / dsv0;
    derived.inv_luma_scales[0] = inv_scale;

    // eq. 120 at i = 0: diffVal = OutRangesL[1] * invScale.
    // out_ranges_l[1] has already been post-shifted by eq. 122 inside
    // `derive_dra_state`. The eq. 121 numerator wants the
    // pre-eq-122 `OutRangesL[1]` to keep the arithmetic dimensionally
    // consistent with the i ≥ 1 entries (where eq. 120 was evaluated
    // before eq. 122). We reconstruct the pre-shift value here by left-
    // shifting by d2 (an exact inverse only when d2 = 0 or eq. 122
    // didn't round up; this matches what `derive_dra_state` does
    // implicitly at i ≥ 1 because it evaluates eq. 120 first, then
    // applies eq. 122 separately).
    let out1_pre_shift = if d2 == 0 {
        derived.out_ranges_l[1]
    } else {
        derived.out_ranges_l[1] << d2
    };
    let diff_val = out1_pre_shift * inv_scale;

    // eq. 121: DraOffsets[0] = ((InDraRange[1] << invScalePrec)
    //                           − diffVal + (1 << (d2 − 1))) >> d2.
    let term1 = (derived.in_dra_range[1] as i64) << INV_SCALE_PREC;
    let dra_off = if d2 == 0 {
        term1 - diff_val
    } else {
        (term1 - diff_val + (1i64 << (d2 - 1))) >> d2
    };
    derived.dra_offsets[0] = dra_off;

    Ok(())
}

/// Apply §8.9.3 (eq. 1374-1376) to a `u16` luma plane in-place.
///
/// Each sample is mapped per:
///
/// ```text
///   rangeIdx     = find_range_idx(lumaSample, &out_ranges_l_i32, num_ranges)
///   incrValue    = InvLumaScales[rangeIdx] * lumaSample              (1374)
///   mappedSample = (DraOffsets[rangeIdx] + incrValue + 256) >> 9     (1375)
///   invLumaSample = Clip3(0, (1 << bit_depth_y) − 1, mappedSample)   (1376)
/// ```
///
/// `bit_depth_y` controls the `Clip1Y` upper bound; samples are read and
/// written as `u16` so this covers 8 / 10 / 12-bit luma uniformly.
///
/// ## DOCS-GAP awareness
///
/// `derived.inv_luma_scales[0]` and `derived.dra_offsets[0]` are read
/// for every sample whose value falls into range 0 (the lowest segment
/// after §8.9.5). The default `derive_dra_state` leaves both at zero
/// (literal spec); call [`fill_inv_luma_scales_range_zero`] before
/// invoking this helper if you need non-degenerate output for the
/// lowest segment. See the module-level docs-gap notes for the full
/// rationale.
pub fn apply_luma_inverse_mapping(plane: &mut [u16], derived: &DraDerived, bit_depth_y: u32) {
    if derived.num_ranges == 0 {
        return;
    }
    let ranges_array = out_ranges_l_as_i32(derived);
    let clip_max = ((1u32 << bit_depth_y).saturating_sub(1)) as i64;
    for sample in plane.iter_mut() {
        *sample = map_one_luma_sample(*sample as i64, &ranges_array, derived, clip_max) as u16;
    }
}

/// Apply §8.9.3 to a `u8` luma plane in-place (8-bit shortcut).
///
/// Equivalent to [`apply_luma_inverse_mapping`] with `bit_depth_y = 8`
/// against a `u16`-widened plane, but works directly on `u8` so callers
/// holding 8-bit pictures don't need to widen-then-narrow. Internally
/// builds a 256-entry LUT via [`build_inv_luma_lut_8bit`] and applies it
/// to every sample.
pub fn apply_luma_inverse_mapping_u8(plane: &mut [u8], derived: &DraDerived) {
    if derived.num_ranges == 0 {
        return;
    }
    let lut = build_inv_luma_lut_8bit(derived);
    for sample in plane.iter_mut() {
        *sample = lut[*sample as usize];
    }
}

/// Build a 256-entry LUT applying §8.9.3 to every 8-bit luma sample
/// value `[0, 255]`.
///
/// For 8-bit pictures the §8.9.3 process is a pure function of the
/// 8-bit sample value, so the per-sample apply degenerates to a LUT
/// lookup. Each entry is computed by running [`map_one_luma_sample`]
/// against `derived`'s `out_ranges_l` / `inv_luma_scales` /
/// `dra_offsets` and clipping to `[0, 255]`.
///
/// Returns an identity LUT when `derived.num_ranges == 0`.
pub fn build_inv_luma_lut_8bit(derived: &DraDerived) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if derived.num_ranges == 0 {
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = i as u8;
        }
        return lut;
    }
    let ranges_array = out_ranges_l_as_i32(derived);
    let clip_max = 255i64;
    for (i, slot) in lut.iter_mut().enumerate() {
        *slot = map_one_luma_sample(i as i64, &ranges_array, derived, clip_max) as u8;
    }
    lut
}

/// Materialise `derived.out_ranges_l[0..=num_ranges]` as an `i32` vector
/// suitable for [`find_range_idx`]. Saturates at `i32::MAX` if the
/// post-eq-122 value somehow exceeds (it cannot in spec-valid bit
/// depths, but the saturation guards against contrived inputs).
fn out_ranges_l_as_i32(derived: &DraDerived) -> Vec<i32> {
    let n = derived.num_ranges + 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = derived.out_ranges_l[i];
        let clamped = v.clamp(i32::MIN as i64, i32::MAX as i64);
        out.push(clamped as i32);
    }
    out
}

/// Apply §8.9.3 eq. 1374-1376 to a single luma sample.
///
/// Pure function: `(lumaSample, OutRangesL, DraDerived, clip_max) →
/// invLumaSample`. Returns an `i64` to avoid lossy narrowing; callers
/// down-convert to the plane element type.
#[inline]
fn map_one_luma_sample(
    luma_sample: i64,
    ranges_array: &[i32],
    derived: &DraDerived,
    clip_max: i64,
) -> i64 {
    let range_idx = find_range_idx(luma_sample as i32, ranges_array, derived.num_ranges);
    let inv_scale = derived.inv_luma_scales[range_idx];
    let dra_offset = derived.dra_offsets[range_idx];
    // eq. 1374-1375.
    let incr_value = inv_scale * luma_sample;
    let mapped = (dra_offset + incr_value + (1i64 << 8)) >> 9;
    // eq. 1376: Clip1Y.
    mapped.clamp(0, clip_max)
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    fn emit_minimal_dra_data(present: bool) -> Vec<u8> {
        let mut e = BitEmitter::new();
        e.u(1, if present { 1 } else { 0 });
        if present {
            e.u(4, 0); // 1 range (num_ranges_minus1 = 0)
            e.u(10, 0); // range_l[0] = 0
            e.u(11, 8); // dra_scale[0] = 8 (Q8.3 → scale = 1.0)
            e.u(1, 0); // luma_inverted = 0
                       // 0 chroma entries (num_ranges-1 = 0 → loop runs 0 times)
        }
        e.finish_with_trailing_bits();
        e.into_bytes()
    }

    #[test]
    fn parse_dra_not_present() {
        let payload = emit_minimal_dra_data(false);
        let dra = parse_dra_data(&payload).unwrap();
        assert!(!dra.descriptor_present);
    }

    #[test]
    fn parse_dra_single_range_identity() {
        // One segment starting at 0 with scale = 1.0 (Q8.3 = 8).
        let payload = emit_minimal_dra_data(true);
        let dra = parse_dra_data(&payload).unwrap();
        assert!(dra.descriptor_present);
        assert_eq!(dra.num_ranges, 1);
        assert_eq!(dra.range_l[0], 0);
        assert_eq!(dra.scale[0], 8); // Q8.3 → 1.0
    }

    #[test]
    fn parse_dra_two_ranges() {
        let mut e = BitEmitter::new();
        e.u(1, 1); // present
        e.u(4, 1); // 2 ranges
        e.u(10, 0); // range_l[0] = 0
        e.u(10, 128); // range_l[1] = 128
        e.u(11, 8); // scale[0] = 8 (1.0)
                    // scale_delta[1] = +4 → scale[1] = 12
        e.u(12, 4); // delta = +4, unsigned 12-bit (positive)
        e.u(1, 0); // not inverted
                   // 1 chroma entry (i in 0..num_ranges-1 = 0..1)
        e.u(8, 16); // chroma_qp_scale[0] = 16
        e.u(8, 255); // chroma_qp_offset[0] = -1 (signed 8-bit: 0xFF)
        e.finish_with_trailing_bits();
        let payload = e.into_bytes();
        let dra = parse_dra_data(&payload).unwrap();
        assert_eq!(dra.num_ranges, 2);
        assert_eq!(dra.range_l[0], 0);
        assert_eq!(dra.range_l[1], 128);
        assert_eq!(dra.scale[0], 8);
        assert_eq!(dra.scale[1], 12);
        assert_eq!(dra.chroma_qp_scale[0], 16);
        assert_eq!(dra.chroma_qp_offset[0], -1);
    }

    #[test]
    fn parse_dra_rejects_empty_payload() {
        assert!(parse_dra_data(&[]).is_err());
    }

    #[test]
    fn lut_identity_single_range() {
        // Scale = 1.0 (Q8.3 = 8), range starting at 0.
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 1,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.scale[0] = 8; // Q8.3 → 1.0
        let lut = build_luma_lut(&dra, 8);
        // Identity mapping: LUT[v] == v for all v.
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8, "LUT[{i}] should be {i}");
        }
    }

    #[test]
    fn lut_not_present_is_identity() {
        let dra = DraData::default(); // descriptor_present = false
        let lut = build_luma_lut(&dra, 8);
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8);
        }
    }

    #[test]
    fn lut_double_range_maps_half_values() {
        // Scale = 2.0 (Q8.3 = 16) → LUT[v] = 2*v, clipped at 255.
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 1,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.scale[0] = 16; // Q8.3 → 2.0
        let lut = build_luma_lut(&dra, 8);
        // LUT[0] = 0, LUT[1] = 2, LUT[100] = 200, LUT[128] = 255 (clipped).
        assert_eq!(lut[0], 0);
        assert_eq!(lut[1], 2); // (16*1+4)>>3 = 2
        assert_eq!(lut[100], 200);
        assert_eq!(lut[128], 255); // clipped
        assert_eq!(lut[200], 255);
    }

    #[test]
    fn apply_dra_noop_when_not_present() {
        let mut pic = crate::picture::YuvPicture::new(8, 8, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let dra = DraData::default();
        apply_dra(&mut pic, &dra, 8, 8);
        for v in pic.y.iter() {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn apply_dra_identity_mapping_preserves_picture() {
        let mut pic = crate::picture::YuvPicture::new(8, 8, 1, 8).unwrap();
        for (i, v) in pic.y.iter_mut().enumerate() {
            *v = (i % 256) as u8;
        }
        let snapshot = pic.y.clone();
        // Single segment, scale = 1.0 → identity LUT.
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 1,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.scale[0] = 8;
        apply_dra(&mut pic, &dra, 8, 8);
        assert_eq!(pic.y, snapshot);
    }

    #[test]
    fn apply_dra_chroma_offset_shifts_cb_cr() {
        let mut pic = crate::picture::YuvPicture::new(8, 8, 1, 8).unwrap();
        for v in pic.cb.iter_mut() {
            *v = 100;
        }
        for v in pic.cr.iter_mut() {
            *v = 100;
        }
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 2,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 128;
        dra.scale[0] = 8; // 1.0
        dra.scale[1] = 8;
        dra.chroma_qp_offset[0] = 10;
        // Fill luma so it stays in seg 0 (DRA maps it identity).
        for v in pic.y.iter_mut() {
            *v = 50;
        }
        apply_dra(&mut pic, &dra, 8, 8);
        // chroma_offset_for_segment(0) = 10, so Cb/Cr should shift by +10.
        for v in pic.cb.iter() {
            assert_eq!(*v, 110);
        }
        for v in pic.cr.iter() {
            assert_eq!(*v, 110);
        }
    }

    #[test]
    fn find_segment_returns_correct_index() {
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 3,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 64;
        dra.range_l[2] = 192;
        assert_eq!(find_segment(&dra, 0), 0);
        assert_eq!(find_segment(&dra, 63), 0);
        assert_eq!(find_segment(&dra, 64), 1);
        assert_eq!(find_segment(&dra, 191), 1);
        assert_eq!(find_segment(&dra, 192), 2);
        assert_eq!(find_segment(&dra, 255), 2);
    }

    // -----------------------------------------------------------------
    // Round 148 — §8.9.5 range-idx helper + per-sample chroma offset.
    // -----------------------------------------------------------------

    #[test]
    fn round148_find_range_idx_eq1383_three_ranges() {
        // ranges_array = [0, 64, 192, 256]; num_ranges = 3.
        // Per eq. 1383:
        //   input <  64  → idx 0
        //   input < 192  → idx 1
        //   otherwise    → idx 2 (clamped to num_ranges − 1)
        let ranges = [0i32, 64, 192, 256];
        assert_eq!(find_range_idx(0, &ranges, 3), 0);
        assert_eq!(find_range_idx(63, &ranges, 3), 0);
        assert_eq!(find_range_idx(64, &ranges, 3), 1);
        assert_eq!(find_range_idx(191, &ranges, 3), 1);
        assert_eq!(find_range_idx(192, &ranges, 3), 2);
        assert_eq!(find_range_idx(255, &ranges, 3), 2);
        // Above the top boundary: range-not-found ⇒ clamped to numRanges − 1.
        assert_eq!(find_range_idx(1000, &ranges, 3), 2);
    }

    #[test]
    fn round148_find_range_idx_single_range_always_zero() {
        let ranges = [0i32, 256];
        for v in (0..256).step_by(17) {
            assert_eq!(find_range_idx(v, &ranges, 1), 0);
        }
        // Even above the top boundary.
        assert_eq!(find_range_idx(999, &ranges, 1), 0);
    }

    #[test]
    fn round148_find_range_idx_zero_ranges_returns_zero() {
        assert_eq!(find_range_idx(42, &[], 0), 0);
    }

    #[test]
    fn round148_build_ranges_array_appends_top_sentinel() {
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 3,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 64;
        dra.range_l[2] = 192;
        let arr = build_ranges_array(&dra, 8);
        assert_eq!(arr, vec![0, 64, 192, 256]);
        // 10-bit top sentinel = 1024.
        let arr10 = build_ranges_array(&dra, 10);
        assert_eq!(arr10, vec![0, 64, 192, 1024]);
    }

    #[test]
    fn round148_apply_dra_chroma_offset_uses_colocated_luma_segment() {
        // Three luma segments: [0, 64), [64, 192), [192, 255].
        // Per-segment chroma offsets: seg 0 = +5, seg 1 = +10, seg 2 = +15.
        // Build an 8×8 4:4:4 picture so each chroma sample directly maps
        // to a luma sample at the same (x, y) (SubWidthC = SubHeightC = 1).
        let mut pic = crate::picture::YuvPicture::new(8, 8, 3, 8).unwrap();
        // Fill luma so column x maps to segment ⌊x/3⌋ (cols 0..2 → seg 0,
        // cols 3..5 → seg 1, cols 6..7 → seg 2). Use luma values 0 / 128 / 192.
        for y in 0..8 {
            for x in 0..8 {
                let seg = if x < 3 {
                    0
                } else if x < 6 {
                    1
                } else {
                    2
                };
                let luma_val = match seg {
                    0 => 0u8,
                    1 => 128u8,
                    _ => 192u8,
                };
                pic.y[y * 8 + x] = luma_val;
            }
        }
        // Fill chroma planes with a known base value.
        for v in pic.cb.iter_mut() {
            *v = 100;
        }
        for v in pic.cr.iter_mut() {
            *v = 100;
        }
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 3,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 64;
        dra.range_l[2] = 192;
        // Identity luma scales so the LUT doesn't perturb luma values.
        dra.scale[0] = 8;
        dra.scale[1] = 8;
        dra.scale[2] = 8;
        dra.chroma_qp_offset[0] = 5;
        dra.chroma_qp_offset[1] = 10;
        dra.chroma_qp_offset[2] = 15;
        apply_dra(&mut pic, &dra, 8, 8);
        // Verify per-sample chroma offset uses the *co-located* luma's
        // segment, not segment 0 uniformly.
        for y in 0..8 {
            for x in 0..8 {
                let expect_off = if x < 3 {
                    5
                } else if x < 6 {
                    10
                } else {
                    15
                };
                assert_eq!(
                    pic.cb[y * 8 + x],
                    (100 + expect_off) as u8,
                    "Cb at ({x},{y}) wrong segment offset"
                );
                assert_eq!(
                    pic.cr[y * 8 + x],
                    (100 + expect_off) as u8,
                    "Cr at ({x},{y}) wrong segment offset"
                );
            }
        }
    }

    #[test]
    fn round148_apply_dra_uses_pre_dra_luma_for_chroma_segment_lookup() {
        // Build a 4:4:4 8×8 picture with luma values that fall in segment 1
        // (luma 100, with range boundaries [0, 64, 192)).
        // The luma LUT is configured with scale 2.0 so post-DRA luma = 200,
        // which would re-classify into segment 2 if the chroma lookup
        // (incorrectly) read post-DRA luma instead of pre-DRA.
        // Segment 1 offset = 7; segment 2 offset = 99 (sentinel for failure).
        let mut pic = crate::picture::YuvPicture::new(4, 4, 3, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        for v in pic.cb.iter_mut() {
            *v = 50;
        }
        for v in pic.cr.iter_mut() {
            *v = 50;
        }
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 2,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 192;
        // Scale 2.0 in seg 0 (Q8.3 = 16) → luma 100 → 200.
        dra.scale[0] = 16;
        dra.scale[1] = 8;
        // If chroma lookup uses *pre-DRA* luma (100 → seg 0), expect +7.
        // If it (wrongly) uses *post-DRA* luma (200 → seg 1), would get +99.
        dra.chroma_qp_offset[0] = 7;
        dra.chroma_qp_offset[1] = 99;
        apply_dra(&mut pic, &dra, 8, 8);
        // Sanity: luma was mapped 100 → 200.
        assert_eq!(pic.y[0], 200);
        // Chroma offset must reflect pre-DRA luma's segment (0 → +7).
        for v in pic.cb.iter() {
            assert_eq!(*v, 57, "Cb must use pre-DRA luma segment (got {v})");
        }
        for v in pic.cr.iter() {
            assert_eq!(*v, 57, "Cr must use pre-DRA luma segment (got {v})");
        }
    }

    #[test]
    fn round148_apply_dra_chroma_offset_420_uses_subsampled_luma() {
        // 4:2:0 picture: chroma sample (x, y) reads luma (x*2, y*2).
        // 8×8 luma plane, 4×4 chroma. Configure luma so the upper-left
        // 4×4 luma quadrant is in segment 0 and the lower-right 4×4 is
        // in segment 1; chroma should pick up the matching offsets at
        // the subsampled positions.
        let mut pic = crate::picture::YuvPicture::new(8, 8, 1, 8).unwrap();
        // luma upper half = 0 (seg 0); lower half = 128 (seg 1).
        for y in 0..8 {
            for x in 0..8 {
                pic.y[y * 8 + x] = if y < 4 { 0 } else { 128 };
            }
        }
        for v in pic.cb.iter_mut() {
            *v = 60;
        }
        for v in pic.cr.iter_mut() {
            *v = 60;
        }
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 2,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 64;
        dra.scale[0] = 8;
        dra.scale[1] = 8;
        dra.chroma_qp_offset[0] = 3;
        dra.chroma_qp_offset[1] = 9;
        apply_dra(&mut pic, &dra, 8, 8);
        // Chroma is 4×4 — rows 0,1 read luma rows 0,2 (both seg 0 → +3);
        // rows 2,3 read luma rows 4,6 (both seg 1 → +9).
        for y in 0..4 {
            for x in 0..4 {
                let expect = if y < 2 { 63u8 } else { 69u8 };
                assert_eq!(
                    pic.cb[y * 4 + x],
                    expect,
                    "Cb chroma at ({x},{y}) expected {expect}"
                );
                assert_eq!(
                    pic.cr[y * 4 + x],
                    expect,
                    "Cr chroma at ({x},{y}) expected {expect}"
                );
            }
        }
    }

    #[test]
    fn round148_apply_dra_clips_chroma_offset() {
        // Co-located luma is 200 → falls into a segment with offset +60;
        // chroma starts at 250 → 250 + 60 = 310 → clamped to 255.
        let mut pic = crate::picture::YuvPicture::new(4, 4, 3, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 200;
        }
        for v in pic.cb.iter_mut() {
            *v = 250;
        }
        for v in pic.cr.iter_mut() {
            *v = 250;
        }
        let mut dra = DraData {
            descriptor_present: true,
            num_ranges: 2,
            ..DraData::default()
        };
        dra.range_l[0] = 0;
        dra.range_l[1] = 192;
        dra.scale[0] = 8;
        dra.scale[1] = 8;
        // Segment 1 covers [192, 255]; offset = +60.
        dra.chroma_qp_offset[0] = 0;
        dra.chroma_qp_offset[1] = 60;
        apply_dra(&mut pic, &dra, 8, 8);
        for v in pic.cb.iter() {
            assert_eq!(*v, 255, "Cb should clip to 255");
        }
        for v in pic.cr.iter() {
            assert_eq!(*v, 255, "Cr should clip to 255");
        }
    }

    // -----------------------------------------------------------------
    // Round 151 — §7.3.6-faithful dra_data() parser + §7.4.7 derivation.
    // -----------------------------------------------------------------
    //
    // These exercise the new spec-faithful pair (`parse_dra_syntax` +
    // `derive_dra_state`), independent of the legacy `DraData` /
    // `parse_dra_data` / `apply_dra` chain above. They are written so a
    // follow-up round wiring §8.9.3 (luma inverse mapping) and §8.9.6
    // (chroma scale derivation) can build directly on `DraDerived`'s
    // `OutRangesL` / `InvLumaScales` / `DraOffsets` / `InDraRange` fields.

    /// Emit a minimal §7.3.6 `dra_data()` payload with current-spec defaults
    /// (`dra_descriptor1 = 4`, `dra_descriptor2 = 9`).
    #[allow(clippy::too_many_arguments)]
    fn emit_dra_syntax_minimal(
        num_ranges_minus1: u32,
        equal_ranges_flag: bool,
        global_offset: u32,
        delta_ranges: &[u32],
        scale_values: &[u32],
        cb_scale: u32,
        cr_scale: u32,
        table_idx: u32,
    ) -> Vec<u8> {
        let mut e = BitEmitter::new();
        e.u(4, 4); // dra_descriptor1 = 4 (spec-restricted)
        e.u(4, 9); // dra_descriptor2 = 9 (spec-restricted)
        e.ue(num_ranges_minus1); // dra_number_ranges_minus1
        e.u(1, if equal_ranges_flag { 1 } else { 0 });
        e.u(10, global_offset);
        if equal_ranges_flag {
            e.u(10, delta_ranges[0]);
        } else {
            for d in delta_ranges.iter().take(num_ranges_minus1 as usize + 1) {
                e.u(10, *d);
            }
        }
        // numBitsDraScale = 4 + 9 = 13.
        for s in scale_values.iter().take(num_ranges_minus1 as usize + 1) {
            e.u(13, *s);
        }
        e.u(13, cb_scale);
        e.u(13, cr_scale);
        e.ue(table_idx);
        e.finish_with_trailing_bits();
        e.into_bytes()
    }

    #[test]
    fn round151_parse_dra_syntax_minimal_single_range() {
        // Single range, equal_ranges_flag = 1, scale 512 (= 1.0 in Q4.9),
        // chroma scales also 512, table_idx 58 (DraJoinedScaleFlag = 0).
        let payload = emit_dra_syntax_minimal(
            0,    // num_ranges_minus1 = 0 ⇒ 1 range
            true, // equal_ranges_flag
            1,    // global_offset = 1 (must be ≥ 1)
            &[1],
            &[512],
            512,
            512,
            58,
        );
        let (syn, der) = parse_dra_syntax(&payload, 10).unwrap();
        assert_eq!(syn.dra_descriptor1, 4);
        assert_eq!(syn.dra_descriptor2, 9);
        assert_eq!(syn.dra_number_ranges_minus1, 0);
        assert!(syn.dra_equal_ranges_flag);
        assert_eq!(syn.dra_global_offset, 1);
        assert_eq!(syn.dra_delta_range[0], 1);
        assert_eq!(syn.dra_scale_value[0], 512);
        assert_eq!(syn.dra_cb_scale_value, 512);
        assert_eq!(syn.dra_cr_scale_value, 512);
        assert_eq!(syn.dra_table_idx, 58);
        assert_eq!(der.num_bits_dra_scale, 13);
        assert_eq!(der.num_ranges, 1);
        // dra_table_idx == 58 ⇒ DraJoinedScaleFlag = 0 (spec page 86).
        assert!(!der.joined_scale_flag);
        // BitDepthY = 10 ⇒ shift = 0 ⇒ InDraRange[0] = 1.
        assert_eq!(der.in_dra_range[0], 1);
        // InDraRange[1] = 1 + 1 = 2.
        assert_eq!(der.in_dra_range[1], 2);
    }

    #[test]
    fn round151_parse_dra_syntax_table_idx_zero_sets_joined_flag() {
        let payload = emit_dra_syntax_minimal(
            0,
            true,
            1,
            &[1],
            &[512],
            512,
            512,
            0, // table_idx != 58 ⇒ DraJoinedScaleFlag = 1
        );
        let (_, der) = parse_dra_syntax(&payload, 10).unwrap();
        assert!(der.joined_scale_flag);
    }

    #[test]
    fn round151_parse_dra_syntax_equal_ranges_distributes_delta() {
        // 4 equal-sized ranges, delta = 100; global_offset = 50.
        // InDraRange = [50, 150, 250, 350, 450] (BitDepthY = 10 → no shift).
        let payload = emit_dra_syntax_minimal(
            3, // 4 ranges
            true,
            50,
            &[100],
            &[512, 512, 512, 512],
            512,
            512,
            58,
        );
        let (_, der) = parse_dra_syntax(&payload, 10).unwrap();
        assert_eq!(der.num_ranges, 4);
        assert_eq!(der.in_dra_range[0], 50);
        assert_eq!(der.in_dra_range[1], 150);
        assert_eq!(der.in_dra_range[2], 250);
        assert_eq!(der.in_dra_range[3], 350);
        assert_eq!(der.in_dra_range[4], 450);
    }

    #[test]
    fn round151_parse_dra_syntax_unequal_ranges_uses_per_range_delta() {
        // 3 ranges with deltas 64, 128, 256; global_offset = 16.
        // InDraRange = [16, 80, 208, 464].
        let payload = emit_dra_syntax_minimal(
            2,
            false,
            16,
            &[64, 128, 256],
            &[512, 512, 512],
            512,
            512,
            58,
        );
        let (_, der) = parse_dra_syntax(&payload, 10).unwrap();
        assert_eq!(der.num_ranges, 3);
        assert_eq!(der.in_dra_range[0], 16);
        assert_eq!(der.in_dra_range[1], 80);
        assert_eq!(der.in_dra_range[2], 208);
        assert_eq!(der.in_dra_range[3], 464);
    }

    #[test]
    fn round151_parse_dra_syntax_bit_depth_8_shift_is_zero() {
        // BitDepthY = 8 ⇒ Max(0, 8 − 10) = 0 (saturating), so InDraRange
        // values are read unshifted as well — global_offset 1 + delta 1
        // ⇒ InDraRange = [1, 2].
        let payload = emit_dra_syntax_minimal(0, true, 1, &[1], &[512], 512, 512, 58);
        let (_, der) = parse_dra_syntax(&payload, 8).unwrap();
        assert_eq!(der.in_dra_range[0], 1);
        assert_eq!(der.in_dra_range[1], 2);
    }

    #[test]
    fn round151_parse_dra_syntax_bit_depth_12_applies_shift_by_2() {
        // BitDepthY = 12 ⇒ Max(0, 12 − 10) = 2; global_offset 1 shifted
        // by 2 ⇒ InDraRange[0] = 4, +(1 << 2) = 8 for InDraRange[1].
        let payload = emit_dra_syntax_minimal(0, true, 1, &[1], &[512], 512, 512, 58);
        let (_, der) = parse_dra_syntax(&payload, 12).unwrap();
        assert_eq!(der.in_dra_range[0], 4);
        assert_eq!(der.in_dra_range[1], 8);
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_empty_payload() {
        assert!(parse_dra_syntax(&[], 10).is_err());
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_zero_scale_value() {
        // dra_scale_value[0] = 0 violates §7.4.7's "shall not be equal to 0".
        let payload = emit_dra_syntax_minimal(0, true, 1, &[1], &[0], 512, 512, 58);
        let err = parse_dra_syntax(&payload, 10).unwrap_err();
        assert!(
            format!("{err}").contains("dra_scale_value"),
            "expected dra_scale_value error, got: {err}"
        );
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_overlarge_scale_value() {
        // dra_scale_value[0] must be < (4 << dra_descriptor2) = 4 << 9 =
        // 2048. Try 2048 exactly (rejected — strictly less).
        let payload = emit_dra_syntax_minimal(0, true, 1, &[1], &[2048], 512, 512, 58);
        let err = parse_dra_syntax(&payload, 10).unwrap_err();
        assert!(
            format!("{err}").contains("dra_scale_value"),
            "expected dra_scale_value error, got: {err}"
        );
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_zero_global_offset() {
        // global_offset = 0 violates §7.4.7's "[1, Min(1023, …)]".
        let payload = emit_dra_syntax_minimal(0, true, 0, &[1], &[512], 512, 512, 58);
        let err = parse_dra_syntax(&payload, 10).unwrap_err();
        assert!(
            format!("{err}").contains("dra_global_offset"),
            "expected dra_global_offset error, got: {err}"
        );
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_overlarge_table_idx() {
        // dra_table_idx must be ≤ 58 — try 59.
        let payload = emit_dra_syntax_minimal(0, true, 1, &[1], &[512], 512, 512, 59);
        let err = parse_dra_syntax(&payload, 10).unwrap_err();
        assert!(
            format!("{err}").contains("dra_table_idx"),
            "expected dra_table_idx error, got: {err}"
        );
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_overlarge_num_ranges() {
        // dra_number_ranges_minus1 must be ≤ 31 — try 32.
        let mut e = BitEmitter::new();
        e.u(4, 4);
        e.u(4, 9);
        e.ue(32); // out of range
        e.finish_with_trailing_bits();
        let payload = e.into_bytes();
        let err = parse_dra_syntax(&payload, 10).unwrap_err();
        assert!(
            format!("{err}").contains("dra_number_ranges_minus1"),
            "expected dra_number_ranges_minus1 error, got: {err}"
        );
    }

    #[test]
    fn round151_parse_dra_syntax_rejects_in_dra_range_overflow() {
        // BitDepthY = 8 ⇒ shift = 0, max InDraRange = 255.
        // global_offset 200 + delta 100 ⇒ InDraRange[1] = 300 > 255.
        let payload = emit_dra_syntax_minimal(0, true, 200, &[100], &[512], 512, 512, 58);
        let err = parse_dra_syntax(&payload, 8).unwrap_err();
        assert!(
            format!("{err}").contains("InDraRange"),
            "expected InDraRange overflow error, got: {err}"
        );
    }

    #[test]
    fn round151_derive_dra_state_out_ranges_recursion_identity_scale() {
        // 3 equal-sized ranges, scale = 512 (= 1.0 in Q4.9), so each
        // outDelta = 512 * 100 = 51200, and OutRangesL post eq. 122
        // (>> 9, rounded) ≈ 100 per step ⇒ [0, 100, 200, 300].
        let syn = DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 2, // 3 ranges
            dra_equal_ranges_flag: true,
            dra_global_offset: 50,
            dra_delta_range: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 100;
                a
            },
            dra_scale_value: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 512;
                a[1] = 512;
                a[2] = 512;
                a
            },
            dra_cb_scale_value: 512,
            dra_cr_scale_value: 512,
            dra_table_idx: 58,
        };
        let der = derive_dra_state(&syn, 10).unwrap();
        assert_eq!(der.num_ranges, 3);
        assert_eq!(der.in_dra_range[0], 50);
        assert_eq!(der.in_dra_range[3], 350);
        // OutRangesL post eq. 122: each outDelta is 51200, divided by 512
        // (= 1 << 9) with eq. 122's +(1 << 8) rounding ⇒ exactly 100 per
        // step. Final values: 0, 100, 200, 300.
        assert_eq!(der.out_ranges_l[0], 0);
        assert_eq!(der.out_ranges_l[1], 100);
        assert_eq!(der.out_ranges_l[2], 200);
        assert_eq!(der.out_ranges_l[3], 300);
    }

    #[test]
    fn round151_derive_dra_state_inv_luma_scale_eq118_identity() {
        // Single non-trivial range: with dra_scale_value = 512 (1.0 in
        // Q4.9), eq. 118's invScale = ((1 << 18) + 256) / 512 = 512.5 →
        // 512 (integer division).
        let syn = DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 1, // 2 ranges
            dra_equal_ranges_flag: true,
            dra_global_offset: 16,
            dra_delta_range: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 64;
                a
            },
            dra_scale_value: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 512;
                a[1] = 512;
                a
            },
            dra_cb_scale_value: 512,
            dra_cr_scale_value: 512,
            dra_table_idx: 58,
        };
        let der = derive_dra_state(&syn, 10).unwrap();
        // eq. 118: invScale = ((1 << 18) + (512 >> 1)) / 512
        //                   = (262144 + 256) / 512 = 262400 / 512 = 512.
        assert_eq!(der.inv_luma_scales[1], 512);
    }

    #[test]
    fn round151_derive_dra_state_inv_luma_scale_eq118_doubled() {
        // dra_scale_value = 256 (= 0.5 in Q4.9) → eq. 118: invScale ≈ 1024.
        let syn = DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 1,
            dra_equal_ranges_flag: true,
            dra_global_offset: 16,
            dra_delta_range: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 64;
                a
            },
            dra_scale_value: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 256;
                a[1] = 256;
                a
            },
            dra_cb_scale_value: 256,
            dra_cr_scale_value: 256,
            dra_table_idx: 58,
        };
        let der = derive_dra_state(&syn, 10).unwrap();
        // (262144 + 128) / 256 = 1024.5 → 1024.
        assert_eq!(der.inv_luma_scales[1], 1024);
    }

    // -----------------------------------------------------------------
    // Round 174 — §8.9.3 luma inverse mapping (eq. 1374-1376).
    // -----------------------------------------------------------------

    /// Helper: build a `DraSyntax` for a 2-range identity DRA at 10-bit
    /// luma (every `dra_scale_value[j] = 512 = 1 << 9` reproduces the
    /// Q4.9 representation of scale 1.0).
    fn identity_two_range_syntax(bit_depth: u32) -> DraSyntax {
        let _ = bit_depth; // accepted for future extension, currently fixed
        let mut delta = [0u16; DRA_MAX_RANGES_V2];
        delta[0] = 64;
        let mut scales = [0u16; DRA_MAX_RANGES_V2];
        scales[0] = 512;
        scales[1] = 512;
        DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 1,
            dra_equal_ranges_flag: true,
            dra_global_offset: 16,
            dra_delta_range: delta,
            dra_scale_value: scales,
            dra_cb_scale_value: 512,
            dra_cr_scale_value: 512,
            dra_table_idx: 58,
        }
    }

    #[test]
    fn round174_literal_spec_range0_is_degenerate() {
        // Without the off-by-one fill, every sample whose value falls in
        // segment 0 (the lowest segment after §8.9.5) reads
        // InvLumaScales[0] = DraOffsets[0] = 0, so eq. 1375 collapses to
        // (0 + 0*sample + 256) >> 9 = 0. This is the documented
        // §7.4.7 docs-gap symptom (see module-level notes).
        let syn = identity_two_range_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        assert_eq!(der.inv_luma_scales[0], 0, "literal spec leaves [0] unset");
        assert_eq!(der.dra_offsets[0], 0, "literal spec leaves [0] unset");
        // Sample whose value is in segment 0's range maps to 0.
        let ranges = out_ranges_l_as_i32(&der);
        // OutRangesL is small in identity case; pick a sample value that
        // definitely falls in range 0 after §8.9.5. InDraRange[0] = 16,
        // InDraRange[1] = 80; OutRangesL[1] after eq. 122 ≈ 64. So a
        // small lumaSample like 10 falls into range 0.
        let mapped = map_one_luma_sample(10, &ranges, &der, 1023);
        assert_eq!(mapped, 0, "literal spec produces 0 in segment 0");
    }

    #[test]
    fn round174_fill_inv_luma_range0_restores_non_degenerate() {
        // After the off-by-one reconciliation, range-0 samples no longer
        // collapse to 0. We don't assert bit-exactness against the §7.4.7
        // identity (the spec's identity-DRA round-trip is itself
        // affected by the docs gap); we assert the qualitative
        // non-degeneracy: a non-zero, non-saturated, monotonic mapping.
        let syn = identity_two_range_syntax(10);
        let mut der = derive_dra_state(&syn, 10).unwrap();
        fill_inv_luma_scales_range_zero(&mut der, &syn).unwrap();
        assert_eq!(der.inv_luma_scales[0], 512, "eq. 118 at i=0 with scale=512");
        // Two distinct range-0 samples must map to two distinct outputs
        // (no degenerate flattening).
        let ranges = out_ranges_l_as_i32(&der);
        let m0 = map_one_luma_sample(5, &ranges, &der, 1023);
        let m1 = map_one_luma_sample(10, &ranges, &der, 1023);
        assert_ne!(m0, m1, "range-0 mapping must not be flat");
        assert!(m1 > m0, "range-0 mapping must be monotonic");
        // Mapping is bounded by Clip1Y.
        assert!((0..=1023).contains(&m0));
        assert!((0..=1023).contains(&m1));
    }

    #[test]
    fn round174_fill_range0_rejects_zero_scale() {
        // Defence-in-depth: dra_scale_value[0] = 0 is already forbidden
        // by the parser; the helper rejects it explicitly so a
        // hand-constructed DraSyntax doesn't trigger a divide-by-zero.
        let mut syn = identity_two_range_syntax(10);
        syn.dra_scale_value[0] = 0;
        let mut der = DraDerived::empty();
        der.num_ranges = 2;
        assert!(fill_inv_luma_scales_range_zero(&mut der, &syn).is_err());
    }

    #[test]
    fn round174_fill_range0_noop_when_num_ranges_zero() {
        // Empty DraDerived → no-op (no division, no error).
        let syn = identity_two_range_syntax(10);
        let mut der = DraDerived::empty();
        // num_ranges intentionally left at 0.
        assert!(fill_inv_luma_scales_range_zero(&mut der, &syn).is_ok());
        assert_eq!(der.inv_luma_scales[0], 0);
        assert_eq!(der.dra_offsets[0], 0);
    }

    #[test]
    fn round174_apply_eq1376_clips_to_bit_depth() {
        // Construct a DraDerived that would, without Clip1Y, produce a
        // mapped value outside [0, (1 << bit_depth_y) − 1]. The clip
        // brings it back into range.
        let mut der = DraDerived::empty();
        der.num_ranges = 1;
        // Single range covers [0, 1024) at 10-bit.
        der.out_ranges_l[0] = 0;
        der.out_ranges_l[1] = 1024;
        // Pick an InvLumaScales that, multiplied by a maxed-out sample,
        // overshoots: invScale * 1023 + (dra_offset + 256) >> 9 ≫ 1023.
        der.inv_luma_scales[0] = 1024; // 2× nominal
        der.dra_offsets[0] = 100_000; // bias upward
                                      // ranges array for find_range_idx.
        let mapped = map_one_luma_sample(1023, &out_ranges_l_as_i32(&der), &der, 1023);
        assert_eq!(mapped, 1023, "Clip1Y caps at (1<<10) − 1");
    }

    #[test]
    fn round174_apply_eq1376_clips_negative_to_zero() {
        // Symmetric clip: very negative DraOffsets pushes mappedSample
        // negative; Clip1Y(_) at lower bound is 0.
        let mut der = DraDerived::empty();
        der.num_ranges = 1;
        der.out_ranges_l[0] = 0;
        der.out_ranges_l[1] = 1024;
        der.inv_luma_scales[0] = 1;
        der.dra_offsets[0] = -1_000_000;
        let mapped = map_one_luma_sample(512, &out_ranges_l_as_i32(&der), &der, 1023);
        assert_eq!(mapped, 0);
    }

    #[test]
    fn round174_apply_u16_in_place() {
        // Plane-level apply: every sample is run through eq. 1374-1376.
        // Use a degenerate-by-design DraDerived where InvLumaScales[0]
        // is non-zero (skipping the spec-literal range-0 trap) so we can
        // verify the loop body, not the gap.
        let mut der = DraDerived::empty();
        der.num_ranges = 1;
        der.out_ranges_l[0] = 0;
        der.out_ranges_l[1] = 1024;
        der.inv_luma_scales[0] = 512; // Q18 representation of 1.0
        der.dra_offsets[0] = 0;
        // Per-sample: mapped = (0 + 512*v + 256) >> 9 = v + (256 >> 9) = v.
        let mut plane = vec![0u16, 100, 500, 1023];
        apply_luma_inverse_mapping(&mut plane, &der, 10);
        assert_eq!(plane, vec![0u16, 100, 500, 1023]);
    }

    #[test]
    fn round174_apply_u8_in_place_matches_lut() {
        // 8-bit shortcut: u8 plane apply equals LUT-of-LUT lookup.
        let mut der = DraDerived::empty();
        der.num_ranges = 1;
        der.out_ranges_l[0] = 0;
        der.out_ranges_l[1] = 256;
        der.inv_luma_scales[0] = 512;
        der.dra_offsets[0] = 0;
        let lut = build_inv_luma_lut_8bit(&der);
        let mut plane = vec![0u8, 1, 50, 128, 200, 255];
        let expected: Vec<u8> = plane.iter().map(|&v| lut[v as usize]).collect();
        apply_luma_inverse_mapping_u8(&mut plane, &der);
        assert_eq!(plane, expected);
        // For this identity-Q18 case the LUT is identity.
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8, "LUT[{i}] should be {i}");
        }
    }

    #[test]
    fn round174_apply_noop_when_num_ranges_zero() {
        // Empty derived: helper is a no-op, plane is unchanged.
        let der = DraDerived::empty();
        let mut plane = vec![10u16, 100, 1000];
        apply_luma_inverse_mapping(&mut plane, &der, 10);
        assert_eq!(plane, vec![10u16, 100, 1000]);
        let mut plane8 = vec![10u8, 100, 250];
        apply_luma_inverse_mapping_u8(&mut plane8, &der);
        assert_eq!(plane8, vec![10u8, 100, 250]);
    }

    #[test]
    fn round174_build_lut_identity_when_num_ranges_zero() {
        let der = DraDerived::empty();
        let lut = build_inv_luma_lut_8bit(&der);
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8);
        }
    }

    #[test]
    fn round174_apply_multi_range_segment_dispatch() {
        // Two ranges with distinct InvLumaScales/DraOffsets — verify
        // §8.9.5 picks the right one and eq. 1374-1376 applies
        // per-range. Construct OutRangesL = [0, 128, 256] (mid-point
        // split for an 8-bit sample space) and check a value below /
        // above 128 maps via the correct entry.
        let mut der = DraDerived::empty();
        der.num_ranges = 2;
        der.out_ranges_l[0] = 0;
        der.out_ranges_l[1] = 128;
        der.out_ranges_l[2] = 256;
        // Range 0: scale = 1.0 (Q18 = 512), offset = 0.
        der.inv_luma_scales[0] = 512;
        der.dra_offsets[0] = 0;
        // Range 1: scale = 2.0 (Q18 = 1024), offset = -65_536 to
        // re-anchor (1024*128 - 65536 = 65536 vs 512*128 = 65536 ≈
        // continuity at the boundary).
        der.inv_luma_scales[1] = 1024;
        der.dra_offsets[1] = -65_536;
        let ranges = out_ranges_l_as_i32(&der);
        // Below split: range 0 path → v.
        let v_lo = map_one_luma_sample(64, &ranges, &der, 255);
        // (0 + 512*64 + 256) >> 9 = 32768 + 256 >> 9 = 64.
        assert_eq!(v_lo, 64);
        // At split: range 1 path → 2*v − 128.
        let v_hi = map_one_luma_sample(200, &ranges, &der, 255);
        // (−65536 + 1024*200 + 256) >> 9 = (−65536 + 204800 + 256) >> 9
        //                                 = 139520 >> 9 = 272 → Clip1Y(255) = 255.
        assert_eq!(v_hi, 255);
        // A middle range-1 value just above the split.
        let v_mid = map_one_luma_sample(150, &ranges, &der, 255);
        // (−65536 + 153600 + 256) >> 9 = 88320 >> 9 = 172.
        assert_eq!(v_mid, 172);
    }
}
