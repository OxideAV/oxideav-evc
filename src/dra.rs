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
// Round 187 — §8.9.7 / §8.9.8 chroma derived state + §8.9.6 chromaScale
// (DraJoinedScaleFlag = 0 path).
// =====================================================================
//
// Round 174 / 181 landed the §8.9.3 luma inverse mapping helpers; the
// outstanding "lacks" tail on the workspace README row is the matching
// chroma derivation (eq. 1384-1385 for the §8.9.6 entry point;
// eq. 1386-1393 for the §8.9.7 derived state; eq. 1394 for the §8.9.8
// adjusted chroma DRA scales).
//
// §8.9.8 has two branches:
//
//   - `DraJoinedScaleFlag == 0` (the simpler path, `dra_table_idx == 58`
//     per §7.4.7 page 85): `chromaScale = (cIdx == 0) ? dra_cb_scale_value
//     : dra_cr_scale_value` (eq. 1394). The chroma scale is independent
//     of the luma scale entry: `chromaScales[cIdx][i]` collapses to the
//     same value for every `i ∈ [0, numRangesL − 1]`.
//
//   - `DraJoinedScaleFlag == 1` (the table-driven path,
//     `dra_table_idx ∈ [0, 57]`): runs the long eq. 1395-1419 chain
//     against `ChromaQpTable` (which is itself an SPS-derived table per
//     §7.4.3 page 78) and the §8.9.5 `ScaleQP` / `QpScale` tables
//     (eq. 1420-1421). This requires plumbing through the active SPS's
//     `ChromaQpTable` (an inter-crate surface), so round 187 surfaces
//     the joined path as a documented `Err(Error::Unsupported)` and
//     leaves it for a follow-up round once the SPS half is wired in.
//
// Round 187 lands the §8.9.7 + §8.9.8-unjoined chain end-to-end so
// callers can apply §8.9.6 chroma scaling for the `DraJoinedScaleFlag = 0`
// path; the joined path is signalled at the derive-time entry point so
// the caller surfaces the gap rather than silently producing the wrong
// scale.
//
// Stays bit-faithful to the spec: every eq. number quoted in a code
// comment below is a direct transcription of the ISO/IEC 23094-1:2020(E)
// PDF.

/// Maximum size of the §8.9.7 chroma-range arrays. `numRangesC =
/// dra_number_ranges_minus1 + 2`, so the array has `DRA_MAX_RANGES_V2 + 1`
/// in-spec entries (32 + 1 = 33). §8.9.5 indexes one entry past the
/// `numRanges`-th slot (the top sentinel), so the storage needs
/// `DRA_MAX_RANGES_V2 + 2 = 34` slots.
pub const DRA_MAX_RANGES_C: usize = DRA_MAX_RANGES_V2 + 2;

/// §8.9.7-derived per-APS chroma DRA state for a single chroma component
/// `cIdx` (Cb when `cIdx == 0`, Cr when `cIdx == 1`).
///
/// Built by [`derive_dra_chroma_state`] from a [`DraSyntax`] +
/// [`DraDerived`] pair (the round-151 luma half).
///
/// Layout mirrors §8.9.7 (page 306):
///
/// * `out_ranges_c[0..=num_ranges_l]` — boundary array indexed
///   `[0, numRangesC − 1] = [0, numRangesL]`. Index 0 is
///   `OutRangesL[0] = 0` (the line before eq. 1387). Indices
///   `[1, numRangesL]` come from eq. 1387 (`(OutRangesL[i] +
///   OutRangesL[i − 1]) >> 1`). The total of `numRangesL + 1` boundaries
///   matches the §8.9.5 invocation that takes
///   `dra_number_ranges_minus1 + 2 = numRangesC` as its `numRanges`.
/// * `chroma_scales[i]` — eq. 1394 evaluated for `cIdx`; constant
///   across `i ∈ [0, numRangesL − 1]` under `DraJoinedScaleFlag == 0`.
/// * `inv_chroma_scales[i]` — eq. 1386 for `i ∈ [0, numRangesL − 1]`.
/// * `out_scales_c[i]` / `out_offsets_c[i]` for `i ∈
///   [0, numRangesC − 1] = [0, numRangesL]`:
///   - `i = 0`: explicit `OutScalesC[0] = OutOffsetsC[0] = 0` +
///     `OutRangesC[0] = OutRangesL[0]` per the line above eq. 1387.
///     The fact that `OutOffsetsC[0] = invChromaScales[0]` is also a
///     valid reading: spec page 306 line above eq. 1387 says
///     "Variables OutScalesC[cIdx][0], OutOffsetsC[cIdx][0] and
///     OutRangesC[0] are set equal to 0, invChromaScales[cIdx][0] and
///     OutRangesL[0], respectively." — i.e. `OutOffsetsC[0] =
///     invChromaScales[0]`, not 0. The "respectively" pattern makes
///     the spec text ambiguous at first reading; round 187 follows
///     the parallel structure of the sentence (three left-hand
///     variables matched to three right-hand values position-wise),
///     so `OutOffsetsC[0]` is set equal to `invChromaScales[0]`,
///     `OutScalesC[0]` to 0, `OutRangesC[0]` to `OutRangesL[0]`.
///   - `i ∈ [1, numRangesL − 1]`: eq. 1391 + eq. 1389.
///   - `i = numRangesL`: eq. 1392 + eq. 1393.
#[derive(Clone, Debug)]
pub struct DraChromaDerived {
    /// Which chroma component this state was derived for.
    pub cidx: ChromaIdx,
    /// `numRangesL = dra_number_ranges_minus1 + 1`.
    pub num_ranges_l: usize,
    /// `OutRangesC[0..=num_ranges_l]` — the §8.9.5 `rangesArray` input
    /// for §8.9.6, with `numRangesC = num_ranges_l + 1` total entries.
    pub out_ranges_c: [i64; DRA_MAX_RANGES_C],
    /// `chromaScales[cIdx][i]` per eq. 1394 (DraJoinedScaleFlag = 0).
    /// Sized to `num_ranges_l` entries.
    pub chroma_scales: [i64; DRA_MAX_RANGES_V2],
    /// `invChromaScales[cIdx][i]` per eq. 1386.
    pub inv_chroma_scales: [i64; DRA_MAX_RANGES_V2],
    /// `OutScalesC[cIdx][i]` for `i ∈ [0, numRangesC − 1]` — §8.9.6
    /// reads `OutScalesC[cIdx][rangeIdx]` in eq. 1385.
    pub out_scales_c: [i64; DRA_MAX_RANGES_C],
    /// `OutOffsetsC[cIdx][i]` for `i ∈ [0, numRangesC − 1]` — §8.9.6
    /// reads `OutOffsetsC[cIdx][rangeIdx]` in eq. 1385.
    pub out_offsets_c: [i64; DRA_MAX_RANGES_C],
}

/// Chroma component index used by §8.9.6 / §8.9.7 / §8.9.8. EVC has two
/// chroma components: `Cb` (`cIdx == 0`) and `Cr` (`cIdx == 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaIdx {
    /// Cb — `cIdx == 0`. Eq. 1394 picks `dra_cb_scale_value`.
    Cb,
    /// Cr — `cIdx == 1`. Eq. 1394 picks `dra_cr_scale_value`.
    Cr,
}

impl ChromaIdx {
    /// Numeric value of `cIdx` per spec convention.
    #[inline]
    pub fn as_u32(self) -> u32 {
        match self {
            ChromaIdx::Cb => 0,
            ChromaIdx::Cr => 1,
        }
    }
}

impl DraChromaDerived {
    /// Zero-initialised state for a `cidx` with no ranges populated. The
    /// §8.9.6 helpers no-op on this (zero `num_ranges_l`).
    fn empty(cidx: ChromaIdx) -> Self {
        Self {
            cidx,
            num_ranges_l: 0,
            out_ranges_c: [0i64; DRA_MAX_RANGES_C],
            chroma_scales: [0i64; DRA_MAX_RANGES_V2],
            inv_chroma_scales: [0i64; DRA_MAX_RANGES_V2],
            out_scales_c: [0i64; DRA_MAX_RANGES_C],
            out_offsets_c: [0i64; DRA_MAX_RANGES_C],
        }
    }
}

/// Derive the §8.9.7 chroma DRA state for chroma component `cidx`, given
/// the parsed [`DraSyntax`] + the round-151 §7.4.7 [`DraDerived`].
///
/// Runs eq. 1386-1393 transcribed verbatim. For the §8.9.8 chromaScales
/// derivation it follows the `DraJoinedScaleFlag == 0` branch (eq. 1394)
/// — the simpler "use `dra_cb_scale_value` / `dra_cr_scale_value`
/// directly" path.
///
/// Returns an [`Error::Unsupported`] when `derived.joined_scale_flag` is
/// `true` (the `dra_table_idx != 58` joined path; eq. 1395-1419 — needs
/// the SPS's `ChromaQpTable` which round 187 does not thread through).
///
/// Returns an [`Error::Invalid`] when:
///
/// * `dra_cb_scale_value == 0` (cIdx = Cb path); a zero divides
///   eq. 1386's `(1 << 18) / chromaScales[0]`.
/// * `dra_cr_scale_value == 0` (cIdx = Cr path), same reason.
///
/// Both cases are forbidden by §7.4.7's scale-value range
/// `[1, (4 << dra_descriptor2) − 1]`, but we check defensively so a
/// malformed APS doesn't panic on division.
pub fn derive_dra_chroma_state(
    syntax: &DraSyntax,
    derived: &DraDerived,
    cidx: ChromaIdx,
    bit_depth_y: u32,
) -> Result<DraChromaDerived> {
    if derived.joined_scale_flag {
        return Err(Error::unsupported(
            "evc dra: §8.9.8 DraJoinedScaleFlag = 1 (joined chroma scale via ChromaQpTable) \
             requires the joined entry point — call \
             `derive_dra_chroma_state_joined(syntax, derived, cidx, bit_depth_y, &chroma_qp_table)` \
             with a `ChromaQpTable` from `default_chroma_qp_table` (no signalled SPS table) \
             or from the SPS-signalled chroma QP mapping (round-193 followup)",
        ));
    }
    if !(8..=16).contains(&bit_depth_y) {
        return Err(Error::invalid(
            "evc dra: bit_depth_y must be in [8, 16] (per SPS §7.4.3.1)",
        ));
    }
    if derived.num_ranges == 0 {
        return Ok(DraChromaDerived::empty(cidx));
    }

    let mut c = DraChromaDerived::empty(cidx);
    c.num_ranges_l = derived.num_ranges;
    let num_ranges_l = c.num_ranges_l;
    // numRangesC = dra_number_ranges_minus1 + 2 = num_ranges_l + 1.
    let num_ranges_c = num_ranges_l + 1;

    // eq. 1394 (§8.9.8 DraJoinedScaleFlag = 0 path): chromaScale =
    // (cIdx == 0) ? dra_cb_scale_value : dra_cr_scale_value.
    // Independent of `i` — so `chromaScales[cIdx][i]` is constant across
    // the luma range index `i`.
    let scale_value = match cidx {
        ChromaIdx::Cb => syntax.dra_cb_scale_value as i64,
        ChromaIdx::Cr => syntax.dra_cr_scale_value as i64,
    };
    if scale_value == 0 {
        return Err(Error::invalid(
            "evc dra: chroma scale value == 0 (forbidden by §7.4.7); cannot derive invChromaScales",
        ));
    }

    // eq. 1386: invChromaScales[cIdx][i] = ((1 << 18) + (chromaScales[cIdx][i] >> 1))
    //                                       / chromaScales[cIdx][i].
    // Since chromaScales is constant across `i` under DraJoinedScaleFlag = 0,
    // invChromaScales is also constant.
    let inv_scale = ((1i64 << 18) + (scale_value >> 1)) / scale_value;
    for i in 0..num_ranges_l {
        c.chroma_scales[i] = scale_value;
        c.inv_chroma_scales[i] = inv_scale;
    }

    // Line above eq. 1387: "Variables OutScalesC[cIdx][0],
    // OutOffsetsC[cIdx][0] and OutRangesC[0] are set equal to 0,
    // invChromaScales[cIdx][0] and OutRangesL[0], respectively."
    c.out_scales_c[0] = 0;
    c.out_offsets_c[0] = inv_scale;
    c.out_ranges_c[0] = derived.out_ranges_l[0];

    // eq. 1387: OutRangesC[i] = (OutRangesL[i] + OutRangesL[i − 1]) >> 1
    // for i in [1, numRangesC − 1] = [1, numRangesL].
    for i in 1..num_ranges_c {
        c.out_ranges_c[i] = (derived.out_ranges_l[i] + derived.out_ranges_l[i - 1]) >> 1;
    }

    // eq. 1388-1391: for i in [1, dra_number_ranges_minus1] =
    // [1, num_ranges_l − 1].
    for i in 1..num_ranges_l {
        // eq. 1388: deltaRange = OutRangesC[i + 1] − OutRangesC[i].
        let delta_range = c.out_ranges_c[i + 1] - c.out_ranges_c[i];
        // eq. 1389: OutOffsetsC[cIdx][i] = invChromaScales[cIdx][i − 1].
        c.out_offsets_c[i] = c.inv_chroma_scales[i - 1];
        // eq. 1390: deltaScale = invChromaScales[cIdx][i] − invChromaScales[cIdx][i − 1].
        let delta_scale = c.inv_chroma_scales[i] - c.inv_chroma_scales[i - 1];
        // eq. 1391: OutScalesC[cIdx][i] = ((deltaScale << 10) + (deltaRange >> 1)) / deltaRange.
        // delta_range == 0 would divide by zero — that requires
        // OutRangesC[i + 1] == OutRangesC[i], which only happens with
        // degenerate DRA syntax (zero-width range). Guard defensively.
        c.out_scales_c[i] = if delta_range == 0 {
            0
        } else {
            ((delta_scale << 10) + (delta_range >> 1)) / delta_range
        };
    }

    // eq. 1392-1393: i = numRangesL.
    // OutOffsetsC[cIdx][numRangesL] = invChromaScales[cIdx][numRangesL − 1].
    c.out_offsets_c[num_ranges_l] = c.inv_chroma_scales[num_ranges_l - 1];
    // OutScalesC[cIdx][numRangesL] = 0.
    c.out_scales_c[num_ranges_l] = 0;

    // §8.9.5 top-sentinel: §8.9.6 invokes §8.9.5 with `numRanges =
    // dra_number_ranges_minus1 + 2 = num_ranges_c`, which reads
    // `rangesArray[rangeIdx + 1]` for `rangeIdx ∈ [0, num_ranges_c − 1]`
    // — so the final iteration reads `rangesArray[num_ranges_c]`. §8.9.7
    // only derives OutRangesC up to `[num_ranges_c − 1] = [num_ranges_l]`,
    // so we synthesise a top sentinel at `1 << bit_depth_y` (the sample-
    // space upper bound), matching round 148's `build_ranges_array` for
    // the luma side. Without this, samples larger than every internal
    // chroma boundary would index past the derived state's end.
    c.out_ranges_c[num_ranges_c] = 1i64 << bit_depth_y;

    Ok(c)
}

/// Apply §8.9.6 to a single luma sample: derive the §8.9.6 `chromaScale`
/// for a given luma sample value, using the round-187 [`DraChromaDerived`]
/// state.
///
/// Spec text:
///
/// ```text
/// rangeIdx = §8.9.5(lumaSample, OutRangesC, dra_number_ranges_minus1 + 2)
/// incValue = lumaSample − OutRangesC[cIdx][rangeIdx]                  (1384)
/// chromaScale = OutOffsetsC[apsId][cIdx][rangeIdx] +
///   (OutScalesC[apsId][cIdx][rangeIdx] * incValue + (1 << 9)) >> 10   (1385)
/// ```
///
/// `chroma_derived` must be the [`DraChromaDerived`] returned by
/// [`derive_dra_chroma_state`] (it already encodes `cIdx`). The caller
/// passes `lumaSample` as the **decoded pre-mapping luma sample at the
/// co-located position** (§8.9.2 input contract; see round 148 for the
/// matching subsampling logic).
///
/// Returns the §8.9.6 `chromaScale` as `i64` so callers can apply
/// eq. 1378 (signed-magnitude chroma multiply) without losing
/// precision. Returns 0 when `num_ranges_l == 0` (empty state).
pub fn chroma_scale_for_luma_sample(luma_sample: i64, chroma_derived: &DraChromaDerived) -> i64 {
    if chroma_derived.num_ranges_l == 0 {
        return 0;
    }
    // §8.9.5 over OutRangesC with numRanges = numRangesC = num_ranges_l + 1.
    let ranges_array = out_ranges_c_as_i32(chroma_derived);
    let num_ranges_c = chroma_derived.num_ranges_l + 1;
    let range_idx = find_range_idx(luma_sample as i32, &ranges_array, num_ranges_c);
    // eq. 1384.
    let inc_value = luma_sample - chroma_derived.out_ranges_c[range_idx];
    // eq. 1385.
    chroma_derived.out_offsets_c[range_idx]
        + ((chroma_derived.out_scales_c[range_idx] * inc_value + (1i64 << 9)) >> 10)
}

// =====================================================================
// Round 193 — §8.9.8 DraJoinedScaleFlag = 1 (joined chroma scale via
// ChromaQpTable) + default Table 5/6 ChromaQpTable builder.
// =====================================================================
//
// Round 187 surfaced the joined path (`dra_table_idx ∈ [0, 57]`) as a
// documented `Err(Error::Unsupported)` because the eq. 1395-1419 chain
// needs `ChromaQpTable[cIdx][qPi]` — itself an SPS-derived table per
// §7.4.3 page 67-68 — to translate the integer + fractional `qpDra`
// shifts into a `draChromaScaleShift`. The actual joined chain is
// otherwise a pure function of `(lumaScale, dra_cb_scale_value /
// dra_cr_scale_value, dra_table_idx, ChromaQpTable, ScaleQP, QpScale)`.
//
// Round 193 lands the joined path against an externally-supplied
// `ChromaQpTable`, plus a `default_chroma_qp_table()` builder for the
// `chroma_qp_table_present_flag == 0` case — the simpler and more
// common path where the SPS does not signal a user-defined chroma QP
// mapping and the spec falls back to Table 5 (`sps_iqt_flag == 0`) or
// Table 6 (`sps_iqt_flag == 1`).
//
// The SPS-signalled chroma-QP-table path (eq. 74, with `qpInVal[][]`
// / `qpOutVal[][]` interpolation across user-supplied pivot points)
// stays a documented gap: round 193 has the consumer side complete
// — callers can hand-build a `ChromaQpTable` from any source — but
// the SPS parser still discards `delta_qp_in_val_minus1` and
// `delta_qp_out_val`. That's the round-194 follow-up.
//
// Stays bit-faithful: every eq. number quoted below is a verbatim
// transcription of ISO/IEC 23094-1:2020(E).

/// §8.9.8 eq. 1420: `ScaleQP[]` — the 55-entry integer-QP scale table
/// indexed by `IndexScaleQP ∈ [0, 53]` (one extra trailing entry so
/// eq. 1399's `ScaleQP[IndexScaleQP + 1]` is in-bounds at the top).
pub const SCALE_QP: [i64; 55] = [
    0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 6, 7, 9, 11, 14, 18, 23, 29, 36, 45, 57, 72, 91, 114, 144,
    181, 228, 287, 362, 456, 575, 724, 912, 1149, 1448, 1825, 2299, 2896, 3649, 4598, 5793, 7298,
    9195, 11585, 14596, 18390, 23170, 29193, 36781, 46341, 58386, 73562, 92682, 116772,
];

/// §8.9.8 eq. 1421: `QpScale[]` — the 25-entry inverse-scale table
/// indexed by `idx ∈ [0, 24]`.
pub const QP_SCALE: [i64; 25] = [
    128, 144, 161, 181, 203, 228, 256, 287, 322, 362, 406, 456, 512, 574, 645, 724, 812, 912, 1024,
    1149, 1290, 1448, 1625, 1825, 2048,
];

/// `ChromaQpTable` for a single chroma component, indexed by
/// `qPi ∈ [−QpBdOffsetC, 57]`.
///
/// `QpBdOffsetC = 6 * bit_depth_chroma_minus8`. With
/// `bit_depth_chroma_minus8 ∈ [0, 8]`, the maximum `QpBdOffsetC` is 48
/// and the maximum table length is `48 + 57 + 1 = 106`. The table
/// stores values flat-packed at offset `qp_bd_offset_c` so that
/// `table[(qPi + qp_bd_offset_c) as usize] = ChromaQpTable[qPi]`.
///
/// Use [`ChromaQpTableEntry::lookup`] to index by `qPi` directly with
/// spec-faithful `Clip3(−QpBdOffsetC, 57, qPi)` clamping.
#[derive(Clone, Debug)]
pub struct ChromaQpTableEntry {
    /// `QpBdOffsetC = 6 * bit_depth_chroma_minus8`. The table is indexed
    /// by `qPi ∈ [−qp_bd_offset_c, 57]` with the `−qp_bd_offset_c`
    /// origin packed at vector index 0.
    pub qp_bd_offset_c: i32,
    /// Flat-packed `ChromaQpTable[qPi]` for
    /// `qPi ∈ [−qp_bd_offset_c, 57]`.
    /// Length is `qp_bd_offset_c + 57 + 1 = qp_bd_offset_c + 58`.
    pub table: Vec<i32>,
}

impl ChromaQpTableEntry {
    /// `ChromaQpTable[qPi]` with the spec's `Clip3(−QpBdOffsetC, 57, qPi)`
    /// clamping per eq. 1403 / 1404. Returns the clamped output value
    /// (`QpC` in Tables 5/6).
    #[inline]
    pub fn lookup(&self, qpi: i32) -> i32 {
        let lo = -self.qp_bd_offset_c;
        let hi = 57;
        let clamped = qpi.clamp(lo, hi);
        let idx = (clamped + self.qp_bd_offset_c) as usize;
        self.table[idx]
    }
}

/// Both `ChromaQpTable[0]` (Cb) and `ChromaQpTable[1]` (Cr).
///
/// When the SPS sets `chroma_qp_table_present_flag == 0`, both tables
/// derive from Table 5 (`sps_iqt_flag == 0`) or Table 6 (`sps_iqt_flag
/// == 1`) and are identical across components — use
/// [`default_chroma_qp_table`] to build that case.
///
/// When `chroma_qp_table_present_flag == 1` with
/// `same_qp_table_for_chroma == 1`, both entries hold the same parsed
/// table; with `same_qp_table_for_chroma == 0`, the two entries differ.
#[derive(Clone, Debug)]
pub struct ChromaQpTable {
    /// `ChromaQpTable[0]` — Cb component.
    pub cb: ChromaQpTableEntry,
    /// `ChromaQpTable[1]` — Cr component.
    pub cr: ChromaQpTableEntry,
}

impl ChromaQpTable {
    /// Look up `ChromaQpTable[cIdx][qPi]` with eq. 1403 / 1404 clamping.
    #[inline]
    pub fn lookup(&self, cidx: ChromaIdx, qpi: i32) -> i32 {
        match cidx {
            ChromaIdx::Cb => self.cb.lookup(qpi),
            ChromaIdx::Cr => self.cr.lookup(qpi),
        }
    }
}

/// Build the spec-page-67 default `ChromaQpTable` for the
/// `chroma_qp_table_present_flag == 0`, `ChromaArrayType == 1` (4:2:0)
/// path.
///
/// `sps_iqt_flag == 0` → Table 5; `sps_iqt_flag == 1` → Table 6. Both
/// tables produce the same value for Cb and Cr (spec page 67: "for m
/// being equal to 0 and 1"), so the returned [`ChromaQpTable`] has
/// `cb == cr` byte-for-byte.
///
/// `bit_depth_chroma_minus8` must be in `[0, 8]`. The resulting table
/// is indexed by `qPi ∈ [−QpBdOffsetC, 57]` where
/// `QpBdOffsetC = 6 * bit_depth_chroma_minus8`.
pub fn default_chroma_qp_table(
    sps_iqt_flag: bool,
    bit_depth_chroma_minus8: u32,
) -> Result<ChromaQpTable> {
    if bit_depth_chroma_minus8 > 8 {
        return Err(Error::invalid(
            "evc dra: bit_depth_chroma_minus8 must be in [0, 8]",
        ));
    }
    let qp_bd_offset_c = 6 * bit_depth_chroma_minus8 as i32;
    let len = (qp_bd_offset_c + 58) as usize;
    let mut tbl = Vec::with_capacity(len);
    for qpi in -qp_bd_offset_c..=57 {
        let qp_c = if sps_iqt_flag {
            // Table 6.
            table6_qp_c(qpi)
        } else {
            // Table 5.
            table5_qp_c(qpi)
        };
        tbl.push(qp_c);
    }
    let cb = ChromaQpTableEntry {
        qp_bd_offset_c,
        table: tbl.clone(),
    };
    let cr = ChromaQpTableEntry {
        qp_bd_offset_c,
        table: tbl,
    };
    Ok(ChromaQpTable { cb, cr })
}

/// Build the spec-page-67 identity `ChromaQpTable` for the
/// `chroma_qp_table_present_flag == 0` "Otherwise" branch — every
/// `ChromaArrayType` value other than 1 (i.e. 0 = monochrome,
/// 2 = 4:2:2, 3 = 4:4:4).
///
/// Per spec page 67 (lines 3960-3961): "Otherwise, `ChromaQpTable[m][qPi]`
/// with `m` being equal to 0 and 1, and `qPi` being in the range of
/// `−QpBdOffsetC` to 57 are set equal to the value of `qPi`." Both Cb and
/// Cr are byte-for-byte identical.
///
/// `bit_depth_chroma_minus8` must be in `[0, 8]`. The resulting table is
/// indexed by `qPi ∈ [−QpBdOffsetC, 57]` where `QpBdOffsetC = 6 *
/// bit_depth_chroma_minus8`. The stored value at each `qPi` equals
/// `qPi` itself, so `ChromaQpTable.lookup(cidx, qPi) == qPi` for every
/// in-range `qPi`. Out-of-range lookups still receive the spec's
/// `Clip3(−QpBdOffsetC, 57, qPi)` clamping from
/// [`ChromaQpTableEntry::lookup`].
///
/// # Use
///
/// On a monochrome (`ChromaArrayType == 0`) stream, the §7.4.3.1 parser
/// never enters the `chroma_qp_table_present_flag` body, so
/// [`crate::sps::Sps::chroma_qp_table`] stays `None`. A consumer that
/// nonetheless needs the table — e.g. an exploratory §8.9.8 joined-path
/// invocation that doesn't short-circuit on monochrome — should call
/// this helper. On 4:2:2 / 4:4:4 streams with
/// `chroma_qp_table_present_flag == 0`, the same identity rule applies.
///
/// Prefer [`chroma_qp_table_for_sps`] when the SPS is available; it
/// dispatches between [`default_chroma_qp_table`] (4:2:0),
/// this helper (non-4:2:0 with `chroma_qp_table_present_flag == 0`),
/// and the parsed `sps.chroma_qp_table` (any `ChromaArrayType != 0` with
/// `chroma_qp_table_present_flag == 1`).
pub fn default_chroma_qp_table_identity(bit_depth_chroma_minus8: u32) -> Result<ChromaQpTable> {
    if bit_depth_chroma_minus8 > 8 {
        return Err(Error::invalid(
            "evc dra: bit_depth_chroma_minus8 must be in [0, 8]",
        ));
    }
    let qp_bd_offset_c = 6 * bit_depth_chroma_minus8 as i32;
    let len = (qp_bd_offset_c + 58) as usize;
    let mut tbl = Vec::with_capacity(len);
    for qpi in -qp_bd_offset_c..=57 {
        tbl.push(qpi);
    }
    let cb = ChromaQpTableEntry {
        qp_bd_offset_c,
        table: tbl.clone(),
    };
    let cr = ChromaQpTableEntry {
        qp_bd_offset_c,
        table: tbl,
    };
    Ok(ChromaQpTable { cb, cr })
}

/// Resolve the active `ChromaQpTable` for an SPS — the SPS → table
/// adapter that closes the round-195 "joined entry doesn't yet read
/// `Sps::chroma_qp_table` automatically" follow-up.
///
/// Dispatch (per ISO/IEC 23094-1:2020(E) §7.4.3.1 page 67):
///
/// * `chroma_qp_table_present_flag == 1` (i.e. `sps.chroma_qp_table` is
///   `Some`) → returns that parsed table verbatim. The §7.4.3.1 parser
///   only populates this on `ChromaArrayType != 0` (the spec's syntax
///   gate at line 2177), so a `Some` here implies a non-monochrome
///   stream.
/// * `chroma_qp_table_present_flag == 0` AND `chroma_format_idc == 1`
///   (4:2:0) → returns [`default_chroma_qp_table`] for the SPS's
///   `sps_iqt_flag` and `bit_depth_chroma_minus8`. Table 5
///   (`sps_iqt_flag == 0`) / Table 6 (`sps_iqt_flag == 1`).
/// * `chroma_qp_table_present_flag == 0` AND `chroma_format_idc != 1`
///   (monochrome, 4:2:2, 4:4:4) → returns
///   [`default_chroma_qp_table_identity`] per the spec page-67
///   "Otherwise" branch.
///
/// This lets §8.9.8 callers — including
/// [`derive_dra_chroma_state_joined`] — pull the right table from a
/// parsed [`crate::sps::Sps`] without re-implementing the three-way
/// dispatch at every call site.
///
/// Note: on a strictly monochrome stream there is no chroma plane to
/// run DRA on, so the returned identity table is provided as a safe
/// fallback for tooling that wants to invoke §8.9.8 introspectively
/// (e.g. unit tests on the joined chain). Production paths should
/// short-circuit chroma processing entirely when
/// `chroma_format_idc == 0`.
pub fn chroma_qp_table_for_sps(sps: &crate::sps::Sps) -> Result<ChromaQpTable> {
    if let Some(tbl) = sps.chroma_qp_table.as_ref() {
        return Ok(tbl.clone());
    }
    if sps.chroma_format_idc == 1 {
        default_chroma_qp_table(sps.sps_iqt_flag, sps.bit_depth_chroma_minus8)
    } else {
        default_chroma_qp_table_identity(sps.bit_depth_chroma_minus8)
    }
}

/// SPS-signalled chroma QP mapping table parameters — what the §7.3.2.1
/// `chroma_qp_table_present_flag == 1` body carries through to the eq. 74
/// derivation.
///
/// Layout matches the spec: a one- or two-entry table set, each row a
/// vector of `num_points_in_qp_table_minus1 + 1` pivot points. When
/// `same_qp_table_for_chroma == 1` only `tables[0]` is signalled and Cr
/// is set equal to Cb by [`build_signalled_chroma_qp_table`].
#[derive(Clone, Debug, Default)]
pub struct SignalledChromaQpTableParams {
    /// `same_qp_table_for_chroma` from the SPS body. When `true`, only
    /// `tables[0]` is parsed and Cr aliases Cb.
    pub same_qp_table_for_chroma: bool,
    /// `global_offset_flag` from the SPS body. Selects `startQP = 16`
    /// when `true`, `startQP = -QpBdOffsetC` when `false`.
    pub global_offset_flag: bool,
    /// One or two pivot-point sets, `tables[i]` carrying the i-th
    /// chroma-QP mapping table's pivot points. Length is 1 when
    /// `same_qp_table_for_chroma == true`, 2 otherwise.
    pub tables: Vec<SignalledChromaQpTablePivots>,
}

/// One chroma component's pivot-point set, parsed verbatim from the SPS.
///
/// `delta_qp_in_val_minus1[j]` corresponds to the spec syntax element of
/// the same name; `delta_qp_out_val[j]` is signed (`se(v)`).
#[derive(Clone, Debug, Default)]
pub struct SignalledChromaQpTablePivots {
    /// `delta_qp_in_val_minus1[j]` for `j ∈ [0, num_points_in_qp_table_minus1]`.
    pub delta_qp_in_val_minus1: Vec<u32>,
    /// `delta_qp_out_val[j]` for `j ∈ [0, num_points_in_qp_table_minus1]`.
    pub delta_qp_out_val: Vec<i32>,
}

/// Build a [`ChromaQpTable`] from SPS-signalled pivot points per
/// ISO/IEC 23094-1:2020(E) §7.4.3.1 page 67–68 (eq. 74 + the surrounding
/// fill loops).
///
/// `bit_depth_chroma_minus8` selects `QpBdOffsetC = 6 *
/// bit_depth_chroma_minus8` (eq. 39 + the chroma table layout).
///
/// Algorithm transcribed verbatim from the spec text:
///
/// ```text
/// startQp = ( global_offset_flag == 1 ) ? 16 : −QpBdOffsetC
/// qpInVal[i][0]  = startQP + delta_qp_in_val_minus1[i][0]
/// qpOutVal[i][0] = startQP + delta_qp_in_val_minus1[i][0]
///                          + delta_qp_out_val[i][0]
/// for j = 1..=num_points_in_qp_table_minus1[i]:
///     qpInVal[i][j]  = qpInVal[i][j−1]  + delta_qp_in_val_minus1[i][j] + 1
///     qpOutVal[i][j] = qpOutVal[i][j−1] + ( delta_qp_in_val_minus1[i][j] + 1
///                                          − delta_qp_out_val[i][j] )      (74)
/// ChromaQpTable[i][qpInVal[i][0]] = qpOutVal[i][0]
/// for k = qpInVal[i][0] − 1 down to −QpBdOffsetC:
///     ChromaQpTable[i][k] = Clip3(−QpBdOffsetC, 57,
///                                  ChromaQpTable[i][k+1] − 1)
/// for j = 0..num_points_in_qp_table_minus1[i] − 1:
///     sh = ( delta_qp_in_val_minus1[i][j+1] + 1 ) >> 1
///     for k = qpInVal[i][j] + 1, m = 1; k <= qpInVal[i][j+1]; k+=1, m+=1:
///         ChromaQpTable[i][k] = ChromaQpTable[i][qpInVal[i][j]]
///             + ( delta_qp_out_val[i][j+1] * m + sh )
///             / ( delta_qp_in_val_minus1[i][j+1] + 1 )
/// for k = qpInVal[i][last] + 1..=57:
///     ChromaQpTable[i][k] = Clip3(−QpBdOffsetC, 57,
///                                  ChromaQpTable[i][k−1] + 1)
/// ```
///
/// When `same_qp_table_for_chroma == 1`, `ChromaQpTable[1][k]` is set
/// equal to `ChromaQpTable[0][k]` for `k ∈ [−QpBdOffsetC, 57]`.
///
/// # Spec-text note (eq. 74)
///
/// The literal eq. 74 line in the 2020-published PDF lacks the trailing
/// `)` after `delta_qp_out_val[i][j]`. The bracketing we follow here
/// closes the `(` opened just before `delta_qp_in_val_minus1[i][j] + 1`,
/// so the recurrence is
/// `qpOutVal[i][j] = qpOutVal[i][j−1] + (delta_qp_in_val_minus1[i][j] + 1 − delta_qp_out_val[i][j])`.
///
/// `qpOutVal[]` is **not** used to fill `ChromaQpTable[]` except at
/// `qpInVal[0]` (line 4011 of the spec text); the per-segment loop
/// (lines 4015–4019) uses `delta_qp_out_val[]` directly. So the only
/// observable consequence of this bracketing choice is the value of
/// `ChromaQpTable[qpInVal[0]] = qpOutVal[0]` (which is bracketed
/// unambiguously, line 4005). A docs collaborator should confirm
/// eq. 74's bracketing before any conformance fixture exercises a
/// `chroma_qp_table_present_flag == 1` stream whose validity check
/// hinges on `qpOutVal[]`.
pub fn build_signalled_chroma_qp_table(
    params: &SignalledChromaQpTableParams,
    bit_depth_chroma_minus8: u32,
) -> Result<ChromaQpTable> {
    if bit_depth_chroma_minus8 > 8 {
        return Err(Error::invalid(
            "evc dra: bit_depth_chroma_minus8 must be in [0, 8]",
        ));
    }
    let n_tables_expected = if params.same_qp_table_for_chroma {
        1
    } else {
        2
    };
    if params.tables.len() != n_tables_expected {
        return Err(Error::invalid(format!(
            "evc dra: SignalledChromaQpTableParams expected {} pivot \
             set(s) (same_qp_table_for_chroma = {}), got {}",
            n_tables_expected,
            params.same_qp_table_for_chroma,
            params.tables.len()
        )));
    }

    let qp_bd_offset_c = 6 * bit_depth_chroma_minus8 as i32;
    let start_qp = if params.global_offset_flag {
        16i32
    } else {
        -qp_bd_offset_c
    };

    let mut built: [Option<ChromaQpTableEntry>; 2] = [None, None];
    for (i, pivots) in params.tables.iter().enumerate() {
        let table_i = build_one_signalled_table(pivots, start_qp, qp_bd_offset_c)?;
        built[i] = Some(table_i);
    }

    let cb = built[0]
        .take()
        .ok_or_else(|| Error::invalid("evc dra: missing Cb pivot set"))?;
    let cr = if params.same_qp_table_for_chroma {
        // Spec page 68: "When same_qp_table_for_chroma is equal to 1,
        // ChromaQpTable[1][k] is set equal to ChromaQpTable[0][k] for
        // k = −QpBdOffsetC..57."
        cb.clone()
    } else {
        built[1]
            .take()
            .ok_or_else(|| Error::invalid("evc dra: missing Cr pivot set"))?
    };

    Ok(ChromaQpTable { cb, cr })
}

/// Build one chroma component's `ChromaQpTable[i][k]` for
/// `k ∈ [−QpBdOffsetC, 57]` from the spec's pivot-point construction.
fn build_one_signalled_table(
    pivots: &SignalledChromaQpTablePivots,
    start_qp: i32,
    qp_bd_offset_c: i32,
) -> Result<ChromaQpTableEntry> {
    let n_points = pivots.delta_qp_in_val_minus1.len();
    if n_points == 0 {
        return Err(Error::invalid(
            "evc dra: signalled chroma QP table needs at least one pivot point",
        ));
    }
    if pivots.delta_qp_out_val.len() != n_points {
        return Err(Error::invalid(format!(
            "evc dra: delta_qp_in_val_minus1 / delta_qp_out_val length \
             mismatch ({} vs {})",
            n_points,
            pivots.delta_qp_out_val.len()
        )));
    }
    let num_points_minus1 = (n_points - 1) as i32;

    // qpInVal[] / qpOutVal[] derivation (eq. 74 + line 4004 / 4005).
    let mut qp_in_val = Vec::with_capacity(n_points);
    let mut qp_out_val = Vec::with_capacity(n_points);
    // j == 0
    let delta_in_0 = pivots.delta_qp_in_val_minus1[0] as i32;
    let delta_out_0 = pivots.delta_qp_out_val[0];
    qp_in_val.push(start_qp + delta_in_0);
    qp_out_val.push(start_qp + delta_in_0 + delta_out_0);
    for j in 1..n_points {
        let delta_in_j = pivots.delta_qp_in_val_minus1[j] as i32;
        let delta_out_j = pivots.delta_qp_out_val[j];
        let qp_in_j = qp_in_val[j - 1] + delta_in_j + 1;
        // eq. 74 — see module note above re: bracketing.
        let qp_out_j = qp_out_val[j - 1] + (delta_in_j + 1 - delta_out_j);
        qp_in_val.push(qp_in_j);
        qp_out_val.push(qp_out_j);
    }

    // Spec page 68: "The values of qpInVal[i][j] and qpOutval[i][j] shall
    // be in the range of −QpBdOffsetC to 57, inclusive." Catch the
    // qpInVal[] over-run (qpOutVal[] is informational with the eq. 74
    // bracketing — see fn doc) so we surface bad streams instead of
    // panicking on the indexing below.
    for (j, &in_v) in qp_in_val.iter().enumerate() {
        if !(-qp_bd_offset_c..=57).contains(&in_v) {
            return Err(Error::invalid(format!(
                "evc dra: qpInVal[{j}] = {in_v} out of range [{}, 57]",
                -qp_bd_offset_c
            )));
        }
    }

    let len = (qp_bd_offset_c + 58) as usize;
    let mut table = vec![0i32; len];

    let pack = |k: i32| -> usize { (k + qp_bd_offset_c) as usize };
    let clip = |v: i32| -> i32 { v.clamp(-qp_bd_offset_c, 57) };

    // Anchor: ChromaQpTable[qpInVal[0]] = qpOutVal[0] (line 4011).
    let qp_in_0 = qp_in_val[0];
    table[pack(qp_in_0)] = clip(qp_out_val[0]);

    // Down-fill below the first pivot (line 4012 / 4013).
    let mut k = qp_in_0 - 1;
    while k >= -qp_bd_offset_c {
        table[pack(k)] = clip(table[pack(k + 1)] - 1);
        k -= 1;
    }

    // Per-segment linear interpolation between consecutive pivots
    // (lines 4015–4019).
    for j in 0..(num_points_minus1 as usize) {
        let d_next = pivots.delta_qp_in_val_minus1[j + 1] as i32 + 1;
        let sh = d_next >> 1;
        let out_next = pivots.delta_qp_out_val[j + 1];
        let anchor = table[pack(qp_in_val[j])];
        let mut m = 1;
        let mut k = qp_in_val[j] + 1;
        while k <= qp_in_val[j + 1] {
            // Spec uses integer division on a numerator that can be
            // negative when delta_qp_out_val[] is negative; preserve
            // that semantics by using i32 truncated division (Rust's
            // `/` matches the spec's truncate-toward-zero behaviour).
            let v = anchor + (out_next * m + sh) / d_next;
            table[pack(k)] = clip(v);
            k += 1;
            m += 1;
        }
    }

    // Up-fill above the last pivot (line 4022 / 4023).
    let qp_in_last = qp_in_val[n_points - 1];
    let mut k = qp_in_last + 1;
    while k <= 57 {
        table[pack(k)] = clip(table[pack(k - 1)] + 1);
        k += 1;
    }

    Ok(ChromaQpTableEntry {
        qp_bd_offset_c,
        table,
    })
}

/// Table 5 (ISO/IEC 23094-1:2020(E) page 67) — `QpC` as a function of
/// `qPi` when `sps_iqt_flag == 0`.
fn table5_qp_c(qpi: i32) -> i32 {
    // Tabulated entries for qPi ∈ [30, 57]. qPi < 30 ⇒ QpC = qPi.
    // qPi >= 58 is out-of-range per §8.9.8 (Clip3(−QpBdOffsetC, 57, ...))
    // but we tolerate it defensively by returning the qPi == 57 value.
    if qpi < 30 {
        return qpi;
    }
    const TBL: [i32; 28] = [
        29, 29, 29, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, // 30..=43
        36, 36, 37, 37, 37, 38, 38, 39, 39, 40, 40, 40, 41, 41, // 44..=57
    ];
    let idx = (qpi.min(57) - 30) as usize;
    TBL[idx]
}

/// Table 6 (ISO/IEC 23094-1:2020(E) page 67) — `QpC` as a function of
/// `qPi` when `sps_iqt_flag == 1`.
fn table6_qp_c(qpi: i32) -> i32 {
    // qPi < 30 ⇒ QpC = qPi. qPi > 43 ⇒ QpC = qPi − 3. Tabulated for
    // qPi ∈ [30, 43].
    if qpi < 30 {
        return qpi;
    }
    if qpi > 43 {
        return qpi - 3;
    }
    const TBL: [i32; 14] = [29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 40];
    let idx = (qpi - 30) as usize;
    TBL[idx]
}

/// §8.9.5 helper specialised for the `ScaleQP` array — find the largest
/// `IndexScaleQP ∈ [0, len − 1]` such that `ScaleQP[IndexScaleQP] <=
/// scale_dra_norm < ScaleQP[IndexScaleQP + 1]`.
///
/// `ScaleQP` is monotonically non-decreasing (eq. 1420), so the spec's
/// `inputSample < rangesArray[rangeIdx + 1]` early-out lookup is well
/// defined. The §8.9.8 invocation passes `size = 54` and the table has
/// 55 entries (one extra for the upper sentinel) so the +1 is safe.
fn scale_qp_range_idx(scale_dra_norm: i64) -> usize {
    // §8.9.5 with numRanges = 54, rangesArray = ScaleQP.
    let num_ranges: usize = 54;
    for range_idx in 0..num_ranges {
        if scale_dra_norm < SCALE_QP[range_idx + 1] {
            return range_idx;
        }
    }
    num_ranges - 1
}

/// §8.9.8 (joined path) — derive `chromaScale` for a single `lumaScale`
/// + `cIdx` input. Pure function: every input is on the call line and
///   the only state read is the eq. 1420 / 1421 `SCALE_QP` / `QP_SCALE`
///   constants + the caller-supplied `ChromaQpTable`.
///
/// Transcribes eq. 1395 → 1419 verbatim. Used by
/// [`derive_dra_chroma_state_joined`] to fill `chroma_scales[i]` for
/// every luma range `i`.
///
/// `dra_table_idx` must be in `[0, 57]` (joined path); the unjoined
/// shortcut at `dra_table_idx == 58` is handled by
/// [`derive_dra_chroma_state`] (eq. 1394).
pub fn chroma_scale_joined(
    luma_scale: i64,
    dra_cb_scale_value: u16,
    dra_cr_scale_value: u16,
    cidx: ChromaIdx,
    dra_table_idx: u8,
    chroma_qp_table: &ChromaQpTable,
) -> i64 {
    // Eq. 1395: scaleDra = lumaScale * ((cIdx == 0) ? dra_cb_scale_value
    //                                                : dra_cr_scale_value).
    let component_scale = match cidx {
        ChromaIdx::Cb => dra_cb_scale_value as i64,
        ChromaIdx::Cr => dra_cr_scale_value as i64,
    };
    let scale_dra = luma_scale * component_scale;
    // Eq. 1396: scaleDraNorm = (scaleDra + (1 << 8)) >> 9.
    let scale_dra_norm = (scale_dra + (1i64 << 8)) >> 9;

    // IndexScaleQP via §8.9.5 over ScaleQP.
    let index_scale_qp = scale_qp_range_idx(scale_dra_norm);

    // Eq. 1397: qpDraInt = 2 * IndexScaleQP − 60.
    let mut qp_dra_int: i64 = 2 * (index_scale_qp as i64) - 60;

    // Eq. 1398-1399.
    let table_num = scale_dra_norm - SCALE_QP[index_scale_qp];
    let table_delta = SCALE_QP[index_scale_qp + 1] - SCALE_QP[index_scale_qp];

    // Spec page 308: "If tableNum is equal to 0, the variable qpDraFrac
    // is set equal to 0, and the variable qpDraInt is decreased by 1,
    // otherwise the variables qpDraInt, qpDraFrac and draChromaQpShift
    // are derived as follows: …"
    //
    // Both branches end up computing draChromaQpShift (eq. 1409 in the
    // "otherwise" branch). The tableNum == 0 branch leaves
    // draChromaQpShift = ChromaQpTable[cIdx][dra_table_idx] − qp0 −
    // qpDraIntAdj − qpDraInt with qpDraIntAdj = 0 and qpDraFracAdj = 0
    // (since qpDraFrac is set to 0 and qp1 − qp0 multiplied by 0 is 0);
    // the qpDraInt -= 1 line then matches the qpDraInt += (qpDraFrac
    // >> 9) of the otherwise branch under qpDraFrac == 0 (which gives
    // qpDraFrac = 1 << 9 after eq. 1402, then qpDraInt += 1; net zero
    // — except for the explicit -= 1 in the tableNum == 0 branch).
    let (qp_dra_frac_adj, dra_chroma_qp_shift) = if table_num == 0 {
        // qpDraFrac = 0; qpDraInt -= 1.
        qp_dra_int -= 1;
        let qp_dra_frac_adj: i64 = 0;
        let idx0 = (dra_table_idx as i64 - qp_dra_int)
            .clamp(-chroma_qp_table.cb.qp_bd_offset_c as i64, 57);
        let qp0 = chroma_qp_table.lookup(cidx, idx0 as i32) as i64;
        // qpDraIntAdj = 0 (qpDraFrac == 0 ⇒ (qp1 − qp0) * 0 >> 9 = 0).
        let dra_chroma_qp_shift =
            chroma_qp_table.lookup(cidx, dra_table_idx as i32) as i64 - qp0 - qp_dra_int;
        (qp_dra_frac_adj, dra_chroma_qp_shift)
    } else {
        // Eq. 1400: qpDraFrac = (tableNum << 10) / tableDelta.
        let mut qp_dra_frac = (table_num << 10) / table_delta;
        // Eq. 1401: qpDraInt += (qpDraFrac >> 9).
        qp_dra_int += qp_dra_frac >> 9;
        // Eq. 1402: qpDraFrac = (1 << 9) − qpDraFrac % (1 << 9).
        qp_dra_frac = (1i64 << 9) - qp_dra_frac.rem_euclid(1i64 << 9);

        // Eq. 1403 / 1404: idx0, idx1 with Clip3.
        let lo = -chroma_qp_table.cb.qp_bd_offset_c as i64;
        let hi = 57i64;
        let idx0 = (dra_table_idx as i64 - qp_dra_int).clamp(lo, hi);
        let idx1 = (dra_table_idx as i64 - qp_dra_int + 1).clamp(lo, hi);

        // Eq. 1405 / 1406: qp0, qp1.
        let qp0 = chroma_qp_table.lookup(cidx, idx0 as i32) as i64;
        let qp1 = chroma_qp_table.lookup(cidx, idx1 as i32) as i64;

        // Eq. 1407: qpDraIntAdj = ((qp1 − qp0) * qpDraFrac) >> 9.
        let qp_dra_int_adj = ((qp1 - qp0) * qp_dra_frac) >> 9;

        // Eq. 1408: qpDraFracAdj = qpDraFrac − (((qp1 − qp0) * qpDraFrac) % (1 << 9)).
        let mut qp_dra_frac_adj = qp_dra_frac - (((qp1 - qp0) * qp_dra_frac).rem_euclid(1i64 << 9));

        // Eq. 1409: draChromaQpShift = ChromaQpTable[cIdx][dra_table_idx]
        //                              − qp0 − qpDraIntAdj − qpDraInt.
        let mut dra_chroma_qp_shift = chroma_qp_table.lookup(cidx, dra_table_idx as i32) as i64
            - qp0
            - qp_dra_int_adj
            - qp_dra_int;

        // Eq. 1410-1411: qpDraFracAdj < 0 fix-up.
        if qp_dra_frac_adj < 0 {
            dra_chroma_qp_shift -= 1;
            qp_dra_frac_adj += 1i64 << 9;
        }
        (qp_dra_frac_adj, dra_chroma_qp_shift)
    };

    // Eq. 1412-1414: idx0, idx1, idx2 with Clip3(0, 24, ·).
    let idx0 = (dra_chroma_qp_shift + 12).clamp(0, 24) as usize;
    let idx1 = (dra_chroma_qp_shift + 12 - 1).clamp(0, 24) as usize;
    let idx2 = (dra_chroma_qp_shift + 12 + 1).clamp(0, 24) as usize;

    // Eq. 1415: draChromaScaleShift = QpScale[idx0].
    let mut dra_chroma_scale_shift = QP_SCALE[idx0];
    // Eq. 1416 / 1417.
    let dra_chroma_scale_shift_frac = if dra_chroma_qp_shift < 0 {
        QP_SCALE[idx0] - QP_SCALE[idx1]
    } else {
        QP_SCALE[idx2] - QP_SCALE[idx0]
    };
    // Eq. 1418: draChromaScaleShift = draChromaScaleShift +
    //   (draChromaScaleShiftFrac * qpDraFracAdj + (1 << 8)) >> 9.
    // Spec text uses a single >> 9 of the whole sum-with-add — the
    // outer addition rebases the shift on draChromaScaleShift. Reading
    // the formula as written places the >> 9 outside the parenthesised
    // sum, so we shift only the (frac * adj + 256) term.
    dra_chroma_scale_shift += (dra_chroma_scale_shift_frac * qp_dra_frac_adj + (1i64 << 8)) >> 9;

    // Eq. 1419: chromaScale = ((scaleDra * draChromaScaleShift) + (1 << 17)) >> 18.
    ((scale_dra * dra_chroma_scale_shift) + (1i64 << 17)) >> 18
}

/// Derive the §8.9.7 chroma DRA state for chroma component `cidx` under
/// the **joined-scale path** (`DraJoinedScaleFlag == 1`,
/// `dra_table_idx ∈ [0, 57]`).
///
/// The structural difference from [`derive_dra_chroma_state`] (which
/// handles the simpler `DraJoinedScaleFlag == 0` / `dra_table_idx ==
/// 58` path) is in eq. 1394: under the joined path,
/// `chromaScales[cIdx][i]` is derived **per luma range** by invoking
/// §8.9.8 with `lumaScale = lumaScales[i] = dra_scale_value[i]` (the
/// per-range luma scale parsed in §7.3.6). Eq. 1386-1393 are
/// identical to the unjoined path, just with the per-`i`
/// `chromaScales[i]` instead of the constant.
///
/// `chroma_qp_table` is the `ChromaQpTable` for the active SPS — built
/// from [`default_chroma_qp_table`] when `chroma_qp_table_present_flag
/// == 0`, or from the SPS-signalled chroma QP mapping (eq. 74) when
/// the flag is set. (Round 193 ships the consumer side; the SPS-side
/// parser still discards the signalled deltas — that's the round-194
/// follow-up.)
///
/// Returns [`Error::Unsupported`] when `derived.joined_scale_flag` is
/// `false` (the caller should use [`derive_dra_chroma_state`] instead).
/// Returns [`Error::Invalid`] when `dra_cb_scale_value == 0` (Cb path)
/// or `dra_cr_scale_value == 0` (Cr path) or any luma
/// `dra_scale_value[i] == 0` — all forbidden by §7.4.7.
pub fn derive_dra_chroma_state_joined(
    syntax: &DraSyntax,
    derived: &DraDerived,
    cidx: ChromaIdx,
    bit_depth_y: u32,
    chroma_qp_table: &ChromaQpTable,
) -> Result<DraChromaDerived> {
    if !derived.joined_scale_flag {
        return Err(Error::unsupported(
            "evc dra: derive_dra_chroma_state_joined invoked with \
             DraJoinedScaleFlag = 0 — call `derive_dra_chroma_state` for \
             the unjoined `dra_table_idx == 58` path",
        ));
    }
    if !(8..=16).contains(&bit_depth_y) {
        return Err(Error::invalid(
            "evc dra: bit_depth_y must be in [8, 16] (per SPS §7.4.3.1)",
        ));
    }
    if derived.num_ranges == 0 {
        return Ok(DraChromaDerived::empty(cidx));
    }
    // Spec §7.4.7 forbids zero scales; defend against the divide-by-zero
    // in eq. 1386 (and the trivial multiply-by-zero in eq. 1395 that
    // would collapse the entire joined chain).
    let component_scale = match cidx {
        ChromaIdx::Cb => syntax.dra_cb_scale_value,
        ChromaIdx::Cr => syntax.dra_cr_scale_value,
    };
    if component_scale == 0 {
        return Err(Error::invalid(
            "evc dra: chroma scale value == 0 (forbidden by §7.4.7); \
             cannot derive joined chromaScales",
        ));
    }
    // dra_table_idx must be in the joined range [0, 57]; the unjoined
    // shortcut at 58 is rejected at the top of the function via the
    // joined_scale_flag check.
    if syntax.dra_table_idx > 57 {
        return Err(Error::invalid(
            "evc dra: dra_table_idx must be in [0, 57] on the joined path",
        ));
    }
    for i in 0..derived.num_ranges {
        if syntax.dra_scale_value[i] == 0 {
            return Err(Error::invalid(
                "evc dra: dra_scale_value[i] == 0 (forbidden by §7.4.7); \
                 joined chromaScale[i] would collapse",
            ));
        }
    }

    let mut c = DraChromaDerived::empty(cidx);
    c.num_ranges_l = derived.num_ranges;
    let num_ranges_l = c.num_ranges_l;
    let num_ranges_c = num_ranges_l + 1;

    // Eq. 1394 (joined path): chromaScales[cIdx][i] = §8.9.8(lumaScales[i],
    //                                                         cIdx).
    // `lumaScales[i]` is the per-range luma scale, i.e. dra_scale_value[i]
    // (§7.4.7 line below eq. 117: "lumaScales[apsId][i] = dra_scale_value[i]").
    for i in 0..num_ranges_l {
        let luma_scale_i = syntax.dra_scale_value[i] as i64;
        let chroma_scale_i = chroma_scale_joined(
            luma_scale_i,
            syntax.dra_cb_scale_value,
            syntax.dra_cr_scale_value,
            cidx,
            syntax.dra_table_idx,
            chroma_qp_table,
        );
        if chroma_scale_i <= 0 {
            return Err(Error::invalid(
                "evc dra: joined chromaScale[i] <= 0 — divide-by-zero in \
                 eq. 1386 would follow",
            ));
        }
        c.chroma_scales[i] = chroma_scale_i;
        // Eq. 1386 reuses chromaScales[i] for invChromaScales[i].
        c.inv_chroma_scales[i] = ((1i64 << 18) + (chroma_scale_i >> 1)) / chroma_scale_i;
    }

    // Line above eq. 1387: i = 0 layout.
    c.out_scales_c[0] = 0;
    c.out_offsets_c[0] = c.inv_chroma_scales[0];
    c.out_ranges_c[0] = derived.out_ranges_l[0];

    // Eq. 1387: OutRangesC[i] = (OutRangesL[i] + OutRangesL[i − 1]) >> 1
    // for i in [1, num_ranges_c − 1] = [1, num_ranges_l].
    for i in 1..num_ranges_c {
        c.out_ranges_c[i] = (derived.out_ranges_l[i] + derived.out_ranges_l[i - 1]) >> 1;
    }

    // Eq. 1388-1391: i in [1, num_ranges_l − 1].
    for i in 1..num_ranges_l {
        let delta_range = c.out_ranges_c[i + 1] - c.out_ranges_c[i];
        c.out_offsets_c[i] = c.inv_chroma_scales[i - 1];
        let delta_scale = c.inv_chroma_scales[i] - c.inv_chroma_scales[i - 1];
        c.out_scales_c[i] = if delta_range == 0 {
            0
        } else {
            ((delta_scale << 10) + (delta_range >> 1)) / delta_range
        };
    }

    // Eq. 1392-1393: i = num_ranges_l.
    c.out_offsets_c[num_ranges_l] = c.inv_chroma_scales[num_ranges_l - 1];
    c.out_scales_c[num_ranges_l] = 0;

    // §8.9.5 top sentinel.
    c.out_ranges_c[num_ranges_c] = 1i64 << bit_depth_y;

    Ok(c)
}

/// Resolve §8.9.7 / §8.9.8 chroma derivation directly from a parsed
/// [`crate::sps::Sps`] — the SPS → `DraChromaDerived` adapter that
/// dispatches between the joined (`DraJoinedScaleFlag = 1`) and
/// unjoined (`DraJoinedScaleFlag = 0`, `dra_table_idx == 58`) paths
/// without making the caller re-implement the branch.
///
/// Dispatch (per ISO/IEC 23094-1:2020(E) §8.9.7 + §8.9.8):
///
/// * `derived.joined_scale_flag == false` → invokes
///   [`derive_dra_chroma_state`] with `bit_depth_y = sps.bit_depth_y()`.
///   The `ChromaQpTable` is not used on this path (the unjoined
///   chroma scale is the raw `dra_cb_scale_value` / `dra_cr_scale_value`
///   per eq. 1394, independent of `qPi`); the adapter still resolves
///   the SPS-active table internally for symmetry but discards it.
/// * `derived.joined_scale_flag == true` → invokes
///   [`derive_dra_chroma_state_joined`] with `bit_depth_y =
///   sps.bit_depth_y()` and the SPS-active `ChromaQpTable` from
///   [`chroma_qp_table_for_sps`] (signalled if present; Table 5 / 6
///   on 4:2:0; identity otherwise).
///
/// This closes the SPS → §8.9.6 chroma-scale chain at one call site
/// for both paths, parallelling round-201's
/// [`chroma_qp_table_for_sps`] for the table half. Existing direct-
/// invocation callers ([`derive_dra_chroma_state`] /
/// [`derive_dra_chroma_state_joined`] with a hand-built table) keep
/// working unchanged — this is a thin opt-in adapter, not a behaviour
/// change.
///
/// # Errors
///
/// Bubbles up the underlying derive's error verbatim:
///
/// * `chroma_qp_table_for_sps` failures (e.g. out-of-range
///   `bit_depth_chroma_minus8` from a malformed SPS).
/// * `derive_dra_chroma_state` failures (zero chroma scale, `bit_depth_y`
///   out of range).
/// * `derive_dra_chroma_state_joined` failures (zero per-range luma
///   scale, `dra_table_idx > 57`, zero joined chromaScale).
///
/// # Use
///
/// ```text
/// let (syntax, derived) = parse_dra_syntax(payload, sps.bit_depth_y())?;
/// let cb = derive_dra_chroma_state_for_sps(&syntax, &derived,
///                                          ChromaIdx::Cb, &sps)?;
/// let scale = chroma_scale_for_luma_sample(luma, &cb);
/// ```
///
/// On a strictly monochrome stream (`chroma_format_idc == 0`) there is
/// no chroma plane to run §8.9.6 on; this adapter still synthesises a
/// usable [`DraChromaDerived`] so introspection / unit-test code can
/// exercise the joined chain (identity `ChromaQpTable`). Production
/// chroma processing should short-circuit at the picture level.
pub fn derive_dra_chroma_state_for_sps(
    syntax: &DraSyntax,
    derived: &DraDerived,
    cidx: ChromaIdx,
    sps: &crate::sps::Sps,
) -> Result<DraChromaDerived> {
    let bit_depth_y = sps.bit_depth_y();
    if derived.joined_scale_flag {
        let table = chroma_qp_table_for_sps(sps)?;
        derive_dra_chroma_state_joined(syntax, derived, cidx, bit_depth_y, &table)
    } else {
        derive_dra_chroma_state(syntax, derived, cidx, bit_depth_y)
    }
}

/// Materialise `chroma_derived.out_ranges_c[0..=num_ranges_c]` as an
/// `i32` vector suitable for [`find_range_idx`]. Includes the top
/// sentinel `1 << bit_depth_y` placed at index `num_ranges_c` by
/// [`derive_dra_chroma_state`]. Mirrors [`out_ranges_l_as_i32`] for the
/// chroma side; the +1 over the chroma-derived count reflects §8.9.5's
/// one-past-end indexing.
fn out_ranges_c_as_i32(chroma_derived: &DraChromaDerived) -> Vec<i32> {
    // numRangesC = num_ranges_l + 1; we need num_ranges_l + 2 boundaries
    // (one per range + one top sentinel) so §8.9.5 indexes safely up to
    // rangesArray[numRangesC].
    let n = chroma_derived.num_ranges_l + 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = chroma_derived.out_ranges_c[i];
        let clamped = v.clamp(i32::MIN as i64, i32::MAX as i64);
        out.push(clamped as i32);
    }
    out
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

    // ============================================================
    // Round 187 — §8.9.6 / §8.9.7 / §8.9.8 chroma derivation tests.
    // ============================================================

    /// Three-range syntax with `dra_table_idx = 58` (DraJoinedScaleFlag = 0
    /// path). Cb scale = 256 (Q9 ≈ 0.5), Cr scale = 1024 (Q9 ≈ 2.0).
    fn three_range_unjoined_chroma_syntax(bit_depth: u32) -> DraSyntax {
        let _ = bit_depth;
        let mut delta = [0u16; DRA_MAX_RANGES_V2];
        // Equal-ranges off; per-range distinct deltas.
        delta[0] = 32;
        delta[1] = 64;
        delta[2] = 96;
        let mut scales = [0u16; DRA_MAX_RANGES_V2];
        scales[0] = 512; // Q9 = 1.0
        scales[1] = 512;
        scales[2] = 512;
        DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 2, // 3 luma ranges
            dra_equal_ranges_flag: false,
            dra_global_offset: 16,
            dra_delta_range: delta,
            dra_scale_value: scales,
            dra_cb_scale_value: 256,
            dra_cr_scale_value: 1024,
            dra_table_idx: 58,
        }
    }

    #[test]
    fn round187_chroma_derive_rejects_joined_path() {
        // dra_table_idx != 58 ⇒ DraJoinedScaleFlag = 1 ⇒ §8.9.8
        // eq. 1395-1419 chain (needs ChromaQpTable). Round 187 surfaces
        // this as Err so the caller doesn't silently use a wrong scale.
        let mut syn = three_range_unjoined_chroma_syntax(10);
        syn.dra_table_idx = 0; // joined path
        let der = derive_dra_state(&syn, 10).unwrap();
        let res = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10);
        assert!(res.is_err(), "joined path must error out");
    }

    #[test]
    fn round187_chroma_derive_rejects_zero_cb_scale() {
        // dra_cb_scale_value = 0 is forbidden by §7.4.7; helper
        // defends against the divide-by-zero in eq. 1386.
        let mut syn = three_range_unjoined_chroma_syntax(10);
        syn.dra_cb_scale_value = 0;
        let der = derive_dra_state(&syn, 10).unwrap();
        assert!(derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).is_err());
        // Cr path still derives fine.
        assert!(derive_dra_chroma_state(&syn, &der, ChromaIdx::Cr, 10).is_ok());
    }

    #[test]
    fn round187_chroma_derive_rejects_zero_cr_scale() {
        let mut syn = three_range_unjoined_chroma_syntax(10);
        syn.dra_cr_scale_value = 0;
        let der = derive_dra_state(&syn, 10).unwrap();
        assert!(derive_dra_chroma_state(&syn, &der, ChromaIdx::Cr, 10).is_err());
        assert!(derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).is_ok());
    }

    #[test]
    fn round187_chroma_derive_noop_on_empty_state() {
        // num_ranges = 0 ⇒ chroma derivation is a no-op (empty state).
        let syn = three_range_unjoined_chroma_syntax(10);
        let mut der = DraDerived::empty();
        der.joined_scale_flag = false;
        // num_ranges intentionally 0.
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        assert_eq!(c.num_ranges_l, 0);
        assert_eq!(c.cidx, ChromaIdx::Cb);
        // chroma_scale_for_luma_sample on empty state returns 0.
        assert_eq!(chroma_scale_for_luma_sample(100, &c), 0);
    }

    #[test]
    fn round187_chroma_derive_eq1386_inv_chroma_scales_identity() {
        // Cb scale = 512 (Q9 1.0) ⇒ invChromaScales = ((1<<18) + 256) / 512 = 512.
        let mut syn = three_range_unjoined_chroma_syntax(10);
        syn.dra_cb_scale_value = 512;
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        // ((1 << 18) + (512 >> 1)) / 512 = (262144 + 256) / 512 = 262400 / 512 = 512.
        let expected_inv = ((1i64 << 18) + (512 >> 1)) / 512;
        assert_eq!(expected_inv, 512);
        for i in 0..c.num_ranges_l {
            assert_eq!(c.chroma_scales[i], 512, "chroma_scales[{i}]");
            assert_eq!(c.inv_chroma_scales[i], 512, "inv_chroma_scales[{i}]");
        }
    }

    #[test]
    fn round187_chroma_derive_eq1386_doubled_chroma_scale() {
        // Cb scale = 1024 (Q9 = 2.0) ⇒ invChromaScales = ((1<<18) + 512) / 1024 = 256.
        // The Cb path uses dra_cb_scale_value (256), so override for this test.
        let mut syn = three_range_unjoined_chroma_syntax(10);
        syn.dra_cb_scale_value = 1024;
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        // ((1<<18) + 512) / 1024 = 262656 / 1024 = 256.
        for i in 0..c.num_ranges_l {
            assert_eq!(c.chroma_scales[i], 1024);
            assert_eq!(c.inv_chroma_scales[i], 256);
        }
    }

    #[test]
    fn round187_chroma_derive_eq1387_out_ranges_c_midpoints() {
        // OutRangesC[0] = OutRangesL[0] = 0.
        // OutRangesC[i] = (OutRangesL[i] + OutRangesL[i-1]) >> 1 for i in [1, numRangesL].
        // With identity-Q9 luma scale and 3 ranges, OutRangesL is monotonic so
        // OutRangesC[i] sits halfway between OutRangesL[i-1] and OutRangesL[i].
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        // OutRangesC[0] == OutRangesL[0].
        assert_eq!(c.out_ranges_c[0], der.out_ranges_l[0]);
        // OutRangesC[i] = midpoint.
        for i in 1..=der.num_ranges {
            let mid = (der.out_ranges_l[i] + der.out_ranges_l[i - 1]) >> 1;
            assert_eq!(c.out_ranges_c[i], mid, "OutRangesC[{i}]");
        }
    }

    #[test]
    fn round187_chroma_derive_eq1389_eq1391_out_scales_offsets_unjoined() {
        // Under DraJoinedScaleFlag = 0, invChromaScales is constant across i,
        // so deltaScale = invChromaScales[i] − invChromaScales[i−1] = 0 for
        // every i ∈ [1, num_ranges_minus1]. Therefore OutScalesC[i] = 0 for
        // every such i. OutOffsetsC[i] = invChromaScales[i−1] = the constant.
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        // num_ranges_minus1 = 2 ⇒ loop runs i ∈ [1, 2].
        for i in 1..der.num_ranges {
            assert_eq!(
                c.out_scales_c[i], 0,
                "OutScalesC[{i}] = 0 (deltaScale is 0 under joined=0)"
            );
            assert_eq!(
                c.out_offsets_c[i], c.inv_chroma_scales[0],
                "OutOffsetsC[{i}] = invChromaScales[i−1]"
            );
        }
    }

    #[test]
    fn round187_chroma_derive_eq1392_eq1393_top_sentinel() {
        // i = numRangesL: OutOffsetsC[numRangesL] = invChromaScales[numRangesL − 1]
        //                 OutScalesC[numRangesL] = 0.
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cr, 10).unwrap();
        let top = c.num_ranges_l;
        assert_eq!(c.out_offsets_c[top], c.inv_chroma_scales[top - 1]);
        assert_eq!(c.out_scales_c[top], 0);
    }

    #[test]
    fn round187_chroma_derive_index_zero_layout_matches_spec() {
        // Line above eq. 1387: "Variables OutScalesC[cIdx][0],
        // OutOffsetsC[cIdx][0] and OutRangesC[0] are set equal to 0,
        // invChromaScales[cIdx][0] and OutRangesL[0], respectively."
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        assert_eq!(c.out_scales_c[0], 0);
        assert_eq!(c.out_offsets_c[0], c.inv_chroma_scales[0]);
        assert_eq!(c.out_ranges_c[0], der.out_ranges_l[0]);
    }

    #[test]
    fn round187_chroma_derive_cb_cr_distinct() {
        // Cb and Cr paths derive independently from their own scale values.
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let cb = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        let cr = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cr, 10).unwrap();
        // Cb uses dra_cb_scale_value = 256; Cr uses dra_cr_scale_value = 1024.
        assert_eq!(cb.chroma_scales[0], 256);
        assert_eq!(cr.chroma_scales[0], 1024);
        // Different inverse: ((1<<18) + 128) / 256 = 1024 vs 256.
        assert_eq!(cb.inv_chroma_scales[0], ((1i64 << 18) + 128) / 256);
        assert_eq!(cr.inv_chroma_scales[0], ((1i64 << 18) + 512) / 1024);
        // OutRangesC is component-agnostic — it's a function of OutRangesL.
        for i in 0..=cb.num_ranges_l {
            assert_eq!(cb.out_ranges_c[i], cr.out_ranges_c[i]);
        }
    }

    #[test]
    fn round187_chroma_scale_for_sample_eq1384_eq1385_unjoined() {
        // Under joined = 0, eq. 1385 with constant invChromaScales:
        //   OutOffsetsC[0] = invChromaScales[0] = const,
        //   OutScalesC[0] = 0 ⇒ eq. 1385 collapses to chromaScale =
        //     OutOffsetsC[0] + ((0 * incValue + 512) >> 10) = invChromaScales[0].
        // i.e. the §8.9.6 chromaScale is constant across luma samples
        // (matches the §8.9.8 joined-flag-0 reading where the scale is a
        // single value per chroma component, not per luma range).
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        let expected = c.inv_chroma_scales[0];
        // Pick a luma sample inside the first range. OutRangesC[0] = 0,
        // OutRangesC[1] is some positive midpoint, so 10 will hit range 0.
        let s0 = chroma_scale_for_luma_sample(10, &c);
        assert_eq!(s0, expected);
        // And a higher sample value: still picks a range with same constant
        // OutOffsetsC + zero OutScalesC contribution.
        // But the higher ranges use OutOffsetsC[i] = invChromaScales[i−1] =
        // the same constant. So every range yields the same scale.
        let s_mid = chroma_scale_for_luma_sample(c.out_ranges_c[1] + 1, &c);
        let s_hi = chroma_scale_for_luma_sample(c.out_ranges_c[c.num_ranges_l], &c);
        assert_eq!(s_mid, expected);
        assert_eq!(s_hi, expected);
    }

    #[test]
    fn round187_chroma_scale_constant_property_unjoined() {
        // The whole point of DraJoinedScaleFlag = 0 is that §8.9.6's
        // chromaScale collapses to a single value per chroma component —
        // verify across the full 10-bit sample range that
        // chroma_scale_for_luma_sample is constant.
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cr, 10).unwrap();
        let baseline = chroma_scale_for_luma_sample(0, &c);
        for v in (0..1024).step_by(13) {
            let s = chroma_scale_for_luma_sample(v as i64, &c);
            assert_eq!(s, baseline, "chromaScale should be constant; v = {v}");
        }
    }

    #[test]
    fn round187_chroma_scale_for_sample_handles_out_of_range_high() {
        // §8.9.5 ends with Min(rangeIdx, numRanges − 1) — a luma sample
        // larger than OutRangesC's top boundary still picks the last range
        // (no panic, no out-of-bounds). With joined = 0 the constant chroma
        // scale is unchanged.
        let syn = three_range_unjoined_chroma_syntax(10);
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        let baseline = chroma_scale_for_luma_sample(0, &c);
        let s = chroma_scale_for_luma_sample(i32::MAX as i64 / 2, &c);
        assert_eq!(s, baseline);
    }

    #[test]
    fn round187_chroma_derive_single_range_identity() {
        // num_ranges_minus1 = 0 ⇒ num_ranges_l = 1, num_ranges_c = 2.
        // Eq. 1388-1391 loop body is empty (i ∈ [1, 0] = ∅).
        // Only eq. 1392-1393 (top sentinel) and the i = 0 line apply.
        let mut delta = [0u16; DRA_MAX_RANGES_V2];
        delta[0] = 64;
        let mut scales = [0u16; DRA_MAX_RANGES_V2];
        scales[0] = 512;
        let syn = DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 0,
            dra_equal_ranges_flag: true,
            dra_global_offset: 16,
            dra_delta_range: delta,
            dra_scale_value: scales,
            dra_cb_scale_value: 512,
            dra_cr_scale_value: 512,
            dra_table_idx: 58,
        };
        let der = derive_dra_state(&syn, 10).unwrap();
        let c = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10).unwrap();
        assert_eq!(c.num_ranges_l, 1);
        // OutRangesC has 2 entries: [OutRangesL[0], (OutRangesL[1] + OutRangesL[0]) >> 1].
        assert_eq!(c.out_ranges_c[0], der.out_ranges_l[0]);
        assert_eq!(
            c.out_ranges_c[1],
            (der.out_ranges_l[1] + der.out_ranges_l[0]) >> 1
        );
        // i = 0 layout.
        assert_eq!(c.out_scales_c[0], 0);
        assert_eq!(c.out_offsets_c[0], c.inv_chroma_scales[0]);
        // Top sentinel at i = num_ranges_l = 1.
        assert_eq!(c.out_offsets_c[1], c.inv_chroma_scales[0]);
        assert_eq!(c.out_scales_c[1], 0);
    }

    #[test]
    fn round187_chroma_idx_as_u32() {
        assert_eq!(ChromaIdx::Cb.as_u32(), 0);
        assert_eq!(ChromaIdx::Cr.as_u32(), 1);
    }

    // ============================================================
    // Round 193 — §8.9.8 joined chroma scale (DraJoinedScaleFlag = 1)
    //              + default Table 5 / Table 6 ChromaQpTable.
    // ============================================================

    #[test]
    fn round193_scale_qp_table_eq1420_first_and_last_entries() {
        // Spot-check the spec literal (page 308 eq. 1420). The table is
        // monotonically non-decreasing — required for the §8.9.5 lookup.
        assert_eq!(SCALE_QP[0], 0);
        assert_eq!(SCALE_QP[1], 1);
        assert_eq!(SCALE_QP[6], 2);
        assert_eq!(SCALE_QP[7], 2);
        assert_eq!(SCALE_QP[10], 4);
        assert_eq!(SCALE_QP[21], 57);
        assert_eq!(SCALE_QP[54], 116772);
        // Monotonicity (strict above the leading zero).
        for i in 1..SCALE_QP.len() {
            assert!(
                SCALE_QP[i] >= SCALE_QP[i - 1],
                "SCALE_QP non-monotonic at i={i}"
            );
        }
    }

    #[test]
    fn round193_qp_scale_table_eq1421_first_and_last_entries() {
        // Spec literal page 308 eq. 1421.
        assert_eq!(QP_SCALE[0], 128);
        assert_eq!(QP_SCALE[6], 256);
        assert_eq!(QP_SCALE[12], 512);
        assert_eq!(QP_SCALE[18], 1024);
        assert_eq!(QP_SCALE[24], 2048);
        // Monotonically increasing.
        for i in 1..QP_SCALE.len() {
            assert!(
                QP_SCALE[i] > QP_SCALE[i - 1],
                "QP_SCALE non-monotonic at i={i}"
            );
        }
        assert_eq!(QP_SCALE.len(), 25);
    }

    #[test]
    fn round193_default_chroma_qp_table_iqt_off_table5_spot_checks() {
        // sps_iqt_flag == 0 ⇒ Table 5.
        // qPi < 30 ⇒ QpC = qPi.
        // qPi 30 → 29, qPi 33 → 30, qPi 35 → 32, qPi 43 → 36, qPi 50 → 38,
        // qPi 57 → 41.
        let t = default_chroma_qp_table(false, 0).unwrap();
        assert_eq!(t.cb.qp_bd_offset_c, 0);
        assert_eq!(t.lookup(ChromaIdx::Cb, 0), 0);
        assert_eq!(t.lookup(ChromaIdx::Cb, 10), 10);
        assert_eq!(t.lookup(ChromaIdx::Cb, 29), 29);
        assert_eq!(t.lookup(ChromaIdx::Cb, 30), 29);
        assert_eq!(t.lookup(ChromaIdx::Cb, 33), 30);
        assert_eq!(t.lookup(ChromaIdx::Cb, 35), 32);
        assert_eq!(t.lookup(ChromaIdx::Cb, 43), 36);
        assert_eq!(t.lookup(ChromaIdx::Cb, 50), 38);
        assert_eq!(t.lookup(ChromaIdx::Cb, 57), 41);
        // Cb and Cr identical under chroma_qp_table_present_flag == 0.
        for qpi in 0..=57 {
            assert_eq!(t.lookup(ChromaIdx::Cb, qpi), t.lookup(ChromaIdx::Cr, qpi));
        }
    }

    #[test]
    fn round193_default_chroma_qp_table_iqt_on_table6_spot_checks() {
        // sps_iqt_flag == 1 ⇒ Table 6.
        // qPi < 30 ⇒ QpC = qPi. qPi > 43 ⇒ QpC = qPi − 3.
        // qPi 30 → 29, qPi 31 → 30, qPi 40 → 37, qPi 43 → 40.
        // qPi 44 → 41, qPi 57 → 54.
        let t = default_chroma_qp_table(true, 0).unwrap();
        assert_eq!(t.lookup(ChromaIdx::Cb, 29), 29);
        assert_eq!(t.lookup(ChromaIdx::Cb, 30), 29);
        assert_eq!(t.lookup(ChromaIdx::Cb, 31), 30);
        // Tabulated for qPi 30..=43; qPi=40 → tbl[10] = 38.
        assert_eq!(t.lookup(ChromaIdx::Cb, 40), 38);
        assert_eq!(t.lookup(ChromaIdx::Cb, 43), 40);
        // qPi > 43: QpC = qPi − 3.
        assert_eq!(t.lookup(ChromaIdx::Cb, 44), 41);
        assert_eq!(t.lookup(ChromaIdx::Cb, 50), 47);
        assert_eq!(t.lookup(ChromaIdx::Cb, 57), 54);
    }

    #[test]
    fn round193_default_chroma_qp_table_bit_depth_10_negative_qpi() {
        // bit_depth_chroma_minus8 == 2 ⇒ QpBdOffsetC = 12.
        // qPi ∈ [−12, 57]. Negative qPi ⇒ Table 5 path qPi < 30 ⇒ QpC = qPi.
        let t = default_chroma_qp_table(false, 2).unwrap();
        assert_eq!(t.cb.qp_bd_offset_c, 12);
        assert_eq!(t.cb.table.len(), 12 + 58);
        for qpi in -12..30 {
            assert_eq!(t.lookup(ChromaIdx::Cb, qpi), qpi);
        }
        // Out-of-range clamps to qPi == −12 on the low side.
        assert_eq!(t.lookup(ChromaIdx::Cb, -100), -12);
        // …and to qPi == 57 on the high side.
        assert_eq!(t.lookup(ChromaIdx::Cb, 100), 41);
    }

    #[test]
    fn round193_default_chroma_qp_table_rejects_out_of_range_bit_depth() {
        assert!(default_chroma_qp_table(false, 9).is_err());
        assert!(default_chroma_qp_table(true, 100).is_err());
    }

    /// Three-range joined-path syntax: dra_table_idx = 30 (mid Table 5
    /// range), Cb = 256, Cr = 1024, per-range luma scales 256/512/1024.
    fn three_range_joined_chroma_syntax() -> DraSyntax {
        let mut delta = [0u16; DRA_MAX_RANGES_V2];
        delta[0] = 32;
        delta[1] = 64;
        delta[2] = 96;
        let mut scales = [0u16; DRA_MAX_RANGES_V2];
        scales[0] = 256;
        scales[1] = 512;
        scales[2] = 1024;
        DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 2,
            dra_equal_ranges_flag: false,
            dra_global_offset: 16,
            dra_delta_range: delta,
            dra_scale_value: scales,
            dra_cb_scale_value: 256,
            dra_cr_scale_value: 1024,
            dra_table_idx: 30,
        }
    }

    #[test]
    fn round193_chroma_scale_joined_returns_positive_for_table5() {
        // Reasonable joined inputs: luma scale 512 (Q9 = 1.0), Cb scale
        // 256 (Q9 ≈ 0.5), dra_table_idx 30. Output must be > 0 (the
        // §8.9.6 eq. 1386 invChromaScale divides by chromaScale, so a
        // zero return would blow up).
        let t = default_chroma_qp_table(false, 0).unwrap();
        let s = chroma_scale_joined(512, 256, 1024, ChromaIdx::Cb, 30, &t);
        assert!(s > 0, "joined chromaScale should be positive, got {s}");
        let s_cr = chroma_scale_joined(512, 256, 1024, ChromaIdx::Cr, 30, &t);
        assert!(
            s_cr > 0,
            "joined chromaScale (Cr) should be positive, got {s_cr}"
        );
    }

    #[test]
    fn round193_chroma_scale_joined_scales_with_luma_scale() {
        // §8.9.8 eq. 1395 multiplies lumaScale * component_scale: doubling
        // lumaScale roughly doubles scaleDra (and the eventual chromaScale,
        // since draChromaScaleShift varies smoothly with the QP-domain
        // shift). Verify the joined chromaScale grows monotonically.
        let t = default_chroma_qp_table(false, 0).unwrap();
        let s1 = chroma_scale_joined(256, 256, 1024, ChromaIdx::Cb, 30, &t);
        let s2 = chroma_scale_joined(512, 256, 1024, ChromaIdx::Cb, 30, &t);
        let s4 = chroma_scale_joined(1024, 256, 1024, ChromaIdx::Cb, 30, &t);
        assert!(
            s2 > s1,
            "doubling lumaScale should grow chromaScale (got {s1} → {s2})"
        );
        assert!(
            s4 > s2,
            "doubling lumaScale should grow chromaScale (got {s2} → {s4})"
        );
    }

    #[test]
    fn round193_derive_dra_chroma_state_joined_basic() {
        // End-to-end joined-path derivation: three-range syntax →
        // DraChromaDerived with per-range chroma_scales[i] populated.
        let syn = three_range_joined_chroma_syntax();
        let der = derive_dra_state(&syn, 10).unwrap();
        assert!(der.joined_scale_flag, "table_idx 30 ⇒ joined = 1");
        let t = default_chroma_qp_table(false, 0).unwrap();
        let c = derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).unwrap();
        assert_eq!(c.num_ranges_l, der.num_ranges);
        // Each chroma_scales[i] must be positive (eq. 1386 reciprocates).
        for i in 0..c.num_ranges_l {
            assert!(
                c.chroma_scales[i] > 0,
                "chroma_scales[{i}] = {} should be positive",
                c.chroma_scales[i]
            );
            assert!(
                c.inv_chroma_scales[i] > 0,
                "inv_chroma_scales[{i}] = {} should be positive",
                c.inv_chroma_scales[i]
            );
        }
    }

    #[test]
    fn round193_derive_joined_per_range_chromascales_differ() {
        // The whole point of the joined path is that chromaScales[i]
        // depend on lumaScales[i] (eq. 1395), so with distinct
        // dra_scale_value[i] entries (256/512/1024) the per-range
        // chroma_scales[] should NOT be constant — distinguishing from
        // the unjoined path where they collapse to a single value.
        let syn = three_range_joined_chroma_syntax();
        let der = derive_dra_state(&syn, 10).unwrap();
        let t = default_chroma_qp_table(false, 0).unwrap();
        let c = derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).unwrap();
        let all_same = c.chroma_scales[0..c.num_ranges_l]
            .iter()
            .all(|&v| v == c.chroma_scales[0]);
        assert!(
            !all_same,
            "joined chroma_scales should vary across ranges; got {:?}",
            &c.chroma_scales[0..c.num_ranges_l]
        );
    }

    #[test]
    fn round193_derive_joined_rejects_unjoined_state() {
        // Unjoined state (dra_table_idx == 58 ⇒ joined_scale_flag == 0)
        // should be rejected — caller should go through
        // `derive_dra_chroma_state` instead.
        let mut syn = three_range_joined_chroma_syntax();
        syn.dra_table_idx = 58;
        let der = derive_dra_state(&syn, 10).unwrap();
        assert!(!der.joined_scale_flag);
        let t = default_chroma_qp_table(false, 0).unwrap();
        assert!(derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).is_err());
    }

    #[test]
    fn round193_derive_joined_rejects_zero_cb_scale() {
        let mut syn = three_range_joined_chroma_syntax();
        syn.dra_cb_scale_value = 0;
        let der = derive_dra_state(&syn, 10).unwrap();
        let t = default_chroma_qp_table(false, 0).unwrap();
        assert!(derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).is_err());
        // Cr still derives (its own scale is non-zero).
        assert!(derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cr, 10, &t).is_ok());
    }

    #[test]
    fn round193_derive_joined_rejects_zero_luma_scale() {
        // Setting dra_scale_value[1] = 0 violates §7.4.7's scale range
        // [1, (4 << descriptor2) − 1] — but defensively, the joined helper
        // must reject before it would feed a zero into eq. 1395.
        //
        // We can't go through `derive_dra_state` to produce the
        // `DraDerived` companion here because that helper itself divides
        // by `dra_scale_value[i]` (eq. 118 InvLumaScales) and would panic
        // on a zero. Hand-build the derived state to isolate the joined
        // helper's own defensive check.
        let mut syn = three_range_joined_chroma_syntax();
        syn.dra_scale_value[1] = 0;
        let mut der = DraDerived::empty();
        der.joined_scale_flag = true;
        der.num_ranges = 3; // matches syn.dra_number_ranges_minus1 + 1
        let t = default_chroma_qp_table(false, 0).unwrap();
        assert!(derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).is_err());
    }

    #[test]
    fn round193_derive_joined_noop_on_empty_state() {
        let syn = three_range_joined_chroma_syntax();
        let mut der = DraDerived::empty();
        der.joined_scale_flag = true;
        // num_ranges intentionally 0.
        let t = default_chroma_qp_table(false, 0).unwrap();
        let c = derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).unwrap();
        assert_eq!(c.num_ranges_l, 0);
        assert_eq!(chroma_scale_for_luma_sample(100, &c), 0);
    }

    #[test]
    fn round193_chroma_scale_joined_pure_function_property() {
        // chroma_scale_joined is a pure function: same inputs ⇒ same output.
        let t = default_chroma_qp_table(false, 0).unwrap();
        let a = chroma_scale_joined(512, 256, 1024, ChromaIdx::Cb, 30, &t);
        let b = chroma_scale_joined(512, 256, 1024, ChromaIdx::Cb, 30, &t);
        assert_eq!(a, b);
    }

    #[test]
    fn round193_chroma_scale_for_sample_works_with_joined_state() {
        // The §8.9.6 entry point (chroma_scale_for_luma_sample) should
        // work transparently on a DraChromaDerived produced by either the
        // joined or unjoined builder — eq. 1384 / 1385 only consume
        // out_offsets_c / out_scales_c / out_ranges_c, which are filled
        // identically (modulo the per-i chroma_scales).
        let syn = three_range_joined_chroma_syntax();
        let der = derive_dra_state(&syn, 10).unwrap();
        let t = default_chroma_qp_table(false, 0).unwrap();
        let c = derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, 10, &t).unwrap();
        // Sample at OutRangesC[0] (= OutRangesL[0]) lands in range 0 with
        // incValue = 0; eq. 1385 collapses to OutOffsetsC[0] +
        // ((OutScalesC[0] * 0 + 512) >> 10) = OutOffsetsC[0]
        // (= inv_chroma_scales[0]). Use the actual OutRangesC[0] to avoid
        // depending on the §7.4.7 luma-derivation arithmetic.
        let s0 = chroma_scale_for_luma_sample(c.out_ranges_c[0], &c);
        assert_eq!(s0, c.out_offsets_c[0]);
        // Cross-check: a sample beyond every OutRangesC boundary lands in
        // the last range (§8.9.5 Min(rangeIdx, numRanges − 1)) and returns
        // OutOffsetsC[num_ranges_l] (= inv_chroma_scales[num_ranges_l-1],
        // with OutScalesC[num_ranges_l] = 0 per eq. 1393).
        let s_top = chroma_scale_for_luma_sample(i32::MAX as i64, &c);
        assert_eq!(s_top, c.out_offsets_c[c.num_ranges_l]);
        // Joined-path chromaScales[] differ across ranges, so the §8.9.6
        // outputs at distinct ranges must differ too.
        assert_ne!(
            s0, s_top,
            "joined chroma_scale_for_sample should vary across ranges"
        );
    }

    #[test]
    fn round193_chroma_qp_table_lookup_clamps_to_table_range() {
        // Direct clamping property of ChromaQpTable::lookup.
        let t = default_chroma_qp_table(false, 1).unwrap();
        // QpBdOffsetC = 6. Lookup at qPi = -1000 clamps to -6.
        assert_eq!(t.lookup(ChromaIdx::Cb, -1000), t.lookup(ChromaIdx::Cb, -6));
        // Lookup at qPi = 1000 clamps to 57.
        assert_eq!(t.lookup(ChromaIdx::Cb, 1000), t.lookup(ChromaIdx::Cb, 57));
    }

    #[test]
    fn round193_unjoined_derive_error_hint_mentions_joined_entry() {
        // The unjoined derive_dra_chroma_state still returns
        // Err(Unsupported) on a joined-flag state — round 193 updates its
        // error message to point the caller at the joined entry instead
        // of saying "round 187 doesn't thread through".
        let mut syn = three_range_joined_chroma_syntax();
        syn.dra_table_idx = 30; // joined
        let der = derive_dra_state(&syn, 10).unwrap();
        assert!(der.joined_scale_flag);
        let res = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, 10);
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(
            msg.contains("derive_dra_chroma_state_joined"),
            "error message should point at the joined entry; got: {msg}"
        );
    }

    // -------------------------------------------------------------------
    // Round 195 — SPS eq. 74 `ChromaQpTable` build from signalled pivots.
    // -------------------------------------------------------------------

    #[test]
    fn round195_signalled_chroma_qp_table_same_qp_table_for_chroma_one_pivot() {
        // Minimal valid input: one chroma component, one pivot. With
        // bit_depth_chroma_minus8 = 0, global_offset_flag = false ⇒
        // startQP = 0, QpBdOffsetC = 0. delta_qp_in_val_minus1[0] = 0 +
        // delta_qp_out_val[0] = 0 ⇒ qpInVal[0] = qpOutVal[0] = 0.
        //
        // ChromaQpTable[k] for k ∈ [0, 57]:
        //  * k == 0 ⇒ qpOutVal[0] = 0 (line 4011)
        //  * k > 0 ⇒ up-fill +1 each step (line 4022 / 4023)
        // So table[k] = k (clamped to [0, 57]).
        let params = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![0],
                delta_qp_out_val: vec![0],
            }],
        };
        let t = build_signalled_chroma_qp_table(&params, 0).unwrap();
        // Cb and Cr aliased.
        for qpi in 0..=57 {
            assert_eq!(t.lookup(ChromaIdx::Cb, qpi), qpi);
            assert_eq!(t.lookup(ChromaIdx::Cr, qpi), qpi);
        }
    }

    #[test]
    fn round195_signalled_chroma_qp_table_two_pivots_interpolates_linearly() {
        // Two pivots at qPi = 10 and qPi = 20, with
        // delta_qp_out_val[1] = 10 ⇒ ChromaQpTable[20] − ChromaQpTable[10]
        // = 10 (slope 1 across a 10-unit input span).
        //
        // QpBdOffsetC = 0, startQP = 0:
        //  * delta_qp_in_val_minus1[0] = 10 ⇒ qpInVal[0] = 10
        //  * delta_qp_out_val[0] = 5 ⇒ qpOutVal[0] = 15 ⇒ table[10] = 15
        //  * delta_qp_in_val_minus1[1] = 9 ⇒ qpInVal[1] = 10 + 9 + 1 = 20
        //  * delta_qp_out_val[1] = 10
        //  * sh = (9 + 1) >> 1 = 5
        //  * for k = 11..=20, m = 1..=10:
        //      table[k] = 15 + (10*m + 5) / 10
        let params = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: false,
            global_offset_flag: false,
            tables: vec![
                SignalledChromaQpTablePivots {
                    delta_qp_in_val_minus1: vec![10, 9],
                    delta_qp_out_val: vec![5, 10],
                },
                // Cr distinct: just verify it's parsed independently.
                SignalledChromaQpTablePivots {
                    delta_qp_in_val_minus1: vec![10, 9],
                    delta_qp_out_val: vec![5, 6],
                },
            ],
        };
        let t = build_signalled_chroma_qp_table(&params, 0).unwrap();
        // Cb pivot anchor at qpInVal[0] = 10:
        assert_eq!(t.lookup(ChromaIdx::Cb, 10), 15);
        // Linear interp from pivot 10 to pivot 20:
        for m in 1i32..=10 {
            let k = 10 + m;
            let expected = 15 + (10 * m + 5) / 10;
            assert_eq!(
                t.lookup(ChromaIdx::Cb, k),
                expected,
                "Cb interp at qPi = {k} (m = {m})"
            );
        }
        // ChromaQpTable[20] = 25 (slope 1 across 10 input units).
        assert_eq!(t.lookup(ChromaIdx::Cb, 20), 25);
        // Cr signalled distinctly with a different delta_qp_out_val[1]:
        // slope is 6/10, so table[20] = 15 + (6*10 + 5)/10 = 15 + 6 = 21.
        assert_eq!(t.lookup(ChromaIdx::Cr, 20), 21);
        // Cb and Cr differ — same_qp_table_for_chroma == false honoured.
        assert_ne!(
            t.lookup(ChromaIdx::Cb, 20),
            t.lookup(ChromaIdx::Cr, 20),
            "Cb and Cr must differ when same_qp_table_for_chroma == 0"
        );
    }

    #[test]
    fn round195_signalled_chroma_qp_table_down_fill_below_first_pivot_clamps() {
        // bit_depth_chroma_minus8 = 1 ⇒ QpBdOffsetC = 6.
        // Pivot at qPi = 4 with output 4. Down-fill from qPi = 3 down to
        // qPi = -6 (every step = -1, then clamped to [-6, 57]).
        let params = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![10],
                delta_qp_out_val: vec![0], // qpOutVal[0] = -6 + 10 + 0 = 4
            }],
        };
        let t = build_signalled_chroma_qp_table(&params, 1).unwrap();
        // qpInVal[0] = -6 + 10 = 4, qpOutVal[0] = 4.
        assert_eq!(t.lookup(ChromaIdx::Cb, 4), 4);
        // Down-fill: table[3] = clip(4 - 1) = 3, etc., until clamped at -6.
        assert_eq!(t.lookup(ChromaIdx::Cb, 3), 3);
        assert_eq!(t.lookup(ChromaIdx::Cb, 0), 0);
        assert_eq!(t.lookup(ChromaIdx::Cb, -3), -3);
        assert_eq!(t.lookup(ChromaIdx::Cb, -6), -6);
        // Up-fill from 4 ⇒ table[5] = 5, ..., table[57] = 57.
        for qpi in 4..=57 {
            assert_eq!(t.lookup(ChromaIdx::Cb, qpi), qpi);
        }
    }

    #[test]
    fn round195_signalled_chroma_qp_table_global_offset_flag_uses_startqp_16() {
        // global_offset_flag = true ⇒ startQP = 16. One pivot:
        //  qpInVal[0] = 16 + 4 = 20, qpOutVal[0] = 16 + 4 + 3 = 23
        let params = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: true,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![4],
                delta_qp_out_val: vec![3],
            }],
        };
        let t = build_signalled_chroma_qp_table(&params, 0).unwrap();
        assert_eq!(t.lookup(ChromaIdx::Cb, 20), 23);
        // Down-fill: table[19] = 22, table[18] = 21, ... table[0] = 3.
        assert_eq!(t.lookup(ChromaIdx::Cb, 0), 3);
        // Up-fill: table[21] = 24, ..., table[57] = 60 ⇒ clamped to 57.
        assert_eq!(t.lookup(ChromaIdx::Cb, 21), 24);
        assert_eq!(t.lookup(ChromaIdx::Cb, 57), 57);
    }

    #[test]
    fn round195_signalled_chroma_qp_table_rejects_mismatched_table_count() {
        // same_qp_table_for_chroma == true requires exactly 1 pivot set;
        // passing 2 should error.
        let bad = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![
                SignalledChromaQpTablePivots {
                    delta_qp_in_val_minus1: vec![0],
                    delta_qp_out_val: vec![0],
                },
                SignalledChromaQpTablePivots {
                    delta_qp_in_val_minus1: vec![0],
                    delta_qp_out_val: vec![0],
                },
            ],
        };
        assert!(build_signalled_chroma_qp_table(&bad, 0).is_err());
    }

    #[test]
    fn round195_signalled_chroma_qp_table_rejects_qpinval_out_of_range() {
        // qpInVal[0] = 0 + 100 = 100 > 57 ⇒ rejected.
        let bad = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![100],
                delta_qp_out_val: vec![0],
            }],
        };
        assert!(build_signalled_chroma_qp_table(&bad, 0).is_err());
    }

    #[test]
    fn round195_signalled_chroma_qp_table_rejects_empty_pivots() {
        let bad = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![],
                delta_qp_out_val: vec![],
            }],
        };
        assert!(build_signalled_chroma_qp_table(&bad, 0).is_err());
    }

    #[test]
    fn round195_signalled_chroma_qp_table_rejects_mismatched_pivot_lengths() {
        let bad = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![0, 1],
                delta_qp_out_val: vec![0], // length mismatch
            }],
        };
        assert!(build_signalled_chroma_qp_table(&bad, 0).is_err());
    }

    // -------------------------------------------------------------------
    // Round 201 — default_chroma_qp_table_identity + chroma_qp_table_for_sps
    // adapter (closes round 195 followups for monochrome / non-4:2:0).
    // -------------------------------------------------------------------

    /// Helper: a minimal `Sps` for the round-201 adapter tests. Only the
    /// fields the adapter inspects (`chroma_format_idc`, `sps_iqt_flag`,
    /// `bit_depth_chroma_minus8`, `chroma_qp_table`) need to vary across
    /// the test cases; everything else is set to a benign default.
    fn round201_sps_for_adapter(
        chroma_format_idc: u32,
        sps_iqt_flag: bool,
        bit_depth_chroma_minus8: u32,
        chroma_qp_table: Option<ChromaQpTable>,
    ) -> crate::sps::Sps {
        crate::sps::Sps {
            sps_seq_parameter_set_id: 0,
            profile_idc: 0,
            level_idc: 30,
            toolset_idc_h: 0,
            toolset_idc_l: 0,
            chroma_format_idc,
            pic_width_in_luma_samples: 64,
            pic_height_in_luma_samples: 64,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8,
            sps_btt_flag: false,
            log2_ctu_size_minus5: 0,
            log2_min_cb_size_minus2: 0,
            log2_diff_ctu_max_14_cb_size: 0,
            log2_diff_ctu_max_tt_cb_size: 0,
            log2_diff_min_cb_min_tt_cb_size_minus2: 0,
            sps_suco_flag: false,
            log2_diff_ctu_size_max_suco_cb_size: 0,
            log2_diff_max_suco_min_suco_cb_size: 0,
            sps_admvp_flag: false,
            sps_affine_flag: false,
            sps_amvr_flag: false,
            sps_dmvr_flag: false,
            sps_mmvd_flag: false,
            sps_hmvp_flag: false,
            sps_eipd_flag: false,
            sps_ibc_flag: false,
            log2_max_ibc_cand_size_minus2: 0,
            sps_cm_init_flag: false,
            sps_adcc_flag: false,
            sps_iqt_flag,
            sps_ats_flag: false,
            sps_addb_flag: false,
            sps_alf_flag: false,
            sps_htdf_flag: false,
            sps_rpl_flag: false,
            sps_pocs_flag: false,
            sps_dquant_flag: false,
            sps_dra_flag: false,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            log2_sub_gop_length: 0,
            log2_ref_pic_gap_length: 0,
            max_num_tid0_ref_pics: 0,
            sps_max_dec_pic_buffering_minus1: 0,
            long_term_ref_pics_flag: false,
            rpl1_same_as_rpl0_flag: false,
            num_ref_pic_lists_in_sps_l0: 0,
            num_ref_pic_lists_in_sps_l1: 0,
            ref_pic_list_structs_l0: Vec::new(),
            ref_pic_list_structs_l1: Vec::new(),
            picture_cropping_flag: false,
            picture_crop_left_offset: 0,
            picture_crop_right_offset: 0,
            picture_crop_top_offset: 0,
            picture_crop_bottom_offset: 0,
            chroma_qp_table,
            vui_parameters_present_flag: false,
        }
    }

    #[test]
    fn round201_identity_chroma_qp_table_8bit_returns_qpi() {
        // bit_depth_chroma_minus8 = 0 ⇒ QpBdOffsetC = 0, qPi ∈ [0, 57].
        let t = default_chroma_qp_table_identity(0).unwrap();
        for qpi in 0..=57 {
            assert_eq!(
                t.lookup(ChromaIdx::Cb, qpi),
                qpi,
                "Cb identity must return qPi at {qpi}"
            );
            assert_eq!(
                t.lookup(ChromaIdx::Cr, qpi),
                qpi,
                "Cr identity must return qPi at {qpi}"
            );
        }
        // Cb and Cr are byte-for-byte equal per spec page-67 "Otherwise"
        // ("m being equal to 0 and 1").
        assert_eq!(t.cb.table, t.cr.table);
        assert_eq!(t.cb.qp_bd_offset_c, 0);
        assert_eq!(t.cr.qp_bd_offset_c, 0);
    }

    #[test]
    fn round201_identity_chroma_qp_table_10bit_negative_qpi_in_range() {
        // bit_depth_chroma_minus8 = 2 ⇒ QpBdOffsetC = 12, qPi ∈ [-12, 57].
        let t = default_chroma_qp_table_identity(2).unwrap();
        assert_eq!(t.cb.qp_bd_offset_c, 12);
        assert_eq!(t.cb.table.len(), 12 + 58);
        for qpi in -12..=57 {
            assert_eq!(t.lookup(ChromaIdx::Cb, qpi), qpi);
            assert_eq!(t.lookup(ChromaIdx::Cr, qpi), qpi);
        }
    }

    #[test]
    fn round201_identity_chroma_qp_table_lookup_clamps_out_of_range() {
        // Spec eq. 1403 / 1404 clamping still applies on the identity
        // table — values outside [-QpBdOffsetC, 57] clamp into range,
        // so the returned QpC is the clamped boundary (NOT the original
        // qPi).
        let t = default_chroma_qp_table_identity(1).unwrap();
        // QpBdOffsetC = 6. -1000 clamps to -6; +1000 clamps to 57.
        assert_eq!(t.lookup(ChromaIdx::Cb, -1000), -6);
        assert_eq!(t.lookup(ChromaIdx::Cb, 1000), 57);
        assert_eq!(t.lookup(ChromaIdx::Cr, -1000), -6);
        assert_eq!(t.lookup(ChromaIdx::Cr, 1000), 57);
    }

    #[test]
    fn round201_identity_chroma_qp_table_rejects_out_of_range_bit_depth() {
        assert!(default_chroma_qp_table_identity(9).is_err());
        assert!(default_chroma_qp_table_identity(100).is_err());
    }

    #[test]
    fn round201_identity_differs_from_table5_default() {
        // The spec-page-67 "Otherwise" identity table is materially
        // different from the Table-5 default (which folds the [29, 41]
        // range of qPi ∈ [30, 57] to a non-trivial QpC). This test
        // documents that distinction so a future refactor doesn't
        // collapse the two helpers.
        let id = default_chroma_qp_table_identity(0).unwrap();
        let t5 = default_chroma_qp_table(false, 0).unwrap();
        // qPi = 30 ⇒ identity says 30, Table 5 says 29.
        assert_eq!(id.lookup(ChromaIdx::Cb, 30), 30);
        assert_eq!(t5.lookup(ChromaIdx::Cb, 30), 29);
        // qPi = 57 ⇒ identity says 57, Table 5 says 41.
        assert_eq!(id.lookup(ChromaIdx::Cb, 57), 57);
        assert_eq!(t5.lookup(ChromaIdx::Cb, 57), 41);
    }

    #[test]
    fn round201_adapter_some_chroma_qp_table_returns_it_verbatim() {
        // Build a recognisable signalled table and stash it on the SPS;
        // the adapter must surface it without re-deriving.
        let params = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![0],
                delta_qp_out_val: vec![0],
            }],
        };
        let parsed = build_signalled_chroma_qp_table(&params, 0).unwrap();
        let sps = round201_sps_for_adapter(1, false, 0, Some(parsed.clone()));
        let out = chroma_qp_table_for_sps(&sps).unwrap();
        assert_eq!(out.cb.qp_bd_offset_c, parsed.cb.qp_bd_offset_c);
        assert_eq!(out.cb.table, parsed.cb.table);
        assert_eq!(out.cr.qp_bd_offset_c, parsed.cr.qp_bd_offset_c);
        assert_eq!(out.cr.table, parsed.cr.table);
    }

    #[test]
    fn round201_adapter_420_no_signalled_falls_to_table5() {
        // chroma_format_idc == 1, sps_iqt_flag == 0, no signalled table
        // ⇒ default_chroma_qp_table(false, bit_depth_chroma_minus8).
        let sps = round201_sps_for_adapter(1, false, 0, None);
        let out = chroma_qp_table_for_sps(&sps).unwrap();
        let expected = default_chroma_qp_table(false, 0).unwrap();
        assert_eq!(out.cb.table, expected.cb.table);
        // Spot check: qPi = 30 ⇒ 29 (Table 5).
        assert_eq!(out.lookup(ChromaIdx::Cb, 30), 29);
    }

    #[test]
    fn round201_adapter_420_no_signalled_iqt_falls_to_table6() {
        // chroma_format_idc == 1, sps_iqt_flag == 1, no signalled table
        // ⇒ default_chroma_qp_table(true, bit_depth_chroma_minus8).
        let sps = round201_sps_for_adapter(1, true, 0, None);
        let out = chroma_qp_table_for_sps(&sps).unwrap();
        let expected = default_chroma_qp_table(true, 0).unwrap();
        assert_eq!(out.cb.table, expected.cb.table);
        // Spot check: qPi = 30 ⇒ 29 (Table 6), qPi = 31 ⇒ 30.
        assert_eq!(out.lookup(ChromaIdx::Cb, 30), 29);
        assert_eq!(out.lookup(ChromaIdx::Cb, 31), 30);
    }

    #[test]
    fn round201_adapter_monochrome_no_signalled_falls_to_identity() {
        // chroma_format_idc == 0 ⇒ spec page-67 "Otherwise" identity
        // table. The §7.4.3.1 parser never even enters the
        // chroma_qp_table_present_flag body for monochrome, so the
        // adapter MUST synthesise the identity on demand.
        let sps = round201_sps_for_adapter(0, false, 0, None);
        let out = chroma_qp_table_for_sps(&sps).unwrap();
        for qpi in 0..=57 {
            assert_eq!(out.lookup(ChromaIdx::Cb, qpi), qpi);
            assert_eq!(out.lookup(ChromaIdx::Cr, qpi), qpi);
        }
    }

    #[test]
    fn round201_adapter_422_no_signalled_falls_to_identity() {
        // chroma_format_idc == 2 (4:2:2) ⇒ "Otherwise" branch identity.
        // Differs from the 4:2:0 case: Table 5/6 are gated on
        // ChromaArrayType == 1.
        let sps = round201_sps_for_adapter(2, false, 0, None);
        let out = chroma_qp_table_for_sps(&sps).unwrap();
        let id = default_chroma_qp_table_identity(0).unwrap();
        assert_eq!(out.cb.table, id.cb.table);
        // Specifically NOT Table 5 — the 4:2:2 fallback is identity,
        // not Table 5/6.
        let t5 = default_chroma_qp_table(false, 0).unwrap();
        assert_ne!(out.cb.table, t5.cb.table);
    }

    #[test]
    fn round201_adapter_444_no_signalled_falls_to_identity() {
        // chroma_format_idc == 3 (4:4:4) ⇒ same "Otherwise" branch as
        // 4:2:2 / monochrome.
        let sps = round201_sps_for_adapter(3, true, 0, None);
        let out = chroma_qp_table_for_sps(&sps).unwrap();
        let id = default_chroma_qp_table_identity(0).unwrap();
        assert_eq!(out.cb.table, id.cb.table);
        assert_eq!(out.cr.table, id.cr.table);
        // sps_iqt_flag is irrelevant in the "Otherwise" branch (Tables
        // 5/6 don't apply): the identity table is the same whether
        // sps_iqt_flag is 0 or 1.
        let mut sps_iqt_off = sps.clone();
        sps_iqt_off.sps_iqt_flag = false;
        let out_iqt_off = chroma_qp_table_for_sps(&sps_iqt_off).unwrap();
        assert_eq!(out.cb.table, out_iqt_off.cb.table);
    }

    #[test]
    fn round201_adapter_chains_into_derive_dra_chroma_state_joined() {
        // End-to-end check: the adapter's output is directly consumable
        // by the round-193 joined §8.9.7 derivation. On a monochrome
        // SPS, the adapter synthesises the identity table; the joined
        // derive accepts it and returns positive per-range chromaScales.
        let sps = round201_sps_for_adapter(0, false, 2, None); // 10-bit mono
        let table = chroma_qp_table_for_sps(&sps).unwrap();
        // bit_depth_chroma_minus8 = 2 ⇒ QpBdOffsetC = 12.
        assert_eq!(table.cb.qp_bd_offset_c, 12);
        // Borrow the round-193 three-range joined syntax helper for a
        // realistic invocation. bit_depth_y must match Sps.
        let mut syn = three_range_joined_chroma_syntax();
        syn.dra_table_idx = 30; // joined branch
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(der.joined_scale_flag);
        let chroma =
            derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, sps.bit_depth_y(), &table)
                .unwrap();
        // All per-range chromaScales must be strictly positive (eq.
        // 1386 reciprocates these; zero would be rejected upstream).
        for i in 0..chroma.num_ranges_l {
            assert!(
                chroma.chroma_scales[i] > 0,
                "chromaScales[{i}] = {} must be > 0",
                chroma.chroma_scales[i]
            );
            assert!(
                chroma.inv_chroma_scales[i] > 0,
                "invChromaScales[{i}] = {} must be > 0",
                chroma.inv_chroma_scales[i]
            );
        }
    }

    // ============================================================
    // Round 207 — SPS-aware `derive_dra_chroma_state_for_sps`
    // adapter: closes the SPS → §8.9.6 chroma chain on both the
    // joined (§8.9.8) and unjoined (§8.9.7) paths.
    // ============================================================

    /// Round-207 SPS builder. Same shape as `round201_sps_for_adapter`
    /// but also exposes `bit_depth_luma_minus8` so the chained tests
    /// can exercise the spec's 10-bit branch on the joined chroma
    /// derivation.
    fn round207_sps_for_adapter(
        chroma_format_idc: u32,
        sps_iqt_flag: bool,
        bit_depth_luma_minus8: u32,
        bit_depth_chroma_minus8: u32,
        chroma_qp_table: Option<ChromaQpTable>,
    ) -> crate::sps::Sps {
        crate::sps::Sps {
            sps_seq_parameter_set_id: 0,
            profile_idc: 0,
            level_idc: 30,
            toolset_idc_h: 0,
            toolset_idc_l: 0,
            chroma_format_idc,
            pic_width_in_luma_samples: 64,
            pic_height_in_luma_samples: 64,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            sps_btt_flag: false,
            log2_ctu_size_minus5: 0,
            log2_min_cb_size_minus2: 0,
            log2_diff_ctu_max_14_cb_size: 0,
            log2_diff_ctu_max_tt_cb_size: 0,
            log2_diff_min_cb_min_tt_cb_size_minus2: 0,
            sps_suco_flag: false,
            log2_diff_ctu_size_max_suco_cb_size: 0,
            log2_diff_max_suco_min_suco_cb_size: 0,
            sps_admvp_flag: false,
            sps_affine_flag: false,
            sps_amvr_flag: false,
            sps_dmvr_flag: false,
            sps_mmvd_flag: false,
            sps_hmvp_flag: false,
            sps_eipd_flag: false,
            sps_ibc_flag: false,
            log2_max_ibc_cand_size_minus2: 0,
            sps_cm_init_flag: false,
            sps_adcc_flag: false,
            sps_iqt_flag,
            sps_ats_flag: false,
            sps_addb_flag: false,
            sps_alf_flag: false,
            sps_htdf_flag: false,
            sps_rpl_flag: false,
            sps_pocs_flag: false,
            sps_dquant_flag: false,
            sps_dra_flag: false,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            log2_sub_gop_length: 0,
            log2_ref_pic_gap_length: 0,
            max_num_tid0_ref_pics: 0,
            sps_max_dec_pic_buffering_minus1: 0,
            long_term_ref_pics_flag: false,
            rpl1_same_as_rpl0_flag: false,
            num_ref_pic_lists_in_sps_l0: 0,
            num_ref_pic_lists_in_sps_l1: 0,
            ref_pic_list_structs_l0: Vec::new(),
            ref_pic_list_structs_l1: Vec::new(),
            picture_cropping_flag: false,
            picture_crop_left_offset: 0,
            picture_crop_right_offset: 0,
            picture_crop_top_offset: 0,
            picture_crop_bottom_offset: 0,
            chroma_qp_table,
            vui_parameters_present_flag: false,
        }
    }

    #[test]
    fn round207_for_sps_unjoined_matches_direct_unjoined() {
        // dra_table_idx = 58 ⇒ DraJoinedScaleFlag = 0 ⇒ §8.9.7 unjoined
        // path. The SPS adapter must produce byte-identical output to
        // `derive_dra_chroma_state` invoked directly with the same
        // `bit_depth_y`. ChromaQpTable is irrelevant here per eq. 1394.
        let sps = round207_sps_for_adapter(1, false, 2, 2, None); // 10-bit
        let syn = three_range_unjoined_chroma_syntax(sps.bit_depth_y());
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(
            !der.joined_scale_flag,
            "dra_table_idx = 58 must clear joined_scale_flag"
        );
        let direct = derive_dra_chroma_state(&syn, &der, ChromaIdx::Cb, sps.bit_depth_y()).unwrap();
        let via_sps = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cb, &sps).unwrap();
        // Byte-identical state on the unjoined path.
        assert_eq!(direct.num_ranges_l, via_sps.num_ranges_l);
        assert_eq!(direct.chroma_scales, via_sps.chroma_scales);
        assert_eq!(direct.inv_chroma_scales, via_sps.inv_chroma_scales);
        assert_eq!(direct.out_scales_c, via_sps.out_scales_c);
        assert_eq!(direct.out_offsets_c, via_sps.out_offsets_c);
        assert_eq!(direct.out_ranges_c, via_sps.out_ranges_c);
    }

    #[test]
    fn round207_for_sps_joined_matches_direct_joined() {
        // dra_table_idx ∈ [0, 57] ⇒ DraJoinedScaleFlag = 1 ⇒ §8.9.8
        // joined path. The SPS adapter must produce byte-identical
        // output to `derive_dra_chroma_state_joined` invoked directly
        // with the SPS-active `ChromaQpTable`.
        let sps = round207_sps_for_adapter(1, false, 0, 0, None); // 8-bit, 4:2:0
        let syn = three_range_joined_chroma_syntax();
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(
            der.joined_scale_flag,
            "dra_table_idx != 58 must set joined_scale_flag"
        );
        let table = chroma_qp_table_for_sps(&sps).unwrap();
        let direct =
            derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cr, sps.bit_depth_y(), &table)
                .unwrap();
        let via_sps = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cr, &sps).unwrap();
        assert_eq!(direct.num_ranges_l, via_sps.num_ranges_l);
        assert_eq!(direct.chroma_scales, via_sps.chroma_scales);
        assert_eq!(direct.inv_chroma_scales, via_sps.inv_chroma_scales);
        assert_eq!(direct.out_scales_c, via_sps.out_scales_c);
        assert_eq!(direct.out_offsets_c, via_sps.out_offsets_c);
        assert_eq!(direct.out_ranges_c, via_sps.out_ranges_c);
    }

    #[test]
    fn round207_for_sps_dispatches_on_joined_scale_flag() {
        // Same SPS, same derived state — only the syntax's
        // dra_table_idx changes between the two invocations. The
        // adapter must dispatch correctly off `derived.joined_scale_flag`
        // (set by `derive_dra_state` from `dra_table_idx`), NOT off
        // any SPS field.
        let sps = round207_sps_for_adapter(1, false, 2, 2, None); // 10-bit
                                                                  // Unjoined branch.
        let syn_u = three_range_unjoined_chroma_syntax(sps.bit_depth_y());
        let der_u = derive_dra_state(&syn_u, sps.bit_depth_y()).unwrap();
        assert!(!der_u.joined_scale_flag);
        let out_u = derive_dra_chroma_state_for_sps(&syn_u, &der_u, ChromaIdx::Cb, &sps).unwrap();
        // §8.9.7: chromaScales[i] is constant across i (= dra_cb_scale_value).
        for i in 1..out_u.num_ranges_l {
            assert_eq!(
                out_u.chroma_scales[i], out_u.chroma_scales[0],
                "unjoined chromaScales[i] must be constant"
            );
            assert_eq!(out_u.chroma_scales[i], syn_u.dra_cb_scale_value as i64);
        }
        // Joined branch.
        let syn_j = three_range_joined_chroma_syntax();
        let der_j = derive_dra_state(&syn_j, sps.bit_depth_y()).unwrap();
        assert!(der_j.joined_scale_flag);
        let out_j = derive_dra_chroma_state_for_sps(&syn_j, &der_j, ChromaIdx::Cb, &sps).unwrap();
        // §8.9.8: chromaScales[i] is a per-range function of
        // `chroma_scale_joined(lumaScales[i], …)` — at minimum not
        // a single constant when the underlying luma scales differ.
        // Verify all are strictly positive (so eq. 1386 reciprocates).
        for i in 0..out_j.num_ranges_l {
            assert!(
                out_j.chroma_scales[i] > 0,
                "joined chromaScales[{i}] must be > 0"
            );
        }
        // And the joined+unjoined paths produce different states
        // (under three_range_joined the per-range luma scales vary
        // 256/512/1024, so the joined chroma scales must vary too).
        let varies =
            (1..out_j.num_ranges_l).any(|i| out_j.chroma_scales[i] != out_j.chroma_scales[0]);
        assert!(varies, "joined chromaScales[i] should vary across i");
    }

    #[test]
    fn round207_for_sps_uses_signalled_chroma_qp_table_on_joined_path() {
        // Build a non-trivial signalled ChromaQpTable, stash it on the
        // SPS, and verify the joined dispatch consumes it (not the
        // Table 5/6 default). Effect is observable because the
        // signalled table differs from Table 5 at qPi = 30 (signalled
        // is identity ⇒ 30 here; Table 5 ⇒ 29).
        let params = SignalledChromaQpTableParams {
            same_qp_table_for_chroma: true,
            global_offset_flag: false,
            // Single pivot: identity ramp (delta_in = 0 ⇒ minus1 = 0,
            // delta_out = 0). Spec pivot-fill at the start qpInVal[0]
            // anchors to 0 and propagates identity outward.
            tables: vec![SignalledChromaQpTablePivots {
                delta_qp_in_val_minus1: vec![0],
                delta_qp_out_val: vec![0],
            }],
        };
        let signalled = build_signalled_chroma_qp_table(&params, 0).unwrap();
        let sps = round207_sps_for_adapter(1, false, 0, 0, Some(signalled.clone()));
        // Use the joined three-range syntax (dra_table_idx = 30 ⇒
        // joined_scale_flag = true).
        let syn = three_range_joined_chroma_syntax();
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(der.joined_scale_flag);
        // Drive both via_sps and direct with the signalled table; they
        // must match. Then verify Table-5 default produces a
        // *different* output — proving the dispatch is reading
        // sps.chroma_qp_table, not falling back to Table 5.
        let via_sps = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cb, &sps).unwrap();
        let direct_signalled = derive_dra_chroma_state_joined(
            &syn,
            &der,
            ChromaIdx::Cb,
            sps.bit_depth_y(),
            &signalled,
        )
        .unwrap();
        assert_eq!(via_sps.chroma_scales, direct_signalled.chroma_scales);
        let default = default_chroma_qp_table(false, 0).unwrap();
        let direct_default =
            derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, sps.bit_depth_y(), &default)
                .unwrap();
        // Signalled (identity) vs default (Table 5) must produce
        // observably different state, otherwise the test can't
        // distinguish the two dispatch outcomes.
        assert_ne!(
            direct_signalled.chroma_scales, direct_default.chroma_scales,
            "signalled identity ChromaQpTable must differ from Table 5 default"
        );
    }

    #[test]
    fn round207_for_sps_propagates_zero_scale_error_unjoined() {
        // Zero chroma scale on the unjoined path ⇒ the underlying
        // `derive_dra_chroma_state` errors (eq. 1386 divide-by-zero
        // guard). The adapter must surface the error verbatim.
        let sps = round207_sps_for_adapter(1, false, 0, 0, None);
        let mut syn = three_range_unjoined_chroma_syntax(sps.bit_depth_y());
        syn.dra_cb_scale_value = 0;
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(!der.joined_scale_flag);
        let err = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cb, &sps);
        assert!(err.is_err());
    }

    #[test]
    fn round207_for_sps_propagates_zero_cb_scale_error_joined() {
        // Zero `dra_cb_scale_value` on the joined path ⇒ the underlying
        // `derive_dra_chroma_state_joined` errors (§7.4.7 forbids
        // zero scales; eq. 1386 would divide by zero). Adapter must
        // surface this verbatim.
        let sps = round207_sps_for_adapter(1, false, 0, 0, None);
        let mut syn = three_range_joined_chroma_syntax();
        syn.dra_cb_scale_value = 0;
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(der.joined_scale_flag);
        let err = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cb, &sps);
        assert!(err.is_err(), "adapter must surface zero-Cb-scale error");
        // Cr path with the same syntax still derives fine (Cr scale
        // is non-zero).
        let ok = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cr, &sps);
        assert!(ok.is_ok(), "Cr path should still derive cleanly");
    }

    #[test]
    fn round207_for_sps_monochrome_synthesises_identity_on_joined_path() {
        // Monochrome SPS (chroma_format_idc = 0): the §7.4.3.1 parser
        // never enters the chroma_qp_table_present_flag body, so
        // sps.chroma_qp_table = None. The round-201 adapter
        // synthesises the spec-page-67 "Otherwise" identity table.
        // The round-207 adapter then feeds it to the joined derive
        // and the chain stays bit-faithful (every chromaScales[i] > 0).
        let sps = round207_sps_for_adapter(0, false, 2, 2, None); // 10-bit mono
        let syn = three_range_joined_chroma_syntax();
        let der = derive_dra_state(&syn, sps.bit_depth_y()).unwrap();
        assert!(der.joined_scale_flag);
        let out = derive_dra_chroma_state_for_sps(&syn, &der, ChromaIdx::Cb, &sps).unwrap();
        for i in 0..out.num_ranges_l {
            assert!(out.chroma_scales[i] > 0);
            assert!(out.inv_chroma_scales[i] > 0);
        }
        // And direct invocation with the round-201 SPS-adapted table
        // must match the round-207 SPS dispatch.
        let table = chroma_qp_table_for_sps(&sps).unwrap();
        let direct =
            derive_dra_chroma_state_joined(&syn, &der, ChromaIdx::Cb, sps.bit_depth_y(), &table)
                .unwrap();
        assert_eq!(out.chroma_scales, direct.chroma_scales);
        assert_eq!(out.out_ranges_c, direct.out_ranges_c);
    }

    #[test]
    fn round207_for_sps_threads_bit_depth_y_from_sps() {
        // The adapter must read bit_depth_y from sps.bit_depth_y(), not
        // assume 8-bit / 10-bit. Build two SPSes that differ only in
        // bit_depth_luma_minus8 and verify the chroma derivation
        // captures the difference (§8.9.5 top sentinel `1 <<
        // bit_depth_y` differs).
        let sps_8 = round207_sps_for_adapter(1, false, 0, 0, None); // 8-bit
        let sps_10 = round207_sps_for_adapter(1, false, 2, 2, None); // 10-bit
        let syn = three_range_unjoined_chroma_syntax(10);
        let der_8 = derive_dra_state(&syn, sps_8.bit_depth_y()).unwrap();
        let der_10 = derive_dra_state(&syn, sps_10.bit_depth_y()).unwrap();
        let out_8 = derive_dra_chroma_state_for_sps(&syn, &der_8, ChromaIdx::Cb, &sps_8).unwrap();
        let out_10 =
            derive_dra_chroma_state_for_sps(&syn, &der_10, ChromaIdx::Cb, &sps_10).unwrap();
        // Top sentinel reflects the bit-depth difference: §8.9.5 top
        // = 1 << bit_depth_y. At num_ranges_c the sentinel sits.
        let num_ranges_c_8 = out_8.num_ranges_l + 1;
        let num_ranges_c_10 = out_10.num_ranges_l + 1;
        assert_eq!(out_8.out_ranges_c[num_ranges_c_8], 1i64 << 8);
        assert_eq!(out_10.out_ranges_c[num_ranges_c_10], 1i64 << 10);
    }

    // ------------------------------------------------------------------
    // Round 284 — §8.9.8 tableNum == 0 branch pins (spec page 308).
    //
    // The spec's "If tableNum is equal to 0, the variable qpDraFrac is
    // set equal to 0, and the variable qpDraInt is decreased by 1"
    // sentence leaves qpDraFracAdj and draChromaQpShift (eq. 1408/1409,
    // printed inside the "otherwise" arm) underivable by literal
    // reading. The in-repo errata file carries no §8.9.8 entry yet
    // (docs task #1278), so these tests pin the documented in-tree
    // reading: qpDraFracAdj = 0 (eq. 1418 then adds exactly
    // (1 << 8) >> 9 = 0) and draChromaQpShift =
    // ChromaQpTable[cIdx][dra_table_idx] − qp0 − qpDraInt with the
    // decremented qpDraInt and idx0 = Clip3(−QpBdOffsetC, 57,
    // dra_table_idx − qpDraInt). Under that reading the whole branch
    // collapses to the closed form asserted below.
    // ------------------------------------------------------------------

    /// The tableNum == 0 closed form: chromaScale =
    /// (scaleDra * QpScale[Clip3(0, 24, shift + 12)] + (1 << 17)) >> 18
    /// with shift = ChromaQpTable[dra_table_idx] − ChromaQpTable[idx0]
    /// − qpDraInt and qpDraInt = 2 * IndexScaleQP − 61.
    fn table_num_zero_closed_form(
        scale_dra: i64,
        index_scale_qp: i64,
        dra_table_idx: i32,
        t: &ChromaQpTable,
        cidx: ChromaIdx,
    ) -> i64 {
        let qp_dra_int = 2 * index_scale_qp - 60 - 1;
        let lo = -t.cb.qp_bd_offset_c as i64;
        let idx0 = (dra_table_idx as i64 - qp_dra_int).clamp(lo, 57);
        let shift =
            t.lookup(cidx, dra_table_idx) as i64 - t.lookup(cidx, idx0 as i32) as i64 - qp_dra_int;
        let scale_shift = QP_SCALE[(shift + 12).clamp(0, 24) as usize];
        ((scale_dra * scale_shift) + (1i64 << 17)) >> 18
    }

    #[test]
    fn round284_table_num_zero_knot_1448_closed_form() {
        // lumaScale 2896 × Cb scale 256 → scaleDra = 741376,
        // scaleDraNorm = (741376 + 256) >> 9 = 1448 = ScaleQP[35]
        // exactly, so tableNum = 0 with IndexScaleQP = 35.
        let t = default_chroma_qp_table(false, 0).unwrap();
        let got = chroma_scale_joined(2896, 256, 1024, ChromaIdx::Cb, 30, &t);
        let want = table_num_zero_closed_form(2896 * 256, 35, 30, &t, ChromaIdx::Cb);
        assert_eq!(got, want);
        assert!(got > 0);
    }

    #[test]
    fn round284_table_num_zero_knot_724_closed_form() {
        // lumaScale 1448 × Cb scale 256 → scaleDra = 370688,
        // scaleDraNorm = (370688 + 256) >> 9 = 724 = ScaleQP[32]
        // exactly: a second knot with a different (and small) qpDraInt
        // (2 * 32 − 61 = 3), exercising the idx0 clip away from the
        // qpDraInt = 9 case.
        let t = default_chroma_qp_table(false, 0).unwrap();
        for cidx in [ChromaIdx::Cb, ChromaIdx::Cr] {
            let (cb, cr) = (256u16, 256u16);
            let got = chroma_scale_joined(1448, cb, cr, cidx, 30, &t);
            let want = table_num_zero_closed_form(1448 * 256, 32, 30, &t, cidx);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn round284_table_num_zero_neighbour_takes_otherwise_branch() {
        // One scaleDraNorm step past the 1448 knot (lumaScale 2897 →
        // scaleDra = 741632, norm = 1449) lands in the eq. 1400-1411
        // arm; it must stay positive and within the same ballpark as
        // the knot value (the QP-domain shift moves by well under one
        // QpScale octave for a single norm step).
        let t = default_chroma_qp_table(false, 0).unwrap();
        let at_knot = chroma_scale_joined(2896, 256, 1024, ChromaIdx::Cb, 30, &t);
        let past_knot = chroma_scale_joined(2897, 256, 1024, ChromaIdx::Cb, 30, &t);
        assert!(past_knot > 0);
        assert!(
            past_knot > at_knot / 2 && past_knot < at_knot * 2,
            "branch seam should not jump an octave: {at_knot} vs {past_knot}"
        );
    }
}
