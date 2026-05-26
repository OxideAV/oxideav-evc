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
/// `OutRangesC` from §8.9.7 + §8.9.8) is still parked pending the
/// §7.3.6-faithful APS parser rewrite.
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
}
