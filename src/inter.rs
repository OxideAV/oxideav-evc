//! EVC inter prediction (ISO/IEC 23094-1 §8.5).
//!
//! Round-4 scope: Baseline-profile P / B slice support — a single
//! reference picture per list, no advanced merge / HMVP / temporal-merge /
//! MMVD / CIIP / GPM / affine / DMVR. The Baseline toolset constraint set
//! (Annex A.3.2) wires `sps_admvp_flag = sps_amvr_flag = sps_mmvd_flag =
//! sps_affine_flag = sps_dmvr_flag = sps_hmvp_flag = 0`, which collapses
//! every motion-vector path through:
//!
//! * **AMVP** (§8.5.2.4 path with `sps_admvp_flag = 0`): the `mvp_idx_lX`
//!   ∈ 0..=3 syntax element selects one of four candidates from
//!   §8.5.2.4.3 (left at `(xCb-1, yCb)`, above at `(xCb, yCb-1)`, above-
//!   right at `(xCb+nCbW, yCb-1)`, and a temporal slot which under
//!   round-4 simplifies to "zero MV"). When a spatial neighbour is
//!   unavailable the spec says the candidate is `(1, 1)`.
//! * **MV reconstruction**: `mvLX = wrap16(mvpLX + mvdLX)` per eq.
//!   436–439 (the `2^16` modulus reduces to a 16-bit signed wrap).
//! * **Conversion 1/4 → 1/16** for luma: `mv <<= 2` (§8.5.2.9 / eq. 400).
//! * **Chroma MV** at 1/32-pel: `mvCLX = mvLX * 2 / SubWidthC` (4:2:0
//!   gives a passthrough, since the luma MV is already in 1/16 luma-pel
//!   and 1/32 chroma-pel == 1/16 luma-pel for 4:2:0).
//!
//! ## Sub-pel interpolation
//!
//! Two filter banks ship as part of round-4:
//!
//! * **Luma**: an 8-tap separable filter at 1/16-pel resolution. The
//!   Baseline coefficient table is Table 25 (`sps_admvp_flag = 0`); only
//!   phases 4, 8 and 12 (the 1/4-pel grid) are populated, the rest are
//!   "na". Round-4 keeps that constraint: a non-baseline phase (1, 2, 3,
//!   5, 6, 7, 9, 10, 11, 13, 14, 15) surfaces as `Error::Unsupported`.
//! * **Chroma**: a 4-tap separable filter at 1/32-pel resolution
//!   (Table 27). Baseline populates phases 4, 8, 12, 16, 20, 24, 28 only
//!   (i.e. the 1/8-pel grid).
//!
//! ## Default-weighted bi-prediction
//!
//! For B slices with `predFlagL0 == predFlagL1 == 1`:
//!
//! ```text
//! pbSamples[x][y] = (predSamplesL0[x][y] + predSamplesL1[x][y] + 1) >> 1
//! ```
//!
//! per §8.5.4.5 eq. 988. Explicit weights (sps_dra_flag path) are out of
//! scope.
//!
//! All section / clause numbers refer to **ISO/IEC 23094-1:2020(E)**.

use oxideav_core::{Error, Result};

/// A 2-D luma motion vector in 1/4-pel resolution (signed 16-bit per
/// component). After §8.5.2.9 conversion the same vector lives at 1/16
/// luma-pel resolution; we never represent the intermediate state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub x: i32,
    pub y: i32,
}

impl MotionVector {
    /// Construct from raw 1/4-pel components.
    pub fn quarter_pel(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// 16-bit modular addition per eq. 436–439. The spec's published
    /// expressions (`+ 216` / `% 216`) are typesetting artefacts of
    /// `+ 2^16` / `% 2^16`; the result is the signed 16-bit wrap of
    /// `mvp + mvd`.
    pub fn wrapping_add(&self, mvd: &Self) -> Self {
        let x = wrap16(self.x.wrapping_add(mvd.x));
        let y = wrap16(self.y.wrapping_add(mvd.y));
        Self { x, y }
    }

    /// Convert from 1/4-pel to 1/16-pel (luma reference grid). Eq. 680/681.
    pub fn quarter_to_sixteenth(&self) -> Self {
        Self {
            x: self.x << 2,
            y: self.y << 2,
        }
    }
}

/// Wrap to signed 16-bit (i.e. range `[-32768, 32767]`). Eq. 437/439 says
/// `mv = u >= 2^15 ? u - 2^16 : u`. We compute the mod-2^16 reduction
/// then sign-extend.
fn wrap16(v: i32) -> i32 {
    let u = (v as u32) & 0xFFFF;
    if u >= 0x8000 {
        (u as i32) - 0x10000
    } else {
        u as i32
    }
}

/// §8.5.3.10 Rounding process for motion vectors — eq. 907/908/909.
///
/// Inputs:
///   * `mv` — the motion vector to round.
///   * `right_shift` — right-shift parameter for rounding.
///   * `left_shift` — left-shift parameter for resolution increase
///     after rounding.
///
/// Output: the rounded motion vector. Per the spec, the offset is
/// `(rightShift == 0) ? 0 : (1 << (rightShift − 1))`, and each component
/// is rounded as `((mv + offset − (mv >= 0 ? 1 : 0)) >> rightShift) << leftShift`.
///
/// The `− (mv >= 0)` term gives "round half away from zero" for negative
/// values and "round half toward negative infinity" for positives, which
/// matches the spec's expression `mv[ k ] >= 0` (a boolean coerced to
/// `{ 0, 1 }`). This is the rounding mode used by §8.5.3 affine derivation
/// (eq. 911, 918, 953, 962, 1023) and by AMVR resolution scaling.
pub fn round_motion_vector(mv: MotionVector, right_shift: u32, left_shift: u32) -> MotionVector {
    let offset: i32 = if right_shift == 0 {
        0
    } else {
        1i32 << (right_shift - 1)
    };
    let round_component = |c: i32| -> i32 {
        let nonneg_adj = if c >= 0 { 1 } else { 0 };
        ((c + offset - nonneg_adj) >> right_shift) << left_shift
    };
    MotionVector {
        x: round_component(mv.x),
        y: round_component(mv.y),
    }
}

/// A reference picture as the inter-prediction module sees it: borrowed
/// luma + chroma planes plus their dimensions. The buffer is owned by
/// the decoder's reference-frame cache; this view is `Copy`-cheap.
#[derive(Clone, Copy, Debug)]
pub struct RefPictureView<'a> {
    pub y: &'a [u8],
    pub cb: &'a [u8],
    pub cr: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub y_stride: usize,
    pub c_stride: usize,
    pub chroma_format_idc: u32,
}

impl<'a> RefPictureView<'a> {
    fn pic_w_c(&self) -> u32 {
        match self.chroma_format_idc {
            0 => 0,
            1 | 2 => self.width.div_ceil(2),
            3 => self.width,
            _ => 0,
        }
    }
    fn pic_h_c(&self) -> u32 {
        match self.chroma_format_idc {
            0 => 0,
            2 | 3 => self.height,
            1 => self.height.div_ceil(2),
            _ => 0,
        }
    }
}

/// Table 25 (ISO/IEC 23094-1:2020 §8.5.4.3.2). Baseline luma 8-tap
/// interpolation filter coefficients at 1/16-pel resolution. Only phases
/// 4, 8 and 12 are non-`na` — Baseline restricts MVs to the 1/4-pel grid.
/// Indexed `[phase][tap]` with phase ∈ 0..=15 and tap ∈ 0..=7. Phase 0
/// is the "integer sample" stub: never applied via the filter, it lives
/// here so the table is densely indexable.
const LUMA_FILTER_TABLE25: [[i32; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],      // 0  — full sample (caller short-circuits)
    [0, 0, 0, 0, 0, 0, 0, 0],       // 1  — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 2  — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 3  — na
    [0, 1, -5, 52, 20, -5, 1, 0],   // 4
    [0, 0, 0, 0, 0, 0, 0, 0],       // 5  — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 6  — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 7  — na
    [0, 2, -10, 40, 40, -10, 2, 0], // 8
    [0, 0, 0, 0, 0, 0, 0, 0],       // 9  — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 10 — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 11 — na
    [0, 1, -5, 20, 52, -5, 1, 0],   // 12
    [0, 0, 0, 0, 0, 0, 0, 0],       // 13 — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 14 — na
    [0, 0, 0, 0, 0, 0, 0, 0],       // 15 — na
];

/// Table 27 (ISO/IEC 23094-1:2020 §8.5.4.3.3). Baseline chroma 4-tap
/// interpolation filter coefficients at 1/32-pel resolution. Only phases
/// 4, 8, 12, 16, 20, 24, 28 are non-`na` — Baseline restricts chroma MV
/// fractions to the 1/8-pel grid (which doubles into the 1/32 grid via
/// the `* 2` chroma MV derivation in §8.5.2.6).
const CHROMA_FILTER_TABLE27: [[i32; 4]; 32] = [
    [0, 64, 0, 0],    // 0 (integer; caller short-circuits)
    [0, 0, 0, 0],     // 1
    [0, 0, 0, 0],     // 2
    [0, 0, 0, 0],     // 3
    [-2, 58, 10, -2], // 4
    [0, 0, 0, 0],     // 5
    [0, 0, 0, 0],     // 6
    [0, 0, 0, 0],     // 7
    [-4, 52, 20, -4], // 8
    [0, 0, 0, 0],     // 9
    [0, 0, 0, 0],     // 10
    [0, 0, 0, 0],     // 11
    [-6, 46, 30, -6], // 12
    [0, 0, 0, 0],     // 13
    [0, 0, 0, 0],     // 14
    [0, 0, 0, 0],     // 15
    [-8, 40, 40, -8], // 16
    [0, 0, 0, 0],     // 17
    [0, 0, 0, 0],     // 18
    [0, 0, 0, 0],     // 19
    [-4, 28, 46, -6], // 20
    [0, 0, 0, 0],     // 21
    [0, 0, 0, 0],     // 22
    [0, 0, 0, 0],     // 23
    [-4, 20, 52, -4], // 24
    [0, 0, 0, 0],     // 25
    [0, 0, 0, 0],     // 26
    [0, 0, 0, 0],     // 27
    [-2, 10, 58, -2], // 28
    [0, 0, 0, 0],     // 29
    [0, 0, 0, 0],     // 30
    [0, 0, 0, 0],     // 31
];

/// Table 24 (ISO/IEC 23094-1:2020 §8.5.4.3.2). Main-profile
/// (`sps_admvp_flag == 1`) luma 8-tap interpolation filter coefficients —
/// every 1/16-pel phase is defined, as the §8.5.3.7 affine subblock
/// motion field (and the §8.5.5 DMVR refinement) address the full
/// 1/16-pel grid. Indexed `[phase][tap]`; phase 0 is the integer-sample
/// stub (caller short-circuits).
const LUMA_FILTER_TABLE24: [[i32; 8]; 16] = [
    [0, 0, 0, 64, 0, 0, 0, 0],        // 0 — full sample
    [0, 1, -3, 63, 4, -2, 1, 0],      // 1
    [-1, 2, -5, 62, 8, -3, 1, 0],     // 2
    [-1, 3, -8, 60, 13, -4, 1, 0],    // 3
    [-1, 4, -10, 58, 17, -5, 1, 0],   // 4
    [-1, 4, -11, 52, 26, -8, 3, -1],  // 5
    [-1, 3, -9, 47, 31, -10, 4, -1],  // 6
    [-1, 4, -11, 45, 34, -10, 4, -1], // 7
    [-1, 4, -11, 40, 40, -11, 4, -1], // 8
    [-1, 4, -10, 34, 45, -11, 4, -1], // 9
    [-1, 4, -10, 31, 47, -9, 3, -1],  // 10
    [-1, 3, -8, 26, 52, -11, 4, -1],  // 11
    [0, 1, -5, 17, 58, -10, 4, -1],   // 12
    [0, 1, -4, 13, 60, -8, 3, -1],    // 13
    [0, 1, -3, 8, 62, -5, 2, -1],     // 14
    [0, 1, -2, 4, 63, -3, 1, 0],      // 15
];

/// Table 26 (ISO/IEC 23094-1:2020 §8.5.4.3.3). Main-profile
/// (`sps_admvp_flag == 1`) chroma 4-tap interpolation filter
/// coefficients — every 1/32-pel phase is defined.
const CHROMA_FILTER_TABLE26: [[i32; 4]; 32] = [
    [0, 64, 0, 0],    // 0 — full sample
    [-1, 63, 2, 0],   // 1
    [-2, 62, 4, 0],   // 2
    [-2, 60, 7, -1],  // 3
    [-2, 58, 10, -2], // 4
    [-3, 57, 12, -2], // 5
    [-4, 56, 14, -2], // 6
    [-4, 55, 15, -2], // 7
    [-4, 54, 16, -2], // 8
    [-5, 53, 18, -2], // 9
    [-6, 52, 20, -2], // 10
    [-6, 49, 24, -3], // 11
    [-6, 46, 28, -4], // 12
    [-5, 44, 29, -4], // 13
    [-4, 42, 30, -4], // 14
    [-4, 39, 33, -4], // 15
    [-4, 36, 36, -4], // 16
    [-4, 33, 39, -4], // 17
    [-4, 30, 42, -4], // 18
    [-4, 29, 44, -5], // 19
    [-4, 28, 46, -6], // 20
    [-3, 24, 49, -6], // 21
    [-2, 20, 52, -6], // 22
    [-2, 18, 53, -5], // 23
    [-2, 16, 54, -4], // 24
    [-2, 15, 55, -4], // 25
    [-2, 14, 56, -4], // 26
    [-2, 12, 57, -3], // 27
    [-2, 10, 58, -2], // 28
    [-1, 7, 60, -2],  // 29
    [0, 4, 62, -2],   // 30
    [0, 2, 63, -1],   // 31
];

fn baseline_luma_phase_supported(phase: u32) -> bool {
    matches!(phase, 0 | 4 | 8 | 12)
}

fn baseline_chroma_phase_supported(phase: u32) -> bool {
    matches!(phase, 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28)
}

/// Sample one luma reference position, clipping coordinates to the
/// reference picture extent.  Implements the `Clip3(0, picW − 1, …)` step
/// in eq. 921/922. Shared with the §8.5.5.2 DMVR bilinear interpolation.
pub(crate) fn sample_luma_clipped(refp: RefPictureView<'_>, x: i32, y: i32) -> i32 {
    let xc = x.clamp(0, refp.width as i32 - 1) as usize;
    let yc = y.clamp(0, refp.height as i32 - 1) as usize;
    refp.y[yc * refp.y_stride + xc] as i32
}

fn sample_chroma_clipped(refp: RefPictureView<'_>, c_idx: u32, x: i32, y: i32) -> i32 {
    let plane = match c_idx {
        1 => refp.cb,
        2 => refp.cr,
        _ => panic!("evc inter: chroma sample requested with cIdx={c_idx}"),
    };
    let xc = x.clamp(0, refp.pic_w_c() as i32 - 1) as usize;
    let yc = y.clamp(0, refp.pic_h_c() as i32 - 1) as usize;
    plane[yc * refp.c_stride + xc] as i32
}

/// §8.5.4.3.1 — the top-left of the *bounding block for reference sample
/// padding*: `( xSbIntL, ySbIntL ) = ( xSb + ( mvLX[0] >> 4 ), ySb +
/// ( mvLX[1] >> 4 ) )` where `mvLX = refMvLX − mvOffset` is the
/// **unrefined** motion vector. When threaded into the §8.5.4.3.2 /
/// §8.5.4.3.3 sample interpolation, every reference fetch is clamped
/// into the padded window around this anchor (eqs. 923/924 luma:
/// `[xSbIntL − 3, xSbIntL + sbWidth + 3]`; eqs. 932/933 chroma:
/// `[xSbIntC − 1, xSbIntC + sbWidth + 1]`) — so a DMVR-refined MV never
/// requires reference samples beyond the window the original MV
/// established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadAnchor {
    /// `xSbIntL` (luma) or `xSbIntC` (chroma), in the plane's own
    /// full-sample units.
    pub x_sb_int: i32,
    /// `ySbIntL` / `ySbIntC`.
    pub y_sb_int: i32,
}

/// Interpolate a luma block of size `sb_width × sb_height` at the given
/// `mv` (1/16-pel) anchored at (`x_sb`, `y_sb`) in the current picture
/// coordinate space. Outputs are i32 samples, clipped to `[0, 2^bd-1]`.
///
/// Implements the §8.5.4.3.2 luma interpolation with the Baseline
/// Table 25 filters (`sps_admvp_flag == 0`). Non-Baseline phases (the
/// `na` rows of Table 25) return `Error::Unsupported`.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_block(
    refp: RefPictureView<'_>,
    x_sb: i32,
    y_sb: i32,
    mv: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    out: &mut [i32],
) -> Result<()> {
    interpolate_luma_block_with(
        refp,
        x_sb,
        y_sb,
        mv,
        sb_width,
        sb_height,
        bit_depth,
        &LUMA_FILTER_TABLE25,
        true,
        None,
        out,
    )
}

/// §8.5.4.3.2 luma interpolation with the Main-profile Table 24 filters
/// (`sps_admvp_flag == 1`) — every 1/16-pel phase is legal, as the
/// affine subblock field addresses the full grid.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_block_main(
    refp: RefPictureView<'_>,
    x_sb: i32,
    y_sb: i32,
    mv: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    out: &mut [i32],
) -> Result<()> {
    interpolate_luma_block_with(
        refp,
        x_sb,
        y_sb,
        mv,
        sb_width,
        sb_height,
        bit_depth,
        &LUMA_FILTER_TABLE24,
        false,
        None,
        out,
    )
}

/// §8.5.4.3.2 Main-profile luma interpolation with the eqs.-923/924
/// subblock padding clamp: every fetched full-sample position is first
/// clipped to the picture (eqs. 921/922) then into
/// `[xSbIntL − 3, xSbIntL + sbWidth + 3] × [ySbIntL − 3, ySbIntL +
/// sbHeight + 3]` around the caller-supplied [`PadAnchor`]. Used by the
/// DMVR-refined motion compensation, whose refined MV must not fetch
/// beyond the padding window of the original (unrefined) MV.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma_block_main_padded(
    refp: RefPictureView<'_>,
    x_sb: i32,
    y_sb: i32,
    mv: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    pad: PadAnchor,
    out: &mut [i32],
) -> Result<()> {
    interpolate_luma_block_with(
        refp,
        x_sb,
        y_sb,
        mv,
        sb_width,
        sb_height,
        bit_depth,
        &LUMA_FILTER_TABLE24,
        false,
        Some(pad),
        out,
    )
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn interpolate_luma_block_with(
    refp: RefPictureView<'_>,
    x_sb: i32,
    y_sb: i32,
    mv: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    table: &[[i32; 8]; 16],
    baseline_phase_check: bool,
    pad: Option<PadAnchor>,
    out: &mut [i32],
) -> Result<()> {
    if out.len() != sb_width * sb_height {
        return Err(Error::invalid("evc inter: luma out buffer size mismatch"));
    }
    // §8.5.4.3.1 eq. 913–916: integer sample is mv >> 4, fractional is mv & 15.
    let int_x = mv.x >> 4;
    let int_y = mv.y >> 4;
    let frac_x = (mv.x & 15) as u32;
    let frac_y = (mv.y & 15) as u32;
    if baseline_phase_check
        && (!baseline_luma_phase_supported(frac_x) || !baseline_luma_phase_supported(frac_y))
    {
        return Err(Error::unsupported(format!(
            "evc inter: round-4 luma phase ({frac_x},{frac_y}) outside Baseline 1/4-pel grid"
        )));
    }
    // eqs. 921/922 then 923/924 — the picture clip first, then (when a
    // padding anchor is threaded) the subblock bounding-window clip. The
    // final fetch re-clips to the picture for memory safety; on
    // conforming streams the padded window never leaves the picture once
    // the eq.-921/922 clip has run, so the re-clip is the identity.
    let pic_w = refp.width as i32;
    let pic_h = refp.height as i32;
    let cl_x = |v: i32| -> i32 {
        let c = v.clamp(0, pic_w - 1);
        match pad {
            Some(p) => c.clamp(p.x_sb_int - 3, p.x_sb_int + sb_width as i32 + 3),
            None => c,
        }
    };
    let cl_y = |v: i32| -> i32 {
        let c = v.clamp(0, pic_h - 1);
        match pad {
            Some(p) => c.clamp(p.y_sb_int - 3, p.y_sb_int + sb_height as i32 + 3),
            None => c,
        }
    };
    let max_val = (1i32 << bit_depth) - 1;
    let shift1 = (4u32).min(bit_depth.saturating_sub(8));
    let shift2 = (8u32).max(20u32.saturating_sub(bit_depth));
    let offset2: i32 = 1 << (shift2 - 1);
    if frac_x == 0 && frac_y == 0 {
        // Integer sample copy (eq. 925).
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let s = sample_luma_clipped(
                    refp,
                    cl_x(x_sb + int_x + xl as i32),
                    cl_y(y_sb + int_y + yl as i32),
                );
                out[yl * sb_width + xl] = s;
            }
        }
        return Ok(());
    }
    let fx = table[frac_x as usize];
    let fy = table[frac_y as usize];
    if frac_y == 0 {
        // Horizontal-only filter (eq. 926).
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let bx = x_sb + int_x + xl as i32;
                let by = y_sb + int_y + yl as i32;
                let mut acc: i32 = 0;
                for i in 0..8 {
                    let s = sample_luma_clipped(refp, cl_x(bx + i as i32 - 3), cl_y(by));
                    acc += fx[i] * s;
                }
                out[yl * sb_width + xl] = (acc >> 6).clamp(0, max_val);
            }
        }
        return Ok(());
    }
    if frac_x == 0 {
        // Vertical-only filter (eq. 927).
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let bx = x_sb + int_x + xl as i32;
                let by = y_sb + int_y + yl as i32;
                let mut acc: i32 = 0;
                for i in 0..8 {
                    let s = sample_luma_clipped(refp, cl_x(bx), cl_y(by + i as i32 - 3));
                    acc += fy[i] * s;
                }
                out[yl * sb_width + xl] = (acc >> 6).clamp(0, max_val);
            }
        }
        return Ok(());
    }
    // Separable horizontal-then-vertical (eq. 928–929). We compute one
    // (sb_width)x(sb_height + 7) intermediate buffer in 32-bit precision
    // then apply the vertical filter.
    let temp_h = sb_height + 7;
    let mut temp = vec![0i32; sb_width * temp_h];
    for yt in 0..temp_h {
        for xl in 0..sb_width {
            let bx = x_sb + int_x + xl as i32;
            let by = y_sb + int_y + yt as i32 - 3;
            let mut acc: i32 = 0;
            for i in 0..8 {
                let s = sample_luma_clipped(refp, cl_x(bx + i as i32 - 3), cl_y(by));
                acc += fx[i] * s;
            }
            // shift1 == 0 for 8-bit per spec.
            temp[yt * sb_width + xl] = acc >> shift1;
        }
    }
    for yl in 0..sb_height {
        for xl in 0..sb_width {
            let mut acc: i32 = 0;
            for i in 0..8 {
                acc += fy[i] * temp[(yl + i) * sb_width + xl];
            }
            out[yl * sb_width + xl] = ((acc + offset2) >> shift2).clamp(0, max_val);
        }
    }
    Ok(())
}

/// Interpolate a chroma block (`c_idx ∈ {1, 2}`) at the given chroma
/// `mvC` in 1/32-chroma-pel units, anchored at chroma coordinates
/// (`x_sb_c`, `y_sb_c`). Implements §8.5.4.3.3 with the Baseline
/// Table 27 filters (`sps_admvp_flag == 0`).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_block(
    refp: RefPictureView<'_>,
    c_idx: u32,
    x_sb_c: i32,
    y_sb_c: i32,
    mv_c: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    out: &mut [i32],
) -> Result<()> {
    interpolate_chroma_block_with(
        refp,
        c_idx,
        x_sb_c,
        y_sb_c,
        mv_c,
        sb_width,
        sb_height,
        bit_depth,
        &CHROMA_FILTER_TABLE27,
        true,
        None,
        out,
    )
}

/// §8.5.4.3.3 chroma interpolation with the Main-profile Table 26
/// filters (`sps_admvp_flag == 1`) — every 1/32-pel phase is legal.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_block_main(
    refp: RefPictureView<'_>,
    c_idx: u32,
    x_sb_c: i32,
    y_sb_c: i32,
    mv_c: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    out: &mut [i32],
) -> Result<()> {
    interpolate_chroma_block_with(
        refp,
        c_idx,
        x_sb_c,
        y_sb_c,
        mv_c,
        sb_width,
        sb_height,
        bit_depth,
        &CHROMA_FILTER_TABLE26,
        false,
        None,
        out,
    )
}

/// §8.5.4.3.3 Main-profile chroma interpolation with the eqs.-932/933
/// subblock padding clamp around the caller-supplied [`PadAnchor`]
/// (`[xSbIntC − 1, xSbIntC + sbWidth + 1]` per axis). The anchor is in
/// chroma full-sample units (`xSbIntC = xSb / SubWidthC + ( mvLX[0] >> 5 )`
/// per §8.5.4.3.1, with `mvLX` the unrefined MV).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma_block_main_padded(
    refp: RefPictureView<'_>,
    c_idx: u32,
    x_sb_c: i32,
    y_sb_c: i32,
    mv_c: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    pad: PadAnchor,
    out: &mut [i32],
) -> Result<()> {
    interpolate_chroma_block_with(
        refp,
        c_idx,
        x_sb_c,
        y_sb_c,
        mv_c,
        sb_width,
        sb_height,
        bit_depth,
        &CHROMA_FILTER_TABLE26,
        false,
        Some(pad),
        out,
    )
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn interpolate_chroma_block_with(
    refp: RefPictureView<'_>,
    c_idx: u32,
    x_sb_c: i32,
    y_sb_c: i32,
    mv_c: MotionVector,
    sb_width: usize,
    sb_height: usize,
    bit_depth: u32,
    table: &[[i32; 4]; 32],
    baseline_phase_check: bool,
    pad: Option<PadAnchor>,
    out: &mut [i32],
) -> Result<()> {
    if out.len() != sb_width * sb_height {
        return Err(Error::invalid("evc inter: chroma out buffer size mismatch"));
    }
    let int_x = mv_c.x >> 5;
    let int_y = mv_c.y >> 5;
    let frac_x = (mv_c.x & 31) as u32;
    let frac_y = (mv_c.y & 31) as u32;
    if baseline_phase_check
        && (!baseline_chroma_phase_supported(frac_x) || !baseline_chroma_phase_supported(frac_y))
    {
        return Err(Error::unsupported(format!(
            "evc inter: round-4 chroma phase ({frac_x},{frac_y}) outside Baseline 1/8-pel grid"
        )));
    }
    // eqs. 930/931 then 932/933 — picture clip, then the subblock
    // padding-window clip when a [`PadAnchor`] is threaded (the chroma
    // window is ±1 around the anchored subblock).
    let pic_w_c = refp.pic_w_c() as i32;
    let pic_h_c = refp.pic_h_c() as i32;
    let cl_x = |v: i32| -> i32 {
        let c = v.clamp(0, pic_w_c - 1);
        match pad {
            Some(p) => c.clamp(p.x_sb_int - 1, p.x_sb_int + sb_width as i32 + 1),
            None => c,
        }
    };
    let cl_y = |v: i32| -> i32 {
        let c = v.clamp(0, pic_h_c - 1);
        match pad {
            Some(p) => c.clamp(p.y_sb_int - 1, p.y_sb_int + sb_height as i32 + 1),
            None => c,
        }
    };
    let max_val = (1i32 << bit_depth) - 1;
    let shift1 = (4u32).min(bit_depth.saturating_sub(8));
    let shift2 = (8u32).max(20u32.saturating_sub(bit_depth));
    let offset2: i32 = 1 << (shift2 - 1);
    if frac_x == 0 && frac_y == 0 {
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let s = sample_chroma_clipped(
                    refp,
                    c_idx,
                    cl_x(x_sb_c + int_x + xl as i32),
                    cl_y(y_sb_c + int_y + yl as i32),
                );
                out[yl * sb_width + xl] = s;
            }
        }
        return Ok(());
    }
    let fx = table[frac_x as usize];
    let fy = table[frac_y as usize];
    if frac_y == 0 {
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let bx = x_sb_c + int_x + xl as i32;
                let by = y_sb_c + int_y + yl as i32;
                let mut acc: i32 = 0;
                for i in 0..4 {
                    let s = sample_chroma_clipped(refp, c_idx, cl_x(bx + i as i32 - 1), cl_y(by));
                    acc += fx[i] * s;
                }
                out[yl * sb_width + xl] = (acc >> 6).clamp(0, max_val);
            }
        }
        return Ok(());
    }
    if frac_x == 0 {
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let bx = x_sb_c + int_x + xl as i32;
                let by = y_sb_c + int_y + yl as i32;
                let mut acc: i32 = 0;
                for i in 0..4 {
                    let s = sample_chroma_clipped(refp, c_idx, cl_x(bx), cl_y(by + i as i32 - 1));
                    acc += fy[i] * s;
                }
                out[yl * sb_width + xl] = (acc >> 6).clamp(0, max_val);
            }
        }
        return Ok(());
    }
    let temp_h = sb_height + 3;
    let mut temp = vec![0i32; sb_width * temp_h];
    for yt in 0..temp_h {
        for xl in 0..sb_width {
            let bx = x_sb_c + int_x + xl as i32;
            let by = y_sb_c + int_y + yt as i32 - 1;
            let mut acc: i32 = 0;
            for i in 0..4 {
                let s = sample_chroma_clipped(refp, c_idx, cl_x(bx + i as i32 - 1), cl_y(by));
                acc += fx[i] * s;
            }
            temp[yt * sb_width + xl] = acc >> shift1;
        }
    }
    for yl in 0..sb_height {
        for xl in 0..sb_width {
            let mut acc: i32 = 0;
            for i in 0..4 {
                acc += fy[i] * temp[(yl + i) * sb_width + xl];
            }
            out[yl * sb_width + xl] = ((acc + offset2) >> shift2).clamp(0, max_val);
        }
    }
    Ok(())
}

/// Default-weighted bi-prediction average (eq. 988).
pub fn average_bipred(pred_l0: &[i32], pred_l1: &[i32], out: &mut [i32]) {
    debug_assert_eq!(pred_l0.len(), pred_l1.len());
    debug_assert_eq!(pred_l0.len(), out.len());
    for ((a, b), o) in pred_l0.iter().zip(pred_l1.iter()).zip(out.iter_mut()) {
        *o = (*a + *b + 1) >> 1;
    }
}

/// Derive the chroma motion vector from the luma MV per eq. 676/677. For
/// 4:2:0 (`chroma_format_idc = 1`), `mvCLX = mvLX * 2 / 2 = mvLX` — the
/// luma 1/16-pel components are reused as chroma 1/32-pel components
/// (since 1 luma sample = 2 chroma samples in 4:2:0, and 1/16 luma-pel
/// equals 1/32 chroma-pel).
pub fn derive_chroma_mv(mv_luma_sixteenth: MotionVector, chroma_format_idc: u32) -> MotionVector {
    let (sub_w, sub_h) = match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1), // monochrome: caller never uses this
    };
    MotionVector {
        x: mv_luma_sixteenth.x * 2 / sub_w,
        y: mv_luma_sixteenth.y * 2 / sub_h,
    }
}

// =====================================================================
// AMVP candidate list construction (§8.5.2.4.3 — `sps_admvp_flag = 0`).
// =====================================================================

/// AMVP candidate slot in the Baseline mvpList (§8.5.2.4.3). The default
/// `(1, 1)` substitution applies when the spatial neighbour is
/// unavailable (eq. 648/649, 651/652, 654/655).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AmvpCandidate(pub MotionVector);

/// Build the four-entry AMVP `mvpList` for a CU at `(x_cb, y_cb)` of
/// size `n_cb_w × n_cb_h`. Spatial neighbours come from a caller-
/// supplied lookup `mv_at(xN, yN) → Option<MotionVector>` that returns
/// `Some` iff the neighbour is in-picture and was coded as inter with
/// the same reference index. `temporal_zero` parametrises mvpList[3];
/// round-4 simplifies it to `MotionVector::default()` (zero MV).
pub fn build_amvp_list_baseline<F>(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    _n_cb_h: i32,
    mut mv_at: F,
    temporal_zero: MotionVector,
) -> [AmvpCandidate; 4]
where
    F: FnMut(i32, i32) -> Option<MotionVector>,
{
    let unavailable = MotionVector::quarter_pel(1, 1);
    let nb0 = mv_at(x_cb - 1, y_cb).unwrap_or(unavailable);
    let nb1 = mv_at(x_cb, y_cb - 1).unwrap_or(unavailable);
    let nb2 = mv_at(x_cb + n_cb_w, y_cb - 1).unwrap_or(unavailable);
    [
        AmvpCandidate(nb0),
        AmvpCandidate(nb1),
        AmvpCandidate(nb2),
        AmvpCandidate(temporal_zero),
    ]
}

// ===========================================================================
// §8.5 AMVR — Adaptive Motion Vector Resolution helpers
// ===========================================================================
//
// AMVR is a Main-profile tool (gated by `sps_amvr_flag = 1`) which lets the
// encoder signal a coarser resolution for both the per-CU motion-vector
// difference (`MvdLX`, eq. 145) and the AMVP predictor (`mvpLX`,
// eq. 645/646). The signal is `amvr_idx[ x0 ][ y0 ]`, a TR-binarised
// syntax element with `cMax = 4` (§9.3.3 binarization table), so the
// valid range is `0..=4`:
//
// * `amvr_idx == 0` — 1/4-pel resolution (Baseline, no shifting).
// * `amvr_idx == 1` — 1/2-pel.
// * `amvr_idx == 2` — integer-pel.
// * `amvr_idx == 3` — 2-pel.
// * `amvr_idx == 4` — 4-pel.
//
// The Main-profile decode path slots these helpers in after the standard
// AMVP MVD reconstruction (eq. 144) and the MVP candidate derivation
// (eq. 619-644). Baseline streams set `sps_amvr_flag = 0` and skip both —
// the helpers below remain dead code on that path, so adding them does
// not perturb the existing Baseline pixel pipeline.
//
// The §9.3.4 ctxIdxInc for `amvr_idx`'s TR-binarised bins is just the
// bin position (Table 67 carries 4 trained states per `initType`, both
// initTypes spanning ctxIdx ranges `0..3` and `4..7` — there is no
// neighbour-derived `ctxInc` term, so bin `k` simply maps to ctx `k`).

/// Maximum legal value of `amvr_idx` (TR cMax). See §9.3.3 binarization
/// table.
pub const AMVR_IDX_MAX: u32 = 4;

/// Eq. 145 — `MvdLX[ x0 ][ y0 ][ compIdx ] = MvdLX[ x0 ][ y0 ][ compIdx ]
/// << amvr_idx[ x0 ][ y0 ]`. Applies the AMVR resolution shift to one
/// component of the motion-vector difference. The caller is responsible
/// for gating on `sps_amvr_flag == 1`; with `sps_amvr_flag == 0` the
/// shift would never be invoked (Baseline forces `amvr_idx == 0` anyway,
/// so this is a left-shift by 0 ≡ identity).
///
/// `amvr_idx` is rejected with [`Error::Unsupported`] outside `0..=4`
/// (the TR cMax range).
pub fn amvr_apply_to_mvd(mvd_component: i32, amvr_idx: u32) -> Result<i32> {
    if amvr_idx > AMVR_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: amvr_idx = {amvr_idx} exceeds TR cMax = {AMVR_IDX_MAX} (§8.5 eq. 145)"
        )));
    }
    Ok(mvd_component << amvr_idx)
}

/// Vector form of [`amvr_apply_to_mvd`] — applies eq. 145 to both
/// components of an MVD.
pub fn amvr_apply_to_mvd_vector(mvd: MotionVector, amvr_idx: u32) -> Result<MotionVector> {
    Ok(MotionVector {
        x: amvr_apply_to_mvd(mvd.x, amvr_idx)?,
        y: amvr_apply_to_mvd(mvd.y, amvr_idx)?,
    })
}

/// Eq. 645/646 — magnitude-preserving AMVR round of one component of the
/// motion-vector predictor `mvpLX[ k ]`. The spec writes the operation
/// branchless on the sign of the predictor:
///
/// ```text
/// mvpLX[ k ] = mvpLX[ k ] >= 0 ?
///   ( ( mvpLX[ k ] + ( 1 << ( amvr_idx - 1 ) ) ) >> amvr_idx ) << amvr_idx :
///   −( ( ( −mvpLX[ k ] + ( 1 << ( amvr_idx - 1 ) ) ) >> amvr_idx ) << amvr_idx )
/// ```
///
/// This rounds magnitude-towards-zero with a "round-half-away-from-zero"
/// tie break — i.e. `+2` at `amvr_idx == 2` rounds to `+4`, `−2` rounds
/// to `−4`. The sign-symmetric branching distinguishes it from
/// [`round_motion_vector`] (§8.5.3.10 eq. 907-909), which is a
/// round-toward-negative-infinity flavour for affine MV derivation.
///
/// The caller is responsible for gating on `sps_amvr_flag == 1 &&
/// amvr_idx != 0`; with `amvr_idx == 0` this returns the input
/// unchanged. `amvr_idx` outside `0..=4` is rejected.
pub fn amvr_round_mvp(mvp_component: i32, amvr_idx: u32) -> Result<i32> {
    if amvr_idx > AMVR_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: amvr_idx = {amvr_idx} exceeds TR cMax = {AMVR_IDX_MAX} (§8.5 eq. 645/646)"
        )));
    }
    if amvr_idx == 0 {
        return Ok(mvp_component);
    }
    let shift = amvr_idx;
    let half: i32 = 1i32 << (shift - 1);
    Ok(if mvp_component >= 0 {
        ((mvp_component + half) >> shift) << shift
    } else {
        // Spec: −((((−mvp) + half) >> shift) << shift)
        let m = -mvp_component;
        -(((m + half) >> shift) << shift)
    })
}

/// Vector form of [`amvr_round_mvp`].
pub fn amvr_round_mvp_vector(mvp: MotionVector, amvr_idx: u32) -> Result<MotionVector> {
    Ok(MotionVector {
        x: amvr_round_mvp(mvp.x, amvr_idx)?,
        y: amvr_round_mvp(mvp.y, amvr_idx)?,
    })
}

/// §9.3.4 ctxIdxInc for the `amvr_idx` TR-binarised bins. Table 67's
/// 4-per-initType layout (positions `0..3` / `4..7`) means each bin maps
/// 1-to-1 to a ctxIdx within the initType range; the assignment is
/// purely positional (no neighbour-derived `ctxInc` term — `amvr_idx` is
/// **not** one of the four entries of Table 96).
///
/// * `bin_idx` — 0-based bin position within the TR string (0..=3, since
///   `cMax = 4` means at most 4 prefix bins).
///
/// Returns [`Error::Unsupported`] for `bin_idx >= 4`.
pub fn amvr_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= AMVR_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: amvr_idx bin_idx = {bin_idx} exceeds TR cMax = {AMVR_IDX_MAX} prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

// ===========================================================================
// §9.3.3 / §9.3.4.2 — merge_idx (regular merge candidate selector)
// ===========================================================================
//
// `merge_idx[ x0 ][ y0 ]` selects the regular merging candidate on the
// `sps_admvp_flag == 1` non-affine / non-MMVD merge path (spec lines
// 2830 / 2908) and on the §7.3.8.4 cu_skip non-affine fall-through.
//
// §9.3.3 binarization: TR, `cMax = ( nCbW * nCbH <= 32 ) ? 3 : 5`,
// `cRiceParam = 0`. The prefix-bin ctxInc (§9.3.4.2 table, `merge_idx`
// row) is purely positional: bin `k` → ctxInc `k` for k ∈ {0,1,2,3,4}
// (Table 49 carries 5 trained states per `initType`). There is no
// neighbour-derived ctxInc term.

/// Hard cap on `merge_idx` prefix bins — the larger of the two §9.3.3
/// `cMax` branches (`cMax = 5` when `nCbW * nCbH > 32`).
pub const MERGE_IDX_MAX: u32 = 5;

/// §9.3.3 — `merge_idx` TR `cMax` as a function of the coding block area:
/// `( nCbW * nCbH <= 32 ) ? 3 : 5`.
pub fn merge_idx_c_max(n_cb_w: u32, n_cb_h: u32) -> u32 {
    if n_cb_w.saturating_mul(n_cb_h) <= 32 {
        3
    } else {
        5
    }
}

/// §9.3.4.2 — positional ctxInc for `merge_idx` prefix bin `bin_idx`
/// (0-based). Bin `k` maps 1-to-1 to ctxIdx `k` within the `initType`
/// range; valid for `bin_idx ∈ 0..=4` (the `cMax = 5` branch needs at
/// most 5 prefix bins). Returns [`Error::Unsupported`] for `bin_idx >= 5`.
pub fn merge_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= MERGE_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: merge_idx bin_idx = {bin_idx} exceeds TR cMax = {MERGE_IDX_MAX} prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

// ===========================================================================
// §9.3.3 / §9.3.4.2 — inter_pred_idc + bi_pred_idx (explicit-AMVP path)
// ===========================================================================
//
// On the `sps_admvp_flag == 1` explicit-AMVP path (spec lines 2912-3025)
// the B-slice prediction direction `inter_pred_idc[ x0 ][ y0 ]` is read,
// and when it resolves to PRED_BI the `bi_pred_idx[ x0 ][ y0 ]` element
// selects which of the two lists carry an MVD (Table 71 semantics:
// 0 = both lists present, 1 = list-1 MVD absent, 2 = list-0 MVD absent).
//
// §9.3.3 binarizations:
//   * `inter_pred_idc` — TR, `cMax = ( !sps_admvp_flag || nCbW + nCbH >
//     12 ) ? 2 : 1`, cRiceParam 0. The §9.3.4.2 ctxInc is positional
//     0,1 (Table 69).
//   * `bi_pred_idx` — TR, `cMax = 2`, cRiceParam 0. The §9.3.4.2 ctxInc
//     is positional 0,1 (Table 71).

/// PRED_L0 (uni-directional, list 0). Table 8 mapping.
pub const PRED_L0: u32 = 0;
/// PRED_L1 (uni-directional, list 1). Table 8 mapping.
pub const PRED_L1: u32 = 1;
/// PRED_BI (bi-directional). Table 8 mapping.
pub const PRED_BI: u32 = 2;

/// §9.3.3 — `inter_pred_idc` TR `cMax` on the §7.3.8.4 inter path:
/// `( !sps_admvp_flag || nCbW + nCbH > 12 ) ? 2 : 1`. The `cMax = 1`
/// branch (small admvp block) caps the directionality to uni-pred.
pub fn inter_pred_idc_c_max(sps_admvp_flag: bool, n_cb_w: u32, n_cb_h: u32) -> u32 {
    if !sps_admvp_flag || n_cb_w + n_cb_h > 12 {
        2
    } else {
        1
    }
}

/// §9.3.4.2 — positional ctxInc for `inter_pred_idc` prefix bin
/// (0,1 over Table 69). Valid for `bin_idx ∈ {0, 1}`.
pub fn inter_pred_idc_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= 2 {
        return Err(Error::unsupported(format!(
            "evc inter: inter_pred_idc bin_idx = {bin_idx} exceeds TR cMax = 2 prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

/// Maximum legal value of `bi_pred_idx` (TR cMax). §9.3.3 binarization.
pub const BI_PRED_IDX_MAX: u32 = 2;

/// §9.3.4.2 — positional ctxInc for `bi_pred_idx` prefix bin (0,1 over
/// Table 71). Valid for `bin_idx ∈ {0, 1}`.
pub fn bi_pred_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= BI_PRED_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: bi_pred_idx bin_idx = {bin_idx} exceeds TR cMax = {BI_PRED_IDX_MAX} prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

// ===========================================================================
// §7.4.7 / §9.3.3 / §9.3.4 — MMVD (Merge with Motion Vector Difference)
// distance / sign / offset derivation
// ===========================================================================
//
// MMVD is a Main-profile tool (gated by `sps_mmvd_flag = 1`) which lets the
// encoder modify a regular merge candidate by a small extra MV offset
// `MmvdOffset[ x0 ][ y0 ][ k ]` (k ∈ {0, 1}). The offset is derived from
// two enumerative syntax elements `mmvd_distance_idx[ x0 ][ y0 ]` and
// `mmvd_direction_idx[ x0 ][ y0 ]` via spec Tables 9 and 10:
//
//   * `mmvd_distance_idx` ∈ 0..=7 → `MmvdDistance ∈ { 1, 2, 4, 8, 16, 32,
//     64, 128 }` (i.e. `MmvdDistance = 1 << mmvd_distance_idx`). The
//     §9.3.3 binarization is TR with `cMax = 7, cRiceParam = 0`.
//   * `mmvd_direction_idx` ∈ 0..=3 → `(MmvdSign[0], MmvdSign[1])` ∈
//     `{ (+1, 0), (−1, 0), (0, +1), (0, −1) }` (Table 10). The §9.3.3
//     binarization is FL with `cMax = 3` (two bits).
//
// Eqs. 133 / 134 then combine the two:
//
//   `MmvdOffset[ x0 ][ y0 ][ k ] = MmvdDistance * MmvdSign[ k ]`
//
// for k ∈ { 0, 1 }. Note `MmvdOffset` is always axis-aligned (one of the
// two components is zero), since each of the four `mmvd_direction_idx`
// rows of Table 10 has at most one non-zero `MmvdSign` component.
//
// Round 218 ships the table + offset derivation as opt-in helpers, plus
// the §9.3.4 ctxIdxInc for the five MMVD syntax elements (`mmvd_flag`,
// `mmvd_group_idx`, `mmvd_merge_idx`, `mmvd_distance_idx`,
// `mmvd_direction_idx`). The Main-profile decode path threads
// `sps_mmvd_flag` + the parsed `mmvd_*_idx` values into them; Baseline
// streams (`sps_mmvd_flag = 0`) infer `mmvd_flag = 0` per §7.4.7 and the
// helpers stay dead code on that path.
//
// The downstream §8.5.2.3.9 derivation (eqs. 531-616, "Derivation process
// for MMVD motion vector") which adds the offset to a merge candidate
// while POC-scaling between L0 and L1 still needs the merge-candidate
// list + DiffPicOrderCnt + reference-list infrastructure that has not
// landed yet; it is the documented follow-up after round 218.

/// Maximum legal value of `mmvd_distance_idx` (TR cMax). §9.3.3
/// binarization table.
pub const MMVD_DISTANCE_IDX_MAX: u32 = 7;

/// Maximum legal value of `mmvd_direction_idx` (FL cMax). §9.3.3
/// binarization table.
pub const MMVD_DIRECTION_IDX_MAX: u32 = 3;

/// Maximum legal value of `mmvd_group_idx` (TR cMax). §9.3.3 binarization
/// table.
pub const MMVD_GROUP_IDX_MAX: u32 = 2;

/// Maximum legal value of `mmvd_merge_idx` (TR cMax). §9.3.3 binarization
/// table.
pub const MMVD_MERGE_IDX_MAX: u32 = 3;

/// Table 9 — derivation of `MmvdDistance[ x0 ][ y0 ]` from
/// `mmvd_distance_idx[ x0 ][ y0 ]`. The table is `1 << mmvd_distance_idx`
/// (i.e. `{ 1, 2, 4, 8, 16, 32, 64, 128 }` for indices 0..7), kept as
/// a transcribed lookup so a future spec amendment that breaks the
/// power-of-two pattern is a single edit.
const TABLE_9_MMVD_DISTANCE: [i32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Table 9 — return `MmvdDistance` for a parsed `mmvd_distance_idx`.
/// Out-of-range indices surface `Error::Unsupported`.
pub fn mmvd_distance(mmvd_distance_idx: u32) -> Result<i32> {
    if mmvd_distance_idx > MMVD_DISTANCE_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_distance_idx = {mmvd_distance_idx} exceeds TR cMax = {MMVD_DISTANCE_IDX_MAX} (Table 9 / §9.3.3)"
        )));
    }
    Ok(TABLE_9_MMVD_DISTANCE[mmvd_distance_idx as usize])
}

/// Table 10 — derivation of `(MmvdSign[ 0 ], MmvdSign[ 1 ])` from
/// `mmvd_direction_idx[ x0 ][ y0 ]`. Each row of Table 10 has exactly
/// one non-zero component (axis-aligned), and that component is ±1.
const TABLE_10_MMVD_SIGN: [(i32, i32); 4] = [
    (1, 0),  // mmvd_direction_idx = 0 → (+1,  0)
    (-1, 0), // mmvd_direction_idx = 1 → (−1,  0)
    (0, 1),  // mmvd_direction_idx = 2 → ( 0, +1)
    (0, -1), // mmvd_direction_idx = 3 → ( 0, −1)
];

/// Table 10 — return `(MmvdSign[ 0 ], MmvdSign[ 1 ])` for a parsed
/// `mmvd_direction_idx`. Out-of-range indices surface
/// `Error::Unsupported`.
pub fn mmvd_sign(mmvd_direction_idx: u32) -> Result<(i32, i32)> {
    if mmvd_direction_idx > MMVD_DIRECTION_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_direction_idx = {mmvd_direction_idx} exceeds FL cMax = {MMVD_DIRECTION_IDX_MAX} (Table 10 / §9.3.3)"
        )));
    }
    Ok(TABLE_10_MMVD_SIGN[mmvd_direction_idx as usize])
}

/// Eq. 133 + 134 — derive `MmvdOffset[ x0 ][ y0 ]` from the parsed
/// `mmvd_distance_idx[ x0 ][ y0 ]` and `mmvd_direction_idx[ x0 ][ y0 ]`.
/// Returns a [`MotionVector`] in the spec's raw integer-pel units; the
/// downstream §8.5.2.3.9 derivation consumes the same units when adding
/// the offset to the merge candidate's `mvL0` / `mvL1`.
///
/// Because Table 10's sign vector is axis-aligned at each row, the
/// returned offset always has exactly one non-zero component (or both
/// zero — but Table 10 has no all-zero row, so in practice always one
/// non-zero component, ranging in magnitude over `{ 1, 2, 4, 8, 16, 32,
/// 64, 128 }`).
pub fn mmvd_offset(mmvd_distance_idx: u32, mmvd_direction_idx: u32) -> Result<MotionVector> {
    let dist = mmvd_distance(mmvd_distance_idx)?;
    let (sx, sy) = mmvd_sign(mmvd_direction_idx)?;
    Ok(MotionVector {
        x: dist * sx, // eq. 133
        y: dist * sy, // eq. 134
    })
}

/// §9.3.4 ctxIdxInc for the `mmvd_flag` FL bin. `mmvd_flag` is **not**
/// in Table 96, so the assignment is purely positional. Table 50 carries
/// a single trained state per `initType` (initType-0 at ctxIdx 0,
/// initType-1 at ctxIdx 1; cMax = 1 ⇒ exactly one bin); the only valid
/// `bin_idx` is 0.
pub fn mmvd_flag_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx != 0 {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_flag bin_idx = {bin_idx} exceeds FL cMax = 1 single bin (§9.3.4)"
        )));
    }
    Ok(0)
}

/// §9.3.4 ctxIdxInc for the `mmvd_group_idx` TR bins (cMax = 2 ⇒ up to
/// two prefix bins). Not in Table 96 ⇒ purely positional. Table 51
/// carries 2 trained states per `initType` (ctxIdx 0..1 and 2..3).
pub fn mmvd_group_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= MMVD_GROUP_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_group_idx bin_idx = {bin_idx} exceeds TR cMax = {MMVD_GROUP_IDX_MAX} prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

/// §9.3.4 ctxIdxInc for the `mmvd_merge_idx` TR bins (cMax = 3 ⇒ up to
/// three prefix bins). Not in Table 96 ⇒ purely positional. Table 52
/// carries 3 trained states per `initType` (ctxIdx 0..2 and 3..5).
pub fn mmvd_merge_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= MMVD_MERGE_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_merge_idx bin_idx = {bin_idx} exceeds TR cMax = {MMVD_MERGE_IDX_MAX} prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

/// §9.3.4 ctxIdxInc for the `mmvd_distance_idx` TR bins (cMax = 7 ⇒ up
/// to seven prefix bins). Not in Table 96 ⇒ purely positional. Table 53
/// carries 7 trained states per `initType` (ctxIdx 0..6 and 7..13).
pub fn mmvd_distance_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    if bin_idx >= MMVD_DISTANCE_IDX_MAX {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_distance_idx bin_idx = {bin_idx} exceeds TR cMax = {MMVD_DISTANCE_IDX_MAX} prefix bins (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

/// §9.3.4 ctxIdxInc for the `mmvd_direction_idx` FL bins (cMax = 3 ⇒
/// exactly two bits). Not in Table 96 ⇒ purely positional. Table 54
/// carries 2 trained states per `initType` (ctxIdx 0..1 and 2..3).
pub fn mmvd_direction_idx_ctx_inc(bin_idx: u32) -> Result<usize> {
    // FL with cMax = 3 means a 2-bit code (`Ceil(Log2(cMax + 1)) = 2`),
    // so bins 0 and 1 are the only valid positions.
    if bin_idx >= 2 {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_direction_idx bin_idx = {bin_idx} exceeds FL cMax = {MMVD_DIRECTION_IDX_MAX} 2-bit code (§9.3.4)"
        )));
    }
    Ok(bin_idx as usize)
}

// ===========================================================================
// §8.5.2.3.9 — Bipred MMVD offset distribution (eqs. 591-616)
// ===========================================================================
//
// Round 223 lands the symmetric / asymmetric bipred branch of the §8.5.2.3.9
// "Derivation process for MMVD motion vector". Round 218 covered Table 9 +
// Table 10 + eqs. 133/134 (`MmvdOffset`); this round consumes that offset
// plus the POC distances of `(currPic, RefPicList0[refIdxL0])` and
// `(currPic, RefPicList1[refIdxL1])` to produce the per-list MV deltas
// `mMvdL0` and `mMvdL1`, and accumulates them into `mMvL0` / `mMvL1`.
//
// The bipred sub-process (eqs. 591-616) splits on the relative magnitudes
// of the two POC diffs:
//
// 1. `Abs(currPocDiffL0) == Abs(currPocDiffL1)` — symmetric case
//    (eqs. 593-596): both `mMvdLX = MmvdOffset` directly.
//
// 2. `Abs(currPocDiffL0) > Abs(currPocDiffL1)` — L1 closer
//    (eqs. 597-601): `mMvdL1 = MmvdOffset`, then `mMvdL0` is scaled from
//    `mMvdL1` by `distScaleFactor = (Abs(L1) << 5) / Abs(L0)` via the
//    round-half-up form `Clip3(-32768, 32767, (sf * mMvdL1[k] + 16) >> 5)`.
//
// 3. `Abs(currPocDiffL0) < Abs(currPocDiffL1)` — L0 closer
//    (eqs. 602-606): symmetric to case 2 with the roles swapped.
//
// After case 2 / case 3, if `currPocDiffL0 * currPocDiffL1 < 0` (the
// reference pictures sit on opposite sides of `currPic` in display order),
// eqs. 607-610 negate `mMvdL1` to flip its sign relative to `mMvdL0`. (The
// no-op identities `mMvdL0 = mMvdL0` at eqs. 607/608 are the spec's way of
// emphasising that only `mMvdL1` is touched.)
//
// The "Otherwise" branch at eqs. 611-612 covers the one-list-active case
// (`predFlagL0 ^ predFlagL1 == 1`): each active list gets `MmvdOffset`, the
// inactive list gets zero.
//
// Eqs. 613-616 close out the process by adding `mMvdLX` to the merge
// candidate's `mvLX[0][0]`, producing the final MMVD motion vectors.
//
// All arithmetic uses signed 32-bit intermediates and clips the per-list
// MVD components to signed 16-bit range. The spec's `Clip3(-32768, 32767,
// ...)` wrapping ensures the upstream `mMvL0 += mMvdL0` accumulation
// (eqs. 613-614) operates on already-clipped values; the addition then
// goes through the same `wrap16` modular semantics as eqs. 436/439.
//
// Spec typo: eq. 601 (page 170) reads
// `Clip3( −32768, 32767 ( distScaleFactor * mMvdL1[ 1 ] + 16 ) >> 5 )` —
// the comma between `32767` and the inner expression is missing in the
// typeset PDF. Context-determined: the form is identical to eq. 600 on
// the y component.
//
// Docs gap none. The §8.5.2.3.9 entry-point process (eqs. 531-590) covers
// the `mmvd_group_idx ∈ { 1, 2 }` ref-list reassignment branches plus a
// `slice_type == P` sub-branch (eq. 545-553, 574-582), both of which need
// `slice_type`, `NumRefIdxActive[]`, and the populated `RefPicList0/1`
// arrays — wiring that needs an §8.5.2.3.x merge-candidate-list pass to
// land first. This round confines itself to the bipred and one-list-active
// distribution + final accumulation, which is self-contained given the
// inputs.

/// §8.5.2.3.9 / eq. 588 (P-slice form), eq. 599, eq. 604 — POC-difference
/// scaling factor used by the bipred MMVD branch:
///
/// ```text
/// distScaleFactor = ( |pocDiffNum| << 5 ) / |pocDiffDen|
/// ```
///
/// The numerator and denominator forms in eqs. 599 / 604 explicitly take
/// `Abs(currPocDiffL?)` of both arguments, so this helper accepts already-
/// absolute values. The result is always non-negative.
///
/// `pocDiffDen` must be non-zero; the bipred branch only enters with both
/// L0 and L1 ref pictures active, and POC equality already routed the
/// caller into the symmetric eqs. 593-596 path.
pub fn mmvd_dist_scale_factor(abs_poc_num: i32, abs_poc_den: i32) -> Result<i32> {
    if abs_poc_den <= 0 {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_dist_scale_factor needs strictly-positive denominator, got {abs_poc_den} (§8.5.2.3.9 eq. 599/604)"
        )));
    }
    if abs_poc_num < 0 {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_dist_scale_factor needs non-negative numerator, got {abs_poc_num} (§8.5.2.3.9 eq. 599/604)"
        )));
    }
    Ok((abs_poc_num << 5) / abs_poc_den)
}

/// `Clip3(-32768, 32767, ...)` — eq. 600 / 601 / 605 / 606 component
/// clipper. The bipred MMVD path produces 32-bit intermediates (POC scale
/// factor × MV component + bias) which the spec then clamps back into
/// signed 16-bit range, matching the §6.5 storage of MV components.
fn clip_mvd_component(v: i32) -> i32 {
    v.clamp(-32768, 32767)
}

/// §8.5.2.3.9 — round-half-up POC scaling of a single MV component, the
/// form used by eqs. 600 / 601 / 605 / 606:
///
/// ```text
/// out = Clip3(-32768, 32767, (distScaleFactor * mvComponent + 16) >> 5)
/// ```
///
/// The `+16` bias plus `>> 5` is the spec's round-half-up reduction of a
/// 5-bit fractional scaling. Rust's arithmetic right-shift on signed
/// integers makes the semantics deterministic for both signs of the
/// product.
fn mmvd_scale_component(dist_scale_factor: i32, mv_component: i32) -> i32 {
    let raw = dist_scale_factor
        .wrapping_mul(mv_component)
        .wrapping_add(16)
        >> 5;
    clip_mvd_component(raw)
}

/// §8.5.2.3.9 / eqs. 591-616 — bipred + one-list-active MMVD offset
/// distribution and final motion-vector accumulation.
///
/// This is the tail of §8.5.2.3.9: it runs after the §7.4.7 / Table-9-10
/// derivation produced `MmvdOffset`, after merge-candidate selection
/// produced `mvL0` / `mvL1`, and after `mmvd_group_idx ∈ { 1, 2 }`
/// reassignment (if any) settled `refIdxLX` / `predFlagLX`.
///
/// Inputs:
///
/// * `mv_l0`, `mv_l1` — the merge candidate's per-list motion vectors,
///   already loaded into `mMvL0` / `mMvL1` by eqs. 531-532. Components
///   live in the same signed 16-bit range as the spec's `mvLX[0][0]`.
/// * `mmvd_offset` — the §7.4.7 / round-218 axis-aligned `MmvdOffset`
///   (eqs. 133-134). Magnitude is one of `{ 1, 2, 4, 8, 16, 32, 64, 128 }`
///   with exactly one non-zero component.
/// * `curr_poc_diff_l0` — `DiffPicOrderCnt(currPic, RefPicList0[refIdxL0])`
///   from eqs. 586 / 591. Signed: positive when the L0 reference precedes
///   `currPic` in display order.
/// * `curr_poc_diff_l1` — `DiffPicOrderCnt(currPic, RefPicList1[refIdxL1])`
///   from eqs. 587 / 592. Signed; same convention.
/// * `pred_flag_l0`, `pred_flag_l1` — `predFlagL{0,1}[0][0]` (the
///   `bool` form). Both true ⇒ bipred branch (eqs. 591-610); exactly
///   one true ⇒ "Otherwise" branch (eqs. 611-612); both false is a
///   caller bug — the §8.5.2.3.9 entry never enters the eqs. 591-616
///   region with both list flags zero.
///
/// Output: the post-accumulation `(mMvL0, mMvL1)` (eqs. 613-616). When
/// `pred_flag_lX == false` the corresponding output equals the input
/// `mv_lX` unchanged (the spec's `mMvdLX = 0` then `mMvLX += 0`).
///
/// Round 223 deliberately keeps this as a pure helper: no `Sps`, no
/// `Slice`, no merge-candidate-list state. It will be invoked by a future
/// §8.5.2.3.9 entry point that resolves `mmvd_group_idx` and threads the
/// POC arithmetic from a populated `RefPicList0 / RefPicList1`.
pub fn mmvd_apply_bipred_offset(
    mv_l0: MotionVector,
    mv_l1: MotionVector,
    mmvd_offset: MotionVector,
    curr_poc_diff_l0: i32,
    curr_poc_diff_l1: i32,
    pred_flag_l0: bool,
    pred_flag_l1: bool,
) -> Result<(MotionVector, MotionVector)> {
    if !pred_flag_l0 && !pred_flag_l1 {
        return Err(Error::unsupported(
            "evc inter: mmvd_apply_bipred_offset called with predFlagL0 == predFlagL1 == 0 (§8.5.2.3.9 eqs. 591-616 require at least one active list)"
                .to_string(),
        ));
    }

    if pred_flag_l0 && pred_flag_l1 {
        // Bipred branch — eqs. 591-610.
        let abs_l0 = curr_poc_diff_l0.unsigned_abs() as i32;
        let abs_l1 = curr_poc_diff_l1.unsigned_abs() as i32;

        let (mut mvd_l0, mut mvd_l1) = if abs_l0 == abs_l1 {
            // Symmetric — eqs. 593-596.
            (mmvd_offset, mmvd_offset)
        } else if abs_l0 > abs_l1 {
            // L1 closer — eqs. 597-601. mMvdL1 = MmvdOffset; mMvdL0 is the
            // round-half-up POC scaling of mMvdL1 by (|L1| << 5) / |L0|.
            let sf = mmvd_dist_scale_factor(abs_l1, abs_l0)?;
            let mvd_l1 = mmvd_offset;
            let mvd_l0 = MotionVector {
                x: mmvd_scale_component(sf, mvd_l1.x),
                y: mmvd_scale_component(sf, mvd_l1.y),
            };
            (mvd_l0, mvd_l1)
        } else {
            // L0 closer — eqs. 602-606. Symmetric to case 2.
            let sf = mmvd_dist_scale_factor(abs_l0, abs_l1)?;
            let mvd_l0 = mmvd_offset;
            let mvd_l1 = MotionVector {
                x: mmvd_scale_component(sf, mvd_l0.x),
                y: mmvd_scale_component(sf, mvd_l0.y),
            };
            (mvd_l0, mvd_l1)
        };

        // Eqs. 607-610: opposite-side POCs flip the sign of mMvdL1.
        // currPocDiffL0 * currPocDiffL1 < 0 ⇔ signs of the two POC diffs
        // disagree (neither is zero — the symmetric branch handled the
        // zero-magnitude tie, and a single zero would route through that
        // branch with abs equality).
        if (curr_poc_diff_l0 < 0) ^ (curr_poc_diff_l1 < 0)
            && curr_poc_diff_l0 != 0
            && curr_poc_diff_l1 != 0
        {
            mvd_l1 = MotionVector {
                x: -mvd_l1.x,
                y: -mvd_l1.y,
            };
            // eqs. 607/608 leave mvd_l0 unchanged (`mMvdL0 = mMvdL0`).
            let _ = &mut mvd_l0;
        }

        // Eqs. 613-616: mMvLX += mMvdLX with the same wrap16 semantics
        // shared with eqs. 436/439.
        Ok((mv_l0.wrapping_add(&mvd_l0), mv_l1.wrapping_add(&mvd_l1)))
    } else {
        // One-list-active "Otherwise" branch — eqs. 611-612.
        // mMvdLX = predFlagLX == 1 ? MmvdOffset : 0
        let mvd_l0 = if pred_flag_l0 {
            mmvd_offset
        } else {
            MotionVector::default()
        };
        let mvd_l1 = if pred_flag_l1 {
            mmvd_offset
        } else {
            MotionVector::default()
        };
        Ok((mv_l0.wrapping_add(&mvd_l0), mv_l1.wrapping_add(&mvd_l1)))
    }
}

// =====================================================================
// §8.5.2.3.9 — Round 229: entry-process signed POC scaling primitives
//
// The §8.5.2.3.9 entry process (eqs. 531-590) reassigns the merge
// candidate's `(refIdxL0, refIdxL1, predFlagL0[0][0], predFlagL1[0][0])`
// and pre-scales one list's MV against the other when
// `mmvd_group_idx ∈ { 1, 2 }`. The scaling form that repeats across
// eqs. 543/544, 552/553, 560/561, 572/573, 581/582, 589/590 is
//
//   distScaleFactor = ( currPocDiffNum << 5 ) / currPocDiffDen      [signed]
//   mMv[k] = Clip3(-32768, 32767,
//               Sign(distScaleFactor * mMv[k])
//             * ( ( Abs(distScaleFactor * mMv[k]) + 16 ) >> 5 ))
//
// This is arithmetically distinct from the bipred-tail eqs. 600/601/
// 605/606 form (which uses absolute POC diffs into
// `mmvd_dist_scale_factor` and plain `(s*v + 16) >> 5`): here the
// POC diffs flow in **signed**, and the magnitude is scaled with
// **round-toward-zero** semantics via `Sign(x) * ((Abs(x) + 16) >> 5)`.
// Negative products therefore round differently from r223's arithmetic
// right-shift form.
//
// Round 229 lands the two shared helpers + the `targetRefIdxL0 ==
// refIdxL0` magic offset constant (eqs. 547 / 576). The full §8.5.2.3.9
// entry process (mmvd_group_idx + slice_type dispatch, refIdx /
// predFlag updates) still waits on the merge-candidate-list builder +
// `NumRefIdxActive[]` / `RefPicListX` threading that the round-218 /
// round-223 followups documented.
// =====================================================================

/// §8.5.2.3.9 eqs. 547 / 576 — "same target ref" P-slice offset.
///
/// In the `mmvd_group_idx == 1` / `slice_type == P` branch (eqs. 545-548)
/// the entry process adds 3 to `mMvL0[0]` when the selected
/// `targetRefIdxL0 == refIdxL0`. The `mmvd_group_idx == 2` mirror branch
/// (eqs. 574-577) subtracts 3 instead. The y component is untouched in
/// both cases.
///
/// The constant is exposed so the future §8.5.2.3.9 entry-point can use
/// the same symbol on both sign sides.
pub const MMVD_P_SAME_TARGET_SHIFT: i32 = 3;

/// §8.5.2.3.9 eqs. 542 / 551 / 559 / 571 / 580 / 588 — signed POC-diff
/// scaling factor used by the entry-process branches:
///
/// ```text
/// distScaleFactor = ( pocDiffNum << 5 ) / pocDiffDen
/// ```
///
/// Both operands flow in **signed** here (unlike `mmvd_dist_scale_factor`
/// which takes pre-absoluted operands). A zero denominator surfaces
/// `Error::Unsupported`; the §8.5.2.3.9 sub-branches all guard against
/// the zero-POC-diff case upstream (the spec enters the scaled form only
/// after picking a `currPocDiffL0` from a distinct reference picture).
pub fn mmvd_signed_dist_scale_factor(poc_diff_num: i32, poc_diff_den: i32) -> Result<i32> {
    if poc_diff_den == 0 {
        return Err(Error::unsupported(format!(
            "evc inter: mmvd_signed_dist_scale_factor needs non-zero denominator, got {poc_diff_den} (§8.5.2.3.9 eqs. 542/551/559/571/580/588)"
        )));
    }
    // Signed left-shift then signed integer division. `i64` widening
    // avoids overflow when `poc_diff_num` lives near i32 bounds; in
    // practice POC diffs are tiny but the spec's arithmetic domain is
    // the same i32 the surrounding equations use.
    let num = (poc_diff_num as i64) << 5;
    let den = poc_diff_den as i64;
    Ok((num / den) as i32)
}

/// §8.5.2.3.9 eqs. 543 / 544 / 552 / 553 / 560 / 561 / 572 / 573 / 581 /
/// 582 / 589 / 590 — round-toward-zero scaling of one MV component:
///
/// ```text
/// out = Clip3(-32768, 32767,
///             Sign(distScaleFactor * v)
///           * ( ( Abs(distScaleFactor * v) + 16 ) >> 5 ))
/// ```
///
/// The `Sign(x) * ((Abs(x) + 16) >> 5)` form is **round-toward-zero**
/// (symmetric in `±x`) with a half-up bias on the magnitude. The bipred
/// tail (eqs. 600 / 601 / 605 / 606, round-223 `mmvd_scale_component`)
/// uses the simpler `Clip3(-32768, 32767, (s * v + 16) >> 5)` arithmetic
/// right-shift form, which rounds toward `-infinity` for negative
/// products. For most operand pairs both forms agree, but the symmetric
/// form is what §8.5.2.3.9 spells out for every entry-process scaling
/// equation (eqs. 543-590), so we keep it as the literal arithmetic.
///
/// Worked example: `s = 32`, `v = -1` ⇒ product `-32`,
/// `(|−32| + 16) >> 5 = 48 >> 5 = 1`, `Sign(−32) = −1`, result `−1`.
/// At `s = 32`, `v = 1` the same calculation flips sign back to `+1`.
pub fn mmvd_signed_scale_component(dist_scale_factor: i32, mv_component: i32) -> i32 {
    let product = (dist_scale_factor as i64).wrapping_mul(mv_component as i64);
    let sign: i64 = match product.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    };
    let abs = product.unsigned_abs() as i64;
    let mag = (abs + 16) >> 5;
    let scaled = sign * mag;
    scaled.clamp(-32768, 32767) as i32
}

/// §8.5.2.3.9 — scale both axes of an MV with the round-toward-zero
/// signed form. Convenience wrapper around `mmvd_signed_scale_component`
/// matching the shape the entry-process sub-branches use (each emits
/// two eqs., one per axis, with the same `distScaleFactor`).
pub fn mmvd_signed_scale_mv(dist_scale_factor: i32, mv: MotionVector) -> MotionVector {
    MotionVector {
        x: mmvd_signed_scale_component(dist_scale_factor, mv.x),
        y: mmvd_signed_scale_component(dist_scale_factor, mv.y),
    }
}

// ----------------------------------------------------------------
// §8.5.2.3.10 — Derivation process for motion vector prediction
// redundancy check
// ----------------------------------------------------------------
//
// Round-232 lands the de-duplication step that closes every §8.5.2.3.x
// merge-candidate-list append in the spec text. The §8.5.2.3.1 –
// §8.5.2.3.8 append paths each finish by invoking
// `merge_cand_redundancy_check` with the (potentially-grown) list and
// the current `numCurrMergeCand`. The check compares the just-appended
// tail entry (`mergeCandList[numCurrMergeCand - 1]`) against every
// entry already in the list, in ascending index order, until either
// (a) it finds a duplicate (in which case the tail is "absorbed" — the
// count is decremented by 1 to reclaim the slot), or (b) it has
// compared against every prior entry (then the tail is genuinely new
// and the count stays put).
//
// The matching predicate per the §8.5.2.3.10 ordered steps:
//
//   1) Number of available reference lists agrees (per-LX `predFlag`s
//      together encode availability — 0/1/2 lists active).
//   2) Same available reference list indices (i.e. the L0/L1
//      activation bitmask agrees, not just the count).
//   3) Same `refIdxLX` in each active list.
//   4) Same `mvLX` in each active list.
//
// Inactive lists are skipped for the (3) / (4) compares: the spec's
// "corresponding to available reference lists" qualifier means a
// dormant `predFlagLX = 0` slot's refIdx / MV values are not part of
// the predicate.

/// §8.5.2.3.x merge-candidate entry — the per-CU motion descriptor
/// produced by the spatial / temporal / HMVP / MMVD append paths and
/// consumed by §8.5.2.3.10. Compact value type so the redundancy-check
/// loop can iterate without bounds checks fighting the borrow checker.
///
/// `pred_flag_lX` agrees with `PredFlagLX[ xCb ][ yCb ]` at the
/// candidate's representative sample (§8.5.2.3 inputs). `ref_idx_lX`
/// and `mv_lX` only carry meaning when the corresponding flag is set;
/// the redundancy check explicitly ignores them when the flag is 0
/// (per the §8.5.2.3.10 "corresponding to available reference lists"
/// scoping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeCand {
    /// `predFlagL0[ 0 ][ 0 ]` for the candidate.
    pub pred_flag_l0: bool,
    /// `predFlagL1[ 0 ][ 0 ]` for the candidate.
    pub pred_flag_l1: bool,
    /// `refIdxL0` — meaningful only when `pred_flag_l0` is set.
    pub ref_idx_l0: i32,
    /// `refIdxL1` — meaningful only when `pred_flag_l1` is set.
    pub ref_idx_l1: i32,
    /// `mvL0` — meaningful only when `pred_flag_l0` is set.
    pub mv_l0: MotionVector,
    /// `mvL1` — meaningful only when `pred_flag_l1` is set.
    pub mv_l1: MotionVector,
}

/// §8.5.2.3.10 matching predicate — `mergeCandList[ a ]` and
/// `mergeCandList[ b ]` satisfy every condition in the four ordered
/// steps.
///
/// Equivalent to a structural equality on the active-list-restricted
/// projection: inactive lists never participate in the compare, so two
/// candidates with the same `predFlag` bitmask but a stale residual
/// `refIdxL1` from a previous L0-only append are still considered
/// equal under this predicate.
///
/// Used internally by `merge_cand_redundancy_check`; exposed because
/// the §8.5.2.3.x append paths in future rounds will call it directly
/// when they need a single-pair compare without the trim loop.
pub fn merge_cand_matches(a: &MergeCand, b: &MergeCand) -> bool {
    // Step 1+2: number of available reference lists + which lists are
    // available collapse to "same `predFlag` bitmask".
    if a.pred_flag_l0 != b.pred_flag_l0 || a.pred_flag_l1 != b.pred_flag_l1 {
        return false;
    }
    // Step 3: same `refIdxLX` in each active list.
    if a.pred_flag_l0 && a.ref_idx_l0 != b.ref_idx_l0 {
        return false;
    }
    if a.pred_flag_l1 && a.ref_idx_l1 != b.ref_idx_l1 {
        return false;
    }
    // Step 4: same `mvLX` in each active list.
    if a.pred_flag_l0 && a.mv_l0 != b.mv_l0 {
        return false;
    }
    if a.pred_flag_l1 && a.mv_l1 != b.mv_l1 {
        return false;
    }
    true
}

/// §8.5.2.3.10 — trim a freshly-appended merge candidate from the tail
/// of `merge_cand_list` if it duplicates any earlier entry.
///
/// `num_curr_merge_cand` is the count after the §8.5.2.3.x append that
/// just placed a candidate at index `num_curr_merge_cand - 1`. The
/// scan walks `cand_indx` from 0 up to (but not equal to)
/// `num_curr_merge_cand - 1`, terminating at the first duplicate it
/// finds. When a duplicate is found, the returned count is
/// `num_curr_merge_cand - 1` (the spec's "decremented by 1"); the
/// caller treats indices ≥ the returned count as logically removed —
/// the list buffer itself is unchanged (the spec doesn't mandate
/// zeroing).
///
/// When `num_curr_merge_cand ≤ 1` the routine is a no-op (the spec's
/// pre-test "When numCurrMergeCand is greater than 1"); the input
/// count is returned untouched.
///
/// Returns the updated `numCurrMergeCand`. Errors:
///
/// * `Error::Unsupported` if `num_curr_merge_cand` exceeds the slice's
///   capacity — that mismatch implies an upstream §8.5.2.3.x append
///   wrote past the array bound and is a bookkeeping bug, not a
///   stream-recoverable condition.
pub fn merge_cand_redundancy_check(
    merge_cand_list: &[MergeCand],
    num_curr_merge_cand: usize,
) -> Result<usize> {
    if num_curr_merge_cand > merge_cand_list.len() {
        return Err(Error::Unsupported(format!(
            "evc inter: merge_cand_redundancy_check num_curr_merge_cand={num_curr_merge_cand} exceeds buffer len={} (§8.5.2.3.10 caller bug)",
            merge_cand_list.len()
        )));
    }
    // Spec pre-condition: "When numCurrMergeCand is greater than 1".
    if num_curr_merge_cand <= 1 {
        return Ok(num_curr_merge_cand);
    }
    let tail_indx = num_curr_merge_cand - 1;
    let tail = &merge_cand_list[tail_indx];
    let mut cand_indx = 0usize;
    // "repeated until candIsNew is equal to FALSE or candIndx is equal
    // to numCurrMergeCand − 2": after the loop body increments
    // `candIndx`, the spec re-checks both exit predicates. At
    // `cand_indx == num_curr_merge_cand - 2` (i.e. `cand_indx ==
    // tail_indx - 1`) the next compare is against the entry directly
    // before the tail; if that fails, the loop body increments
    // `cand_indx` to `tail_indx - 1 + 1 = tail_indx` which would
    // compare the tail against itself — exactly the case the spec's
    // exit predicate forbids. So the active scan range is `0 ..
    // tail_indx` (i.e. `0 ..= tail_indx - 1`).
    while cand_indx < tail_indx {
        let cand_is_new = !merge_cand_matches(&merge_cand_list[cand_indx], tail);
        if !cand_is_new {
            // Spec: "When candIsNew is equal to FALSE, the variable
            // numCurrMergeCand is decremented by 1." The exit clause
            // fires immediately.
            return Ok(num_curr_merge_cand - 1);
        }
        cand_indx += 1;
    }
    // Walked every prior entry without a match: tail is genuinely new.
    Ok(num_curr_merge_cand)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `wrap16` reproduces the spec's mod-2^16 / sign-extension rule.
    #[test]
    fn wrap16_round_trip() {
        assert_eq!(wrap16(0), 0);
        assert_eq!(wrap16(32767), 32767);
        assert_eq!(wrap16(32768), -32768);
        assert_eq!(wrap16(-1), -1);
        assert_eq!(wrap16(0x10000 + 5), 5);
        assert_eq!(wrap16(-32769), 32767);
    }

    /// Eq. 436–439 wrap MV addition into 16-bit signed range.
    #[test]
    fn mv_wrapping_add() {
        let a = MotionVector::quarter_pel(32000, -32000);
        let b = MotionVector::quarter_pel(1000, -1000);
        let c = a.wrapping_add(&b);
        assert_eq!(c, MotionVector::quarter_pel(-32536, 32536));
    }

    /// Integer-sample copy short-circuit: a zero MV reads the integer
    /// reference verbatim.
    #[test]
    fn luma_zero_mv_copies_reference() {
        let mut y = vec![0u8; 16 * 16];
        for j in 0..16 {
            for i in 0..16 {
                y[j * 16 + i] = (i * 16 + j) as u8;
            }
        }
        let cb = vec![128u8; 64];
        let cr = vec![128u8; 64];
        let refp = RefPictureView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 16,
            height: 16,
            y_stride: 16,
            c_stride: 8,
            chroma_format_idc: 1,
        };
        let mut out = vec![0i32; 4 * 4];
        interpolate_luma_block(refp, 4, 4, MotionVector::default(), 4, 4, 8, &mut out).unwrap();
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(
                    out[j * 4 + i] as u8,
                    y[(4 + j) * 16 + (4 + i)],
                    "row {j} col {i}"
                );
            }
        }
    }

    /// 1/2-pel luma filter sums to 64 (the >>6 normalisation factor).
    #[test]
    fn luma_half_pel_filter_sums_to_64() {
        let coeffs = LUMA_FILTER_TABLE25[8];
        let s: i32 = coeffs.iter().sum();
        assert_eq!(s, 64);
    }

    /// 1/4-pel luma filter sums to 64 too.
    #[test]
    fn luma_quarter_pel_filter_sums_to_64() {
        let s: i32 = LUMA_FILTER_TABLE25[4].iter().sum();
        assert_eq!(s, 64);
        let s: i32 = LUMA_FILTER_TABLE25[12].iter().sum();
        assert_eq!(s, 64);
    }

    /// All Baseline chroma filters sum to 64.
    #[test]
    fn chroma_filters_sum_to_64() {
        for &p in &[4u32, 8, 12, 16, 20, 24, 28] {
            let s: i32 = CHROMA_FILTER_TABLE27[p as usize].iter().sum();
            assert_eq!(s, 64, "chroma phase {p}");
        }
    }

    /// 1/2-pel luma filter on a uniform reference picture leaves the
    /// signal at the same DC value.
    #[test]
    fn luma_half_pel_filter_dc_invariant() {
        let y = vec![100u8; 32 * 32];
        let cb = vec![128u8; 16 * 16];
        let cr = vec![128u8; 16 * 16];
        let refp = RefPictureView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 32,
            height: 32,
            y_stride: 32,
            c_stride: 16,
            chroma_format_idc: 1,
        };
        // mv = (8, 8) at 1/16-pel → 1/2-pel in both directions. int = 0,
        // frac = 8 each.
        let mv = MotionVector::quarter_pel(8, 8);
        let mut out = vec![0i32; 4 * 4];
        interpolate_luma_block(refp, 8, 8, mv, 4, 4, 8, &mut out).unwrap();
        for v in out {
            assert_eq!(v, 100, "DC must survive 1/2-pel filter");
        }
    }

    /// Non-Baseline luma phase is rejected.
    #[test]
    fn rejects_non_baseline_luma_phase() {
        let y = vec![0u8; 16 * 16];
        let cb = vec![0u8; 64];
        let cr = vec![0u8; 64];
        let refp = RefPictureView {
            y: &y,
            cb: &cb,
            cr: &cr,
            width: 16,
            height: 16,
            y_stride: 16,
            c_stride: 8,
            chroma_format_idc: 1,
        };
        let mv = MotionVector::quarter_pel(1, 0); // 1/16-pel — not in Baseline grid
        let mut out = vec![0i32; 16];
        let err = interpolate_luma_block(refp, 0, 0, mv, 4, 4, 8, &mut out).unwrap_err();
        assert!(format!("{err}").contains("Baseline"));
    }

    /// Bi-prediction averaging.
    #[test]
    fn average_bipred_rounds_correctly() {
        let a = vec![100, 100, 200, 200];
        let b = vec![100, 101, 200, 201];
        let mut o = vec![0i32; 4];
        average_bipred(&a, &b, &mut o);
        assert_eq!(o, vec![100, 101, 200, 201]); // (100+100+1)/2=100, (100+101+1)/2=101
    }

    /// AMVP list with all neighbours unavailable falls back to (1,1) for
    /// spatial slots and zero MV for the temporal slot.
    #[test]
    fn amvp_fallback_for_unavailable_neighbours() {
        let list = build_amvp_list_baseline(0, 0, 16, 16, |_, _| None, MotionVector::default());
        assert_eq!(list[0].0, MotionVector::quarter_pel(1, 1));
        assert_eq!(list[1].0, MotionVector::quarter_pel(1, 1));
        assert_eq!(list[2].0, MotionVector::quarter_pel(1, 1));
        assert_eq!(list[3].0, MotionVector::default());
    }

    /// AMVP list pulls the requested neighbour MVs.
    #[test]
    fn amvp_picks_neighbour_mvs() {
        let nb_a = MotionVector::quarter_pel(4, -4);
        let nb_b = MotionVector::quarter_pel(-8, 0);
        let list = build_amvp_list_baseline(
            16,
            16,
            16,
            16,
            |x, y| match (x, y) {
                (15, 16) => Some(nb_a), // left at (xCb-1, yCb)
                (16, 15) => Some(nb_b), // above at (xCb, yCb-1)
                _ => None,
            },
            MotionVector::default(),
        );
        assert_eq!(list[0].0, nb_a);
        assert_eq!(list[1].0, nb_b);
        assert_eq!(list[2].0, MotionVector::quarter_pel(1, 1));
        assert_eq!(list[3].0, MotionVector::default());
    }

    /// Chroma MV derivation: 4:2:0 keeps the same numeric components.
    #[test]
    fn chroma_mv_passthrough_420() {
        let mv = MotionVector::quarter_pel(40, 20); // 1/16-pel luma
        let mvc = derive_chroma_mv(mv, 1);
        assert_eq!(mvc, MotionVector::quarter_pel(40, 20));
    }

    /// Quarter-to-sixteenth conversion is a left-shift by 2.
    #[test]
    fn quarter_to_sixteenth_shift() {
        let mv = MotionVector::quarter_pel(3, -5);
        let mv16 = mv.quarter_to_sixteenth();
        assert_eq!(mv16, MotionVector::quarter_pel(12, -20));
    }

    // -------- §8.5.3.10 round_motion_vector (eq. 907/908/909) --------

    /// `right_shift == 0` short-circuits the offset to 0, so the round
    /// reduces to a pure `(mv − (mv >= 0)) << leftShift`.
    #[test]
    fn round_mv_zero_right_shift_no_offset() {
        // Positive component: (5 - 1) >> 0 << 0 = 4.
        // Negative component: (-5 - 0) >> 0 << 0 = -5.
        let mv = MotionVector::quarter_pel(5, -5);
        let r = round_motion_vector(mv, 0, 0);
        assert_eq!(r, MotionVector::quarter_pel(4, -5));
    }

    /// Round-toward-negative-infinity flavour on positive values:
    /// rounding by 4 (right_shift = 2) with no left_shift sends 7 to 1 (not 2),
    /// because the spec subtracts 1 from positives before the shift.
    #[test]
    fn round_mv_positive_right_shift_two() {
        // For mv = 7: (7 + 2 - 1) >> 2 << 0 = 8 >> 2 = 2.
        // For mv = 8: (8 + 2 - 1) >> 2 << 0 = 9 >> 2 = 2.
        // For mv = 4: (4 + 2 - 1) >> 2 << 0 = 5 >> 2 = 1.
        let r = round_motion_vector(MotionVector::quarter_pel(7, 8), 2, 0);
        assert_eq!(r, MotionVector::quarter_pel(2, 2));
        let r = round_motion_vector(MotionVector::quarter_pel(4, 5), 2, 0);
        assert_eq!(r, MotionVector::quarter_pel(1, 1));
    }

    /// On negatives, the `nonneg_adj = 0` branch fires so the offset is
    /// applied directly; the arithmetic right shift rounds toward
    /// negative infinity.
    #[test]
    fn round_mv_negative_right_shift_two() {
        // For mv = -7: (-7 + 2 - 0) >> 2 << 0 = -5 >> 2 = -2 (arith shift
        // rounds toward -inf: -5 = ..1011 → ..1111110 = -2).
        // For mv = -8: (-8 + 2 - 0) >> 2 << 0 = -6 >> 2 = -2.
        // For mv = -4: (-4 + 2 - 0) >> 2 << 0 = -2 >> 2 = -1.
        // For mv = -5: (-5 + 2 - 0) >> 2 << 0 = -3 >> 2 = -1.
        let r = round_motion_vector(MotionVector::quarter_pel(-7, -8), 2, 0);
        assert_eq!(r, MotionVector::quarter_pel(-2, -2));
        let r = round_motion_vector(MotionVector::quarter_pel(-4, -5), 2, 0);
        assert_eq!(r, MotionVector::quarter_pel(-1, -1));
    }

    /// Combined right-then-left shift: round to 1/4-pel, scale back to
    /// 1/16-pel. (rightShift=2, leftShift=2.)
    #[test]
    fn round_mv_then_resolution_increase() {
        // mv = 11: (11 + 2 - 1) >> 2 << 2 = (12 >> 2) << 2 = 3 << 2 = 12.
        // mv = -11: (-11 + 2 - 0) >> 2 << 2 = (-9 >> 2) << 2 = -3 << 2 = -12.
        let r = round_motion_vector(MotionVector::quarter_pel(11, -11), 2, 2);
        assert_eq!(r, MotionVector::quarter_pel(12, -12));
    }

    /// At right_shift == 1, the offset is `1 << 0 = 1` so the round
    /// becomes `(mv + 1 - (mv >= 0)) >> 1 << leftShift` — which for
    /// mv == 0 cancels exactly (1 + 0 - 1 = 0) and stays at 0.
    #[test]
    fn round_mv_zero_round_trip_at_right_shift_one() {
        let r = round_motion_vector(MotionVector::default(), 1, 0);
        assert_eq!(r, MotionVector::default());
    }

    // ====================================================================
    // §8.5 AMVR — round-213 (eq. 145, eq. 645/646, §9.3.4 ctx layout).
    // ====================================================================

    /// Eq. 145 is a plain left-shift; `amvr_idx == 0` is identity.
    #[test]
    fn round213_amvr_apply_to_mvd_zero_idx_identity() {
        assert_eq!(amvr_apply_to_mvd(5, 0).unwrap(), 5);
        assert_eq!(amvr_apply_to_mvd(-7, 0).unwrap(), -7);
        assert_eq!(amvr_apply_to_mvd(0, 0).unwrap(), 0);
    }

    /// Eq. 145 shifts MVD components left by amvr_idx (signed).
    #[test]
    fn round213_amvr_apply_to_mvd_shift_examples() {
        // amvr_idx = 1 (1/2-pel): MVD scales by 2.
        assert_eq!(amvr_apply_to_mvd(3, 1).unwrap(), 6);
        assert_eq!(amvr_apply_to_mvd(-3, 1).unwrap(), -6);
        // amvr_idx = 2 (integer-pel): MVD scales by 4.
        assert_eq!(amvr_apply_to_mvd(3, 2).unwrap(), 12);
        assert_eq!(amvr_apply_to_mvd(-3, 2).unwrap(), -12);
        // amvr_idx = 4 (4-pel): MVD scales by 16.
        assert_eq!(amvr_apply_to_mvd(1, 4).unwrap(), 16);
        assert_eq!(amvr_apply_to_mvd(-1, 4).unwrap(), -16);
    }

    /// Vector form mirrors the component form on both axes.
    #[test]
    fn round213_amvr_apply_to_mvd_vector_both_axes() {
        let mvd = MotionVector::quarter_pel(3, -5);
        let scaled = amvr_apply_to_mvd_vector(mvd, 2).unwrap();
        assert_eq!(scaled, MotionVector::quarter_pel(12, -20));
    }

    /// `amvr_idx > 4` is outside the TR cMax range.
    #[test]
    fn round213_amvr_apply_to_mvd_rejects_oob_idx() {
        assert!(amvr_apply_to_mvd(0, 5).is_err());
        assert!(amvr_apply_to_mvd_vector(MotionVector::default(), 5).is_err());
    }

    /// Eq. 645/646 with `amvr_idx == 0` returns the predictor unchanged.
    /// (The spec gates the entire eq. 645/646 block on `amvr_idx != 0`,
    /// but the helper is defined for the full `0..=4` range so callers
    /// can lift the gate without a special case.)
    #[test]
    fn round213_amvr_round_mvp_zero_idx_identity() {
        assert_eq!(amvr_round_mvp(13, 0).unwrap(), 13);
        assert_eq!(amvr_round_mvp(-13, 0).unwrap(), -13);
    }

    /// Eq. 645/646 — sign-symmetric round-half-up worked examples at
    /// `amvr_idx == 2` (mask = 0b11, half = 2, shift = 2):
    ///   +1 → ((+1+2)>>2)<<2 = 0
    ///   +2 → ((+2+2)>>2)<<2 = 4
    ///   +3 → ((+3+2)>>2)<<2 = 4
    ///   −1 → −(((+1+2)>>2)<<2) = 0
    ///   −2 → −(((+2+2)>>2)<<2) = −4
    ///   −3 → −(((+3+2)>>2)<<2) = −4
    #[test]
    fn round213_amvr_round_mvp_sign_symmetric_at_idx2() {
        assert_eq!(amvr_round_mvp(1, 2).unwrap(), 0);
        assert_eq!(amvr_round_mvp(2, 2).unwrap(), 4);
        assert_eq!(amvr_round_mvp(3, 2).unwrap(), 4);
        assert_eq!(amvr_round_mvp(-1, 2).unwrap(), 0);
        assert_eq!(amvr_round_mvp(-2, 2).unwrap(), -4);
        assert_eq!(amvr_round_mvp(-3, 2).unwrap(), -4);
    }

    /// The sign-symmetric AMVR round distinguishes itself from the
    /// §8.5.3.10 round_motion_vector at `mv = -2, amvr_idx = 2`:
    /// AMVR returns −4 (rounds toward larger magnitude), affine round
    /// returns 0 (rounds toward zero). This is the load-bearing
    /// distinction between the two rounding modes — using one in place
    /// of the other yields non-conforming MV reconstruction.
    #[test]
    fn round213_amvr_round_mvp_differs_from_affine_round_for_negatives() {
        let mv = MotionVector::quarter_pel(-2, -2);
        let amvr = amvr_round_mvp_vector(mv, 2).unwrap();
        let affine = round_motion_vector(mv, 2, 2);
        assert_eq!(amvr, MotionVector::quarter_pel(-4, -4));
        assert_eq!(affine, MotionVector::quarter_pel(0, 0));
        assert_ne!(amvr, affine);
    }

    /// Eq. 645/646 at `amvr_idx == 1` rounds 1/4-pel toward 1/2-pel.
    #[test]
    fn round213_amvr_round_mvp_half_pel() {
        // +1 → ((1+1)>>1)<<1 = 2; -1 → -2.
        assert_eq!(amvr_round_mvp(1, 1).unwrap(), 2);
        assert_eq!(amvr_round_mvp(-1, 1).unwrap(), -2);
        // +3 → 4; -3 → -4. Even values unchanged.
        assert_eq!(amvr_round_mvp(3, 1).unwrap(), 4);
        assert_eq!(amvr_round_mvp(-3, 1).unwrap(), -4);
        assert_eq!(amvr_round_mvp(4, 1).unwrap(), 4);
        assert_eq!(amvr_round_mvp(-4, 1).unwrap(), -4);
    }

    /// Eq. 645/646 at `amvr_idx == 4` rounds to multiples of 16.
    #[test]
    fn round213_amvr_round_mvp_four_pel() {
        // +7 → ((7+8)>>4)<<4 = 0; +8 → ((8+8)>>4)<<4 = 16.
        assert_eq!(amvr_round_mvp(7, 4).unwrap(), 0);
        assert_eq!(amvr_round_mvp(8, 4).unwrap(), 16);
        // -7 → 0; -8 → -16.
        assert_eq!(amvr_round_mvp(-7, 4).unwrap(), 0);
        assert_eq!(amvr_round_mvp(-8, 4).unwrap(), -16);
    }

    /// Vector form mirrors the component form on both axes.
    #[test]
    fn round213_amvr_round_mvp_vector_both_axes() {
        let mv = MotionVector::quarter_pel(7, -8);
        // amvr_idx = 3 → multiples of 8, half = 4.
        // +7 → ((7+4)>>3)<<3 = (11>>3)<<3 = 8.
        // -8 → -(((8+4)>>3)<<3) = -((12>>3)<<3) = -8.
        let r = amvr_round_mvp_vector(mv, 3).unwrap();
        assert_eq!(r, MotionVector::quarter_pel(8, -8));
    }

    /// `amvr_idx > 4` outside TR cMax range.
    #[test]
    fn round213_amvr_round_mvp_rejects_oob_idx() {
        assert!(amvr_round_mvp(0, 5).is_err());
        assert!(amvr_round_mvp_vector(MotionVector::default(), 5).is_err());
    }

    /// §9.3.4 ctxInc for `amvr_idx` is positional (Table 67 ranges
    /// `0..3` / `4..7`; `amvr_idx` is **not** in Table 96, so there is
    /// no neighbour-derived term).
    #[test]
    fn round213_amvr_idx_ctx_inc_is_positional() {
        assert_eq!(amvr_idx_ctx_inc(0).unwrap(), 0);
        assert_eq!(amvr_idx_ctx_inc(1).unwrap(), 1);
        assert_eq!(amvr_idx_ctx_inc(2).unwrap(), 2);
        assert_eq!(amvr_idx_ctx_inc(3).unwrap(), 3);
    }

    /// `bin_idx >= 4` is outside the TR prefix range (cMax = 4 means at
    /// most 4 prefix bins — the 4-bit TR codeword "1111" has bin 3 as
    /// its last bin and no terminator).
    #[test]
    fn round213_amvr_idx_ctx_inc_rejects_oob_bin() {
        assert!(amvr_idx_ctx_inc(4).is_err());
        assert!(amvr_idx_ctx_inc(99).is_err());
    }

    /// §9.3.3 — `merge_idx` cMax is area-dependent: `nCbW*nCbH <= 32`
    /// → 3, else 5. The boundary is exactly 32 samples.
    #[test]
    fn round377_merge_idx_c_max_area_boundary() {
        assert_eq!(merge_idx_c_max(4, 8), 3); // 32 → small branch
        assert_eq!(merge_idx_c_max(8, 4), 3); // 32 → small branch
        assert_eq!(merge_idx_c_max(8, 8), 5); // 64 → large branch
        assert_eq!(merge_idx_c_max(4, 16), 5); // 64 → large branch
        assert_eq!(merge_idx_c_max(4, 4), 3); // 16 → small branch
    }

    /// §9.3.4.2 — `merge_idx` ctxInc is positional (Table 49 ranges
    /// `0..4` / `5..9`), bin k → ctx k for k ∈ 0..=4.
    #[test]
    fn round377_merge_idx_ctx_inc_is_positional() {
        for k in 0..MERGE_IDX_MAX {
            assert_eq!(merge_idx_ctx_inc(k).unwrap(), k as usize);
        }
        assert!(merge_idx_ctx_inc(MERGE_IDX_MAX).is_err());
        assert!(merge_idx_ctx_inc(42).is_err());
    }

    /// §9.3.3 — `inter_pred_idc` cMax: admvp small block (nCbW+nCbH<=12)
    /// caps at 1 (uni-pred only); larger admvp blocks or any non-admvp
    /// block allow cMax 2.
    #[test]
    fn round377_inter_pred_idc_c_max() {
        // admvp, 4+4=8 <= 12 → 1.
        assert_eq!(inter_pred_idc_c_max(true, 4, 4), 1);
        // admvp, 8+8=16 > 12 → 2.
        assert_eq!(inter_pred_idc_c_max(true, 8, 8), 2);
        // admvp boundary: 4+8=12, not > 12 → 1.
        assert_eq!(inter_pred_idc_c_max(true, 4, 8), 1);
        // admvp boundary: 8+8 already covered; 4+16=20 > 12 → 2.
        assert_eq!(inter_pred_idc_c_max(true, 4, 16), 2);
        // !admvp → always 2 regardless of size.
        assert_eq!(inter_pred_idc_c_max(false, 4, 4), 2);
    }

    /// §9.3.4.2 — `inter_pred_idc` / `bi_pred_idx` ctxInc positional 0,1.
    #[test]
    fn round377_inter_pred_idc_and_bi_pred_idx_ctx_inc() {
        assert_eq!(inter_pred_idc_ctx_inc(0).unwrap(), 0);
        assert_eq!(inter_pred_idc_ctx_inc(1).unwrap(), 1);
        assert!(inter_pred_idc_ctx_inc(2).is_err());
        assert_eq!(bi_pred_idx_ctx_inc(0).unwrap(), 0);
        assert_eq!(bi_pred_idx_ctx_inc(1).unwrap(), 1);
        assert!(bi_pred_idx_ctx_inc(BI_PRED_IDX_MAX).is_err());
        // Table 8 direction constants.
        assert_eq!((PRED_L0, PRED_L1, PRED_BI), (0, 1, 2));
    }

    /// The AMVR-MVD shift (eq. 145) and the AMVR-MVP round (eq. 645/646)
    /// commute with the "fully encoded → fully reconstructed" pipeline
    /// at `amvr_idx == 0` — i.e. a Baseline-style stream sees `mv_recon
    /// = mvp + mvd` unchanged. This is the round-trip property the
    /// Baseline pipeline relies on (`sps_amvr_flag = 0` ⇒ all AMVR
    /// helpers are no-ops).
    #[test]
    fn round213_amvr_baseline_pipeline_identity_at_idx0() {
        let mvp = MotionVector::quarter_pel(12, -8);
        let mvd = MotionVector::quarter_pel(3, 5);
        let mvp_rounded = amvr_round_mvp_vector(mvp, 0).unwrap();
        let mvd_shifted = amvr_apply_to_mvd_vector(mvd, 0).unwrap();
        let mv_recon = mvp_rounded.wrapping_add(&mvd_shifted);
        assert_eq!(mvp_rounded, mvp);
        assert_eq!(mvd_shifted, mvd);
        assert_eq!(mv_recon, MotionVector::quarter_pel(15, -3));
    }

    /// Worked Main-profile reconstruction example at `amvr_idx == 2`
    /// (integer-pel resolution): mvp at 1/4-pel `(13, -10)` rounds to
    /// `(12, -12)` (sign-symmetric round-half-up), mvd at 1/4-pel
    /// `(2, -3)` shifts to `(8, -12)`, sum is `(20, -24)`. Demonstrates
    /// the chain pulls cleanly out of the spec, no off-by-one anywhere.
    #[test]
    fn round213_amvr_worked_chain_at_idx2() {
        let mvp = MotionVector::quarter_pel(13, -10);
        let mvd = MotionVector::quarter_pel(2, -3);
        let mvp_rounded = amvr_round_mvp_vector(mvp, 2).unwrap();
        let mvd_shifted = amvr_apply_to_mvd_vector(mvd, 2).unwrap();
        // mvp: +13 → ((13+2)>>2)<<2 = 12; -10 → -(((10+2)>>2)<<2) = -12.
        assert_eq!(mvp_rounded, MotionVector::quarter_pel(12, -12));
        // mvd: 2<<2 = 8; -3<<2 = -12.
        assert_eq!(mvd_shifted, MotionVector::quarter_pel(8, -12));
        let mv_recon = mvp_rounded.wrapping_add(&mvd_shifted);
        assert_eq!(mv_recon, MotionVector::quarter_pel(20, -24));
    }

    // ---- Round 218 — MMVD distance / sign / offset / ctxInc -------------

    /// Table 9 — `MmvdDistance ∈ { 1, 2, 4, 8, 16, 32, 64, 128 }` across
    /// the full `mmvd_distance_idx ∈ 0..=7` range.
    #[test]
    fn round218_mmvd_distance_table9_full_range() {
        let expected: [i32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        for (idx, &want) in expected.iter().enumerate() {
            assert_eq!(
                mmvd_distance(idx as u32).unwrap(),
                want,
                "mmvd_distance_idx = {idx}"
            );
        }
    }

    /// Table 9 — `mmvd_distance_idx = 8` is past TR cMax = 7.
    #[test]
    fn round218_mmvd_distance_rejects_oob_idx() {
        assert!(mmvd_distance(8).is_err());
        assert!(mmvd_distance(255).is_err());
    }

    /// Table 10 — all four `mmvd_direction_idx` rows are axis-aligned and
    /// of unit magnitude.
    #[test]
    fn round218_mmvd_sign_table10_full_range() {
        let expected: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for (idx, &want) in expected.iter().enumerate() {
            assert_eq!(mmvd_sign(idx as u32).unwrap(), want, "row {idx}");
        }
    }

    /// Table 10 — `mmvd_direction_idx = 4` is past FL cMax = 3.
    #[test]
    fn round218_mmvd_sign_rejects_oob_idx() {
        assert!(mmvd_sign(4).is_err());
        assert!(mmvd_sign(7).is_err());
    }

    /// Eq. 133 / 134 — `MmvdOffset = MmvdDistance * MmvdSign` for every
    /// `(mmvd_distance_idx, mmvd_direction_idx)` pair. Spot-check
    /// `(idx_d = 5, idx_dir = 1) ⇒ (−32, 0)` and
    /// `(idx_d = 7, idx_dir = 2) ⇒ (0, 128)`.
    #[test]
    fn round218_mmvd_offset_eq133_eq134_spot_checks() {
        assert_eq!(mmvd_offset(5, 1).unwrap(), MotionVector { x: -32, y: 0 });
        assert_eq!(mmvd_offset(7, 2).unwrap(), MotionVector { x: 0, y: 128 });
        // Smallest non-trivial: idx_d = 0 (distance 1), idx_dir = 0
        // (+x axis) ⇒ (1, 0).
        assert_eq!(mmvd_offset(0, 0).unwrap(), MotionVector { x: 1, y: 0 });
        // Largest negative-y: idx_d = 7, idx_dir = 3 ⇒ (0, −128).
        assert_eq!(mmvd_offset(7, 3).unwrap(), MotionVector { x: 0, y: -128 });
    }

    /// Eq. 133 / 134 — the offset is always axis-aligned. Iterates the
    /// full 8 × 4 = 32-entry Cartesian product and asserts at least one
    /// component is zero. Pins the Table-10 "single non-zero axis"
    /// property so a future column swap surfaces.
    #[test]
    fn round218_mmvd_offset_always_axis_aligned() {
        for d in 0..=MMVD_DISTANCE_IDX_MAX {
            for dir in 0..=MMVD_DIRECTION_IDX_MAX {
                let off = mmvd_offset(d, dir).unwrap();
                assert!(
                    off.x == 0 || off.y == 0,
                    "axis-aligned violated at (d = {d}, dir = {dir}): {off:?}"
                );
            }
        }
    }

    /// Eq. 133 / 134 — offset magnitude equals `MmvdDistance` on the
    /// non-zero axis for every direction. Confirms the Table-10 sign
    /// component is unit magnitude (±1) at every row.
    #[test]
    fn round218_mmvd_offset_magnitude_equals_distance() {
        for d in 0..=MMVD_DISTANCE_IDX_MAX {
            let want_mag = mmvd_distance(d).unwrap();
            for dir in 0..=MMVD_DIRECTION_IDX_MAX {
                let off = mmvd_offset(d, dir).unwrap();
                let mag = off.x.abs().max(off.y.abs());
                assert_eq!(mag, want_mag, "(d = {d}, dir = {dir})");
            }
        }
    }

    /// Eq. 133 / 134 — out-of-range `mmvd_distance_idx` propagates as
    /// `Error::Unsupported` from the offset entry too (defence in depth
    /// at the boundary).
    #[test]
    fn round218_mmvd_offset_propagates_oob_distance_idx() {
        assert!(mmvd_offset(8, 0).is_err());
    }

    /// Eq. 133 / 134 — out-of-range `mmvd_direction_idx` propagates from
    /// the offset entry.
    #[test]
    fn round218_mmvd_offset_propagates_oob_direction_idx() {
        assert!(mmvd_offset(0, 4).is_err());
    }

    /// §9.3.4 — `mmvd_flag` is a single FL bit; bin 0 → ctx 0, bin 1+
    /// rejects.
    #[test]
    fn round218_mmvd_flag_ctx_inc_positional() {
        assert_eq!(mmvd_flag_ctx_inc(0).unwrap(), 0);
        assert!(mmvd_flag_ctx_inc(1).is_err());
    }

    /// §9.3.4 — `mmvd_group_idx` TR bins map 1-to-1 to ctx slots
    /// 0..MMVD_GROUP_IDX_MAX.
    #[test]
    fn round218_mmvd_group_idx_ctx_inc_positional() {
        assert_eq!(mmvd_group_idx_ctx_inc(0).unwrap(), 0);
        assert_eq!(mmvd_group_idx_ctx_inc(1).unwrap(), 1);
        assert!(mmvd_group_idx_ctx_inc(MMVD_GROUP_IDX_MAX).is_err());
    }

    /// §9.3.4 — `mmvd_merge_idx` TR bins map 1-to-1 to ctx slots
    /// 0..MMVD_MERGE_IDX_MAX.
    #[test]
    fn round218_mmvd_merge_idx_ctx_inc_positional() {
        for bin in 0..MMVD_MERGE_IDX_MAX {
            assert_eq!(mmvd_merge_idx_ctx_inc(bin).unwrap(), bin as usize);
        }
        assert!(mmvd_merge_idx_ctx_inc(MMVD_MERGE_IDX_MAX).is_err());
    }

    /// §9.3.4 — `mmvd_distance_idx` TR bins map 1-to-1 to ctx slots
    /// 0..MMVD_DISTANCE_IDX_MAX (the seven trained states of
    /// Table 53 per `initType`).
    #[test]
    fn round218_mmvd_distance_idx_ctx_inc_positional() {
        for bin in 0..MMVD_DISTANCE_IDX_MAX {
            assert_eq!(mmvd_distance_idx_ctx_inc(bin).unwrap(), bin as usize);
        }
        assert!(mmvd_distance_idx_ctx_inc(MMVD_DISTANCE_IDX_MAX).is_err());
    }

    /// §9.3.4 — `mmvd_direction_idx` is FL with cMax = 3, encoded as a
    /// 2-bit FL code (`Ceil(Log2(cMax + 1)) = 2`). The two bins map to
    /// ctx 0 and ctx 1 (Table 54 carries two trained states per
    /// `initType`). bin 2+ rejects.
    #[test]
    fn round218_mmvd_direction_idx_ctx_inc_positional() {
        assert_eq!(mmvd_direction_idx_ctx_inc(0).unwrap(), 0);
        assert_eq!(mmvd_direction_idx_ctx_inc(1).unwrap(), 1);
        assert!(mmvd_direction_idx_ctx_inc(2).is_err());
    }

    /// End-to-end: a parsed `(mmvd_distance_idx, mmvd_direction_idx) =
    /// (3, 2)` ⇒ `MmvdDistance = 8`, `MmvdSign = (0, +1)`, ⇒ `MmvdOffset
    /// = (0, 8)`. This is the offset that the §8.5.2.3.9 derivation
    /// adds to the selected merge candidate's `mvL0` / `mvL1`. Round
    /// 218 ships the (idx → offset) chain; the merge-candidate add /
    /// POC-scaling part is the follow-up.
    #[test]
    fn round218_mmvd_worked_chain_dist3_dir2() {
        let off = mmvd_offset(3, 2).unwrap();
        assert_eq!(off, MotionVector { x: 0, y: 8 });
    }

    /// Baseline pipeline (`sps_mmvd_flag = 0` ⇒ `mmvd_flag = 0`) skips
    /// the MMVD path entirely. No call into any of the round-218
    /// helpers happens — assertion: the helpers' constants are
    /// consistent with the §9.3.3 binarization table (no integer
    /// arithmetic surprise from a refactor changing cMax values).
    #[test]
    fn round218_mmvd_binarization_cmax_constants_match_spec() {
        assert_eq!(MMVD_DISTANCE_IDX_MAX, 7);
        assert_eq!(MMVD_DIRECTION_IDX_MAX, 3);
        assert_eq!(MMVD_GROUP_IDX_MAX, 2);
        assert_eq!(MMVD_MERGE_IDX_MAX, 3);
    }

    // -----------------------------------------------------------------
    // round 223 — §8.5.2.3.9 bipred MMVD offset distribution
    // -----------------------------------------------------------------

    /// Eq. 588 / 599 / 604 — distScaleFactor is `(|num| << 5) / |den|`.
    /// Worked example from a symmetric same-magnitude case: `|num| = 4`,
    /// `|den| = 4` ⇒ `sf = 32`.
    #[test]
    fn round223_dist_scale_factor_eq599_eq604_form() {
        assert_eq!(mmvd_dist_scale_factor(4, 4).unwrap(), 32);
        assert_eq!(mmvd_dist_scale_factor(2, 4).unwrap(), 16);
        assert_eq!(mmvd_dist_scale_factor(4, 2).unwrap(), 64);
        assert_eq!(mmvd_dist_scale_factor(0, 4).unwrap(), 0);
    }

    /// Zero / negative denominator surfaces `Error::Unsupported` rather
    /// than panicking; negative numerator likewise (the caller's bipred
    /// branch always passes `Abs(currPocDiffL?)`).
    #[test]
    fn round223_dist_scale_factor_rejects_bad_inputs() {
        assert!(mmvd_dist_scale_factor(4, 0).is_err());
        assert!(mmvd_dist_scale_factor(4, -1).is_err());
        assert!(mmvd_dist_scale_factor(-1, 4).is_err());
    }

    /// Eqs. 593-596 — when `Abs(currPocDiffL0) == Abs(currPocDiffL1)`,
    /// both `mMvdLX = MmvdOffset`, so `mMvLX = mvLX + MmvdOffset` for
    /// both lists. POC signs equal (no opposite-side sign flip).
    #[test]
    fn round223_bipred_symmetric_same_magnitude_same_sign() {
        let mv_l0 = MotionVector::quarter_pel(10, -4);
        let mv_l1 = MotionVector::quarter_pel(-20, 8);
        let offset = MotionVector::quarter_pel(0, 16);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 4, 4, true, true).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(10, 12));
        assert_eq!(out_l1, MotionVector::quarter_pel(-20, 24));
    }

    /// Eqs. 593-596 plus eqs. 607-610 — symmetric magnitudes but
    /// opposite POC signs ⇒ `mMvdL1` gets negated, `mMvdL0` keeps the
    /// raw offset.
    #[test]
    fn round223_bipred_symmetric_opposite_sign_flips_l1() {
        let mv_l0 = MotionVector::quarter_pel(0, 0);
        let mv_l1 = MotionVector::quarter_pel(0, 0);
        let offset = MotionVector::quarter_pel(32, 0);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 4, -4, true, true).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(32, 0));
        assert_eq!(out_l1, MotionVector::quarter_pel(-32, 0));
    }

    /// Eqs. 597-601 — `Abs(L0) > Abs(L1)` ⇒ `mMvdL1 = MmvdOffset`,
    /// `mMvdL0` is scaled by `sf = (|L1| << 5) / |L0|`. With
    /// `|L0| = 8`, `|L1| = 4`, `MmvdOffset = (64, 0)`, scaled component
    /// is `((4<<5)/8 * 64 + 16) >> 5 = (16 * 64 + 16) >> 5 = 1040 >> 5
    /// = 32`. Same-sign POCs, no sign flip.
    #[test]
    fn round223_bipred_l1_closer_scales_l0() {
        let mv_l0 = MotionVector::quarter_pel(0, 0);
        let mv_l1 = MotionVector::quarter_pel(0, 0);
        let offset = MotionVector::quarter_pel(64, 0);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 8, 4, true, true).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(32, 0));
        assert_eq!(out_l1, MotionVector::quarter_pel(64, 0));
    }

    /// Eqs. 602-606 — `Abs(L0) < Abs(L1)` ⇒ `mMvdL0 = MmvdOffset`,
    /// `mMvdL1` is scaled by `sf = (|L0| << 5) / |L1|`. With `|L0| = 4`,
    /// `|L1| = 8`, `MmvdOffset = (0, 64)`, scaled y is `((4<<5)/8 * 64
    /// + 16) >> 5 = 32`.
    #[test]
    fn round223_bipred_l0_closer_scales_l1() {
        let mv_l0 = MotionVector::quarter_pel(0, 0);
        let mv_l1 = MotionVector::quarter_pel(0, 0);
        let offset = MotionVector::quarter_pel(0, 64);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 4, 8, true, true).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(0, 64));
        assert_eq!(out_l1, MotionVector::quarter_pel(0, 32));
    }

    /// Eqs. 597-601 followed by 607-610 — asymmetric magnitudes with
    /// opposite POC signs ⇒ the scaled `mMvdL1` is negated.
    /// `|L0| = 8`, `|L1| = 4`, POC signs differ, `MmvdOffset = (64, 0)`:
    /// `mMvdL1 = (64, 0)` (raw offset), then negated to `(-64, 0)`;
    /// `mMvdL0` = scaled = `(32, 0)` (no sign flip on L0).
    #[test]
    fn round223_bipred_l1_closer_opposite_sign_flips_l1_only() {
        let mv_l0 = MotionVector::quarter_pel(0, 0);
        let mv_l1 = MotionVector::quarter_pel(0, 0);
        let offset = MotionVector::quarter_pel(64, 0);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 8, -4, true, true).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(32, 0));
        assert_eq!(out_l1, MotionVector::quarter_pel(-64, 0));
    }

    /// Eqs. 611-612 — only L0 active. `mMvdL0 = MmvdOffset`, `mMvdL1 =
    /// 0`. POC diffs are irrelevant to the "Otherwise" branch.
    #[test]
    fn round223_one_list_active_l0_only() {
        let mv_l0 = MotionVector::quarter_pel(100, 100);
        let mv_l1 = MotionVector::quarter_pel(50, 50);
        let offset = MotionVector::quarter_pel(16, 0);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 0, 0, true, false).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(116, 100));
        assert_eq!(out_l1, MotionVector::quarter_pel(50, 50));
    }

    /// Eqs. 611-612 — only L1 active. Symmetric to L0-only.
    #[test]
    fn round223_one_list_active_l1_only() {
        let mv_l0 = MotionVector::quarter_pel(50, 50);
        let mv_l1 = MotionVector::quarter_pel(100, 100);
        let offset = MotionVector::quarter_pel(0, -8);
        let (out_l0, out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 0, 0, false, true).unwrap();
        assert_eq!(out_l0, MotionVector::quarter_pel(50, 50));
        assert_eq!(out_l1, MotionVector::quarter_pel(100, 92));
    }

    /// Both `predFlagLX == 0` is a caller bug — the §8.5.2.3.9 entry
    /// process never enters eqs. 591-616 with both list flags zero.
    /// Surface `Error::Unsupported`, do not silently return zeroes.
    #[test]
    fn round223_rejects_both_lists_inactive() {
        let mv = MotionVector::default();
        let offset = MotionVector::quarter_pel(1, 0);
        assert!(mmvd_apply_bipred_offset(mv, mv, offset, 1, 1, false, false).is_err());
    }

    /// Eqs. 613-616 — the final accumulation goes through the same
    /// `wrap16` semantics as eqs. 436/439. Verify by pushing `mvL0`
    /// near the positive 16-bit boundary and observing the wrap.
    #[test]
    fn round223_accumulation_wraps_into_signed_16bit() {
        let mv_l0 = MotionVector::quarter_pel(32000, 0);
        let mv_l1 = MotionVector::quarter_pel(0, 0);
        let offset = MotionVector::quarter_pel(1000, 0);
        let (out_l0, _out_l1) =
            mmvd_apply_bipred_offset(mv_l0, mv_l1, offset, 4, 4, true, true).unwrap();
        // 32000 + 1000 = 33000 wraps to -32536 per wrap16.
        assert_eq!(out_l0.x, -32536);
    }

    /// Eqs. 600 / 601 / 605 / 606 — the clip step caps the post-scale
    /// MV component to signed 16-bit. Force a giant scale factor and
    /// confirm the clip engages (rather than overflowing into the wrap
    /// semantics of the eq. 613 accumulation).
    #[test]
    fn round223_scaled_component_clips_to_signed16() {
        // sf = (1 << 5) / 1 = 32; mv component = 8000.
        // raw = (32 * 8000 + 16) >> 5 = 256016 >> 5 = 8000 (fits, identity).
        // We instead pin clip explicitly:
        assert_eq!(clip_mvd_component(40000), 32767);
        assert_eq!(clip_mvd_component(-40000), -32768);
        assert_eq!(clip_mvd_component(0), 0);
    }

    /// `mmvd_scale_component` matches eq. 600's `(sf * mv + 16) >> 5`
    /// form on a worked positive example and on a negative-mv example
    /// (Rust's arithmetic right-shift on signed types makes the result
    /// deterministic on both signs).
    #[test]
    fn round223_scale_component_eq600_form() {
        // (32 * 100 + 16) >> 5 = 3216 >> 5 = 100. Identity sf = 32.
        assert_eq!(mmvd_scale_component(32, 100), 100);
        // (16 * 100 + 16) >> 5 = 1616 >> 5 = 50.5 ⇒ 50 (round-half-up
        // semantics on positive values give 50 here because 1616 = 50*32
        // + 16, the round-half-up of 50.5 gives 51 — but the spec's form
        // is `(x + 16) >> 5` which is "add half, truncate"; with x=1600
        // we get (1600 + 16) >> 5 = 1616 >> 5 = 50).
        assert_eq!(mmvd_scale_component(16, 100), 50);
        // Negative MV: (16 * -100 + 16) >> 5 = (-1584) >> 5 = -50
        // (arithmetic right shift: -1584 / 32 = -49.5 ⇒ -50 floor).
        assert_eq!(mmvd_scale_component(16, -100), -50);
    }

    /// Cross-check: when the bipred branch enters with all-zero
    /// MV inputs and a non-zero offset and symmetric POC magnitudes,
    /// the result equals `(offset, offset)` for same-sign POCs and
    /// `(offset, -offset)` for opposite-sign POCs.
    #[test]
    fn round223_bipred_symmetric_property_offset_distribution() {
        let zero = MotionVector::default();
        for (k, offset) in [
            MotionVector::quarter_pel(1, 0),
            MotionVector::quarter_pel(-1, 0),
            MotionVector::quarter_pel(0, 32),
            MotionVector::quarter_pel(0, -128),
        ]
        .iter()
        .enumerate()
        {
            let (l0, l1) = mmvd_apply_bipred_offset(zero, zero, *offset, 4, 4, true, true).unwrap();
            assert_eq!(l0, *offset, "same-sign k = {k}");
            assert_eq!(l1, *offset, "same-sign k = {k}");

            let (l0, l1) =
                mmvd_apply_bipred_offset(zero, zero, *offset, 4, -4, true, true).unwrap();
            assert_eq!(l0, *offset, "opp-sign k = {k}");
            assert_eq!(
                l1,
                MotionVector {
                    x: -offset.x,
                    y: -offset.y
                },
                "opp-sign k = {k}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Round 229 — §8.5.2.3.9 entry-process signed POC scaling primitives
    // -----------------------------------------------------------------

    /// Eq. 542 / 551 / 559 / 571 / 580 / 588 worked examples. Equal POCs
    /// land at `sf = 32` (`+1` after the `>> 5`). Opposite-sign inputs
    /// yield negative `sf`. Mixed magnitudes match `(num << 5) / den`
    /// with truncation toward zero.
    #[test]
    fn round229_signed_dist_scale_factor_worked_examples() {
        // Symmetric: same POC distance → 32 (which downstream rounds
        // back to ±1 after >>5).
        assert_eq!(mmvd_signed_dist_scale_factor(4, 4).unwrap(), 32);
        // num = 4, den = 8 → ( 4 << 5 ) / 8 = 128 / 8 = 16
        assert_eq!(mmvd_signed_dist_scale_factor(4, 8).unwrap(), 16);
        // num = -4, den = 8 → ( -4 << 5 ) / 8 = -128 / 8 = -16
        assert_eq!(mmvd_signed_dist_scale_factor(-4, 8).unwrap(), -16);
        // num = 4, den = -8 → ( 4 << 5 ) / -8 = 128 / -8 = -16
        assert_eq!(mmvd_signed_dist_scale_factor(4, -8).unwrap(), -16);
        // num = -4, den = -8 → ( -4 << 5 ) / -8 = -128 / -8 = 16
        assert_eq!(mmvd_signed_dist_scale_factor(-4, -8).unwrap(), 16);
        // L0-closer worked example (eq. 559 form): num = 8, den = 4
        // → ( 8 << 5 ) / 4 = 256 / 4 = 64.
        assert_eq!(mmvd_signed_dist_scale_factor(8, 4).unwrap(), 64);
    }

    /// Eq. 542 / 551 / 559 / 571 / 580 / 588 — zero denominator surfaces
    /// `Error::Unsupported`. The entry-process callers guard the zero-
    /// POC case upstream, but a self-protective check is cheap and
    /// makes the helper safe to call directly.
    #[test]
    fn round229_signed_dist_scale_factor_rejects_zero_denominator() {
        assert!(mmvd_signed_dist_scale_factor(4, 0).is_err());
        assert!(mmvd_signed_dist_scale_factor(0, 0).is_err());
        assert!(mmvd_signed_dist_scale_factor(-4, 0).is_err());
    }

    /// Zero numerator with non-zero denominator is well-defined: 0
    /// regardless of `pocDiffDen`. Used by §8.5.2.3.9 sub-branches
    /// where the merge candidate's reference picture is currPic.
    #[test]
    fn round229_signed_dist_scale_factor_zero_numerator() {
        assert_eq!(mmvd_signed_dist_scale_factor(0, 4).unwrap(), 0);
        assert_eq!(mmvd_signed_dist_scale_factor(0, -4).unwrap(), 0);
    }

    /// Eq. 543 / 544 / 552 / 553 / … symmetric round-toward-zero
    /// scaling. `sf = 32`, `v = ±1` ⇒ ±1 (magnitude (1+16)>>5 ... no,
    /// 32*1 = 32, |32|+16 = 48, 48>>5 = 1). `sf = 16`, `v = 2` ⇒
    /// 32 → (|32|+16)>>5 = 1, sign +1 → +1.
    #[test]
    fn round229_signed_scale_component_symmetric_in_sign() {
        assert_eq!(mmvd_signed_scale_component(32, 1), 1);
        assert_eq!(mmvd_signed_scale_component(32, -1), -1);
        assert_eq!(mmvd_signed_scale_component(-32, 1), -1);
        assert_eq!(mmvd_signed_scale_component(-32, -1), 1);

        // sf = 16, v = 2 → product 32, (32+16)>>5 = 1
        assert_eq!(mmvd_signed_scale_component(16, 2), 1);
        assert_eq!(mmvd_signed_scale_component(16, -2), -1);
    }

    /// The form rounds **toward zero** with a half-up bias. `sf=1`,
    /// `v=15` ⇒ product 15, (15+16)>>5 = 31>>5 = 0. `sf=1`, `v=16`
    /// ⇒ product 16, (16+16)>>5 = 32>>5 = 1. The threshold is at
    /// `|product| = 16`.
    #[test]
    fn round229_signed_scale_component_half_up_threshold() {
        assert_eq!(mmvd_signed_scale_component(1, 15), 0);
        assert_eq!(mmvd_signed_scale_component(1, -15), 0);
        // Same threshold on the other axis of the `Sign(x) *` form.
        assert_eq!(mmvd_signed_scale_component(1, 16), 1);
        assert_eq!(mmvd_signed_scale_component(1, -16), -1);
    }

    /// Result is clipped to signed-16-bit. Pick a giant product to
    /// exercise both clamps. `sf = 32767`, `v = 32767` produces a
    /// magnitude well above `i16::MAX`; after the `>>5` reduction it
    /// still exceeds 32767, so the clip engages.
    #[test]
    fn round229_signed_scale_component_clamps_to_signed16() {
        let out = mmvd_signed_scale_component(32767, 32767);
        assert_eq!(out, 32767);
        let out = mmvd_signed_scale_component(32767, -32767);
        assert_eq!(out, -32768);
        let out = mmvd_signed_scale_component(-32767, 32767);
        assert_eq!(out, -32768);
    }

    /// `mmvd_signed_scale_mv` applies the same `distScaleFactor` to
    /// both axes (eqs. 543 + 544 pair, 552 + 553 pair, etc.).
    #[test]
    fn round229_signed_scale_mv_applies_to_both_axes() {
        let mv = MotionVector { x: 16, y: -16 };
        let scaled = mmvd_signed_scale_mv(32, mv);
        // sf=32, v=16 → product 512, (512+16)>>5 = 528>>5 = 16
        // sf=32, v=-16 → product -512, (|−512|+16)>>5 = 16, sign -1 → -16
        assert_eq!(scaled, MotionVector { x: 16, y: -16 });

        // Opposite-sign sf flips both axes (the entry-process branches
        // that ride on a sign-flipped POC distance).
        let scaled = mmvd_signed_scale_mv(-32, mv);
        assert_eq!(scaled, MotionVector { x: -16, y: 16 });
    }

    /// Round-trip property: scaling by `sf = 32` (i.e. `pocDiffNum ==
    /// pocDiffDen`) is `+v` on both axes for non-saturating inputs.
    /// This is the symmetric / "same POC distance" identity the
    /// §8.5.2.3.9 sub-branches reduce to when the L0 / L1 distances
    /// match.
    #[test]
    fn round229_signed_scale_component_unit_factor_identity() {
        for v in [-100, -32, -1, 0, 1, 32, 100, 500] {
            let s = mmvd_signed_scale_component(32, v);
            // sf * v + 16 = 32 * v + 16. For |v| < 1024 the (Abs+16) >>
            // 5 path reduces to |v| exactly when v.abs() % 1 == 0
            // (always); verify it.
            assert_eq!(s, v, "sf=32, v={v}");
        }
    }

    /// Zero-product short-circuit: `sf == 0` or `v == 0` ⇒ 0.
    #[test]
    fn round229_signed_scale_component_zero_inputs() {
        assert_eq!(mmvd_signed_scale_component(0, 32767), 0);
        assert_eq!(mmvd_signed_scale_component(32767, 0), 0);
        assert_eq!(mmvd_signed_scale_component(0, 0), 0);
    }

    /// Eqs. 547 / 576 constant: the §8.5.2.3.9 P-slice / same-target
    /// branch adds (group 1) or subtracts (group 2) 3 from `mMvL0[0]`.
    /// The y component is untouched. The pinned constant catches
    /// future refactors that drift the magnitude.
    #[test]
    fn round229_p_same_target_shift_constant_pinned() {
        assert_eq!(MMVD_P_SAME_TARGET_SHIFT, 3);
    }

    /// Differential property: at the half-up rounding boundary the
    /// symmetric and arithmetic-shift forms can diverge for negative
    /// products. Eq. 600 (round-223 `mmvd_scale_component`) uses
    /// `(s*v + 16) >> 5` (arithmetic right shift). Eq. 543 (round-229
    /// `mmvd_signed_scale_component`) uses `Sign(s*v) * ((|s*v| + 16)
    /// >> 5)`. For `s = 1`, `v = -1` both forms produce 0:
    /// arithmetic-shift form `(-1 + 16) >> 5 = 15 >> 5 = 0`; symmetric
    /// form `Sign(-1) * ((1 + 16) >> 5) = -1 * 0 = 0`. Establish the
    /// agreement-at-zero baseline; the negative-rounding bias only
    /// matters for products where `((|p| - 16) / 32)` and
    /// `((p + 16) / 32)` disagree in sign.
    #[test]
    fn round229_signed_scale_agreement_at_zero_threshold() {
        // Both eq. 600 form (mmvd_scale_component) and eq. 543 form
        // (mmvd_signed_scale_component) agree at the zero crossing.
        assert_eq!(mmvd_signed_scale_component(1, -1), 0);
        assert_eq!(mmvd_scale_component(1, -1), 0);
    }

    // ----------------------------------------------------------------
    // Round 232 — §8.5.2.3.10 motion vector prediction redundancy
    // check
    // ----------------------------------------------------------------

    /// Helper for the §8.5.2.3.10 fixtures: build a candidate with both
    /// lists active and explicit refIdx / MV per list.
    fn mc_bipred(rl0: i32, mvl0: (i32, i32), rl1: i32, mvl1: (i32, i32)) -> MergeCand {
        MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: true,
            ref_idx_l0: rl0,
            ref_idx_l1: rl1,
            mv_l0: MotionVector::quarter_pel(mvl0.0, mvl0.1),
            mv_l1: MotionVector::quarter_pel(mvl1.0, mvl1.1),
        }
    }

    /// Helper for the §8.5.2.3.10 fixtures: build a candidate with only
    /// L0 active. Residual L1 fields are explicitly set to "junk" so
    /// the predicate's L1-skip behaviour is observable.
    fn mc_l0(rl0: i32, mvl0: (i32, i32)) -> MergeCand {
        MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: rl0,
            ref_idx_l1: 999,
            mv_l0: MotionVector::quarter_pel(mvl0.0, mvl0.1),
            mv_l1: MotionVector::quarter_pel(9999, 9999),
        }
    }

    /// §8.5.2.3.10 step (1) — when the two candidates use a different
    /// `predFlag` bitmask, the predicate fails on the first ordered
    /// step. A bipred candidate cannot match an L0-only candidate even
    /// when L0's MV / refIdx coincide.
    #[test]
    fn round232_pred_flag_bitmask_mismatch_blocks_match() {
        let a = mc_bipred(0, (1, 2), 0, (3, 4));
        let b = mc_l0(0, (1, 2));
        assert!(!merge_cand_matches(&a, &b));
        assert!(!merge_cand_matches(&b, &a));
    }

    /// §8.5.2.3.10 step (3) — when both candidates carry the same
    /// `predFlag` bitmask and same MVs, a single-list refIdx
    /// disagreement is enough to make the predicate fail.
    #[test]
    fn round232_ref_idx_mismatch_blocks_match() {
        let a = mc_bipred(0, (1, 2), 0, (3, 4));
        let b = mc_bipred(1, (1, 2), 0, (3, 4));
        assert!(!merge_cand_matches(&a, &b));
    }

    /// §8.5.2.3.10 step (4) — same `predFlag` bitmask and same
    /// refIdxs, but a single MV component differs.
    #[test]
    fn round232_mv_component_mismatch_blocks_match() {
        let a = mc_bipred(0, (1, 2), 0, (3, 4));
        let b = mc_bipred(0, (1, 2), 0, (3, 5));
        assert!(!merge_cand_matches(&a, &b));
    }

    /// §8.5.2.3.10 "corresponding to available reference lists" — when
    /// `predFlagL1 = 0` for both candidates, the spec masks out L1's
    /// refIdx / MV from the compare. Residual stale L1 values do NOT
    /// disturb the match.
    #[test]
    fn round232_inactive_list_fields_are_ignored() {
        let a = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: 5, // junk
            mv_l0: MotionVector::quarter_pel(1, 2),
            mv_l1: MotionVector::quarter_pel(100, 200), // junk
        };
        let b = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -3, // different junk
            mv_l0: MotionVector::quarter_pel(1, 2),
            mv_l1: MotionVector::quarter_pel(-50, 50), // different junk
        };
        assert!(merge_cand_matches(&a, &b));
    }

    /// §8.5.2.3.10 pre-test — when `numCurrMergeCand ≤ 1` the routine
    /// returns the count untouched (the spec's outer "When
    /// numCurrMergeCand is greater than 1" guard).
    #[test]
    fn round232_pre_test_no_op_when_count_le_1() {
        let list = vec![mc_l0(0, (1, 2)); 4];
        assert_eq!(merge_cand_redundancy_check(&list, 0).unwrap(), 0);
        assert_eq!(merge_cand_redundancy_check(&list, 1).unwrap(), 1);
    }

    /// §8.5.2.3.10 happy path — tail entry duplicates an earlier
    /// entry, count is decremented by 1.
    #[test]
    fn round232_duplicate_tail_drops_count() {
        let mut list = vec![MergeCand::default(); 6];
        list[0] = mc_l0(0, (1, 2));
        list[1] = mc_l0(1, (3, 4));
        list[2] = mc_l0(2, (5, 6));
        list[3] = mc_l0(1, (3, 4)); // duplicates index 1
        let out = merge_cand_redundancy_check(&list, 4).unwrap();
        assert_eq!(out, 3);
    }

    /// §8.5.2.3.10 happy path — tail entry is genuinely new, count is
    /// preserved.
    #[test]
    fn round232_new_tail_preserves_count() {
        let mut list = vec![MergeCand::default(); 6];
        list[0] = mc_l0(0, (1, 2));
        list[1] = mc_l0(1, (3, 4));
        list[2] = mc_l0(2, (5, 6));
        list[3] = mc_l0(3, (7, 8));
        let out = merge_cand_redundancy_check(&list, 4).unwrap();
        assert_eq!(out, 4);
    }

    /// §8.5.2.3.10 scan ordering — when both index 0 AND index 1
    /// would match the tail, the spec stops at the FIRST duplicate
    /// (candIndx == 0 path). Verify the loop exits before reaching
    /// index 1 by mutating index 1 to a different value: the count
    /// is still decremented exactly once (count - 1).
    #[test]
    fn round232_first_duplicate_short_circuits_scan() {
        let mut list = vec![MergeCand::default(); 6];
        list[0] = mc_l0(0, (1, 2));
        list[1] = mc_l0(1, (3, 4));
        list[2] = mc_l0(2, (5, 6));
        list[3] = mc_l0(0, (1, 2)); // matches list[0]
        let out = merge_cand_redundancy_check(&list, 4).unwrap();
        // Decrement by 1 (single-pass behaviour), not 2.
        assert_eq!(out, 3);
    }

    /// §8.5.2.3.10 boundary — when only the entry directly before the
    /// tail matches, the spec's exit predicate (`candIndx ==
    /// numCurrMergeCand - 2`) still admits the compare; the routine
    /// correctly decrements.
    #[test]
    fn round232_penultimate_duplicate_decrements() {
        let mut list = vec![MergeCand::default(); 6];
        list[0] = mc_l0(0, (1, 2));
        list[1] = mc_l0(1, (3, 4));
        list[2] = mc_l0(2, (5, 6));
        list[3] = mc_l0(2, (5, 6)); // matches index 2 only
        let out = merge_cand_redundancy_check(&list, 4).unwrap();
        assert_eq!(out, 3);
    }

    /// §8.5.2.3.10 `numCurrMergeCand == 2` smallest non-trivial case
    /// — duplicate.
    #[test]
    fn round232_two_element_duplicate() {
        let list = vec![mc_l0(0, (1, 2)), mc_l0(0, (1, 2))];
        let out = merge_cand_redundancy_check(&list, 2).unwrap();
        assert_eq!(out, 1);
    }

    /// §8.5.2.3.10 `numCurrMergeCand == 2` distinct case.
    #[test]
    fn round232_two_element_distinct() {
        let list = vec![mc_l0(0, (1, 2)), mc_l0(0, (1, 3))];
        let out = merge_cand_redundancy_check(&list, 2).unwrap();
        assert_eq!(out, 2);
    }

    /// §8.5.2.3.10 — bipred duplicate where both L0 and L1 must agree
    /// for the predicate to fire.
    #[test]
    fn round232_bipred_full_match_drops_count() {
        let mut list = vec![MergeCand::default(); 4];
        list[0] = mc_bipred(0, (1, 2), 1, (3, 4));
        list[1] = mc_bipred(2, (5, 6), 3, (7, 8));
        list[2] = mc_bipred(0, (1, 2), 1, (3, 4)); // matches index 0
        let out = merge_cand_redundancy_check(&list, 3).unwrap();
        assert_eq!(out, 2);
    }

    /// §8.5.2.3.10 — bipred near-match where ONLY L1 differs. The
    /// predicate must reject the partial match.
    #[test]
    fn round232_bipred_l1_only_difference_preserves_count() {
        let mut list = vec![MergeCand::default(); 4];
        list[0] = mc_bipred(0, (1, 2), 1, (3, 4));
        list[1] = mc_bipred(2, (5, 6), 3, (7, 8));
        list[2] = mc_bipred(0, (1, 2), 1, (3, 5)); // L1 mvy differs
        let out = merge_cand_redundancy_check(&list, 3).unwrap();
        assert_eq!(out, 3);
    }

    /// §8.5.2.3.10 — caller bug: `num_curr_merge_cand` exceeds buffer.
    #[test]
    fn round232_oversize_count_errors() {
        let list = vec![mc_l0(0, (1, 2)); 2];
        let err = merge_cand_redundancy_check(&list, 3).unwrap_err();
        match err {
            Error::Unsupported(msg) => {
                assert!(msg.contains("§8.5.2.3.10"));
            }
            _ => panic!("expected Error::Unsupported"),
        }
    }

    /// §8.5.2.3.10 — reflexive identity: the matching predicate is
    /// reflexive (a candidate always matches itself), as it must be
    /// for the trim loop to ever fire.
    #[test]
    fn round232_predicate_reflexive() {
        let cand = mc_bipred(2, (-10, 7), -1, (300, -300));
        assert!(merge_cand_matches(&cand, &cand));
        let cand_l0 = mc_l0(5, (1, 2));
        assert!(merge_cand_matches(&cand_l0, &cand_l0));
    }

    /// §8.5.2.3.10 — symmetric: `matches(a, b) == matches(b, a)`. The
    /// spec's "have all the following conditions met" wording defines
    /// an equivalence relation on the active-list-restricted
    /// projection, so the predicate must be symmetric even in the
    /// inactive-list-ignored case.
    #[test]
    fn round232_predicate_symmetric() {
        let a = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: 5,
            mv_l0: MotionVector::quarter_pel(1, 2),
            mv_l1: MotionVector::quarter_pel(99, 99),
        };
        let b = MergeCand {
            pred_flag_l0: true,
            pred_flag_l1: false,
            ref_idx_l0: 0,
            ref_idx_l1: -3,
            mv_l0: MotionVector::quarter_pel(1, 2),
            mv_l1: MotionVector::quarter_pel(-1, -1),
        };
        assert_eq!(merge_cand_matches(&a, &b), merge_cand_matches(&b, &a));
        assert!(merge_cand_matches(&a, &b));
    }

    // --- eqs. 923/924 + 932/933 subblock padding clamp -------------------

    /// Build a 32×32 reference with a horizontal luma ramp (y = 4·x) and
    /// distinct flat chroma.
    fn ramp_ref<'a>(
        y: &'a mut Vec<u8>,
        cb: &'a mut Vec<u8>,
        cr: &'a mut Vec<u8>,
    ) -> RefPictureView<'a> {
        *y = (0..32 * 32).map(|i| ((i % 32) * 4) as u8).collect();
        *cb = (0..16 * 16).map(|i| ((i % 16) * 8) as u8).collect();
        *cr = vec![77u8; 16 * 16];
        RefPictureView {
            y,
            cb,
            cr,
            width: 32,
            height: 32,
            y_stride: 32,
            c_stride: 16,
            chroma_format_idc: 1,
        }
    }

    /// With the anchor placed at the fetch MV itself (`mvOffset == 0`),
    /// the eqs.-923/924 window covers every fetched position — the padded
    /// interpolation is byte-identical to the unpadded one, including a
    /// fractional phase (all 8 taps live inside `[xSbIntL−3, +sbW+3]`).
    #[test]
    fn padded_luma_identity_when_anchor_matches_mv() {
        let (mut y, mut cb, mut cr) = (Vec::new(), Vec::new(), Vec::new());
        let refp = ramp_ref(&mut y, &mut cb, &mut cr);
        let mv = MotionVector {
            x: (2 << 4) + 5,
            y: 1 << 4,
        }; // +2 int +5/16 frac, +1 int
        let mut plain = vec![0i32; 8 * 8];
        let mut padded = vec![0i32; 8 * 8];
        interpolate_luma_block_main(refp, 8, 8, mv, 8, 8, 8, &mut plain).unwrap();
        let pad = PadAnchor {
            x_sb_int: 8 + (mv.x >> 4),
            y_sb_int: 8 + (mv.y >> 4),
        };
        interpolate_luma_block_main_padded(refp, 8, 8, mv, 8, 8, 8, pad, &mut padded).unwrap();
        assert_eq!(plain, padded);
    }

    /// A refined MV that walks +6 integer pels right of the anchor MV has
    /// its right-edge fetches clamped to `xSbIntL + sbWidth + 3` — the
    /// clamped columns replicate the window-edge sample instead of
    /// reading fresh reference columns (eq. 923).
    #[test]
    fn padded_luma_clamps_beyond_anchor_window() {
        let (mut y, mut cb, mut cr) = (Vec::new(), Vec::new(), Vec::new());
        let refp = ramp_ref(&mut y, &mut cb, &mut cr);
        // 4×4 block at (8, 8). Anchor MV = 0 → window x ∈ [5, 15].
        let pad = PadAnchor {
            x_sb_int: 8,
            y_sb_int: 8,
        };
        // Refined MV = +6 integer pels → raw fetches x = 14..17; the last
        // two columns clamp to 15.
        let mv = MotionVector { x: 6 << 4, y: 0 };
        let mut out = vec![0i32; 4 * 4];
        interpolate_luma_block_main_padded(refp, 8, 8, mv, 4, 4, 8, pad, &mut out).unwrap();
        // Columns 0/1 fetch x = 14/15 (in-window); columns 2/3 clamp to 15.
        assert_eq!(out[0], 14 * 4);
        assert_eq!(out[1], 15 * 4);
        assert_eq!(out[2], 15 * 4, "eq. 923 clamps to xSbIntL+sbW+3");
        assert_eq!(out[3], 15 * 4);
        // Unpadded reads the genuine columns 16/17.
        let mut plain = vec![0i32; 4 * 4];
        interpolate_luma_block_main(refp, 8, 8, mv, 4, 4, 8, &mut plain).unwrap();
        assert_eq!(plain[2], 16 * 4);
        assert_eq!(plain[3], 17 * 4);
    }

    /// Chroma eq.-932 window is only ±1 around the anchored subblock: a
    /// +3-chroma-sample refined fetch clamps every column past
    /// `xSbIntC + sbWidth + 1` to the window edge.
    #[test]
    fn padded_chroma_clamps_beyond_anchor_window() {
        let (mut y, mut cb, mut cr) = (Vec::new(), Vec::new(), Vec::new());
        let refp = ramp_ref(&mut y, &mut cb, &mut cr);
        // 4×4 chroma block at (4, 4). Anchor MV = 0 → window x ∈ [3, 9].
        let pad = PadAnchor {
            x_sb_int: 4,
            y_sb_int: 4,
        };
        // Refined chroma MV = +3 integer chroma pels (1/32 units).
        let mv_c = MotionVector { x: 3 << 5, y: 0 };
        let mut out = vec![0i32; 4 * 4];
        interpolate_chroma_block_main_padded(refp, 1, 4, 4, mv_c, 4, 4, 8, pad, &mut out).unwrap();
        // Raw fetches x = 7..10; the last column clamps to 9 (= 9·8 = 72).
        assert_eq!(out[0], 7 * 8);
        assert_eq!(out[1], 8 * 8);
        assert_eq!(out[2], 9 * 8);
        assert_eq!(out[3], 9 * 8, "eq. 932 clamps to xSbIntC+sbW+1");
    }
}
