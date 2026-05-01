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

fn baseline_luma_phase_supported(phase: u32) -> bool {
    matches!(phase, 0 | 4 | 8 | 12)
}

fn baseline_chroma_phase_supported(phase: u32) -> bool {
    matches!(phase, 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28)
}

/// Sample one luma reference position, clipping coordinates to the
/// reference picture extent.  Implements the `Clip3(0, picW − 1, …)` step
/// in eq. 921/922.
fn sample_luma_clipped(refp: RefPictureView<'_>, x: i32, y: i32) -> i32 {
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

/// Interpolate a luma block of size `sb_width × sb_height` at the given
/// `mv` (1/16-pel) anchored at (`x_sb`, `y_sb`) in the current picture
/// coordinate space. Outputs are i32 samples, clipped to `[0, 2^bd-1]`.
///
/// Implements the §8.5.4.3.2 luma interpolation. Round-4 only supports
/// the Baseline-table (Table 25) phases — non-baseline phases return
/// `Error::Unsupported`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
    if out.len() != sb_width * sb_height {
        return Err(Error::invalid("evc inter: luma out buffer size mismatch"));
    }
    // §8.5.4.3.1 eq. 913–916: integer sample is mv >> 4, fractional is mv & 15.
    let int_x = mv.x >> 4;
    let int_y = mv.y >> 4;
    let frac_x = (mv.x & 15) as u32;
    let frac_y = (mv.y & 15) as u32;
    if !baseline_luma_phase_supported(frac_x) || !baseline_luma_phase_supported(frac_y) {
        return Err(Error::unsupported(format!(
            "evc inter: round-4 luma phase ({frac_x},{frac_y}) outside Baseline 1/4-pel grid"
        )));
    }
    let max_val = (1i32 << bit_depth) - 1;
    let shift1 = (4u32).min(bit_depth.saturating_sub(8));
    let shift2 = (8u32).max(20u32.saturating_sub(bit_depth));
    let offset2: i32 = 1 << (shift2 - 1);
    if frac_x == 0 && frac_y == 0 {
        // Integer sample copy (eq. 925).
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let s =
                    sample_luma_clipped(refp, x_sb + int_x + xl as i32, y_sb + int_y + yl as i32);
                out[yl * sb_width + xl] = s;
            }
        }
        return Ok(());
    }
    let fx = LUMA_FILTER_TABLE25[frac_x as usize];
    let fy = LUMA_FILTER_TABLE25[frac_y as usize];
    if frac_y == 0 {
        // Horizontal-only filter (eq. 926).
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let bx = x_sb + int_x + xl as i32;
                let by = y_sb + int_y + yl as i32;
                let mut acc: i32 = 0;
                for i in 0..8 {
                    let s = sample_luma_clipped(refp, bx + i as i32 - 3, by);
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
                    let s = sample_luma_clipped(refp, bx, by + i as i32 - 3);
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
                let s = sample_luma_clipped(refp, bx + i as i32 - 3, by);
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
/// (`x_sb_c`, `y_sb_c`). Implements §8.5.4.3.3.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
    if out.len() != sb_width * sb_height {
        return Err(Error::invalid("evc inter: chroma out buffer size mismatch"));
    }
    let int_x = mv_c.x >> 5;
    let int_y = mv_c.y >> 5;
    let frac_x = (mv_c.x & 31) as u32;
    let frac_y = (mv_c.y & 31) as u32;
    if !baseline_chroma_phase_supported(frac_x) || !baseline_chroma_phase_supported(frac_y) {
        return Err(Error::unsupported(format!(
            "evc inter: round-4 chroma phase ({frac_x},{frac_y}) outside Baseline 1/8-pel grid"
        )));
    }
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
                    x_sb_c + int_x + xl as i32,
                    y_sb_c + int_y + yl as i32,
                );
                out[yl * sb_width + xl] = s;
            }
        }
        return Ok(());
    }
    let fx = CHROMA_FILTER_TABLE27[frac_x as usize];
    let fy = CHROMA_FILTER_TABLE27[frac_y as usize];
    if frac_y == 0 {
        for yl in 0..sb_height {
            for xl in 0..sb_width {
                let bx = x_sb_c + int_x + xl as i32;
                let by = y_sb_c + int_y + yl as i32;
                let mut acc: i32 = 0;
                for i in 0..4 {
                    let s = sample_chroma_clipped(refp, c_idx, bx + i as i32 - 1, by);
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
                    let s = sample_chroma_clipped(refp, c_idx, bx, by + i as i32 - 1);
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
                let s = sample_chroma_clipped(refp, c_idx, bx + i as i32 - 1, by);
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
}
