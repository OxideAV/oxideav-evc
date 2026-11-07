//! Encoder **adaptive quantisation** (round 483): a per-CU luma QP
//! chosen from the source picture's local activity, signalled through
//! the §7.3.8.5 `cu_qp_delta_abs` / `cu_qp_delta_sign_flag` pair under
//! `cu_qp_delta_enabled_flag = 1` (PPS) with `sps_dquant_flag = 0`.
//!
//! ## Signalling (§7.3.8.5, §7.4.9.5, §8.7.1)
//!
//! With `sps_dquant_flag == 0` the delta is present in every
//! `transform_unit()` that carries a non-zero CBF (`cbf_luma || cbf_cb
//! || cbf_cr`), right after `cbf_luma` and ahead of the ATS group. The
//! decoder folds it into the eq. 1042 chain
//! `QpY = ( QpY_PREV + CuQpDelta + 52 ) % 52`, `QpY_PREV` being the
//! `QpY` of the previous CU in decoding order (`slice_qp` at the start
//! of the tile). A CU without residual reads no delta (`CuQpDelta`
//! inferred 0) and so takes `QpY_PREV` — its QP only matters to the
//! §8.8.2 deblocking of its edges, and the encoder stamps exactly that
//! value. The delta written is the representative of
//! `target − QpY_PREV` modulo 52 inside the §7.4.9.5 range −26..=25.
//! `cu_qp_delta_abs` is U-binarised with every bin on ctxInc 0 of
//! Table 78 (Table 95), the sign a bypass bin (Table 92 lists it
//! `bypass`).
//!
//! ## The policy
//!
//! The picture is cut into 8×8 luma units; each unit's activity is
//! `A = log2( σ² + 29 )` of its source samples (σ² the variance, scaled
//! to the 8-bit range so the policy is bit-depth invariant; the floor
//! of 29 — half the SSIM `C2 = (0.03 · 255)²` — keeps near-flat units,
//! whose faint noise no viewer sees, from pulling the QP down to code
//! that noise). A unit's
//! QP offset is `s · ( A − Ā )` — `Ā` the picture mean of `A`, `s` the
//! strength in QP per doubling of the variance — clamped to
//! `±MAX_OFFSET`. Busy texture (where coding error is masked) is
//! quantised coarser, flat areas (where it shows as banding and
//! blocking) finer; the zero-mean offsets keep the picture's rate near
//! that of the plain slice QP. A CU takes the rounded mean offset of
//! the units it covers (the unit holding it when it is smaller than
//! one). Predicted (P / B) pictures run at half the strength: their
//! quality distribution is largely inherited from the references, and
//! a full-strength spread there re-codes per-frame noise in the flat
//! areas the intra picture already settled (measured: the full spread
//! loses SSIM BD-rate on P GOPs of the mixed corpus, half gains it).
//! Everything is integer arithmetic — the `log2` is the exact
//! repeated-squaring expansion — so the chosen QPs, and the stream, are
//! identical on every platform.

use crate::cabac::BinSink;
use crate::cabac_init::{CtxSel, MainCtxTable};
use crate::deblock::SideInfoGrid;
use crate::picture::YuvPicture;

/// log2 of the activity unit (8×8 luma samples).
const UNIT_LOG2: u32 = 3;
/// The largest magnitude of a unit's QP offset.
pub const MAX_OFFSET: i32 = 8;
/// Variance floor of the activity measure (8-bit units).
const ACTIVITY_FLOOR: u64 = 29;
/// Fractional bits of the fixed-point activity / offset arithmetic.
const FRAC: u32 = 16;

/// `log2( v )` in Q16 for `v ≥ 1`: the integer part from the leading
/// bit, the 16 fractional bits by repeated squaring of the normalised
/// mantissa (each squaring that reaches 2 yields a 1 bit).
fn log2_q16(v: u64) -> i64 {
    debug_assert!(v >= 1);
    let e = 63 - v.leading_zeros();
    // Mantissa in Q30, in [1, 2).
    let mut m: u64 = if e >= 30 {
        v >> (e - 30)
    } else {
        v << (30 - e)
    };
    let mut frac = 0i64;
    for bit in (0..FRAC).rev() {
        m = (m * m) >> 30;
        if m >= 2 << 30 {
            m >>= 1;
            frac |= 1 << bit;
        }
    }
    (i64::from(e) << FRAC) | frac
}

/// A picture's per-unit QP offsets (Q16) — see the module doc.
#[derive(Clone, Debug)]
pub struct AqMap {
    cols: usize,
    rows: usize,
    pic_w: u32,
    pic_h: u32,
    offsets_q16: Vec<i64>,
}

impl AqMap {
    /// Build the map of `src` at `strength_q8` (QP per doubling of the
    /// variance, Q8; zero gives an all-zero map — every CU at the
    /// slice QP, still signalling its delta).
    pub fn build(src: &YuvPicture, strength_q8: u32) -> Self {
        let unit = 1usize << UNIT_LOG2;
        let cols = (src.width as usize).div_ceil(unit);
        let rows = (src.height as usize).div_ceil(unit);
        let stride = src.y_stride();
        let shift = 2 * (src.bit_depth.saturating_sub(8));
        let mut act = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                let (x0, y0) = (c * unit, r * unit);
                let x1 = (x0 + unit).min(src.width as usize);
                let y1 = (y0 + unit).min(src.height as usize);
                let (mut s, mut s2, mut n) = (0u64, 0u64, 0u64);
                for y in y0..y1 {
                    for &v in &src.y[y * stride + x0..y * stride + x1] {
                        s += u64::from(v);
                        s2 += u64::from(v) * u64::from(v);
                        n += 1;
                    }
                }
                // n² σ² = n Σv² − (Σv)², exact in integers.
                let var_n2 = n * s2 - s * s;
                let var = (var_n2 / (n * n)) >> shift;
                act.push(log2_q16(var + ACTIVITY_FLOOR));
            }
        }
        let mean = act.iter().sum::<i64>() / act.len() as i64;
        let lim = i64::from(MAX_OFFSET) << FRAC;
        let offsets_q16 = act
            .iter()
            .map(|&a| ((i64::from(strength_q8) * (a - mean)) >> 8).clamp(-lim, lim))
            .collect();
        Self {
            cols,
            rows,
            pic_w: src.width,
            pic_h: src.height,
            offsets_q16,
        }
    }

    /// The QP offset of the CU at `( x0, y0 )` of size
    /// `( 1 << lw ) × ( 1 << lh )`: the mean of the offsets of the units
    /// it covers (inside the picture), rounded half away from zero.
    pub fn cu_offset(&self, x0: u32, y0: u32, lw: u32, lh: u32) -> i32 {
        let c0 = (x0 >> UNIT_LOG2) as usize;
        let r0 = (y0 >> UNIT_LOG2) as usize;
        let x1 = (x0 + (1 << lw)).min(self.pic_w);
        let y1 = (y0 + (1 << lh)).min(self.pic_h);
        let c1 = ((x1 as usize).div_ceil(1 << UNIT_LOG2)).clamp(c0 + 1, self.cols);
        let r1 = ((y1 as usize).div_ceil(1 << UNIT_LOG2)).clamp(r0 + 1, self.rows);
        let (mut sum, mut n) = (0i64, 0i64);
        for r in r0..r1 {
            for c in c0..c1 {
                sum += self.offsets_q16[r * self.cols + c];
                n += 1;
            }
        }
        let half = 1i64 << (FRAC - 1);
        let mean = sum / n.max(1);
        let rounded = if mean >= 0 {
            (mean + half) >> FRAC
        } else {
            -((-mean + half) >> FRAC)
        };
        rounded as i32
    }

    /// The target `QpY` of a CU at slice QP `slice_qp` (0..=51).
    pub fn cu_qp(&self, slice_qp: i32, x0: u32, y0: u32, lw: u32, lh: u32) -> i32 {
        (slice_qp + self.cu_offset(x0, y0, lw, lh)).clamp(0, 51)
    }
}

/// The §7.4.9.5 `CuQpDelta` (−26..=25) that moves the eq. 1042 chain
/// from `qp_prev` to `target` (both 0..=51).
pub fn wrap_delta(target: i32, qp_prev: i32) -> i32 {
    (target - qp_prev + 26).rem_euclid(52) - 26
}

/// Write the §7.3.8.5 `cu_qp_delta_abs` (U, Table 78 ctxInc 0 on
/// every bin) and, for a non-zero magnitude, the bypass
/// `cu_qp_delta_sign_flag` (1 = negative, eq. 148 per errata #238 (b))
/// — the dual of the decoder's read.
pub fn emit_cu_qp_delta<S: BinSink>(enc: &mut S, sel: CtxSel, delta: i32) {
    debug_assert!((-26..=25).contains(&delta));
    let (t, i) = sel.ctx(MainCtxTable::CuQpDeltaAbs, 0);
    enc.encode_u_regular_capped(delta.unsigned_abs(), 63, t, |_| i);
    if delta != 0 {
        enc.encode_bypass(u8::from(delta < 0));
    }
}

/// Overwrite the `qp_y` of every 4×4 cell of a CU already stamped into
/// `grid` (the §8.8.2.3 step-3 deblocking input) with its eq. 1042
/// `QpY`.
pub fn restamp_qp(grid: &mut SideInfoGrid, x0: u32, y0: u32, lw: u32, lh: u32, qp_y: i32) {
    let c0 = (x0 >> 2) as usize;
    let r0 = (y0 >> 2) as usize;
    let c1 = (((x0 + (1 << lw)) >> 2) as usize).min(grid.w_cells);
    let r1 = (((y0 + (1 << lh)) >> 2) as usize).min(grid.h_cells);
    for r in r0..r1 {
        for c in c0..c1 {
            grid.at_mut(c, r).qp_y = qp_y.clamp(0, 51) as u8;
        }
    }
}

/// The decode-order eq. 1042 chain as the encoder tracks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QpChain {
    /// `QpY_PREV`.
    pub prev: i32,
}

impl QpChain {
    pub fn new(slice_qp: i32) -> Self {
        Self { prev: slice_qp }
    }

    /// The delta a CU coded at `target` signals (when its TU carries a
    /// CBF), or `None` when no delta is present.
    pub fn delta(&self, enabled: bool, target: i32, cbf_any: bool) -> Option<i32> {
        (enabled && cbf_any).then(|| wrap_delta(target, self.prev))
    }

    /// The `QpY` a CU coded at `target` ends up with (the chain left
    /// unchanged) — see [`Self::advance`].
    pub fn qp_y(&self, enabled: bool, target: i32, cbf_any: bool) -> i32 {
        if enabled && cbf_any {
            target
        } else {
            self.prev
        }
    }

    /// Advance past a CU: its `QpY` — `target` when it carried a delta,
    /// `QpY_PREV` otherwise.
    pub fn advance(&mut self, enabled: bool, target: i32, cbf_any: bool) -> i32 {
        if enabled && cbf_any {
            self.prev = target;
        }
        self.prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{CabacEncoder, CabacEngine, InitType};

    #[test]
    fn log2_q16_is_exact_on_powers_and_monotone() {
        for e in 0..40u32 {
            assert_eq!(log2_q16(1u64 << e), i64::from(e) << 16);
        }
        // log2(3) = 1.5849625… → Q16 103872.
        assert_eq!(log2_q16(3) >> 4, 103_872 >> 4);
        let mut last = -1;
        for v in 1..5000u64 {
            let l = log2_q16(v);
            assert!(l > last, "monotone at {v}");
            last = l;
        }
    }

    #[test]
    fn wrap_delta_hits_every_target_in_range() {
        for prev in 0..52 {
            for target in 0..52 {
                let d = wrap_delta(target, prev);
                assert!((-26..=25).contains(&d));
                assert_eq!((prev + d + 52).rem_euclid(52), target, "eq. 1042");
            }
        }
    }

    /// The writer is the dual of the decoder's U + bypass read on both
    /// entropy shapes.
    #[test]
    fn cu_qp_delta_reads_back() {
        for cm_init in [false, true] {
            let sel = CtxSel::new(cm_init, InitType::Pb);
            let mut enc = CabacEncoder::new();
            if cm_init {
                enc.init_main_profile(InitType::Pb, 30);
            }
            let deltas: Vec<i32> = (-26..=25).chain([0, 0, 3, -3, 25, -26]).collect();
            for &d in &deltas {
                emit_cu_qp_delta(&mut enc, sel, d);
            }
            enc.encode_terminate(true);
            let bytes = enc.finish();
            let mut eng = CabacEngine::new(&bytes).unwrap();
            if cm_init {
                crate::cabac_init::init_main_profile_contexts(&mut eng, InitType::Pb, 30).unwrap();
            }
            for &d in &deltas {
                let (t, i) = sel.ctx(MainCtxTable::CuQpDeltaAbs, 0);
                let abs = eng.decode_u_regular(t, |_| i).unwrap() as i32;
                let got = if abs > 0 && eng.decode_bypass().unwrap() != 0 {
                    -abs
                } else {
                    abs
                };
                assert_eq!(got, d, "cm_init {cm_init}");
            }
        }
    }

    #[test]
    fn map_raises_busy_and_lowers_flat_units() {
        let (w, h) = (64u32, 32u32);
        let mut pic = YuvPicture::new(w, h, 1, 8).unwrap();
        let mut s = 1u32;
        for y in 0..h as usize {
            for x in 0..w as usize {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Left half flat, right half ±40 noise.
                pic.y[y * w as usize + x] = if x < 32 {
                    100
                } else {
                    (100 + (s >> 24) % 81) as u16 - 40
                };
            }
        }
        let map = AqMap::build(&pic, 256);
        let flat = map.cu_offset(0, 0, 4, 4);
        let busy = map.cu_offset(32, 0, 4, 4);
        assert!(flat < 0 && busy > 0, "flat {flat} busy {busy}");
        assert!(flat >= -MAX_OFFSET && busy <= MAX_OFFSET);
        // A CU straddling both takes the mean.
        let both = map.cu_offset(0, 0, 6, 5);
        assert!(both > flat && both < busy);
        assert_eq!(map.cu_qp(50, 32, 0, 4, 4), (50 + busy).min(51));
        assert_eq!(AqMap::build(&pic, 0).cu_offset(32, 0, 4, 4), 0);
    }

    #[test]
    fn chain_tracks_eq_1042() {
        let mut c = QpChain::new(30);
        assert_eq!(c.delta(true, 34, true), Some(4));
        assert_eq!(c.delta(true, 34, false), None);
        assert_eq!(c.delta(false, 34, true), None);
        assert_eq!(c.advance(true, 34, false), 30, "no residual keeps QpY_PREV");
        assert_eq!(c.advance(true, 34, true), 34);
        assert_eq!(c.advance(true, 20, true), 20);
    }
}
