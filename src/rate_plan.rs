//! Two-pass rate control — the first pass's per-frame statistics, the
//! per-frame QP plan a second pass derives from them under a buffer
//! model, and the deterministic `2^(x/6)` arithmetic both share with
//! the one-pass controller.
//!
//! ## Model
//!
//! Every frame `i` the first pass coded at `qp_i` with `bits_i` bits;
//! the second pass predicts `bits_i( q ) = bits_i · 2^( −s · ( q − qp_i ) )`
//! for any `q`, with one sequence-wide **slope** `s` (bits halve every
//! `1 / s` QP steps). The classic law has `s = 1/6`; on this crate's
//! 24-QP-finer quantizer scale ([`crate::slice_enc::rd_lambda`]) the
//! corpus measures `s ≈ 0.07` at the 40-55 dB operating points
//! ([`DEFAULT_SLOPE`]), and the second pass **refines `s` online**:
//! every frame it codes at a QP different from the first pass's is a
//! direct slope observation for that very frame (`log2( bits_1 /
//! bits_2 ) / ( qp_2 − qp_1 )`), averaged into the estimate and used to
//! re-plan the frames still to come.
//!
//! ## Plan
//!
//! Under a total budget with that law, every frame at the **same** QP
//! minimises the summed distortion (the marginal bit cost of a QP step
//! is the same everywhere), so the plan starts from one sequence-wide
//! fractional `q_base` found by bisection so that the modelled total
//! hits `bitrate · frames / fps`. Per frame the integer QP dithers
//! between `⌊q_base⌋` and `⌈q_base⌉` so the cumulative modelled bits
//! track the fractional target, and a leaky-bucket **buffer** walk
//! bends the plan where the flat allocation would break the buffer:
//! a frame that would underflow the buffer (more bits than have
//! arrived) is raised in QP until it fits. A surplus that would
//! overflow the buffer is left unused (the channel idles) rather than
//! forced into the picture — the flat allocation is what minimises
//! distortion for the budget. The buffer holds `vbv` bits, starts
//! full, and is fed `bitrate / fps` bits per frame.
//!
//! The second-pass encoder re-plans before every frame over the frames
//! still to come, with the bits still allowed, the buffer as it
//! actually stands and the refined slope — so first-pass model errors
//! are paid off over what is left rather than at the end. All of it is
//! pure arithmetic on the `2^( x / 6 )` ladder (logarithms by
//! bisection on that ladder), hence platform-deterministic.

use oxideav_core::{Error, Result};

/// One first-pass frame: key picture or not, the QP it was coded at,
/// and its access-unit size in bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStat {
    pub idr: bool,
    pub qp: i32,
    pub bits: f64,
}

/// Header line of the stats text.
const STATS_HEADER: &str = "# oxideav-evc first-pass stats v1";

/// Serialise first-pass statistics (one `idr qp bits` line per frame).
pub fn format_stats(stats: &[FrameStat]) -> String {
    let mut out = String::from(STATS_HEADER);
    out.push('\n');
    for s in stats {
        out.push_str(&format!("{} {} {}\n", u8::from(s.idr), s.qp, s.bits as u64));
    }
    out
}

/// Parse [`format_stats`] output.
pub fn parse_stats(text: &str) -> Result<Vec<FrameStat>> {
    let mut lines = text.lines();
    match lines.next() {
        Some(h) if h.trim() == STATS_HEADER => {}
        _ => {
            return Err(Error::invalid(
                "evc two-pass: stats file lacks the `# oxideav-evc first-pass stats v1` header",
            ))
        }
    }
    let mut out = Vec::new();
    for (n, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let bad = || Error::invalid(format!("evc two-pass: stats line {} malformed", n + 2));
        let idr = match it.next().ok_or_else(bad)? {
            "0" => false,
            "1" => true,
            _ => return Err(bad()),
        };
        let qp: i32 = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
        let bits: u64 = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
        if !(0..=51).contains(&qp) {
            return Err(bad());
        }
        out.push(FrameStat {
            idr,
            qp,
            bits: bits as f64,
        });
    }
    Ok(out)
}

/// `2^( qp / 6 )` for an integer `qp` (negative allowed), evaluated
/// deterministically — an exact power of two times a sixth-root-of-two
/// constant — so rate decisions are byte-identical on every platform.
pub fn pow2_qp_over6(qp: i32) -> f64 {
    const SIXTH: [f64; 6] = [
        1.0,
        1.122_462_048_309_373,
        1.259_921_049_894_873_2,
        std::f64::consts::SQRT_2,
        1.587_401_051_968_199_4,
        1.781_797_436_280_678_6,
    ];
    2f64.powi(qp.div_euclid(6)) * SIXTH[qp.rem_euclid(6) as usize]
}

/// `2^( x / 6 )` for a fractional `x`: the integer ladder above times
/// `2^r` for the remainder `r ∈ [0, 1/6)`, the latter from the first
/// terms of its power series (error < 1e−9 over that interval) — pure
/// arithmetic, hence deterministic.
pub fn pow2_frac_over6(x: f64) -> f64 {
    let k = x.floor();
    let r = (x - k) / 6.0; // ∈ [0, 1/6)
    let a = std::f64::consts::LN_2 * r;
    let poly =
        1.0 + a + a * a / 2.0 + a * a * a / 6.0 + a * a * a * a / 24.0 + a * a * a * a * a / 120.0;
    pow2_qp_over6(k as i32) * poly
}

/// `log2( ratio )` for `ratio > 0` by bisection on [`pow2_frac_over6`]
/// (no transcendental call; deterministic), to ~1e−9.
pub fn log2_ratio(ratio: f64) -> f64 {
    debug_assert!(ratio > 0.0);
    let (mut lo, mut hi) = (-1200.0f64, 1200.0f64); // x = 6·log2, |log2| ≤ 200
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if pow2_frac_over6(mid) < ratio {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi) / 6.0
}

/// The rate slope prior — bits halve every `1 / DEFAULT_SLOPE ≈ 14`
/// QP steps (corpus measurement, round 455; refined online).
pub const DEFAULT_SLOPE: f64 = 0.07;

/// Slope estimates are kept inside this range (bits halving every
/// 2 .. 50 QP steps).
pub const SLOPE_RANGE: (f64, f64) = (0.02, 0.5);

/// The second pass's plan.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Per-frame slice QP.
    pub qps: Vec<i32>,
    /// Per-frame modelled bits at that QP.
    pub planned_bits: Vec<f64>,
    /// The sequence-wide fractional base QP the plan dithers around.
    pub base_qp: f64,
}

/// Buffer parameters of the plan walk.
#[derive(Clone, Copy, Debug)]
pub struct BufferModel {
    /// Bits per frame arriving in the buffer (`bitrate / fps`).
    pub inflow: f64,
    /// Buffer capacity in bits.
    pub vbv: f64,
}

/// Modelled bits of a first-pass frame re-coded at (fractional) `q`.
#[inline]
fn model_bits(s: &FrameStat, slope: f64, q: f64) -> f64 {
    s.bits.max(8.0) * pow2_frac_over6(-6.0 * slope * (q - s.qp as f64))
}

/// One walk at a fractional `q_base`: per-frame integer QPs (dithered
/// around `q_base`, bent by the buffer), their modelled bits, and the
/// total. `fullness0` is the buffer level ahead of the first frame.
fn walk(
    stats: &[FrameStat],
    slope: f64,
    q_base: f64,
    buf: BufferModel,
    fullness0: f64,
) -> (Vec<i32>, Vec<f64>, f64) {
    let n = stats.len();
    let mut qps = Vec::with_capacity(n);
    let mut bits_out = Vec::with_capacity(n);
    let mut total = 0.0;
    let mut frac_cum = 0.0;
    let mut fullness = fullness0;
    let lo = q_base.floor().clamp(0.0, 51.0) as i32;
    let hi = (lo + 1).min(51);
    for st in stats {
        frac_cum += model_bits(st, slope, q_base);
        // Dither: the integer QP that keeps the cumulative modelled
        // bits nearest the fractional-model cumulative target.
        let b_lo = model_bits(st, slope, lo as f64);
        let b_hi = model_bits(st, slope, hi as f64);
        let mut q = if ((total + b_lo) - frac_cum).abs() <= ((total + b_hi) - frac_cum).abs() {
            lo
        } else {
            hi
        };
        let mut b = model_bits(st, slope, q as f64);
        // Buffer: raise the QP until the frame fits what has arrived.
        // A surplus that would overflow the buffer is simply left
        // unused (the channel idles): forcing it to be spent would
        // push every other frame up in QP for no distortion gain.
        while b > fullness && q < 51 {
            q += 1;
            b = model_bits(st, slope, q as f64);
        }
        fullness = (fullness - b + buf.inflow).min(buf.vbv);
        total += b;
        qps.push(q);
        bits_out.push(b);
    }
    (qps, bits_out, total)
}

/// Derive the plan for `stats` (the frames still to code) for
/// `target_total` bits under `buf`, with the buffer at `fullness0`
/// ahead of the first frame and the rate slope `slope`.
pub fn plan_two_pass(
    stats: &[FrameStat],
    target_total: f64,
    buf: BufferModel,
    fullness0: f64,
    slope: f64,
) -> Result<Plan> {
    if stats.is_empty() {
        return Err(Error::invalid("evc two-pass: stats hold no frames"));
    }
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !positive(target_total) || !positive(buf.inflow) || !positive(buf.vbv) {
        return Err(Error::invalid(
            "evc two-pass: target, inflow and vbv must be positive",
        ));
    }
    if slope.is_nan() || !(SLOPE_RANGE.0..=SLOPE_RANGE.1).contains(&slope) {
        return Err(Error::invalid(format!(
            "evc two-pass: slope {slope} outside {SLOPE_RANGE:?}"
        )));
    }
    // Bisection on the fractional base QP: the modelled total is
    // non-increasing in q_base.
    let (mut lo, mut hi) = (0.0f64, 51.0f64);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let (_, _, total) = walk(stats, slope, mid, buf, fullness0);
        if total > target_total {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (q_lo, b_lo, t_lo) = walk(stats, slope, lo, buf, fullness0);
    let (q_hi, b_hi, t_hi) = walk(stats, slope, hi, buf, fullness0);
    let (qps, planned_bits, base_qp) = if (t_lo - target_total).abs() <= (t_hi - target_total).abs()
    {
        (q_lo, b_lo, lo)
    } else {
        (q_hi, b_hi, hi)
    };
    Ok(Plan {
        qps,
        planned_bits,
        base_qp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_stats(n: usize, gop: usize) -> Vec<FrameStat> {
        (0..n)
            .map(|i| {
                let idr = i % gop == 0;
                let mut s = 0x9E37_79B9u32.wrapping_mul(i as u32 + 1);
                s ^= s >> 13;
                let jitter = 0.8 + (s % 41) as f64 / 100.0;
                FrameStat {
                    idr,
                    qp: 30,
                    bits: if idr { 60_000.0 } else { 6_000.0 } * jitter,
                }
            })
            .collect()
    }

    #[test]
    fn stats_round_trip() {
        let stats = synth_stats(7, 3);
        let text = format_stats(&stats);
        let back = parse_stats(&text).unwrap();
        assert_eq!(back.len(), stats.len());
        for (a, b) in stats.iter().zip(back.iter()) {
            assert_eq!(a.idr, b.idr);
            assert_eq!(a.qp, b.qp);
            assert_eq!(a.bits.floor(), b.bits);
        }
        assert!(parse_stats("1 30 100\n").is_err(), "missing header");
        assert!(parse_stats("# oxideav-evc first-pass stats v1\n1 99 100\n").is_err());
        assert!(parse_stats("# oxideav-evc first-pass stats v1\nx\n").is_err());
    }

    #[test]
    fn pow2_ladder_is_consistent() {
        for k in -20..=60 {
            let a = pow2_qp_over6(k);
            let b = pow2_frac_over6(k as f64);
            assert!((a - b).abs() < 1e-9 * a, "k {k}: {a} vs {b}");
        }
        // Midpoints sit between the neighbouring integer rungs.
        for k in 0..51 {
            let m = pow2_frac_over6(k as f64 + 0.5);
            assert!(m > pow2_qp_over6(k) && m < pow2_qp_over6(k + 1));
        }
        assert!((pow2_frac_over6(3.0) - std::f64::consts::SQRT_2).abs() < 1e-12);
        for &r in &[0.01, 0.5, 0.75, 1.0, 1.5, 2.0, 1000.0] {
            assert!((log2_ratio(r) - f64::log2(r)).abs() < 1e-7, "log2 {r}");
        }
    }

    /// With a roomy buffer the plan is flat (one QP, ±1 dither) and its
    /// modelled total lands on the target.
    #[test]
    fn flat_plan_hits_target_with_roomy_buffer() {
        let stats = synth_stats(48, 8);
        let first_pass_total: f64 = stats.iter().map(|s| s.bits).sum();
        for &scale in &[0.5, 1.0, 2.0] {
            let target = first_pass_total * scale;
            let buf = BufferModel {
                inflow: target / 48.0,
                vbv: target, // the whole sequence fits
            };
            let plan = plan_two_pass(&stats, target, buf, buf.vbv, 1.0 / 6.0).unwrap();
            let total: f64 = plan.planned_bits.iter().sum();
            assert!(
                (total - target).abs() < 0.01 * target,
                "scale {scale}: planned {total} vs target {target}"
            );
            let (mn, mx) = (
                *plan.qps.iter().min().unwrap(),
                *plan.qps.iter().max().unwrap(),
            );
            assert!(mx - mn <= 1, "scale {scale}: qps {:?}", plan.qps);
            if scale == 1.0 {
                assert!((plan.base_qp - 30.0).abs() < 0.6, "base {}", plan.base_qp);
            } else if scale == 0.5 {
                assert!((plan.base_qp - 36.0).abs() < 0.6, "base {}", plan.base_qp);
            }
        }
    }

    /// A tight buffer bends the plan: the key frames are raised in QP
    /// so they fit what has arrived, and the modelled walk never
    /// underflows.
    #[test]
    fn tight_buffer_never_underflows() {
        let stats = synth_stats(48, 8);
        let target: f64 = stats.iter().map(|s| s.bits).sum();
        let inflow = target / 48.0;
        let buf = BufferModel {
            inflow,
            vbv: 2.5 * inflow,
        };
        let plan = plan_two_pass(&stats, target, buf, buf.vbv, 1.0 / 6.0).unwrap();
        let mut fullness = buf.vbv;
        for (i, b) in plan.planned_bits.iter().enumerate() {
            assert!(
                *b <= fullness + 1e-6,
                "frame {i}: {b} bits vs {fullness} buffered"
            );
            fullness = (fullness - b + inflow).min(buf.vbv);
        }
        // The IDR frames run at a higher QP than their neighbours.
        for i in (8..48).step_by(8) {
            assert!(
                plan.qps[i] > plan.qps[i + 1],
                "idr {i}: {:?}",
                &plan.qps[i..i + 2]
            );
        }
        let total: f64 = plan.planned_bits.iter().sum();
        assert!(
            (total - target).abs() < 0.05 * target,
            "{total} vs {target}"
        );
    }

    /// A flatter slope needs a bigger QP move for the same halving.
    #[test]
    fn slope_scales_the_qp_move() {
        let stats = synth_stats(24, 8);
        let total: f64 = stats.iter().map(|s| s.bits).sum();
        let buf = BufferModel {
            inflow: total / 24.0,
            vbv: total,
        };
        let steep = plan_two_pass(&stats, total * 0.5, buf, buf.vbv, 1.0 / 6.0).unwrap();
        let flat = plan_two_pass(&stats, total * 0.5, buf, buf.vbv, 1.0 / 12.0).unwrap();
        assert!((steep.base_qp - 36.0).abs() < 0.6, "{}", steep.base_qp);
        assert!((flat.base_qp - 42.0).abs() < 0.6, "{}", flat.base_qp);
    }

    #[test]
    fn rejects_bad_inputs() {
        let buf = BufferModel {
            inflow: 100.0,
            vbv: 1000.0,
        };
        assert!(plan_two_pass(&[], 1000.0, buf, buf.vbv, DEFAULT_SLOPE).is_err());
        let stats = synth_stats(4, 4);
        assert!(plan_two_pass(&stats, 0.0, buf, buf.vbv, DEFAULT_SLOPE).is_err());
        assert!(plan_two_pass(
            &stats,
            1000.0,
            BufferModel {
                inflow: 0.0,
                vbv: 1.0
            },
            1.0,
            DEFAULT_SLOPE,
        )
        .is_err());
        assert!(plan_two_pass(&stats, 1000.0, buf, buf.vbv, 5.0).is_err());
    }
}
