//! Encoder-side forward transform + quantization — the inverse of the
//! decoder's §8.7.2/§8.7.3/§8.7.4 scale-and-inverse-transform chain
//! ([`crate::dequant::scale_and_inverse_transform`]).
//!
//! ## Derivation
//!
//! The decoder computes (Baseline, `sps_iqt_flag == 0`, trType 0):
//!
//! ```text
//! scaled[·]  = round( level · levelScale[qp%6] · 2^(qp/6) · rectNorm / 2^bdShift )   (eq. 1059)
//! R          = A · scaled · Bᵀ            A = transMatrix(nTbH), B = transMatrix(nTbW)  (eq. 1062)
//! residual   = round( R / 2^bdShiftPost ) with bdShiftPost = (20 − bitDepth) + 7        (eq. 1055)
//! ```
//!
//! so the encoder needs `scaled* = A⁻¹ · (S · 2^bdShiftPost) · B⁻ᵀ` for a
//! target residual `S`, then `level = round(scaled* · 2^bdShift /
//! (levelScale·2^(qp/6)·rectNorm))`. The integer DCT-II family has
//! (near-)orthogonal **rows** — `M·Mᵀ = D` with `D` diagonal exactly for
//! nTbS ∈ {2, 4} and diagonally dominant for the larger kernels — so
//! `M⁻¹ ≈ Mᵀ·D⁻¹` with per-row norms `D[m][m] = Σₙ M[m][n]²`. The
//! residual error of that approximation is ≲ 0.2 % of amplitude (the
//! worst off-diagonal row product is −50 against norms ≈ 2¹⁵), well
//! under one quantization step at every legal QP; the round-trip
//! fixtures below pin the achieved accuracy.
//!
//! No external encoder is being mirrored here: this is plain linear
//! inversion of the spec's decode chain, computed in `f64` (exact for
//! the ≤ 2⁵³ magnitudes involved, IEEE-deterministic).

use oxideav_core::{Error, Result};

use crate::dequant::{rect_norm, scaling_bd_shift, LEVEL_SCALE_BASELINE};
use crate::transform::trans_matrix;

/// Forward-transform + quantize one TB of residual samples into
/// `TransCoeffLevel`s that the decoder's
/// [`crate::dequant::scale_and_inverse_transform`] turns back into
/// (approximately) `residual`. Returns `true` when any level is
/// non-zero (the CBF the caller must signal).
///
/// `residual` and `levels` are row-major `n_tb_w × n_tb_h`; dimensions
/// must be powers of two in `{2, 4, 8, 16, 32, 64}`; `qp ∈ 0..=63`
/// (the §8.7 scaling domain — Baseline slice QPs occupy 0..=51).
pub fn forward_quantize(
    residual: &[i32],
    levels: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    bit_depth: u32,
) -> Result<bool> {
    let n = n_tb_w * n_tb_h;
    if residual.len() != n || levels.len() != n {
        return Err(Error::invalid(format!(
            "evc forward_quantize: length mismatch (res={}, levels={}, expected {n})",
            residual.len(),
            levels.len()
        )));
    }
    forward_quantize_pass(residual, levels, n_tb_w, n_tb_h, qp, bit_depth)?;

    // One refinement iteration against the *exact* decode chain: the
    // Mᵀ·D⁻¹ inversion is approximate for nTbS ≥ 8 (rows of the integer
    // kernels are not perfectly orthogonal), so re-quantize the decoded
    // error and fold the correction in. This squares the approximation
    // error away, leaving only the irreducible quantization rounding.
    let mut back = vec![0i32; n];
    crate::dequant::scale_and_inverse_transform(levels, &mut back, n_tb_w, n_tb_h, qp, bit_depth)?;
    let err: Vec<i32> = residual
        .iter()
        .zip(back.iter())
        .map(|(&want, &got)| want - got)
        .collect();
    if err.iter().any(|&e| e != 0) {
        let mut delta = vec![0i32; n];
        forward_quantize_pass(&err, &mut delta, n_tb_w, n_tb_h, qp, bit_depth)?;
        for (lvl, d) in levels.iter_mut().zip(delta.iter()) {
            *lvl = (*lvl + *d).clamp(-32768, 32767);
        }
    }
    Ok(levels.iter().any(|&v| v != 0))
}

/// One linear-inversion pass (no refinement) — see [`forward_quantize`].
fn forward_quantize_pass(
    residual: &[i32],
    levels: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    bit_depth: u32,
) -> Result<bool> {
    let n = n_tb_w * n_tb_h;
    if !matches!(n_tb_w, 2 | 4 | 8 | 16 | 32 | 64) || !matches!(n_tb_h, 2 | 4 | 8 | 16 | 32 | 64) {
        return Err(Error::unsupported(format!(
            "evc forward_quantize: nTbS ∈ {{2,4,8,16,32,64}}; got {n_tb_w}x{n_tb_h}"
        )));
    }
    if !(0..=63).contains(&qp) {
        return Err(Error::invalid(format!(
            "evc forward_quantize: qp {qp} out of range [0,63]"
        )));
    }
    debug_assert_eq!(residual.len(), n);
    debug_assert_eq!(levels.len(), n);

    let a = trans_matrix(n_tb_h); // vertical kernel (H×H)
    let b = trans_matrix(n_tb_w); // horizontal kernel (W×W)
    let d_a = row_norms(a, n_tb_h);
    let d_b = row_norms(b, n_tb_w);

    // scaled* = Aᵀ·D_A⁻¹ · S · D_B⁻¹·B, i.e.
    // scaled*[i][j] = Σ_{k,l} A[k][i]·S[k][l]·B[l][j] / (d_A[k]·d_B[l]).
    // Two 1-D passes; `t` holds (Aᵀ·D_A⁻¹·S), an H×W array.
    let mut t = vec![0f64; n];
    for i in 0..n_tb_h {
        for l in 0..n_tb_w {
            let mut acc = 0f64;
            for k in 0..n_tb_h {
                acc += (a[k * n_tb_h + i] as f64) * (residual[k * n_tb_w + l] as f64) / d_a[k];
            }
            t[i * n_tb_w + l] = acc;
        }
    }

    // Undo the decoder's two shifts: multiply by 2^bdShiftPost (eq.
    // 1055) and 2^bdShift / (levelScale·2^(qp/6)·rectNorm) (eq. 1059).
    let bd_shift_post = (20 - bit_depth) + 7;
    let bd_shift = scaling_bd_shift(n_tb_w, n_tb_h, bit_depth);
    let rect = rect_norm(n_tb_w, n_tb_h) as f64;
    let q_step =
        (LEVEL_SCALE_BASELINE[(qp % 6) as usize] as f64) * f64::from(1u32 << (qp / 6) as u32);
    let gain = f64::from(1u32 << bd_shift_post) * f64::from(1u32 << bd_shift) / (q_step * rect);

    let mut any = false;
    for i in 0..n_tb_h {
        for j in 0..n_tb_w {
            let mut acc = 0f64;
            for l in 0..n_tb_w {
                acc += t[i * n_tb_w + l] * (b[l * n_tb_w + j] as f64) / d_b[l];
            }
            let lvl = (acc * gain).round();
            let lvl = lvl.clamp(-32768.0, 32767.0) as i32;
            levels[i * n_tb_w + j] = lvl;
            any |= lvl != 0;
        }
    }
    Ok(any)
}

/// Per-row squared norms of a row-major `n × n` kernel — the diagonal of
/// `M·Mᵀ` (e.g. 16384 / 16562 alternating for the 4-point DCT-II).
fn row_norms(mat: &[i16], n: usize) -> Vec<f64> {
    (0..n)
        .map(|m| {
            mat[m * n..(m + 1) * n]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::scale_and_inverse_transform;

    fn round_trip_max_err(residual: &[i32], w: usize, h: usize, qp: i32) -> i32 {
        let mut levels = vec![0i32; w * h];
        forward_quantize(residual, &mut levels, w, h, qp, 8).unwrap();
        let mut back = vec![0i32; w * h];
        scale_and_inverse_transform(&levels, &mut back, w, h, qp, 8).unwrap();
        residual
            .iter()
            .zip(back.iter())
            .map(|(&a, &b)| (a - b).abs())
            .max()
            .unwrap()
    }

    /// Zero residual quantizes to all-zero levels (CBF 0).
    #[test]
    fn zero_residual_zero_levels() {
        let res = vec![0i32; 64];
        let mut levels = vec![0i32; 64];
        let any = forward_quantize(&res, &mut levels, 8, 8, 30, 8).unwrap();
        assert!(!any);
        assert!(levels.iter().all(|&v| v == 0));
    }

    /// At QP 0 the quantizer step in the scaled domain is ≈ 1.25, so a
    /// full-range residual must survive the decode chain near-losslessly
    /// across every square TB size.
    #[test]
    fn near_lossless_at_qp0_all_sizes() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 511) - 255
        };
        for s in [4usize, 8, 16, 32, 64] {
            let res: Vec<i32> = (0..s * s).map(|_| next()).collect();
            let err = round_trip_max_err(&res, s, s, 0);
            assert!(err <= 2, "{s}x{s} qp0 max err {err}");
        }
    }

    /// Rectangular TBs (the 4:2:0 chroma shapes of a rectangular luma
    /// CB) round-trip through the rectNorm = 181 branch.
    #[test]
    fn rectangular_blocks_round_trip() {
        let mut seed = 0x9e37_79b9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 201) - 100
        };
        for (w, h) in [(4usize, 8usize), (8, 4), (16, 8), (8, 32)] {
            let res: Vec<i32> = (0..w * h).map(|_| next()).collect();
            let err = round_trip_max_err(&res, w, h, 0);
            assert!(err <= 2, "{w}x{h} qp0 max err {err}");
        }
    }

    /// Reconstruction error grows monotonically-ish with QP but stays
    /// bounded by the quantization step (levelScale-derived): a coarse
    /// sanity envelope over the whole legal slice-QP range.
    #[test]
    fn error_bounded_by_step_across_qp() {
        let mut seed = 0xdead_beefu32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 401) - 200
        };
        let res: Vec<i32> = (0..64).map(|_| next()).collect();
        for qp in [0, 6, 12, 22, 30, 40, 51] {
            let err = round_trip_max_err(&res, 8, 8, qp);
            // step in the residual domain ≈ levelScale·2^(qp/6)·2^bdShift
            // /2^(bdShift+post)·‖basis‖ — empirically ≲ 2^((qp/6)+2).
            let bound = 4 << (qp / 6).max(0);
            assert!(err <= bound, "qp {qp}: err {err} > bound {bound}");
        }
    }

    /// A DC-flat residual (the most common intra shape) is representable
    /// at moderate QP with small error.
    #[test]
    fn flat_residual_moderate_qp() {
        let res = vec![37i32; 16];
        let err = round_trip_max_err(&res, 4, 4, 22);
        assert!(err <= 4, "flat 4x4 qp22 err {err}");
    }

    /// Bad inputs are refused.
    #[test]
    fn rejects_bad_inputs() {
        let res = vec![0i32; 16];
        let mut levels = vec![0i32; 16];
        assert!(forward_quantize(&res, &mut levels, 4, 4, 64, 8).is_err());
        assert!(forward_quantize(&res, &mut levels, 3, 4, 22, 8).is_err());
        let mut short = vec![0i32; 15];
        assert!(forward_quantize(&res, &mut short, 4, 4, 22, 8).is_err());
    }
}
