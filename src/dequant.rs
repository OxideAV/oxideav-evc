//! EVC dequantization / scaling process (ISO/IEC 23094-1 §8.7.3 +
//! §8.7.2 final renormalisation).
//!
//! Round-3 scope: Baseline profile with `sps_iqt_flag = 0`, 8-bit luma /
//! chroma. The scaled transform coefficient `d[x][y]` is computed from
//! the bitstream's `TransCoeffLevel[x][y]` per eq. 1059, then the
//! cascaded inverse-transform output is renormalised per eq. 1055 with
//! the bit-depth-driven `bdShift`.

use oxideav_core::{Error, Result};

use crate::transform::inverse_transform;

/// `levelScale` — eq. 1059. Round-3 only ships the `sps_iqt_flag = 0`
/// list (the `sps_iqt_flag = 1` path swaps the trailing 71 for 72).
pub const LEVEL_SCALE_BASELINE: [i32; 6] = [40, 45, 51, 57, 64, 71];

/// Compute `bdShift` for the scaling process (eq. 1056 / 1057).
fn scaling_bd_shift(n_tb_w: usize, n_tb_h: usize, bit_depth: u32) -> u32 {
    let log2_w = (n_tb_w as u32).trailing_zeros();
    let log2_h = (n_tb_h as u32).trailing_zeros();
    let logsum = log2_w + log2_h;
    bit_depth + ((logsum & 1) * 8 + logsum / 2) - 5
}

/// Compute `rectNorm` per eq. 1058. Returns 181 if `(log2W + log2H)` is
/// odd, else 1.
fn rect_norm(n_tb_w: usize, n_tb_h: usize) -> i32 {
    let log2_w = (n_tb_w as u32).trailing_zeros();
    let log2_h = (n_tb_h as u32).trailing_zeros();
    if (log2_w + log2_h) & 1 == 1 {
        181
    } else {
        1
    }
}

/// Scaling + transformation + final renorm per §8.7.2.
///
/// `levels` is the per-position `TransCoeffLevel[x][y]` array (length
/// `n_tb_w * n_tb_h`, row-major) and `dst` receives the
/// renormalised residual samples `r[x][y]` in the same shape.
///
/// `qp` is the unsigned Qp' per eq. 1043 / 1048 / 1049 (i.e. the
/// `bit-depth offset` is already added — for 8-bit, `QpBdOffsetY = 0` so
/// `Qp'Y = QpY`). `bit_depth` is the relevant component bit depth (8 for
/// luma in round 3).
pub fn scale_and_inverse_transform(
    levels: &[i32],
    dst: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    bit_depth: u32,
) -> Result<()> {
    if levels.len() != n_tb_w * n_tb_h || dst.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(format!(
            "evc dequant: length mismatch (levels={}, dst={}, expected {}*{}={})",
            levels.len(),
            dst.len(),
            n_tb_w,
            n_tb_h,
            n_tb_w * n_tb_h
        )));
    }
    if !(0..=63).contains(&qp) {
        return Err(Error::invalid(format!(
            "evc dequant: qp {qp} out of range [0,63]"
        )));
    }
    // Step 1: scaling per §8.7.3 eq. 1059.
    let bd_shift = scaling_bd_shift(n_tb_w, n_tb_h, bit_depth);
    let rect = rect_norm(n_tb_w, n_tb_h);
    let level_scale = LEVEL_SCALE_BASELINE[(qp % 6) as usize];
    let level_shift = qp / 6;
    let one_shl = 1i32 << (bd_shift - 1);
    let mut scaled = vec![0i32; n_tb_w * n_tb_h];
    for (idx, &lvl) in levels.iter().enumerate() {
        // (TransCoeffLevel * levelScale[qP%6]) << (qP/6) * rectNorm
        // + (1 << (bdShift - 1))) >> bdShift, clipped to [-32768, 32767].
        let raw = ((lvl as i64) * (level_scale as i64)) << level_shift;
        let v = (raw * (rect as i64) + (one_shl as i64)) >> bd_shift;
        let clipped = v.clamp(-32768, 32767) as i32;
        scaled[idx] = clipped;
    }
    // Step 2: inverse transform (cascaded 1-D matrix mul, eq. 1062).
    inverse_transform(&mut scaled, n_tb_w, n_tb_h)?;
    // Step 3: final renormalisation per eq. 1055. With sps_iqt_flag = 0:
    //   bdShift_post = (20 - bitDepth) + 7
    let bd_shift_post = (20 - bit_depth) + 7;
    let one_shl_post = 1i32 << (bd_shift_post - 1);
    for (out, &v) in dst.iter_mut().zip(scaled.iter()) {
        *out = (v + one_shl_post) >> bd_shift_post;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero levels produce zero residuals at any QP.
    #[test]
    fn zero_levels_produce_zero_residuals() {
        let levels = vec![0i32; 16];
        let mut dst = vec![0i32; 16];
        scale_and_inverse_transform(&levels, &mut dst, 4, 4, 22, 8).unwrap();
        assert!(dst.iter().all(|&v| v == 0));
    }

    /// `bdShift` per eq. 1056 with bit_depth = 8:
    /// 4×4: log2W+log2H = 4 → even → bdShift = 8 + 0*8 + 2 - 5 = 5
    /// 8×8: 6 → even → 8 + 0 + 3 - 5 = 6
    /// 16×16: 8 → even → 8 + 0 + 4 - 5 = 7
    /// 32×32: 10 → even → 8 + 0 + 5 - 5 = 8
    /// 4×8: 5 → odd → 8 + 8 + 2 - 5 = 13
    #[test]
    fn scaling_bd_shift_values() {
        assert_eq!(scaling_bd_shift(4, 4, 8), 5);
        assert_eq!(scaling_bd_shift(8, 8, 8), 6);
        assert_eq!(scaling_bd_shift(16, 16, 8), 7);
        assert_eq!(scaling_bd_shift(32, 32, 8), 8);
        assert_eq!(scaling_bd_shift(4, 8, 8), 13);
    }

    /// `rect_norm` per eq. 1058: 181 when (log2W + log2H) is odd, else 1.
    #[test]
    fn rect_norm_values() {
        assert_eq!(rect_norm(4, 4), 1);
        assert_eq!(rect_norm(8, 8), 1);
        assert_eq!(rect_norm(4, 8), 181);
        assert_eq!(rect_norm(8, 4), 181);
    }

    /// QP out of range surfaces as Invalid.
    #[test]
    fn rejects_out_of_range_qp() {
        let levels = vec![0i32; 16];
        let mut dst = vec![0i32; 16];
        let err = scale_and_inverse_transform(&levels, &mut dst, 4, 4, 64, 8).unwrap_err();
        assert!(format!("{err}").contains("qp"));
    }

    /// Length mismatch produces Invalid.
    #[test]
    fn rejects_length_mismatch() {
        let levels = vec![0i32; 15];
        let mut dst = vec![0i32; 16];
        let err = scale_and_inverse_transform(&levels, &mut dst, 4, 4, 22, 8).unwrap_err();
        assert!(format!("{err}").contains("length"));
    }
}
