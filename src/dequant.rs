//! EVC dequantization / scaling process (ISO/IEC 23094-1 §8.7.3 +
//! §8.7.2 final renormalisation).
//!
//! Round-3 scope: Baseline profile with `sps_iqt_flag = 0`, 8-bit luma /
//! chroma. The scaled transform coefficient `d[x][y]` is computed from
//! the bitstream's `TransCoeffLevel[x][y]` per eq. 1059, then the
//! cascaded inverse-transform output is renormalised per eq. 1055 with
//! the bit-depth-driven `bdShift`.

use oxideav_core::{Error, Result};

use crate::transform::{inverse_transform, inverse_transform_ats};

/// `levelScale` — §8.7.3, the `sps_iqt_flag == 0` list.
pub const LEVEL_SCALE_BASELINE: [i32; 6] = [40, 45, 51, 57, 64, 71];

/// `levelScale` — §8.7.3, the `sps_iqt_flag == 1` list (the improved
/// quantization swaps the trailing 71 for 72).
pub const LEVEL_SCALE_IQT: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// §8.7.1 eq. 1043 — `Qp′Y = QpY + QpBdOffsetY` with
/// `QpBdOffsetY = 6 * (BitDepthY − 8)`: the luma quantization parameter
/// the §8.7.2 scaling consumes (eq. 1050).
pub fn qp_prime_y(qp_y: i32, bit_depth_luma: u32) -> i32 {
    qp_y + 6 * (bit_depth_luma as i32 - 8)
}

/// §8.7.1 eqs. 1044-1049 — the chroma quantization parameter `Qp′C` the
/// §8.7.2 scaling consumes (eqs. 1051/1052): `qPi = Clip3(−QpBdOffsetC,
/// 57, QpY + slice_c*_qp_offset)` mapped through `ChromaQpTable` (the
/// spec-page-67 default Table 5 / Table 6 selected by `sps_iqt_flag` for
/// 4:2:0 with `chroma_qp_table_present_flag == 0`) plus `QpBdOffsetC`.
pub fn qp_prime_c(
    qp_y: i32,
    chroma_qp_offset: i32,
    bit_depth_chroma: u32,
    sps_iqt_flag: bool,
) -> i32 {
    let qp_bd_offset_c = 6 * (bit_depth_chroma as i32 - 8);
    let qpi = (qp_y + chroma_qp_offset).clamp(-qp_bd_offset_c, 57);
    let qp_c = if sps_iqt_flag {
        crate::dra::table6_qp_c(qpi)
    } else {
        crate::dra::table5_qp_c(qpi)
    };
    qp_c + qp_bd_offset_c
}

/// Compute `bdShift` for the scaling process (eq. 1056 / 1057).
/// `pub(crate)` so the encoder-side quantizer (`crate::quant_enc`)
/// inverts the exact scale the decoder applies.
pub(crate) fn scaling_bd_shift(n_tb_w: usize, n_tb_h: usize, bit_depth: u32) -> u32 {
    let log2_w = (n_tb_w as u32).trailing_zeros();
    let log2_h = (n_tb_h as u32).trailing_zeros();
    let logsum = log2_w + log2_h;
    bit_depth + ((logsum & 1) * 8 + logsum / 2) - 5
}

/// Compute `rectNorm` per eq. 1058. Returns 181 if `(log2W + log2H)` is
/// odd, else 1. `pub(crate)` for the encoder-side quantizer.
pub(crate) fn rect_norm(n_tb_w: usize, n_tb_h: usize) -> i32 {
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
    sps_iqt_flag: bool,
) -> Result<()> {
    // trType 0/0 → the plain §8.7.4.3 DCT-II path.
    scale_and_inverse_transform_typed(
        levels,
        dst,
        n_tb_w,
        n_tb_h,
        qp,
        bit_depth,
        0,
        0,
        sps_iqt_flag,
    )
}

/// §8.7 scaling + **ATS-selected** inverse transform (§7.3.8.5 / Table 30):
/// like [`scale_and_inverse_transform`] but the step-2 inverse transform
/// selects the per-direction kernel `(tr_type_hor, tr_type_ver)` per
/// §8.7.4.2. With `(0, 0)` it is byte-for-byte the DCT-II path;
/// `(1, ·)`/`(2, ·)` engage DST-VII / DCT-VIII on the luma block an
/// `ats_cu_intra_flag == 1` (or `ats_cu_inter_flag == 1`) CU selects.
#[allow(clippy::too_many_arguments)]
pub fn scale_and_inverse_transform_ats(
    levels: &[i32],
    dst: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    bit_depth: u32,
    tr_type_hor: u32,
    tr_type_ver: u32,
    sps_iqt_flag: bool,
) -> Result<()> {
    scale_and_inverse_transform_typed(
        levels,
        dst,
        n_tb_w,
        n_tb_h,
        qp,
        bit_depth,
        tr_type_hor,
        tr_type_ver,
        sps_iqt_flag,
    )
}

/// Shared §8.7 scale (eq. 1059) → inverse transform (eq. 1062) → final
/// renorm (eq. 1055) core. `tr_type_hor`/`tr_type_ver == 0` routes the
/// step-2 transform through the plain DCT-II [`inverse_transform`]; any
/// non-zero type routes through [`inverse_transform_ats`].
#[allow(clippy::too_many_arguments)]
fn scale_and_inverse_transform_typed(
    levels: &[i32],
    dst: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    qp: i32,
    bit_depth: u32,
    tr_type_hor: u32,
    tr_type_ver: u32,
    sps_iqt_flag: bool,
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
    // qP is Qp′ (eqs. 1050-1052): QpY/QpC plus the bit-depth offset
    // 6 * (BitDepth − 8), so the legal ceiling grows with depth
    // (51 + 48 at the §7.4.3.1 16-bit maximum).
    if !(0..=99).contains(&qp) {
        return Err(Error::invalid(format!(
            "evc dequant: qP {qp} out of range [0,99]"
        )));
    }
    // Step 1: scaling per §8.7.3 eq. 1059.
    let bd_shift = scaling_bd_shift(n_tb_w, n_tb_h, bit_depth);
    let rect = rect_norm(n_tb_w, n_tb_h);
    let level_scale = if sps_iqt_flag {
        LEVEL_SCALE_IQT[(qp % 6) as usize]
    } else {
        LEVEL_SCALE_BASELINE[(qp % 6) as usize]
    };
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
    // Step 2: inverse transform (cascaded 1-D matrix mul, eq. 1062;
    // the §8.7.4.1 step-2 eq. 1060/1061 intermediate rides inside).
    if tr_type_hor == 0 && tr_type_ver == 0 {
        inverse_transform(&mut scaled, n_tb_w, n_tb_h, sps_iqt_flag)?;
    } else {
        inverse_transform_ats(
            &mut scaled,
            n_tb_w,
            n_tb_h,
            tr_type_hor,
            tr_type_ver,
            sps_iqt_flag,
        )?;
    }
    // Step 3: final renormalisation per eq. 1055 with the §8.7.2 step-3
    // bdShift — eq. 1053 (`(20 − BitDepth) + 7`) under
    // `sps_iqt_flag == 0`, eq. 1054 (`20 − BitDepth`) under `== 1` (the
    // 7-bit renorm already happened at the eq. 1060 intermediate).
    let bd_shift_post = if sps_iqt_flag {
        20 - bit_depth
    } else {
        (20 - bit_depth) + 7
    };
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
        scale_and_inverse_transform(&levels, &mut dst, 4, 4, 22, 8, false).unwrap();
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
        let err = scale_and_inverse_transform(&levels, &mut dst, 4, 4, 100, 8, false).unwrap_err();
        assert!(format!("{err}").contains("qP"));
    }

    /// Length mismatch produces Invalid.
    #[test]
    fn rejects_length_mismatch() {
        let levels = vec![0i32; 15];
        let mut dst = vec![0i32; 16];
        let err = scale_and_inverse_transform(&levels, &mut dst, 4, 4, 22, 8, false).unwrap_err();
        assert!(format!("{err}").contains("length"));
    }
}
