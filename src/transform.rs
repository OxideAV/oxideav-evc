//! EVC inverse transform (ISO/IEC 23094-1 §8.7.4).
//!
//! Round-5 scope: Baseline profile with `sps_iqt_flag = 0` and
//! `sps_ats_flag = 0`. Both `trTypeHor` and `trTypeVer` are forced to 0 →
//! every transform block uses the DCT-II family with kernel size
//! `nTbS ∈ {2, 4, 8, 16, 32, 64}`. The 64-point matrix is built from the
//! closed-form `M[m][n] = round(64·√2·cos(π·m·(2n+1)/128))` for m≥1
//! (M[0][n] = 64), which we verified against every printed entry in
//! eq. 1072 / 1074 of the spec — the printed `transMatrixCol0to15`
//! (rows 0..63 × cols 0..15) and `transMatrixCol16to31` (rows 0..63 ×
//! cols 16..31) both match the formula. The remaining cols 32..63 fall
//! out of the same formula, sidestepping the m/n labelling typos in
//! eq. 1075/1076.
//!
//! The DST-VII variants (trType = 1, 2 — used by Main-profile ATS) and
//! the alternative DCT (trType = 1 / nTbS=2 in `sps_iqt_flag=1` mode)
//! are deferred to round 4.
//!
//! ## Two-stage cascade (§8.7.4.1)
//!
//! 1. **Vertical pass** — invoke the 1-D transform on each column of the
//!    `(nTbW)x(nTbH)` array of scaled coefficients.
//! 2. **Intermediate clamp** (when `sps_iqt_flag = 0`, the round-3 path):
//!    no rescale; intermediate values pass straight through (eq. 1061).
//! 3. **Horizontal pass** — invoke the 1-D transform on each row.
//!
//! The 1-D transform is `y[i] = Σ transMatrix[i][j] · x[j]` (eq. 1062).
//! The transform matrices are the well-known integer DCT-II tables from
//! the HEVC family (§8.7.4.3 eq. 1063–1070 here).

use oxideav_core::{Error, Result};

/// Apply the EVC two-stage inverse transform to an `(nTbW)x(nTbH)`
/// array of scaled coefficients in `coeffs` (row-major). `coeffs` is
/// overwritten with the residual sample array.
///
/// Both dimensions must be powers of two in `{2, 4, 8, 16, 32, 64}`.
pub fn inverse_transform(coeffs: &mut [i32], n_tb_w: usize, n_tb_h: usize) -> Result<()> {
    if coeffs.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(format!(
            "evc inverse_transform: buffer len {} != {}*{} = {}",
            coeffs.len(),
            n_tb_w,
            n_tb_h,
            n_tb_w * n_tb_h
        )));
    }
    if !is_supported_size(n_tb_w) || !is_supported_size(n_tb_h) {
        return Err(Error::unsupported(format!(
            "evc inverse_transform: nTbS ∈ {{2,4,8,16,32,64}}; got {n_tb_w}x{n_tb_h}"
        )));
    }
    // Step 1: vertical 1-D transform on each column → e[x][y].
    let mut e = vec![0i32; n_tb_w * n_tb_h];
    let m_v = trans_matrix(n_tb_h);
    let mut col = vec![0i32; n_tb_h];
    let mut out_v = vec![0i32; n_tb_h];
    for x in 0..n_tb_w {
        for y in 0..n_tb_h {
            col[y] = coeffs[y * n_tb_w + x];
        }
        transform_1d(&col, &mut out_v, m_v, n_tb_h);
        for y in 0..n_tb_h {
            e[y * n_tb_w + x] = out_v[y];
        }
    }
    // Step 2 (sps_iqt_flag = 0, eq. 1061): g[x][y] = e[x][y].
    let g = e;
    // Step 3: horizontal 1-D transform on each row → r[x][y].
    let m_h = trans_matrix(n_tb_w);
    let mut out_h = vec![0i32; n_tb_w];
    for y in 0..n_tb_h {
        transform_1d(&g[y * n_tb_w..y * n_tb_w + n_tb_w], &mut out_h, m_h, n_tb_w);
        for x in 0..n_tb_w {
            coeffs[y * n_tb_w + x] = out_h[x];
        }
    }
    Ok(())
}

fn is_supported_size(n: usize) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32 | 64)
}

/// §8.7.4.1 ATS-intra inverse transform: like [`inverse_transform`] but
/// each 1-D stage selects its kernel by transform type
/// (`tr_type_ver` for the column stage, `tr_type_hor` for the row stage),
/// per §8.7.4.2. `trType ∈ {0 = DCT-II, 1 = DST-VII, 2 = DCT-VIII}`. The
/// non-DCT-II kernels (DST-VII / DCT-VIII) are defined for `nTbS ∈ {4, 8,
/// 16, 32}`; the §7.3.8.5 presence predicate already constrains an
/// ATS-intra luma TB to `log2 <= 5` (`nTbS <= 32`).
///
/// chroma always passes `trType = 0` (§8.7.4.1: "trType set equal to … if
/// cIdx is equal to 0, otherwise 0"), so chroma callers use the plain
/// [`inverse_transform`].
pub fn inverse_transform_ats(
    coeffs: &mut [i32],
    n_tb_w: usize,
    n_tb_h: usize,
    tr_type_hor: u32,
    tr_type_ver: u32,
) -> Result<()> {
    if coeffs.len() != n_tb_w * n_tb_h {
        return Err(Error::invalid(format!(
            "evc inverse_transform_ats: buffer len {} != {}*{}",
            coeffs.len(),
            n_tb_w,
            n_tb_h
        )));
    }
    if !is_supported_size(n_tb_w) || !is_supported_size(n_tb_h) {
        return Err(Error::unsupported(format!(
            "evc inverse_transform_ats: nTbS ∈ {{2,4,8,16,32,64}}; got {n_tb_w}x{n_tb_h}"
        )));
    }
    let m_v = trans_matrix_typed(n_tb_h, tr_type_ver)?;
    let m_h = trans_matrix_typed(n_tb_w, tr_type_hor)?;
    // Step 1: vertical 1-D transform per column with trTypeVer.
    let mut e = vec![0i32; n_tb_w * n_tb_h];
    let mut col = vec![0i32; n_tb_h];
    let mut out_v = vec![0i32; n_tb_h];
    for x in 0..n_tb_w {
        for y in 0..n_tb_h {
            col[y] = coeffs[y * n_tb_w + x];
        }
        transform_1d(&col, &mut out_v, m_v, n_tb_h);
        for y in 0..n_tb_h {
            e[y * n_tb_w + x] = out_v[y];
        }
    }
    // Step 2 (sps_iqt_flag = 0, eq. 1061): g = e.
    let g = e;
    // Step 3: horizontal 1-D transform per row with trTypeHor.
    let mut out_h = vec![0i32; n_tb_w];
    for y in 0..n_tb_h {
        transform_1d(&g[y * n_tb_w..y * n_tb_w + n_tb_w], &mut out_h, m_h, n_tb_w);
        for x in 0..n_tb_w {
            coeffs[y * n_tb_w + x] = out_h[x];
        }
    }
    Ok(())
}

/// Look up the transform matrix for `(n_tb_s, tr_type)`. trType 0 reuses
/// the DCT-II matrices; trType 1 (DST-VII) / trType 2 (DCT-VIII) are
/// available for `nTbS ∈ {4, 8, 16, 32}` (§8.7.4.3 eqs. 1077-1090).
fn trans_matrix_typed(n_tb_s: usize, tr_type: u32) -> Result<&'static [i16]> {
    match tr_type {
        0 => Ok(trans_matrix(n_tb_s)),
        1 => match n_tb_s {
            4 => Ok(&DST7_4),
            8 => Ok(&DST7_8),
            _ => Err(Error::unsupported(format!(
                "evc ATS DST-VII: nTbS ∈ {{4,8}} implemented; got {n_tb_s}"
            ))),
        },
        2 => match n_tb_s {
            4 => Ok(&DCT8_4),
            8 => Ok(&DCT8_8),
            _ => Err(Error::unsupported(format!(
                "evc ATS DCT-VIII: nTbS ∈ {{4,8}} implemented; got {n_tb_s}"
            ))),
        },
        _ => Err(Error::invalid(format!(
            "evc ATS: trType ∈ {{0,1,2}}; got {tr_type}"
        ))),
    }
}

/// 1-D transform: `y[i] = Σ_j transMatrix[i][j] * x[j]` (§8.7.4.2 eq.
/// 1062). `mat` is row-major `n_tb_s × n_tb_s`.
fn transform_1d(x: &[i32], y: &mut [i32], mat: &[i16], n_tb_s: usize) {
    debug_assert_eq!(x.len(), n_tb_s);
    debug_assert_eq!(y.len(), n_tb_s);
    debug_assert_eq!(mat.len(), n_tb_s * n_tb_s);
    for i in 0..n_tb_s {
        let mut acc: i64 = 0;
        for j in 0..n_tb_s {
            acc += (mat[i * n_tb_s + j] as i64) * (x[j] as i64);
        }
        y[i] = acc as i32;
    }
}

/// Look up the trType=0 (DCT-II) transform matrix for the given size.
/// Returns a row-major `n_tb_s × n_tb_s` slice.
fn trans_matrix(n_tb_s: usize) -> &'static [i16] {
    match n_tb_s {
        2 => &MAT_2,
        4 => &MAT_4,
        8 => &MAT_8,
        16 => &MAT_16,
        32 => mat_32(),
        64 => mat_64(),
        _ => panic!("unsupported transform size {n_tb_s}"),
    }
}

// -------------------------------------------------------------------
// Transform matrices — trType = 0, §8.7.4.3 eq. 1063 onward.
// Row m, column n stored as `m * n_tb_s + n`.
// -------------------------------------------------------------------

#[rustfmt::skip]
static MAT_2: [i16; 4] = [
    64,  64,
    64, -64,
];

#[rustfmt::skip]
static MAT_4: [i16; 16] = [
    64,  64,  64,  64,
    84,  35, -35, -84,
    64, -64, -64,  64,
    35, -84,  84, -35,
];

#[rustfmt::skip]
static MAT_8: [i16; 64] = [
    64,  64,  64,  64,  64,  64,  64,  64,
    89,  75,  50,  18, -18, -50, -75, -89,
    84,  35, -35, -84, -84, -35,  35,  84,
    75, -18, -89, -50,  50,  89,  18, -75,
    64, -64, -64,  64,  64, -64, -64,  64,
    50, -89,  18,  75, -75, -18,  89, -50,
    35, -84,  84, -35, -35,  84, -84,  35,
    18, -50,  75, -89,  89, -75,  50, -18,
];

#[rustfmt::skip]
static MAT_16: [i16; 256] = [
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    90, 87, 80, 70, 57, 43, 26,  9, -9, -26, -43, -57, -70, -80, -87, -90,
    89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
    87, 57,  9, -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87,
    84, 35, -35, -84, -84, -35, 35, 84, 84, 35, -35, -84, -84, -35, 35, 84,
    80,  9, -70, -87, -26, 57, 90, 43, -43, -90, -57, 26, 87, 70, -9, -80,
    75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
    70, -43, -87,  9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70,
    64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    57, -80, -26, 90, -9, -87, 43, 70, -70, -43, 87,  9, -90, 26, 80, -57,
    50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
    43, -90, 57, 26, -87, 70,  9, -80, 80, -9, -70, 87, -26, -57, 90, -43,
    35, -84, 84, -35, -35, 84, -84, 35, 35, -84, 84, -35, -35, 84, -84, 35,
    26, -70, 90, -80, 43,  9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -26,
    18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
     9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9,
];

// -------------------------------------------------------------------
// ATS-intra kernels — trType = 1 (DST-VII) and trType = 2 (DCT-VIII),
// §8.7.4.3 eqs. 1077/1078 (DST-VII 4/8) and 1084/1085 (DCT-VIII 4/8).
// Row m, column n stored as `m * n_tb_s + n`.
// -------------------------------------------------------------------

/// DST-VII 4×4 — §8.7.4.3 eq. 1077.
#[rustfmt::skip]
static DST7_4: [i16; 16] = [
    29,  55,  74,  84,
    74,  74,   0, -74,
    84, -29, -74,  55,
    55, -84,  74, -29,
];

/// DST-VII 8×8 — §8.7.4.3 eq. 1078.
#[rustfmt::skip]
static DST7_8: [i16; 64] = [
    16,  32,  46,  59,  70,  79,  84,  87,
    46,  79,  87,  70,  32, -16, -59, -84,
    70,  84,  32, -46, -87, -59,  16,  79,
    84,  46, -59, -79,  16,  87,  32, -70,
    87, -16, -84,  32,  79, -46, -70,  59,
    79, -70, -16,  84, -59, -32,  87, -46,
    59, -87,  70, -16, -46,  84, -79,  32,
    32, -59,  79, -87,  84, -70,  46, -16,
];

/// DCT-VIII 4×4 — §8.7.4.3 eq. 1084.
#[rustfmt::skip]
static DCT8_4: [i16; 16] = [
    84,  74,  55,  29,
    74,   0, -74, -74,
    55, -74, -29,  84,
    29, -74,  84, -55,
];

/// DCT-VIII 8×8 — §8.7.4.3 eq. 1085.
#[rustfmt::skip]
static DCT8_8: [i16; 64] = [
    87,  84,  79,  70,  59,  46,  32,  16,
    84,  59,  16, -32, -70, -87, -79, -46,
    79,  16, -59, -87, -46,  32,  84,  70,
    70, -32, -87, -16,  79,  59, -46, -84,
    59, -70, -46,  79,  32, -84, -16,  87,
    46, -87,  32,  59, -84,  16,  70, -79,
    32, -79,  84, -46, -16,  70, -87,  59,
    16, -46,  70, -84,  87, -79,  59, -32,
];

// 32x32 matrix from §8.7.4.3 eq. 1067 onward — rebuilt from the spec's
// two halves COL0TO15 (32 rows × 16 cols filling cols 0..15 of all rows)
// and COL16TO31 (32 rows × 16 cols filling cols 16..31 of all rows). The
// spec's textual indexing in eq. 1067/1069 has the m/n ranges
// transposed; the natural reading (and the only one consistent with the
// 32×16 source-table dimensions) is used here.
static MAT_32_BUF: std::sync::OnceLock<Vec<i16>> = std::sync::OnceLock::new();

fn mat_32() -> &'static [i16] {
    MAT_32_BUF
        .get_or_init(|| {
            #[rustfmt::skip]
            const COL0TO15: [i16; 32 * 16] = [
                64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13,  4,
                90, 87, 80, 70, 57, 43, 26,  9, -9, -26, -43, -57, -70, -80, -87, -90,
                90, 82, 67, 47, 22, -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13,
                89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
                88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39, 73, 90, 85, 61, 22,
                87, 57,  9, -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87,
                85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30,
                84, 35, -35, -84, -84, -35, 35, 84, 84, 35, -35, -84, -84, -35, 35, 84,
                82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67,  4, 73, 88, 39,
                80,  9, -70, -87, -26, 57, 90, 43, -43, -90, -57, 26, 87, 70, -9, -80,
                78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 30, 90, 54, -39, -90, -47,
                75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
                73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61, -47, -88, -4, 85, 54,
                70, -43, -87,  9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70,
                67, -54, -78, 39, 85, -22, -90,  4, 90, 13, -88, -30, 82, 47, -73, -61,
                64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
                61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67,
                57, -80, -26, 90, -9, -87, 43, 70, -70, -43, 87,  9, -90, 26, 80, -57,
                54, -85, -4, 88, -47, -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73,
                50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
                47, -90, 39, 54, -90, 30, 61, -88, 22, 67, -85, 13, 73, -82,  4, 78,
                43, -90, 57, 26, -87, 70,  9, -80, 80, -9, -70, 87, -26, -57, 90, -43,
                39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82,
                35, -84, 84, -35, -35, 84, -84, 35, 35, -84, 84, -35, -35, 84, -84, 35,
                30, -78, 90, -61,  4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85,
                26, -70, 90, -80, 43,  9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -26,
                22, -61, 85, -90, 73, -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88,
                18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
                13, -39, 61, -78, 88, -90, 85, -73, 54, -30,  4, 22, -47, 67, -82, 90,
                 9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9,
                 4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
            ];
            #[rustfmt::skip]
            const COL16TO31: [i16; 32 * 16] = [
                64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                -4, -13, -22, -30, -39, -47, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
                -90, -87, -80, -70, -57, -43, -26, -9,  9, 26, 43, 57, 70, 80, 87, 90,
                13, 39, 61, 78, 88, 90, 85, 73, 54, 30,  4, -22, -47, -67, -82, -90,
                89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
                -22, -61, -85, -90, -73, -39,  4, 47, 78, 90, 82, 54, 13, -30, -67, -88,
                -87, -57, -9, 43, 80, 90, 70, 26, -26, -70, -90, -80, -43,  9, 57, 87,
                30, 78, 90, 61,  4, -54, -88, -82, -39, 22, 73, 90, 67, 13, -47, -85,
                84, 35, -35, -84, -84, -35, 35, 84, 84, 35, -35, -84, -84, -35, 35, 84,
                -39, -88, -73, -4, 67, 90, 47, -30, -85, -78, -13, 61, 90, 54, -22, -82,
                -80, -9, 70, 87, 26, -57, -90, -43, 43, 90, 57, -26, -87, -70,  9, 80,
                47, 90, 39, -54, -90, -30, 61, 88, 22, -67, -85, -13, 73, 82,  4, -78,
                75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
                -54, -85,  4, 88, 47, -61, -82, 13, 90, 39, -67, -78, 22, 90, 30, -73,
                -70, 43, 87, -9, -90, -26, 80, 57, -57, -80, 26, 90,  9, -87, -43, 70,
                61, 73, -47, -82, 30, 88, -13, -90, -4, 90, 22, -85, -39, 78, 54, -67,
                64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
                -67, -54, 78, 39, -85, -22, 90,  4, -90, 13, 88, -30, -82, 47, 73, -61,
                -57, 80, 26, -90,  9, 87, -43, -70, 70, 43, -87, -9, 90, -26, -80, 57,
                73, 30, -90, 22, 78, -67, -39, 90, -13, -82, 61, 47, -88,  4, 85, -54,
                50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
                -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -30, 90, -54, -39, 90, -47,
                -43, 90, -57, -26, 87, -70, -9, 80, -80,  9, 70, -87, 26, 57, -90, 43,
                82, -22, -54, 90, -61, -13, 78, -85, 30, 47, -90, 67,  4, -73, 88, -39,
                35, -84, 84, -35, -35, 84, -84, 35, 35, -84, 84, -35, -35, 84, -84, 35,
                -85, 47, 13, -67, 90, -73, 22, 39, -82, 88, -54, -4, 61, -90, 78, -30,
                -26, 70, -90, 80, -43, -9, 57, -87, 87, -57,  9, 43, -80, 90, -70, 26,
                88, -67, 30, 13, -54, 82, -90, 78, -47,  4, 39, -73, 90, -85, 61, -22,
                18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
                -90, 82, -67, 47, -22, -4, 30, -54, 73, -85, 90, -88, 78, -61, 39, -13,
                -9, 26, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -26,  9,
                90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 47, -39, 30, -22, 13, -4,
            ];
            let mut m = vec![0i16; 32 * 32];
            for r in 0..32 {
                for c in 0..16 {
                    m[r * 32 + c] = COL0TO15[r * 16 + c];
                    m[r * 32 + 16 + c] = COL16TO31[r * 16 + c];
                }
            }
            m
        })
        .as_slice()
}

/// 64×64 matrix from §8.7.4.3 eq. 1071-1076. The spec's `transMatrixCol0to15`
/// and `transMatrixCol16to31` cover cols 0..31 of all 64 rows; we verified
/// every printed entry against the closed-form
/// `M[m][n] = round(64·√2·cos(π·m·(2n+1)/128))` for m≥1 (M[0][n] = 64).
/// Cols 32..63 are filled from the same formula, dodging the m/n typos
/// in eq. 1075 / 1076.
static MAT_64_BUF: std::sync::OnceLock<Vec<i16>> = std::sync::OnceLock::new();

fn mat_64() -> &'static [i16] {
    MAT_64_BUF
        .get_or_init(|| {
            // Build all 64×64 entries via the EVC integer-DCT-II formula.
            //   M[0][n] = 64
            //   M[m][n] = round(64 · sqrt(2) · cos(pi · m · (2n+1) / 128))  for m >= 1
            // f64 trig with `as i16` rounding is exact for every entry we
            // could cross-check against the spec's printed COL0TO15 /
            // COL16TO31 (cols 0..31, rows 0..63).
            let mut m = vec![0i16; 64 * 64];
            for slot in m.iter_mut().take(64) {
                *slot = 64; // row 0 = DC
            }
            let scale = 64.0_f64 * std::f64::consts::SQRT_2;
            for row in 1..64 {
                for col in 0..64 {
                    let theta =
                        std::f64::consts::PI * (row as f64) * (2.0 * (col as f64) + 1.0) / 128.0;
                    let v = (scale * theta.cos()).round() as i32;
                    debug_assert!(
                        (-90..=90).contains(&v),
                        "row {row} col {col} out of [-90,90]: {v}"
                    );
                    m[row * 64 + col] = v as i16;
                }
            }
            m
        })
        .as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x4 inverse DCT applied to a single DC coefficient at (0,0).
    /// The cascaded matrix multiply (no renorm) produces a basis-vector
    /// outer product of the DCT-II first column with itself.
    ///
    /// Vertical pass on col 0 = [1,0,0,0] yields `mat_4_col[0] = [64, 84,
    /// 64, 35]` placed in column 0 of the intermediate.
    ///
    /// Horizontal pass on row 0 = [64,0,0,0] yields `[4096, 5376, 4096,
    /// 2240]` (i.e. 64 × first row of mat_4).
    ///
    /// In row-major flat storage:
    ///   coeffs[0] = 64*64 = 4096 (row 0 col 0)
    ///   coeffs[1] = 64*84 = 5376 (row 0 col 1)
    ///   coeffs[4] = 84*64 = 5376 (row 1 col 0)
    ///   coeffs[5] = 84*84 = 7056 (row 1 col 1)
    #[test]
    fn inverse_transform_dc_only_4x4() {
        let mut coeffs = vec![0i32; 16];
        coeffs[0] = 1;
        inverse_transform(&mut coeffs, 4, 4).unwrap();
        assert_eq!(coeffs[0], 4096);
        assert_eq!(coeffs[1], 5376);
        assert_eq!(coeffs[4], 5376);
        assert_eq!(coeffs[5], 7056);
    }

    /// ATS trType 0/0 must reproduce the plain DCT-II inverse exactly
    /// (the trType-0 lookup reuses the same matrices).
    #[test]
    fn ats_trtype0_matches_plain() {
        let mut a = vec![0i32; 16];
        a[0] = 1;
        a[5] = 3;
        let mut b = a.clone();
        inverse_transform(&mut a, 4, 4).unwrap();
        inverse_transform_ats(&mut b, 4, 4, 0, 0).unwrap();
        assert_eq!(a, b);
    }

    /// DST-VII 4×4 single-DC. Both stages DST-VII (trType 1). Vertical pass
    /// on col0 = [1,0,0,0] yields DST-VII column 0 = [29,74,84,55] into the
    /// intermediate's column 0. Each row stage is `out[i] = DST7[i][0] *
    /// e_row0_val`. coeffs[0] = 29*29, coeffs[1] = 74*29, coeffs[4] = 29*74,
    /// coeffs[5] = 74*74.
    #[test]
    fn ats_dst7_4x4_dc() {
        let mut coeffs = vec![0i32; 16];
        coeffs[0] = 1;
        inverse_transform_ats(&mut coeffs, 4, 4, 1, 1).unwrap();
        assert_eq!(coeffs[0], 29 * 29);
        assert_eq!(coeffs[1], 74 * 29);
        assert_eq!(coeffs[4], 29 * 74);
        assert_eq!(coeffs[5], 74 * 74);
    }

    /// DCT-VIII 4×4 single-DC. Both stages DCT-VIII (trType 2), column 0 =
    /// [84,74,55,29]. coeffs[0] = 84*84, coeffs[1] = 74*84, coeffs[4] =
    /// 84*74, coeffs[5] = 74*74.
    #[test]
    fn ats_dct8_4x4_dc() {
        let mut coeffs = vec![0i32; 16];
        coeffs[0] = 1;
        inverse_transform_ats(&mut coeffs, 4, 4, 2, 2).unwrap();
        assert_eq!(coeffs[0], 84 * 84);
        assert_eq!(coeffs[1], 74 * 84);
        assert_eq!(coeffs[4], 84 * 74);
        assert_eq!(coeffs[5], 74 * 74);
    }

    /// Mixed kernels: trTypeHor = DST-VII (1), trTypeVer = DCT-VIII (2).
    /// Vertical (col) stage uses DCT-VIII → its column 0 is [84,74,55,29],
    /// so e[row0] = 84. Horizontal (row) stage uses DST-VII →
    /// `out[i] = DST7[i][0] * 84` with DST-VII column 0 = [29,74,84,55].
    /// coeffs[0] = 84*29, coeffs[1] = 84*74, coeffs[4] = 74*29.
    #[test]
    fn ats_mixed_kernels_dc() {
        let mut coeffs = vec![0i32; 16];
        coeffs[0] = 1;
        inverse_transform_ats(&mut coeffs, 4, 4, 1, 2).unwrap();
        assert_eq!(coeffs[0], 84 * 29);
        assert_eq!(coeffs[1], 84 * 74);
        assert_eq!(coeffs[4], 74 * 29);
    }

    /// 8×8 DST-VII / DCT-VIII zero input → zero output (matrix integrity).
    #[test]
    fn ats_8x8_zero() {
        for (h, v) in [(1u32, 1u32), (2, 2), (1, 2), (2, 1)] {
            let mut coeffs = vec![0i32; 64];
            inverse_transform_ats(&mut coeffs, 8, 8, h, v).unwrap();
            assert!(coeffs.iter().all(|&c| c == 0));
        }
    }

    /// DST-VII / DCT-VIII at unimplemented sizes surface Unsupported (the
    /// matrices for 16/32 are deferred); never a panic.
    #[test]
    fn ats_unsupported_size_errors() {
        let mut coeffs = vec![0i32; 16 * 16];
        assert!(inverse_transform_ats(&mut coeffs, 16, 16, 1, 1).is_err());
        assert!(inverse_transform_ats(&mut coeffs, 16, 16, 2, 0).is_err());
    }

    /// 8x8 zero coefficients yield zero residuals.
    #[test]
    fn inverse_transform_zero_8x8() {
        let mut coeffs = vec![0i32; 64];
        inverse_transform(&mut coeffs, 8, 8).unwrap();
        assert!(coeffs.iter().all(|&v| v == 0));
    }

    /// 2x2 inverse DCT — just verify the matrix multiply is right.
    /// coeffs = [1, 0; 0, 0] →
    ///   col 0 = [1, 0]; row 0 of mat_2 = [64, 64]
    ///   so vert pass column 0 = [64, 64]; column 1 = 0.
    /// After vert pass: row 0 = [64, 0]; row 1 = [64, 0].
    /// horiz pass per row: row 0 → [64*64, 64*64] = [4096, 4096].
    #[test]
    fn inverse_transform_2x2() {
        let mut coeffs = vec![1, 0, 0, 0];
        inverse_transform(&mut coeffs, 2, 2).unwrap();
        for v in &coeffs {
            assert_eq!(*v, 4096);
        }
    }

    /// 32x32 size: ensure mat_32() construction yields the right shape
    /// and the DC row is all 64s.
    #[test]
    fn mat_32_initialises() {
        let m = mat_32();
        assert_eq!(m.len(), 32 * 32);
        for (c, v) in m.iter().enumerate().take(32) {
            assert_eq!(*v, 64, "row 0 col {c} must be 64");
        }
    }

    /// A 16x16 zero coefficient block also produces zero output.
    #[test]
    fn inverse_transform_zero_16x16() {
        let mut coeffs = vec![0i32; 256];
        inverse_transform(&mut coeffs, 16, 16).unwrap();
        assert!(coeffs.iter().all(|&v| v == 0));
    }

    /// Round-5: 64×64 transform supported.
    #[test]
    fn inverse_transform_zero_64x64() {
        let mut coeffs = vec![0i32; 64 * 64];
        inverse_transform(&mut coeffs, 64, 64).unwrap();
        assert!(coeffs.iter().all(|&v| v == 0));
    }

    /// 64×64 DC-only at (0,0) → coeffs[y][x] = mat[x][0] * mat[y][0]
    /// (cascaded matrix multiplication, no renorm — dequant adds the
    /// bit-depth shift). Storage convention: `coeffs[y*64+x]` (row-major
    /// with x as inner dimension).
    #[test]
    fn inverse_transform_dc_only_64x64() {
        let mut coeffs = vec![0i32; 64 * 64];
        coeffs[0] = 1; // (y=0, x=0)
        inverse_transform(&mut coeffs, 64, 64).unwrap();
        // After cascade: coeffs[y*64 + x] = mat_64[x][0] * mat_64[y][0].
        // mat[0][0]=64, mat[1][0]=90, mat[63][0]=2.
        assert_eq!(coeffs[0], 64 * 64); // (y=0, x=0)
        assert_eq!(coeffs[1], 90 * 64); // (y=0, x=1) = AC1 first col times DC
        assert_eq!(coeffs[63], 2 * 64); // (y=0, x=63)
        assert_eq!(coeffs[64], 64 * 90); // (y=1, x=0)
        assert_eq!(coeffs[63 * 64], 64 * 2); // (y=63, x=0)
        assert_eq!(coeffs[63 * 64 + 63], 2 * 2); // (y=63, x=63)
    }

    /// Unsupported size (3) — 64 is now supported.
    #[test]
    fn rejects_unsupported_size() {
        let mut coeffs = vec![0i32; 9];
        let err = inverse_transform(&mut coeffs, 3, 3).unwrap_err();
        assert!(format!("{err}").contains("nTbS"));
    }

    /// 64×64 matrix DC row is all 64s (trivial spec sanity).
    #[test]
    fn mat_64_dc_row_is_64s() {
        let m = mat_64();
        assert_eq!(m.len(), 64 * 64);
        for (c, &v) in m.iter().take(64).enumerate() {
            assert_eq!(v, 64, "row 0 col {c} must be 64");
        }
        // Spec eq. 1072 row 1 col 0..15 should match.
        let row1_col0_15 = [
            90, 90, 90, 89, 88, 87, 86, 84, 83, 81, 79, 76, 74, 71, 69, 66,
        ];
        for (c, &expect) in row1_col0_15.iter().enumerate() {
            assert_eq!(m[64 + c], expect, "row 1 col {c}");
        }
        // Spec eq. 1074 row 1 col 16..31.
        let row1_col16_31 = [62, 59, 56, 52, 48, 45, 41, 37, 33, 28, 24, 20, 15, 11, 7, 2];
        for (c, &expect) in row1_col16_31.iter().enumerate() {
            assert_eq!(m[64 + 16 + c], expect, "row 1 col {}", 16 + c);
        }
    }

    /// Length mismatch on the input buffer.
    #[test]
    fn rejects_wrong_buffer_len() {
        let mut coeffs = vec![0i32; 5];
        let err = inverse_transform(&mut coeffs, 4, 4).unwrap_err();
        assert!(format!("{err}").contains("buffer len"));
    }

    /// 32x32 zero coefficients yield zero output.
    #[test]
    fn inverse_transform_zero_32x32() {
        let mut coeffs = vec![0i32; 1024];
        inverse_transform(&mut coeffs, 32, 32).unwrap();
        assert!(coeffs.iter().all(|&v| v == 0));
    }

    /// 32x32 DC-only at (0,0) → top-left output equals 64 * 64 * dc =
    /// 4096 (no renorm here; dequant adds the bit-depth shift).
    #[test]
    fn inverse_transform_dc_only_32x32() {
        let mut coeffs = vec![0i32; 1024];
        coeffs[0] = 1;
        inverse_transform(&mut coeffs, 32, 32).unwrap();
        assert_eq!(coeffs[0], 4096);
    }
}
