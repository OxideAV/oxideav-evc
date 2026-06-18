//! EVC Main-profile **extended intra prediction** (EIPD) sample-derivation
//! modes (ISO/IEC 23094-1:2020 §8.4.4.8 / §8.4.4.9 / §8.4.4.10).
//!
//! When `sps_eipd_flag == 1` the intra mode set grows from the five
//! Baseline modes (`intra.rs`) to the full EIPD set of Table 15:
//!
//! | mode | name        | source                       |
//! |------|-------------|------------------------------|
//! | 0    | INTRA_DC    | §8.4.4.3 (Baseline, eipd=1 eq.) |
//! | 1    | INTRA_PLN   | §8.4.4.9                     |
//! | 2    | INTRA_BI    | §8.4.4.8                     |
//! | 3..32| directional | §8.4.4.10 (Table 20)         |
//!
//! with the named anchors `INTRA_DIA_L = 6`, `INTRA_VER = 12`,
//! `INTRA_DIA_R = 18`, `INTRA_HOR = 24`, `INTRA_DIA_U = 30`.
//!
//! This module implements the *prediction-sample derivation* for the three
//! EIPD-specific families (`INTRA_BI`, `INTRA_PLN`, the 33 directional
//! modes). The mode-index syntax/derivation (§8.4.2 MPM list) and the
//! reference-sample construction/substitution (§8.4.4.1 / §8.4.4.2) are
//! the caller's responsibility; this module consumes a ready
//! [`EipdRefSamples`] neighbourhood and writes `predSamples` row-major.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

/// Named EIPD intra-mode anchors (Table 15, `sps_eipd_flag == 1`).
pub const INTRA_DC: i32 = 0;
pub const INTRA_PLN: i32 = 1;
pub const INTRA_BI: i32 = 2;
pub const INTRA_DIA_L: i32 = 6;
pub const INTRA_VER: i32 = 12;
pub const INTRA_DIA_R: i32 = 18;
pub const INTRA_HOR: i32 = 24;
pub const INTRA_DIA_U: i32 = 30;

/// §6.4 left/right neighbour availability (`availLR`, eq. 23).
///
/// `availLR = availableL + availableR * 2`, so the four codes are:
/// `LR_00 = 0` (neither), `LR_10 = 1` (left only), `LR_01 = 2` (right
/// only), `LR_11 = 3` (both). When `sps_suco_flag == 0` only `LR_00`
/// and `LR_10` occur.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvailLr {
    /// `LR_00` — neither left nor right available.
    Lr00,
    /// `LR_10` — left available, right not.
    Lr10,
    /// `LR_01` — right available, left not.
    Lr01,
    /// `LR_11` — both available.
    Lr11,
}

impl AvailLr {
    /// Decode the eq. (23) numeric code (`availableL + availableR * 2`).
    pub fn from_code(code: u32) -> Self {
        match code & 3 {
            0 => Self::Lr00,
            1 => Self::Lr10,
            2 => Self::Lr01,
            _ => Self::Lr11,
        }
    }
}

/// Table 17 — `divScaleMult[ idx ]` for `idx = 0..7` (INTRA_BI scaling).
const DIV_SCALE_MULT: [i64; 8] = [2048, 1365, 819, 455, 241, 124, 63, 32];
/// §8.4.4.8 normalisation factor `divScaleShift`.
const DIV_SCALE_SHIFT: i64 = 12;

/// Table 18 — `weightFactor[ absLog2DiffWH ]` for `absLog2DiffWH = 1..5`.
/// Index 0 is unused (the lookup is keyed on `absLog2DiffWH >= 1`).
const WEIGHT_FACTOR: [i64; 6] = [0, 341, 205, 114, 60, 31];

/// Table 19 — `mult[ i ]` / `shift[ i ]` for `i = 2..7` (INTRA_PLN).
/// Indices 0/1 are unused (`idxW`/`idxH = Max( log2, 2 )`).
const PLN_MULT: [i64; 8] = [0, 0, 13, 17, 5, 11, 23, 47];
const PLN_SHIFT: [i64; 8] = [0, 0, 7, 10, 11, 15, 19, 23];

/// Table 20 — `(dirXYSign, divDxy, divDyx)` for the directional modes
/// `predModeIntra = 3..32`. Modes 12 (`INTRA_VER`) and 24 (`INTRA_HOR`)
/// are the "dash" rows: they are dispatched specially before this table
/// is consulted, so their tuples here are never read (filled with the
/// neutral `(0, 0, 0)`).
const DIR_TABLE: [(i32, i64, i64); 30] = [
    /* 3 */ (-1, 2816, 372),
    /* 4 */ (-1, 2048, 512),
    /* 5 */ (-1, 1408, 744),
    /* 6 */ (-1, 1024, 1024),
    /* 7 */ (-1, 744, 1408),
    /* 8 */ (-1, 512, 2048),
    /* 9 */ (-1, 372, 2816),
    /* 10 */ (-1, 256, 4096),
    /* 11 */ (-1, 128, 8192),
    /* 12 */ (0, 0, 0), // INTRA_VER — dispatched specially
    /* 13 */ (1, 128, 8192),
    /* 14 */ (1, 256, 4096),
    /* 15 */ (1, 372, 2816),
    /* 16 */ (1, 512, 2048),
    /* 17 */ (1, 744, 1408),
    /* 18 */ (1, 1024, 1024),
    /* 19 */ (1, 1408, 744),
    /* 20 */ (1, 2048, 512),
    /* 21 */ (1, 2816, 372),
    /* 22 */ (1, 4096, 256),
    /* 23 */ (1, 8192, 128),
    /* 24 */ (0, 0, 0), // INTRA_HOR — dispatched specially
    /* 25 */ (-1, 8192, 128),
    /* 26 */ (-1, 4096, 256),
    /* 27 */ (-1, 2816, 372),
    /* 28 */ (-1, 2048, 512),
    /* 29 */ (-1, 1408, 744),
    /* 30 */ (-1, 1024, 1024),
    /* 31 */ (-1, 744, 1408),
    /* 32 */ (-1, 512, 2048),
];

/// Reference-sample neighbourhood for EIPD prediction.
///
/// Mirrors the §8.4.4.1 sample array `p[ x ][ y ]`. The EIPD modes index
/// three half-open spans plus the four corners:
///
/// * `top[i]   = p[ i ][ -1 ]`   for `i = -1 .. nCbW + nCbH - 1`
/// * `left[j]  = p[ -1 ][ j ]`   for `j = -1 .. nCbH + nCbW - 1`
/// * `right[j] = p[ nCbW ][ j ]` for `j = -1 .. nCbH + nCbW - 1`
///   (only populated / read on the SUCO `sps_suco_flag == 1` path).
///
/// To keep `p[ x ][ -1 ]` and `p[ -1 ][ y ]` addressable with a `-1`
/// origin, each vector stores the `-1` element first: index `k` in the
/// vector holds `p[ k - 1 ][ -1 ]` (for `top`) or `p[ -1 ][ k - 1 ]`
/// (for `left` / `right`). The [`top`](Self::top) / [`left`](Self::left)
/// / [`right`](Self::right) accessors take the spec index (which may be
/// `-1`) and apply the offset.
pub struct EipdRefSamples {
    /// `p[ x ][ -1 ]` for `x = -1 .. nCbW + nCbH - 1` (offset +1).
    top_buf: Vec<i32>,
    /// `p[ -1 ][ y ]` for `y = -1 .. nCbH + nCbW - 1` (offset +1).
    left_buf: Vec<i32>,
    /// `p[ nCbW ][ y ]` for `y = -1 .. nCbH + nCbW - 1` (offset +1).
    right_buf: Vec<i32>,
}

impl EipdRefSamples {
    /// Allocate a neighbourhood pre-filled with the `1 << (bitDepth - 1)`
    /// "not available" substitute (§8.4.4.2). Overlay real samples with
    /// [`set_top`](Self::set_top) / [`set_left`](Self::set_left) /
    /// [`set_right`](Self::set_right) / [`set_top_left`](Self::set_top_left).
    pub fn unavailable(n_cb_w: usize, n_cb_h: usize, bit_depth: u32) -> Self {
        let fill = 1i32 << (bit_depth - 1);
        // length = 1 (for the -1 slot) + (nCbW + nCbH) span.
        let len = 1 + n_cb_w + n_cb_h;
        Self {
            top_buf: vec![fill; len],
            left_buf: vec![fill; len],
            right_buf: vec![fill; len],
        }
    }

    /// `p[ x ][ -1 ]` (top row), `x ∈ -1 .. nCbW + nCbH - 1`.
    #[inline]
    pub fn top(&self, x: i32) -> i32 {
        self.top_buf[(x + 1) as usize]
    }

    /// `p[ -1 ][ y ]` (left column), `y ∈ -1 .. nCbH + nCbW - 1`.
    #[inline]
    pub fn left(&self, y: i32) -> i32 {
        self.left_buf[(y + 1) as usize]
    }

    /// `p[ nCbW ][ y ]` (right column), `y ∈ -1 .. nCbH + nCbW - 1`.
    #[inline]
    pub fn right(&self, y: i32) -> i32 {
        self.right_buf[(y + 1) as usize]
    }

    /// `p[ -1 ][ -1 ]` corner.
    #[inline]
    pub fn top_left(&self) -> i32 {
        self.top_buf[0]
    }

    /// Overlay `p[ x ][ -1 ] = v`, `x ∈ -1 .. nCbW + nCbH - 1`.
    pub fn set_top(&mut self, x: i32, v: i32) {
        self.top_buf[(x + 1) as usize] = v;
    }

    /// Overlay `p[ -1 ][ y ] = v`, `y ∈ -1 .. nCbH + nCbW - 1`.
    pub fn set_left(&mut self, y: i32, v: i32) {
        self.left_buf[(y + 1) as usize] = v;
    }

    /// Overlay `p[ nCbW ][ y ] = v`, `y ∈ -1 .. nCbH + nCbW - 1`.
    pub fn set_right(&mut self, y: i32, v: i32) {
        self.right_buf[(y + 1) as usize] = v;
    }

    /// Overlay the `p[ -1 ][ -1 ]` corner (writes both the top and left
    /// `-1` slots, which both denote that corner).
    pub fn set_top_left(&mut self, v: i32) {
        self.top_buf[0] = v;
        self.left_buf[0] = v;
    }
}

#[inline]
fn clip3(lo: i64, hi: i64, v: i64) -> i64 {
    v.clamp(lo, hi)
}

#[inline]
fn log2_usize(v: usize) -> i64 {
    (usize::BITS - 1 - v.leading_zeros()) as i64
}

/// Top-level EIPD prediction dispatch (§8.4.4.1 ordered branch into
/// §8.4.4.8 / §8.4.4.9 / §8.4.4.10).
///
/// `pred_mode_intra` is the EIPD mode index (0..32). `dst` is written
/// row-major (`n_cb_w` columns, `n_cb_h` rows).
pub fn predict_eipd(
    pred_mode_intra: i32,
    refs: &EipdRefSamples,
    n_cb_w: usize,
    n_cb_h: usize,
    bit_depth: u32,
    avail_lr: AvailLr,
    dst: &mut [i32],
) {
    debug_assert_eq!(dst.len(), n_cb_w * n_cb_h);
    match pred_mode_intra {
        INTRA_DC => predict_dc_eipd(refs, n_cb_w, n_cb_h, avail_lr, dst),
        INTRA_PLN => predict_pln(refs, n_cb_w, n_cb_h, bit_depth, avail_lr, dst),
        INTRA_BI => predict_bi(refs, n_cb_w, n_cb_h, bit_depth, avail_lr, dst),
        _ => predict_directional(
            pred_mode_intra,
            refs,
            n_cb_w,
            n_cb_h,
            bit_depth,
            avail_lr,
            dst,
        ),
    }
}

/// §8.4.4.3 INTRA_DC, `sps_eipd_flag == 1` path (eqs. 286/287/288).
///
/// Because the top sum spans `nCbW` samples and the side sum spans
/// `nCbH` samples, the total is in general not a power of two, so the
/// average is taken via the `divScaleMult` reciprocal-scaling table
/// (Table 17) keyed on the aspect-ratio index, rather than a single
/// shift. `availLR` selects which side column joins the average and
/// the `log2AspRatio` / `aspRatioShift` formulation.
fn predict_dc_eipd(
    refs: &EipdRefSamples,
    n_cb_w: usize,
    n_cb_h: usize,
    avail_lr: AvailLr,
    dst: &mut [i32],
) {
    let log2_w = log2_usize(n_cb_w);
    let log2_h = log2_usize(n_cb_h);
    let w = n_cb_w as i64;
    let h = n_cb_h as i64;

    let sum_top: i64 = (0..n_cb_w as i32).map(|x| refs.top(x) as i64).sum();
    let sum_left: i64 = (0..n_cb_h as i32).map(|y| refs.left(y) as i64).sum();
    let sum_right: i64 = (0..n_cb_h as i32).map(|y| refs.right(y) as i64).sum();

    let dc = match avail_lr {
        AvailLr::Lr11 => {
            // eq. (286)
            let asp_ratio_shift = if w > 2 * h { log2_h + 1 } else { log2_w };
            let log2_asp_ratio = if w > 2 * h {
                log2_w - log2_h - 1
            } else {
                log2_h - log2_w + 1
            };
            let num = sum_top + sum_left + sum_right;
            ((num + ((w + h + h) >> 1)) * DIV_SCALE_MULT[log2_asp_ratio as usize])
                >> (DIV_SCALE_SHIFT + asp_ratio_shift)
        }
        AvailLr::Lr01 => {
            // eq. (287) — uses right column instead of left.
            let asp_ratio_shift = if w > h { log2_h } else { log2_w };
            let log2_asp_ratio = if w > h {
                log2_w - log2_h
            } else {
                log2_h - log2_w
            };
            ((sum_top + sum_right + ((w + h) >> 1)) * DIV_SCALE_MULT[log2_asp_ratio as usize])
                >> (DIV_SCALE_SHIFT + asp_ratio_shift)
        }
        AvailLr::Lr10 | AvailLr::Lr00 => {
            // eq. (288)
            let asp_ratio_shift = if w > h { log2_h } else { log2_w };
            let log2_asp_ratio = if w > h {
                log2_w - log2_h
            } else {
                log2_h - log2_w
            };
            ((sum_top + sum_left + ((w + h) >> 1)) * DIV_SCALE_MULT[log2_asp_ratio as usize])
                >> (DIV_SCALE_SHIFT + asp_ratio_shift)
        }
    };
    let dc = dc as i32;
    for v in dst.iter_mut() {
        *v = dc;
    }
}

/// §8.4.4.8 INTRA_BI (bilinear). `availLR` selects between the
/// right-reference forms (LR_11 / LR_01, eqs. 297/304) and the
/// left-reference forms (LR_10 / LR_00, eq. 311).
fn predict_bi(
    refs: &EipdRefSamples,
    n_cb_w: usize,
    n_cb_h: usize,
    bit_depth: u32,
    avail_lr: AvailLr,
    dst: &mut [i32],
) {
    let max_val = (1i64 << bit_depth) - 1;
    let log2_w = log2_usize(n_cb_w);
    let log2_h = log2_usize(n_cb_h);
    let w = n_cb_w as i64;
    let h = n_cb_h as i64;

    match avail_lr {
        AvailLr::Lr11 => {
            // eq. (297) — bilinear from left + right columns and the top row.
            let div_mult = DIV_SCALE_MULT[log2_w as usize];
            for y in 0..n_cb_h {
                let yi = y as i64;
                let p_l = refs.left(y as i32) as i64; // p[-1][y]
                let p_r = refs.right(y as i32) as i64; // p[nCbW][y]
                                                       // p[-1][nCbH-1], p[nCbW][nCbH-1]
                let p_l_b = refs.left(n_cb_h as i32 - 1) as i64;
                let p_r_b = refs.right(n_cb_h as i32 - 1) as i64;
                for x in 0..n_cb_w {
                    let xi = x as i64;
                    let p_t = refs.top(x as i32) as i64; // p[x][-1]
                    let horiz_top = ((p_l * (w - xi) + p_r * (xi + 1) + (w >> 1)) * div_mult)
                        >> DIV_SCALE_SHIFT;
                    let horiz_bot = ((p_l_b * (w - xi) + p_r_b * (xi + 1) + (w >> 1)) * div_mult)
                        >> DIV_SCALE_SHIFT;
                    let vert = (p_t * (h - 1 - yi) + horiz_bot * (yi + 1) + (h >> 1)) >> log2_h;
                    let val = (horiz_top + vert + 1) >> 1;
                    dst[y * n_cb_w + x] = clip3(0, max_val, val) as i32;
                }
            }
        }
        AvailLr::Lr01 => {
            // eqs. (298)-(304) — corner-anchored plane from right column.
            let i_a = refs.top_left() as i64; // p[-1][-1]
            let i_b = refs.right(n_cb_h as i32) as i64; // p[nCbW][nCbH]
            let i_c = bi_ic(i_a, i_b, n_cb_w, n_cb_h, log2_w, log2_h);
            for y in 0..n_cb_h {
                let yi = y as i64;
                for x in 0..n_cb_w {
                    let xi = x as i64;
                    let p_t = refs.top(x as i32) as i64; // p[x][-1]
                    let p_r = refs.right(y as i32) as i64; // p[nCbW][y]
                    let val = ((((i_a - p_r) * (xi + 1)) << log2_h)
                        + (((i_b - p_t) * (yi + 1)) << log2_w)
                        + ((p_t + p_r) << (log2_w + log2_h))
                        + ((i_c << 1) - i_a - i_b) * xi * yi
                        + (1i64 << (log2_w + log2_h)))
                        >> (log2_w + log2_h + 1);
                    dst[y * n_cb_w + x] = clip3(0, max_val, val) as i32;
                }
            }
        }
        AvailLr::Lr10 | AvailLr::Lr00 => {
            // eqs. (305)-(311) — corner-anchored plane from left column.
            let i_a = refs.top(n_cb_w as i32) as i64; // p[nCbW][-1]
            let i_b = refs.left(n_cb_h as i32) as i64; // p[-1][nCbH]
            let i_c = bi_ic(i_a, i_b, n_cb_w, n_cb_h, log2_w, log2_h);
            for y in 0..n_cb_h {
                let yi = y as i64;
                for x in 0..n_cb_w {
                    let xi = x as i64;
                    let p_t = refs.top(x as i32) as i64; // p[x][-1]
                    let p_l = refs.left(y as i32) as i64; // p[-1][y]
                    let val = ((((i_a - p_l) * (xi + 1)) << log2_h)
                        + (((i_b - p_t) * (yi + 1)) << log2_w)
                        + ((p_t + p_l) << (log2_w + log2_h))
                        + ((i_c << 1) - i_a - i_b) * xi * yi
                        + (1i64 << (log2_w + log2_h)))
                        >> (log2_w + log2_h + 1);
                    dst[y * n_cb_w + x] = clip3(0, max_val, val) as i32;
                }
            }
        }
    }
}

/// INTRA_BI corner-interpolation coefficient `iC` (eqs. 300/303 and
/// 307/310 — identical form for both LR cases).
fn bi_ic(i_a: i64, i_b: i64, n_cb_w: usize, n_cb_h: usize, log2_w: i64, log2_h: i64) -> i64 {
    if n_cb_w == n_cb_h {
        (i_a + i_b + 1) >> 1
    } else {
        let i_shift = log2_w.min(log2_h);
        let abs_diff = (log2_w - log2_h).unsigned_abs() as usize;
        let wf = WEIGHT_FACTOR[abs_diff];
        (((i_a << log2_w) + (i_b << log2_h)) * wf + (1i64 << (i_shift + 9))) >> (i_shift + 10)
    }
}

/// §8.4.4.9 INTRA_PLN (planar). `availLR` selects the right-anchored
/// (LR_11 / LR_01, eqs. 314-319) vs. left-anchored (LR_10 / LR_00,
/// eqs. 320-325) gradient accumulation.
fn predict_pln(
    refs: &EipdRefSamples,
    n_cb_w: usize,
    n_cb_h: usize,
    bit_depth: u32,
    avail_lr: AvailLr,
    dst: &mut [i32],
) {
    let max_val = (1i64 << bit_depth) - 1;
    let log2_w = log2_usize(n_cb_w);
    let log2_h = log2_usize(n_cb_h);
    let idx_w = log2_w.max(2) as usize;
    let idx_h = log2_h.max(2) as usize;
    let half_w = (n_cb_w / 2) as i32;
    let half_h = (n_cb_h / 2) as i32;

    let right = matches!(avail_lr, AvailLr::Lr11 | AvailLr::Lr01);

    // iH / iV accumulators (eqs. 314/315 vs. 320/321).
    let mut i_h: i64 = 0;
    for xp in 0..half_w {
        let term = if right {
            // p[(nCbW/2) - xp - 1][-1] - p[(nCbW/2) + xp + 1][-1]
            refs.top(half_w - xp - 1) as i64 - refs.top(half_w + xp + 1) as i64
        } else {
            // p[(nCbW/2) + xp][-1] - p[(nCbW/2) - xp - 2][-1]
            refs.top(half_w + xp) as i64 - refs.top(half_w - xp - 2) as i64
        };
        i_h += (xp as i64 + 1) * term;
    }
    let mut i_v: i64 = 0;
    for yp in 0..half_h {
        let term = if right {
            // p[nCbW][(nCbH/2) + yp] - p[nCbW][(nCbH/2) - yp - 2]
            refs.right(half_h + yp) as i64 - refs.right(half_h - yp - 2) as i64
        } else {
            // p[-1][(nCbH/2) + yp] - p[-1][(nCbH/2) - yp - 2]
            refs.left(half_h + yp) as i64 - refs.left(half_h - yp - 2) as i64
        };
        i_v += (yp as i64 + 1) * term;
    }

    // iA (eq. 316 vs. 322).
    let i_a = if right {
        (refs.top(0) as i64 + refs.right(n_cb_h as i32 - 1) as i64) << 4
    } else {
        (refs.top(n_cb_w as i32 - 1) as i64 + refs.left(n_cb_h as i32 - 1) as i64) << 4
    };

    // iB / iC (eqs. 317/318, identical form for both LR cases).
    let mult_h = PLN_MULT[idx_h];
    let shift_h = PLN_SHIFT[idx_h];
    let mult_w = PLN_MULT[idx_w];
    let shift_w = PLN_SHIFT[idx_w];
    let i_b = ((i_h << 5) * mult_h + (1i64 << (shift_h - 1))) >> shift_h;
    let i_c = ((i_v << 5) * mult_w + (1i64 << (shift_w - 1))) >> shift_w;

    let off_x = (n_cb_w >> 1) as i64 - 1;
    let off_y = (n_cb_h >> 1) as i64 - 1;
    for y in 0..n_cb_h {
        let yi = y as i64;
        for x in 0..n_cb_w {
            let xi = x as i64;
            let val = (i_a + (xi - off_x) * i_b + (yi - off_y) * i_c + 16) >> 5;
            dst[y * n_cb_w + x] = clip3(0, max_val, val) as i32;
        }
    }
}

/// §8.4.4.10 directional modes (`predModeIntra = 3..32`, excluding the
/// `INTRA_VER`/`INTRA_HOR` special anchors handled separately below).
///
/// Implements the full two-step process: step 1 derives
/// `iOffset / iX / iY / refPosition` per the `availLR` + mode quadrant
/// branch (eqs. 326-364); step 2 applies the 4-tap fractional filter
/// against the selected reference array (eqs. 365-385).
#[allow(clippy::too_many_arguments)]
fn predict_directional(
    pred_mode_intra: i32,
    refs: &EipdRefSamples,
    n_cb_w: usize,
    n_cb_h: usize,
    bit_depth: u32,
    avail_lr: AvailLr,
    dst: &mut [i32],
) {
    // INTRA_VER (12) and INTRA_HOR (24) are pure copies, dispatched here
    // before consulting the Table 20 (their rows are dashes).
    if pred_mode_intra == INTRA_VER {
        for y in 0..n_cb_h {
            for x in 0..n_cb_w {
                dst[y * n_cb_w + x] = refs.top(x as i32);
            }
        }
        return;
    }
    if pred_mode_intra == INTRA_HOR {
        for y in 0..n_cb_h {
            let v = refs.left(y as i32);
            for x in 0..n_cb_w {
                dst[y * n_cb_w + x] = v;
            }
        }
        return;
    }

    let (dir_xy_sign, div_dxy, div_dyx) = DIR_TABLE[(pred_mode_intra - 3) as usize];
    let max_val = (1i64 << bit_depth) - 1;
    let w = n_cb_w as i64;
    let clip_max = (n_cb_w + n_cb_h - 1) as i64;
    let clip_min = -1i64;

    let lr_right_branch = matches!(avail_lr, AvailLr::Lr01 | AvailLr::Lr11);

    for y in 0..n_cb_h {
        let yi = y as i64;
        for x in 0..n_cb_w {
            let xi = x as i64;
            // Step 1: derive iOffset, iX/iY, refPosition.
            let (i_offset, ref_pos, i_coord) = if lr_right_branch {
                dir_step1_lr_right(pred_mode_intra, avail_lr, xi, yi, w, div_dxy, div_dyx)
            } else {
                dir_step1_lr_left(pred_mode_intra, xi, yi, div_dxy, div_dyx)
            };

            // Step 2: 4-tap fractional filter (eqs. 365-385).
            let val = dir_step2(
                refs,
                ref_pos,
                i_coord,
                i_offset,
                dir_xy_sign,
                clip_min,
                clip_max,
            );
            dst[y * n_cb_w + x] = clip3(0, max_val, val) as i32;
        }
    }
}

/// Which reference array the directional filter taps (eqs. 330/333/...).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RefPos {
    Up,
    Left,
    Right,
}

/// Step-1 derivation for the `availLR ∈ { LR_01, LR_11 }` branch
/// (eqs. 326-350). Returns `(iOffset, refPosition, iCoord)` where
/// `iCoord` is `iX` for `refPosition == Up` and `iY` for Left/Right.
fn dir_step1_lr_right(
    pred_mode_intra: i32,
    avail_lr: AvailLr,
    xi: i64,
    yi: i64,
    w: i64,
    div_dxy: i64,
    div_dyx: i64,
) -> (i64, RefPos, i64) {
    let i_tan_y = ((yi + 1) * div_dxy) >> 10;

    if pred_mode_intra < INTRA_VER {
        if xi < w - i_tan_y {
            // eqs. 328-330 — upper sample.
            let i_offset = (((yi + 1) * div_dxy) >> 5) - (i_tan_y << 5);
            let i_x = xi + i_tan_y;
            (i_offset, RefPos::Up, i_x)
        } else {
            // eqs. 331-333 — right sample.
            let i_offset = (((w - xi) * div_dyx) >> 5) - ((((w - xi) * div_dyx) >> 10) << 5);
            let i_y = yi + (((w - xi) * div_dyx) >> 10);
            (i_offset, RefPos::Right, i_y)
        }
    } else if pred_mode_intra > INTRA_HOR {
        // eqs. 334/335 recompute iTanX/iTanY from (nCbW - x).
        let i_tan_x = ((w - xi) * div_dyx) >> 10;
        let i_tan_y2 = ((w - xi) * div_dxy) >> 10;
        if yi < i_tan_x {
            // eqs. 336-338 — upper sample.
            let i_offset = (((w - xi) * div_dxy) >> 5) - (i_tan_y2 << 5);
            let i_x = xi + i_tan_y2;
            (i_offset, RefPos::Up, i_x)
        } else {
            // eqs. 339-341 — right sample.
            let i_offset = (((w - xi) * div_dxy) >> 5) - (i_tan_x << 5);
            let i_y = yi - i_tan_x;
            (i_offset, RefPos::Right, i_y)
        }
    } else {
        // 3 <= predModeIntra <= INTRA_HOR (and >= INTRA_VER): the
        // "Otherwise" arm (eqs. 342-350). iTanX here is the eq. 327 form.
        let i_tan_x = ((xi + 1) * div_dyx) >> 10;
        if yi < i_tan_x {
            // eqs. 342-344 — upper sample.
            let i_offset = (((yi + 1) * div_dxy) >> 5) - (i_tan_y << 5);
            let i_x = xi - i_tan_y;
            (i_offset, RefPos::Up, i_x)
        } else if avail_lr == AvailLr::Lr01 {
            // eqs. 345-347 — right sample.
            let i_offset = (((w - xi) * div_dxy) >> 5) - ((((w - xi) * div_dyx) >> 10) << 5);
            let i_y = yi + (((w - xi) * div_dyx) >> 10);
            (i_offset, RefPos::Right, i_y)
        } else {
            // eqs. 348-350 — left sample.
            let i_offset = (((xi + 1) * div_dyx) >> 5) - (i_tan_x << 5);
            let i_y = yi - i_tan_x;
            (i_offset, RefPos::Left, i_y)
        }
    }
}

/// Step-1 derivation for the `availLR ∈ { LR_10, LR_00 }` branch
/// (eqs. 351-364).
fn dir_step1_lr_left(
    pred_mode_intra: i32,
    xi: i64,
    yi: i64,
    div_dxy: i64,
    div_dyx: i64,
) -> (i64, RefPos, i64) {
    let i_tan_y = ((yi + 1) * div_dxy) >> 10;
    let i_tan_x = ((xi + 1) * div_dyx) >> 10;

    if pred_mode_intra < INTRA_VER {
        // eqs. 353-355 — upper sample.
        let i_offset = (((yi + 1) * div_dxy) >> 5) - (i_tan_y << 5);
        let i_x = xi + i_tan_y;
        (i_offset, RefPos::Up, i_x)
    } else if pred_mode_intra > INTRA_HOR {
        // eqs. 356-358 — left sample.
        let i_offset = (((xi + 1) * div_dyx) >> 5) - (i_tan_x << 5);
        let i_y = yi + i_tan_x;
        (i_offset, RefPos::Left, i_y)
    } else if yi < i_tan_x {
        // eqs. 359-361 — upper sample.
        let i_offset = (((yi + 1) * div_dxy) >> 5) - (i_tan_y << 5);
        let i_x = xi - i_tan_y;
        (i_offset, RefPos::Up, i_x)
    } else {
        // eqs. 362-364 — left sample.
        let i_offset = (((xi + 1) * div_dyx) >> 5) - (i_tan_x << 5);
        let i_y = yi - i_tan_x;
        (i_offset, RefPos::Left, i_y)
    }
}

/// Step-2 4-tap fractional filter (eqs. 365-385). `i_coord` is `iX` for
/// `RefPos::Up`, `iY` for `RefPos::Left` / `RefPos::Right`.
#[allow(clippy::too_many_arguments)]
fn dir_step2(
    refs: &EipdRefSamples,
    ref_pos: RefPos,
    i_coord: i64,
    i_offset: i64,
    dir_xy_sign: i32,
    clip_min: i64,
    clip_max: i64,
) -> i64 {
    // Tap offsets depend on dirXYSign and refPosition (eqs. 365-384).
    // For Up + Left: sign==-1 → {+1,+2,-1}; else → {-1,-2,+1}.
    // For Right:     sign==-1 → {-1,-2,+1}; else → {+1,+2,-1}.
    let (d_n, d_p2, d_nn1) = match ref_pos {
        RefPos::Right => {
            if dir_xy_sign == -1 {
                (-1, -2, 1)
            } else {
                (1, 2, -1)
            }
        }
        _ => {
            if dir_xy_sign == -1 {
                (1, 2, -1)
            } else {
                (-1, -2, 1)
            }
        }
    };
    let i_n = clip3(clip_min, clip_max, i_coord + d_n);
    let i_n_p2 = clip3(clip_min, clip_max, i_coord + d_p2);
    let i_n_n1 = clip3(clip_min, clip_max, i_coord + d_nn1);
    let i_c = clip3(clip_min, clip_max, i_coord);

    let (s_nn1, s_c, s_n, s_np2) = match ref_pos {
        RefPos::Up => (
            refs.top(i_n_n1 as i32) as i64,
            refs.top(i_c as i32) as i64,
            refs.top(i_n as i32) as i64,
            refs.top(i_n_p2 as i32) as i64,
        ),
        RefPos::Left => (
            refs.left(i_n_n1 as i32) as i64,
            refs.left(i_c as i32) as i64,
            refs.left(i_n as i32) as i64,
            refs.left(i_n_p2 as i32) as i64,
        ),
        // refs.right already encodes p[ nCbW ][ · ].
        RefPos::Right => (
            refs.right(i_n_n1 as i32) as i64,
            refs.right(i_c as i32) as i64,
            refs.right(i_n as i32) as i64,
            refs.right(i_n_p2 as i32) as i64,
        ),
    };

    (s_nn1 * (32 - i_offset)
        + s_c * (64 - i_offset)
        + s_n * (32 + i_offset)
        + s_np2 * i_offset
        + 64)
        >> 7
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_refs(n_cb_w: usize, n_cb_h: usize, val: i32) -> EipdRefSamples {
        let mut r = EipdRefSamples::unavailable(n_cb_w, n_cb_h, 8);
        let span = (n_cb_w + n_cb_h) as i32;
        r.set_top_left(val);
        for i in -1..span {
            r.set_top(i, val);
            r.set_left(i, val);
            r.set_right(i, val);
        }
        r
    }

    #[test]
    fn avail_lr_codes() {
        assert_eq!(AvailLr::from_code(0), AvailLr::Lr00);
        assert_eq!(AvailLr::from_code(1), AvailLr::Lr10);
        assert_eq!(AvailLr::from_code(2), AvailLr::Lr01);
        assert_eq!(AvailLr::from_code(3), AvailLr::Lr11);
    }

    /// All EIPD modes must reproduce a flat reference field exactly.
    /// A constant neighbourhood is a fixed point of every linear
    /// predictor in the spec.
    #[test]
    fn all_modes_flat_field() {
        for &(w, h) in &[(8usize, 8usize), (16, 8), (8, 16), (4, 4)] {
            let refs = flat_refs(w, h, 100);
            for &lr in &[AvailLr::Lr00, AvailLr::Lr10, AvailLr::Lr01, AvailLr::Lr11] {
                for mode in 0..=32 {
                    let mut dst = vec![0i32; w * h];
                    predict_eipd(mode, &refs, w, h, 8, lr, &mut dst);
                    assert!(
                        dst.iter().all(|&v| v == 100),
                        "mode {mode} ({w}x{h}, {lr:?}) must reproduce flat 100, got {:?}",
                        &dst[..dst.len().min(8)]
                    );
                }
            }
        }
    }

    /// INTRA_VER (mode 12) copies the top row into every output row.
    #[test]
    fn ver_copies_top_row() {
        let mut refs = EipdRefSamples::unavailable(4, 4, 8);
        for x in 0..4i32 {
            refs.set_top(x, 40 + x);
        }
        let mut dst = vec![0i32; 16];
        predict_eipd(INTRA_VER, &refs, 4, 4, 8, AvailLr::Lr00, &mut dst);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 40 + x as i32);
            }
        }
    }

    /// INTRA_HOR (mode 24) copies the left column into every output column.
    #[test]
    fn hor_copies_left_column() {
        let mut refs = EipdRefSamples::unavailable(4, 4, 8);
        for y in 0..4i32 {
            refs.set_left(y, 70 + y);
        }
        let mut dst = vec![0i32; 16];
        predict_eipd(INTRA_HOR, &refs, 4, 4, 8, AvailLr::Lr00, &mut dst);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], 70 + y as i32);
            }
        }
    }

    /// DC averages top + left with the eipd=1 rounding/shift.
    #[test]
    fn dc_average() {
        // 4x4: top all 100, left all 200, right unused.
        let mut refs = EipdRefSamples::unavailable(4, 4, 8);
        for i in 0..4i32 {
            refs.set_top(i, 100);
            refs.set_left(i, 200);
        }
        let mut dst = vec![0i32; 16];
        predict_eipd(INTRA_DC, &refs, 4, 4, 8, AvailLr::Lr00, &mut dst);
        // sum = 4*100 + 4*200 = 1200; n = 8; (1200 + 4) >> 3 = 150.
        assert!(dst.iter().all(|&v| v == 150), "got {:?}", dst);
    }

    /// Outputs are always clipped into the valid 8-bit pixel range, even
    /// for steep directional gradients off a non-flat field.
    #[test]
    fn directional_in_range() {
        let mut refs = EipdRefSamples::unavailable(8, 8, 8);
        let span = 16i32;
        for i in -1..span {
            // a steep ramp on every reference array
            refs.set_top(i, ((i + 1) * 15).clamp(0, 255));
            refs.set_left(i, 255 - ((i + 1) * 15).clamp(0, 255));
            refs.set_right(i, ((i + 1) * 9).clamp(0, 255));
        }
        for mode in 3..=32 {
            for &lr in &[AvailLr::Lr00, AvailLr::Lr10, AvailLr::Lr01, AvailLr::Lr11] {
                let mut dst = vec![0i32; 64];
                predict_eipd(mode, &refs, 8, 8, 8, lr, &mut dst);
                assert!(
                    dst.iter().all(|&v| (0..=255).contains(&v)),
                    "mode {mode} {lr:?} produced out-of-range sample"
                );
            }
        }
    }
}
