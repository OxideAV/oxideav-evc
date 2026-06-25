//! EVC **Hadamard Transform Domain Filter** (HTDF), the §8.7.6
//! post-reconstruction filter (`sps_htdf_flag == 1`).
//!
//! When `sps_htdf_flag == 1` the §8.4.1 / §8.6 reconstruction invokes
//! this luma-only post-reconstruction filter on each coding block before
//! the in-loop deblocking / ALF stages. It runs a sliding 2×2 Hadamard
//! transform over the (padded) reconstructed luma, soft-thresholds the
//! three AC spectral coefficients through a QP-selected look-up table,
//! inverse-transforms and accumulates overlapping 2×2 contributions, then
//! rounds the four-fold accumulation back into the plane.
//!
//! The module splits into the three spec sub-processes:
//!
//! * §8.7.6.3 [`derive_htdf_lut`] — selects `(bLUT, aTHR, tblShift)` from
//!   the `setOfLUT[5][16]` / `tblThrLog2[5]` tables (eqs. 1106-1111)
//!   keyed on `QpY` and the inter-square-≥32 predicate.
//! * §8.7.6.2 [`pad_rec_samples`] — builds the `(nCbW+2)×(nCbH+2)`
//!   replicate-padded sample array, with the §6.4.1 availability /
//!   `constrained_intra_pred_flag` `dx`/`dy` clamp on the unavailable
//!   borders (eqs. of §8.7.6.2).
//! * §8.7.6.1 [`filter_block`] — the sliding-Hadamard accumulation +
//!   rounding (eqs. 1093-1105), plus the four §8.7.6.1 applicability
//!   gates ([`htdf_applies`]).
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).
//! The numeric look-up tables (`setOfLUT`, `tblThrLog2`) are transcribed
//! verbatim from §8.7.6.3 of the staged spec.

/// §8.7.6.3 `setOfLUT[5][16]` (eq. 1108) — the five QP-indexed
/// soft-threshold look-up tables.
const SET_OF_LUT: [[i32; 16]; 5] = [
    [0, 0, 2, 6, 10, 14, 19, 23, 28, 32, 36, 41, 45, 49, 53, 57],
    [
        0, 0, 5, 12, 20, 29, 38, 47, 56, 65, 73, 82, 90, 98, 107, 115,
    ],
    [0, 0, 1, 4, 9, 16, 24, 32, 41, 50, 59, 68, 77, 86, 94, 103],
    [
        0, 0, 3, 9, 19, 32, 47, 64, 81, 99, 117, 135, 154, 179, 205, 230,
    ],
    [
        0, 0, 0, 2, 6, 11, 18, 27, 38, 51, 64, 96, 128, 160, 192, 224,
    ],
];

/// §8.7.6.3 `tblThrLog2[5]` (eq. 1110).
const TBL_THR_LOG2: [i32; 5] = [6, 7, 7, 8, 8];

/// §8.7.6.1 scan template `scanTmpl = { (0,0), (0,1), (1,0), (1,1) }`
/// (eq. 1093). Each entry is `(scanTmpl[i][0], scanTmpl[i][1])` =
/// `(dy, dx)`.
const SCAN_TMPL: [(i32, i32); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

/// §8.7.6.3 look-up-table parameters for a given `QpY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HtdfLut {
    /// `bLUT` — the 16-entry soft-threshold table (eq. 1107).
    pub b_lut: [i32; 16],
    /// `aTHR` — look-up-table access threshold (eq. 1111).
    pub a_thr: i32,
    /// `tblShift` — table index shift (eq. 1109).
    pub tbl_shift: i32,
}

/// §8.7.6.3 — derive `(bLUT, aTHR, tblShift)` for the block (eqs.
/// 1106-1111).
///
/// `is_inter_square_ge32` is the eq.-1106 predicate
/// `LumaPredMode == MODE_INTER && nCbW == nCbH && Min(nCbW, nCbH) >= 32`.
pub fn derive_htdf_lut(qp_y: i32, is_inter_square_ge32: bool) -> HtdfLut {
    // eq. 1106: qpIdx selection.
    let qp_idx = if is_inter_square_ge32 {
        ((qp_y - 28 + (1 << 2)) >> 3).clamp(0, 4)
    } else {
        ((qp_y - 20 + (1 << 2)) >> 3).clamp(0, 4)
    } as usize;

    let b_lut = SET_OF_LUT[qp_idx];
    // eq. 1109: tblShift = tblThrLog2[qpIdx] − 4.
    let tbl_shift = TBL_THR_LOG2[qp_idx] - 4;
    // eq. 1111: aTHR = (1 << tblThrLog2[qpIdx]) − (1 << tblShift).
    let a_thr = (1 << TBL_THR_LOG2[qp_idx]) - (1 << tbl_shift);
    HtdfLut {
        b_lut,
        a_thr,
        tbl_shift,
    }
}

/// §8.7.6.1 applicability gate. Returns `false` (filter not applied) for
/// any of the four disqualifying conditions.
///
/// * `nCbW * nCbH < 64`
/// * `Max(nCbW, nCbH) >= 128`
/// * `Min(nCbW, nCbH) >= 32 && !is_intra`
/// * `QpY <= 17`
pub fn htdf_applies(n_cb_w: usize, n_cb_h: usize, qp_y: i32, is_intra: bool) -> bool {
    if n_cb_w * n_cb_h < 64 {
        return false;
    }
    if n_cb_w.max(n_cb_h) >= 128 {
        return false;
    }
    if n_cb_w.min(n_cb_h) >= 32 && !is_intra {
        return false;
    }
    if qp_y <= 17 {
        return false;
    }
    true
}

/// §8.7.6.2 border-availability for one padded-edge sample. Returns the
/// §6.4.1 `availableN` folded with the `constrained_intra_pred_flag`
/// intra predicate: `true` means the bordering neighbour at
/// `(xCb + x, yCb + y)` may be sampled directly; `false` triggers the
/// `dx`/`dy` replicate clamp toward the block interior.
pub trait BorderAvailability {
    /// `availableN && !(constrained_intra_pred_flag && neighbour
    /// not intra)` at component-domain offset `(x, y)` from the block's
    /// top-left.
    fn border_available(&self, x: i32, y: i32) -> bool;
}

/// A border rule that treats every position inside the picture extent as
/// available (single-slice, single-tile, all-intra fixtures). `x`/`y` are
/// component-domain offsets from the block top-left.
pub struct InPictureBorder {
    /// Block top-left x (component domain).
    pub x_cb: i32,
    /// Block top-left y (component domain).
    pub y_cb: i32,
    /// Picture width (component domain).
    pub width: i32,
    /// Picture height (component domain).
    pub height: i32,
}

impl BorderAvailability for InPictureBorder {
    fn border_available(&self, x: i32, y: i32) -> bool {
        let ax = self.x_cb + x;
        let ay = self.y_cb + y;
        ax >= 0 && ay >= 0 && ax < self.width && ay < self.height
    }
}

/// §8.7.6.2 — build the `(nCbW+2)×(nCbH+2)` replicate-padded sample
/// array. Indexing convention: `recSamplesPad[x][y]` for
/// `x ∈ −1..nCbW`, `y ∈ −1..nCbH`, flattened row-major with `x` the
/// fast axis as `pad[(y+1)*(nCbW+2) + (x+1)]`.
///
/// * `sample(ax, ay)` returns `recSamples[ax][ay]` at a picture-domain
///   luma location (already validated in-bounds by the caller's clamp).
/// * `border` resolves §6.4.1 availability + the intra predicate.
pub fn pad_rec_samples<S, B>(
    x_cb: i32,
    y_cb: i32,
    n_cb_w: usize,
    n_cb_h: usize,
    sample: S,
    border: &B,
) -> Vec<i32>
where
    S: Fn(i32, i32) -> i32,
    B: BorderAvailability,
{
    let w = n_cb_w as i32;
    let h = n_cb_h as i32;
    let stride = n_cb_w + 2;
    let mut pad = vec![0i32; stride * (n_cb_h + 2)];

    for y in -1..=h {
        for x in -1..=w {
            let v = if (0..w).contains(&x) && (0..h).contains(&y) {
                // Interior: direct copy.
                sample(x_cb + x, y_cb + y)
            } else {
                // Border: §8.7.6.2 dx/dy replicate clamp.
                let avail = border.border_available(x, y);
                let mut dx = 0;
                let mut dy = 0;
                if !avail {
                    if x == -1 {
                        dx = 1;
                    }
                    if x == w {
                        dx = -1;
                    }
                    if y == -1 {
                        dy = 1;
                    }
                    if y == h {
                        dy = -1;
                    }
                }
                sample(x_cb + x + dx, y_cb + y + dy)
            };
            pad[((y + 1) as usize) * stride + (x + 1) as usize] = v;
        }
    }
    pad
}

/// §8.7.6.1 soft-threshold of one AC Hadamard coefficient (eqs.
/// 1098/1099). `f_had` is `fHad[i]` for `i ∈ {1,2,3}`.
#[inline]
fn filt_coeff(f_had: i32, lut: &HtdfLut, bit_depth: u32) -> i32 {
    if bit_depth < 10 {
        // eq. 1098.
        let bd_shift = 10 - bit_depth as i32;
        if (f_had.abs() << bd_shift) >= lut.a_thr {
            f_had
        } else if f_had > 0 {
            let idx =
                (((f_had << bd_shift) + (1 << (lut.tbl_shift - 1))) >> lut.tbl_shift) as usize;
            lut.b_lut[idx] >> bd_shift
        } else {
            let idx =
                ((((-f_had) << bd_shift) + (1 << (lut.tbl_shift - 1))) >> lut.tbl_shift) as usize;
            -(lut.b_lut[idx] >> bd_shift)
        }
    } else {
        // eq. 1099.
        let bd_shift = bit_depth as i32 - 10;
        if (f_had.abs() >> bd_shift) >= lut.a_thr {
            f_had
        } else if f_had > 0 {
            let idx =
                (((f_had >> bd_shift) + (1 << (lut.tbl_shift - 1))) >> lut.tbl_shift) as usize;
            lut.b_lut[idx] << bd_shift
        } else {
            let idx =
                ((((-f_had) >> bd_shift) + (1 << (lut.tbl_shift - 1))) >> lut.tbl_shift) as usize;
            -(lut.b_lut[idx] << bd_shift)
        }
    }
}

/// §8.7.6.1 — filter the block. Consumes the §8.7.6.2 padded array and
/// returns the `nCbW * nCbH` modified luma samples (row-major), each
/// `Clip1Y((accFlt + 2) >> 2)` (eq. 1105). The caller writes them back
/// into the plane.
///
/// Pre-condition: [`htdf_applies`] returned `true`. `pad` must be a
/// `(nCbW+2)×(nCbH+2)` array from [`pad_rec_samples`].
pub fn filter_block(
    pad: &[i32],
    n_cb_w: usize,
    n_cb_h: usize,
    lut: &HtdfLut,
    bit_depth: u32,
) -> Vec<i32> {
    let stride = n_cb_w + 2;
    let w = n_cb_w as i32;
    let h = n_cb_h as i32;
    let max_val = (1i32 << bit_depth) - 1;

    // accFlt indexed [x][y] for x = −1..nCbW−1, y = −1..nCbH−1, i.e. an
    // nCbW × nCbH window offset by +1. Each sliding 2×2 contributes to a
    // 2×2 footprint, so the accumulation footprint is (nCbW)×(nCbH) at
    // logical x,y in 0..nCbW-1 (the eq.-1104 writes can reach x+1/y+1 =
    // nCbW-1+1, but those land outside the final 0..nCbW-1 output window
    // and are discarded by eq. 1105). We size the buffer to nCbW × nCbH
    // and ignore writes outside it.
    let mut acc = vec![0i32; n_cb_w * n_cb_h];

    // pad accessor: padded[x][y] with x,y in −1..n*, +1 origin.
    let p = |x: i32, y: i32| -> i32 { pad[((y + 1) as usize) * stride + (x + 1) as usize] };

    for y in -1..h {
        for x in -1..w {
            // eq. 1093: inFilt[i] = recSamplesPad[x+scanTmpl[i][1]][y+scanTmpl[i][0]].
            let mut in_filt = [0i32; 4];
            for (i, &(sdy, sdx)) in SCAN_TMPL.iter().enumerate() {
                in_filt[i] = p(x + sdx, y + sdy);
            }

            // eqs. 1094-1097: forward Hadamard.
            let f_had = [
                in_filt[0] + in_filt[2] + in_filt[1] + in_filt[3],
                in_filt[0] + in_filt[2] - in_filt[1] - in_filt[3],
                in_filt[0] - in_filt[2] + in_filt[1] - in_filt[3],
                in_filt[0] - in_filt[2] - in_filt[1] + in_filt[3],
            ];

            // eqs. 1098/1099 soft-threshold the AC coefficients; the DC
            // term fHadFilt[0] = fHad[0] is kept unchanged (§8.7.6.1 text
            // following eq. 1099).
            let mut f_filt = [f_had[0], 0, 0, 0];
            for i in 1..4 {
                f_filt[i] = filt_coeff(f_had[i], lut, bit_depth);
            }

            // eqs. 1100-1103: inverse Hadamard.
            let inv = [
                f_filt[0] + f_filt[2] + f_filt[1] + f_filt[3],
                f_filt[0] + f_filt[2] - f_filt[1] - f_filt[3],
                f_filt[0] - f_filt[2] + f_filt[1] - f_filt[3],
                f_filt[0] - f_filt[2] - f_filt[1] + f_filt[3],
            ];

            // eq. 1104: accumulate the 2×2 footprint.
            for (i, &(sdy, sdx)) in SCAN_TMPL.iter().enumerate() {
                let ax = x + sdx;
                let ay = y + sdy;
                if (0..w).contains(&ax) && (0..h).contains(&ay) {
                    acc[(ay as usize) * n_cb_w + ax as usize] += inv[i] >> 2;
                }
            }
        }
    }

    // eq. 1105: SL = Clip1Y((accFlt + 2) >> 2).
    let mut out = vec![0i32; n_cb_w * n_cb_h];
    for (o, a) in out.iter_mut().zip(acc.iter()) {
        *o = ((a + 2) >> 2).clamp(0, max_val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applicability_gates() {
        // < 64 samples (4×4 = 16).
        assert!(!htdf_applies(4, 4, 30, true));
        // 8×8 = 64, qp > 17, intra → applies.
        assert!(htdf_applies(8, 8, 30, true));
        // Max dim >= 128.
        assert!(!htdf_applies(128, 8, 30, true));
        // Min dim >= 32 and inter → not applied.
        assert!(!htdf_applies(32, 32, 30, false));
        // Min dim >= 32 but intra → applies.
        assert!(htdf_applies(32, 32, 30, true));
        // qp <= 17.
        assert!(!htdf_applies(8, 8, 17, true));
    }

    #[test]
    fn lut_selection_qp_idx() {
        // Non-inter branch: qpIdx = Clip3(0,4,(QpY−20+4)>>3).
        // QpY = 20 → (4)>>3 = 0.
        let l = derive_htdf_lut(20, false);
        assert_eq!(l.b_lut, SET_OF_LUT[0]);
        assert_eq!(l.tbl_shift, TBL_THR_LOG2[0] - 4);
        assert_eq!(
            l.a_thr,
            (1 << TBL_THR_LOG2[0]) - (1 << (TBL_THR_LOG2[0] - 4))
        );
        // QpY = 51 → (51−20+4)>>3 = 35>>3 = 4.
        let l = derive_htdf_lut(51, false);
        assert_eq!(l.b_lut, SET_OF_LUT[4]);
        // Inter-square-≥32 branch: qpIdx = Clip3(0,4,(QpY−28+4)>>3).
        // QpY = 28 → (4)>>3 = 0.
        let l = derive_htdf_lut(28, true);
        assert_eq!(l.b_lut, SET_OF_LUT[0]);
    }

    /// A perfectly flat block has all-zero AC coefficients (`fHad[1..3]`),
    /// so the filter is the identity on a constant field.
    #[test]
    fn flat_field_is_identity() {
        let n = 8usize;
        let val = 100i32;
        let border = InPictureBorder {
            x_cb: 0,
            y_cb: 0,
            width: 64,
            height: 64,
        };
        let pad = pad_rec_samples(0, 0, n, n, |_ax, _ay| val, &border);
        let lut = derive_htdf_lut(30, false);
        let out = filter_block(&pad, n, n, &lut, 8);
        assert!(out.iter().all(|&v| v == val), "got {out:?}");
    }

    /// The padding replicates the block-interior border into the
    /// unavailable ring: a constant block pads to a constant array.
    #[test]
    fn padding_replicates_border() {
        let n = 8usize;
        // Border declares everything unavailable so the dx/dy clamp fires.
        struct NoBorder;
        impl BorderAvailability for NoBorder {
            fn border_available(&self, _x: i32, _y: i32) -> bool {
                false
            }
        }
        // sample returns the x-coordinate so we can see the clamp.
        let pad = pad_rec_samples(0, 0, n, n, |ax, _ay| ax, &NoBorder);
        let stride = n + 2;
        // Left border column x = −1 (pad index 0) clamps dx=+1 → reads x=0.
        for y in 0..n {
            let left = pad[(y + 1) * stride]; // x = −1
            assert_eq!(left, 0, "left border at y={y}");
        }
        // Right border column x = nCbW (pad index n+1) clamps dx=−1 →
        // reads x = nCbW−1.
        for y in 0..n {
            let right = pad[(y + 1) * stride + (n + 1)];
            assert_eq!(right, (n as i32) - 1, "right border at y={y}");
        }
    }

    /// A single bright impulse on a flat field is smoothed (the AC energy
    /// is thresholded), so the output peak is no larger than the input
    /// peak and the surrounding samples move toward it.
    #[test]
    fn impulse_is_smoothed() {
        let n = 8usize;
        let base = 100i32;
        let mut field = vec![base; (n + 2) * (n + 2)]; // oversized scratch
        let stride_pic = n + 2;
        // Put an impulse at interior (4,4).
        field[4 * stride_pic + 4] = 200;
        let border = InPictureBorder {
            x_cb: 0,
            y_cb: 0,
            width: n as i32,
            height: n as i32,
        };
        let pad = pad_rec_samples(
            0,
            0,
            n,
            n,
            |ax, ay| field[(ay as usize) * stride_pic + ax as usize],
            &border,
        );
        let lut = derive_htdf_lut(40, false);
        let out = filter_block(&pad, n, n, &lut, 8);
        let peak = out[4 * n + 4];
        // The filter never amplifies above the input impulse.
        assert!(peak <= 200, "peak={peak}");
        // Energy is preserved in the neighbourhood (output stays within
        // the input value range).
        assert!(out.iter().all(|&v| (90..=200).contains(&v)), "{out:?}");
    }
}
