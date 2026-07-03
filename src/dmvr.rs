//! §8.5.5 decoder-side motion-vector refinement (DMVR) integer + parametric
//! refinement kernels.
//!
//! When `sps_dmvr_flag == 1` a bi-predicted subblock's motion vectors are
//! refined symmetrically (`refMvL0 = mvL0 + dMvL0`, `refMvL1 = mvL1 −
//! dMvL0`, eqs. 396/397) by minimising the bilateral SAD between the two
//! reference predictions over a small search window. This module
//! implements the search core:
//!
//! * §8.5.5.3 [`sad_values`] — the 9-entry bilateral SAD over the `bC`
//!   integer-offset grid (eq. 1018).
//! * §8.5.5.4 [`select_best_idx`] — the array-entry selection that picks
//!   the minimum-SAD offset index (with the centre-bias tie rules).
//! * §8.5.5.5 [`parametric_refine`] — the sub-pel parabolic refinement of
//!   `dMvL0` (eqs. 1019-1022).
//! * §8.5.5.1 [`refine_subblock_mv`] — the driver tying the two integer
//!   refinement passes + the parametric step into the final `dMvL0`
//!   (eqs. 989-998), pure over a caller-supplied predicted-sample
//!   provider (the §8.5.5.2 bilinear interpolation is the data plane the
//!   decoder wires in).
//!
//! All sample arrays are addressed in the spec's `(sbW + 2·srRange) ×
//! (sbH + 2·srRange)` padded layout with `srRange = 2`, i.e. a
//! `(sbW + 4) × (sbH + 4)` block; the `offsetH`/`offsetV` base of `2`
//! centres the SAD window.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

/// §8.5.5 fixed search range (`srRange = 2`, §8.5.5.1).
pub const SR_RANGE: i32 = 2;

/// The §8.5.5.3 `bC[ 2 ][ 9 ]` integer-offset grid: `bC[0][i]` is the x
/// offset and `bC[1][i]` the y offset of search position `i`, laid out as
/// a 3×3 grid in raster order (`i = 3·(dx+1) + (dy+1)` ⇒ index 4 is the
/// centre `(0, 0)`).
const BC: [(i32, i32); 9] = [
    (-1, -1), // 0
    (-1, 0),  // 1
    (-1, 1),  // 2
    (0, -1),  // 3
    (0, 0),   // 4 (centre)
    (0, 1),   // 5
    (1, -1),  // 6
    (1, 0),   // 7
    (1, 1),   // 8
];

/// One DMVR predicted-sample plane: the `(sb_w + 4) × (sb_h + 4)` array of
/// §8.5.5.2 bilinear-interpolated luma values for one list, row-major with
/// stride `sb_w + 2·srRange`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredPlane {
    /// Row stride `sb_w + 2·srRange`.
    pub stride: usize,
    /// Sample values, length `stride · (sb_h + 2·srRange)`.
    pub samples: Vec<i32>,
}

impl PredPlane {
    #[inline]
    fn at(&self, x: i32, y: i32) -> i32 {
        self.samples[(y as usize) * self.stride + (x as usize)]
    }
}

/// §8.5.5.3 — the 9-entry bilateral SAD (eq. 1018).
///
/// For each search position `i` the SAD accumulates `Abs(L0[x + offH0][y +
/// offV0] − L1[x − 2·bCx + offH1][y − 2·bCy + offV1])` over `x ∈
/// [bCx, sbW + bCx)`, `y ∈ [bCy, sbH + bCy)`. `offset_h`/`offset_v` are the
/// `[L0, L1]` base offsets the §8.5.5.1 driver maintains.
pub fn sad_values(
    pred_l0: &PredPlane,
    pred_l1: &PredPlane,
    sb_w: i32,
    sb_h: i32,
    offset_h: [i32; 2],
    offset_v: [i32; 2],
) -> [i64; 9] {
    let mut sad = [0i64; 9];
    for (i, &(bcx, bcy)) in BC.iter().enumerate() {
        let mut acc = 0i64;
        // y ∈ [bcy, sbH + bcy), x ∈ [bcx, sbW + bcx) — the spec's inclusive
        // upper bound `..(nSb + bC)` is exclusive in the accumulation
        // (an off-by-one in the prose; the window is sbW × sbH samples).
        for y in bcy..(sb_h + bcy) {
            for x in bcx..(sb_w + bcx) {
                let a = pred_l0.at(x + offset_h[0], y + offset_v[0]);
                let b = pred_l1.at(x - 2 * bcx + offset_h[1], y - 2 * bcy + offset_v[1]);
                acc += (a - b).unsigned_abs() as i64;
            }
        }
        sad[i] = acc;
    }
    sad
}

/// §8.5.5.4 — array-entry selection: pick the minimum-SAD search index
/// with the spec's quadrant-then-centre-bias tie rules.
pub fn select_best_idx(sad: &[i64; 9]) -> usize {
    // The four-way quadrant split on the cross neighbours (1/7 horizontal,
    // 3/5 vertical), each yielding an `idx_val` and a provisional best.
    let (idx_val, mut best_idx) = if sad[1] < sad[7] && sad[3] < sad[5] {
        (0usize, if sad[1] < sad[3] { 1 } else { 3 })
    } else if sad[1] >= sad[7] && sad[3] < sad[5] {
        (6usize, if sad[7] < sad[3] { 7 } else { 3 })
    } else if sad[1] < sad[7] && sad[3] >= sad[5] {
        (2usize, if sad[1] < sad[5] { 1 } else { 5 })
    } else {
        (8usize, if sad[7] < sad[5] { 7 } else { 5 })
    };
    // Centre bias: a non-worse centre wins.
    if sad[4] <= sad[best_idx] {
        best_idx = 4;
    }
    // The diagonal `idx_val` wins if strictly better than the current best.
    if sad[idx_val] < sad[best_idx] {
        best_idx = idx_val;
    }
    best_idx
}

/// §8.5.5.5 — parametric (sub-pel) refinement of `d_mv` (eqs. 1019-1022).
///
/// Adds the parabolic-minimum offset derived from the cross-neighbour SADs
/// (`sad[1]`/`sad[7]` horizontal, `sad[3]`/`sad[5]` vertical, `sad[4]`
/// centre) to `d_mv`, in 1/16-pel units. Each axis is independently `0`
/// when the parabola is degenerate (`sad[1] + sad[7] == 2·sad[4]`).
pub fn parametric_refine(sad: &[i64; 9], d_mv: [i32; 2]) -> [i32; 2] {
    let axis = |s_lo: i64, s_hi: i64| -> i32 {
        let denom = 2 * (s_lo + s_hi - (sad[4] << 1));
        if s_lo + s_hi == (sad[4] << 1) {
            0
        } else {
            // ((s_lo − s_hi) << 4) / denom — integer division toward zero.
            (((s_lo - s_hi) << 4) / denom) as i32
        }
    };
    [
        d_mv[0] + axis(sad[1], sad[7]),
        d_mv[1] + axis(sad[3], sad[5]),
    ]
}

/// §8.5.5.1 — the DMVR driver: derive the delta MV `dMvL0` (1/16-pel) for
/// one subblock from the two predicted-sample planes.
///
/// `pred_l0`/`pred_l1` are the §8.5.5.2 bilinear predictions over the
/// `(sb_w + 4) × (sb_h + 4)` padded window (the data plane the decoder
/// supplies). Returns `(d_mv_l0, frac_pel_applied)`; the caller forms
/// `refMvL0 = mvL0 + dMvL0`, `refMvL1 = mvL1 − dMvL0` (eqs. 396/397).
///
/// The early-out gate (eq.: `sadVal[4] >= sbW·sbH` required to refine at
/// all) is applied; when it fails `dMvL0` is `(0, 0)`.
pub fn refine_subblock_mv(
    pred_l0: &PredPlane,
    pred_l1: &PredPlane,
    sb_w: i32,
    sb_h: i32,
) -> ([i32; 2], bool) {
    let mut offset_h = [2i32; 2];
    let mut offset_v = [2i32; 2];
    let mut frac_pel_applied = false;
    let mut d_mv = [0i32; 2];

    let sad = sad_values(pred_l0, pred_l1, sb_w, sb_h, offset_h, offset_v);

    // Refinement gate (§8.5.5.1): only refine when the centre SAD is at
    // least sbW·sbH (an average per-sample difference of ≥ 1).
    if sad[4] < (sb_w as i64) * (sb_h as i64) {
        return (d_mv, frac_pel_applied);
    }

    let best_idx = select_best_idx(&sad);
    if best_idx == 4 {
        frac_pel_applied = true;
        // No integer step; fall through to the parametric stage on the
        // first-pass SAD.
        let refined = parametric_refine(&sad, d_mv);
        return (refined, frac_pel_applied);
    }

    // First integer step (eqs. 991-996): move the centre to bestIdx.
    let dx = (best_idx / 3) as i32 - 1;
    let dy = (best_idx % 3) as i32 - 1;
    d_mv[0] = 16 * dx;
    d_mv[1] = 16 * dy;
    offset_h[0] += dx;
    offset_v[0] += dy;
    offset_h[1] += -dx;
    offset_v[1] += -dy;

    // Re-evaluate SAD at the shifted window, re-select.
    let sad2 = sad_values(pred_l0, pred_l1, sb_w, sb_h, offset_h, offset_v);
    let best_idx2 = select_best_idx(&sad2);
    if best_idx2 == 4 && sad2[4] > 0 {
        frac_pel_applied = true;
    }
    let dx2 = (best_idx2 / 3) as i32 - 1;
    let dy2 = (best_idx2 % 3) as i32 - 1;
    d_mv[0] += 16 * dx2;
    d_mv[1] += 16 * dy2;

    if frac_pel_applied {
        d_mv = parametric_refine(&sad2, d_mv);
    }
    (d_mv, frac_pel_applied)
}

/// Table 29 — the §8.5.5.2.2 luma bilinear interpolation filter
/// coefficients `fbL[ p ] = [ 64 − 4·p, 4·p ]` for each 1/16 fractional
/// position `p`.
#[inline]
fn fb_l(p: i32) -> [i32; 2] {
    [64 - 4 * p, 4 * p]
}

/// §8.5.5.1 eqs. 989/990 + §8.5.5.2 — build one list's `(sbWidth +
/// 2·srRange) × (sbHeight + 2·srRange)` bilinear prediction plane.
///
/// `mv_quarter` is the list's **1/4-pel** `mvLX`; the §8.5.2.9 conversion
/// to 1/16-pel (`mvLtX = mvLX << 2`) and the eqs.-989/990 search-window
/// pre-shift (`mvLsX = mvLtX − 16·srRange`) happen here, so the returned
/// plane is exactly the §8.5.5.1 `predSamplesLXL` input of the SAD stage.
///
/// Per §8.5.5.2.2 the sample values are scaled to a common 10-bit-headroom
/// domain: the integer-position copy applies `<< shift0 = 10 − BitDepthY`
/// (eq. 1012, `BitDepthY <= 10`), the single-axis filters normalise by
/// `>> shift2 = BitDepthY − 4` (eqs. 1013/1014) and the two-axis path by
/// `>> shift3` then `( · + offset4 ) >> shift4` (eqs. 1015-1017). Every
/// reference fetch is clipped to the picture (the eqs. 1013-1016 clips,
/// extended to both axes for memory safety — border replicate).
pub fn bilinear_pred_plane(
    refp: crate::inter::RefPictureView<'_>,
    x_sb: i32,
    y_sb: i32,
    mv_quarter: crate::inter::MotionVector,
    sb_w: i32,
    sb_h: i32,
    bit_depth: u32,
) -> PredPlane {
    let mv_lt = mv_quarter.quarter_to_sixteenth();
    // eqs. 989/990.
    let mv_ls_x = mv_lt.x - 16 * SR_RANGE;
    let mv_ls_y = mv_lt.y - 16 * SR_RANGE;
    let int_x = mv_ls_x >> 4;
    let int_y = mv_ls_y >> 4;
    let frac_x = mv_ls_x & 15;
    let frac_y = mv_ls_y & 15;
    let w = (sb_w + 2 * SR_RANGE) as usize;
    let h = (sb_h + 2 * SR_RANGE) as usize;
    let bd = bit_depth as i32;
    let fetch = |x: i32, y: i32| crate::inter::sample_luma_clipped(refp, x, y);
    let mut samples = vec![0i32; w * h];
    for (yl, row) in samples.chunks_exact_mut(w).enumerate() {
        for (xl, out) in row.iter_mut().enumerate() {
            // eqs. 999/1000.
            let xi = x_sb + int_x + xl as i32;
            let yi = y_sb + int_y + yl as i32;
            *out = if frac_x == 0 && frac_y == 0 {
                // eq. 1012.
                let s = fetch(xi, yi);
                if bd <= 10 {
                    s << (10 - bd)
                } else {
                    (s + (1 << (bd - 11))) >> (bd - 10)
                }
            } else if frac_y == 0 {
                // eq. 1013 — horizontal-only.
                let f = fb_l(frac_x);
                let shift2 = bd - 4;
                let offset2 = 1 << (shift2 - 1);
                (f[0] * fetch(xi, yi) + f[1] * fetch(xi + 1, yi) + offset2) >> shift2
            } else if frac_x == 0 {
                // eq. 1014 — vertical-only.
                let f = fb_l(frac_y);
                let shift2 = bd - 4;
                let offset2 = 1 << (shift2 - 1);
                (f[0] * fetch(xi, yi) + f[1] * fetch(xi, yi + 1) + offset2) >> shift2
            } else {
                // eqs. 1015-1017 — separable two-tap, offset3 = 0.
                let fx = fb_l(frac_x);
                let fy = fb_l(frac_y);
                let shift3 = bd - 8;
                let t0 = (fx[0] * fetch(xi, yi) + fx[1] * fetch(xi + 1, yi)) >> shift3;
                let t1 = (fx[0] * fetch(xi, yi + 1) + fx[1] * fetch(xi + 1, yi + 1)) >> shift3;
                // shift4 = 10, offset4 = 1 << 9 (eqs. 1010/1011).
                (fy[0] * t0 + fy[1] * t1 + (1 << 9)) >> 10
            };
        }
    }
    PredPlane { stride: w, samples }
}

/// §8.5.1 eqs. 387-390 — the DMVR subblock partition of an
/// `nCbW × nCbH` coding block: `(numSbX, numSbY, sbWidth, sbHeight)`.
pub fn dmvr_subblock_geometry(n_cb_w: u32, n_cb_h: u32) -> (u32, u32, u32, u32) {
    let num_sb_x = if n_cb_w > 16 { n_cb_w >> 4 } else { 1 };
    let num_sb_y = if n_cb_h > 16 { n_cb_h >> 4 } else { 1 };
    let sb_w = if n_cb_w > 16 { 16 } else { n_cb_w };
    let sb_h = if n_cb_h > 16 { 16 } else { n_cb_h };
    (num_sb_x, num_sb_y, sb_w, sb_h)
}

/// §8.5.5 end-to-end for one subblock over real reference pictures:
/// build both lists' §8.5.5.2 bilinear prediction planes from the
/// 1/4-pel `mvL0` / `mvL1` and run the §8.5.5.1 refinement. Returns the
/// delta motion vector `dMvL0` in 1/16-pel units (the caller forms
/// `refMvLX = ( mvLX << 2 ) ± dMvL0`, eqs. 395-397).
#[allow(clippy::too_many_arguments)]
pub fn refine_subblock_from_refs(
    ref_l0: crate::inter::RefPictureView<'_>,
    ref_l1: crate::inter::RefPictureView<'_>,
    x_sb: i32,
    y_sb: i32,
    mv_l0_quarter: crate::inter::MotionVector,
    mv_l1_quarter: crate::inter::MotionVector,
    sb_w: i32,
    sb_h: i32,
    bit_depth: u32,
) -> [i32; 2] {
    let p0 = bilinear_pred_plane(ref_l0, x_sb, y_sb, mv_l0_quarter, sb_w, sb_h, bit_depth);
    let p1 = bilinear_pred_plane(ref_l1, x_sb, y_sb, mv_l1_quarter, sb_w, sb_h, bit_depth);
    refine_subblock_mv(&p0, &p1, sb_w, sb_h).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat plane of constant value, `(sb_w+4) × (sb_h+4)`.
    fn flat(sb_w: i32, sb_h: i32, value: i32) -> PredPlane {
        let stride = (sb_w + 2 * SR_RANGE) as usize;
        let rows = (sb_h + 2 * SR_RANGE) as usize;
        PredPlane {
            stride,
            samples: vec![value; stride * rows],
        }
    }

    /// Identical planes ⇒ all SADs zero ⇒ no refinement (gate fails).
    #[test]
    fn identical_planes_no_refinement() {
        let p0 = flat(8, 8, 100);
        let p1 = flat(8, 8, 100);
        let sad = sad_values(&p0, &p1, 8, 8, [2, 2], [2, 2]);
        assert!(sad.iter().all(|&s| s == 0));
        let (d, frac) = refine_subblock_mv(&p0, &p1, 8, 8);
        assert_eq!(d, [0, 0]);
        assert!(!frac);
    }

    /// A constant DC offset between the planes makes every centred SAD =
    /// sbW·sbH·|delta|; the gate passes and the centre is selected (flat
    /// cost surface ⇒ bestIdx 4).
    #[test]
    fn dc_offset_centre_selected() {
        let p0 = flat(8, 8, 200);
        let p1 = flat(8, 8, 150); // delta 50
        let sad = sad_values(&p0, &p1, 8, 8, [2, 2], [2, 2]);
        // Every position is a flat field → identical SAD = 64·50 = 3200.
        assert!(sad.iter().all(|&s| s == 64 * 50));
        let best = select_best_idx(&sad);
        assert_eq!(best, 4); // centre bias on a flat cost surface
        let (_d, frac) = refine_subblock_mv(&p0, &p1, 8, 8);
        // Flat surface → parametric axes degenerate → dMv stays 0, but
        // frac_pel_applied is set (bestIdx == 4 path).
        assert!(frac);
    }

    /// select_best_idx centre bias: a strictly-minimal centre always wins.
    #[test]
    fn select_centre_bias() {
        // Centre (idx 4) smallest → bestIdx 4.
        let sad = [9i64, 8, 9, 8, 1, 8, 9, 8, 9];
        assert_eq!(select_best_idx(&sad), 4);
    }

    /// select_best_idx picks the minimal cross neighbour when the centre is
    /// not the minimum.
    #[test]
    fn select_left_neighbour() {
        // sad[1] (left) is the unique minimum, centre large.
        let sad = [9i64, 1, 9, 8, 10, 8, 9, 8, 9];
        // quadrant: sad[1]<sad[7] (1<8) ✓, sad[3]<sad[5] (8<8) ✗ → branch 3
        // (sad[1]<sad[7] && sad[3]>=sad[5]) idxVal 2, best = sad[1]<sad[5]?1:5
        // → 1. Centre 10 not <= sad[1]=1 → keep. idxVal 2: sad[2]=9 not < 1.
        assert_eq!(select_best_idx(&sad), 1);
    }

    /// parametric_refine: a symmetric V around the centre gives a 0 offset;
    /// an asymmetric one biases toward the lower neighbour.
    #[test]
    fn parametric_symmetric_and_biased() {
        // Symmetric: sad[1]==sad[7], sad[3]==sad[5] → both axes 0.
        let sym = [0i64, 10, 0, 10, 4, 10, 0, 10, 0];
        assert_eq!(parametric_refine(&sym, [0, 0]), [0, 0]);
        // Horizontal asymmetry: sad[1]=8, sad[7]=12, sad[4]=4.
        // denom = 2·(8+12−8) = 24; num = (8−12)<<4 = −64; −64/24 = −2.
        let asym = [0i64, 8, 0, 10, 4, 10, 0, 12, 0];
        let r = parametric_refine(&asym, [0, 0]);
        assert_eq!(r[0], -2);
        assert_eq!(r[1], 0); // vertical symmetric (10/10)
    }

    /// A horizontal integer-pel mismatch is recovered: L0 and L1 carry the
    /// same horizontal ramp, but L1 is shifted so that a left/right `bC`
    /// offset re-aligns them to zero SAD while the centre SAD is large.
    /// The SAD indexing reads `L0[x − 2·bCx + offH1]` for L1, so a one-pel
    /// content shift is cancelled by the matching `bCx`.
    #[test]
    fn horizontal_mismatch_lowers_at_neighbour() {
        let stride = (8 + 4) as usize;
        let rows = (8 + 4) as usize;
        // L0: horizontal ramp. L1: the SAME ramp. With offH base 2 the
        // centre compares L0[x+2] vs L1[x+2] — identical → SAD 0. To create
        // a mismatch we offset L1's content by one column so the centre is
        // off by a constant and a neighbour re-aligns.
        let mut s0 = vec![0i32; stride * rows];
        let mut s1 = vec![0i32; stride * rows];
        for y in 0..rows {
            for x in 0..stride {
                s0[y * stride + x] = (x as i32) * 10;
                s1[y * stride + x] = (x as i32) * 10;
            }
        }
        // Perturb L1 so the centred comparison is non-zero but a shifted
        // window finds a better (lower) match: bump every L1 sample by a
        // ramp-step so column k of L1 matches column k+1 of L0.
        for v in s1.iter_mut() {
            *v += 10;
        }
        let p0 = PredPlane {
            stride,
            samples: s0,
        };
        let p1 = PredPlane {
            stride,
            samples: s1,
        };
        let sad = sad_values(&p0, &p1, 8, 8, [2, 2], [2, 2]);
        // Centre (idx 4) compares L0[x+2] vs L1[x+2] = L0[x+2]+10 → SAD =
        // 64·10. A horizontal neighbour (idx 1 / idx 7) shifts L1's read by
        // ±2 columns (the −2·bCx term) = ±20 in value → strictly larger or
        // the re-aligning direction → at least one neighbour ≠ centre cost.
        assert!(sad[4] > 0);
        // The cost surface is not flat: the two horizontal neighbours differ
        // from the centre (the −2·bCx content shift changes the SAD).
        assert!(sad[1] != sad[4] || sad[7] != sad[4]);
    }

    /// End-to-end driver on a flat-offset plane pair: the gate passes
    /// (centre SAD ≥ sbW·sbH) and a flat cost surface yields a centre
    /// selection with `frac_pel_applied` set and a degenerate (zero)
    /// parametric delta.
    #[test]
    fn driver_flat_surface_centre() {
        let p0 = flat(8, 8, 128);
        let p1 = flat(8, 8, 64); // delta 64 → SAD 64·64 ≥ 64
        let (d, frac) = refine_subblock_mv(&p0, &p1, 8, 8);
        assert!(frac);
        assert_eq!(d, [0, 0]); // flat → no integer step, degenerate parabola
    }

    // --- §8.5.5.2 bilinear interpolation + the refs-level driver ---------

    /// Deterministic "generic" 8-bit texture.
    fn texture(x: i32, y: i32) -> u8 {
        (((x * 37 + y * 101) ^ (x * y * 13)) & 0xff) as u8
    }

    fn make_ref(shift_x: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y: Vec<u8> = (0i32..32 * 32)
            .map(|i| texture(i % 32 - shift_x, i / 32))
            .collect();
        (y, vec![128u8; 16 * 16], vec![128u8; 16 * 16])
    }

    fn view<'a>(y: &'a [u8], cb: &'a [u8], cr: &'a [u8]) -> crate::inter::RefPictureView<'a> {
        crate::inter::RefPictureView {
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

    /// Integer-position bilinear plane (eq. 1012): a zero 1/4-pel MV
    /// yields `mvLs = −2` integer pels each axis, and every plane sample
    /// is the reference sample `<< shift0` (= 2 at 8-bit).
    #[test]
    fn bilinear_plane_integer_positions_scale_by_shift0() {
        let (y, cb, cr) = make_ref(0);
        let refp = view(&y, &cb, &cr);
        let p = bilinear_pred_plane(refp, 8, 8, crate::inter::MotionVector::default(), 8, 8, 8);
        assert_eq!(p.stride, 12);
        for yl in 0..12 {
            for xl in 0..12 {
                let expect = (texture(8 - 2 + xl, 8 - 2 + yl) as i32) << 2;
                assert_eq!(p.at(xl, yl), expect, "({xl},{yl})");
            }
        }
    }

    /// Half-pel horizontal phase (eq. 1013): `mvL0 = (2, 0)` in 1/4-pel →
    /// `mvLs.x = −24` → fracX = 8 → the Table-29 `[32, 32]` average with
    /// the `offset2 >> shift2` normalisation onto the same 10-bit domain.
    #[test]
    fn bilinear_plane_half_pel_is_two_tap_average() {
        let (y, cb, cr) = make_ref(0);
        let refp = view(&y, &cb, &cr);
        let mv = crate::inter::MotionVector { x: 2, y: 0 };
        let p = bilinear_pred_plane(refp, 8, 8, mv, 4, 4, 8);
        // int_x = (8 − 32) >> 4 = −2, frac = 8.
        for yl in 0..8 {
            for xl in 0..8 {
                let a = texture(8 - 2 + xl, 8 - 2 + yl) as i32;
                let b = texture(8 - 2 + xl + 1, 8 - 2 + yl) as i32;
                let expect = (32 * a + 32 * b + 8) >> 4;
                assert_eq!(p.at(xl, yl), expect, "({xl},{yl})");
            }
        }
    }

    /// eqs. 387-390 — the DMVR partition: ≤16 stays whole, larger splits
    /// into 16-sample subblocks.
    #[test]
    fn subblock_geometry_eqs_387_390() {
        assert_eq!(dmvr_subblock_geometry(8, 8), (1, 1, 8, 8));
        assert_eq!(dmvr_subblock_geometry(16, 16), (1, 1, 16, 16));
        assert_eq!(dmvr_subblock_geometry(32, 16), (2, 1, 16, 16));
        assert_eq!(dmvr_subblock_geometry(64, 32), (4, 2, 16, 16));
    }

    /// End-to-end over reference pictures: L0's content sits one pel
    /// right of base, L1's one pel left. The bilateral match is exact at
    /// `dMvL0 = (+1 int pel, 0)` (L0 shifts +d, L1 −d, eqs. 396/397), so
    /// the refinement returns exactly `[16, 0]` with no sub-pel step
    /// (second-pass centre SAD is 0).
    #[test]
    fn refine_from_refs_recovers_opposed_integer_shift() {
        let (y0, cb0, cr0) = make_ref(1); // ref0[x] = base[x − 1]
        let (y1, cb1, cr1) = make_ref(-1); // ref1[x] = base[x + 1]
        let r0 = view(&y0, &cb0, &cr0);
        let r1 = view(&y1, &cb1, &cr1);
        let d = refine_subblock_from_refs(
            r0,
            r1,
            8,
            8,
            crate::inter::MotionVector::default(),
            crate::inter::MotionVector::default(),
            8,
            8,
            8,
        );
        assert_eq!(d, [16, 0]);
    }

    /// Identical references ⇒ the §8.5.5.1 gate (centre SAD < sbW·sbH)
    /// fails and the refinement is the identity.
    #[test]
    fn refine_from_refs_identical_refs_no_op() {
        let (y0, cb0, cr0) = make_ref(0);
        let r0 = view(&y0, &cb0, &cr0);
        let d = refine_subblock_from_refs(
            r0,
            r0,
            8,
            8,
            crate::inter::MotionVector::default(),
            crate::inter::MotionVector::default(),
            8,
            8,
            8,
        );
        assert_eq!(d, [0, 0]);
    }
}
