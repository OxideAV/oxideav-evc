//! **ALF encoder** (round 458): adaptive-loop-filter design for the
//! §8.8.4 filters the decoder applies — Wiener estimation per
//! §8.8.4.3 class, class merging, coefficient quantisation, the
//! per-CTB `alf_ctb_flag` election, the §7.3.5 `alf_data()` writer
//! (dual of [`crate::alf::parse_alf_data`]) and the APS NAL, with the
//! decoder's own apply as the reconstruction reference.
//!
//! ## Estimation
//!
//! The decoder's luma filter (§8.8.4.2 eqs. 1281-1288) is
//! `out = ( Σ_k c_k · ( p_k + p'_k ) + c_12 · p_0 + 256 ) >> 9` with
//! `c_12 = 512 − 2 Σ_k c_k` (eq. 104 — unity DC gain), `k` over the 12
//! symmetric 7×7-diamond taps and the tap order permuted per sample by
//! its `transposeIdx` (eqs. 1282-1285). Substituting the constraint,
//! `out − p_0 ≈ Σ_k c_k · d_k / 512` with `d_k = p_k + p'_k − 2 p_0`,
//! so per class the least-squares filter solves the 12×12 normal
//! equations `A c = 512 b` accumulated over the class's samples with
//! the regressors indexed in the **canonical** tap order (the inverse
//! of the sample's transpose). The inputs are exactly the decoder's:
//! the §8.8.4.5 padded per-CTB arrays under the slice's edge
//! availability and the §8.8.4.3 classification over them.
//!
//! Classes are merged greedily (the pair whose joint filter loses the
//! least energy first), every merge level's filter set is quantised
//! (rounded, then refined coordinate-wise on the quadratic) and written,
//! the CTBs are elected on / off against the filtered picture, and the
//! level with the least `SSE + λ · ( APS bits + flag bits )` wins. Chroma
//! designs one 5×5-diamond filter jointly over Cb and Cr (the APS
//! carries a single chroma filter) and elects each plane.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

// Dense small-matrix arithmetic reads clearest with explicit indices.
#![allow(clippy::needless_range_loop)]

use oxideav_core::Result;

use crate::alf::{
    self, AlfCtbMap, AlfData, AlfInputAvailability, CHROMA_MAX_GOLOMB_IDX, CHROMA_TAPS,
    CHROMA_TAPS_SYM, GOLOMB_ORDER_IDX_C, GOLOMB_ORDER_IDX_Y, LUMA_TAPS, LUMA_TAPS_SYM,
    NUM_ALF_FILTERS,
};
use crate::bitwriter::BitWriter;
use crate::picture::YuvPicture;

/// Luma taps (free coefficients).
const NL: usize = 12;
/// Chroma taps (free coefficients).
const NC: usize = 6;

/// The slice-level outcome of an ALF design: what the slice header and
/// the access unit carry, and the CTB map the `coding_tree_unit()`
/// prefix writes.
#[derive(Clone, Debug)]
pub struct AlfSliceParams {
    /// The complete APS RBSP (`adaptation_parameter_set_id = 0`,
    /// `aps_params_type = 0`, `alf_data()`, extension flag, trailing
    /// bits).
    pub aps_rbsp: Vec<u8>,
    /// `slice_alf_enabled_flag`.
    pub luma_enabled: bool,
    /// `slice_alf_map_flag` — the per-CTB `alf_ctb_flag` is coded.
    pub map_flag: bool,
    /// `slice_alf_chroma_idc` (0 none, 1 Cb, 2 Cr, 3 both).
    pub chroma_idc: u32,
    /// Per-CTB luma flags (raster order) — all `true` when `map_flag`
    /// is 0.
    pub ctb_luma: Vec<bool>,
    /// The parsed-back filter data the decoder will hold.
    pub data: AlfData,
}

impl AlfSliceParams {
    /// The resolved `AlfCtbMap` the decoder builds from this slice.
    pub fn ctb_map(&self, pic_w: u32, pic_h: u32, ctb_log2: u32) -> AlfCtbMap {
        let mut map = AlfCtbMap::new(pic_w, pic_h, ctb_log2);
        let cb = self.chroma_idc == 1 || self.chroma_idc == 3;
        let cr = self.chroma_idc == 2 || self.chroma_idc == 3;
        for i in 0..map.luma.len() {
            let luma = self.luma_enabled && self.ctb_luma.get(i).copied().unwrap_or(false);
            map.set(i, luma, cb, cr);
        }
        map
    }
}

/// Least-squares accumulator of one filter class: `A` (N×N), `b` (N),
/// `Σ y²` and the sample count.
#[derive(Clone)]
struct Stats<const N: usize> {
    a: [[f64; N]; N],
    b: [f64; N],
    sy2: f64,
    n: f64,
}

impl<const N: usize> Stats<N> {
    fn zero() -> Self {
        Self {
            a: [[0.0; N]; N],
            b: [0.0; N],
            sy2: 0.0,
            n: 0.0,
        }
    }
    fn add(&mut self, d: &[f64; N], y: f64) {
        for i in 0..N {
            self.b[i] += d[i] * y;
            for j in i..N {
                self.a[i][j] += d[i] * d[j];
            }
        }
        self.sy2 += y * y;
        self.n += 1.0;
    }
    fn merge(&self, o: &Self) -> Self {
        let mut out = self.clone();
        for i in 0..N {
            out.b[i] += o.b[i];
            for j in i..N {
                out.a[i][j] += o.a[i][j];
            }
        }
        out.sy2 += o.sy2;
        out.n += o.n;
        out
    }
    /// The real-valued least-squares filter (units of 1/512) — `None`
    /// when the class carries no usable energy.
    fn solve(&self) -> Option<[f64; N]> {
        if self.n < 1.0 {
            return None;
        }
        // Symmetric fill + a mild ridge for near-singular classes.
        let mut m = [[0.0f64; N]; N];
        let mut trace = 0.0;
        for i in 0..N {
            trace += self.a[i][i];
        }
        let ridge = trace / (N as f64) * 1e-6 + 1e-9;
        for i in 0..N {
            for j in 0..N {
                m[i][j] = if j >= i { self.a[i][j] } else { self.a[j][i] };
            }
            m[i][i] += ridge;
        }
        let mut rhs = [0.0f64; N];
        for i in 0..N {
            rhs[i] = 512.0 * self.b[i];
        }
        // Gaussian elimination with partial pivoting.
        for col in 0..N {
            let mut piv = col;
            for r in col + 1..N {
                if m[r][col].abs() > m[piv][col].abs() {
                    piv = r;
                }
            }
            if m[piv][col].abs() < 1e-12 {
                return None;
            }
            m.swap(col, piv);
            rhs.swap(col, piv);
            for r in col + 1..N {
                let f = m[r][col] / m[col][col];
                if f != 0.0 {
                    for c in col..N {
                        m[r][c] -= f * m[col][c];
                    }
                    rhs[r] -= f * rhs[col];
                }
            }
        }
        let mut x = [0.0f64; N];
        for i in (0..N).rev() {
            let mut acc = rhs[i];
            for j in i + 1..N {
                acc -= m[i][j] * x[j];
            }
            x[i] = acc / m[i][i];
        }
        Some(x)
    }
    /// `SSE` of the class under integer filter `c` (units of 1/512):
    /// `Σ y² − 2 bᵀc / 512 + cᵀ A c / 512²`.
    fn sse(&self, c: &[i32; N]) -> f64 {
        let mut bt = 0.0;
        let mut quad = 0.0;
        for i in 0..N {
            bt += self.b[i] * f64::from(c[i]);
            for j in 0..N {
                let a = if j >= i { self.a[i][j] } else { self.a[j][i] };
                quad += a * f64::from(c[i]) * f64::from(c[j]);
            }
        }
        self.sy2 - 2.0 * bt / 512.0 + quad / (512.0 * 512.0)
    }
    /// Round the real filter and refine it coordinate-wise on the
    /// quadratic, within `[lo, hi]`.
    fn quantize(&self, lo: i32, hi: i32) -> [i32; N] {
        let mut c = [0i32; N];
        if let Some(f) = self.solve() {
            for i in 0..N {
                c[i] = f[i].round().clamp(f64::from(lo), f64::from(hi)) as i32;
            }
        }
        let mut best = self.sse(&c);
        for _ in 0..4 {
            let mut improved = false;
            for i in 0..N {
                for delta in [-1i32, 1] {
                    let v = c[i] + delta;
                    if v < lo || v > hi {
                        continue;
                    }
                    let mut t = c;
                    t[i] = v;
                    let s = self.sse(&t);
                    if s < best - 1e-9 {
                        best = s;
                        c = t;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        c
    }
}

/// The canonical tap index each transposed tap position reads
/// (`coef[k] = f[perm[k]]`, eqs. 1282-1285).
fn transpose_perm(t: u8) -> [usize; 13] {
    let id: [i16; 13] = std::array::from_fn(|i| i as i16);
    let p = alf::transpose_luma_coeffs(&id, t);
    std::array::from_fn(|k| p[k] as usize)
}

/// `Ceil( Log2( n ) )`.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Write `alf_data()` (§7.3.5) for the given filter set: `luma`
/// (signalled filters, coefficients `[0..12)`), the class → filter map
/// `delta_idx`, and the optional chroma filter. Fixed-filter usage 0,
/// `alf_luma_type_flag = 1` (13 taps), `alf_luma_coeff_delta_flag = 0`;
/// the Exp-Golomb orders and the delta-prediction flag are chosen for
/// the fewest bits. Returns the bit writer positioned after the
/// structure.
fn write_alf_data(
    w: &mut BitWriter,
    luma: Option<(&[[i32; NL]], &[u8; NUM_ALF_FILTERS])>,
    chroma: Option<&[i32; NC]>,
) {
    w.u1(luma.is_some()); // alf_luma_filter_signal_flag
    w.u1(chroma.is_some()); // alf_chroma_filter_signal_flag
    if let Some((filters, delta_idx)) = luma {
        let n = filters.len();
        w.ue(n as u32 - 1); // alf_luma_num_filters_signalled_minus1
        w.u1(true); // alf_luma_type_flag (13 taps)
        if n > 1 {
            let bits = ceil_log2(n as u32);
            for &d in delta_idx.iter() {
                w.u(bits, u32::from(d)); // alf_luma_coeff_delta_idx[i]
            }
        }
        w.uek(0, 0); // alf_luma_fixed_filter_usage_pattern = 0
        w.u1(false); // alf_luma_coeff_delta_flag = 0
                     // Delta prediction: code either the filters or their successive
                     // differences (eq. 97), whichever is cheaper.
        let diffs: Vec<[i32; NL]> = (0..n)
            .map(|i| {
                if i == 0 {
                    filters[0]
                } else {
                    std::array::from_fn(|j| filters[i][j] - filters[i - 1][j])
                }
            })
            .collect();
        let (pred, orders, min_order) = if n > 1 {
            let (o_a, m_a, bits_a) = best_luma_orders(filters);
            let (o_b, m_b, bits_b) = best_luma_orders(&diffs);
            if bits_b < bits_a {
                (true, o_b, m_b)
            } else {
                (false, o_a, m_a)
            }
        } else {
            let (o, m, _) = best_luma_orders(filters);
            (false, o, m)
        };
        if n > 1 {
            w.u1(pred); // alf_luma_coeff_delta_prediction_flag
        }
        w.ue(min_order - 1); // alf_luma_min_eg_order_minus1
        let mut k = min_order;
        for &o in orders.iter().take(3) {
            w.u1(o > k); // alf_luma_eg_order_increase_flag[i]
            k = o;
        }
        let coded = if pred { &diffs } else { filters };
        for f in coded.iter() {
            for (j, &v) in f.iter().enumerate() {
                let order = orders[GOLOMB_ORDER_IDX_Y[j].min(2)];
                w.uek(order, v.unsigned_abs());
                if v != 0 {
                    w.u1(v > 0); // sign: 0 negative, 1 positive
                }
            }
        }
    }
    if let Some(c) = chroma {
        let (orders, min_order, _) = best_chroma_orders(c);
        w.ue(min_order - 1); // alf_chroma_min_eg_order_minus1
        let mut k = min_order;
        for &o in orders.iter().take(CHROMA_MAX_GOLOMB_IDX) {
            w.u1(o > k);
            k = o;
        }
        for (j, &v) in c.iter().enumerate() {
            let order = orders[GOLOMB_ORDER_IDX_C[j].min(CHROMA_MAX_GOLOMB_IDX - 1)];
            w.uek(order, v.unsigned_abs());
            if v != 0 {
                w.u1(v > 0);
            }
        }
    }
}

/// Bits of `uek(k, v)`.
fn uek_bits(k: u32, v: u32) -> u32 {
    let mut m = 0u32;
    while ((1u64 << (m + 1)) - 1) << k <= u64::from(v) {
        m += 1;
    }
    2 * m + 1 + k
}

/// The cheapest `(expGoOrderY[0..3], alf_luma_min_eg_order, bits)` for
/// a luma coefficient block: orders `k0 <= k1 <= k2` with unit steps,
/// `k0 ∈ 1..=7`.
fn best_luma_orders(filters: &[[i32; NL]]) -> ([u32; 3], u32, u32) {
    let mut best = ([1u32; 3], 1u32, u32::MAX);
    for k0 in 1..=7u32 {
        for inc1 in 0..=1u32 {
            for inc2 in 0..=1u32 {
                let orders = [k0, k0 + inc1, k0 + inc1 + inc2];
                let mut bits = 0u32;
                for f in filters {
                    for (j, &v) in f.iter().enumerate() {
                        let k = orders[GOLOMB_ORDER_IDX_Y[j].min(2)];
                        bits += uek_bits(k, v.unsigned_abs()) + u32::from(v != 0);
                    }
                }
                if bits < best.2 {
                    best = (orders, k0, bits);
                }
            }
        }
    }
    best
}

/// The cheapest `(expGoOrderC[0..2], min order, bits)` for the chroma
/// coefficients.
fn best_chroma_orders(c: &[i32; NC]) -> ([u32; 2], u32, u32) {
    let mut best = ([1u32; 2], 1u32, u32::MAX);
    for k0 in 1..=7u32 {
        for inc1 in 0..=1u32 {
            let orders = [k0, k0 + inc1];
            let mut bits = 0u32;
            for (j, &v) in c.iter().enumerate() {
                let k = orders[GOLOMB_ORDER_IDX_C[j].min(1)];
                bits += uek_bits(k, v.unsigned_abs()) + u32::from(v != 0);
            }
            if bits < best.2 {
                best = (orders, k0, bits);
            }
        }
    }
    best
}

/// The complete APS RBSP around an `alf_data()` (§7.3.2.3):
/// `adaptation_parameter_set_id = 0`, `aps_params_type = 0` (ALF), the
/// structure, `aps_extension_flag = 0`, `rbsp_trailing_bits()`.
pub fn write_alf_aps_rbsp(
    luma: Option<(&[[i32; NL]], &[u8; NUM_ALF_FILTERS])>,
    chroma: Option<&[i32; NC]>,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(5, 0); // adaptation_parameter_set_id
    w.u(3, 0); // aps_params_type = ALF
    write_alf_data(&mut w, luma, chroma);
    w.u1(false); // aps_extension_flag
    w.rbsp_trailing_bits();
    w.into_bytes()
}

/// One evaluated luma filter set: `(cost, filters, class map, CTB
/// flags, map partial)`.
type LumaCandidate = (f64, Vec<[i32; NL]>, [u8; NUM_ALF_FILTERS], Vec<bool>, bool);

/// Per-CTB luma statistics of one picture: the 25 class accumulators
/// (the padded input and classification exactly as the decoder derives
/// them) and, per CTB, the unfiltered SSE.
struct LumaStats {
    classes: Vec<Stats<NL>>,
}

fn gather_luma(
    src: &YuvPicture,
    rec: &YuvPicture,
    avail: &AlfInputAvailability<'_>,
    ctb_log2: u32,
) -> LumaStats {
    let w = rec.width as usize;
    let h = rec.height as usize;
    let stride = rec.y_stride();
    let ctb = 1usize << ctb_log2;
    let perms: [[usize; 13]; 4] = std::array::from_fn(|t| transpose_perm(t as u8));
    let mut classes = vec![Stats::<NL>::zero(); NUM_ALF_FILTERS];
    for y_ctb in (0..h).step_by(ctb) {
        for x_ctb in (0..w).step_by(ctb) {
            let blk_w = (x_ctb + ctb).min(w) - x_ctb;
            let blk_h = (y_ctb + ctb).min(h) - y_ctb;
            let pad = alf::derive_alf_input(
                &rec.y, stride, w, h, x_ctb, y_ctb, blk_w, blk_h, 1, 1, avail,
            );
            let cls = alf::derive_alf_classification_padded(&pad, blk_w, blk_h, rec.bit_depth);
            for y in 0..blk_h {
                for x in 0..blk_w {
                    let c = cls.filt_idx_at(x, y) as usize;
                    let perm = &perms[cls.transpose_idx_at(x, y) as usize];
                    let centre = pad.at(x as i32, y as i32);
                    let mut d = [0f64; NL];
                    for k in 0..NL {
                        let (dy0, dx0) = LUMA_TAPS[k];
                        let (dy1, dx1) = LUMA_TAPS_SYM[k];
                        let pair = pad.at(x as i32 + dx0, y as i32 + dy0)
                            + pad.at(x as i32 + dx1, y as i32 + dy1)
                            - 2 * centre;
                        d[perm[k]] += f64::from(pair);
                    }
                    let s = src.y[(y_ctb + y) * src.y_stride() + (x_ctb + x)] as i32;
                    classes[c].add(&d, f64::from(s - centre));
                }
            }
        }
    }
    LumaStats { classes }
}

fn gather_chroma(
    src: &YuvPicture,
    rec: &YuvPicture,
    avail: &AlfInputAvailability<'_>,
    ctb_log2: u32,
) -> Stats<NC> {
    let cw = rec.width.div_ceil(2) as usize;
    let chh = rec.height.div_ceil(2) as usize;
    let stride = rec.c_stride();
    let ctb_c = (1usize << ctb_log2) / 2;
    let mut st = Stats::<NC>::zero();
    for (plane, splane) in [(&rec.cb, &src.cb), (&rec.cr, &src.cr)] {
        for y_ctb in (0..chh).step_by(ctb_c) {
            for x_ctb in (0..cw).step_by(ctb_c) {
                let blk_w = (x_ctb + ctb_c).min(cw) - x_ctb;
                let blk_h = (y_ctb + ctb_c).min(chh) - y_ctb;
                let pad = alf::derive_alf_input(
                    plane, stride, cw, chh, x_ctb, y_ctb, blk_w, blk_h, 2, 2, avail,
                );
                for y in 0..blk_h {
                    for x in 0..blk_w {
                        let centre = pad.at(x as i32, y as i32);
                        let mut d = [0f64; NC];
                        for k in 0..NC {
                            let (dy0, dx0) = CHROMA_TAPS[k];
                            let (dy1, dx1) = CHROMA_TAPS_SYM[k];
                            d[k] = f64::from(
                                pad.at(x as i32 + dx0, y as i32 + dy0)
                                    + pad.at(x as i32 + dx1, y as i32 + dy1)
                                    - 2 * centre,
                            );
                        }
                        let s = splane[(y_ctb + y) * src.c_stride() + (x_ctb + x)] as i32;
                        st.add(&d, f64::from(s - centre));
                    }
                }
            }
        }
    }
    st
}

/// SSE of `a` against `b` over a rectangle of a plane.
fn sse_rect(a: &[u16], b: &[u16], stride: usize, x0: usize, y0: usize, w: usize, h: usize) -> f64 {
    let mut acc = 0f64;
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let d = a[y * stride + x] as f64 - b[y * stride + x] as f64;
            acc += d * d;
        }
    }
    acc
}

/// Greedy class merging: the class → group assignment at every level
/// from 25 groups down to 1 (index `k − 1` holds the `k`-group map).
fn merge_levels(classes: &[Stats<NL>]) -> Vec<[u8; NUM_ALF_FILTERS]> {
    let sse_opt = |s: &Stats<NL>| -> f64 {
        match s.solve() {
            Some(f) => {
                let c: [i32; NL] = std::array::from_fn(|i| f[i].round() as i32);
                s.sse(&c).min(s.sy2)
            }
            None => s.sy2,
        }
    };
    let mut groups: Vec<(Vec<usize>, Stats<NL>, f64)> = classes
        .iter()
        .enumerate()
        .map(|(i, s)| (vec![i], s.clone(), sse_opt(s)))
        .collect();
    let mut levels = vec![[0u8; NUM_ALF_FILTERS]; NUM_ALF_FILTERS];
    loop {
        let k = groups.len();
        let mut map = [0u8; NUM_ALF_FILTERS];
        for (g, (members, _, _)) in groups.iter().enumerate() {
            for &m in members {
                map[m] = g as u8;
            }
        }
        levels[k - 1] = map;
        if k == 1 {
            break;
        }
        let mut best: Option<(usize, usize, f64, Stats<NL>, f64)> = None;
        for i in 0..k {
            for j in i + 1..k {
                let merged = groups[i].1.merge(&groups[j].1);
                let s = sse_opt(&merged);
                let delta = s - groups[i].2 - groups[j].2;
                if best.as_ref().map_or(true, |b| delta < b.2) {
                    best = Some((i, j, delta, merged, s));
                }
            }
        }
        let (i, j, _, merged, s) = best.expect("two groups");
        let (mut mi, _, _) = groups.remove(j);
        let (mj, _, _) = groups.remove(i);
        mi.extend(mj);
        groups.insert(i, (mi, merged, s));
    }
    levels
}

/// Design the ALF for one picture: `src` the source, `rec` the
/// reconstruction after deblocking (the decoder's ALF input), `avail`
/// the slice's §8.8.4.5/.6 edge availability, `lambda` the RD multiplier
/// in SSE-per-bit units. Returns `None` when no filter pays for itself;
/// otherwise the slice parameters, with `rec` **filtered in place**
/// exactly as the decoder will filter it.
pub fn design_and_apply(
    src: &YuvPicture,
    rec: &mut YuvPicture,
    avail: &AlfInputAvailability<'_>,
    lambda: f64,
    ctb_log2: u32,
) -> Result<Option<AlfSliceParams>> {
    let bd = rec.bit_depth;
    let w = rec.width as usize;
    let h = rec.height as usize;
    let ctb = 1usize << ctb_log2;
    let ctbs_w = w.div_ceil(ctb);
    let ctbs_h = h.div_ceil(ctb);
    let n_ctb = ctbs_w * ctbs_h;
    let nal_overhead_bits = 8.0 * 6.0; // length prefix + NAL header

    // ---- luma ----
    let luma = gather_luma(src, rec, avail, ctb_log2);
    let levels = merge_levels(&luma.classes);
    let mut ctb_sse_off = vec![0f64; n_ctb];
    for ry in 0..ctbs_h {
        for rx in 0..ctbs_w {
            let (x0, y0) = (rx * ctb, ry * ctb);
            let (bw, bh) = ((x0 + ctb).min(w) - x0, (y0 + ctb).min(h) - y0);
            ctb_sse_off[ry * ctbs_w + rx] =
                sse_rect(&src.y, &rec.y, rec.y_stride(), x0, y0, bw, bh);
        }
    }
    let luma_off_cost: f64 = ctb_sse_off.iter().sum();
    // Candidate filter counts; each evaluated with real CTB elections.
    let mut best_luma: Option<LumaCandidate> = None;
    for &k in &[1usize, 2, 3, 4, 6, 8, 12, 16, 25] {
        let map = levels[k - 1];
        let mut group_stats = vec![Stats::<NL>::zero(); k];
        for (c, s) in luma.classes.iter().enumerate() {
            group_stats[map[c] as usize] = group_stats[map[c] as usize].merge(s);
        }
        let filters: Vec<[i32; NL]> = group_stats.iter().map(|s| s.quantize(-512, 511)).collect();
        let aps = write_alf_aps_rbsp(Some((&filters, &map)), None);
        let data = alf::parse_alf_data(&aps[1..])?;
        // Filter every CTB once (the decoder's apply is per-CTB over a
        // pre-ALF snapshot, so the all-on picture holds each CTB's result).
        let mut all_on = AlfCtbMap::new(rec.width, rec.height, ctb_log2);
        for i in 0..n_ctb {
            all_on.set(i, true, false, false);
        }
        let mut filtered = rec.clone();
        alf::apply_alf_luma_availability(&mut filtered, &data, &all_on, avail, bd);
        let mut flags = vec![false; n_ctb];
        let mut on_sse = 0f64;
        for ry in 0..ctbs_h {
            for rx in 0..ctbs_w {
                let i = ry * ctbs_w + rx;
                let (x0, y0) = (rx * ctb, ry * ctb);
                let (bw, bh) = ((x0 + ctb).min(w) - x0, (y0 + ctb).min(h) - y0);
                let s_on = sse_rect(&src.y, &filtered.y, rec.y_stride(), x0, y0, bw, bh);
                if s_on < ctb_sse_off[i] {
                    flags[i] = true;
                    on_sse += s_on;
                } else {
                    on_sse += ctb_sse_off[i];
                }
            }
        }
        if !flags.iter().any(|&f| f) {
            continue;
        }
        let all = flags.iter().all(|&f| f);
        let flag_bits = if all { 0.0 } else { n_ctb as f64 };
        let cost = on_sse + lambda * (8.0 * aps.len() as f64 + nal_overhead_bits + flag_bits);
        if best_luma.as_ref().map_or(true, |b| cost < b.0) {
            best_luma = Some((cost, filters, map, flags, !all));
        }
    }
    let luma_choice = match best_luma {
        Some(b) if b.0 < luma_off_cost => Some(b),
        _ => None,
    };

    // ---- chroma (one filter for both planes; per-plane election) ----
    let cst = gather_chroma(src, rec, avail, ctb_log2);
    let cfilter = cst.quantize(-512, 511);
    let chroma_choice = if cfilter.iter().any(|&v| v != 0) {
        let aps_probe = write_alf_aps_rbsp(None, Some(&cfilter));
        let data = alf::parse_alf_data(&aps_probe[1..])?;
        let mut all_on = AlfCtbMap::new(rec.width, rec.height, ctb_log2);
        for i in 0..n_ctb {
            all_on.set(i, false, true, true);
        }
        let mut filtered = rec.clone();
        alf::apply_alf_chroma_availability(
            &mut filtered,
            &data.chroma_filters[0],
            &all_on,
            avail,
            1,
            bd,
        );
        alf::apply_alf_chroma_availability(
            &mut filtered,
            &data.chroma_filters[0],
            &all_on,
            avail,
            2,
            bd,
        );
        let cw = rec.width.div_ceil(2) as usize;
        let chh = rec.height.div_ceil(2) as usize;
        let cs = rec.c_stride();
        let cb_off = sse_rect(&src.cb, &rec.cb, cs, 0, 0, cw, chh);
        let cb_on = sse_rect(&src.cb, &filtered.cb, cs, 0, 0, cw, chh);
        let cr_off = sse_rect(&src.cr, &rec.cr, cs, 0, 0, cw, chh);
        let cr_on = sse_rect(&src.cr, &filtered.cr, cs, 0, 0, cw, chh);
        let coef_bits = 8.0 * (aps_probe.len() as f64 - 1.0);
        let gain = (cb_off - cb_on).max(0.0) + (cr_off - cr_on).max(0.0);
        if gain > lambda * coef_bits {
            let idc = u32::from(cb_on < cb_off) | (u32::from(cr_on < cr_off) << 1);
            Some((cfilter, idc))
        } else {
            None
        }
    } else {
        None
    };

    if luma_choice.is_none() && chroma_choice.is_none() {
        return Ok(None);
    }
    // Chroma rides only with a luma-enabled slice (§7.3.4: the chroma
    // idc is coded inside the slice_alf_enabled_flag block for 4:2:0);
    // a chroma-only design enables luma with an all-off map.
    let (filters, map, mut flags, mut map_flag) = match luma_choice {
        Some((_, f, m, fl, mf)) => (Some(f), m, fl, mf),
        None => (None, [0u8; NUM_ALF_FILTERS], vec![false; n_ctb], true),
    };
    if filters.is_none() && !map_flag {
        map_flag = true;
        flags.iter_mut().for_each(|f| *f = false);
    }
    let luma_arg = filters.as_ref().map(|f| (f.as_slice(), &map));
    let chroma_arg = chroma_choice.as_ref().map(|(c, _)| c);
    let aps_rbsp = write_alf_aps_rbsp(luma_arg, chroma_arg);
    let data = alf::parse_alf_data(&aps_rbsp[1..])?;
    let chroma_idc = chroma_choice.map_or(0, |(_, idc)| idc);
    let params = AlfSliceParams {
        aps_rbsp,
        luma_enabled: true,
        map_flag,
        chroma_idc,
        ctb_luma: flags,
        data,
    };
    apply_decided(rec, &params, avail, ctb_log2);
    Ok(Some(params))
}

/// Filter `pic` exactly as the decoder's post-filter pass will for a
/// slice carrying `params` (`apply_post_filters`: the classified luma
/// apply over the padded inputs when any CTB is on, then each enabled
/// chroma plane over its per-CTB map).
pub fn apply_decided(
    pic: &mut YuvPicture,
    params: &AlfSliceParams,
    avail: &AlfInputAvailability<'_>,
    ctb_log2: u32,
) {
    let map = params.ctb_map(pic.width, pic.height, ctb_log2);
    let bd = pic.bit_depth;
    if map.any_luma_on() {
        alf::apply_alf_luma_availability(pic, &params.data, &map, avail, bd);
    }
    let cb = params.chroma_idc == 1 || params.chroma_idc == 3;
    let cr = params.chroma_idc == 2 || params.chroma_idc == 3;
    crate::decoder::apply_chroma_alf_masked_or_whole_plane(
        pic,
        &map,
        Some(&params.data),
        Some(&params.data),
        cb,
        cr,
        avail,
        bd,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `alf_data()` writer is the exact dual of the parser: random
    /// filter sets at every signalled count, both delta-prediction
    /// choices and every Golomb order the writer may pick read back
    /// into the same derived coefficients; the chroma filter and the
    /// eq. 104 / 110 DC terms match.
    #[test]
    fn alf_data_writer_reads_back() {
        let mut seed = 0xA1F0_0458u32;
        let mut next = |range: i32| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % (2 * range + 1)) - range
        };
        for &n in &[1usize, 2, 5, 13, 25] {
            for &amp in &[2i32, 20, 200] {
                let filters: Vec<[i32; NL]> = (0..n)
                    .map(|_| std::array::from_fn(|_| next(amp).clamp(-512, 511)))
                    .collect();
                let map: [u8; NUM_ALF_FILTERS] = std::array::from_fn(|i| ((i * 7 + 3) % n) as u8);
                let chroma: [i32; NC] = std::array::from_fn(|_| next(amp).clamp(-512, 511));
                let rbsp = write_alf_aps_rbsp(Some((&filters, &map)), Some(&chroma));
                let aps = crate::aps::parse(&rbsp).unwrap();
                assert!(aps.is_alf() && !aps.aps_extension_flag);
                let data = alf::parse_alf_data(&aps.payload_raw).unwrap();
                assert!(data.luma_filter_signal && data.chroma_filter_signal);
                assert_eq!(data.num_signalled_luma_filters, n);
                for c in 0..NUM_ALF_FILTERS {
                    let f = &filters[map[c] as usize];
                    let got = data.luma_filters[c].coef;
                    for j in 0..NL {
                        assert_eq!(i32::from(got[j]), f[j], "n{n} amp{amp} class {c} tap {j}");
                    }
                    let dc = 512 - 2 * f.iter().sum::<i32>();
                    assert_eq!(i32::from(got[12]), dc.clamp(-1024, 1023));
                }
                let cf = data.chroma_filters[0].coef;
                for j in 0..NC {
                    assert_eq!(i32::from(cf[j]), chroma[j]);
                }
                assert_eq!(
                    i32::from(cf[6]),
                    (512 - 2 * chroma.iter().sum::<i32>()).clamp(-1024, 1023)
                );
            }
        }
        // Luma-only and chroma-only shapes.
        let f = [[3i32, -2, 0, 1, 0, 0, 5, 0, -1, 0, 0, 7]];
        let map = [0u8; NUM_ALF_FILTERS];
        let d = alf::parse_alf_data(&write_alf_aps_rbsp(Some((&f, &map)), None)[1..]).unwrap();
        assert!(d.luma_filter_signal && !d.chroma_filter_signal);
        let d =
            alf::parse_alf_data(&write_alf_aps_rbsp(None, Some(&[1, 2, 3, 4, 5, 6]))[1..]).unwrap();
        assert!(!d.luma_filter_signal && d.chroma_filter_signal);
    }

    /// The least-squares solver recovers a planted filter from its own
    /// regressors, and the quantiser never scores worse than rounding.
    #[test]
    fn wiener_solver_recovers_planted_filter() {
        let planted: [i32; NL] = [7, -3, 12, 0, 5, -20, 2, 9, -6, 1, 4, 30];
        let mut st = Stats::<NL>::zero();
        let mut seed = 0x77u32;
        for _ in 0..4000 {
            let mut d = [0f64; NL];
            for v in d.iter_mut() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = f64::from(((seed >> 16) as i32 % 61) - 30);
            }
            let y: f64 = d
                .iter()
                .zip(planted.iter())
                .map(|(&x, &c)| x * f64::from(c))
                .sum::<f64>()
                / 512.0;
            st.add(&d, y);
        }
        let f = st.solve().unwrap();
        for j in 0..NL {
            assert!(
                (f[j] - f64::from(planted[j])).abs() < 1e-3,
                "tap {j}: {}",
                f[j]
            );
        }
        assert_eq!(st.quantize(-512, 511), planted);
        assert!(st.sse(&planted) < 1e-3);
    }

    /// `uek_bits` agrees with the writer.
    #[test]
    fn uek_bit_count_matches_writer() {
        for k in 0..4u32 {
            for v in [0u32, 1, 2, 3, 7, 8, 100, 1000] {
                let mut w = BitWriter::new();
                w.uek(k, v);
                assert_eq!(w.bit_position() as u32, uek_bits(k, v), "k{k} v{v}");
                w.align_to_byte_zero();
                let bytes = w.into_bytes();
                let mut r = crate::bitreader::BitReader::new(&bytes);
                assert_eq!(r.uek(k).unwrap(), v);
            }
        }
    }
}
