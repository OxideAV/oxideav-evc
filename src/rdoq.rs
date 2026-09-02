//! Rate-distortion optimised quantization (RDOQ) for the §7.3.8.7
//! run-length residual syntax.
//!
//! The plain quantizer rounds every coefficient to its nearest level.
//! RDOQ instead chooses, per transform block, the level vector that
//! minimises `D + λ · R` where `D` is the (transform-domain, SSE-scaled)
//! quantization error and `R` the exact number of bits the
//! `residual_coding_rle()` string costs at the entropy coder's current
//! context state — level decisions (nearest vs. one lower vs. zero),
//! last-position trimming (dropping a trailing tail of small
//! coefficients) and the whole-block zero-out (`cbf = 0`) all fall out
//! of one dynamic programme.
//!
//! ## The syntax's rate structure
//!
//! `residual_coding_rle()` (§7.3.8.7) codes each non-zero coefficient in
//! §6.5.2 scan order as `coeff_zero_run` (U, Table 84), `coeff_abs_level_minus1`
//! (U, Table 85), `coeff_sign_flag` (bypass) and, unless the scan is
//! exhausted, `coeff_last_flag` (Table 86). The run/level bins take the
//! §9.3.4.2.2 eq. 1434/1435 contexts `( Min( PrevLevel − 1, 5 ) << 1 ) +
//! Min( binIdx, 1 )` (`+ 12` for chroma) under `sps_cm_init_flag == 1`,
//! and the Table-95 `cIdx == 0 ? 0 : 2` / `+ Min( binIdx, 1 )` pair on
//! the shared context table under `== 0`. So the rate of a coefficient
//! depends on its own magnitude, the zero run before it and the
//! **bucket** `Min( PrevLevel − 1, 5 )` of the previous non-zero level
//! (`PrevLevel = 6` at the start). With the per-bin costs held at the
//! current context state, a run of `r ≥ 1` zeros costs `A + B · r` bits
//! (bin 0 as `1`, `r − 1` continuation `1`s, one terminating `0`) and a
//! run of 0 costs the single bin-0 `0` — linear in `r`, which is what
//! makes the trellis below linear in the block size.
//!
//! ## The trellis
//!
//! State = (scan position `p` of a non-zero level, bucket `b` of that
//! level). `best[ p ][ b ]` is the least `D + λ · R` of any level vector
//! whose last non-zero so far sits at `p` with bucket `b`, the
//! `coeff_last_flag` of that coefficient provisionally `0`. A
//! transition from `( q, b′ )` adds the zero distortion of positions
//! `q + 1 .. p − 1` (prefix sums), the run cost at bucket `b′`, the
//! level/sign/last bins and the coefficient's own distortion. The
//! `r ≥ 1` transitions are folded into one running minimum per `b′`
//! (`best[ q ][ b′ ] − Z[ q + 1 ] − λ · B_b′ · q`), so each position
//! costs `O( buckets × candidate levels )`. The block's answer is the
//! best `( p, b )` with its last flag flipped to `1` (or absent at the
//! final scan position) plus the tail's zero distortion, compared
//! against the all-zero block under the caller's `cbf` bin costs.
//!
//! Candidate magnitudes per position are `⌊|c|⌋`, `⌈|c|⌉` and `0`
//! around the fractional level `c` ([`crate::quant_enc::forward_transform_fractional`]).
//! Everything is `f64` arithmetic on integer-derived costs, so the
//! choice — and the emitted bytes — are platform-deterministic.

use crate::bin_cost::{bin_cost, BitCostModel};
use crate::cabac_init::{ctx_inc_coeff_zero_run, CtxSel, MainCtxTable};
use crate::slice_data::zigzag_scan;

/// Per-bucket static bin costs of the run and level U strings.
#[derive(Clone, Copy)]
struct BucketCosts {
    /// `coeff_zero_run == 0`: the lone bin-0 `0`.
    run_zero: f64,
    /// `coeff_zero_run == r ≥ 1`: `run_a + run_b · r`.
    run_a: f64,
    run_b: f64,
    /// `coeff_abs_level_minus1 == 0`: the lone bin-0 `0`.
    lvl_one: f64,
    /// `coeff_abs_level_minus1 == m ≥ 1`: `lvl_a + lvl_b · m`.
    lvl_a: f64,
    lvl_b: f64,
}

/// The `( ctxTable, ctxIdx )` of run/level bin `bin_idx` at bucket `b`
/// (`PrevLevel − 1 = b`) under the selector's entropy shape.
fn run_lvl_ctx(
    sel: CtxSel,
    table: MainCtxTable,
    c_idx: u32,
    b: usize,
    bin_idx: u32,
) -> (usize, usize) {
    if sel.cm_init {
        (
            table.as_usize(),
            table.ctx_idx_offset(sel.init_type)
                + ctx_inc_coeff_zero_run(bin_idx, c_idx, b as u32 + 1),
        )
    } else {
        let chroma = if c_idx == 0 { 0 } else { 2 };
        (
            0,
            table.cm0_ctx_idx_offset(sel.init_type) + chroma + (bin_idx.min(1) as usize),
        )
    }
}

fn bucket_costs(model: &BitCostModel, sel: CtxSel, c_idx: u32, b: usize) -> BucketCosts {
    let cost = |table: MainCtxTable, bin_idx: u32, bin: u8| -> f64 {
        let (t, i) = run_lvl_ctx(sel, table, c_idx, b, bin_idx);
        bin_cost(model.context(t, i), bin)
    };
    let r00 = cost(MainCtxTable::CoeffZeroRun, 0, 0);
    let r01 = cost(MainCtxTable::CoeffZeroRun, 0, 1);
    let r10 = cost(MainCtxTable::CoeffZeroRun, 1, 0);
    let r11 = cost(MainCtxTable::CoeffZeroRun, 1, 1);
    let l00 = cost(MainCtxTable::CoeffAbsLevelMinus1, 0, 0);
    let l01 = cost(MainCtxTable::CoeffAbsLevelMinus1, 0, 1);
    let l10 = cost(MainCtxTable::CoeffAbsLevelMinus1, 1, 0);
    let l11 = cost(MainCtxTable::CoeffAbsLevelMinus1, 1, 1);
    BucketCosts {
        run_zero: r00,
        // r ≥ 1: bin0 = 1, (r − 1) × 1, then 0  ⇒  (r01 + r10 − r11) + r11·r
        run_a: r01 + r10 - r11,
        run_b: r11,
        lvl_one: l00,
        lvl_a: l01 + l10 - l11,
        lvl_b: l11,
    }
}

/// Bucket of a non-zero magnitude: `Min( |level| − 1, 5 )`.
#[inline]
fn bucket_of(abs_level: u32) -> usize {
    (abs_level - 1).min(5) as usize
}

/// Everything the trellis needs from the caller.
pub struct RdoqInputs<'a> {
    /// `λ` in the caller's SSE-per-bit units.
    pub lambda: f64,
    /// The entropy shape.
    pub sel: CtxSel,
    /// The context state the block will be coded at.
    pub model: &'a BitCostModel,
    /// Colour component (context selection).
    pub c_idx: u32,
    /// `( ctxTable, ctxIdx )` of the `cbf` bin that gates this block.
    pub cbf_ctx: (usize, usize),
}

impl<'a> RdoqInputs<'a> {
    /// Inputs for a block gated by `cbf_table`'s ctxInc-0 bin.
    pub fn new(
        model: &'a BitCostModel,
        lambda: f64,
        sel: CtxSel,
        c_idx: u32,
        cbf_table: MainCtxTable,
    ) -> Self {
        Self {
            lambda,
            sel,
            model,
            c_idx,
            cbf_ctx: sel.ctx(cbf_table, 0),
        }
    }
}

/// Trellis over one TB: `frac[ blkPos ]` are the fractional levels,
/// `weights[ blkPos ]` the SSE-per-unit-level² weights (both row-major
/// `blk_w × blk_h`). Returns the chosen levels (row-major), the `cbf`,
/// and the block's `D + λ · R` (including the `cbf` bin).
pub fn rdoq_rle(
    frac: &[f64],
    weights: &[f64],
    blk_w: usize,
    blk_h: usize,
    inputs: &RdoqInputs<'_>,
) -> (Vec<i32>, bool, f64) {
    let n = blk_w * blk_h;
    debug_assert_eq!(frac.len(), n);
    debug_assert_eq!(weights.len(), n);
    let lambda = inputs.lambda;
    let model = inputs.model;
    let sel = inputs.sel;
    let c_idx = inputs.c_idx;
    let scan = zigzag_scan(blk_w, blk_h);

    // Zero distortion prefix: z[i] = Σ_{p < i} dist(p, 0) (scan order).
    let mut z = vec![0f64; n + 1];
    for p in 0..n {
        let c = frac[scan[p]];
        z[p + 1] = z[p] + c * c * weights[scan[p]];
    }
    let z_all = z[n];
    let cbf0 = bin_cost(model.context(inputs.cbf_ctx.0, inputs.cbf_ctx.1), 0);
    let cbf1 = bin_cost(model.context(inputs.cbf_ctx.0, inputs.cbf_ctx.1), 1);
    let zero_cost = z_all + lambda * cbf0;

    // Trivially all-zero: nothing rounds to a non-zero level.
    if frac.iter().all(|c| c.abs() < 0.5) {
        return (vec![0i32; n], false, zero_cost);
    }

    let bc: [BucketCosts; 6] = core::array::from_fn(|b| bucket_costs(model, sel, c_idx, b));
    let last_inc = if c_idx == 0 { 0 } else { 1 };
    let (lt, li) = sel.ctx(MainCtxTable::CoeffLastFlag, last_inc);
    let last0 = bin_cost(model.context(lt, li), 0);
    let last1 = bin_cost(model.context(lt, li), 1);
    let sign = 1.0;

    const INF: f64 = f64::INFINITY;
    // best[p][b]: cost through a non-zero at p with bucket b (last = 0
    // provisional). back[p][b] = (q, b_prev, abs_level); q = -1 = start.
    let mut best = vec![[INF; 6]; n];
    let mut back = vec![[(-1i32, 0u8, 0u32); 6]; n];
    // Running minima for the r ≥ 1 transitions, per previous bucket;
    // the start pseudo-state (q = −1, PrevLevel 6 ⇒ bucket 5).
    let mut run_min = [INF; 6];
    run_min[5] = lambda * bc[5].run_b; // 0 − z[0] − λ·B·(−1)
    let mut run_min_arg = [(-1i32, 0u8); 6];
    // best[p − 1][·] with the start state at p = 0.
    let mut prev = [INF; 6];
    prev[5] = 0.0;
    let mut prev_q = -1i32;

    for p in 0..n {
        let c = frac[scan[p]];
        let w = weights[scan[p]];
        let mag = c.abs();
        let lo = mag.floor() as u32;
        let hi = lo + 1;
        let mut cands = [0u32; 2];
        let mut n_cand = 0;
        if lo >= 1 {
            cands[n_cand] = lo.min(32767);
            n_cand += 1;
        }
        if mag >= 0.25 && hi <= 32767 && (n_cand == 0 || cands[0] != hi) {
            cands[n_cand] = hi;
            n_cand += 1;
        }
        let mut cur = [INF; 6];
        let mut cur_back = [(-1i32, 0u8, 0u32); 6];
        let last_here = if p + 1 == n { 0.0 } else { last0 };
        for &abs in &cands[..n_cand] {
            let b = bucket_of(abs);
            let dist = (mag - abs as f64) * (mag - abs as f64) * w;
            for bp in 0..6 {
                // Reach p from bucket bp with run 0 (q = p − 1) or ≥ 1.
                let c1 = if prev[bp] < INF {
                    prev[bp] + lambda * bc[bp].run_zero
                } else {
                    INF
                };
                let c2 = if run_min[bp] < INF {
                    run_min[bp] + z[p] + lambda * (bc[bp].run_a + bc[bp].run_b * (p as f64 - 1.0))
                } else {
                    INF
                };
                let (base, from_q) = if c1 <= c2 {
                    (c1, prev_q)
                } else {
                    (c2, run_min_arg[bp].0)
                };
                if base >= INF {
                    continue;
                }
                let m = abs - 1;
                let lvl_bits = if m == 0 {
                    bc[bp].lvl_one
                } else {
                    bc[bp].lvl_a + bc[bp].lvl_b * m as f64
                };
                let total = base + lambda * (lvl_bits + sign + last_here) + dist;
                if total < cur[b] {
                    cur[b] = total;
                    cur_back[b] = (from_q, bp as u8, abs);
                }
            }
        }
        // Fold q = p − 1 into the r ≥ 1 running minima for p + 1 onward
        // (it must not serve p itself: that is the run-0 transition).
        for bp in 0..6 {
            if prev[bp] < INF {
                let v = prev[bp] - z[(prev_q + 1) as usize] - lambda * bc[bp].run_b * prev_q as f64;
                if v < run_min[bp] {
                    run_min[bp] = v;
                    run_min_arg[bp] = (prev_q, bp as u8);
                }
            }
        }
        best[p] = cur;
        back[p] = cur_back;
        prev = cur;
        prev_q = p as i32;
    }

    // Close: flip the final coefficient's last flag, add the tail zeros.
    let mut best_total = INF;
    let mut best_end = (0usize, 0usize);
    for p in 0..n {
        let close = if p + 1 == n { 0.0 } else { last1 - last0 };
        for (b, &cost_pb) in best[p].iter().enumerate() {
            if cost_pb < INF {
                let total = cost_pb + lambda * close + (z_all - z[p + 1]);
                if total < best_total {
                    best_total = total;
                    best_end = (p, b);
                }
            }
        }
    }
    let nz_cost = best_total + lambda * cbf1;
    if nz_cost >= zero_cost {
        return (vec![0i32; n], false, zero_cost);
    }
    let mut levels = vec![0i32; n];
    let (mut p, mut b) = best_end;
    loop {
        let (q, bp, abs) = back[p][b];
        let signed = if frac[scan[p]] < 0.0 {
            -(abs as i32)
        } else {
            abs as i32
        };
        levels[scan[p]] = signed;
        if q < 0 {
            break;
        }
        p = q as usize;
        b = bp as usize;
    }
    (levels, true, nz_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{BinSink, InitType};
    use crate::quant_enc::{
        forward_quantize, forward_transform_fractional, level_unit_sse_weights,
    };
    use crate::slice_enc::emit_residual_rle;

    fn inputs(model: &BitCostModel, sel: CtxSel, lambda: f64) -> RdoqInputs<'_> {
        RdoqInputs {
            lambda,
            sel,
            model,
            c_idx: 0,
            cbf_ctx: sel.ctx(MainCtxTable::CbfLuma, 0),
        }
    }

    /// The block's own D + λ·R accounting equals a re-measure of the
    /// chosen levels' bin string plus the transform-domain distortion.
    fn account(
        levels: &[i32],
        frac: &[f64],
        weights: &[f64],
        w: usize,
        h: usize,
        inp: &RdoqInputs<'_>,
    ) -> f64 {
        let mut m = inp.model.clone();
        let cbf = levels.iter().any(|&l| l != 0);
        let bits = m.measure(|m| {
            m.encode_decision(inp.cbf_ctx.0, inp.cbf_ctx.1, u8::from(cbf));
            if cbf {
                emit_residual_rle(
                    m,
                    inp.sel,
                    inp.c_idx,
                    levels,
                    (w as u32).trailing_zeros(),
                    (h as u32).trailing_zeros(),
                );
            }
        });
        let dist: f64 = frac
            .iter()
            .zip(levels.iter())
            .zip(weights.iter())
            .map(|((&c, &l), &wt)| (c - l as f64) * (c - l as f64) * wt)
            .sum();
        dist + inp.lambda * bits
    }

    fn noisy(n: usize, seed: u32, amp: i32) -> Vec<i32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 16) as i32 % (2 * amp + 1)) - amp
            })
            .collect()
    }

    /// With λ = 0 the trellis is the nearest-level quantizer.
    #[test]
    fn zero_lambda_is_nearest_rounding() {
        let (w, h) = (8usize, 8usize);
        let res = noisy(w * h, 7, 120);
        let frac = forward_transform_fractional(&res, w, h, 30, 8).unwrap();
        let weights = level_unit_sse_weights(w, h, 30, 8);
        let model = BitCostModel::new();
        let sel = CtxSel::baseline();
        let (levels, cbf, _) = rdoq_rle(&frac, &weights, w, h, &inputs(&model, sel, 0.0));
        assert!(cbf);
        for (l, c) in levels.iter().zip(frac.iter()) {
            assert_eq!(*l, c.round() as i32, "λ = 0 must round: {c} → {l}");
        }
    }

    /// The trellis never does worse than plain rounding under its own
    /// objective, on both entropy shapes and a spread of λ / QP.
    #[test]
    fn never_worse_than_rounding() {
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 8), (32, 32)] {
            for &qp in &[12i32, 30, 45] {
                for &cm in &[false, true] {
                    let res = noisy(w * h, 0x1234 + qp as u32, 60);
                    let frac = forward_transform_fractional(&res, w, h, qp, 8).unwrap();
                    let weights = level_unit_sse_weights(w, h, qp, 8);
                    let mut model = BitCostModel::new();
                    let sel = CtxSel::new(cm, InitType::I);
                    if cm {
                        model.init_main_profile(InitType::I, qp);
                    }
                    let lambda = crate::slice_enc::rd_lambda(qp, 8);
                    let inp = inputs(&model, sel, lambda);
                    let (levels, _cbf, cost) = rdoq_rle(&frac, &weights, w, h, &inp);
                    let mut plain = vec![0i32; w * h];
                    forward_quantize(&res, &mut plain, w, h, qp, 8).unwrap();
                    let plain_cost = account(&plain, &frac, &weights, w, h, &inp);
                    let own = account(&levels, &frac, &weights, w, h, &inp);
                    assert!(
                        (own - cost).abs() < 1e-6 * cost.max(1.0),
                        "{w}x{h} qp{qp} cm{cm}: accounting {own} vs trellis {cost}"
                    );
                    assert!(
                        cost <= plain_cost + 1e-9,
                        "{w}x{h} qp{qp} cm{cm}: trellis {cost} worse than rounding {plain_cost}"
                    );
                }
            }
        }
    }

    /// A block whose rounding gives a lone trailing ±1 far down the scan
    /// drops it once λ makes the run + level + last bins dearer than its
    /// distortion.
    #[test]
    fn trims_expensive_trailing_ones() {
        let (w, h) = (8usize, 8usize);
        let mut frac = vec![0f64; w * h];
        frac[0] = 12.3;
        frac[1] = -3.9;
        frac[w * h - 2] = 0.62; // far tail
        let weights = vec![1.0f64; w * h];
        let model = BitCostModel::new();
        let sel = CtxSel::baseline();
        let (levels, cbf, _) = rdoq_rle(&frac, &weights, w, h, &inputs(&model, sel, 0.5));
        assert!(cbf);
        assert_eq!(levels[0], 12);
        assert_eq!(levels[1], -4);
        assert_eq!(levels[w * h - 2], 0, "tail must be trimmed: {levels:?}");
        let (levels, _, _) = rdoq_rle(&frac, &weights, w, h, &inputs(&model, sel, 0.0));
        assert_eq!(levels[w * h - 2], 1, "λ = 0 keeps it");
    }

    /// A block of small coefficients zeroes out entirely at a high λ.
    #[test]
    fn zeroes_out_block_when_bits_outweigh_distortion() {
        let (w, h) = (4usize, 4usize);
        let frac: Vec<f64> = (0..16)
            .map(|i| if i % 5 == 0 { 0.7 } else { 0.1 })
            .collect();
        let weights = vec![1.0f64; 16];
        let model = BitCostModel::new();
        let sel = CtxSel::baseline();
        let (levels, cbf, _) = rdoq_rle(&frac, &weights, w, h, &inputs(&model, sel, 3.0));
        assert!(!cbf, "{levels:?}");
        assert!(levels.iter().all(|&l| l == 0));
        let (_, cbf, _) = rdoq_rle(&frac, &weights, w, h, &inputs(&model, sel, 0.01));
        assert!(cbf);
    }
}
