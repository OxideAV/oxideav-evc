//! The RDOQ trellis on fuzzed transform blocks: fractional levels and
//! SSE weights from a fuzzed residual at a fuzzed size / QP / λ /
//! entropy shape, then (a) structural invariants of the chosen levels,
//! (b) the chosen block never scores worse than nearest rounding under
//! the trellis's own D + λ·R objective, and (c) the §7.3.8.7 RLE string
//! the encoder writes for it decodes back to the same levels through
//! the decoder's `decode_residual_coding_rle`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_evc::bin_cost::BitCostModel;
use oxideav_evc::cabac::{BinSink, CabacEncoder, CabacEngine, InitType};
use oxideav_evc::cabac_init::{init_main_profile_contexts, CtxSel, MainCtxTable};
use oxideav_evc::quant_enc::{forward_quantize, forward_transform_fractional, level_unit_sse_weights};
use oxideav_evc::rdoq::{rdoq_rle, RdoqInputs};
use oxideav_evc::slice_data::decode_residual_coding_rle;
use oxideav_evc::slice_enc::emit_residual_rle;

fn account(
    levels: &[i32],
    frac: &[f64],
    weights: &[f64],
    log2_w: u32,
    log2_h: u32,
    inp: &RdoqInputs<'_>,
) -> f64 {
    let mut m = inp.model.clone();
    let cbf = levels.iter().any(|&l| l != 0);
    let bits = m.measure(|m| {
        m.encode_decision(inp.cbf_ctx.0, inp.cbf_ctx.1, u8::from(cbf));
        if cbf {
            emit_residual_rle(m, inp.sel, inp.c_idx, levels, log2_w, log2_h);
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

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let log2_w = 1 + u32::from(data[0] % 6); // 2..=64
    let log2_h = 1 + u32::from(data[1] % 6);
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    let n = w * h;
    let bit_depth = if data[2] & 1 != 0 { 10 } else { 8 };
    let qp = i32::from(data[2] >> 1) % 64; // the §8.7 Qp′ domain
    let cm_init = data[3] & 1 != 0;
    let c_idx = u32::from((data[3] >> 1) % 3);
    let init_type = if data[3] & 8 != 0 {
        InitType::Pb
    } else {
        InitType::I
    };
    let slice_qp = i32::from(data[4] % 52);
    let lambda = f64::from(data[5]) * f64::from(data[5]) / 16.0;
    let amp = 1i32 << bit_depth;
    let payload = &data[6..];
    let residual: Vec<i32> = (0..n)
        .map(|i| {
            if payload.is_empty() {
                0
            } else {
                let b = i32::from(payload[i % payload.len()]);
                let sign = if (i / payload.len()) % 2 == 0 { 1 } else { -1 };
                (sign * b * amp / 256).clamp(-amp, amp)
            }
        })
        .collect();

    let frac = forward_transform_fractional(&residual, w, h, qp, bit_depth).expect("fractional");
    let weights = level_unit_sse_weights(w, h, qp, bit_depth);
    let sel = CtxSel::new(cm_init, init_type);
    let mut model = BitCostModel::new();
    if cm_init {
        model.init_main_profile(init_type, slice_qp);
    }
    // Some context state: run a few bins so the states are not all default.
    model.commit(|m| {
        for i in 0..usize::from(data[4]) {
            let (t, ci) = sel.ctx(MainCtxTable::CoeffLastFlag, i % 2);
            m.encode_decision(t, ci, (i % 3 == 0) as u8);
        }
    });
    let table = match c_idx {
        0 => MainCtxTable::CbfLuma,
        1 => MainCtxTable::CbfCb,
        _ => MainCtxTable::CbfCr,
    };
    let inp = RdoqInputs::new(&model, lambda, sel, c_idx, table);
    let (levels, cbf, cost) = rdoq_rle(&frac, &weights, w, h, &inp);

    // (a) invariants
    assert_eq!(levels.len(), n);
    assert_eq!(cbf, levels.iter().any(|&l| l != 0));
    assert!(levels.iter().all(|l| (-32768..=32767).contains(l)));
    assert!(cost.is_finite() && cost >= 0.0);
    for (l, c) in levels.iter().zip(frac.iter()) {
        // Never further from the fractional value than rounding is, on
        // the same side (or zero).
        assert!(*l == 0 || (*l > 0) == (*c > 0.0), "sign: {l} vs {c}");
        assert!((f64::from(*l) - c).abs() <= c.abs() + 1.0, "magnitude: {l} vs {c}");
    }

    // (b) never worse than rounding under the model's objective
    let mut plain = vec![0i32; n];
    forward_quantize(&residual, &mut plain, w, h, qp, bit_depth).expect("quantize");
    let own = account(&levels, &frac, &weights, log2_w, log2_h, &inp);
    let rounded = account(&plain, &frac, &weights, log2_w, log2_h, &inp);
    assert!((own - cost).abs() <= 1e-6 * cost.max(1.0), "accounting {own} vs {cost}");
    assert!(own <= rounded + 1e-6 * rounded.max(1.0), "trellis {own} worse than rounding {rounded}");

    // (c) the RLE string round-trips through the decoder
    if cbf {
        let mut enc = CabacEncoder::new();
        if cm_init {
            enc.init_main_profile(init_type, slice_qp);
        }
        emit_residual_rle(&mut enc, sel, c_idx, &levels, log2_w, log2_h);
        enc.encode_terminate(true);
        let bytes = enc.finish();
        let mut eng = CabacEngine::new(&bytes).expect("engine");
        if cm_init {
            init_main_profile_contexts(&mut eng, init_type, slice_qp).expect("init");
        }
        let mut decoded = vec![0i32; n];
        let mut runs = 0u32;
        decode_residual_coding_rle(&mut eng, sel, c_idx, &mut decoded, &mut runs, log2_w, log2_h)
            .expect("decode");
        assert_eq!(decoded, levels, "RLE duality");
    }
});
