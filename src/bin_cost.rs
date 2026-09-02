//! Exact CABAC bin costs for the encoder's decide pass.
//!
//! The §9.3.4.3.2 decision engine splits `ivlCurrRange` in proportion
//! to `valState / 512` (eq. 1474: `ivlLpsRange = ( valState *
//! ivlCurrRange ) >> 9`), so a context variable `(valState, valMps)`
//! models `P( LPS ) = valState / 512`. The ideal cost of coding a bin
//! is therefore `−log2( valState / 512 )` bits for the LPS and
//! `−log2( ( 512 − valState ) / 512 )` for the MPS; a bypass bin costs
//! exactly one bit. [`BitCostModel`] keeps a full context table
//! (mirroring [`CabacEncoder`](crate::cabac::CabacEncoder) bin for bin),
//! implements [`BinSink`] so every syntax writer in `slice_enc` /
//! `slice_enc_p` can be replayed into it, and accumulates those costs
//! instead of a codeword — the rate term of the encoder's RD decisions
//! is then the entropy coder's own cost of the exact bin string it will
//! emit, at the context state it will emit it in.
//!
//! ## Determinism
//!
//! The `−log2` table is built by an integer binary-logarithm (repeated
//! mantissa squaring) in a `const fn`, not by `f64::log2`, so the costs
//! — and hence every RD decision and every emitted byte — are identical
//! on every platform. This is what keeps the MD5-pinned stream fixtures
//! honest across CI hosts.

use crate::cabac::{
    init_contexts_from_init_value, next_state, BinSink, ContextVar, InitType, MAX_CTX_PER_TABLE,
    MAX_CTX_TABLES,
};
use crate::cabac_init::MainCtxTable;

/// Fixed-point (Q16) `−log2( s / 512 )` for `s ∈ 1..=512`; entry 0 is
/// unused (a `valState` of 0 never occurs — §9.3.4.3.2.2 clamps to
/// `1..=511`) and aliases entry 1.
const COST_Q16: [u32; 513] = build_cost_table();

/// `−log2( s / 512 ) · 2^16`, rounded down, from an integer algorithm:
/// `log2( s ) = k + frac` with `k = floor( log2 s )` and the fraction
/// extracted bit by bit by squaring the normalised mantissa.
const fn build_cost_table() -> [u32; 513] {
    let mut table = [0u32; 513];
    let mut s = 1u32;
    while s <= 512 {
        let k = 31 - s.leading_zeros();
        // Mantissa in Q31, in [1, 2).
        let mut m: u64 = (s as u64) << (31 - k);
        let mut frac: u32 = 0;
        let mut i = 0;
        while i < 16 {
            m = (m * m) >> 31;
            frac <<= 1;
            if m >= (2u64 << 31) {
                frac |= 1;
                m >>= 1;
            }
            i += 1;
        }
        let log2_q16 = (k << 16) | frac;
        table[s as usize] = (9u32 << 16) - log2_q16;
        s += 1;
    }
    table[0] = table[1];
    table
}

/// The cost in bits of coding `bin` against `var` (static — the state
/// is not advanced).
#[inline]
pub fn bin_cost(var: ContextVar, bin: u8) -> f64 {
    let s = (var.val_state as usize).clamp(1, 511);
    let idx = if bin == var.val_mps { 512 - s } else { s };
    f64::from(COST_Q16[idx]) / 65536.0
}

/// A [`BinSink`] that accumulates the exact bin costs of everything
/// written into it against its own context table.
#[derive(Clone, Debug)]
pub struct BitCostModel {
    ctx: Vec<Vec<ContextVar>>,
    bits: f64,
    /// When false the context table is left untouched (a "what would
    /// this cost right now" probe); when true the states advance
    /// exactly like the real coder's.
    adapt: bool,
}

impl Default for BitCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl BitCostModel {
    /// Every context at the §9.3.2.2 case-1 default `(256, 0)` — the
    /// `sps_cm_init_flag == 0` shape.
    pub fn new() -> Self {
        Self {
            ctx: vec![vec![ContextVar::default(); MAX_CTX_PER_TABLE]; MAX_CTX_TABLES],
            bits: 0.0,
            adapt: true,
        }
    }

    /// §9.3.2.2 case 2 — the Main-profile init at `slice_qp`, the
    /// same table set `CabacEncoder::init_main_profile` installs.
    pub fn init_main_profile(&mut self, init_type: InitType, slice_qp: i32) {
        for &table in MainCtxTable::ALL {
            let (start, end) = table.init_type_range(init_type);
            let values = table.init_values();
            for (ctx_idx, &init_value) in values.iter().enumerate().take(end).skip(start) {
                self.ctx[table.as_usize()][ctx_idx] =
                    init_contexts_from_init_value(init_value, slice_qp);
            }
        }
    }

    /// The current state of one context variable.
    #[inline]
    pub fn context(&self, ctx_table: usize, ctx_idx: usize) -> ContextVar {
        self.ctx[ctx_table][ctx_idx]
    }

    /// Static cost of one regular bin at the current state.
    #[inline]
    pub fn decision_cost(&self, ctx_table: usize, ctx_idx: usize, bin: u8) -> f64 {
        bin_cost(self.ctx[ctx_table][ctx_idx], bin)
    }

    /// Bits accumulated so far.
    pub fn bits(&self) -> f64 {
        self.bits
    }

    /// Cost `f`'s bins at the current context state **without**
    /// advancing it — the probe for a candidate the encoder may not
    /// pick. Returns the bits `f` cost.
    pub fn measure<F: FnOnce(&mut Self)>(&mut self, f: F) -> f64 {
        let before = self.bits;
        let adapt = self.adapt;
        self.adapt = false;
        f(self);
        self.adapt = adapt;
        let cost = self.bits - before;
        self.bits = before;
        cost
    }

    /// Replay `f`'s bins **with** context adaptation — the decided
    /// syntax, in decode order, so the model's state tracks the real
    /// coder's. Returns the bits `f` cost.
    pub fn commit<F: FnOnce(&mut Self)>(&mut self, f: F) -> f64 {
        let before = self.bits;
        let adapt = self.adapt;
        self.adapt = true;
        f(self);
        self.adapt = adapt;
        self.bits - before
    }
}

impl BinSink for BitCostModel {
    #[inline]
    fn encode_decision(&mut self, ctx_table: usize, ctx_idx: usize, bin: u8) {
        let var = self.ctx[ctx_table][ctx_idx];
        self.bits += bin_cost(var, bin);
        if self.adapt {
            self.ctx[ctx_table][ctx_idx] = next_state(var, bin);
        }
    }

    #[inline]
    fn encode_bypass(&mut self, _bin: u8) {
        self.bits += 1.0;
    }

    fn encode_terminate(&mut self, _terminate: bool) {
        // §9.3.4.3.5: the terminate bin narrows the range by one unit
        // — a cost of −log2( 1 − 1/ivlCurrRange ) ≈ 0 bits, and it is
        // emitted once per slice regardless of any decision.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacEncoder;

    /// The integer table agrees with `f64::log2` to within its Q16
    /// resolution at every state.
    #[test]
    fn cost_table_matches_log2() {
        for s in 1..=512u32 {
            let want = -(f64::from(s) / 512.0).log2();
            let got = f64::from(COST_Q16[s as usize]) / 65536.0;
            assert!(
                (want - got).abs() < 2.0 / 65536.0,
                "state {s}: table {got} vs log2 {want}"
            );
        }
        assert_eq!(COST_Q16[512], 0);
        assert_eq!(COST_Q16[256], 1 << 16); // p = 1/2 → exactly one bit
        assert_eq!(COST_Q16[128], 2 << 16);
    }

    /// The equiprobable default state charges one bit either way; a
    /// skewed state charges the MPS less than the LPS.
    #[test]
    fn costs_follow_probabilities() {
        let d = ContextVar::default();
        assert_eq!(bin_cost(d, 0), 1.0);
        assert_eq!(bin_cost(d, 1), 1.0);
        let skewed = ContextVar {
            val_state: 32,
            val_mps: 1,
        };
        assert!(bin_cost(skewed, 1) < 0.1);
        assert!(bin_cost(skewed, 0) > 3.9);
    }

    /// The model's accumulated cost tracks the real coder's output
    /// length on a long adaptive bin string (within the arithmetic
    /// coder's few bits of flush overhead).
    #[test]
    fn model_tracks_real_codeword_length() {
        let mut seed = 0x1234_5678u32;
        let mut bins = Vec::new();
        for _ in 0..4000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            // ~85 % zeros so the context skews.
            bins.push(u8::from((seed >> 24) % 100 < 15));
        }
        let mut enc = CabacEncoder::new();
        enc.init_main_profile(InitType::I, 30);
        let mut model = BitCostModel::new();
        model.init_main_profile(InitType::I, 30);
        let cost = model.commit(|m| {
            for (i, &b) in bins.iter().enumerate() {
                let ctx = i % 3;
                m.encode_decision(MainCtxTable::CbfLuma.as_usize(), ctx, b);
                CabacEncoder::encode_decision(&mut enc, MainCtxTable::CbfLuma.as_usize(), ctx, b);
            }
        });
        enc.encode_terminate(true);
        let real_bits = enc.finish().len() as f64 * 8.0;
        assert!(
            (cost - real_bits).abs() < 40.0,
            "model {cost} bits vs coder {real_bits} bits"
        );
        assert!(
            cost < 0.8 * bins.len() as f64,
            "skewed source must beat 1 bit/bin: {cost}"
        );
    }

    /// `measure` leaves the state untouched; `commit` advances it.
    #[test]
    fn measure_is_side_effect_free() {
        let mut model = BitCostModel::new();
        let c1 = model.measure(|m| {
            for _ in 0..50 {
                m.encode_decision(0, 0, 1);
            }
        });
        let c2 = model.measure(|m| {
            for _ in 0..50 {
                m.encode_decision(0, 0, 1);
            }
        });
        assert_eq!(c1, c2);
        assert_eq!(c1, 50.0);
        assert_eq!(model.bits(), 0.0);
        let c3 = model.commit(|m| {
            for _ in 0..50 {
                m.encode_decision(0, 0, 1);
            }
        });
        assert!(c3 < c1, "adaptation must make a run of MPS cheaper: {c3}");
        assert_eq!(model.bits(), c3);
        assert!(model.context(0, 0).val_state < 256);
    }
}
