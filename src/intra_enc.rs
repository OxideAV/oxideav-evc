//! Encoder-side **EIPD** intra prediction (`sps_eipd_flag == 1`,
//! §7.3.8.4 / §8.4.2 / §8.4.3 / §8.4.4): the mode search shared by the
//! I and P/B slice encoders, and the write-side duals of the decoder's
//! `eipd_syntax` reads.
//!
//! ## Syntax duals
//!
//! The luma mode is signalled through the three §8.4.2 lists the
//! decoder derives from the L / A / R neighbour modes
//! (`candModeList[2]`, `extCandModeList[8]`, `remModeList[33]`):
//!
//! ```text
//!   intra_luma_pred_mpm_flag      FL cMax 1, Table 63 ctxInc 0
//!   ├─ 1: intra_luma_pred_mpm_idx FL cMax 1, Table 64 ctxInc 0
//!   └─ 0: intra_luma_pred_pims_flag   bypass
//!         ├─ 1: intra_luma_pred_pims_idx  FL cMax 7, 3 bypass bins
//!         └─ 0: intra_luma_pred_rem_mode  TB cMax 22, bypass
//!   intra_chroma_pred_mode        Table 93: 1 / 00 / 010 / 0110 / 0111
//!                                 (bin 0 Table 65 ctxInc 0, rest bypass)
//! ```
//!
//! [`selector_for`] inverts the decoder's `EipdModeLists::select`:
//! a mode found in `candModeList` is sent as MPM, else in
//! `extCandModeList` as PIMS, else as the `remModeList` index minus 10
//! (the decoder reads `remModeList[ rem_mode + 10 ]`; the first ten
//! entries are the MPM/PIMS modes, which are never sent that way).
//!
//! ## Mode search
//!
//! Full RD over all 33 modes would cost ~7× the Baseline five; the
//! search runs in two stages: every mode is predicted and ranked by
//! `SAD + λ_SAD · bits( mode syntax )` (`λ_SAD = √λ`, the usual
//! first-order relation between the L1 and L2 Lagrangians), and the
//! best [`RD_CANDIDATES`] plus both MPM entries go through the exact RD
//! (RDOQ residual, true reconstruction SSE, exact bins). Chroma then
//! tries all five `intra_chroma_pred_mode` values (DM plus the four
//! Table-16 modes after the §8.4.3 collision skip) under the same RD.

use crate::cabac::BinSink;
use crate::cabac_init::{CtxSel, MainCtxTable};
use crate::deblock::{CuPredMode, SideInfoGrid};
use crate::eipd::{predict_eipd, AvailLr};
use crate::eipd_mode::{derive_mode_lists, EipdModeLists, ModeSelector, NeighbourMode};
use crate::picture::YuvPicture;

/// Number of EIPD luma modes (Table 15: 0..=32).
pub const EIPD_MODES: i32 = 33;

/// SAD-ranked modes that go through the full RD (besides the MPMs).
pub const RD_CANDIDATES: usize = 3;

/// One chroma RD choice: `(intra_chroma_pred_mode, IntraPredModeC,
/// levels_cb, cbf_cb, res_cb, levels_cr, cbf_cr, res_cr)`.
pub type ChromaChoice = (i32, i32, Vec<i32>, bool, Vec<i32>, Vec<i32>, bool, Vec<i32>);

/// The intra-mode syntax of one CU as the encoder decided it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntraSel {
    /// `sps_eipd_flag == 0`: the Baseline `intra_pred_mode` index
    /// (Table 13 order DC / HOR / VER / UL / UR); chroma is DM.
    Baseline(usize),
    /// `sps_eipd_flag == 1`: `IntraPredModeY` (0..=32) and the raw
    /// `intra_chroma_pred_mode` (0 = DM, 1..=4 per §8.4.3).
    Eipd { mode_y: i32, chroma_raw: i32 },
}

impl IntraSel {
    /// The value the decoder stamps as `intra_luma_mode` (the Baseline
    /// index, or the EIPD mode) — what later CUs' §8.4.2 neighbour
    /// probes read.
    pub fn stamp_value(self) -> u8 {
        match self {
            IntraSel::Baseline(m) => m as u8,
            IntraSel::Eipd { mode_y, .. } => mode_y.clamp(0, 32) as u8,
        }
    }
}

/// §8.4.2 step 1-2 — one neighbouring location's candidate mode over
/// the encoder's decode-order grid (the dual of the decoder's probe:
/// inside the picture, already stamped, `MODE_INTRA`).
fn neighbour_mode(grid: &SideInfoGrid, pic_w: u32, pic_h: u32, x: i64, y: i64) -> NeighbourMode {
    if x < 0 || y < 0 || x >= i64::from(pic_w) || y >= i64::from(pic_h) {
        return NeighbourMode::invalid();
    }
    let xc = (x as u32 >> 2) as usize;
    let yc = (y as u32 >> 2) as usize;
    if xc >= grid.w_cells || yc >= grid.h_cells {
        return NeighbourMode::invalid();
    }
    let cell = grid.at(xc, yc);
    if cell.cu_log2_w == 0 || cell.pred_mode != CuPredMode::Intra {
        return NeighbourMode::invalid();
    }
    NeighbourMode::valid(i32::from(cell.intra_luma_mode))
}

/// The §8.4.2 mode lists of the CU at `(x0, y0)` from its
/// `(xCb − 1, yCb)`, `(xCb, yCb − 1)`, `(xCb + nCbW, yCb)` neighbours.
pub fn mode_lists(
    grid: &SideInfoGrid,
    pic_w: u32,
    pic_h: u32,
    x0: u32,
    y0: u32,
    log2_w: u32,
) -> EipdModeLists {
    let w = 1i64 << log2_w;
    let a = neighbour_mode(grid, pic_w, pic_h, i64::from(x0) - 1, i64::from(y0));
    let b = neighbour_mode(grid, pic_w, pic_h, i64::from(x0), i64::from(y0) - 1);
    let c = neighbour_mode(grid, pic_w, pic_h, i64::from(x0) + w, i64::from(y0));
    derive_mode_lists(a, b, c)
}

/// The selector that makes the decoder's `EipdModeLists::select`
/// return `mode` — the write-side inverse of §8.4.2 step 6.
pub fn selector_for(lists: &EipdModeLists, mode: i32) -> ModeSelector {
    if let Some(i) = lists.cand_mode_list.iter().position(|&m| m == mode) {
        return ModeSelector::Mpm(i);
    }
    if let Some(i) = lists.ext_cand_mode_list.iter().position(|&m| m == mode) {
        return ModeSelector::Pims(i);
    }
    let i = lists.rem_mode_list[10..]
        .iter()
        .position(|&m| m == mode)
        .expect("every EIPD mode is in one of the three lists");
    ModeSelector::Rem(i)
}

/// Write the luma mode group for `selector`.
pub fn emit_luma_selector<S: BinSink>(enc: &mut S, sel: CtxSel, selector: ModeSelector) {
    let (t, i) = sel.ctx(MainCtxTable::IntraLumaPredMpmFlag, 0);
    match selector {
        ModeSelector::Mpm(idx) => {
            enc.encode_decision(t, i, 1);
            let (t, i) = sel.ctx(MainCtxTable::IntraLumaPredMpmIdx, 0);
            enc.encode_decision(t, i, idx as u8);
        }
        ModeSelector::Pims(idx) => {
            enc.encode_decision(t, i, 0);
            enc.encode_bypass(1);
            // FL cMax 7: three bypass bins, MSB first.
            for b in (0..3).rev() {
                enc.encode_bypass(((idx >> b) & 1) as u8);
            }
        }
        ModeSelector::Rem(rem) => {
            enc.encode_decision(t, i, 0);
            enc.encode_bypass(0);
            emit_tb_bypass(enc, rem as u32, 22);
        }
    }
}

/// TB (§9.3.3.6) bypass write — the dual of `CabacEngine::decode_tb_bypass`.
pub fn emit_tb_bypass<S: BinSink>(enc: &mut S, value: u32, c_max: u32) {
    if c_max == 0 {
        return;
    }
    let n = c_max + 1;
    let k = 31 - n.leading_zeros();
    let u = (1u32 << (k + 1)) - n;
    let (codeword, bits) = if value < u {
        (value, k)
    } else {
        (value + u, k + 1)
    };
    for b in (0..bits).rev() {
        enc.encode_bypass(((codeword >> b) & 1) as u8);
    }
}

/// Write `intra_chroma_pred_mode` (Table 93).
pub fn emit_chroma_pred_mode<S: BinSink>(enc: &mut S, sel: CtxSel, raw: i32) {
    let (t, i) = sel.ctx(MainCtxTable::IntraChromaPredMode, 0);
    if raw == 0 {
        enc.encode_decision(t, i, 1);
        return;
    }
    enc.encode_decision(t, i, 0);
    match raw {
        1 => enc.encode_bypass(0),
        2 => {
            enc.encode_bypass(1);
            enc.encode_bypass(0);
        }
        3 => {
            enc.encode_bypass(1);
            enc.encode_bypass(1);
            enc.encode_bypass(0);
        }
        _ => {
            enc.encode_bypass(1);
            enc.encode_bypass(1);
            enc.encode_bypass(1);
        }
    }
}

/// The EIPD prediction of component `c_idx` of the CU at luma
/// `(x_luma, y_luma)` from the current reconstruction — the decoder's
/// §8.4.4 kernels over its §8.4.4.1 reference construction, under the
/// single-tile, `sps_suco_flag == 0` availability (left iff `x > 0`,
/// never right). Row-major `n_cb_w × n_cb_h` in the component domain.
pub fn predict(
    recon: &YuvPicture,
    x_luma: u32,
    y_luma: u32,
    log2_w: u32,
    log2_h: u32,
    c_idx: u32,
    mode: i32,
) -> Vec<i32> {
    let (x, y, w, h) = if c_idx == 0 {
        (x_luma, y_luma, 1usize << log2_w, 1usize << log2_h)
    } else {
        (
            x_luma / 2,
            y_luma / 2,
            1usize << (log2_w - 1),
            1usize << (log2_h - 1),
        )
    };
    let refs = recon.fetch_eipd_refs(x, y, w, h, c_idx, false, false);
    let avail_lr = if x > 0 { AvailLr::Lr10 } else { AvailLr::Lr00 };
    let mut pred = vec![0i32; w * h];
    predict_eipd(mode, &refs, w, h, recon.bit_depth, avail_lr, &mut pred);
    pred
}

/// The SAD-ranked luma candidate set for the full RD: the best
/// [`RD_CANDIDATES`] modes under `SAD + √λ · bits` plus both MPMs,
/// deduplicated, in ranking order.
#[allow(clippy::too_many_arguments)]
pub fn luma_candidates(
    recon: &YuvPicture,
    src_y: &[i32],
    x0: u32,
    y0: u32,
    log2_w: u32,
    log2_h: u32,
    lists: &EipdModeLists,
    lambda: f64,
    mut mode_bits: impl FnMut(ModeSelector) -> f64,
) -> Vec<i32> {
    let lambda_sad = lambda.sqrt();
    let mut ranked: Vec<(f64, i32)> = (0..EIPD_MODES)
        .map(|mode| {
            let pred = predict(recon, x0, y0, log2_w, log2_h, 0, mode);
            let sad: f64 = src_y
                .iter()
                .zip(pred.iter())
                .map(|(&s, &p)| f64::from((s - p).abs()))
                .sum();
            (
                sad + lambda_sad * mode_bits(selector_for(lists, mode)),
                mode,
            )
        })
        .collect();
    // Deterministic ordering: cost, then mode number.
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let mut out: Vec<i32> = ranked.iter().take(RD_CANDIDATES).map(|r| r.1).collect();
    for &m in &lists.cand_mode_list {
        if !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin_cost::BitCostModel;
    use crate::cabac::{CabacEncoder, CabacEngine, InitType};
    use crate::cabac_init::init_main_profile_contexts;
    use crate::eipd_syntax::{
        read_intra_chroma_pred_mode, read_luma_mode_selector, EipdCtx, EipdSyntaxStats,
    };

    /// Every mode of every list shape maps to a selector the decoder
    /// resolves back to that mode.
    #[test]
    fn selector_round_trips_through_the_lists() {
        let shapes = [
            (
                NeighbourMode::invalid(),
                NeighbourMode::invalid(),
                NeighbourMode::invalid(),
            ),
            (
                NeighbourMode::valid(12),
                NeighbourMode::valid(24),
                NeighbourMode::invalid(),
            ),
            (
                NeighbourMode::valid(5),
                NeighbourMode::valid(5),
                NeighbourMode::valid(7),
            ),
            (
                NeighbourMode::valid(0),
                NeighbourMode::invalid(),
                NeighbourMode::valid(31),
            ),
            (
                NeighbourMode::valid(2),
                NeighbourMode::valid(1),
                NeighbourMode::valid(0),
            ),
        ];
        for (a, b, c) in shapes {
            let lists = derive_mode_lists(a, b, c);
            for mode in 0..EIPD_MODES {
                assert_eq!(
                    lists.select(selector_for(&lists, mode)),
                    mode,
                    "{lists:?} mode {mode}"
                );
            }
        }
    }

    /// The luma group and the chroma mode written here read back through
    /// the decoder's `eipd_syntax` on both entropy shapes.
    #[test]
    fn syntax_duals_read_back() {
        for &cm in &[false, true] {
            for &init in &[InitType::I, InitType::Pb] {
                let sel = CtxSel::new(cm, init);
                let mut enc = CabacEncoder::new();
                if cm {
                    enc.init_main_profile(init, 25);
                }
                let selectors = [
                    ModeSelector::Mpm(0),
                    ModeSelector::Mpm(1),
                    ModeSelector::Pims(0),
                    ModeSelector::Pims(5),
                    ModeSelector::Pims(7),
                    ModeSelector::Rem(0),
                    ModeSelector::Rem(8),
                    ModeSelector::Rem(9),
                    ModeSelector::Rem(15),
                    ModeSelector::Rem(22),
                ];
                for (k, &s) in selectors.iter().enumerate() {
                    emit_luma_selector(&mut enc, sel, s);
                    emit_chroma_pred_mode(&mut enc, sel, (k % 5) as i32);
                }
                enc.encode_terminate(true);
                let bytes = enc.finish();
                let mut eng = CabacEngine::new(&bytes).unwrap();
                if cm {
                    init_main_profile_contexts(&mut eng, init, 25).unwrap();
                }
                let ctx = EipdCtx::for_slice(cm, init);
                let mut stats = EipdSyntaxStats::default();
                for (k, &s) in selectors.iter().enumerate() {
                    assert_eq!(
                        read_luma_mode_selector(&mut eng, ctx, &mut stats).unwrap(),
                        s,
                        "cm{cm} {init:?}"
                    );
                    assert_eq!(
                        read_intra_chroma_pred_mode(&mut eng, ctx, &mut stats).unwrap(),
                        (k % 5) as i32
                    );
                }
            }
        }
    }

    /// The TB writer is the exact inverse of the decoder's TB reader
    /// across the whole `cMax = 22` range.
    #[test]
    fn tb_bypass_dual() {
        let mut enc = CabacEncoder::new();
        for v in 0..=22u32 {
            emit_tb_bypass(&mut enc, v, 22);
        }
        enc.encode_terminate(true);
        let bytes = enc.finish();
        let mut eng = CabacEngine::new(&bytes).unwrap();
        for v in 0..=22u32 {
            assert_eq!(eng.decode_tb_bypass(22).unwrap(), v);
        }
    }

    /// The candidate set always carries both MPMs and never repeats.
    #[test]
    fn candidates_include_mpms_without_repeats() {
        let recon = YuvPicture::new(16, 16, 1, 8).unwrap();
        let src: Vec<i32> = (0..64).map(|i| 100 + (i % 8) * 3).collect();
        let lists = derive_mode_lists(
            NeighbourMode::valid(12),
            NeighbourMode::valid(6),
            NeighbourMode::invalid(),
        );
        let model = BitCostModel::new();
        let sel = CtxSel::baseline();
        let cands = luma_candidates(&recon, &src, 8, 8, 3, 3, &lists, 4.0, |s| {
            let mut m = model.clone();
            m.measure(|m| emit_luma_selector(m, sel, s))
        });
        assert!(cands.len() >= RD_CANDIDATES && cands.len() <= RD_CANDIDATES + 2);
        for &m in &lists.cand_mode_list {
            assert!(cands.contains(&m), "{cands:?} lacks MPM {m}");
        }
        let mut sorted = cands.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), cands.len());
    }
}
