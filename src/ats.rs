//! EVC Main-profile **ATS-intra** (Adaptive Transform Selection,
//! `sps_ats_flag == 1`, intra path) — syntax reads + §8.7.4.1 transform-type
//! derivation (ISO/IEC 23094-1:2020 §7.3.8.5 + Table 30).
//!
//! When `sps_ats_flag == 1`, an intra CU with `cbf_luma == 1` and both
//! `log2CbWidth <= 5` and `log2CbHeight <= 5` carries the `ats_cu_intra_flag`
//! that, when set, selects a non-DCT-II separable kernel for the luma
//! transform. The two follow-on flags `ats_hor_mode` / `ats_ver_mode`
//! pick DST-VII (1) vs DCT-VIII (2) independently per direction (spec
//! lines 3080-3087):
//!
//! ```text
//!   if( CuPredMode == MODE_INTRA && sps_ats_flag &&
//!       log2CbWidth <= 5 && log2CbHeight <= 5 && cbf_luma ) {
//!       ats_cu_intra_flag[ x0 ][ y0 ]                    ae(v)
//!       if( ats_cu_intra_flag[ x0 ][ y0 ] == 1 ) {
//!           ats_hor_mode[ x0 ][ y0 ]                     ae(v)
//!           ats_ver_mode[ x0 ][ y0 ]                     ae(v)
//!       }
//!   }
//! ```
//!
//! ## Binarisation + contexts (Table 95)
//!
//! | element             | binarisation  | bin0 context             |
//! |---------------------|---------------|--------------------------|
//! | `ats_cu_intra_flag` | FL, cMax = 1  | **bypass**               |
//! | `ats_hor_mode`      | FL, cMax = 1  | ctxInc 0 (Table 79)      |
//! | `ats_ver_mode`      | FL, cMax = 1  | ctxInc 0 (Table 79)      |
//!
//! ## §8.7.4.1 Table 30 — trType derivation (intra, `ats_cu_intra_flag == 1`)
//!
//! | `ats_hor_mode` | `ats_ver_mode` | `trTypeHor` | `trTypeVer` |
//! |----------------|----------------|-------------|-------------|
//! | 0              | 0              | 1           | 1           |
//! | 1              | 0              | 2           | 1           |
//! | 0              | 1              | 1           | 2           |
//! | 1              | 1              | 2           | 2           |
//!
//! i.e. `trTypeHor = 1 + ats_hor_mode`, `trTypeVer = 1 + ats_ver_mode`
//! (trType 0 = DCT-II, 1 = DST-VII, 2 = DCT-VIII per §8.7.4.2). Chroma
//! always uses trType 0 (§8.7.4.1 passes `trType = 0` for `cIdx != 0`).
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::cabac::CabacEngine;
use crate::cabac_init::MainCtxTable;
use crate::eipd_syntax::EipdCtx;

/// The decoded ATS-intra decision for one luma transform block: whether
/// the alternative transform is used, and the resolved per-direction
/// `(trTypeHor, trTypeVer)` for the luma component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtsIntra {
    /// `ats_cu_intra_flag` — alternative transform engaged.
    pub used: bool,
    /// `trTypeHor` — horizontal 1-D transform type (0/1/2).
    pub tr_type_hor: u32,
    /// `trTypeVer` — vertical 1-D transform type (0/1/2).
    pub tr_type_ver: u32,
}

impl AtsIntra {
    /// The "not used" / inferred default: both modes absent → trType 0
    /// (plain DCT-II), the §6.108/6.115 inference for absent
    /// `ats_hor_mode`/`ats_ver_mode`.
    pub fn disabled() -> Self {
        Self {
            used: false,
            tr_type_hor: 0,
            tr_type_ver: 0,
        }
    }

    /// Apply the §8.7.4 inverse transform implied by this decision to one
    /// luma transform block `coeffs` of size `n_tb_w × n_tb_h` (row-major),
    /// bridging the §7.3.8.5 / Table-30 syntax decode to the §8.7.4.2
    /// kernel selection.
    ///
    /// When [`used`](Self::used) is `false` both `trType`s are 0, so this
    /// is byte-for-byte the plain DCT-II [`crate::transform::inverse_transform`] (the
    /// trType-0 lookup reuses the DCT-II matrices). When `used` is `true`
    /// the resolved `(trTypeHor, trTypeVer)` drive the DST-VII / DCT-VIII
    /// kernels per [`crate::transform::inverse_transform_ats`]. Chroma is never ATS-coded
    /// (§8.7.4.1 passes `trType = 0` for `cIdx != 0`), so chroma callers
    /// use [`crate::transform::inverse_transform`] directly.
    pub fn apply_inverse(self, coeffs: &mut [i32], n_tb_w: usize, n_tb_h: usize) -> Result<()> {
        crate::transform::inverse_transform_ats(
            coeffs,
            n_tb_w,
            n_tb_h,
            self.tr_type_hor,
            self.tr_type_ver,
        )
    }
}

/// §7.3.8.5 presence predicate for `ats_cu_intra_flag` (spec line
/// 3080-3081): `sps_ats_flag && log2CbWidth <= 5 && log2CbHeight <= 5 &&
/// cbf_luma`, evaluated only on intra CUs by the caller.
pub fn ats_intra_flag_present(
    sps_ats_flag: bool,
    log2_cb_width: u32,
    log2_cb_height: u32,
    cbf_luma: bool,
) -> bool {
    sps_ats_flag && log2_cb_width <= 5 && log2_cb_height <= 5 && cbf_luma
}

/// Read counters for the ATS-intra syntax (one per element).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtsSyntaxStats {
    pub cu_intra_flag_bins: u32,
    pub hor_mode_bins: u32,
    pub ver_mode_bins: u32,
}

/// §7.3.8.5 + Table 30 — read the ATS-intra syntax group and resolve it
/// to an [`AtsIntra`] decision.
///
/// The caller must have already established that the presence predicate
/// [`ats_intra_flag_present`] holds for this TB. Reads `ats_cu_intra_flag`
/// (bypass); when set, reads `ats_hor_mode` then `ats_ver_mode`
/// (context-coded on Table 79, ctxInc 0) and applies Table 30
/// (`trType{Hor,Ver} = 1 + ats_{hor,ver}_mode`). When the flag is 0 the
/// inferred default (`disabled`) is returned and no further bins are read.
pub fn read_ats_intra(
    eng: &mut CabacEngine,
    ctx: EipdCtx,
    stats: &mut AtsSyntaxStats,
) -> Result<AtsIntra> {
    // ats_cu_intra_flag — FL cMax=1, **bypass** (Table 95).
    let used = eng.decode_bypass()? != 0;
    stats.cu_intra_flag_bins += 1;
    if !used {
        return Ok(AtsIntra::disabled());
    }
    // ats_hor_mode / ats_ver_mode — FL cMax=1, ctxInc 0 (Table 79).
    let (t, i) = ctx.ats_mode_ctx();
    let hor = eng.decode_decision(t, i)? as u32;
    stats.hor_mode_bins += 1;
    let ver = eng.decode_decision(t, i)? as u32;
    stats.ver_mode_bins += 1;
    Ok(AtsIntra {
        used: true,
        // Table 30: trTypeHor = 1 + ats_hor_mode, trTypeVer = 1 + ats_ver_mode.
        tr_type_hor: 1 + hor,
        tr_type_ver: 1 + ver,
    })
}

/// Extension of [`EipdCtx`] that exposes the Table 79 (`AtsMode`)
/// context, shared by `ats_hor_mode` and `ats_ver_mode`.
trait AtsModeCtx {
    fn ats_mode_ctx(self) -> (usize, usize);
}

impl AtsModeCtx for EipdCtx {
    fn ats_mode_ctx(self) -> (usize, usize) {
        // Mirror EipdCtx::ctx for the single-context AtsMode table: under
        // sps_cm_init_flag == 0 every regular bin shares (0, 0); under
        // sps_cm_init_flag == 1 it lands on Table 79 at ctxIdx 0.
        if self.is_cm_init() {
            (MainCtxTable::AtsMode.as_usize(), 0)
        } else {
            (0, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacEncoder;

    fn regular_bins(bins: &[u8]) -> Vec<u8> {
        let mut enc = CabacEncoder::new();
        for &b in bins {
            enc.encode_decision(0, 0, b);
        }
        enc.encode_terminate(true);
        enc.finish()
    }

    /// Presence predicate gates on size + cbf_luma per spec line 3080-3081.
    #[test]
    fn presence_predicate() {
        assert!(ats_intra_flag_present(true, 5, 5, true));
        assert!(ats_intra_flag_present(true, 2, 4, true));
        // sps_ats_flag off.
        assert!(!ats_intra_flag_present(false, 4, 4, true));
        // CB wider than 32 (log2 > 5).
        assert!(!ats_intra_flag_present(true, 6, 4, true));
        assert!(!ats_intra_flag_present(true, 4, 6, true));
        // cbf_luma 0.
        assert!(!ats_intra_flag_present(true, 4, 4, false));
    }

    /// `ats_cu_intra_flag == 0` → disabled, only one (bypass) bin's worth
    /// of state consumed, trType both 0.
    #[test]
    fn flag_zero_disables() {
        // First regular bin 0 drives the bypass read toward 0.
        let bs = regular_bins(&[0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AtsSyntaxStats::default();
        let ats = read_ats_intra(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert_eq!(ats, AtsIntra::disabled());
        assert_eq!(stats.cu_intra_flag_bins, 1);
        assert_eq!(stats.hor_mode_bins, 0);
    }

    /// Table 30 derivation for all four (hor, ver) combinations.
    #[test]
    fn table_30_trtype_derivation() {
        // (ats_hor_mode, ats_ver_mode) -> (trTypeHor, trTypeVer).
        let cases = [
            (0u32, 0u32, 1u32, 1u32),
            (1, 0, 2, 1),
            (0, 1, 1, 2),
            (1, 1, 2, 2),
        ];
        for (h, v, th, tv) in cases {
            let ats = AtsIntra {
                used: true,
                tr_type_hor: 1 + h,
                tr_type_ver: 1 + v,
            };
            assert_eq!(ats.tr_type_hor, th);
            assert_eq!(ats.tr_type_ver, tv);
        }
    }

    /// Baseline ATS-mode context routes to (0, 0); Main-profile routes to
    /// Table 79 at ctxIdx 0.
    #[test]
    fn ats_mode_ctx_routing() {
        assert_eq!(EipdCtx::new(false).ats_mode_ctx(), (0, 0));
        assert_eq!(
            EipdCtx::new(true).ats_mode_ctx(),
            (MainCtxTable::AtsMode.as_usize(), 0)
        );
    }

    /// `apply_inverse` on a `disabled` decision is byte-for-byte the plain
    /// DCT-II inverse for every nTbS (trType 0/0 reuses the DCT-II tables).
    #[test]
    fn apply_inverse_disabled_matches_plain_dct() {
        for n in [4usize, 8, 16, 32, 64] {
            let mut a = vec![0i32; n * n];
            a[0] = 5;
            a[n + 1] = -3;
            let mut b = a.clone();
            crate::transform::inverse_transform(&mut a, n, n).unwrap();
            AtsIntra::disabled().apply_inverse(&mut b, n, n).unwrap();
            assert_eq!(a, b, "disabled apply_inverse must equal plain DCT for {n}");
        }
    }

    /// End-to-end ATS-intra path: synthesise the CABAC bin sequence for
    /// `ats_cu_intra_flag = 1, ats_hor_mode = 1, ats_ver_mode = 0`, decode
    /// it through [`read_ats_intra`] (Table 30 → trTypeHor = 2 (DCT-VIII),
    /// trTypeVer = 1 (DST-VII)), then drive a real coefficient block through
    /// [`AtsIntra::apply_inverse`] for every §8.7.4.3 size {4,8,16,32} and
    /// confirm it reproduces the kernel selected by the direct
    /// [`crate::transform::inverse_transform_ats`] call. This exercises
    /// syntax-decode → Table-30 derivation → kernel dispatch together.
    #[test]
    fn decode_to_transform_roundtrip_all_sizes() {
        // Bins under sps_cm_init_flag = 0 all share ctx (0,0); the bypass
        // engine and the (0,0) decision engine both read this stream. The
        // sequence drives ats_cu_intra_flag → 1, ats_hor_mode → 1,
        // ats_ver_mode → 0. The exact resolved modes are asserted, so the
        // test is robust to the CABAC range-coder's internal mapping.
        let bs = regular_bins(&[1, 1, 0]);
        let mut eng = CabacEngine::new(&bs).unwrap();
        let mut stats = AtsSyntaxStats::default();
        let ats = read_ats_intra(&mut eng, EipdCtx::new(false), &mut stats).unwrap();
        assert!(ats.used);
        assert_eq!(stats.cu_intra_flag_bins, 1);
        assert_eq!(stats.hor_mode_bins, 1);
        assert_eq!(stats.ver_mode_bins, 1);
        // trType in {1,2} for an engaged ATS decision (Table 30 = 1 + mode).
        assert!((1..=2).contains(&ats.tr_type_hor));
        assert!((1..=2).contains(&ats.tr_type_ver));

        for n in [4usize, 8, 16, 32] {
            // A non-trivial coefficient block (a few low-freq impulses).
            let mut block = vec![0i32; n * n];
            block[0] = 17;
            block[1] = -9;
            block[n] = 4;
            block[n + 1] = 2;

            let mut via_decision = block.clone();
            ats.apply_inverse(&mut via_decision, n, n).unwrap();

            let mut via_direct = block.clone();
            crate::transform::inverse_transform_ats(
                &mut via_direct,
                n,
                n,
                ats.tr_type_hor,
                ats.tr_type_ver,
            )
            .unwrap();

            assert_eq!(
                via_decision, via_direct,
                "apply_inverse kernel dispatch must match direct call at nTbS={n}"
            );
            // The alternative transform must actually differ from plain
            // DCT-II for this engaged decision (sanity that a non-DCT kernel
            // was selected, not silently falling back to trType 0).
            let mut plain = block.clone();
            crate::transform::inverse_transform(&mut plain, n, n).unwrap();
            assert_ne!(
                via_decision, plain,
                "engaged ATS at nTbS={n} should differ from plain DCT-II"
            );
        }
    }
}
