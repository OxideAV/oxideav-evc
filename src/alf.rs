//! EVC Adaptive Loop Filter (ISO/IEC 23094-1 §8.9 / §7.3.5).
//!
//! The ALF is a post-deblocking in-loop filter that adaptively selects, for
//! each CTU, one of up to 25 luma filter sets and up to 4 chroma filter sets
//! signalled in the ALF APS (`aps_params_type == 0`).  The per-CTU
//! `alf_ctb_flag` (CABAC-coded in `coding_tree_unit()`) can disable the
//! filter for individual CTUs; currently the round-11 decoder reads the
//! `alf_ctb_flag` from the slice header as an enable/disable switch and
//! applies whichever filter the APS selects uniformly to every luma / chroma
//! CTU where `alf_ctb_flag` is true.
//!
//! ## Filter shapes (§8.9.4)
//!
//! The spec defines two diamond-shaped filter footprints:
//!
//! * **Luma** — 7×7 diamond with 13 unique tap positions (the filter matrix
//!   has 25 positions but the diamond pattern reduces to 13 symmetric pairs
//!   and a centre coefficient; the spec lists 13 coefficients per luma filter
//!   set: `c[0]..c[12]`, with c[12] as the DC offset — eq. 1263-1264).
//! * **Chroma** — 5×5 diamond with 7 unique tap positions (7 coefficients per
//!   chroma filter; eq. 1290-1292).
//!
//! ## APS payload (§7.3.5)
//!
//! The `alf_data()` syntax encodes:
//! * `alf_luma_filter_signal_flag` / `alf_chroma_filter_signal_flag`
//! * `new_filter_flag[0]` (luma) / `new_filter_flag[1]` (chroma)
//! * `alf_luma_num_filters_signalled_minus1`: 0..=24 → 1..=25 filter sets
//! * Per filter: `alf_luma_coeff_flag[i]` (1 bit) then, when set, 12 × 6-bit
//!   `alf_luma_coeff_abs` fields and 12 × 1-bit `alf_luma_coeff_sign` fields
//!   (the 13th coefficient is the DC offset derived as
//!   `128 − Σ|c[0..11]| × 2`).
//! * `alf_chroma_num_alt_filters_minus1`: 0..=3 → 1..=4 chroma alternates
//! * Per chroma filter: 6 × 6-bit abs + 6 × 1-bit sign (the 7th is derived).
//!
//! ## Round-11 scope
//!
//! * Full APS payload parsing (replaces the round-1 raw-byte capture).
//! * Luma + chroma filter tap-pass applied uniformly across the full picture
//!   after deblocking when `sps_alf_flag == 1 && slice_alf_enabled_flag`.
//! * Per-CTU `alf_ctb_flag` masking (round 113): `coding_tree_unit()`
//!   (§7.3.8.2) decodes the per-CTU applicability map into an [`AlfCtbMap`];
//!   [`apply_alf_with_map`] / [`apply_alf_luma_masked`] then filter only the
//!   luma coding tree blocks whose `alf_ctb_flag` is 1 per the §8.9 loop
//!   (lines 18059-18074), with the `blkWidth` / `blkHeight` picture-edge
//!   clamp. The whole-plane [`apply_alf`] entry is retained for the
//!   minimal-header / no-map path.
//! * The 25-filter-set `alf_luma_filter_idx` per-CTU selection (§8.9.6) is
//!   deferred — this round always applies filter set 0.
//! * The §8.8.4.3 **ALF transpose + classification filter-index derivation**
//!   (round 117): [`derive_alf_classification`] computes, per luma sample of
//!   a coding tree block, the gradient-classification `filtIdx` (0..24, one of
//!   the 25 spec filter classes — eq. 1319/1320) and `transposeIdx` (0..3 —
//!   eq. 1317/1318) from the local horizontal / vertical / diagonal activity
//!   (eq. 1289-1316). [`transpose_luma_coeffs`] applies the eq. 1282-1285
//!   coefficient permutation a sample's `transposeIdx` selects.
//! * The §7.3.5 **spec-faithful `alf_data()` parser + §8.9.4 `AlfCoeffL`
//!   derivation** (round 120): [`parse_alf_data`] consumes the full §7.3.5
//!   syntax — `alf_luma_type_flag` + `coefPosMap` (eq. 90/91),
//!   `alf_luma_coeff_delta_idx[ ]` (u(v) per eq. just below the table),
//!   `alf_luma_fixed_filter_usage_pattern` (uek(v)) +
//!   `alf_luma_fixed_filter_usage_flag[ ]` + `alf_luma_fixed_filter_set_idx[ ]`,
//!   `alf_luma_coeff_delta_flag` + `alf_luma_coeff_delta_prediction_flag`,
//!   `alf_luma_{min,eg_order_increase}_*` (eq. 92/93/94/95 for
//!   `expGoOrderY[ ]`), per-class `alf_luma_coeff_flag[ ]`, then per-tap
//!   `alf_luma_coeff_delta_abs[ ][ ]` (uek(v) with order picked by eq.
//!   golombOrderIdxY) + `alf_luma_coeff_delta_sign_flag[ ][ ]`. Chroma path
//!   mirrors the luma uek(v) signalling per §7.3.5 second half.
//!   [`derive_alf_coeff_l`] then assembles the 25 per-class
//!   `AlfCoeffL[ filtIdx ][ 0..12 ]` arrays per eq. 96-104: starts from the
//!   fixed-filter row [`crate::alf_tables::ALF_FIX_FILT_COEFF`] (selected via
//!   [`crate::alf_tables::ALF_CLASS_TO_FILT_MAP`] when
//!   `alf_luma_fixed_filter_usage_flag[ filtIdx ] == 1`) or zero (eq. 98/99),
//!   adds the per-position delta from
//!   `filterCoefficients[ alf_luma_coeff_delta_idx[ filtIdx ] ][ coefPosMap[ j ] − 1 ]`
//!   for every j with `coefPosMap[ j ] > 0` (eq. 100), and computes
//!   position 12 (DC offset) per eq. 104. The decoder now caches the
//!   derived per-class coefficients and the per-sample
//!   [`derive_alf_classification`] selects which class's `AlfCoeffL` row
//!   applies — closing the §8.8.4.2 / §8.9.4 wiring loop. Per-CTU
//!   ALF filter-set selection (§8.9.6) and the Main-profile toolset
//!   (BTT/ADMVP/EIPD/ATS/AMVR/affine) remain as documented follow-ups.
//!
//! All clause and equation numbers refer to **ISO/IEC 23094-1:2020(E)**.

use oxideav_core::{Error, Result};

use crate::alf_tables::{ALF_CLASS_TO_FILT_MAP, ALF_FIX_FILT_COEFF};
use crate::bitreader::BitReader;
use crate::picture::YuvPicture;

// =====================================================================
// APS alf_data() parser.
// =====================================================================

/// Number of luma tap positions excluding the derived DC offset (§8.9.4.1).
/// The luma filter has 13 taps; the spec stores 12 explicit abs/sign pairs
/// and derives the 13th (DC offset) as `128 − 2 * Σ abs[0..11]`.
const ALF_LUMA_NUM_COEF: usize = 13;

/// Explicit tap coefficients stored in the APS for luma (one per filter set,
/// all 13 values including the derived DC offset stored after decode).
pub const ALF_MAX_LUMA_FILTERS: usize = 25;

/// Chroma: 7 taps total; 6 explicit + 1 derived.
const ALF_CHROMA_NUM_COEF: usize = 7;

/// Maximum number of chroma alternate filters (§7.3.5).
pub const ALF_MAX_CHROMA_ALTS: usize = 4;

/// One parsed luma filter (13 coefficients, luma-plane signed).
/// Index mapping to the 7×7 diamond follows §8.9.4.1 eq. 1265.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlfLumaFilter {
    /// c[0]..c[12]; c[12] is the DC offset term (eq. 1264).
    pub coef: [i16; ALF_LUMA_NUM_COEF],
}

impl Default for AlfLumaFilter {
    fn default() -> Self {
        // Identity filter: all taps zero, DC offset 64 (no modification).
        let mut coef = [0i16; ALF_LUMA_NUM_COEF];
        coef[12] = 64;
        Self { coef }
    }
}

/// One parsed chroma filter (7 coefficients).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlfChromaFilter {
    /// c[0]..c[6]; c[6] is the derived DC offset (eq. 1292).
    pub coef: [i16; ALF_CHROMA_NUM_COEF],
}

impl Default for AlfChromaFilter {
    fn default() -> Self {
        let mut coef = [0i16; ALF_CHROMA_NUM_COEF];
        coef[6] = 64;
        Self { coef }
    }
}

/// All filter data decoded from a single ALF APS payload.
///
/// The struct carries both the §7.3.5 parsed syntax elements (`luma_*`
/// fields) and the §8.9.4 derived per-class luma filters (`luma_filters`)
/// so callers can either consult `luma_filters[filtIdx]` directly (the
/// post-derivation form ready for §8.9.4.1 eq. 1263) or re-run
/// [`derive_alf_coeff_l`] with a different APS pool when needed.
#[derive(Clone, Debug)]
pub struct AlfData {
    /// Whether luma filters were signalled in this APS.
    pub luma_filter_signal: bool,
    /// Whether chroma filters were signalled.
    pub chroma_filter_signal: bool,
    /// `alf_luma_type_flag` (§7.3.5). When 0: `NumAlfCoefs = 7`, when 1:
    /// `NumAlfCoefs = 13`. The Baseline / Main-profile pure pixel pipeline
    /// uses the 13-tap form (eq. 1265-style 7×7 diamond) as the only one
    /// currently applied; the round-120 parser stores the flag for
    /// completeness.
    pub luma_type_flag: bool,
    /// Number of signalled luma filter sets `NumSignalledFilter` =
    /// `alf_luma_num_filters_signalled_minus1 + 1`, in 1..=25.
    pub num_signalled_luma_filters: usize,
    /// `alf_luma_coeff_delta_idx[ i ]` for `i = 0..24`, the class-to-
    /// signalled-filter mapping (eq. 100). When
    /// `num_signalled_luma_filters == 1` the spec infers every entry as 0.
    pub alf_luma_coeff_delta_idx: [u8; ALF_MAX_LUMA_FILTERS],
    /// `alf_luma_fixed_filter_usage_flag[ filtIdx ]` resolved per class
    /// (eq. 98/99 select).
    pub alf_luma_fixed_filter_usage_flag: [bool; ALF_MAX_LUMA_FILTERS],
    /// `alf_luma_fixed_filter_set_idx[ filtIdx ]` per class, in 0..=15.
    /// Read only when the usage flag is 1 (eq. 98).
    pub alf_luma_fixed_filter_set_idx: [u8; ALF_MAX_LUMA_FILTERS],
    /// Number of luma filter classes that have non-trivial output (always
    /// `NumAlfFilters = 25` for the derived form, kept for compatibility
    /// with the legacy round-11 callers that selected `luma_filters[0]`).
    pub num_luma_filters: usize,
    /// `AlfCoeffL[ filtIdx ][ 0..12 ]` for `filtIdx = 0..24` per §8.9.4
    /// eq. 96-104. Position 12 carries the eq. 104 DC offset.
    pub luma_filters: [AlfLumaFilter; ALF_MAX_LUMA_FILTERS],
    /// Number of chroma alternate filters (1..=4). EVC v1 (§7.3.5) always
    /// signals a single chroma filter; the field stays for forward
    /// compatibility with the future per-CTU `alf_ctb_chroma_alt_idx` work.
    pub num_chroma_alts: usize,
    /// Chroma alternate filters; only slots `0..num_chroma_alts` are valid.
    pub chroma_filters: [AlfChromaFilter; ALF_MAX_CHROMA_ALTS],
}

impl Default for AlfData {
    fn default() -> Self {
        Self {
            luma_filter_signal: false,
            chroma_filter_signal: false,
            luma_type_flag: true,
            num_signalled_luma_filters: 1,
            alf_luma_coeff_delta_idx: [0; ALF_MAX_LUMA_FILTERS],
            alf_luma_fixed_filter_usage_flag: [false; ALF_MAX_LUMA_FILTERS],
            alf_luma_fixed_filter_set_idx: [0; ALF_MAX_LUMA_FILTERS],
            num_luma_filters: 1,
            luma_filters: [AlfLumaFilter::default(); ALF_MAX_LUMA_FILTERS],
            num_chroma_alts: 1,
            chroma_filters: [AlfChromaFilter::default(); ALF_MAX_CHROMA_ALTS],
        }
    }
}

/// §7.3.5 `coefPosMap[ ]` for `alf_luma_type_flag == 0` (eq. 90).
/// 7-tap "small" luma filter: position 0 is the central DC tap, indices
/// 1..6 carry the six spatial taps (mapped to `filterCoefficients[*]`).
const COEF_POS_MAP_7: [usize; 13] = [0, 0, 1, 0, 0, 2, 3, 4, 0, 0, 5, 6, 7];

/// §7.3.5 `coefPosMap[ ]` for `alf_luma_type_flag == 1` (eq. 91).
/// 13-tap "large" luma filter — direct 1:1 mapping, all positions live.
const COEF_POS_MAP_13: [usize; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

/// §7.3.5 `golombOrderIdxY[ ]` (eq. 94) — selects which `expGoOrderY[ ]`
/// entry parses each of the 12 luma coefficient deltas.
const GOLOMB_ORDER_IDX_Y: [usize; 12] = [0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2];

/// §7.3.5 (chroma) `golombOrderIdxC[ ]` — only 6 entries because chroma
/// signals 6 abs deltas (the 7th is derived). The spec text just below the
/// chroma table mirrors the luma `golombOrderIdxY` shape; for 5×5 chroma
/// the index pattern is `[0, 0, 1, 0, 0, 1]` (same prefix as luma).
const GOLOMB_ORDER_IDX_C: [usize; 6] = [0, 0, 1, 0, 0, 1];

/// `LumaMaxGolombIdx` per §7.3.5 — `alf_luma_type_flag == 0` → 2, else 3.
#[inline]
fn luma_max_golomb_idx(type_flag: bool) -> usize {
    if type_flag {
        3
    } else {
        2
    }
}

/// `ChromaMaxGolombIdx` per §7.3.5 — fixed at 2.
const CHROMA_MAX_GOLOMB_IDX: usize = 2;

/// Parse an `alf_data()` payload from the given byte slice.
///
/// The payload begins at bit offset 8 within the APS RBSP (the 1-byte APS
/// header covers `adaptation_parameter_set_id` and `aps_params_type`). The
/// caller passes the APS payload bytes starting immediately after that header.
///
/// Round-120 implements the **full §7.3.5 syntax** plus the §8.9.4
/// derivation (eq. 96-104), populating `luma_filters[ filtIdx ]` directly
/// for `filtIdx = 0..24`. The flow:
///
/// ```text
/// alf_data() {
///   alf_luma_filter_signal_flag                              u(1)
///   alf_chroma_filter_signal_flag                            u(1)
///   if (alf_luma_filter_signal_flag) {
///     alf_luma_num_filters_signalled_minus1                  ue(v)   // 0..24
///     alf_luma_type_flag                                     u(1)
///     if (alf_luma_num_filters_signalled_minus1 > 0)
///       for i in 0..25
///         alf_luma_coeff_delta_idx[i]                        u(v)
///     alf_luma_fixed_filter_usage_pattern                    uek(v) k=0
///     if (pattern == 2)
///       for i in 0..25
///         alf_luma_fixed_filter_usage_flag[i]                u(1)
///     if (pattern > 0)
///       for i in 0..25
///         if (alf_luma_fixed_filter_usage_flag[i])
///           alf_luma_fixed_filter_set_idx[i]                 u(4)
///     alf_luma_coeff_delta_flag                              u(1)
///     if (!alf_luma_coeff_delta_flag && minus1 > 0)
///       alf_luma_coeff_delta_prediction_flag                 u(1)
///     alf_luma_min_eg_order_minus1                           ue(v)
///     for i in 0..LumaMaxGolombIdx
///       alf_luma_eg_order_increase_flag[i]                   u(1)
///     if (alf_luma_coeff_delta_flag)
///       for i in 0..NumSignalledFilter
///         alf_luma_coeff_flag[i]                             u(1)
///     for i in 0..NumSignalledFilter
///       if (alf_luma_coeff_flag[i])
///         for j in 0..12
///           alf_luma_coeff_delta_abs[i][j]                   uek(v)
///           if (alf_luma_coeff_delta_abs[i][j])
///             alf_luma_coeff_delta_sign_flag[i][j]           u(1)
///   }
///   if (alf_chroma_filter_signal_flag) {
///     alf_chroma_min_eg_order_minus1                         ue(v)
///     for i in 0..ChromaMaxGolombIdx
///       alf_chroma_eg_order_increase_flag[i]                 u(1)
///     for j in 0..6
///       alf_chroma_coeff_abs[j]                              uek(v)
///       if (alf_chroma_coeff_abs[j])
///         alf_chroma_coeff_sign_flag[j]                      u(1)
///   }
/// }
/// ```
pub fn parse_alf_data(payload: &[u8]) -> Result<AlfData> {
    if payload.is_empty() {
        return Err(Error::invalid("evc alf: empty ALF APS payload"));
    }
    let mut br = BitReader::new(payload);
    let mut data = AlfData::default();

    let luma_signal = br.u1()? != 0;
    let chroma_signal = br.u1()? != 0;
    data.luma_filter_signal = luma_signal;
    data.chroma_filter_signal = chroma_signal;

    if luma_signal {
        // --- Class count + filter type --------------------------------
        let n_minus1 = br.ue()?;
        if n_minus1 >= ALF_MAX_LUMA_FILTERS as u32 {
            return Err(Error::invalid(format!(
                "evc alf: alf_luma_num_filters_signalled_minus1 {n_minus1} > {}",
                ALF_MAX_LUMA_FILTERS - 1
            )));
        }
        let num_signalled = (n_minus1 + 1) as usize; // NumSignalledFilter
        data.num_signalled_luma_filters = num_signalled;
        data.num_luma_filters = ALF_MAX_LUMA_FILTERS; // Eq. 1320 always yields 0..24

        let type_flag = br.u1()? != 0;
        data.luma_type_flag = type_flag;
        let coef_pos_map: &[usize; ALF_LUMA_NUM_COEF] = if type_flag {
            &COEF_POS_MAP_13
        } else {
            &COEF_POS_MAP_7
        };

        // --- alf_luma_coeff_delta_idx[ ] ------------------------------
        if num_signalled > 1 {
            let bits = ceil_log2(num_signalled as u32);
            for i in 0..ALF_MAX_LUMA_FILTERS {
                let v = br.u(bits)? as usize;
                if v >= num_signalled {
                    return Err(Error::invalid(format!(
                        "evc alf: alf_luma_coeff_delta_idx[{i}] {v} >= NumSignalledFilter {num_signalled}"
                    )));
                }
                data.alf_luma_coeff_delta_idx[i] = v as u8;
            }
        } else {
            // Inferred to 0.
            for slot in data.alf_luma_coeff_delta_idx.iter_mut() {
                *slot = 0;
            }
        }

        // --- Fixed filter pattern + flags ------------------------------
        let fixed_pattern = br.uek(0)?; // k=0 per §7.3.5
        if fixed_pattern > 2 {
            return Err(Error::invalid(format!(
                "evc alf: alf_luma_fixed_filter_usage_pattern {fixed_pattern} out of range 0..2"
            )));
        }
        if fixed_pattern == 2 {
            for i in 0..ALF_MAX_LUMA_FILTERS {
                data.alf_luma_fixed_filter_usage_flag[i] = br.u1()? != 0;
            }
        } else if fixed_pattern == 1 {
            for slot in data.alf_luma_fixed_filter_usage_flag.iter_mut() {
                *slot = true;
            }
        } // pattern == 0 → all false (default).
        if fixed_pattern > 0 {
            for i in 0..ALF_MAX_LUMA_FILTERS {
                if data.alf_luma_fixed_filter_usage_flag[i] {
                    let idx = br.u(4)?;
                    if idx > 15 {
                        return Err(Error::invalid(format!(
                            "evc alf: alf_luma_fixed_filter_set_idx[{i}] {idx} > 15"
                        )));
                    }
                    data.alf_luma_fixed_filter_set_idx[i] = idx as u8;
                }
            }
        }

        // --- coeff_delta_flag + coeff_delta_prediction_flag -----------
        let coeff_delta_flag = br.u1()? != 0;
        let coeff_delta_prediction_flag = if !coeff_delta_flag && num_signalled > 1 {
            br.u1()? != 0
        } else {
            false
        };

        // --- expGoOrderY[ ] -------------------------------------------
        let min_eg_order_minus1 = br.ue()?;
        if min_eg_order_minus1 > 6 {
            return Err(Error::invalid(format!(
                "evc alf: alf_luma_min_eg_order_minus1 {min_eg_order_minus1} > 6"
            )));
        }
        let mut k_min = min_eg_order_minus1 + 1;
        let max_idx_y = luma_max_golomb_idx(type_flag);
        let mut exp_go_order_y = [0u32; 3]; // upper bound LumaMaxGolombIdx ≤ 3
        for slot in exp_go_order_y.iter_mut().take(max_idx_y) {
            let inc = br.u1()?;
            *slot = k_min + inc;
            k_min = *slot;
        }

        // --- alf_luma_coeff_flag[ i ] ---------------------------------
        let mut coeff_flag = [true; ALF_MAX_LUMA_FILTERS];
        if coeff_delta_flag {
            for slot in coeff_flag.iter_mut().take(num_signalled) {
                *slot = br.u1()? != 0;
            }
        }

        // --- alf_luma_coeff_delta_abs / sign --------------------------
        // filterCoefficients[ i ][ j ] for i in 0..num_signalled, j in 0..NumAlfCoefs-1
        let n_coefs = if type_flag { 12 } else { 6 };
        let mut filter_coefficients = vec![[0i16; 12]; num_signalled];
        for i in 0..num_signalled {
            if !coeff_flag[i] {
                continue;
            }
            for j in 0..n_coefs {
                let order_idx = GOLOMB_ORDER_IDX_Y[j].min(max_idx_y - 1);
                let k = exp_go_order_y[order_idx];
                let abs_val = br.uek(k)? as i32;
                let sign = if abs_val != 0 { br.u1()? } else { 1 };
                // Per §7.3.5 the sign convention is the same as luma later: sign==0 → negative, ==1 → positive
                let c = if sign == 0 { -abs_val } else { abs_val };
                filter_coefficients[i][j] = c as i16;
            }
        }
        // Eq. 97: cumulative-sum prediction across i.
        if coeff_delta_prediction_flag {
            for i in 1..num_signalled {
                let prev_row = filter_coefficients[i - 1];
                for (j, prev) in prev_row.iter().enumerate().take(n_coefs) {
                    filter_coefficients[i][j] = filter_coefficients[i][j].saturating_add(*prev);
                }
            }
        }

        // --- §8.9.4 derivation eq. 96-104 (data.luma_filters[ filtIdx ]) ---
        derive_alf_coeff_l(&mut data, &filter_coefficients, coef_pos_map, type_flag);
    }

    if chroma_signal {
        // --- expGoOrderC[ ] -------------------------------------------
        let min_eg_order_minus1 = br.ue()?;
        if min_eg_order_minus1 > 6 {
            return Err(Error::invalid(format!(
                "evc alf: alf_chroma_min_eg_order_minus1 {min_eg_order_minus1} > 6"
            )));
        }
        let mut k_min = min_eg_order_minus1 + 1;
        let mut exp_go_order_c = [0u32; CHROMA_MAX_GOLOMB_IDX];
        for slot in exp_go_order_c.iter_mut() {
            let inc = br.u1()?;
            *slot = k_min + inc;
            k_min = *slot;
        }

        // --- Single chroma alternate (EVC v1) -------------------------
        // §7.3.5 second half does not signal `num_chroma_alts`; the
        // syntax is a flat loop over j = 0..5 yielding one chroma filter
        // (§8.9.4.2 eq. 109 / 110).
        data.num_chroma_alts = 1;
        let mut coef = [0i16; ALF_CHROMA_NUM_COEF];
        let mut sum2: i32 = 0;
        for (j, slot) in coef.iter_mut().enumerate().take(6) {
            let order_idx = GOLOMB_ORDER_IDX_C[j].min(CHROMA_MAX_GOLOMB_IDX - 1);
            let k = exp_go_order_c[order_idx];
            let abs_val = br.uek(k)? as i32;
            let sign = if abs_val != 0 { br.u1()? } else { 1 };
            // §7.3.5 / eq. 109: sign == 0 → negative, sign == 1 → positive.
            let c = if sign == 0 { -abs_val } else { abs_val };
            *slot = c as i16;
            sum2 += c << 1;
        }
        // Eq. 110: AlfCoeffC[ 6 ] = 512 − Σ AlfCoeffC[ k ] << 1.
        let dc = (512 - sum2).clamp(-1024, 1023);
        coef[6] = dc as i16;
        data.chroma_filters[0] = AlfChromaFilter { coef };
    }

    Ok(data)
}

/// `Ceil(Log2(n))` for n >= 1. Returns 0 only when n == 1 (one-element
/// table needs 0 bits to address).
#[inline]
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Derive `AlfCoeffL[ filtIdx ][ 0..12 ]` for every class (§8.9.4 eq.
/// 96-104), writing the result into `data.luma_filters[ filtIdx ]`.
///
/// * eq. 98: if `alf_luma_fixed_filter_usage_flag[ filtIdx ] == 1`, seed
///   `outCoef[ filtIdx ][ j ]` from
///   `AlfFixFiltCoeff[ AlfClassToFiltMap[ filtIdx ][ set_idx ] ][ j ]`.
/// * eq. 99: otherwise zero.
/// * eq. 100: for every `j` with `coefPosMap[ j ] > 0`, add the per-class
///   delta `filterCoefficients[ alf_luma_coeff_delta_idx[ filtIdx ] ][ coefPosMap[ j ] − 1 ]`.
/// * eq. 101: `AlfCoeffL[ ][ j ] = outCoef[ j ]`.
/// * eq. 104: position 12 = `512 − Σ_{k=0..11} AlfCoeffL[ ][ k ] << 1`.
///
/// `filter_coefficients` has shape `[NumSignalledFilter][12]`.
fn derive_alf_coeff_l(
    data: &mut AlfData,
    filter_coefficients: &[[i16; 12]],
    coef_pos_map: &[usize; ALF_LUMA_NUM_COEF],
    type_flag: bool,
) {
    for (filt_idx, class_row) in ALF_CLASS_TO_FILT_MAP
        .iter()
        .enumerate()
        .take(ALF_MAX_LUMA_FILTERS)
    {
        let mut out_coef = [0i32; ALF_LUMA_NUM_COEF];
        // eq. 98 / 99 seed.
        if data.alf_luma_fixed_filter_usage_flag[filt_idx] {
            let set_idx = data.alf_luma_fixed_filter_set_idx[filt_idx] as usize;
            let fix_row = class_row[set_idx] as usize;
            for (j, slot) in out_coef.iter_mut().enumerate().take(12) {
                *slot = ALF_FIX_FILT_COEFF[fix_row][j] as i32;
            }
        }
        // eq. 100 — add per-class deltas at coefPosMap positions.
        let signalled = data.alf_luma_coeff_delta_idx[filt_idx] as usize;
        if signalled < filter_coefficients.len() {
            for (j, &pos) in coef_pos_map.iter().enumerate().take(12) {
                if pos > 0 {
                    let delta = filter_coefficients[signalled][pos - 1] as i32;
                    out_coef[j] += delta;
                }
            }
        }
        // eq. 101 — clamp to spec range −2^9 .. 2^9 - 1 for j = 0..11.
        let mut coef = [0i16; ALF_LUMA_NUM_COEF];
        for (j, slot) in coef.iter_mut().enumerate().take(12) {
            *slot = out_coef[j].clamp(-512, 511) as i16;
        }
        // eq. 104 — position 12 = 512 − 2 × Σ coef[0..11].
        let sum2: i32 = coef.iter().take(12).map(|c| (*c as i32) << 1).sum();
        let dc = (512 - sum2).clamp(-1024, 1023);
        coef[12] = dc as i16;
        data.luma_filters[filt_idx] = AlfLumaFilter { coef };
        // For the 7-tap form some `coefPosMap` entries are 0 (no spatial
        // contribution); eq. 99 ensures those slots stay at the fixed-filter
        // seed (or 0 if no fixed filter), so no special handling is needed
        // beyond the eq. 100 guard above.
        let _ = type_flag; // currently unused outside parser
    }
}

// =====================================================================
// Filter tap-pass (§8.9.4).
// =====================================================================

/// 7×7 diamond luma tap offsets per §8.9.4.1.
///
/// The filter footprint relative to the sample at (x, y) uses the
/// 13 coefficient positions listed in eq. 1265. We store the offsets as
/// (dy, dx) pairs in the same order as c[0]..c[12].
///
/// The diamond is symmetric; eq. 1265 writes the north-half only and the
/// south-half mirrors it. The 13 positions (with c[12] as the centre DC
/// bias term) are (row-offset, col-offset) from the filtered sample:
///
/// ```text
/// c[0]  → (−3,   0)
/// c[1]  → (−2,  −1)   c[2]  → (−2,  0)   (by symmetry c[1] also = (+2, +1))
/// c[3]  → (−1,  −2)   c[4]  → (−1, -1)   c[5]  → (−1,  0)
/// c[6]  → ( 0,  −3)   c[7]  → ( 0, -2)   c[8]  → ( 0, -1)
/// c[9]  → (+1,  −2)   c[10] → (+1, -1)   c[11] → (+1,  0)  (= mirrored c[3..5])
/// c[12] → DC offset (not a spatial tap)
/// ```
///
/// In the spec's symmetric filter notation each c[k] (k < 12) appears twice
/// (for both the north and south diamond arm). `c[12]` is added once.
static LUMA_TAPS: [(i32, i32); 12] = [
    (-3, 0),
    (-2, -1),
    (-2, 0),
    (-1, -2),
    (-1, -1),
    (-1, 0),
    (0, -3),
    (0, -2),
    (0, -1),
    (1, -2),
    (1, -1),
    (1, 0),
];

/// Symmetric counterpart of each tap: the position mirrored through the
/// centre sample. Follows from the diamond-symmetry property of the filter
/// (§8.9.4.1 eq. 1263): sample at `−(dy, dx)` is also weighted by c[k].
static LUMA_TAPS_SYM: [(i32, i32); 12] = [
    (3, 0),
    (2, 1),
    (2, 0),
    (1, 2),
    (1, 1),
    (1, 0),
    (0, 3),
    (0, 2),
    (0, 1),
    (-1, 2),
    (-1, 1),
    (-1, 0),
];

/// 5×5 diamond chroma tap offsets per §8.8.4.4 eq. 1321. The chroma filter
/// has 7 coefficients arranged on a 5×5 cross / X-cross diamond: 6 spatial
/// pairs followed by the DC offset (`coef[6]`, derived per eq. 110).
///
/// Round 145 corrects the round-11 ordering for taps 3-5 — the previous
/// table mapped coef[3] / coef[4] / coef[5] to the wrong geometric
/// positions vs. eq. 1321. The per-tap reads listed by eq. 1321 are:
///
/// | k | first read       | symmetric read   |
/// |---|------------------|------------------|
/// | 0 | rec[x  , y − 2]  | rec[x  , y + 2]  |
/// | 1 | rec[x − 1, y − 1]| rec[x + 1, y + 1]|
/// | 2 | rec[x  , y − 1]  | rec[x  , y + 1]  |
/// | 3 | rec[x + 1, y − 1]| rec[x − 1, y + 1]|
/// | 4 | rec[x − 2, y]    | rec[x + 2, y]    |
/// | 5 | rec[x − 1, y]    | rec[x + 1, y]    |
///
/// `CHROMA_TAPS[k]` is the `(dy, dx)` of the first read; `CHROMA_TAPS_SYM[k]`
/// is its eq. 1321 symmetric partner.
static CHROMA_TAPS: [(i32, i32); 6] = [(-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1)];
static CHROMA_TAPS_SYM: [(i32, i32); 6] = [(2, 0), (1, 1), (1, 0), (1, -1), (0, 2), (0, 1)];

/// Apply the ALF luma filter to the entire Y plane per §8.8.4.2 eq.
/// 1286-1288 (whole-plane, no per-sample classification). Writes results
/// back in-place against a pre-filter snapshot.
///
/// Round-120 update: now uses the spec scaling (sum + 256) >> 9 with
/// `coef[12]` multiplying the centre sample (`recPicture[x, y]`) rather
/// than the legacy `(sum + 64) >> 7` constant-DC approximation. Filter
/// coefficients are expected in the §8.9.4 eq. 101 / eq. 104 range —
/// i.e. those returned by [`parse_alf_data`].
pub fn apply_alf_luma(pic: &mut YuvPicture, filter: &AlfLumaFilter, bit_depth: u32) {
    let w = pic.width as usize;
    let h = pic.height as usize;
    let stride = pic.y_stride();
    let src = pic.y.clone();
    let max_val = ((1u32 << bit_depth) - 1) as i32;

    for row in 0..h {
        for col in 0..w {
            let x = col as i32;
            let y = row as i32;
            let coef = &filter.coef;
            let mut sum: i32 = 0;
            for k in 0..12 {
                let (dy0, dx0) = LUMA_TAPS[k];
                let (dy1, dx1) = LUMA_TAPS_SYM[k];
                let s0 = {
                    let xc = (x + dx0).clamp(0, w as i32 - 1) as usize;
                    let yc = (y + dy0).clamp(0, h as i32 - 1) as usize;
                    src[yc * stride + xc] as i32
                };
                let s1 = {
                    let xc = (x + dx1).clamp(0, w as i32 - 1) as usize;
                    let yc = (y + dy1).clamp(0, h as i32 - 1) as usize;
                    src[yc * stride + xc] as i32
                };
                sum += coef[k] as i32 * (s0 + s1);
            }
            // Eq. 1286: filterCoeff[12] * recPicture[x, y].
            sum += coef[12] as i32 * src[row * stride + col] as i32;
            // Eq. 1287/1288: (sum + 256) >> 9, clipped to bit-depth.
            let out = ((sum + 256) >> 9).clamp(0, max_val);
            pic.y[row * stride + col] = out as u16;
        }
    }
}

/// Apply the ALF chroma filter to one chroma plane (`c_idx` 1 = Cb, 2 = Cr).
/// Round-120 uses the spec scaling per §8.8.4.4: `(sum + 256) >> 9` with
/// `coef[6]` (`AlfCoeffC[6]` per eq. 110) multiplying the centre sample.
pub fn apply_alf_chroma(
    pic: &mut YuvPicture,
    filter: &AlfChromaFilter,
    c_idx: usize,
    bit_depth: u32,
) {
    let (cw, ch) = match pic.chroma_format_idc {
        1 => (
            pic.width.div_ceil(2) as usize,
            pic.height.div_ceil(2) as usize,
        ),
        2 => (pic.width.div_ceil(2) as usize, pic.height as usize),
        _ => (pic.width as usize, pic.height as usize),
    };
    let stride = pic.c_stride();
    let plane = if c_idx == 1 {
        pic.cb.clone()
    } else {
        pic.cr.clone()
    };
    let max_val = ((1u32 << bit_depth) - 1) as i32;
    let dst = if c_idx == 1 { &mut pic.cb } else { &mut pic.cr };

    for row in 0..ch {
        for col in 0..cw {
            let x = col as i32;
            let y = row as i32;
            let coef = &filter.coef;
            let mut sum: i32 = 0;
            for k in 0..6 {
                let (dy0, dx0) = CHROMA_TAPS[k];
                let (dy1, dx1) = CHROMA_TAPS_SYM[k];
                let s0 = {
                    let xc = (x + dx0).clamp(0, cw as i32 - 1) as usize;
                    let yc = (y + dy0).clamp(0, ch as i32 - 1) as usize;
                    plane[yc * stride + xc] as i32
                };
                let s1 = {
                    let xc = (x + dx1).clamp(0, cw as i32 - 1) as usize;
                    let yc = (y + dy1).clamp(0, ch as i32 - 1) as usize;
                    plane[yc * stride + xc] as i32
                };
                sum += coef[k] as i32 * (s0 + s1);
            }
            sum += coef[6] as i32 * plane[row * stride + col] as i32;
            let out = ((sum + 256) >> 9).clamp(0, max_val);
            dst[row * stride + col] = out as u16;
        }
    }
}

/// Apply luma ALF filter set 0 to a picture, then chroma ALF (if present).
/// This is the flat "apply filter globally" entry called by the decoder
/// pipeline when `sps_alf_flag && slice_alf_enabled_flag`.
///
/// Per-CTU `alf_ctb_flag` selection is deferred (§8.9.6) — all CTUs
/// receive the same filter for this round.
pub fn apply_alf(pic: &mut YuvPicture, alf: &AlfData, bit_depth_luma: u32, bit_depth_chroma: u32) {
    if alf.luma_filter_signal && alf.num_luma_filters > 0 {
        apply_alf_luma(pic, &alf.luma_filters[0], bit_depth_luma);
    }
    if alf.chroma_filter_signal && alf.num_chroma_alts > 0 && pic.chroma_format_idc != 0 {
        apply_alf_chroma(pic, &alf.chroma_filters[0], 1, bit_depth_chroma);
        apply_alf_chroma(pic, &alf.chroma_filters[0], 2, bit_depth_chroma);
    }
}

// =====================================================================
// Per-CTB ALF applicability map (§8.9 / §7.3.8.2).
// =====================================================================

/// Resolved per-CTU `alf_ctb_*` applicability decoded by
/// `coding_tree_unit()` (§7.3.8.2). One triplet per CTU in raster scan
/// order (`rx + ry * ctbs_wide`), carrying the resolved (present-or-
/// inferred) luma / Cb / Cr on/off state used by the §8.9 apply to
/// decide which coding tree blocks receive the filter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AlfCtbMap {
    /// `CtbLog2SizeY` — luma CTB side length is `1 << ctb_log2_size_y`.
    pub ctb_log2_size_y: u32,
    /// `PicWidthInCtbsY`.
    pub ctbs_wide: u32,
    /// `PicHeightInCtbsY`.
    pub ctbs_high: u32,
    /// Per-CTU luma `alf_ctb_flag` (raster order, length
    /// `ctbs_wide * ctbs_high`).
    pub luma: Vec<bool>,
    /// Per-CTU Cb `alf_ctb_chroma_flag`.
    pub chroma_cb: Vec<bool>,
    /// Per-CTU Cr `alf_ctb_chroma2_flag`.
    pub chroma_cr: Vec<bool>,
}

impl AlfCtbMap {
    /// Allocate an all-off map sized to a picture of `pic_width × pic_height`
    /// luma samples at `ctb_log2_size_y`.
    pub fn new(pic_width: u32, pic_height: u32, ctb_log2_size_y: u32) -> Self {
        let ctb_size = 1u32 << ctb_log2_size_y;
        let ctbs_wide = pic_width.div_ceil(ctb_size).max(1);
        let ctbs_high = pic_height.div_ceil(ctb_size).max(1);
        let n = (ctbs_wide as usize) * (ctbs_high as usize);
        Self {
            ctb_log2_size_y,
            ctbs_wide,
            ctbs_high,
            luma: vec![false; n],
            chroma_cb: vec![false; n],
            chroma_cr: vec![false; n],
        }
    }

    /// Record the resolved flags for the CTU at raster index `idx`.
    pub fn set(&mut self, idx: usize, luma: bool, chroma_cb: bool, chroma_cr: bool) {
        if idx < self.luma.len() {
            self.luma[idx] = luma;
            self.chroma_cb[idx] = chroma_cb;
            self.chroma_cr[idx] = chroma_cr;
        }
    }

    /// True when at least one CTU has its luma `alf_ctb_flag` set — i.e.
    /// the masked luma apply would touch at least one CTB.
    pub fn any_luma_on(&self) -> bool {
        self.luma.iter().any(|&b| b)
    }
}

/// Apply the luma ALF filter only to coding tree blocks whose
/// `alf_ctb_flag[ rx ][ ry ]` is 1, per the §8.9 loop (lines 18059-18074).
///
/// Like [`apply_alf_luma`], the filter reads from a pre-filter snapshot of
/// the whole plane (so a CTB at the boundary of a filtered region still
/// reads its neighbours' unfiltered values — matching the spec, whose
/// `recPicture` input is the reconstructed array prior to ALF). The §8.9
/// `blkWidth` / `blkHeight` boundary clamp falls out naturally: the
/// per-CTB write range is intersected with the picture extent, so a CTB
/// that overruns the right / bottom edge only filters its in-picture
/// samples.
///
/// Samples in CTBs whose flag is 0 are left at their reconstructed value
/// (the spec initialises `alfPicture` to `recPicture` and only overwrites
/// filtered CTBs).
pub fn apply_alf_luma_masked(
    pic: &mut YuvPicture,
    filter: &AlfLumaFilter,
    map: &AlfCtbMap,
    bit_depth: u32,
) {
    let w = pic.width as usize;
    let h = pic.height as usize;
    let stride = pic.y_stride();
    let src = pic.y.clone();
    let max_val = ((1u32 << bit_depth) - 1) as i32;
    let ctb_size = 1usize << map.ctb_log2_size_y;
    let coef = &filter.coef;

    for ry in 0..map.ctbs_high as usize {
        for rx in 0..map.ctbs_wide as usize {
            let idx = ry * map.ctbs_wide as usize + rx;
            if !map.luma.get(idx).copied().unwrap_or(false) {
                continue;
            }
            // §8.9: blkWidth / blkHeight clamp to the picture edge.
            let x0 = rx * ctb_size;
            let y0 = ry * ctb_size;
            let x1 = (x0 + ctb_size).min(w);
            let y1 = (y0 + ctb_size).min(h);
            for row in y0..y1 {
                for col in x0..x1 {
                    let x = col as i32;
                    let y = row as i32;
                    let mut sum: i32 = 0;
                    for k in 0..12 {
                        let (dy0, dx0) = LUMA_TAPS[k];
                        let (dy1, dx1) = LUMA_TAPS_SYM[k];
                        let s0 = {
                            let xc = (x + dx0).clamp(0, w as i32 - 1) as usize;
                            let yc = (y + dy0).clamp(0, h as i32 - 1) as usize;
                            src[yc * stride + xc] as i32
                        };
                        let s1 = {
                            let xc = (x + dx1).clamp(0, w as i32 - 1) as usize;
                            let yc = (y + dy1).clamp(0, h as i32 - 1) as usize;
                            src[yc * stride + xc] as i32
                        };
                        sum += coef[k] as i32 * (s0 + s1);
                    }
                    sum += coef[12] as i32 * src[row * stride + col] as i32;
                    let out = ((sum + 256) >> 9).clamp(0, max_val);
                    pic.y[row * stride + col] = out as u16;
                }
            }
        }
    }
}

/// Apply the §8.8.4.4 **coding-tree-block chroma type filtering process**
/// to either the Cb or Cr plane, gated per-CTU by the §7.3.8.2 chroma
/// applicability map.
///
/// Round 145 closes the chroma half of round-113's per-CTB apply
/// mechanism. Until now `apply_alf_chroma` filtered the entire chroma
/// plane uniformly; the spec invocation in §8.8.4.1 lines 18079-18089 is
/// per coding tree block, gated on `alf_ctb_chroma_flag[ rx ][ ry ]` /
/// `alf_ctb_chroma2_flag[ rx ][ ry ]` (the `ChromaArrayType == 3` path).
///
/// Per CTU, the chroma CTB is located at
/// `( ( rx << CtbLog2SizeY ) / SubWidthC, ( ry << CtbLog2SizeY ) / SubHeightC )`
/// with size `( blkWidth / SubWidthC, blkHeight / SubHeightC )` — the
/// §8.8.4.1 chroma-CTB derivation (lines 18105-18107). `c_idx` selects the
/// plane (1 = Cb, 2 = Cr) and also picks which side of `map` (`chroma_cb`
/// vs `chroma_cr`) gates the per-CTB application.
///
/// As with [`apply_alf_luma_masked`], the per-CTB writes read from a
/// pre-filter snapshot of the whole plane, so a filtered CTB at the edge of
/// a flagged region still reads its neighbours at their unfiltered values
/// (matching the spec's `recPicture` input which is the array prior to
/// ALF). The §8.8.4.4 derivation already clips to picture extent via the
/// edge clamp, so a CTB that overruns the right / bottom edge only filters
/// its in-picture samples.
///
/// Equation 1321 reads coefficients in geometric order — `coef[0]` is the
/// outer vertical pair, `coef[6]` the centre DC — matching the
/// [`CHROMA_TAPS`] / [`CHROMA_TAPS_SYM`] layout corrected in round 145.
pub fn apply_alf_chroma_masked(
    pic: &mut YuvPicture,
    filter: &AlfChromaFilter,
    map: &AlfCtbMap,
    c_idx: usize,
    bit_depth: u32,
) {
    debug_assert!(c_idx == 1 || c_idx == 2, "c_idx must be 1 (Cb) or 2 (Cr)");
    if pic.chroma_format_idc == 0 {
        return;
    }
    let (sub_w, sub_h) = chroma_sub_sampling(pic.chroma_format_idc);
    let (cw, ch) = chroma_plane_dims(pic.width, pic.height, pic.chroma_format_idc);
    let stride = pic.c_stride();
    let snapshot = if c_idx == 1 {
        pic.cb.clone()
    } else {
        pic.cr.clone()
    };
    let max_val = ((1u32 << bit_depth) - 1) as i32;
    let dst = if c_idx == 1 { &mut pic.cb } else { &mut pic.cr };
    let ctb_size = 1usize << map.ctb_log2_size_y;
    let coef = &filter.coef;
    let plane_flags = if c_idx == 1 {
        &map.chroma_cb
    } else {
        &map.chroma_cr
    };

    for ry in 0..map.ctbs_high as usize {
        for rx in 0..map.ctbs_wide as usize {
            let idx = ry * map.ctbs_wide as usize + rx;
            if !plane_flags.get(idx).copied().unwrap_or(false) {
                continue;
            }
            // §8.8.4.1 lines 18105-18107: luma → chroma CTB coordinates +
            // dims via SubWidthC / SubHeightC. The CTU origin is on the
            // chroma grid because CTB size is a power of two ≥ chroma
            // subsampling (4:2:0 / 4:2:2 → ctb_size is even).
            let x_ctb_c = (rx * ctb_size) / sub_w;
            let y_ctb_c = (ry * ctb_size) / sub_h;
            let blk_w_c = (ctb_size / sub_w).min(cw.saturating_sub(x_ctb_c));
            let blk_h_c = (ctb_size / sub_h).min(ch.saturating_sub(y_ctb_c));
            for row in 0..blk_h_c {
                for col in 0..blk_w_c {
                    let cx = x_ctb_c + col;
                    let cy = y_ctb_c + row;
                    let x = cx as i32;
                    let y = cy as i32;
                    let mut sum: i32 = 0;
                    for k in 0..6 {
                        let (dy0, dx0) = CHROMA_TAPS[k];
                        let (dy1, dx1) = CHROMA_TAPS_SYM[k];
                        let s0 = {
                            let xc = (x + dx0).clamp(0, cw as i32 - 1) as usize;
                            let yc = (y + dy0).clamp(0, ch as i32 - 1) as usize;
                            snapshot[yc * stride + xc] as i32
                        };
                        let s1 = {
                            let xc = (x + dx1).clamp(0, cw as i32 - 1) as usize;
                            let yc = (y + dy1).clamp(0, ch as i32 - 1) as usize;
                            snapshot[yc * stride + xc] as i32
                        };
                        sum += coef[k] as i32 * (s0 + s1);
                    }
                    sum += coef[6] as i32 * snapshot[cy * stride + cx] as i32;
                    let out = ((sum + 256) >> 9).clamp(0, max_val);
                    dst[cy * stride + cx] = out as u16;
                }
            }
        }
    }
}

/// (SubWidthC, SubHeightC) per ChromaArrayType for chroma-sub-sample math.
/// Mirrors the §6.2 picture-format Table 2 mapping; ChromaArrayType 0
/// (monochrome) is caller-gated and returns the harmless (1, 1).
#[inline]
fn chroma_sub_sampling(chroma_format_idc: u32) -> (usize, usize) {
    match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    }
}

/// Chroma plane dimensions in samples per `chroma_format_idc`.
#[inline]
fn chroma_plane_dims(width: u32, height: u32, chroma_format_idc: u32) -> (usize, usize) {
    match chroma_format_idc {
        1 => (width.div_ceil(2) as usize, height.div_ceil(2) as usize),
        2 => (width.div_ceil(2) as usize, height as usize),
        3 => (width as usize, height as usize),
        _ => (0, 0),
    }
}

/// ALF apply driven by a decoded per-CTU applicability map (§8.9).
///
/// The luma filter is masked per [`apply_alf_luma_masked`]. Chroma follows
/// the §8.9 lines 18099-18116 path for `ChromaArrayType` 1..2: when
/// `sliceChromaAlfEnabledFlag` / `sliceChroma2AlfEnabledFlag` holds the
/// chroma plane is filtered as a whole (there is no per-CTB chroma flag in
/// the 4:2:0 / 4:2:2 case — `slice_alf_chroma_idc > 0` enables the plane and
/// the §7.3.8.2 chroma map flags are inferred 0). The caller decides whether
/// chroma is enabled via `chroma_cb_enabled` / `chroma_cr_enabled` (the
/// slice-level enables resolved by the header parser).
pub fn apply_alf_with_map(
    pic: &mut YuvPicture,
    alf: &AlfData,
    map: &AlfCtbMap,
    chroma_cb_enabled: bool,
    chroma_cr_enabled: bool,
    bit_depth_luma: u32,
    bit_depth_chroma: u32,
) {
    if alf.luma_filter_signal && alf.num_luma_filters > 0 {
        apply_alf_luma_masked(pic, &alf.luma_filters[0], map, bit_depth_luma);
    }
    if alf.chroma_filter_signal && alf.num_chroma_alts > 0 && pic.chroma_format_idc != 0 {
        if chroma_cb_enabled {
            apply_alf_chroma(pic, &alf.chroma_filters[0], 1, bit_depth_chroma);
        }
        if chroma_cr_enabled {
            apply_alf_chroma(pic, &alf.chroma_filters[0], 2, bit_depth_chroma);
        }
    }
}

// =====================================================================
// §8.8.4.3 — ALF transpose + classification filter-index derivation.
// =====================================================================

/// Number of distinct ALF luma filter classes (§8.9.4.1: `NumAlfFilters`).
/// The §8.8.4.3 classification yields a `filtIdx` in `0..NUM_ALF_FILTERS`.
pub const NUM_ALF_FILTERS: usize = 25;

/// Per-sample classification output of the §8.8.4.3 derivation.
///
/// One pair per luma sample inside the coding tree block, stored in
/// row-major order (`y * blk_width + x`), with `x = 0..blk_width − 1` and
/// `y = 0..blk_height − 1`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AlfClassification {
    /// Coding-tree-block width in luma samples (`blkWidth`).
    pub blk_width: usize,
    /// Coding-tree-block height in luma samples (`blkHeight`).
    pub blk_height: usize,
    /// `filtIdx[ x ][ y ]` — gradient class in `0..NUM_ALF_FILTERS`
    /// (eq. 1319 / eq. 1320), row-major.
    pub filt_idx: Vec<u8>,
    /// `transposeIdx[ x ][ y ]` — coefficient permutation selector `0..3`
    /// (eq. 1318), row-major.
    pub transpose_idx: Vec<u8>,
}

impl AlfClassification {
    /// `filtIdx[ x ][ y ]` for `x = 0..blk_width − 1`, `y = 0..blk_height − 1`.
    #[inline]
    pub fn filt_idx_at(&self, x: usize, y: usize) -> u8 {
        self.filt_idx[y * self.blk_width + x]
    }

    /// `transposeIdx[ x ][ y ]`.
    #[inline]
    pub fn transpose_idx_at(&self, x: usize, y: usize) -> u8 {
        self.transpose_idx[y * self.blk_width + x]
    }
}

/// §8.8.4.3 `transposeTable[ ]` (eq. 1317).
const ALF_TRANSPOSE_TABLE: [u8; 8] = [0, 1, 0, 2, 2, 3, 1, 3];

/// §8.8.4.3 `varTab[ ]` (eq. 1315).
const ALF_VAR_TAB: [u8; 16] = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4];

/// Read a luma sample at `(xCtb + x, yCtb + y)` clamping to the picture
/// extent, matching the spec's behaviour that the §8.8.4.5 boundary-padding
/// process supplies the nearest in-picture sample for off-edge reads.
#[inline]
fn rec_sample_clamped(src: &[u16], stride: usize, w: usize, h: usize, x: i32, y: i32) -> i32 {
    let xc = x.clamp(0, w as i32 - 1) as usize;
    let yc = y.clamp(0, h as i32 - 1) as usize;
    src[yc * stride + xc] as i32
}

/// Derive the §8.8.4.3 classification (`filtIdx` + `transposeIdx`) for every
/// luma sample of a coding tree block located at `(x_ctb, y_ctb)` in the
/// reconstructed luma plane `src` (`w × h`, row stride `stride`).
///
/// This is a faithful clean-room transcription of ISO/IEC 23094-1 §8.8.4.3,
/// eq. 1289-1320:
///
/// 1. Per-position 1-D Laplacian gradients `filtH / filtV / filtD0 / filtD1`
///    over `x = −2..blkWidth + 1`, `y = −2..blkHeight + 1` (eq. 1289-1292).
/// 2. Per-4×4-subblock window sums `sumH / sumV / sumD0 / sumD1 / sumOfHV`
///    with `i, j = −2..5` (eq. 1293-1297).
/// 3. Per-sample direction strength `dir1 / dir2 / dirS` from the dominant
///    horizontal-vertical vs diagonal activity (eq. 1298-1314).
/// 4. Activity quantisation `avgVar` via `varTab` (eq. 1315-1316).
/// 5. `transposeIdx` (eq. 1317-1318) and the final `filtIdx` with the
///    direction-strength offset (eq. 1319-1320).
///
/// `bit_depth` is `BitDepthY`, used by eq. 1316's `>> (BitDepthY − 2)` shift.
#[allow(clippy::too_many_arguments)]
pub fn derive_alf_classification(
    src: &[u16],
    stride: usize,
    w: usize,
    h: usize,
    x_ctb: usize,
    y_ctb: usize,
    blk_width: usize,
    blk_height: usize,
    bit_depth: u32,
) -> AlfClassification {
    // --- Step 1: gradients over the [−2, blk+1] halo (eq. 1289-1292). ---
    //
    // We index gradient arrays by an offset so x, y can run from −2.
    let gw = blk_width + 4; // x: −2 .. blk_width + 1
    let gh = blk_height + 4; // y: −2 .. blk_height + 1
    let g_index = |gx: i32, gy: i32| -> usize { ((gy + 2) as usize) * gw + ((gx + 2) as usize) };

    let mut filt_h = vec![0i32; gw * gh];
    let mut filt_v = vec![0i32; gw * gh];
    let mut filt_d0 = vec![0i32; gw * gh];
    let mut filt_d1 = vec![0i32; gw * gh];

    for gy in -2..(blk_height as i32 + 2) {
        for gx in -2..(blk_width as i32 + 2) {
            let px = x_ctb as i32 + gx;
            let py = y_ctb as i32 + gy;
            let c = rec_sample_clamped(src, stride, w, h, px, py) << 1;
            let l = rec_sample_clamped(src, stride, w, h, px - 1, py);
            let r = rec_sample_clamped(src, stride, w, h, px + 1, py);
            let u = rec_sample_clamped(src, stride, w, h, px, py - 1);
            let d = rec_sample_clamped(src, stride, w, h, px, py + 1);
            let ul = rec_sample_clamped(src, stride, w, h, px - 1, py - 1);
            let dr = rec_sample_clamped(src, stride, w, h, px + 1, py + 1);
            let ur = rec_sample_clamped(src, stride, w, h, px + 1, py - 1);
            let dl = rec_sample_clamped(src, stride, w, h, px - 1, py + 1);
            let idx = g_index(gx, gy);
            filt_h[idx] = (c - l - r).abs();
            filt_v[idx] = (c - u - d).abs();
            filt_d0[idx] = (c - ul - dr).abs();
            filt_d1[idx] = (c - ur - dl).abs();
        }
    }

    // --- Step 2: per-4×4-subblock window sums (eq. 1293-1297). ---
    //
    // Subblock grid: sx = 0..(blkWidth − 1) >> 2, sy = 0..(blkHeight − 1) >> 2.
    let sub_w = ((blk_width - 1) >> 2) + 1;
    let sub_h = ((blk_height - 1) >> 2) + 1;
    let s_index = |sx: usize, sy: usize| -> usize { sy * sub_w + sx };

    let mut sum_h = vec![0i32; sub_w * sub_h];
    let mut sum_v = vec![0i32; sub_w * sub_h];
    let mut sum_d0 = vec![0i32; sub_w * sub_h];
    let mut sum_d1 = vec![0i32; sub_w * sub_h];
    let mut sum_of_hv = vec![0i32; sub_w * sub_h];

    for sy in 0..sub_h {
        for sx in 0..sub_w {
            let (mut sh, mut sv, mut sd0, mut sd1) = (0i32, 0i32, 0i32, 0i32);
            for j in -2i32..=5 {
                for i in -2i32..=5 {
                    let gx = (sx as i32) * 4 + i;
                    let gy = (sy as i32) * 4 + j;
                    let idx = g_index(gx, gy);
                    sh += filt_h[idx];
                    sv += filt_v[idx];
                    sd0 += filt_d0[idx];
                    sd1 += filt_d1[idx];
                }
            }
            let si = s_index(sx, sy);
            sum_h[si] = sh;
            sum_v[si] = sv;
            sum_d0[si] = sd0;
            sum_d1[si] = sd1;
            sum_of_hv[si] = sh + sv;
        }
    }

    // --- Steps 3-5: per-sample dir / avgVar / filtIdx / transposeIdx. ---
    let mut filt_idx = vec![0u8; blk_width * blk_height];
    let mut transpose_idx = vec![0u8; blk_width * blk_height];
    let hv_shift = bit_depth.saturating_sub(2);

    for y in 0..blk_height {
        for x in 0..blk_width {
            let si = s_index(x >> 2, y >> 2);
            let s_h = sum_h[si];
            let s_v = sum_v[si];
            let s_d0 = sum_d0[si];
            let s_d1 = sum_d1[si];

            // dirHV / hv1 / hv0 (eq. 1298-1303).
            let (hv1, hv0, dir_hv) = if s_v > s_h {
                (s_v, s_h, 1u8)
            } else {
                (s_h, s_v, 3u8)
            };

            // dirD / d1 / d0 (eq. 1304-1309).
            let (d1, d0, dir_d) = if s_d0 > s_d1 {
                (s_d0, s_d1, 0u8)
            } else {
                (s_d1, s_d0, 2u8)
            };

            // hvd1 / hvd0 and the diagonal-vs-HV branch (eq. 1310-1314).
            // Use i64 for the cross products to avoid i32 overflow on large
            // CTBs / high bit depths.
            let diag_dominant = (d1 as i64) * (hv0 as i64) > (hv1 as i64) * (d0 as i64);
            let (hvd1, hvd0) = if diag_dominant { (d1, d0) } else { (hv1, hv0) };
            let dir1 = if diag_dominant { dir_d } else { dir_hv };
            let dir2 = if diag_dominant { dir_hv } else { dir_d };
            let dir_s = if hvd1 > 2 * hvd0 {
                1u8
            } else if (hvd1 as i64) * 2 > 9 * (hvd0 as i64) {
                2u8
            } else {
                0u8
            };

            // avgVar (eq. 1315-1316).
            let var_in = ((sum_of_hv[si] >> hv_shift).clamp(0, 15)) as usize;
            let avg_var = ALF_VAR_TAB[var_in];

            // transposeIdx (eq. 1317-1318).
            let t = ALF_TRANSPOSE_TABLE[(dir1 as usize) * 2 + ((dir2 >> 1) as usize)];

            // filtIdx (eq. 1319-1320).
            let mut fidx = avg_var as i32;
            if dir_s != 0 {
                fidx += (((dir1 & 0x1) << 1) as i32 + dir_s as i32) * 5;
            }

            let out = y * blk_width + x;
            filt_idx[out] = fidx as u8;
            transpose_idx[out] = t;
        }
    }

    AlfClassification {
        blk_width,
        blk_height,
        filt_idx,
        transpose_idx,
    }
}

/// Apply the §8.8.4.2 spec luma filtering loop (eq. 1281-1288) to a single
/// coding tree block at `(x_ctb, y_ctb)` of size `blk_width × blk_height`,
/// driving the per-sample filter selection from the `AlfCoeffL[ filtIdx ]`
/// derivation in `alf.luma_filters` and from the per-sample `filtIdx` /
/// `transposeIdx` arrays in `classification`. Writes results into `pic.y`
/// against a snapshot of the pre-ALF plane.
///
/// This is the closed-loop §8.9.4 + §8.8.4.2 + §8.8.4.3 wiring round-120
/// adds: until now the apply used `alf.luma_filters[0]` uniformly across
/// the whole CTB and ignored the per-sample classification. The new path
/// (1) selects `f[ j ] = AlfCoeffL[ filtIdx[x][y] ][ j ]` per eq. 1281,
/// (2) permutes to `filterCoeff[ ]` per `transposeIdx` per eq. 1282-1285,
/// (3) accumulates `sum` per eq. 1286 (north/south arm symmetric pairs +
/// `coef[12] * recPicture[x, y]`), and (4) clips per eq. 1287/1288.
///
/// `bit_depth` is `BitDepthY`. The picture-edge clamp on the tap reads
/// matches the §8.8.4.5 boundary-padding process.
#[allow(clippy::too_many_arguments)]
pub fn apply_alf_luma_classified(
    pic: &mut YuvPicture,
    alf: &AlfData,
    classification: &AlfClassification,
    x_ctb: usize,
    y_ctb: usize,
    bit_depth: u32,
) {
    let w = pic.width as usize;
    let h = pic.height as usize;
    let stride = pic.y_stride();
    let src = pic.y.clone();
    let max_val = ((1u32 << bit_depth) - 1) as i32;
    let blk_w = classification.blk_width;
    let blk_h = classification.blk_height;

    for y in 0..blk_h {
        for x in 0..blk_w {
            let px = x_ctb + x;
            let py = y_ctb + y;
            if px >= w || py >= h {
                continue;
            }
            let filt_idx = classification.filt_idx_at(x, y) as usize;
            let transpose_idx = classification.transpose_idx_at(x, y);
            // Eq. 1281: f[j] = AlfCoeffL[ apsId ][ filtIdx[ x ][ y ] ][ j ].
            let f = alf.luma_filters[filt_idx].coef;
            // Eq. 1282-1285: per-transpose permutation.
            let coef = transpose_luma_coeffs(&f, transpose_idx);

            // Eq. 1286: sum = Σ filterCoeff[k] * (north + south) +
            //                 filterCoeff[12] * recPicture[x, y].
            let mut sum: i32 = 0;
            for k in 0..12 {
                let (dy0, dx0) = LUMA_TAPS[k];
                let (dy1, dx1) = LUMA_TAPS_SYM[k];
                let s0 = {
                    let xc = (px as i32 + dx0).clamp(0, w as i32 - 1) as usize;
                    let yc = (py as i32 + dy0).clamp(0, h as i32 - 1) as usize;
                    src[yc * stride + xc] as i32
                };
                let s1 = {
                    let xc = (px as i32 + dx1).clamp(0, w as i32 - 1) as usize;
                    let yc = (py as i32 + dy1).clamp(0, h as i32 - 1) as usize;
                    src[yc * stride + xc] as i32
                };
                sum += coef[k] as i32 * (s0 + s1);
            }
            sum += coef[12] as i32 * src[py * stride + px] as i32;
            // Eq. 1287/1288: clip ((sum + 256) >> 9).
            let out = ((sum + 256) >> 9).clamp(0, max_val);
            pic.y[py * stride + px] = out as u16;
        }
    }
}

/// Apply the per-sample classified §8.8.4.2 luma filter to every CTB whose
/// `alf_ctb_flag` is 1 in `map`, deriving the §8.8.4.3 classification per
/// CTB and looking up the §8.9.4-derived per-class filter from `alf`.
///
/// Snapshots the whole plane up front so a filtered CTB's edge taps still
/// read unfiltered neighbours, mirroring [`apply_alf_luma_masked`].
pub fn apply_alf_luma_classified_masked(
    pic: &mut YuvPicture,
    alf: &AlfData,
    map: &AlfCtbMap,
    bit_depth: u32,
) {
    let w = pic.width as usize;
    let h = pic.height as usize;
    let ctb_size = 1usize << map.ctb_log2_size_y;
    for ry in 0..map.ctbs_high as usize {
        for rx in 0..map.ctbs_wide as usize {
            let idx = ry * map.ctbs_wide as usize + rx;
            if !map.luma.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let x_ctb = rx * ctb_size;
            let y_ctb = ry * ctb_size;
            let bw = (x_ctb + ctb_size).min(w) - x_ctb;
            let bh = (y_ctb + ctb_size).min(h) - y_ctb;
            let cls = derive_alf_classification(
                &pic.y,
                pic.y_stride(),
                w,
                h,
                x_ctb,
                y_ctb,
                bw,
                bh,
                bit_depth,
            );
            apply_alf_luma_classified(pic, alf, &cls, x_ctb, y_ctb, bit_depth);
        }
    }
}

/// Apply the §8.8.4.2 transpose permutation (eq. 1282-1285) to a 13-tap luma
/// filter `f[0..12]`, selecting the arrangement that the sample's
/// `transposeIdx` requests. `transpose_idx == 0` returns `f` unchanged
/// (eq. 1285).
pub fn transpose_luma_coeffs(
    f: &[i16; ALF_LUMA_NUM_COEF],
    transpose_idx: u8,
) -> [i16; ALF_LUMA_NUM_COEF] {
    match transpose_idx {
        // eq. 1282.
        1 => [
            f[9], f[4], f[10], f[8], f[1], f[5], f[11], f[7], f[3], f[0], f[2], f[6], f[12],
        ],
        // eq. 1283.
        2 => [
            f[0], f[3], f[2], f[1], f[8], f[7], f[6], f[5], f[4], f[9], f[10], f[11], f[12],
        ],
        // eq. 1284.
        3 => [
            f[9], f[8], f[10], f[4], f[3], f[7], f[11], f[5], f[1], f[0], f[2], f[6], f[12],
        ],
        // eq. 1285 (transpose_idx == 0 and any other value).
        _ => *f,
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    /// Emit a spec-faithful §7.3.5 ALF APS payload for `num_signalled` luma
    /// filter classes. Each per-tap value is encoded as uek(v) with the
    /// shared `expGoOrderY` derived from `eg_min` + `eg_inc[ ]`.
    ///
    /// `tap_values[ i ][ j ]` is the desired signed value of
    /// `filterCoefficients[ i ][ j ]` for `i = 0..num_signalled` and
    /// `j = 0..n_coefs − 1`. The emitter uses `alf_luma_coeff_delta_flag = 0`
    /// (so all classes are signalled with `alf_luma_coeff_flag = 1` inferred)
    /// and `alf_luma_fixed_filter_usage_pattern = 0`, plus
    /// `alf_luma_coeff_delta_idx = identity` when num_signalled == 25
    /// (otherwise inferred to 0).
    #[allow(clippy::too_many_arguments)]
    fn emit_spec_alf_data(
        luma_signal: bool,
        type_flag: bool,
        num_signalled: usize,
        tap_values: &[Vec<i32>],
        chroma_signal: bool,
        chroma_taps: &[i32; 6],
        chroma_eg_min: u32,
        chroma_eg_inc: [u32; 2],
        luma_eg_min: u32,
        luma_eg_inc: &[u32], // length = LumaMaxGolombIdx
    ) -> Vec<u8> {
        let mut e = BitEmitter::new();
        e.u(1, if luma_signal { 1 } else { 0 });
        e.u(1, if chroma_signal { 1 } else { 0 });
        if luma_signal {
            assert!((1..=25).contains(&num_signalled));
            e.ue((num_signalled - 1) as u32);
            e.u(1, if type_flag { 1 } else { 0 });
            if num_signalled > 1 {
                // alf_luma_coeff_delta_idx[0..25] — identity (i → i.min(num−1)).
                let bits = ceil_log2(num_signalled as u32);
                for i in 0..25 {
                    let v = (i as u32).min((num_signalled - 1) as u32);
                    e.u(bits, v);
                }
            }
            // alf_luma_fixed_filter_usage_pattern = 0 (uek(v) k=0).
            e.uek(0, 0);
            // coeff_delta_flag = 0 (so all coeff_flag[i] are inferred 1).
            e.u(1, 0);
            // coeff_delta_prediction_flag = 0 when minus1 > 0, else not signalled.
            if num_signalled > 1 {
                e.u(1, 0);
            }
            // alf_luma_min_eg_order_minus1 + per-idx increase flags.
            assert!(luma_eg_min <= 6);
            e.ue(luma_eg_min);
            assert_eq!(luma_eg_inc.len(), luma_max_golomb_idx(type_flag));
            for &inc in luma_eg_inc.iter() {
                e.u(1, inc);
            }
            // Compute the actual expGoOrderY chain to encode tap values.
            let max_idx_y = luma_max_golomb_idx(type_flag);
            let mut exp_go_order_y = [0u32; 3];
            let mut k_min = luma_eg_min + 1;
            for (i, slot) in exp_go_order_y.iter_mut().take(max_idx_y).enumerate() {
                *slot = k_min + luma_eg_inc[i];
                k_min = *slot;
            }
            let n_coefs = if type_flag { 12 } else { 6 };
            for row in tap_values.iter().take(num_signalled) {
                for j in 0..n_coefs {
                    let order_idx = GOLOMB_ORDER_IDX_Y[j].min(max_idx_y - 1);
                    let k = exp_go_order_y[order_idx];
                    let v = row[j];
                    let abs_val = v.unsigned_abs();
                    e.uek(k, abs_val);
                    if abs_val != 0 {
                        // sign convention: 0 → negative, 1 → positive (per parser).
                        e.u(1, if v >= 0 { 1 } else { 0 });
                    }
                }
            }
        }
        if chroma_signal {
            e.ue(chroma_eg_min);
            for &inc in chroma_eg_inc.iter() {
                e.u(1, inc);
            }
            // Recompute the chroma EG chain.
            let mut k_min = chroma_eg_min + 1;
            let mut exp_go_order_c = [0u32; CHROMA_MAX_GOLOMB_IDX];
            for (i, slot) in exp_go_order_c.iter_mut().enumerate() {
                *slot = k_min + chroma_eg_inc[i];
                k_min = *slot;
            }
            for (j, &v) in chroma_taps.iter().enumerate().take(6) {
                let order_idx = GOLOMB_ORDER_IDX_C[j].min(CHROMA_MAX_GOLOMB_IDX - 1);
                let k = exp_go_order_c[order_idx];
                let abs_val = v.unsigned_abs();
                e.uek(k, abs_val);
                if abs_val != 0 {
                    e.u(1, if v >= 0 { 1 } else { 0 });
                }
            }
        }
        e.finish_with_trailing_bits();
        e.into_bytes()
    }

    #[test]
    fn parse_alf_data_luma_only_identity() {
        // One signalled class, all taps 0 → fixed-filter contribution 0
        // (pattern 0) + zero deltas → every AlfCoeffL[ filtIdx ] is all-zero
        // spatial taps with DC offset = 512 (eq. 104).
        let taps = vec![vec![0i32; 12]];
        let payload = emit_spec_alf_data(
            true,
            true,
            1,
            &taps,
            false,
            &[0; 6],
            0,
            [0; 2],
            0,
            &[0, 0, 0],
        );
        let alf = parse_alf_data(&payload).unwrap();
        assert!(alf.luma_filter_signal);
        assert!(!alf.chroma_filter_signal);
        assert_eq!(alf.num_signalled_luma_filters, 1);
        assert!(alf.luma_type_flag);
        for filt_idx in 0..NUM_ALF_FILTERS {
            for j in 0..12 {
                assert_eq!(
                    alf.luma_filters[filt_idx].coef[j], 0,
                    "tap {j} of class {filt_idx} must be zero (no fixed, no delta)"
                );
            }
            // eq. 104: DC = 512 − 0 = 512.
            assert_eq!(alf.luma_filters[filt_idx].coef[12], 512);
        }
    }

    #[test]
    fn parse_alf_data_luma_with_coeffs() {
        // Single signalled class with tap 0 = +3. Every class maps to this
        // single filter (alf_luma_coeff_delta_idx inferred 0), so every
        // AlfCoeffL[ filtIdx ][ 0 ] = 3 and DC = 512 − 2 × 3 = 506.
        let mut taps = vec![0i32; 12];
        taps[0] = 3;
        let payload = emit_spec_alf_data(
            true,
            true,
            1,
            &[taps],
            false,
            &[0; 6],
            0,
            [0; 2],
            0,
            &[0, 0, 0],
        );
        let alf = parse_alf_data(&payload).unwrap();
        for filt_idx in 0..NUM_ALF_FILTERS {
            assert_eq!(alf.luma_filters[filt_idx].coef[0], 3);
            assert_eq!(alf.luma_filters[filt_idx].coef[12], 506);
        }
    }

    #[test]
    fn parse_alf_data_chroma_only() {
        let payload = emit_spec_alf_data(false, true, 0, &[], true, &[0; 6], 0, [0; 2], 0, &[]);
        let alf = parse_alf_data(&payload).unwrap();
        assert!(!alf.luma_filter_signal);
        assert!(alf.chroma_filter_signal);
        assert_eq!(alf.num_chroma_alts, 1);
        // All-zero chroma taps → DC = 512 − 0 = 512 (eq. 110).
        assert_eq!(alf.chroma_filters[0].coef[6], 512);
        for j in 0..6 {
            assert_eq!(alf.chroma_filters[0].coef[j], 0);
        }
    }

    #[test]
    fn parse_alf_rejects_empty_payload() {
        assert!(parse_alf_data(&[]).is_err());
    }

    #[test]
    fn parse_alf_rejects_too_many_filters() {
        let mut e = BitEmitter::new();
        e.u(1, 1); // luma_signal
        e.u(1, 0);
        e.ue(25); // num_signalled_minus1 = 25 → 26 filters (out of range)
        e.finish_with_trailing_bits();
        assert!(parse_alf_data(&e.into_bytes()).is_err());
    }

    #[test]
    fn alf_luma_filter_dc_is_identity_on_uniform_picture() {
        // Round-120 spec semantics (eq. 1286/1287):
        //   sum = coef[12] * recPicture[x,y] + Σ coef[k] * (s0 + s1)
        //   out = clip3(0, max, (sum + 256) >> 9)
        //
        // On a flat picture filled with value V and all spatial taps 0,
        // the "approximate identity" filter has coef[12] = 512 because
        // (512 * V + 256) >> 9 ≈ V (exact when V is small / clipped).
        let mut pic = crate::picture::YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let filter = AlfLumaFilter {
            coef: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 512],
        };
        apply_alf_luma(&mut pic, &filter, 8);
        // (512 * 100 + 256) >> 9 = (51200 + 256) >> 9 = 51456 >> 9 = 100.
        for v in pic.y.iter() {
            assert_eq!(*v, 100, "DC = 512 must be the spec identity");
        }
    }

    #[test]
    fn alf_luma_filter_sum_check() {
        // All-zero filter (every coef incl. centre is 0) maps every sample
        // to (0 + 256) >> 9 = 0.
        let mut pic = crate::picture::YuvPicture::new(8, 8, 0, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let filter = AlfLumaFilter { coef: [0; 13] };
        apply_alf_luma(&mut pic, &filter, 8);
        for v in pic.y.iter() {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn alf_chroma_filter_applies_without_panic() {
        // Round-120: spec scale (sum + 256) >> 9 with coef[6] multiplying
        // the centre sample. For V = 128 and coef = [0,0,0,0,0,0,4]:
        // (4 * 128 + 256) >> 9 = 768 >> 9 = 1.
        let mut pic = crate::picture::YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.cb.iter_mut() {
            *v = 128;
        }
        let filter = AlfChromaFilter {
            coef: [0, 0, 0, 0, 0, 0, 4],
        };
        apply_alf_chroma(&mut pic, &filter, 1, 8);
        for v in pic.cb.iter() {
            assert_eq!(*v, 1);
        }
    }

    #[test]
    fn apply_alf_noop_when_no_filters_signaled() {
        // When luma_filter_signal=false, apply_alf must not modify the picture.
        let mut pic = crate::picture::YuvPicture::new(8, 8, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 50;
        }
        let alf = AlfData {
            luma_filter_signal: false,
            chroma_filter_signal: false,
            ..AlfData::default()
        };
        apply_alf(&mut pic, &alf, 8, 8);
        for v in pic.y.iter() {
            assert_eq!(*v, 50);
        }
    }

    #[test]
    fn parse_alf_negative_coeff_sign() {
        // tap 0 = -5: sign bit = 0 (negative), abs = 5.
        let mut taps = vec![0i32; 12];
        taps[0] = -5;
        let payload = emit_spec_alf_data(
            true,
            true,
            1,
            &[taps],
            false,
            &[0; 6],
            0,
            [0; 2],
            0,
            &[0, 0, 0],
        );
        let alf = parse_alf_data(&payload).unwrap();
        assert_eq!(alf.luma_filters[0].coef[0], -5);
        // eq. 104: DC = 512 − 2*(-5) = 522.
        assert_eq!(alf.luma_filters[0].coef[12], 522);
    }

    // =================================================================
    // Round 113: per-CTB ALF apply-masking (§8.9 / §7.3.8.2).
    // =================================================================

    /// A "spec-scaled" const filter that maps every uniform-V luma sample
    /// to a fixed output. With round-120 semantics (eq. 1286/1287):
    ///   out = clip((coef[12] * V + 256) >> 9).
    /// For V = 100 and coef[12] = 10 → (1000 + 256) >> 9 = 1256 >> 9 = 2.
    /// Distinct from the reconstructed fill value (100).
    fn const_filter_to_two() -> AlfLumaFilter {
        let mut coef = [0i16; ALF_LUMA_NUM_COEF];
        coef[12] = 10;
        AlfLumaFilter { coef }
    }

    #[test]
    fn alf_ctb_map_new_sizes_to_ctb_grid() {
        // 96×64 picture at CtbLog2SizeY = 5 (32×32 CTBs) → 3 wide × 2 high.
        let map = AlfCtbMap::new(96, 64, 5);
        assert_eq!(map.ctbs_wide, 3);
        assert_eq!(map.ctbs_high, 2);
        assert_eq!(map.luma.len(), 6);
        assert!(!map.any_luma_on());
        // A partial CTB at the edge still counts as one CTB column / row.
        let map2 = AlfCtbMap::new(40, 33, 5);
        assert_eq!(map2.ctbs_wide, 2);
        assert_eq!(map2.ctbs_high, 2);
    }

    #[test]
    fn masked_luma_apply_only_touches_flagged_ctbs() {
        // 64×32 picture, CtbLog2SizeY = 5 → two 32×32 CTBs side by side.
        let mut pic = crate::picture::YuvPicture::new(64, 32, 0, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let mut map = AlfCtbMap::new(64, 32, 5);
        // CTU 0 (left) on, CTU 1 (right) off.
        map.set(0, true, false, false);
        map.set(1, false, false, false);
        let filter = const_filter_to_two();
        apply_alf_luma_masked(&mut pic, &filter, &map, 8);
        let stride = pic.y_stride();
        // Left CTB filtered to 2; right CTB unchanged at 100.
        for row in 0..32usize {
            for col in 0..32usize {
                assert_eq!(pic.y[row * stride + col], 2, "left CTB filtered");
            }
            for col in 32..64usize {
                assert_eq!(pic.y[row * stride + col], 100, "right CTB untouched");
            }
        }
    }

    #[test]
    fn masked_luma_apply_clamps_at_picture_edge() {
        // 40×40 picture, CtbLog2SizeY = 5 → 2×2 CTB grid but only the
        // top-left CTB is full 32×32; the others are partial. Flag the
        // bottom-right CTB on: it covers luma (32..40, 32..40) = 8×8.
        let mut pic = crate::picture::YuvPicture::new(40, 40, 0, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let mut map = AlfCtbMap::new(40, 40, 5);
        // raster idx 3 = (rx=1, ry=1) bottom-right partial CTB.
        map.set(3, true, false, false);
        let filter = const_filter_to_two();
        apply_alf_luma_masked(&mut pic, &filter, &map, 8);
        let stride = pic.y_stride();
        // Only samples in (32..40, 32..40) change; the rest stay 100.
        for row in 0..40usize {
            for col in 0..40usize {
                let expected = if row >= 32 && col >= 32 { 2 } else { 100 };
                assert_eq!(pic.y[row * stride + col], expected, "row={row} col={col}");
            }
        }
    }

    #[test]
    fn masked_apply_all_off_map_leaves_picture_unchanged() {
        let mut pic = crate::picture::YuvPicture::new(64, 32, 0, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 77;
        }
        let map = AlfCtbMap::new(64, 32, 5); // all off
        let filter = const_filter_to_two();
        apply_alf_luma_masked(&mut pic, &filter, &map, 8);
        for v in pic.y.iter() {
            assert_eq!(*v, 77);
        }
    }

    #[test]
    fn apply_alf_with_map_matches_whole_plane_when_all_on() {
        // With every luma CTB flagged on, the masked apply must produce the
        // identical result to the whole-plane apply_alf path.
        let make = || {
            let mut pic = crate::picture::YuvPicture::new(64, 32, 0, 8).unwrap();
            for (i, v) in pic.y.iter_mut().enumerate() {
                *v = (i % 200) as u16;
            }
            pic
        };
        let mut pic_map = make();
        let mut pic_whole = make();
        let mut alf = AlfData {
            luma_filter_signal: true,
            chroma_filter_signal: false,
            num_luma_filters: 1,
            ..AlfData::default()
        };
        // A non-trivial filter so the two paths actually do work.
        alf.luma_filters[0].coef[0] = 4;
        alf.luma_filters[0].coef[12] = 60;

        let mut map = AlfCtbMap::new(64, 32, 5);
        for i in 0..map.luma.len() {
            map.set(i, true, false, false);
        }
        apply_alf_with_map(&mut pic_map, &alf, &map, false, false, 8, 8);
        apply_alf(&mut pic_whole, &alf, 8, 8);
        assert_eq!(pic_map.y, pic_whole.y, "masked-all-on == whole-plane");
    }

    #[test]
    fn apply_alf_with_map_chroma_gated_by_slice_enable() {
        // ChromaArrayType 1 (4:2:0): the per-CTB chroma map flags are
        // inferred 0, so chroma is filtered as a whole plane only when the
        // slice-level enable is passed. Cb enabled, Cr disabled.
        let mut pic = crate::picture::YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.cb.iter_mut() {
            *v = 128;
        }
        for v in pic.cr.iter_mut() {
            *v = 128;
        }
        let mut alf = AlfData {
            luma_filter_signal: false,
            chroma_filter_signal: true,
            num_chroma_alts: 1,
            ..AlfData::default()
        };
        // Spec eq. 1286-style chroma: coef[6] = 4, V = 128 →
        // (4 * 128 + 256) >> 9 = 768 >> 9 = 1 for a uniform plane.
        alf.chroma_filters[0].coef = [0, 0, 0, 0, 0, 0, 4];
        let map = AlfCtbMap::new(16, 16, 5);
        apply_alf_with_map(&mut pic, &alf, &map, true, false, 8, 8);
        assert!(pic.cb.iter().all(|&v| v == 1), "Cb filtered (enabled)");
        assert!(pic.cr.iter().all(|&v| v == 128), "Cr untouched (disabled)");
    }

    // =================================================================
    // Round 117: §8.8.4.3 ALF transpose + classification (eq. 1289-1320).
    // =================================================================

    /// Reference re-derivation of §8.8.4.3 used to cross-check the SUT on
    /// arbitrary input. Independently coded from the spec equations with a
    /// straightforward (non-windowed-prefix) structure so a transcription
    /// typo in the SUT would not be mirrored here.
    #[allow(clippy::too_many_arguments)]
    fn classify_ref(
        src: &[u16],
        stride: usize,
        w: usize,
        h: usize,
        x_ctb: usize,
        y_ctb: usize,
        blk_w: usize,
        blk_h: usize,
        bd: u32,
    ) -> (Vec<u8>, Vec<u8>) {
        let samp = |x: i32, y: i32| -> i32 {
            let xc = x.clamp(0, w as i32 - 1) as usize;
            let yc = y.clamp(0, h as i32 - 1) as usize;
            src[yc * stride + xc] as i32
        };
        let grad = |gx: i32, gy: i32| -> (i32, i32, i32, i32) {
            let px = x_ctb as i32 + gx;
            let py = y_ctb as i32 + gy;
            let c = samp(px, py) << 1;
            let fh = (c - samp(px - 1, py) - samp(px + 1, py)).abs();
            let fv = (c - samp(px, py - 1) - samp(px, py + 1)).abs();
            let fd0 = (c - samp(px - 1, py - 1) - samp(px + 1, py + 1)).abs();
            let fd1 = (c - samp(px + 1, py - 1) - samp(px - 1, py + 1)).abs();
            (fh, fv, fd0, fd1)
        };
        let mut fi = vec![0u8; blk_w * blk_h];
        let mut ti = vec![0u8; blk_w * blk_h];
        for y in 0..blk_h {
            for x in 0..blk_w {
                let (sx, sy) = (x >> 2, y >> 2);
                let (mut sh, mut sv, mut sd0, mut sd1) = (0, 0, 0, 0);
                for j in -2i32..=5 {
                    for i in -2i32..=5 {
                        let (a, b, cc, dd) = grad((sx as i32) * 4 + i, (sy as i32) * 4 + j);
                        sh += a;
                        sv += b;
                        sd0 += cc;
                        sd1 += dd;
                    }
                }
                let (hv1, hv0, dir_hv) = if sv > sh { (sv, sh, 1) } else { (sh, sv, 3) };
                let (d1, d0, dir_d) = if sd0 > sd1 {
                    (sd0, sd1, 0)
                } else {
                    (sd1, sd0, 2)
                };
                let diag = (d1 as i64) * (hv0 as i64) > (hv1 as i64) * (d0 as i64);
                let (hvd1, hvd0) = if diag { (d1, d0) } else { (hv1, hv0) };
                let dir1: i32 = if diag { dir_d } else { dir_hv };
                let dir2: i32 = if diag { dir_hv } else { dir_d };
                let dir_s = if hvd1 > 2 * hvd0 {
                    1
                } else if (hvd1 as i64) * 2 > 9 * (hvd0 as i64) {
                    2
                } else {
                    0
                };
                let var_tab = [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4];
                let var_in = ((sh + sv) >> (bd as i32 - 2)).clamp(0, 15) as usize;
                let avg_var = var_tab[var_in];
                let tt = [0, 1, 0, 2, 2, 3, 1, 3];
                let t = tt[(dir1 * 2 + (dir2 >> 1)) as usize];
                let mut f = avg_var;
                if dir_s != 0 {
                    f += (((dir1 & 0x1) << 1) + dir_s) * 5;
                }
                fi[y * blk_w + x] = f as u8;
                ti[y * blk_w + x] = t as u8;
            }
        }
        (fi, ti)
    }

    #[test]
    fn classification_flat_block_is_class_zero() {
        // A perfectly flat plane has zero gradients everywhere → sumOfHV = 0
        // → avgVar = varTab[0] = 0 and dirS = 0, so filtIdx = 0 for every
        // sample (eq. 1319, no eq. 1320 offset).
        //
        // transposeIdx is *not* 0 in this degenerate case: with all sums
        // equal to 0 the `s_v > s_h` / `s_d0 > s_d1` comparisons both take
        // the else branch, giving dir1 = dirHV = 3 and dir2 = dirD = 2, so
        // transposeTable[3*2 + (2>>1)] = transposeTable[7] = 3 (eq.
        // 1317-1318). The transpose is irrelevant when every tap pair reads
        // the same value, but the spec still defines it as 3 here.
        let w = 16;
        let h = 16;
        let src = vec![100u16; w * h];
        let cls = derive_alf_classification(&src, w, w, h, 0, 0, 8, 8, 8);
        assert_eq!(cls.blk_width, 8);
        assert_eq!(cls.blk_height, 8);
        assert!(cls.filt_idx.iter().all(|&v| v == 0), "flat → class 0");
        assert!(
            cls.transpose_idx.iter().all(|&v| v == 3),
            "flat → transpose 3 (spec eq. 1317-1318 degenerate case)"
        );
    }

    #[test]
    fn classification_subblock_resolution_is_4x4() {
        // filtIdx / transposeIdx are constant within each 4×4 subblock (they
        // index the per-subblock sums via x>>2, y>>2). Build a vertical-edge
        // pattern and assert the four samples of the top-left 4×4 share a
        // class.
        let w = 32;
        let h = 32;
        let mut src = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = if x < 16 { 40 } else { 200 };
            }
        }
        let cls = derive_alf_classification(&src, w, w, h, 0, 0, 16, 16, 8);
        for sy in 0..4usize {
            for sx in 0..4usize {
                let base = cls.filt_idx_at(sx * 4, sy * 4);
                let baset = cls.transpose_idx_at(sx * 4, sy * 4);
                for dy in 0..4 {
                    for dx in 0..4 {
                        assert_eq!(cls.filt_idx_at(sx * 4 + dx, sy * 4 + dy), base);
                        assert_eq!(cls.transpose_idx_at(sx * 4 + dx, sy * 4 + dy), baset);
                    }
                }
            }
        }
    }

    #[test]
    fn classification_matches_independent_reference_on_pseudo_random() {
        // Deterministic LCG-filled plane; cross-check every sample against
        // the independent reference re-derivation.
        let w = 40;
        let h = 36;
        let mut state: u32 = 0x1234_5678;
        let mut src = vec![0u16; w * h];
        for v in src.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (state >> 16) as u16;
        }
        // Two CTBs: a full 32×32 at (0,0) and an 8×4 edge block at (32,32).
        for &(xc, yc, bw, bh) in &[(0usize, 0usize, 32usize, 32usize), (32, 32, 8, 4)] {
            let cls = derive_alf_classification(&src, w, w, h, xc, yc, bw, bh, 8);
            let (fi, ti) = classify_ref(&src, w, w, h, xc, yc, bw, bh, 8);
            assert_eq!(cls.filt_idx, fi, "filtIdx mismatch at ctb ({xc},{yc})");
            assert_eq!(cls.transpose_idx, ti, "transposeIdx mismatch ({xc},{yc})");
        }
    }

    #[test]
    fn classification_horizontal_edge_picks_vertical_direction() {
        // A horizontal edge (rows change, columns flat) makes vertical
        // activity dominate → dirHV branch gives dir = 1, dirS != 0, so
        // filtIdx is bumped by the eq.1320 offset and is non-zero.
        let w = 32;
        let h = 32;
        let mut src = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = if y < 16 { 30 } else { 220 };
            }
        }
        let cls = derive_alf_classification(&src, w, w, h, 0, 0, 16, 16, 8);
        // Sample on the strongest-activity row near the edge (y=15).
        let f = cls.filt_idx_at(0, 12);
        assert!(
            f >= 5,
            "edge sample should land in a directional class: {f}"
        );
        // filtIdx is always within range.
        assert!(cls.filt_idx.iter().all(|&v| (v as usize) < NUM_ALF_FILTERS));
        assert!(cls.transpose_idx.iter().all(|&v| v <= 3));
    }

    #[test]
    fn classification_filtidx_always_in_range() {
        // Adversarial high-contrast checkerboard: ensure no filtIdx escapes
        // 0..24 and no transposeIdx escapes 0..3 (eq. 1320 offset bound).
        let w = 24;
        let h = 24;
        let mut src = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = if (x + y) % 2 == 0 { 0 } else { 255 };
            }
        }
        let cls = derive_alf_classification(&src, w, w, h, 0, 0, 16, 16, 8);
        assert!(cls.filt_idx.iter().all(|&v| (v as usize) < NUM_ALF_FILTERS));
        assert!(cls.transpose_idx.iter().all(|&v| v <= 3));
    }

    #[test]
    fn transpose_coeffs_identity_for_zero() {
        let f: [i16; ALF_LUMA_NUM_COEF] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(transpose_luma_coeffs(&f, 0), f);
        // Out-of-range transpose index also falls through to identity.
        assert_eq!(transpose_luma_coeffs(&f, 9), f);
    }

    #[test]
    fn transpose_coeffs_match_spec_permutations() {
        let f: [i16; ALF_LUMA_NUM_COEF] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        // eq. 1282.
        assert_eq!(
            transpose_luma_coeffs(&f, 1),
            [9, 4, 10, 8, 1, 5, 11, 7, 3, 0, 2, 6, 12]
        );
        // eq. 1283.
        assert_eq!(
            transpose_luma_coeffs(&f, 2),
            [0, 3, 2, 1, 8, 7, 6, 5, 4, 9, 10, 11, 12]
        );
        // eq. 1284.
        assert_eq!(
            transpose_luma_coeffs(&f, 3),
            [9, 8, 10, 4, 3, 7, 11, 5, 1, 0, 2, 6, 12]
        );
    }

    #[test]
    fn transpose_centre_and_dc_are_invariant() {
        // Position 12 (centre/DC) is fixed under every transpose (eq.
        // 1282-1285 all keep f[12] last).
        let f: [i16; ALF_LUMA_NUM_COEF] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77];
        for t in 0..4 {
            assert_eq!(transpose_luma_coeffs(&f, t)[12], 77);
        }
    }

    #[test]
    fn classification_bit_depth_shift_scales_activity() {
        // eq. 1316 shifts sumOfHV by (BitDepthY − 2). At 10-bit the same
        // sample magnitudes shift one extra bit, so a moderate-activity
        // block classifies lower than at 8-bit (or equal, never higher).
        let w = 32;
        let h = 32;
        let mut src = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                // Gentle ramp → small but non-zero activity.
                src[y * w + x] = ((x + y) as u32 % 64 + 64) as u16;
            }
        }
        let c8 = derive_alf_classification(&src, w, w, h, 0, 0, 16, 16, 8);
        let c10 = derive_alf_classification(&src, w, w, h, 0, 0, 16, 16, 10);
        // The varTab input is non-increasing as the shift grows, so the
        // avgVar component of every sample's class at 10-bit is <= the 8-bit
        // one (transpose / direction-offset are bit-depth-independent, so we
        // compare the dirS == 0 samples where filtIdx == avgVar directly).
        for i in 0..c8.filt_idx.len() {
            if c8.transpose_idx[i] == c10.transpose_idx[i]
                && c8.filt_idx[i] < 5
                && c10.filt_idx[i] < 5
            {
                assert!(
                    c10.filt_idx[i] <= c8.filt_idx[i],
                    "10-bit class {} should not exceed 8-bit class {} at sample {i}",
                    c10.filt_idx[i],
                    c8.filt_idx[i]
                );
            }
        }
    }

    // =================================================================
    // Round 120: §7.3.5 spec-faithful parser + §8.9.4 AlfCoeffL +
    // §8.8.4.2 classified apply wiring.
    // =================================================================

    #[test]
    fn ceil_log2_boundary_values() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(25), 5);
    }

    #[test]
    fn round120_parse_uses_alf_luma_type_flag_seven_tap() {
        // alf_luma_type_flag == 0 → NumAlfCoefs = 7, only 6 deltas signalled
        // and coefPosMap = COEF_POS_MAP_7. With one signalled class, no
        // fixed filter, all deltas 0: every AlfCoeffL[ filtIdx ] is all-zero
        // spatial taps with DC = 512.
        let payload = emit_spec_alf_data(
            true,
            false, // type_flag = 0 → 7-tap
            1,
            &[vec![0i32; 12]],
            false,
            &[0; 6],
            0,
            [0; 2],
            0,
            &[0, 0], // LumaMaxGolombIdx = 2
        );
        let alf = parse_alf_data(&payload).unwrap();
        assert!(!alf.luma_type_flag);
        for filt_idx in 0..NUM_ALF_FILTERS {
            for j in 0..12 {
                assert_eq!(alf.luma_filters[filt_idx].coef[j], 0);
            }
            assert_eq!(alf.luma_filters[filt_idx].coef[12], 512);
        }
    }

    #[test]
    fn round120_alf_luma_coeff_delta_idx_routes_classes_to_filters() {
        // 2 signalled filters, alf_luma_coeff_delta_idx identity-mapped
        // i.e. class 0 → filter 0, class 1 → filter 1, classes 2..24 → filter
        // 1 (clamped). Filter 0 has tap 0 = 7; filter 1 has tap 0 = -3.
        let mut taps0 = vec![0i32; 12];
        taps0[0] = 7;
        let mut taps1 = vec![0i32; 12];
        taps1[0] = -3;
        let payload = emit_spec_alf_data(
            true,
            true,
            2,
            &[taps0, taps1],
            false,
            &[0; 6],
            0,
            [0; 2],
            0,
            &[0, 0, 0],
        );
        let alf = parse_alf_data(&payload).unwrap();
        assert_eq!(alf.num_signalled_luma_filters, 2);
        assert_eq!(alf.alf_luma_coeff_delta_idx[0], 0);
        assert_eq!(alf.alf_luma_coeff_delta_idx[1], 1);
        // Identity mapping in the emitter clamps to (num_signalled - 1) = 1
        // for filt_idx >= 2.
        assert_eq!(alf.alf_luma_coeff_delta_idx[2], 1);
        // Class 0 → filter 0: AlfCoeffL[0][0] = 7, DC = 512 − 14 = 498.
        assert_eq!(alf.luma_filters[0].coef[0], 7);
        assert_eq!(alf.luma_filters[0].coef[12], 498);
        // Class 1 → filter 1: AlfCoeffL[1][0] = -3, DC = 512 − (−6) = 518.
        assert_eq!(alf.luma_filters[1].coef[0], -3);
        assert_eq!(alf.luma_filters[1].coef[12], 518);
    }

    #[test]
    fn round120_fixed_filter_pattern1_seeds_every_class() {
        // alf_luma_fixed_filter_usage_pattern = 1 means every class uses
        // the fixed filter. With set_idx = 0 for every class and no
        // signalled deltas, AlfCoeffL[ filtIdx ][ j ] equals
        // ALF_FIX_FILT_COEFF[ AlfClassToFiltMap[ filtIdx ][ 0 ] ][ j ].
        let mut e = BitEmitter::new();
        e.u(1, 1); // luma_signal
        e.u(1, 0); // no chroma
        e.ue(0); // num_signalled_minus1 = 0 → 1 filter
        e.u(1, 1); // type_flag = 1
                   // num_signalled == 1, so alf_luma_coeff_delta_idx loop is skipped.
        e.uek(0, 1); // fixed_filter_usage_pattern = 1
                     // Every class flagged on automatically; need set_idx per class.
        for _ in 0..25 {
            e.u(4, 0);
        }
        e.u(1, 0); // coeff_delta_flag = 0
                   // num_signalled == 1 ⇒ no coeff_delta_prediction_flag.
        e.ue(0); // alf_luma_min_eg_order_minus1 = 0 ⇒ kMin = 1
        for _ in 0..3 {
            e.u(1, 0);
        }
        // coeff_flag[0] inferred 1; 12 abs values, all 0 ⇒ uek(1) zero = "10".
        for _ in 0..12 {
            // uek(k=1) of 0 is "1" + 1-bit suffix "0" = bits "10". No sign.
            e.u(2, 0b10);
        }
        e.finish_with_trailing_bits();
        let alf = parse_alf_data(&e.into_bytes()).unwrap();
        for (filt_idx, class_row) in ALF_CLASS_TO_FILT_MAP
            .iter()
            .enumerate()
            .take(NUM_ALF_FILTERS)
        {
            assert!(alf.alf_luma_fixed_filter_usage_flag[filt_idx]);
            assert_eq!(alf.alf_luma_fixed_filter_set_idx[filt_idx], 0);
            // Eq. 98: outCoef[ filtIdx ][ j ] = ALF_FIX_FILT_COEFF[ map[filtIdx][0] ][ j ].
            let fix_row = class_row[0] as usize;
            for (j, &expected) in ALF_FIX_FILT_COEFF[fix_row].iter().enumerate().take(12) {
                assert_eq!(
                    alf.luma_filters[filt_idx].coef[j], expected,
                    "class {filt_idx} tap {j}"
                );
            }
            // Eq. 104 DC.
            let sum2: i32 = (0..12)
                .map(|j| (alf.luma_filters[filt_idx].coef[j] as i32) << 1)
                .sum();
            assert_eq!(alf.luma_filters[filt_idx].coef[12] as i32, 512 - sum2);
        }
    }

    #[test]
    fn round120_coeff_delta_prediction_chains_deltas_across_filters() {
        // 3 signalled filters, with delta_prediction_flag = 1 (cumulative).
        // emit_spec_alf_data hard-codes prediction_flag = 0, so we build the
        // payload by hand.
        let mut e = BitEmitter::new();
        e.u(1, 1); // luma_signal
        e.u(1, 0); // no chroma
        e.ue(2); // num_signalled - 1 = 2 → 3 filters
        e.u(1, 1); // type_flag = 1
                   // alf_luma_coeff_delta_idx[ 0..24 ] — Ceil(Log2(3)) = 2 bits, identity-clamped.
        for i in 0..25 {
            e.u(2, (i as u32).min(2));
        }
        e.uek(0, 0); // fixed_filter_usage_pattern = 0
        e.u(1, 0); // coeff_delta_flag = 0
        e.u(1, 1); // coeff_delta_prediction_flag = 1
        e.ue(0); // alf_luma_min_eg_order_minus1 = 0 ⇒ kMin = 1
        for _ in 0..3 {
            e.u(1, 0); // expGoOrder all kMin = 1
        }
        // 3 filters × 12 taps each — set tap 0 to (5, 0, -2) per-filter and rest = 0.
        // After eq. 97 cumulative: filter 0 = 5, filter 1 = 5+0 = 5, filter 2 = 5+(-2) = 3.
        // Each value via uek(1): 5 → bits "001110" (zeros=2, suffix 3 bits "110")
        //   wait reader: zeros=2, k=1, total_suffix=3, value = ((1<<2)-1)<<1 + suffix = 6+suffix.
        //   so 5 needs base + suffix = ((1<<1)-1)<<1 = 2 with suffix at zeros=1: 5 = 2 + suffix(suffix bits=2) → suffix=3.
        //   bits: "0" "1" "11" = "0111".
        // 0 via uek(1): "1" "0" = "10". -2 via abs=2 + sign 0:
        //   2 = ((1<<1)-1)<<1 + suffix? with zeros=1: 2 + suffix(2 bits), suffix=0 → "0" "1" "00" = "0100".
        // Use helper instead.
        let emit_uek = |e: &mut BitEmitter, k: u32, v: u32| e.uek(k, v);
        for filter_idx in 0..3 {
            let tap0 = match filter_idx {
                0 => 5i32,
                1 => 0i32,
                _ => -2i32,
            };
            let abs_val = tap0.unsigned_abs();
            emit_uek(&mut e, 1, abs_val);
            if abs_val != 0 {
                e.u(1, if tap0 >= 0 { 1 } else { 0 });
            }
            for _ in 1..12 {
                emit_uek(&mut e, 1, 0);
            }
        }
        e.finish_with_trailing_bits();
        let alf = parse_alf_data(&e.into_bytes()).unwrap();
        // After eq. 97 cumulative-sum prediction:
        //   filter 0 unchanged = 5
        //   filter 1 += filter 0 = 0 + 5 = 5
        //   filter 2 += filter 1 = -2 + 5 = 3
        // Then AlfCoeffL[ filt_idx ][ 0 ] takes the routed filter:
        //   filt 0 → filter 0 (tap0=5);
        //   filt 1 → filter 1 (tap0=5);
        //   filt 2..24 → filter 2 (tap0=3, clamped at emit-time).
        assert_eq!(alf.luma_filters[0].coef[0], 5);
        assert_eq!(alf.luma_filters[1].coef[0], 5);
        assert_eq!(alf.luma_filters[2].coef[0], 3);
    }

    #[test]
    fn round120_derive_alf_coeff_l_eq104_dc_property() {
        // Eq. 104 invariant: for every parsed AlfData, AlfCoeffL[ ][ 12 ]
        // must equal 512 − 2 × Σ AlfCoeffL[ ][ 0..11 ]. Verify on a
        // randomly-chosen multi-class config.
        let mut taps = vec![0i32; 12];
        taps[3] = 8;
        taps[7] = -4;
        taps[11] = 2;
        let payload = emit_spec_alf_data(
            true,
            true,
            1,
            &[taps],
            false,
            &[0; 6],
            0,
            [0; 2],
            0,
            &[0, 0, 0],
        );
        let alf = parse_alf_data(&payload).unwrap();
        for filt_idx in 0..NUM_ALF_FILTERS {
            let sum2: i32 = (0..12)
                .map(|j| (alf.luma_filters[filt_idx].coef[j] as i32) << 1)
                .sum();
            assert_eq!(alf.luma_filters[filt_idx].coef[12] as i32, 512 - sum2);
        }
    }

    #[test]
    fn round120_chroma_eg_chain_progresses() {
        // The chroma path's expGoOrderC chain starts at kMin = min+1 and
        // monotonically increases per the increase flag. With min = 1 and
        // increments [1, 0]: kMin → 2, then exp[0] = 2+1 = 3, exp[1] = 3+0 = 3.
        // For chroma coeffs all 0, parsing must succeed.
        let payload = emit_spec_alf_data(false, true, 0, &[], true, &[0; 6], 1, [1, 0], 0, &[]);
        let alf = parse_alf_data(&payload).unwrap();
        assert!(alf.chroma_filter_signal);
        for j in 0..6 {
            assert_eq!(alf.chroma_filters[0].coef[j], 0);
        }
        assert_eq!(alf.chroma_filters[0].coef[6], 512);
    }

    #[test]
    fn round120_apply_alf_luma_classified_uniform_plane() {
        // On a flat plane every sample classifies the same way (filtIdx = 0,
        // transposeIdx = 3 — see classification_flat_block_is_class_zero).
        // Build an AlfData with AlfCoeffL[0] = identity-DC and verify
        // apply_alf_luma_classified produces the spec identity.
        let mut pic = crate::picture::YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let mut alf = AlfData {
            luma_filter_signal: true,
            num_luma_filters: NUM_ALF_FILTERS,
            ..AlfData::default()
        };
        for filt_idx in 0..NUM_ALF_FILTERS {
            alf.luma_filters[filt_idx].coef = [0; 13];
            alf.luma_filters[filt_idx].coef[12] = 512;
        }
        let cls = derive_alf_classification(&pic.y, pic.y_stride(), 16, 16, 0, 0, 16, 16, 8);
        apply_alf_luma_classified(&mut pic, &alf, &cls, 0, 0, 8);
        for v in pic.y.iter() {
            assert_eq!(*v, 100, "(512 * 100 + 256) >> 9 = 100");
        }
    }

    #[test]
    fn round120_apply_alf_luma_classified_routes_per_class() {
        // Build a vertical-edge picture so different rows / columns will
        // classify into different filtIdx values. Assign one AlfCoeffL row
        // with a non-trivial DC and verify samples in that class change
        // while samples in other classes use their own (here all-zero) row.
        let w = 32usize;
        let h = 32usize;
        let mut pic = crate::picture::YuvPicture::new(w as u32, h as u32, 0, 8).unwrap();
        let stride = pic.y_stride();
        for y in 0..h {
            for x in 0..w {
                pic.y[y * stride + x] = if x < 16 { 60 } else { 180 };
            }
        }
        let cls = derive_alf_classification(&pic.y, pic.y_stride(), w, h, 0, 0, w, h, 8);
        // Pick the class assigned at (0, 0) and give it a non-zero DC.
        let chosen_class = cls.filt_idx_at(0, 0) as usize;
        let mut alf = AlfData {
            luma_filter_signal: true,
            num_luma_filters: NUM_ALF_FILTERS,
            ..AlfData::default()
        };
        for filt_idx in 0..NUM_ALF_FILTERS {
            alf.luma_filters[filt_idx].coef = [0; 13];
        }
        // Identity-DC for the chosen class; every other class stays at 0.
        alf.luma_filters[chosen_class].coef[12] = 512;
        let pre = pic.y.clone();
        apply_alf_luma_classified(&mut pic, &alf, &cls, 0, 0, 8);
        // Samples in the chosen class are unchanged (identity DC).
        // Samples in other classes go to 0 (zero filter).
        let mut at_least_one_chosen = false;
        let mut at_least_one_other = false;
        for y in 0..h {
            for x in 0..w {
                let cur = cls.filt_idx_at(x, y) as usize;
                if cur == chosen_class {
                    assert_eq!(pic.y[y * stride + x], pre[y * stride + x]);
                    at_least_one_chosen = true;
                } else {
                    assert_eq!(pic.y[y * stride + x], 0);
                    at_least_one_other = true;
                }
            }
        }
        assert!(
            at_least_one_chosen && at_least_one_other,
            "vertical edge should split classes"
        );
    }

    #[test]
    fn round120_apply_alf_luma_classified_masked_respects_ctb_flag() {
        // 64×32 picture, two 32×32 CTBs. Only the left CTB has alf_ctb_flag = 1.
        // The classified+masked apply must filter the left CTB and leave the
        // right one untouched. Use an identity-DC AlfCoeffL for every class so
        // the filtered output equals the input on a flat plane.
        let mut pic = crate::picture::YuvPicture::new(64, 32, 0, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let mut map = AlfCtbMap::new(64, 32, 5);
        map.set(0, true, false, false);
        map.set(1, false, false, false);
        let mut alf = AlfData {
            luma_filter_signal: true,
            num_luma_filters: NUM_ALF_FILTERS,
            ..AlfData::default()
        };
        for filt_idx in 0..NUM_ALF_FILTERS {
            alf.luma_filters[filt_idx].coef = [0; 13];
            alf.luma_filters[filt_idx].coef[12] = 512;
        }
        let pre = pic.y.clone();
        apply_alf_luma_classified_masked(&mut pic, &alf, &map, 8);
        let stride = pic.y_stride();
        // The flat plane reproduces under the identity-DC filter, so values
        // are unchanged even where the filter ran. To make a behavioural
        // assertion: switch one CTB to a zero filter and verify it nukes
        // the left CTB while the right one stays at 100.
        for filt_idx in 0..NUM_ALF_FILTERS {
            alf.luma_filters[filt_idx].coef[12] = 0;
        }
        // Restore the pre-filter plane (since the identity run didn't move it).
        pic.y.copy_from_slice(&pre);
        apply_alf_luma_classified_masked(&mut pic, &alf, &map, 8);
        for row in 0..32usize {
            for col in 0..32usize {
                assert_eq!(pic.y[row * stride + col], 0, "left CTB zeroed");
            }
            for col in 32..64usize {
                assert_eq!(pic.y[row * stride + col], 100, "right CTB unchanged");
            }
        }
    }

    #[test]
    fn round120_apply_alf_luma_classified_clamps_partial_edge_ctb() {
        // 40×40 picture, 2×2 CTBs at CtbLog2SizeY = 5. Top-left CTB is full
        // 32×32, bottom-right is 8×8 (partial edge). Flag only the partial
        // CTB. The classified+masked apply must only touch the 8×8 in-pic
        // region.
        let mut pic = crate::picture::YuvPicture::new(40, 40, 0, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 77;
        }
        let mut map = AlfCtbMap::new(40, 40, 5);
        map.set(3, true, false, false);
        let mut alf = AlfData {
            luma_filter_signal: true,
            num_luma_filters: NUM_ALF_FILTERS,
            ..AlfData::default()
        };
        // Every class → zero filter so the affected region becomes 0.
        for filt_idx in 0..NUM_ALF_FILTERS {
            alf.luma_filters[filt_idx].coef = [0; 13];
        }
        apply_alf_luma_classified_masked(&mut pic, &alf, &map, 8);
        let stride = pic.y_stride();
        for row in 0..40usize {
            for col in 0..40usize {
                let expected = if row >= 32 && col >= 32 { 0 } else { 77 };
                assert_eq!(pic.y[row * stride + col], expected);
            }
        }
    }

    #[test]
    fn round120_fixed_filter_table_dimensions_match_spec() {
        // Sanity check on the embedded spec tables.
        assert_eq!(ALF_FIX_FILT_COEFF.len(), 64);
        for row in ALF_FIX_FILT_COEFF.iter() {
            assert_eq!(row.len(), 12);
        }
        assert_eq!(ALF_CLASS_TO_FILT_MAP.len(), 25);
        for row in ALF_CLASS_TO_FILT_MAP.iter() {
            assert_eq!(row.len(), 16);
            for &v in row.iter() {
                assert!(v < 64);
            }
        }
        // Spot-check a couple of spec entries.
        assert_eq!(ALF_FIX_FILT_COEFF[0][0], 0);
        assert_eq!(ALF_FIX_FILT_COEFF[0][11], 30);
        assert_eq!(ALF_FIX_FILT_COEFF[63][6], 16);
        assert_eq!(ALF_CLASS_TO_FILT_MAP[0][15], 63);
        assert_eq!(ALF_CLASS_TO_FILT_MAP[24][0], 8);
    }

    // =================================================================
    // Round 145: §8.8.4.4 per-CTB chroma type filtering (eq. 1321-1323)
    //            + eq. 1321 tap-geometry verification.
    // =================================================================

    /// Compute the eq. 1321 weighted sum at chroma sample (x, y) with
    /// edge clamping over a plane of (cw, ch) dimensions. Independently
    /// transcribed from the spec equation; mirrors what
    /// [`apply_alf_chroma`] / [`apply_alf_chroma_masked`] should produce.
    fn eq_1321_ref(
        plane: &[u16],
        stride: usize,
        cw: usize,
        ch: usize,
        x: i32,
        y: i32,
        coef: &[i16; 7],
    ) -> i32 {
        let read = |xx: i32, yy: i32| -> i32 {
            let xc = xx.clamp(0, cw as i32 - 1) as usize;
            let yc = yy.clamp(0, ch as i32 - 1) as usize;
            plane[yc * stride + xc] as i32
        };
        let s = coef[0] as i32 * (read(x, y + 2) + read(x, y - 2))
            + coef[1] as i32 * (read(x + 1, y + 1) + read(x - 1, y - 1))
            + coef[2] as i32 * (read(x, y + 1) + read(x, y - 1))
            + coef[3] as i32 * (read(x - 1, y + 1) + read(x + 1, y - 1))
            + coef[4] as i32 * (read(x + 2, y) + read(x - 2, y))
            + coef[5] as i32 * (read(x + 1, y) + read(x - 1, y))
            + coef[6] as i32 * read(x, y);
        (s + 256) >> 9
    }

    #[test]
    fn round145_apply_alf_chroma_matches_eq_1321_on_synthetic_plane() {
        // Plant a deterministic gradient in Cb; verify apply_alf_chroma's
        // output matches an independent eq. 1321 evaluation sample-by-sample.
        // This catches the round-11 tap-geometry permutation that was fixed
        // in round 145 (coef[3] / coef[4] / coef[5] had been wired to the
        // wrong geometric positions).
        let mut pic = crate::picture::YuvPicture::new(16, 16, 1, 8).unwrap();
        let stride = pic.c_stride();
        let cw = stride;
        let ch = pic.cb.len() / stride;
        for y in 0..ch {
            for x in 0..cw {
                pic.cb[y * stride + x] = ((x * 7 + y * 11 + 50) % 200 + 20) as u16;
            }
        }
        // Non-trivial coefficients spread across all six spatial taps —
        // these exercise the eq. 1321 tap-position ordering.
        let coef: [i16; 7] = [5, -3, 4, -2, 6, -7, 64];
        let filter = AlfChromaFilter { coef };
        let pre = pic.cb.clone();
        apply_alf_chroma(&mut pic, &filter, 1, 8);
        // Independent re-derivation must match every sample.
        for y in 0..ch {
            for x in 0..cw {
                let want =
                    eq_1321_ref(&pre, stride, cw, ch, x as i32, y as i32, &coef).clamp(0, 255);
                assert_eq!(
                    pic.cb[y * stride + x] as i32,
                    want,
                    "eq.1321 mismatch at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn round145_apply_alf_chroma_masked_matches_whole_plane_when_all_ctus_on() {
        // 32×32 picture, four 16×16 CTBs (CtbLog2SizeY = 4). All four
        // chroma CTBs flagged → per-CTB apply must reproduce whole-plane
        // apply bit-for-bit for both Cb and Cr (eq. 1321-1323 + edge clamp).
        let mut pic_a = crate::picture::YuvPicture::new(32, 32, 3, 8).unwrap();
        let mut pic_b = pic_a.clone();
        let stride = pic_a.c_stride();
        let cw = stride;
        let ch = pic_a.cb.len() / stride;
        // Fill Cb / Cr with two different deterministic patterns.
        for y in 0..ch {
            for x in 0..cw {
                pic_a.cb[y * stride + x] = ((x * 3 + y * 5 + 40) % 220 + 10) as u16;
                pic_a.cr[y * stride + x] = ((x * 5 + y * 3 + 100) % 180 + 30) as u16;
            }
        }
        pic_b.cb.copy_from_slice(&pic_a.cb);
        pic_b.cr.copy_from_slice(&pic_a.cr);
        let filter = AlfChromaFilter {
            coef: [3, -2, 5, -1, 4, -3, 64],
        };
        // Map: all CTUs on for both chroma flags.
        let mut map = AlfCtbMap::new(32, 32, 4);
        for i in 0..(map.ctbs_wide * map.ctbs_high) as usize {
            map.set(i, true, true, true);
        }
        // pic_a uses the whole-plane apply, pic_b the per-CTB masked apply.
        apply_alf_chroma(&mut pic_a, &filter, 1, 8);
        apply_alf_chroma(&mut pic_a, &filter, 2, 8);
        apply_alf_chroma_masked(&mut pic_b, &filter, &map, 1, 8);
        apply_alf_chroma_masked(&mut pic_b, &filter, &map, 2, 8);
        assert_eq!(pic_a.cb, pic_b.cb, "Cb whole vs masked");
        assert_eq!(pic_a.cr, pic_b.cr, "Cr whole vs masked");
    }

    #[test]
    fn round145_apply_alf_chroma_masked_only_touches_flagged_ctus() {
        // 32×16 picture in 4:4:4 (so chroma matches luma dims), CtbLog2Size
        // = 4 → two 16×16 chroma CTBs. Flag only the left chroma_cb CTU and
        // verify only the left half of Cb moves, while Cr and the right
        // half of Cb stay at their initial values.
        let mut pic = crate::picture::YuvPicture::new(32, 16, 3, 8).unwrap();
        let stride = pic.c_stride();
        let cw = stride;
        let ch = pic.cb.len() / stride;
        assert_eq!(cw, 32);
        assert_eq!(ch, 16);
        for v in pic.cb.iter_mut() {
            *v = 80;
        }
        for v in pic.cr.iter_mut() {
            *v = 200;
        }
        let mut map = AlfCtbMap::new(32, 16, 4);
        // Left chroma CTU has Cb on, right chroma CTU has everything off.
        map.set(0, false, true, false);
        map.set(1, false, false, false);
        // Zero filter → output becomes (256 >> 9).clamp = 0 for flagged
        // samples; non-flagged samples stay at their input value.
        let filter = AlfChromaFilter {
            coef: [0, 0, 0, 0, 0, 0, 0],
        };
        apply_alf_chroma_masked(&mut pic, &filter, &map, 1, 8);
        // Cr is not gated by the Cb plane flags — call separately with
        // a map whose Cr flags are all off.
        apply_alf_chroma_masked(&mut pic, &filter, &map, 2, 8);
        for y in 0..ch {
            for x in 0..cw {
                let cb_expect = if x < 16 { 0 } else { 80 };
                assert_eq!(pic.cb[y * stride + x], cb_expect, "Cb ({x},{y})");
                // Cr map is all-off → Cr untouched everywhere.
                assert_eq!(pic.cr[y * stride + x], 200, "Cr ({x},{y})");
            }
        }
    }

    #[test]
    fn round145_apply_alf_chroma_masked_edge_clamp_partial_ctu() {
        // 24×24 picture in 4:2:0 → 12×12 chroma plane. CtbLog2SizeY = 4
        // (luma CTU = 16) → chroma CTU stride = 8. ctbs_wide = 2 (luma),
        // each chroma CTU = 8 wide → the right chroma CTU starts at x=8
        // and only spans 4 columns (clipped by cw=12). Verify the apply
        // doesn't write past column 11 and the unflagged right-edge area
        // outside the flagged CTU stays at the initial value.
        let mut pic = crate::picture::YuvPicture::new(24, 24, 1, 8).unwrap();
        let stride = pic.c_stride();
        let cw = stride;
        let ch = pic.cb.len() / stride;
        assert_eq!(cw, 12);
        assert_eq!(ch, 12);
        for v in pic.cb.iter_mut() {
            *v = 50;
        }
        // Flag only the right CTU (clipped to 4 chroma columns).
        let mut map = AlfCtbMap::new(24, 24, 4);
        // ctbs_wide = ceil(24/16) = 2, ctbs_high = 2 → 4 CTUs total.
        assert_eq!(map.ctbs_wide, 2);
        assert_eq!(map.ctbs_high, 2);
        map.set(0, false, false, false);
        map.set(1, false, true, false); // top-right Cb CTU
        map.set(2, false, false, false);
        map.set(3, false, true, false); // bottom-right Cb CTU
        let filter = AlfChromaFilter {
            coef: [0, 0, 0, 0, 0, 0, 0],
        };
        apply_alf_chroma_masked(&mut pic, &filter, &map, 1, 8);
        // Left columns 0..8 untouched (no flag), right columns 8..12 zeroed.
        for y in 0..ch {
            for x in 0..cw {
                let want = if x < 8 { 50 } else { 0 };
                assert_eq!(pic.cb[y * stride + x], want, "({x},{y})");
            }
        }
        // Sanity: no out-of-bounds write happened (sample at the last
        // column / row of the plane still in [0, 255]).
        assert_eq!(pic.cb[(ch - 1) * stride + (cw - 1)], 0);
    }

    #[test]
    fn round145_apply_alf_chroma_masked_skips_when_monochrome() {
        // chroma_format_idc = 0 → masked chroma apply is a no-op.
        let mut pic = crate::picture::YuvPicture::new(16, 16, 0, 8).unwrap();
        let filter = AlfChromaFilter {
            coef: [0, 0, 0, 0, 0, 0, 0],
        };
        let mut map = AlfCtbMap::new(16, 16, 4);
        map.set(0, false, true, true);
        // Empty cb/cr; the call must not panic on empty plane access.
        apply_alf_chroma_masked(&mut pic, &filter, &map, 1, 8);
        apply_alf_chroma_masked(&mut pic, &filter, &map, 2, 8);
        assert!(pic.cb.is_empty());
        assert!(pic.cr.is_empty());
    }
}
