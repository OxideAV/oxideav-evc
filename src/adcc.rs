//! EVC **ADCC** — advanced residual coding (ISO/IEC 23094-1:2020
//! §7.3.8.8 `residual_coding_adv()`, the `sps_adcc_flag == 1` residual
//! entropy layer that replaces the §7.3.8.7 run-length coding).
//!
//! The syntax walks the transform block backwards from the signalled
//! last significant position in 16-coefficient groups of the §6.5.2
//! zig-zag scan:
//!
//! 1. `last_sig_coeff_x_prefix` / `last_sig_coeff_y_prefix` — TR with
//!    `cMax = (log2TrafoSize << 1) − 1` (Tables 87/88, the §9.3.4.2.6
//!    eq. 1441 ctxInc); prefixes above 3 carry an FL **bypass** suffix
//!    (`cMax = (1 << ((prefix >> 1) − 1)) − 1`), composing per
//!    eqs. 149-152.
//! 2. Per group (`cgIdx` from `scanPosLast >> 4` down to 0):
//!    `sig_coeff_flag` for every scan position below the last
//!    (Table 89, the §9.3.4.2.7 eqs. 1442-1451 neighbour-stencil
//!    ctxInc), collecting the non-zero positions in reverse scan
//!    order;
//! 3. `coeff_abs_level_greaterA_flag` for the first
//!    `Min(numNZ, 8)` non-zero positions and one
//!    `coeff_abs_level_greaterB_flag` at the first greaterA position
//!    (Table 90 — shared A/B context space — with the §9.3.4.2.8/.9
//!    eqs. 1452-1465 stencils, ctxInc 0 at the last position);
//! 4. when escape data is present (a second greaterA, a greaterB, or
//!    `numNZ > 8`): `coeff_abs_level_remaining` per §9.3.3.8 — all
//!    **bypass**: a TR prefix over `cMax = numBinRem << cRiceParam`
//!    (Table 94 `numBinRem`, the §9.3.4.2.10 eqs. 1466-1471 +
//!    Table 98 Rice parameter) chained to a `k = cRiceParam + 1` EGk
//!    suffix, added onto the §7.3.8.8 `baseLevel` (2 + countFirstBCoef
//!    for the first 8 coefficients until a level ≥ 2 lands, 1 after);
//! 5. `coeff_signs_group` — one **bypass** bin per non-zero
//!    coefficient, MSB-first (its §7.4 semantics fix the group width
//!    at `numNZ`; the §7.3.8.8 `<< (32 − numNZ)` walk consumes the
//!    first-read bin for `blkPosArray[0]`).
//!
//! Under `sps_cm_init_flag == 0` every regular bin collapses to the
//! crate's Baseline `(0, 0)` slot (the walker-wide convention); under
//! `== 1` each element lands on its Table 39 table at
//! `ctxIdxOffset(initType) + ctxInc`.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::{Error, Result};

use crate::cabac::{BinSink, CabacEngine};
use crate::cabac_init::{
    ctx_inc_coeff_abs_level_greater_a, ctx_inc_coeff_abs_level_greater_b,
    ctx_inc_last_sig_coeff_prefix, ctx_inc_sig_coeff_flag, rice_param_coeff_abs_level_remaining,
    CtxSel, MainCtxTable,
};

/// Presence-gating tallies for one or more `residual_coding_adv()`
/// invocations (the round-391 fixture style: one counter per element).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdccStats {
    /// `last_sig_coeff_x_prefix` + `last_sig_coeff_y_prefix` regular bins.
    pub last_sig_prefix_bins: u32,
    /// `last_sig_coeff_x_suffix` + `last_sig_coeff_y_suffix` bypass bins.
    pub last_sig_suffix_bins: u32,
    /// `sig_coeff_flag` regular bins.
    pub sig_coeff_bins: u32,
    /// `coeff_abs_level_greaterA_flag` regular bins.
    pub greater_a_bins: u32,
    /// `coeff_abs_level_greaterB_flag` regular bins.
    pub greater_b_bins: u32,
    /// `coeff_abs_level_remaining` symbols decoded (bypass).
    pub remaining_syms: u32,
    /// `coeff_signs_group` bypass bins.
    pub sign_bins: u32,
    /// `residual_coding_adv()` invocations.
    pub blocks: u32,
}

/// Table 94 — `numBinRem` per `cRiceParam`.
const NUM_BIN_REM: [u32; 4] = [6, 5, 6, 3];

/// §9.3.4.2.7/.8/.9/.10 neighbour stencil over the (partially decoded)
/// `TransCoeffLevel` array: fold the five neighbours
/// `(xC+1, yC)`, `(xC+2, yC)`, `(xC+1, yC+1)`, `(xC, yC+1)`,
/// `(xC, yC+2)` (each guarded by the block bounds) through `f`.
fn stencil_sum(
    levels: &[i32],
    xc: u32,
    yc: u32,
    log2_tb_width: u32,
    log2_tb_height: u32,
    f: impl Fn(i32) -> u32,
) -> u32 {
    let w = 1u32 << log2_tb_width;
    let h = 1u32 << log2_tb_height;
    let at = |x: u32, y: u32| levels[(y as usize) << log2_tb_width | x as usize];
    let mut sum = 0u32;
    if xc < w - 1 {
        sum += f(at(xc + 1, yc));
    }
    if xc + 2 < w {
        sum += f(at(xc + 2, yc));
    }
    if xc < w - 1 && yc < h - 1 {
        sum += f(at(xc + 1, yc + 1));
    }
    if yc < h - 1 {
        sum += f(at(xc, yc + 1));
    }
    if yc + 2 < h {
        sum += f(at(xc, yc + 2));
    }
    sum
}

/// Read one `last_sig_coeff_{x,y}_prefix` (TR, Tables 87/88) plus its
/// optional FL bypass suffix, resolving eqs. 149-152 into the
/// coefficient coordinate.
#[allow(clippy::too_many_arguments)]
fn decode_last_sig_coord(
    eng: &mut CabacEngine,
    sel: CtxSel,
    table: MainCtxTable,
    c_idx: u32,
    chroma_array_type: u32,
    log2_trafo_size: u32,
    stats: &mut AdccStats,
) -> Result<u32> {
    let c_max = (log2_trafo_size << 1) - 1;
    let mut bins = 0u32;
    let prefix = if sel.cm_init {
        let off = table.ctx_idx_offset(sel.init_type);
        eng.decode_tr_regular(c_max, 0, table.as_usize(), |bin_idx| {
            bins += 1;
            off + ctx_inc_last_sig_coeff_prefix(bin_idx, c_idx, log2_trafo_size, chroma_array_type)
        })?
    } else {
        // Table 95, sps_cm_init_flag == 0 row: ctxInc = binIdx for luma,
        // 11 + binIdx for chroma, on the shared ctxTable 0 at the
        // element's Table-39 offset.
        let off = table.cm0_ctx_idx_offset(sel.init_type);
        eng.decode_tr_regular(c_max, 0, 0, |bin_idx| {
            bins += 1;
            off + if c_idx == 0 {
                bin_idx as usize
            } else {
                11 + bin_idx as usize
            }
        })?
    };
    stats.last_sig_prefix_bins += bins;
    if prefix > 3 {
        // eq. 150/152: FL bypass suffix of ((prefix >> 1) − 1) bins.
        let suffix_len = (prefix >> 1) - 1;
        let suffix_c_max = (1u32 << suffix_len) - 1;
        let suffix = eng.decode_fl_bypass(suffix_c_max)?;
        stats.last_sig_suffix_bins += suffix_len;
        Ok((1u32 << suffix_len) * (2 + (prefix & 1)) + suffix)
    } else {
        Ok(prefix)
    }
}

/// §9.3.3.8 — decode one `coeff_abs_level_remaining` (all bypass): the
/// TR prefix over `cMax = numBinRem << cRiceParam` chained to the
/// `k = cRiceParam + 1` EGk suffix.
fn decode_abs_level_remaining(eng: &mut CabacEngine, c_rice_param: u32) -> Result<u32> {
    let num_bin_rem = NUM_BIN_REM[c_rice_param.min(3) as usize];
    let c_max = num_bin_rem << c_rice_param;
    let mut prefix = 0u32;
    while prefix < num_bin_rem {
        if eng.decode_bypass()? == 0 {
            break;
        }
        prefix += 1;
    }
    if prefix < num_bin_rem {
        // TR suffix: FL of cRiceParam bypass bits.
        let mut suffix = 0u32;
        for _ in 0..c_rice_param {
            suffix = (suffix << 1) | eng.decode_bypass()? as u32;
        }
        Ok((prefix << c_rice_param) + suffix)
    } else {
        // All-ones prefix → eq. 1433 EGk suffix with k = cRiceParam + 1.
        let suffix = eng.decode_egk_bypass(c_rice_param + 1)?;
        Ok(c_max + suffix)
    }
}

/// §7.3.8.8 `residual_coding_adv()` — decode one transform block's
/// coefficient levels into `levels` (row-major
/// `y << log2_tb_width | x`, length `1 << (log2W + log2H)`, caller
/// pre-zeroed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_residual_coding_adv(
    eng: &mut CabacEngine,
    sel: CtxSel,
    c_idx: u32,
    chroma_array_type: u32,
    levels: &mut [i32],
    stats: &mut AdccStats,
    log2_tb_width: u32,
    log2_tb_height: u32,
) -> Result<()> {
    let blk_w = 1usize << log2_tb_width;
    let blk_h = 1usize << log2_tb_height;
    let total = blk_w * blk_h;
    if levels.len() != total {
        return Err(Error::invalid(format!(
            "evc residual_coding_adv: levels len {} != {}*{} = {}",
            levels.len(),
            blk_w,
            blk_h,
            total
        )));
    }
    if total > (1 << 12) {
        return Err(Error::invalid(format!(
            "evc residual_coding_adv: block too large ({total} > 4096)"
        )));
    }
    stats.blocks += 1;

    // last_sig_coeff_{x,y}: prefix TR + FL bypass suffix (eqs. 149-152).
    let last_x = decode_last_sig_coord(
        eng,
        sel,
        MainCtxTable::LastSigCoeffXPrefix,
        c_idx,
        chroma_array_type,
        log2_tb_width,
        stats,
    )?;
    let last_y = decode_last_sig_coord(
        eng,
        sel,
        MainCtxTable::LastSigCoeffYPrefix,
        c_idx,
        chroma_array_type,
        log2_tb_height,
        stats,
    )?;
    if last_x as usize >= blk_w || last_y as usize >= blk_h {
        return Err(Error::invalid(format!(
            "evc residual_coding_adv: last position ({last_x}, {last_y}) outside {blk_w}x{blk_h}"
        )));
    }

    // ScanOrder / InvScanOrder (§6.5.2 zig-zag).
    let scan = crate::scan::zig_zag_scan(blk_w, blk_h);
    let raster_pos_last = (last_x as usize) + ((last_y as usize) << log2_tb_width);
    let scan_pos_last = scan
        .iter()
        .position(|&p| p as usize == raster_pos_last)
        .ok_or_else(|| Error::invalid("evc residual_coding_adv: InvScanOrder miss"))?;

    let last_coef_group = scan_pos_last >> 4;
    let mut i_pos = scan_pos_last as i64;
    for cg_idx in (0..=last_coef_group as i64).rev() {
        let sub_block_pos = cg_idx << 4;
        let mut escape_data_present = false;
        // (blkPos, xC, yC) of each non-zero coefficient in reverse scan
        // order — the §7.3.8.8 `blkPosArray`.
        let mut nz: Vec<(usize, u32, u32)> = Vec::with_capacity(16);
        while i_pos >= sub_block_pos {
            let blk_pos = scan[i_pos as usize] as usize;
            let xc = (blk_pos & (blk_w - 1)) as u32;
            let yc = (blk_pos >> log2_tb_width) as u32;
            let sig = if i_pos as usize != scan_pos_last {
                // Table 89 / §9.3.4.2.7 stencil over the already-decoded
                // significance map.
                // Table 95: the sps_cm_init_flag == 0 row is
                // `cIdx == 0 ? 0 : 1` (no stencil).
                let cm1_inc = if sel.cm_init {
                    let num_flags =
                        stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                            (v != 0) as u32
                        });
                    ctx_inc_sig_coeff_flag(c_idx, xc, yc, num_flags)
                } else {
                    0
                };
                let cm0_inc = if c_idx == 0 { 0 } else { 1 };
                let (t, i) = sel.ctx_shaped(MainCtxTable::SigCoeffFlag, cm1_inc, cm0_inc);
                let bin = eng.decode_decision(t, i)?;
                stats.sig_coeff_bins += 1;
                bin != 0
            } else {
                true // §7.4: inferred 1 at the last significant position
            };
            if sig {
                levels[blk_pos] = 1;
                nz.push((blk_pos, xc, yc));
            }
            i_pos -= 1;
        }
        let num_nz = nz.len();
        if num_nz == 0 {
            continue;
        }
        // coeff_abs_level_greaterA_flag for the first Min(numNZ, 8).
        let mut last_greater_a: Option<usize> = None;
        let num_c1 = num_nz.min(8);
        for (n, &(blk_pos, xc, yc)) in nz.iter().enumerate().take(num_c1) {
            let is_last = n == 0 && cg_idx as usize == last_coef_group;
            // Table 95: the sps_cm_init_flag == 0 row is
            // `cIdx == 0 ? 0 : 1`.
            let cm1_inc = if sel.cm_init {
                let num_flags = stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                    (v.unsigned_abs() > 1) as u32
                });
                ctx_inc_coeff_abs_level_greater_a(c_idx, xc, yc, is_last, num_flags)
            } else {
                0
            };
            let cm0_inc = if c_idx == 0 { 0 } else { 1 };
            let (t, i) = sel.ctx_shaped(MainCtxTable::CoeffAbsLevelGreaterFlag, cm1_inc, cm0_inc);
            let flag = eng.decode_decision(t, i)?;
            stats.greater_a_bins += 1;
            levels[blk_pos] += flag as i32;
            if flag != 0 {
                if last_greater_a.is_none() {
                    last_greater_a = Some(n);
                } else {
                    escape_data_present = true;
                }
            }
        }
        // coeff_abs_level_greaterB_flag at the first greaterA position.
        if let Some(n) = last_greater_a {
            let (blk_pos, xc, yc) = nz[n];
            let is_last = n == 0 && cg_idx as usize == last_coef_group;
            // Table 95: the sps_cm_init_flag == 0 row is
            // `cIdx == 0 ? 0 : 1`.
            let cm1_inc = if sel.cm_init {
                let num_flags = stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                    (v.unsigned_abs() > 2) as u32
                });
                ctx_inc_coeff_abs_level_greater_b(c_idx, xc, yc, is_last, num_flags)
            } else {
                0
            };
            let cm0_inc = if c_idx == 0 { 0 } else { 1 };
            let (t, i) = sel.ctx_shaped(MainCtxTable::CoeffAbsLevelGreaterFlag, cm1_inc, cm0_inc);
            let flag = eng.decode_decision(t, i)?;
            stats.greater_b_bins += 1;
            levels[blk_pos] += flag as i32;
            if flag != 0 {
                escape_data_present = true;
            }
        }
        let escape_data_present = escape_data_present || num_nz > 8;
        // coeff_abs_level_remaining (§9.3.3.8, all bypass).
        let mut count_first_b_coef = 1i32;
        if escape_data_present {
            for (n, &(blk_pos, xc, yc)) in nz.iter().enumerate() {
                let base_level = if n < 8 { 2 + count_first_b_coef } else { 1 };
                if levels[blk_pos] >= base_level {
                    // §9.3.4.2.10 Rice parameter over the current
                    // TransCoeffLevel neighbourhood (eqs. 1466-1471).
                    let loc_sum = stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                        v.unsigned_abs()
                    }) as i32;
                    let loc_sum_abs = (loc_sum - base_level * 5).clamp(0, 31) as u32;
                    let c_rice = rice_param_coeff_abs_level_remaining(loc_sum_abs);
                    let remaining = decode_abs_level_remaining(eng, c_rice)?;
                    stats.remaining_syms += 1;
                    levels[blk_pos] = (base_level + remaining as i32).min(32767);
                }
                if levels[blk_pos] >= 2 {
                    count_first_b_coef = 0;
                }
            }
        }
        // coeff_signs_group: one bypass bin per non-zero coefficient,
        // MSB-first (the §7.3.8.8 `<< (32 − numNZ)` + top-bit walk).
        for &(blk_pos, _, _) in nz.iter() {
            let sign = eng.decode_bypass()?;
            stats.sign_bins += 1;
            if sign != 0 {
                levels[blk_pos] = -levels[blk_pos];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Round 458 — the encoder's write side.
// ---------------------------------------------------------------------

/// The `last_sig_coeff_{x,y}` prefix / suffix pair of a coordinate
/// (eqs. 149-152 inverted): `v < 4` is the prefix alone; otherwise with
/// `g = ⌊log2 v⌋` the prefix is `2g + ((v >> (g − 1)) & 1)` and the
/// suffix the low `g − 1` bits.
fn last_sig_prefix_suffix(v: u32) -> (u32, u32, u32) {
    if v < 4 {
        return (v, 0, 0);
    }
    let g = 31 - v.leading_zeros();
    let suffix_len = g - 1;
    let prefix = 2 * g + ((v >> suffix_len) & 1);
    (prefix, v & ((1 << suffix_len) - 1), suffix_len)
}

/// Write one `last_sig_coeff_{x,y}_prefix` (TR, Tables 87/88 — the
/// prefix's `cMax` ones carry no terminator) plus its FL bypass suffix.
fn encode_last_sig_coord<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    table: MainCtxTable,
    c_idx: u32,
    chroma_array_type: u32,
    log2_trafo_size: u32,
    v: u32,
) {
    let c_max = (log2_trafo_size << 1) - 1;
    let (prefix, suffix, suffix_len) = last_sig_prefix_suffix(v);
    debug_assert!(prefix <= c_max);
    let (t, ctx_of): (usize, Box<dyn Fn(u32) -> usize>) = if sel.cm_init {
        let off = table.ctx_idx_offset(sel.init_type);
        (
            table.as_usize(),
            Box::new(move |b| {
                off + ctx_inc_last_sig_coeff_prefix(b, c_idx, log2_trafo_size, chroma_array_type)
            }),
        )
    } else {
        let off = table.cm0_ctx_idx_offset(sel.init_type);
        (
            0,
            Box::new(move |b| {
                off + if c_idx == 0 {
                    b as usize
                } else {
                    11 + b as usize
                }
            }),
        )
    };
    for b in 0..prefix {
        enc.encode_decision(t, ctx_of(b), 1);
    }
    if prefix < c_max {
        enc.encode_decision(t, ctx_of(prefix), 0);
    }
    for i in (0..suffix_len).rev() {
        enc.encode_bypass(((suffix >> i) & 1) as u8);
    }
}

/// §9.3.3.8 — write one `coeff_abs_level_remaining` (all bypass): the
/// TR prefix over `cMax = numBinRem << cRiceParam` chained to the
/// `k = cRiceParam + 1` EGk suffix — the dual of `decode_abs_level_remaining`.
fn encode_abs_level_remaining<S: BinSink>(enc: &mut S, c_rice_param: u32, value: u32) {
    let num_bin_rem = NUM_BIN_REM[c_rice_param.min(3) as usize];
    let c_max = num_bin_rem << c_rice_param;
    if value < c_max {
        let prefix = value >> c_rice_param;
        for _ in 0..prefix {
            enc.encode_bypass(1);
        }
        enc.encode_bypass(0);
        for i in (0..c_rice_param).rev() {
            enc.encode_bypass(((value >> i) & 1) as u8);
        }
    } else {
        for _ in 0..num_bin_rem {
            enc.encode_bypass(1);
        }
        encode_egk_bypass(enc, c_rice_param + 1, value - c_max);
    }
}

/// §9.3.3.4 EGk bypass writer — the dual of `CabacEngine::decode_egk_bypass`.
fn encode_egk_bypass<S: BinSink>(enc: &mut S, k_in: u32, mut value: u32) {
    let mut k = k_in;
    while value >= (1u32 << k) {
        enc.encode_bypass(1);
        value -= 1u32 << k;
        k += 1;
    }
    enc.encode_bypass(0);
    for i in (0..k).rev() {
        enc.encode_bypass(((value >> i) & 1) as u8);
    }
}

/// §7.3.8.8 `residual_coding_adv()` **writer** (round 458 encoder) —
/// the exact dual of [`decode_residual_coding_adv`]: the last
/// significant coordinate, then per 16-coefficient group in reverse
/// scan order the significance map, the greaterA / greaterB flags, the
/// escape remainders and the sign group. Every context is derived over
/// a mirror of the decoder's progressively filled `TransCoeffLevel`
/// array (`1` at a significant position until its greater flags land,
/// the full magnitude once the remainder does), so the stencils see
/// exactly what the reader sees. `levels` is row-major, non-empty in at
/// least one position (the caller signals `cbf = 1`).
#[doc(hidden)]
pub fn encode_residual_coding_adv<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    c_idx: u32,
    chroma_array_type: u32,
    levels: &[i32],
    log2_tb_width: u32,
    log2_tb_height: u32,
) {
    let blk_w = 1usize << log2_tb_width;
    let blk_h = 1usize << log2_tb_height;
    let total = blk_w * blk_h;
    debug_assert_eq!(levels.len(), total);
    let scan = crate::scan::zig_zag_scan(blk_w, blk_h);
    let scan_pos_last = scan
        .iter()
        .rposition(|&p| levels[p as usize] != 0)
        .expect("cbf set with all-zero levels");
    let raster_last = scan[scan_pos_last] as usize;
    let last_x = (raster_last & (blk_w - 1)) as u32;
    let last_y = (raster_last >> log2_tb_width) as u32;
    encode_last_sig_coord(
        enc,
        sel,
        MainCtxTable::LastSigCoeffXPrefix,
        c_idx,
        chroma_array_type,
        log2_tb_width,
        last_x,
    );
    encode_last_sig_coord(
        enc,
        sel,
        MainCtxTable::LastSigCoeffYPrefix,
        c_idx,
        chroma_array_type,
        log2_tb_height,
        last_y,
    );

    // The decoder's view of TransCoeffLevel as it fills in.
    let mut dv = vec![0i32; total];
    let last_coef_group = scan_pos_last >> 4;
    let mut i_pos = scan_pos_last as i64;
    for cg_idx in (0..=last_coef_group as i64).rev() {
        let sub_block_pos = cg_idx << 4;
        let mut escape_data_present = false;
        let mut nz: Vec<(usize, u32, u32)> = Vec::with_capacity(16);
        while i_pos >= sub_block_pos {
            let blk_pos = scan[i_pos as usize] as usize;
            let xc = (blk_pos & (blk_w - 1)) as u32;
            let yc = (blk_pos >> log2_tb_width) as u32;
            let sig = levels[blk_pos] != 0;
            if i_pos as usize != scan_pos_last {
                let cm1_inc = if sel.cm_init {
                    let num_flags = stencil_sum(&dv, xc, yc, log2_tb_width, log2_tb_height, |v| {
                        (v != 0) as u32
                    });
                    ctx_inc_sig_coeff_flag(c_idx, xc, yc, num_flags)
                } else {
                    0
                };
                let cm0_inc = if c_idx == 0 { 0 } else { 1 };
                let (t, i) = sel.ctx_shaped(MainCtxTable::SigCoeffFlag, cm1_inc, cm0_inc);
                enc.encode_decision(t, i, u8::from(sig));
            }
            if sig {
                dv[blk_pos] = 1;
                nz.push((blk_pos, xc, yc));
            }
            i_pos -= 1;
        }
        let num_nz = nz.len();
        if num_nz == 0 {
            continue;
        }
        let mut last_greater_a: Option<usize> = None;
        for (n, &(blk_pos, xc, yc)) in nz.iter().enumerate().take(num_nz.min(8)) {
            let is_last = n == 0 && cg_idx as usize == last_coef_group;
            let cm1_inc = if sel.cm_init {
                let num_flags = stencil_sum(&dv, xc, yc, log2_tb_width, log2_tb_height, |v| {
                    (v.unsigned_abs() > 1) as u32
                });
                ctx_inc_coeff_abs_level_greater_a(c_idx, xc, yc, is_last, num_flags)
            } else {
                0
            };
            let cm0_inc = if c_idx == 0 { 0 } else { 1 };
            let (t, i) = sel.ctx_shaped(MainCtxTable::CoeffAbsLevelGreaterFlag, cm1_inc, cm0_inc);
            let flag = levels[blk_pos].unsigned_abs() > 1;
            enc.encode_decision(t, i, u8::from(flag));
            dv[blk_pos] += flag as i32;
            if flag {
                if last_greater_a.is_none() {
                    last_greater_a = Some(n);
                } else {
                    escape_data_present = true;
                }
            }
        }
        if let Some(n) = last_greater_a {
            let (blk_pos, xc, yc) = nz[n];
            let is_last = n == 0 && cg_idx as usize == last_coef_group;
            let cm1_inc = if sel.cm_init {
                let num_flags = stencil_sum(&dv, xc, yc, log2_tb_width, log2_tb_height, |v| {
                    (v.unsigned_abs() > 2) as u32
                });
                ctx_inc_coeff_abs_level_greater_b(c_idx, xc, yc, is_last, num_flags)
            } else {
                0
            };
            let cm0_inc = if c_idx == 0 { 0 } else { 1 };
            let (t, i) = sel.ctx_shaped(MainCtxTable::CoeffAbsLevelGreaterFlag, cm1_inc, cm0_inc);
            let flag = levels[blk_pos].unsigned_abs() > 2;
            enc.encode_decision(t, i, u8::from(flag));
            dv[blk_pos] += flag as i32;
            if flag {
                escape_data_present = true;
            }
        }
        let escape_data_present = escape_data_present || num_nz > 8;
        let mut count_first_b_coef = 1i32;
        if escape_data_present {
            for (n, &(blk_pos, xc, yc)) in nz.iter().enumerate() {
                let base_level = if n < 8 { 2 + count_first_b_coef } else { 1 };
                if dv[blk_pos] >= base_level {
                    let loc_sum = stencil_sum(&dv, xc, yc, log2_tb_width, log2_tb_height, |v| {
                        v.unsigned_abs()
                    }) as i32;
                    let loc_sum_abs = (loc_sum - base_level * 5).clamp(0, 31) as u32;
                    let c_rice = rice_param_coeff_abs_level_remaining(loc_sum_abs);
                    let magnitude = levels[blk_pos].unsigned_abs().min(32767) as i32;
                    debug_assert!(magnitude >= base_level);
                    encode_abs_level_remaining(enc, c_rice, (magnitude - base_level) as u32);
                    dv[blk_pos] = magnitude;
                }
                if dv[blk_pos] >= 2 {
                    count_first_b_coef = 0;
                }
            }
        }
        for &(blk_pos, _, _) in nz.iter() {
            enc.encode_bypass(u8::from(levels[blk_pos] < 0));
            if levels[blk_pos] < 0 {
                dv[blk_pos] = -dv[blk_pos];
            }
        }
    }
}

/// Rate-distortion optimised quantization for the §7.3.8.8 advanced
/// residual syntax (round 458) — a candidate-set search rather than a
/// trellis: the ADCC rate of a coefficient depends on the significance
/// and magnitude stencils of its already-coded neighbours, the
/// per-group greaterA/B budget and the escape state, so the run-length
/// trellis's linear structure does not carry over. Candidates:
/// nearest rounding; the run-length trellis's level vector (a
/// well-shaped `D + λ · R` proxy that already trims small tails); the
/// rounding with every `|level| == 1` coefficient of fractional
/// magnitude below 0.6 dropped; and the rounding with its last one,
/// two or three non-zero coefficients (scan order) dropped. Each is
/// costed as `Σ w · ( c − level )² + λ · R` with `R` the exact bin
/// string of [`encode_residual_coding_adv`] at the model's context
/// state plus the `cbf` bin; the all-zero block competes under its
/// `cbf = 0` cost. Returns `(levels, cbf, cost)` like
/// [`crate::rdoq::rdoq_rle`].
pub fn rdoq_adcc(
    frac: &[f64],
    weights: &[f64],
    blk_w: usize,
    blk_h: usize,
    chroma_array_type: u32,
    inputs: &crate::rdoq::RdoqInputs<'_>,
) -> (Vec<i32>, bool, f64) {
    let n = blk_w * blk_h;
    debug_assert_eq!(frac.len(), n);
    debug_assert_eq!(weights.len(), n);
    let lambda = inputs.lambda;
    let sel = inputs.sel;
    let (cbf_t, cbf_i) = inputs.cbf_ctx;
    let cbf_cost = |bin: u8| crate::bin_cost::bin_cost(inputs.model.context(cbf_t, cbf_i), bin);
    let log2_w = (blk_w as u32).trailing_zeros();
    let log2_h = (blk_h as u32).trailing_zeros();
    let dist = |levels: &[i32]| -> f64 {
        frac.iter()
            .zip(levels.iter())
            .zip(weights.iter())
            .map(|((&c, &l), &w)| {
                let e = c - f64::from(l);
                w * e * e
            })
            .sum()
    };
    let zero_cost = dist(&vec![0i32; n]) + lambda * cbf_cost(0);
    let rounded: Vec<i32> = frac
        .iter()
        .map(|v| v.round().clamp(-32767.0, 32767.0) as i32)
        .collect();
    if rounded.iter().all(|&l| l == 0) {
        return (rounded, false, zero_cost);
    }
    let scan = crate::scan::zig_zag_scan(blk_w, blk_h);
    let mut candidates: Vec<Vec<i32>> = Vec::with_capacity(6);
    candidates.push(rounded.clone());
    let (proxy, proxy_cbf, _) = crate::rdoq::rdoq_rle(frac, weights, blk_w, blk_h, inputs);
    if proxy_cbf {
        candidates.push(proxy);
    }
    let biased: Vec<i32> = rounded
        .iter()
        .zip(frac.iter())
        .map(|(&l, &c)| if l.abs() == 1 && c.abs() < 0.6 { 0 } else { l })
        .collect();
    candidates.push(biased);
    let nz_scan: Vec<usize> = scan
        .iter()
        .map(|&p| p as usize)
        .filter(|&p| rounded[p] != 0)
        .collect();
    for k in 1..=3usize.min(nz_scan.len().saturating_sub(1)) {
        let mut trimmed = rounded.clone();
        for &p in &nz_scan[nz_scan.len() - k..] {
            trimmed[p] = 0;
        }
        candidates.push(trimmed);
    }
    let mut best: Option<(Vec<i32>, f64)> = None;
    for cand in candidates {
        if cand.iter().all(|&l| l == 0) {
            continue;
        }
        let mut model = inputs.model.clone();
        let bits = model.measure(|m| {
            encode_residual_coding_adv(
                m,
                sel,
                inputs.c_idx,
                chroma_array_type,
                &cand,
                log2_w,
                log2_h,
            )
        });
        let cost = dist(&cand) + lambda * (bits + cbf_cost(1));
        if best.as_ref().map_or(true, |b| cost < b.1) {
            best = Some((cand, cost));
        }
    }
    match best {
        Some((levels, cost)) if cost < zero_cost => (levels, true, cost),
        _ => (vec![0i32; n], false, zero_cost),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{CabacEncoder, CabacEngine, InitType};

    /// Round 458: the writer is the exact dual of the reader — random
    /// level blocks of every size (skewed toward small magnitudes with
    /// occasional large escapes), both entropy shapes, both init types,
    /// luma and chroma — and the last-coordinate binarisation inverts
    /// eqs. 149-152 for every coordinate up to 63.
    #[test]
    fn adcc_writer_reads_back() {
        for v in 0..64u32 {
            let (prefix, suffix, len) = last_sig_prefix_suffix(v);
            let back = if prefix > 3 {
                (1u32 << len) * (2 + (prefix & 1)) + suffix
            } else {
                prefix
            };
            assert_eq!(back, v);
            if prefix > 3 {
                assert_eq!((prefix >> 1) - 1, len);
            }
        }
        let mut seed = 0xADCC_0458u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for &cm in &[false, true] {
            for &init in &[InitType::I, InitType::Pb] {
                let sel = CtxSel::new(cm, init);
                let mut enc = CabacEncoder::new();
                if cm {
                    enc.init_main_profile(init, 27);
                }
                let mut written: Vec<(u32, u32, u32, Vec<i32>)> = Vec::new();
                for &(lw, lh) in &[
                    (1u32, 1u32),
                    (2, 2),
                    (3, 2),
                    (2, 4),
                    (3, 3),
                    (4, 4),
                    (5, 3),
                    (6, 6),
                ] {
                    for c_idx in 0..3u32 {
                        for density in [1u32, 4, 12, 40] {
                            let n = 1usize << (lw + lh);
                            let mut levels = vec![0i32; n];
                            for l in levels.iter_mut() {
                                let r = next();
                                if r % 64 < density {
                                    let mag = match r % 16 {
                                        0..=8 => 1,
                                        9..=12 => 2,
                                        13 => 3 + (r % 5) as i32,
                                        14 => 20 + (r % 200) as i32,
                                        _ => 1000 + (r % 31000) as i32,
                                    };
                                    *l = if r & 0x100 != 0 { -mag } else { mag };
                                }
                            }
                            if levels.iter().all(|&l| l == 0) {
                                levels[(next() as usize) % n] = 1;
                            }
                            encode_residual_coding_adv(&mut enc, sel, c_idx, 1, &levels, lw, lh);
                            written.push((lw, lh, c_idx, levels));
                        }
                    }
                }
                enc.encode_terminate(true);
                let bytes = enc.finish();
                let mut eng = CabacEngine::new(&bytes).unwrap();
                if cm {
                    crate::cabac_init::init_main_profile_contexts(&mut eng, init, 27).unwrap();
                }
                let mut stats = AdccStats::default();
                for (lw, lh, c_idx, want) in &written {
                    let mut got = vec![0i32; want.len()];
                    decode_residual_coding_adv(
                        &mut eng, sel, *c_idx, 1, &mut got, &mut stats, *lw, *lh,
                    )
                    .unwrap();
                    assert_eq!(&got, want, "cm{cm} {init:?} {lw}x{lh} c{c_idx}");
                }
                assert!(eng.decode_terminate().unwrap());
            }
        }
    }

    /// The candidate search never scores worse than plain rounding under
    /// its own objective, and its output re-measures to the cost it
    /// reports.
    #[test]
    fn rdoq_adcc_beats_rounding_and_accounts_exactly() {
        use crate::bin_cost::BitCostModel;
        use crate::quant_enc::{forward_transform_fractional, level_unit_sse_weights};
        use crate::rdoq::RdoqInputs;
        let mut seed = 0x0458_ADCCu32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 121) - 60
        };
        for &cm in &[false, true] {
            for &(w, h, qp) in &[(4usize, 4usize, 20), (8, 8, 30), (16, 8, 38), (32, 32, 44)] {
                let sel = CtxSel::new(cm, InitType::I).with_adcc(true);
                let mut model = BitCostModel::new();
                if cm {
                    model.init_main_profile(InitType::I, qp);
                }
                let res: Vec<i32> = (0..w * h).map(|_| next() / 4).collect();
                let frac = forward_transform_fractional(&res, w, h, qp, 8).unwrap();
                let weights = level_unit_sse_weights(w, h, qp, 8);
                let lambda = crate::slice_enc::rd_lambda(qp, 8);
                let inp = RdoqInputs::new(&model, lambda, sel, 0, MainCtxTable::CbfLuma);
                let (levels, cbf, cost) = rdoq_adcc(&frac, &weights, w, h, 1, &inp);
                let dist = |l: &[i32]| -> f64 {
                    frac.iter()
                        .zip(l.iter())
                        .zip(weights.iter())
                        .map(|((&c, &v), &wt)| wt * (c - f64::from(v)).powi(2))
                        .sum()
                };
                let (ct, ci) = inp.cbf_ctx;
                let rounded: Vec<i32> = frac.iter().map(|v| v.round() as i32).collect();
                let round_cost = if rounded.iter().all(|&v| v == 0) {
                    dist(&rounded) + lambda * model.decision_cost(ct, ci, 0)
                } else {
                    let mut m = model.clone();
                    let bits = m.measure(|m| {
                        encode_residual_coding_adv(
                            m,
                            sel,
                            0,
                            1,
                            &rounded,
                            w.trailing_zeros(),
                            h.trailing_zeros(),
                        )
                    });
                    dist(&rounded) + lambda * (bits + model.decision_cost(ct, ci, 1))
                };
                assert!(
                    cost <= round_cost + 1e-9,
                    "cm{cm} {w}x{h} qp{qp}: {cost} > {round_cost}"
                );
                let remeasured = if cbf {
                    let mut m = model.clone();
                    let bits = m.measure(|m| {
                        encode_residual_coding_adv(
                            m,
                            sel,
                            0,
                            1,
                            &levels,
                            w.trailing_zeros(),
                            h.trailing_zeros(),
                        )
                    });
                    dist(&levels) + lambda * (bits + model.decision_cost(ct, ci, 1))
                } else {
                    assert!(levels.iter().all(|&v| v == 0));
                    dist(&levels) + lambda * model.decision_cost(ct, ci, 0)
                };
                assert!(
                    (remeasured - cost).abs() < 1e-6,
                    "cm{cm} {w}x{h}: {remeasured} vs {cost}"
                );
            }
        }
    }

    /// Single DC coefficient of +1 under the Baseline `(0, 0)` collapse:
    /// `last_sig = (0, 0)` (two single-bin TR prefixes), the sig flag is
    /// inferred at the last position, one greaterA "0", no greaterB, no
    /// escape, one bypass sign.
    #[test]
    fn adcc_single_dc_plus_one_cm0() {
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 0); // last_sig_x_prefix = 0
        enc.encode_decision(0, 0, 0); // last_sig_y_prefix = 0
        enc.encode_decision(0, 0, 0); // greaterA[0] = 0
        enc.encode_bypass(0); // sign +
        enc.encode_terminate(true);
        let rbsp = enc.finish();
        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut levels = vec![0i32; 16];
        let mut stats = AdccStats::default();
        decode_residual_coding_adv(
            &mut eng,
            CtxSel::baseline(),
            0,
            1,
            &mut levels,
            &mut stats,
            2,
            2,
        )
        .unwrap();
        assert_eq!(levels[0], 1);
        assert!(levels[1..].iter().all(|&v| v == 0));
        assert_eq!(stats.last_sig_prefix_bins, 2);
        assert_eq!(stats.sig_coeff_bins, 0, "last position is inferred");
        assert_eq!(stats.greater_a_bins, 1);
        assert_eq!(stats.greater_b_bins, 0);
        assert_eq!(stats.remaining_syms, 0);
        assert_eq!(stats.sign_bins, 1);
        assert!(eng.decode_terminate().unwrap());
    }

    /// Two coefficients under `sps_cm_init_flag == 1` (I-slice offsets):
    /// `(0,0) = +5` (greaterA + greaterB + a Rice-0 remaining of 2) and
    /// `(1,0) = −1` (the last significant position). Exercises the
    /// Tables 87/88 prefix contexts, the §9.3.4.2.7 sig stencil, the
    /// shared Table 90 A/B space, the §9.3.3.8 bypass remaining and the
    /// MSB-first sign group.
    #[test]
    fn adcc_two_coeffs_cm1_escape_path() {
        use crate::cabac_init::init_main_profile_contexts;
        let sel = CtxSel::new(true, InitType::I);
        let t87 = MainCtxTable::LastSigCoeffXPrefix;
        let t88 = MainCtxTable::LastSigCoeffYPrefix;
        let t89 = MainCtxTable::SigCoeffFlag;
        let t90 = MainCtxTable::CoeffAbsLevelGreaterFlag;
        let mut enc = CabacEncoder::new();
        enc.init_main_profile(InitType::I, 30);
        // last = (1, 0): x_prefix = 1 → TR "10"; y_prefix = 0 → "0".
        let xi = |b: u32| ctx_inc_last_sig_coeff_prefix(b, 0, 2, 1);
        enc.encode_decision(t87.as_usize(), xi(0), 1);
        enc.encode_decision(t87.as_usize(), xi(1), 0);
        enc.encode_decision(t88.as_usize(), xi(0), 0);
        // sig_coeff_flag at (0,0): the (1,0) neighbour is already 1 →
        // numFlags 1 → sigCtx 2, offset 0 → ctxInc 2.
        enc.encode_decision(t89.as_usize(), ctx_inc_sig_coeff_flag(0, 0, 0, 1), 1);
        // greaterA n=0 at (1,0) — the last position → ctxInc 0; flag 0.
        enc.encode_decision(t90.as_usize(), 0, 0);
        // greaterA n=1 at (0,0): no |v|>1 neighbours → ctxInc 1; flag 1.
        enc.encode_decision(
            t90.as_usize(),
            ctx_inc_coeff_abs_level_greater_a(0, 0, 0, false, 0),
            1,
        );
        // greaterB at n=1 (0,0): no |v|>2 neighbours → ctxInc 1; flag 1.
        enc.encode_decision(
            t90.as_usize(),
            ctx_inc_coeff_abs_level_greater_b(0, 0, 0, false, 0),
            1,
        );
        // remaining for (0,0): baseLevel 3, locSumAbs = clip(1 − 15) = 0
        // → Rice 0, numBinRem 6; remaining = 2 → TR bypass "110".
        enc.encode_bypass(1);
        enc.encode_bypass(1);
        enc.encode_bypass(0);
        // signs MSB-first over blkPosArray = [(1,0), (0,0)]: − then +.
        enc.encode_bypass(1);
        enc.encode_bypass(0);
        enc.encode_terminate(true);
        let rbsp = enc.finish();

        let mut eng = CabacEngine::new(&rbsp).unwrap();
        init_main_profile_contexts(&mut eng, InitType::I, 30).unwrap();
        let mut levels = vec![0i32; 16];
        let mut stats = AdccStats::default();
        decode_residual_coding_adv(&mut eng, sel, 0, 1, &mut levels, &mut stats, 2, 2).unwrap();
        assert_eq!(levels[0], 5, "(0,0) = baseLevel 3 + remaining 2");
        assert_eq!(levels[1], -1, "(1,0) = last coefficient, negative");
        assert!(levels[2..].iter().all(|&v| v == 0));
        assert_eq!(stats.last_sig_prefix_bins, 3);
        assert_eq!(stats.sig_coeff_bins, 1);
        assert_eq!(stats.greater_a_bins, 2);
        assert_eq!(stats.greater_b_bins, 1);
        assert_eq!(stats.remaining_syms, 1);
        assert_eq!(stats.sign_bins, 2);
        assert!(eng.decode_terminate().unwrap());
    }

    /// eqs. 149-152: a prefix above 3 carries an FL bypass suffix. On an
    /// 8×8 block, x_prefix 4 + suffix 1 → LastSignificantCoeffX =
    /// (1 << 1) · (2 + 0) + 1 = 5.
    #[test]
    fn adcc_last_sig_suffix_composition() {
        let mut enc = CabacEncoder::new();
        // x_prefix = 4 → TR "11110" (cMax = 5 on log2 3); baseline
        // per-bin ctxIdx = binIdx for luma (Table 95, cm_init == 0 row).
        for b in 0..4 {
            enc.encode_decision(0, b, 1);
        }
        enc.encode_decision(0, 4, 0);
        enc.encode_bypass(1); // x_suffix = 1 (1 bin)
                              // y_prefix = 0.
        enc.encode_decision(0, 0, 0);
        // The last coefficient at (5, 0): sig walk from scanPosLast down
        // to scan position 0 reads sig flags for every non-last position
        // in the two 16-coefficient groups; encode them all as 0.
        let scan = crate::scan::zig_zag_scan(8, 8);
        let scan_pos_last = scan.iter().position(|&p| p == 5).unwrap();
        for _ in 0..scan_pos_last {
            enc.encode_decision(0, 0, 0);
        }
        enc.encode_decision(0, 0, 0); // greaterA[0] = 0
        enc.encode_bypass(0); // sign +
        enc.encode_terminate(true);
        let rbsp = enc.finish();

        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut levels = vec![0i32; 64];
        let mut stats = AdccStats::default();
        decode_residual_coding_adv(
            &mut eng,
            CtxSel::baseline(),
            0,
            1,
            &mut levels,
            &mut stats,
            3,
            3,
        )
        .unwrap();
        assert_eq!(levels[5], 1, "LastSignificantCoeffX = 5");
        assert_eq!(levels.iter().filter(|&&v| v != 0).count(), 1);
        assert_eq!(stats.last_sig_suffix_bins, 1);
        assert_eq!(stats.sig_coeff_bins, scan_pos_last as u32);
        assert!(eng.decode_terminate().unwrap());
    }

    /// Errata #213(b): `last_sig_coeff_y_suffix` is sized by
    /// `last_sig_coeff_y_prefix`, **not** `last_sig_coeff_x_prefix` (the
    /// Table 91 `cMax` typo copies the x-row). This fixture makes the two
    /// suffix widths differ — a 16×16 block with `last_x = 12` (x_prefix 7,
    /// a **2-bin** x_suffix) and `last_y = 5` (y_prefix 4, a **1-bin**
    /// y_suffix). Had the y_suffix been sized from x_prefix (7 → 2 bins) it
    /// would over-read one bypass bin and desync the entire sig walk, so a
    /// clean decode to `(12, 5)` with a terminating flush is the proof the
    /// crate follows the §7.4 semantics (`y_prefix`) over the printed typo.
    #[test]
    fn adcc_last_sig_y_suffix_sized_by_y_prefix_errata_213b() {
        let mut enc = CabacEncoder::new();
        // last_x = 12: x_prefix = 7 (TR "1111111", cMax = 7 on log2 4, no
        // terminator), 2-bin x_suffix "00" → (1 << 2)·(2 + 1) + 0 = 12.
        // Baseline per-bin ctxIdx = binIdx for luma.
        for b in 0..7 {
            enc.encode_decision(0, b, 1);
        }
        enc.encode_bypass(0);
        enc.encode_bypass(0);
        // last_y = 5: y_prefix = 4 (TR "11110"), 1-bin y_suffix "1" →
        // (1 << 1)·(2 + 0) + 1 = 5.
        for b in 0..4 {
            enc.encode_decision(0, b, 1);
        }
        enc.encode_decision(0, 4, 0);
        enc.encode_bypass(1);
        // Single non-zero coefficient at (12, 5): the reverse sig walk reads
        // a 0 for every non-last scan position below scanPosLast.
        let scan = crate::scan::zig_zag_scan(16, 16);
        let raster_last = 12 + 5 * 16;
        let scan_pos_last = scan.iter().position(|&p| p == raster_last).unwrap();
        for _ in 0..scan_pos_last {
            enc.encode_decision(0, 0, 0);
        }
        enc.encode_decision(0, 0, 0); // greaterA[0] = 0
        enc.encode_bypass(0); // sign +
        enc.encode_terminate(true);
        let rbsp = enc.finish();

        let mut eng = CabacEngine::new(&rbsp).unwrap();
        let mut levels = vec![0i32; 256];
        let mut stats = AdccStats::default();
        decode_residual_coding_adv(
            &mut eng,
            CtxSel::baseline(),
            0,
            1,
            &mut levels,
            &mut stats,
            4,
            4,
        )
        .unwrap();
        // The single coefficient lands at raster (12 + 5·16) = 92.
        assert_eq!(levels[raster_last as usize], 1);
        assert_eq!(levels.iter().filter(|&&v| v != 0).count(), 1);
        // x_suffix (2 bins) + y_suffix (1 bin) = 3 suffix bins total; a
        // typo-following decoder would have consumed 4 and desynced.
        assert_eq!(stats.last_sig_suffix_bins, 3);
        assert_eq!(stats.sig_coeff_bins, scan_pos_last as u32);
        assert!(eng.decode_terminate().unwrap());
    }
}
