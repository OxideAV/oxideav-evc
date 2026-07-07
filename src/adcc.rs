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

use crate::cabac::CabacEngine;
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
        eng.decode_tr_regular(c_max, 0, 0, |_| {
            bins += 1;
            0
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
                let (t, i) = if sel.cm_init {
                    let num_flags =
                        stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                            (v != 0) as u32
                        });
                    sel.ctx(
                        MainCtxTable::SigCoeffFlag,
                        ctx_inc_sig_coeff_flag(c_idx, xc, yc, num_flags),
                    )
                } else {
                    (0, 0)
                };
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
            let (t, i) = if sel.cm_init {
                let num_flags = stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                    (v.unsigned_abs() > 1) as u32
                });
                sel.ctx(
                    MainCtxTable::CoeffAbsLevelGreaterFlag,
                    ctx_inc_coeff_abs_level_greater_a(c_idx, xc, yc, is_last, num_flags),
                )
            } else {
                (0, 0)
            };
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
            let (t, i) = if sel.cm_init {
                let num_flags = stencil_sum(levels, xc, yc, log2_tb_width, log2_tb_height, |v| {
                    (v.unsigned_abs() > 2) as u32
                });
                sel.ctx(
                    MainCtxTable::CoeffAbsLevelGreaterFlag,
                    ctx_inc_coeff_abs_level_greater_b(c_idx, xc, yc, is_last, num_flags),
                )
            } else {
                (0, 0)
            };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{CabacEncoder, CabacEngine, InitType};

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
        // x_prefix = 4 → TR "11110" (cMax = 5 on log2 3).
        for _ in 0..4 {
            enc.encode_decision(0, 0, 1);
        }
        enc.encode_decision(0, 0, 0);
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
}
