//! Intra Block Copy (IBC) primitives — ISO/IEC 23094-1 §8.6.
//!
//! IBC is a Main-profile tool: a coding unit signalled with
//! `ibc_flag = 1` selects a previously-reconstructed region of the
//! **current** picture as its prediction, using a small block-vector
//! (BV) signalled via `abs_mvd_l0` + `mvd_l0_sign_flag` (same syntax as
//! inter `MvdL0`). No inter reference frame is involved.
//!
//! Round-73 lands the IBC scaffold: the spec's three derivation /
//! prediction primitives plus the conformance constraints from
//! §8.6.2.1.  Wiring `coding_unit()` so the CABAC walker actually
//! emits IBC CUs (and lifting the `sps_ibc_flag` gate in the front
//! door) is the next round — but every numeric piece is here and
//! unit-tested.
//!
//! The functions here implement:
//!
//! * `derive_ibc_luma_mv` — §8.6.2.1 eq. 1025-1039.  Folds the parsed
//!   `mvd_l0` into a signed-16-bit `mvL`, then shifts left by 4 to land
//!   on the 1/16-pel grid the §8.5.4.3 interpolator expects.
//! * `derive_ibc_chroma_mv` — §8.6.2.2 eq. 1040-1041.
//! * `validate_ibc_constraints` — eq. 1035-1038.  The §8.6.2.1
//!   bitstream-conformance rules (same-CTU-row, current-or-left CTU
//!   column).  Returns `Error::Invalid` on a non-conformant BV.
//! * `predict_ibc_block` — §8.6.3.  Since `mvL <<= 4` makes the
//!   low-4 bits of `mvL` always zero, IBC always lands on an integer
//!   sample — no fractional interpolation needed.  Copies the
//!   `(nCbW × nCbH)` reference block from the **current** picture's
//!   already-reconstructed region.
//!
//! Everything else (CABAC `ibc_flag` parsing, dual-tree handling,
//! the §8.6.1 5-step decoding pipeline wiring) is deferred to the
//! follow-up round that lifts the SPS gate.

use oxideav_core::{Error, Result};

use crate::inter::MotionVector;
use crate::picture::YuvPicture;

/// 16-bit modular wrap per eq. 1027-1030 (the same `+ 2^16 / % 2^16`
/// pattern used elsewhere in §8.5).  Kept private to the module so
/// callers go through the public derivation helpers.
fn wrap16(v: i32) -> i32 {
    let u = (v as u32) & 0xFFFF;
    if u >= 0x8000 {
        (u as i32) - 0x10000
    } else {
        u as i32
    }
}

/// Derive the IBC luma motion vector per §8.6.2.1, eq. 1025-1039.
///
/// Inputs:
///   * `mvd` — the parsed `MvdL0[xCb][yCb]` (signed, in 1-pel units
///     for IBC — the spec carries it directly through `abs_mvd_l0` +
///     `mvd_l0_sign_flag`).
///
/// Output: `mvL` in 1/16-pel units (the `<< 4` shift of eq. 1039 is
/// folded in), ready to pass to a 1/16-pel block-copy or interpolator.
pub fn derive_ibc_luma_mv(mvd: MotionVector) -> MotionVector {
    // eq. 1027/1029: u = (mvp + mvd + 2^16) % 2^16. With mvp = (0,0)
    // (eq. 1026 "mvp[0] = mvp[1] = 0") this collapses to mvd mod 2^16.
    // eq. 1028/1030: sign-extend back into [-2^15, 2^15).
    let mv_x = wrap16(mvd.x);
    let mv_y = wrap16(mvd.y);
    // eq. 1039: mvL[0] <<= 4, mvL[1] <<= 4 (move to 1/16-pel grid).
    MotionVector {
        x: mv_x << 4,
        y: mv_y << 4,
    }
}

/// Derive the IBC chroma motion vector from the luma MV per §8.6.2.2,
/// eq. 1040-1041. `mvL` is expected in 1/16-pel luma units (the output
/// of `derive_ibc_luma_mv`); the returned MV is in 1/32-chroma-pel
/// units, matching the §8.5.4.3.3 chroma interpolator convention.
///
/// `chroma_format_idc` selects SubWidthC / SubHeightC per Table 2:
///   * 0 (monochrome) → only luma; chroma MV is zero
///   * 1 (4:2:0) → SubWidthC = SubHeightC = 2
///   * 2 (4:2:2) → SubWidthC = 2, SubHeightC = 1
///   * 3 (4:4:4) → SubWidthC = SubHeightC = 1
pub fn derive_ibc_chroma_mv(mv_l: MotionVector, chroma_format_idc: u32) -> MotionVector {
    let (sub_w, sub_h) = match chroma_format_idc {
        0 => return MotionVector { x: 0, y: 0 },
        1 => (2i32, 2i32),
        2 => (2i32, 1i32),
        3 => (1i32, 1i32),
        _ => return MotionVector { x: 0, y: 0 },
    };
    // eq. 1040/1041: mvC[k] = (mvL[k] >> (3 + SubXC)) * 32.
    // (Right-shift on signed i32 is arithmetic in Rust — sign-preserving
    // exactly per spec.)
    MotionVector {
        x: (mv_l.x >> (3 + sub_w)) * 32,
        y: (mv_l.y >> (3 + sub_h)) * 32,
    }
}

/// §8.6.2.1 bitstream-conformance constraints on the IBC BV
/// (eq. 1035-1038 plus the "at least one of …" guard).  Only the
/// Baseline + `sps_suco_flag = 0` subset is encoded here — `sps_suco_flag = 1`
/// adds further rules around the right column that are scope for a
/// follow-up round.
///
/// Inputs:
///   * `mv_l_sixteenth` — the 1/16-pel `mvL` from `derive_ibc_luma_mv`.
///   * `x_cb`, `y_cb` — top-left luma sample of the current CB.
///   * `n_cb_w`, `n_cb_h` — width/height of the current luma CB.
///   * `ctb_log2_size_y` — `CtbLog2SizeY` (5..=7 in EVC).
///
/// Returns `Err(Error::Invalid)` if the BV violates any of:
///   * "at least one of mvL[0]+nCbW ≤ 0, mvL[1]+nCbH ≤ 0" — i.e. the
///     reference block lies strictly above-or-left of the current CB.
///     IBC cannot reference samples that have not yet been
///     reconstructed.
///   * Same-CTU-row constraint (eq. 1035/1036).
///   * Current-or-left CTU-column constraint (eq. 1037/1038).
///
/// Returns `Ok(())` on a conformant BV.
pub fn validate_ibc_constraints(
    mv_l_sixteenth: MotionVector,
    x_cb: i32,
    y_cb: i32,
    n_cb_w: i32,
    n_cb_h: i32,
    ctb_log2_size_y: u32,
) -> Result<()> {
    if n_cb_w <= 0 || n_cb_h <= 0 {
        return Err(Error::invalid("evc ibc: zero / negative CB dimensions"));
    }
    if !(5..=7).contains(&ctb_log2_size_y) {
        return Err(Error::invalid(format!(
            "evc ibc: CtbLog2SizeY {ctb_log2_size_y} outside EVC 5..=7"
        )));
    }
    // Move mvL back to integer-pel for the §8.6.2.1 constraints (the
    // spec writes them on the unshifted mvL).  IBC always lands on
    // integer samples (low 4 bits of mvL_sixteenth are always zero).
    if mv_l_sixteenth.x & 0xF != 0 || mv_l_sixteenth.y & 0xF != 0 {
        return Err(Error::invalid(
            "evc ibc: mvL not aligned to integer-pel (fractional BVs forbidden)",
        ));
    }
    let mv_x = mv_l_sixteenth.x >> 4;
    let mv_y = mv_l_sixteenth.y >> 4;

    // eq. 1031-1034: reference-block corners.
    let x_ref_tl = x_cb + mv_x;
    let y_ref_tl = y_cb + mv_y;
    let x_ref_tr = x_ref_tl + n_cb_w - 1;
    let y_ref_bl = y_ref_tl + n_cb_h - 1;

    // "At least one of … shall be true" guard (sps_suco_flag=0 path):
    //   mvL[0] + nCbW ≤ 0  OR  mvL[1] + nCbH ≤ 0
    // Rephrased: the reference block must lie strictly above-or-left
    // of (xCb, yCb) — its top-right must be < xCb on the row, or its
    // bottom-left must be < yCb on the column.
    let cond_left = mv_x + n_cb_w <= 0; // reference column ends before xCb
    let cond_above = mv_y + n_cb_h <= 0; // reference row ends before yCb
    if !(cond_left || cond_above) {
        return Err(Error::invalid(
            "evc ibc: BV does not point above-or-left of current CB (eq. 1113 guard)",
        ));
    }

    // eq. 1035: yRefTL >> CtbLog2SizeY == yCb >> CtbLog2SizeY.
    if (y_ref_tl >> ctb_log2_size_y) != (y_cb >> ctb_log2_size_y) {
        return Err(Error::invalid(
            "evc ibc: BV crosses CTU row boundary (eq. 1035)",
        ));
    }
    // eq. 1036: yRefBL >> CtbLog2SizeY == yCb >> CtbLog2SizeY.
    if (y_ref_bl >> ctb_log2_size_y) != (y_cb >> ctb_log2_size_y) {
        return Err(Error::invalid(
            "evc ibc: BV bottom row crosses CTU row boundary (eq. 1036)",
        ));
    }
    // eq. 1037: xRefTL >> CtbLog2SizeY >= (xCb >> CtbLog2SizeY) - 1.
    if (x_ref_tl >> ctb_log2_size_y) < (x_cb >> ctb_log2_size_y) - 1 {
        return Err(Error::invalid(
            "evc ibc: BV points more than one CTU to the left (eq. 1037)",
        ));
    }
    // eq. 1038: xRefTR >> CtbLog2SizeY <= xCb >> CtbLog2SizeY.
    if (x_ref_tr >> ctb_log2_size_y) > (x_cb >> ctb_log2_size_y) {
        return Err(Error::invalid(
            "evc ibc: BV reference block crosses into right CTU (eq. 1038)",
        ));
    }
    // Picture-edge sanity check (callers should already have a
    // `xRefTL >= 0` invariant from the conformance suite, but guard
    // anyway — out-of-picture references are not reconstructed).
    if x_ref_tl < 0 || y_ref_tl < 0 {
        return Err(Error::invalid(
            "evc ibc: BV points to negative picture coordinates",
        ));
    }

    Ok(())
}

/// §8.6.3 IBC block prediction.  IBC always lands on integer samples
/// (eq. 1039's `<< 4` is bijective with a zero low-nibble, and the
/// fractional sample interpolation §8.5.4.3.1 collapses to a sample
/// copy at phase 0 for both luma and chroma), so this is just a
/// rectangular memcpy from the current picture's already-reconstructed
/// region.
///
/// Inputs:
///   * `cur_pic` — the current picture (only the already-reconstructed
///     region is read; the validated BV constraint guarantees the
///     samples have been emitted in raster order before this CU).
///   * `x_cb`, `y_cb` — top-left luma sample of the current CB.
///   * `n_cb_w_l`, `n_cb_h_l` — luma CB dimensions.
///   * `mv_l_sixteenth` / `mv_c_thirtysecondth` — pre-validated MVs.
///
/// Outputs:
///   * `pred_y` — `n_cb_w_l × n_cb_h_l` luma samples, row-major.
///   * `pred_cb` / `pred_cr` — same for chroma when `chroma_present`.
///
/// Bit-depth is taken from `cur_pic.bit_depth`; round-73 only supports
/// 8-bit luma + chroma (the picture buffer already enforces this).
#[allow(clippy::too_many_arguments)]
pub fn predict_ibc_block(
    cur_pic: &YuvPicture,
    x_cb: i32,
    y_cb: i32,
    n_cb_w_l: usize,
    n_cb_h_l: usize,
    mv_l_sixteenth: MotionVector,
    mv_c_thirtysecondth: MotionVector,
    chroma_present: bool,
    pred_y: &mut [i32],
    pred_cb: &mut [i32],
    pred_cr: &mut [i32],
) -> Result<()> {
    if pred_y.len() != n_cb_w_l * n_cb_h_l {
        return Err(Error::invalid(
            "evc ibc: pred_y buffer size mismatch (expected nCbW * nCbH)",
        ));
    }
    if mv_l_sixteenth.x & 0xF != 0 || mv_l_sixteenth.y & 0xF != 0 {
        return Err(Error::invalid(
            "evc ibc: mvL not aligned to integer-pel (fractional BVs forbidden)",
        ));
    }
    let mv_x_l = mv_l_sixteenth.x >> 4;
    let mv_y_l = mv_l_sixteenth.y >> 4;
    let stride_y = cur_pic.y_stride();
    let pic_w_y = cur_pic.width as i32;
    let pic_h_y = cur_pic.height as i32;

    for j in 0..n_cb_h_l {
        for i in 0..n_cb_w_l {
            let xr = x_cb + mv_x_l + i as i32;
            let yr = y_cb + mv_y_l + j as i32;
            // Clamp picture-edge per the §6.4.1 + Clip3 convention used
            // throughout §8.5/§8.6.  A conformant BV (validated above)
            // never hits the clamp.
            let xc = xr.clamp(0, pic_w_y - 1) as usize;
            let yc = yr.clamp(0, pic_h_y - 1) as usize;
            pred_y[j * n_cb_w_l + i] = cur_pic.y[yc * stride_y + xc] as i32;
        }
    }
    if !chroma_present {
        return Ok(());
    }
    let (sub_w, sub_h) = match cur_pic.chroma_format_idc {
        0 => return Ok(()),
        1 => (2usize, 2usize),
        2 => (2usize, 1usize),
        3 => (1usize, 1usize),
        _ => {
            return Err(Error::invalid(format!(
                "evc ibc: unsupported chroma_format_idc {}",
                cur_pic.chroma_format_idc
            )))
        }
    };
    let n_cb_w_c = n_cb_w_l / sub_w;
    let n_cb_h_c = n_cb_h_l / sub_h;
    if pred_cb.len() != n_cb_w_c * n_cb_h_c || pred_cr.len() != n_cb_w_c * n_cb_h_c {
        return Err(Error::invalid(
            "evc ibc: pred_cb/pred_cr buffer size mismatch",
        ));
    }
    // mv_c is in 1/32-chroma-pel; mask to chroma integer-pel.
    if mv_c_thirtysecondth.x & 0x1F != 0 || mv_c_thirtysecondth.y & 0x1F != 0 {
        return Err(Error::invalid(
            "evc ibc: mvC not aligned to integer chroma-pel (eq. 1040 guarantees this)",
        ));
    }
    let mv_x_c = mv_c_thirtysecondth.x >> 5;
    let mv_y_c = mv_c_thirtysecondth.y >> 5;
    let stride_c = cur_pic.c_stride();
    let pic_w_c = (cur_pic.width as usize / sub_w) as i32;
    let pic_h_c = (cur_pic.height as usize / sub_h) as i32;
    let x_cb_c = x_cb / sub_w as i32;
    let y_cb_c = y_cb / sub_h as i32;
    for j in 0..n_cb_h_c {
        for i in 0..n_cb_w_c {
            let xr = x_cb_c + mv_x_c + i as i32;
            let yr = y_cb_c + mv_y_c + j as i32;
            let xc = xr.clamp(0, pic_w_c - 1) as usize;
            let yc = yr.clamp(0, pic_h_c - 1) as usize;
            pred_cb[j * n_cb_w_c + i] = cur_pic.cb[yc * stride_c + xc] as i32;
            pred_cr[j * n_cb_w_c + i] = cur_pic.cr[yc * stride_c + xc] as i32;
        }
    }
    Ok(())
}

/// §8.6.1 IBC CU decoding pipeline — steps 1, 2, 3 chained.
///
/// Runs the spec's first three ordered steps for an IBC-coded coding
/// unit:
///
/// 1. derive luma MV (§8.6.2.1, eq. 1025-1039);
/// 2. derive chroma MV when chroma is present (§8.6.2.2, eq. 1040-1041);
/// 3. predict the block from the current picture's already-reconstructed
///    region (§8.6.3).
///
/// Steps 4 (residual decode) and 5 (picture reconstruction prior to
/// in-loop filtering) live elsewhere — they're shared with the inter
/// pipeline (§8.5.6.1) and the picture-construction process (§8.7.5),
/// so they don't belong in this module.
///
/// The MV is also **validated** in-line against §8.6.2.1 bitstream
/// conformance via `validate_ibc_constraints`. A non-conformant BV
/// short-circuits to `Err(Error::Invalid)` before any sample read.
///
/// Inputs:
///   * `cur_pic` — the current picture (luma + chroma planes already
///     populated up to the current CU's position).
///   * `x_cb`, `y_cb` — top-left luma sample of the current CB.
///   * `n_cb_w_l`, `n_cb_h_l` — luma CB dimensions.
///   * `mvd` — parsed `MvdL0` (signed, in integer-pel units for IBC).
///   * `ctb_log2_size_y` — `CtbLog2SizeY` (5..=7 in EVC).
///   * `chroma_present` — whether to fill chroma prediction buffers.
///
/// Outputs:
///   * `pred_y` — `n_cb_w_l × n_cb_h_l` luma prediction samples (row-major).
///   * `pred_cb` / `pred_cr` — chroma prediction samples (row-major) when
///     `chroma_present` is true. The buffers must already be sized to
///     `(n_cb_w_l / SubWidthC) * (n_cb_h_l / SubHeightC)` for the
///     picture's `chroma_format_idc`.
///   * Returns the derived `(mvL, mvC)` pair on success so the caller
///     can drive HMVP update / side-info-grid stamp.
#[allow(clippy::too_many_arguments)]
pub fn decode_ibc_cu(
    cur_pic: &YuvPicture,
    x_cb: i32,
    y_cb: i32,
    n_cb_w_l: usize,
    n_cb_h_l: usize,
    mvd: MotionVector,
    ctb_log2_size_y: u32,
    chroma_present: bool,
    pred_y: &mut [i32],
    pred_cb: &mut [i32],
    pred_cr: &mut [i32],
) -> Result<(MotionVector, MotionVector)> {
    // Step 1: derive luma MV (§8.6.2.1).
    let mv_l = derive_ibc_luma_mv(mvd);
    // Inline conformance check before any sample read. The validator's
    // CB-dimension check covers nCbW / nCbH non-positive, so we don't
    // duplicate that guard here.
    validate_ibc_constraints(
        mv_l,
        x_cb,
        y_cb,
        n_cb_w_l as i32,
        n_cb_h_l as i32,
        ctb_log2_size_y,
    )?;
    // Step 2: derive chroma MV (§8.6.2.2).
    let mv_c = if chroma_present {
        derive_ibc_chroma_mv(mv_l, cur_pic.chroma_format_idc)
    } else {
        MotionVector { x: 0, y: 0 }
    };
    // Step 3: predict (§8.6.3).
    predict_ibc_block(
        cur_pic,
        x_cb,
        y_cb,
        n_cb_w_l,
        n_cb_h_l,
        mv_l,
        mv_c,
        chroma_present,
        pred_y,
        pred_cb,
        pred_cr,
    )?;
    Ok((mv_l, mv_c))
}

/// §7.4.5 `isIbcAllowed` predicate (the structural part — the CABAC
/// walker still has to gate on `treeType` and `predModeConstraint`).
///
/// Returns `true` iff all of:
///   * `sps_ibc_flag == 1`,
///   * `log2CbWidth <= log2MaxIbcCandSize` AND
///     `log2CbHeight <= log2MaxIbcCandSize`.
///
/// The remaining bullet (`treeType != DUAL_TREE_CHROMA`, `predModeConstraint`
/// rules) is decided by the caller from the dual-tree / pred-mode state
/// the walker tracks — keeping it out of this predicate lets unit tests
/// drive the size + flag side of the gate without faking a tree-type
/// enum.
pub fn is_ibc_allowed_for_size(
    sps_ibc_flag: bool,
    log2_max_ibc_cand_size: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
) -> bool {
    sps_ibc_flag
        && log2_cb_width <= log2_max_ibc_cand_size
        && log2_cb_height <= log2_max_ibc_cand_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::picture::YuvPicture;

    // -------- derive_ibc_luma_mv (eq. 1025-1039) --------

    #[test]
    fn ibc_luma_mv_zero_mvd_lands_at_origin() {
        let mv = derive_ibc_luma_mv(MotionVector { x: 0, y: 0 });
        assert_eq!(mv, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn ibc_luma_mv_negative_mvd_shifts_to_sixteenth() {
        let mv = derive_ibc_luma_mv(MotionVector { x: -16, y: -8 });
        // wrap16 of -16 stays -16; <<4 → -256.
        assert_eq!(mv, MotionVector { x: -256, y: -128 });
    }

    #[test]
    fn ibc_luma_mv_positive_mvd_shifts_to_sixteenth() {
        let mv = derive_ibc_luma_mv(MotionVector { x: 32, y: 17 });
        assert_eq!(
            mv,
            MotionVector {
                x: 32 << 4,
                y: 17 << 4
            }
        );
    }

    #[test]
    fn ibc_luma_mv_wraps_at_16_bits() {
        // mvd = 32800 -> wrap16 -> 32800 - 65536 = -32736.  Then <<4
        // would overflow i16 but we hold i32, so this represents the
        // spec's eq. 1027/1028 modular arithmetic.
        let mv = derive_ibc_luma_mv(MotionVector {
            x: 32800,
            y: -32800,
        });
        assert_eq!(mv.x, -32736 << 4);
        assert_eq!(mv.y, 32736 << 4);
    }

    // -------- derive_ibc_chroma_mv (eq. 1040-1041) --------

    #[test]
    fn ibc_chroma_mv_monochrome_is_zero() {
        let mv_c = derive_ibc_chroma_mv(MotionVector { x: 1024, y: 1024 }, 0);
        assert_eq!(mv_c, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn ibc_chroma_mv_420_halves_and_rescales() {
        // mvL = (256, 256) in 1/16-pel = (16, 16) integer luma pels.
        // 4:2:0 chroma MV: mvC[k] = (mvL[k] >> (3 + 2)) * 32 = (mvL >> 5) * 32.
        // 256 >> 5 = 8; * 32 = 256. So integer mvC is (8, 8) chroma pels
        // — half the luma offset, as expected.
        let mv_c = derive_ibc_chroma_mv(MotionVector { x: 256, y: 256 }, 1);
        assert_eq!(mv_c, MotionVector { x: 256, y: 256 });
        // Integer chroma sample part: mv_c >> 5.
        assert_eq!(mv_c.x >> 5, 8);
        assert_eq!(mv_c.y >> 5, 8);
    }

    #[test]
    fn ibc_chroma_mv_422_halves_x_only() {
        // 4:2:2: SubWidthC=2, SubHeightC=1. mvC[0] = (mvL[0] >> 5) * 32,
        // mvC[1] = (mvL[1] >> 4) * 32.
        let mv_c = derive_ibc_chroma_mv(MotionVector { x: 256, y: 256 }, 2);
        // x: 256 >> 5 = 8, * 32 = 256.
        // y: 256 >> 4 = 16, * 32 = 512.
        assert_eq!(mv_c, MotionVector { x: 256, y: 512 });
    }

    #[test]
    fn ibc_chroma_mv_444_matches_luma_when_rescaled() {
        // 4:4:4: SubWidthC=SubHeightC=1. mvC = (mvL >> 4) * 32 — i.e.
        // mvC ranges in 1/32 chroma-pel which IS 1/32 luma-pel since
        // sub-sampling is 1×1.  Integer-pel parts should match.
        let mv_c = derive_ibc_chroma_mv(MotionVector { x: 32, y: 32 }, 3);
        // 32 >> 4 = 2, * 32 = 64.
        assert_eq!(mv_c, MotionVector { x: 64, y: 64 });
        assert_eq!(mv_c.x >> 5, 2);
        assert_eq!(mv_c.y >> 5, 2);
    }

    #[test]
    fn ibc_chroma_mv_negative_sign_preserved() {
        // Arithmetic right shift preserves sign — important for
        // negative BVs (the IBC common case, since the reference must
        // lie above-or-left of the current CB).
        let mv_c = derive_ibc_chroma_mv(MotionVector { x: -256, y: -512 }, 1);
        // -256 >> 5 = -8, * 32 = -256.
        // -512 >> 5 = -16, * 32 = -512.
        assert_eq!(mv_c, MotionVector { x: -256, y: -512 });
    }

    // -------- validate_ibc_constraints (eq. 1035-1038) --------

    fn mv_sixteenth_from_integer(x: i32, y: i32) -> MotionVector {
        MotionVector {
            x: x << 4,
            y: y << 4,
        }
    }

    #[test]
    fn ibc_validate_above_only_reference_is_conformant() {
        // CTB = 32×32 (log2 = 5). Current CB at (16, 16), 16×16.  Reference
        // 16 pels above and overlapping in x → cond_above true, cond_left false.
        // But cond_above OR cond_left is enough.
        let mv = mv_sixteenth_from_integer(0, -16);
        let r = validate_ibc_constraints(mv, 16, 16, 16, 16, 5);
        assert!(r.is_ok(), "expected ok, got {r:?}");
    }

    #[test]
    fn ibc_validate_left_only_reference_is_conformant() {
        // Reference 16 pels to the left, same row → cond_left = (-16 + 16 <= 0) = true.
        let mv = mv_sixteenth_from_integer(-16, 0);
        let r = validate_ibc_constraints(mv, 16, 16, 16, 16, 5);
        assert!(r.is_ok(), "expected ok, got {r:?}");
    }

    #[test]
    fn ibc_validate_overlapping_reference_rejected() {
        // BV (-8, 0): reference top-left = (8, 16), bottom-right = (23, 31).
        // Current CB = (16..32, 16..32) → reference *overlaps* current.
        // cond_left = (-8 + 16 <= 0) = (8 <= 0) = false.
        // cond_above = (0 + 16 <= 0) = false.
        // → rejected by the "at least one of" guard.
        let mv = mv_sixteenth_from_integer(-8, 0);
        let r = validate_ibc_constraints(mv, 16, 16, 16, 16, 5);
        assert!(r.is_err());
    }

    #[test]
    fn ibc_validate_cross_ctu_row_rejected() {
        // CtbLog2SizeY = 5 → CTU=32. CB at (0, 32).
        // BV (0, -16) → yRefTL = 16 → CTU row 0, while current CTU row = 1.
        let mv = mv_sixteenth_from_integer(0, -16);
        let r = validate_ibc_constraints(mv, 0, 32, 16, 16, 5);
        assert!(r.is_err());
    }

    #[test]
    fn ibc_validate_left_neighbour_ctu_allowed() {
        // CtbLog2SizeY = 5 → CTU = 32. CB at (32, 0), 16×16.
        // BV (-16, 0) → xRefTL = 16, xRefTR = 31 → CTU column 0
        // (current col is 1). Spec allows xRefTL >> log2 >= (xCb >> log2) - 1
        // (i.e. current OR left CTU).
        let mv = mv_sixteenth_from_integer(-16, 0);
        let r = validate_ibc_constraints(mv, 32, 0, 16, 16, 5);
        assert!(r.is_ok(), "expected ok, got {r:?}");
    }

    #[test]
    fn ibc_validate_two_ctus_left_rejected() {
        // CTU=32; CB at (64, 0). BV (-48, 0) → xRefTL = 16 → CTU col 0,
        // (xCb >> 5) - 1 = 1. xRefTL >> 5 = 0 < 1 → rejected.
        let mv = mv_sixteenth_from_integer(-48, 0);
        let r = validate_ibc_constraints(mv, 64, 0, 16, 16, 5);
        assert!(r.is_err());
    }

    #[test]
    fn ibc_validate_fractional_bv_rejected() {
        // Low-4-bits non-zero → fractional BV. Spec disallows.
        let mv = MotionVector { x: -257, y: -128 };
        let r = validate_ibc_constraints(mv, 16, 16, 16, 16, 5);
        assert!(r.is_err());
    }

    #[test]
    fn ibc_validate_zero_dims_rejected() {
        let mv = mv_sixteenth_from_integer(-16, 0);
        let r = validate_ibc_constraints(mv, 16, 16, 0, 16, 5);
        assert!(r.is_err());
    }

    #[test]
    fn ibc_validate_bad_ctb_log2_rejected() {
        let mv = mv_sixteenth_from_integer(-16, 0);
        let r = validate_ibc_constraints(mv, 16, 16, 16, 16, 4);
        assert!(r.is_err());
        let r = validate_ibc_constraints(mv, 16, 16, 16, 16, 8);
        assert!(r.is_err());
    }

    #[test]
    fn ibc_validate_negative_reference_origin_rejected() {
        // CB at (0, 8). BV (0, -16) → yRefTL = -8 < 0.
        // Also fails the same-CTU-row test (negative right-shift differs).
        let mv = mv_sixteenth_from_integer(0, -16);
        let r = validate_ibc_constraints(mv, 0, 8, 8, 8, 5);
        assert!(r.is_err());
    }

    // -------- predict_ibc_block (§8.6.3) --------

    fn make_pic_with_gradient(w: u32, h: u32) -> YuvPicture {
        let mut p = YuvPicture::new(w, h, 1, 8).expect("pic alloc");
        for y in 0..(h as usize) {
            for x in 0..(w as usize) {
                p.y[y * w as usize + x] = ((x + y) & 0xFF) as u8;
            }
        }
        let cw = (w as usize) / 2;
        let ch = (h as usize) / 2;
        for y in 0..ch {
            for x in 0..cw {
                p.cb[y * cw + x] = (x & 0xFF) as u8;
                p.cr[y * cw + x] = (y & 0xFF) as u8;
            }
        }
        p
    }

    #[test]
    fn ibc_predict_copies_above_block() {
        // 32×32 pic, gradient Y[y][x] = (x + y) & 0xFF. Current CB at
        // (0, 16), 16×16; BV = (0, -16) → reference is the (0..16, 0..16)
        // top-left block.  Predicted Y[j][i] should equal (i + j) & 0xFF.
        let pic = make_pic_with_gradient(32, 32);
        let mv_l = mv_sixteenth_from_integer(0, -16);
        let mv_c = derive_ibc_chroma_mv(mv_l, 1);
        let mut py = vec![0i32; 16 * 16];
        let mut pcb = vec![0i32; 8 * 8];
        let mut pcr = vec![0i32; 8 * 8];
        predict_ibc_block(
            &pic, 0, 16, 16, 16, mv_l, mv_c, true, &mut py, &mut pcb, &mut pcr,
        )
        .expect("predict ok");
        for j in 0..16usize {
            for i in 0..16usize {
                assert_eq!(py[j * 16 + i], ((i + j) & 0xFF) as i32, "Y[{j}][{i}]");
            }
        }
        for j in 0..8usize {
            for i in 0..8usize {
                // Cb of source is x-only gradient.  Source chroma block
                // at (0, 8)..(8, 16) → reference at (0, 0)..(8, 8)
                // (chroma BV = (0, -8) thanks to 4:2:0).
                assert_eq!(pcb[j * 8 + i], i as i32, "Cb[{j}][{i}]");
                assert_eq!(pcr[j * 8 + i], j as i32, "Cr[{j}][{i}]");
            }
        }
    }

    #[test]
    fn ibc_predict_copies_left_block() {
        let pic = make_pic_with_gradient(32, 32);
        let mv_l = mv_sixteenth_from_integer(-16, 0);
        let mv_c = derive_ibc_chroma_mv(mv_l, 1);
        let mut py = vec![0i32; 16 * 16];
        let mut pcb = vec![0i32; 8 * 8];
        let mut pcr = vec![0i32; 8 * 8];
        predict_ibc_block(
            &pic, 16, 0, 16, 16, mv_l, mv_c, true, &mut py, &mut pcb, &mut pcr,
        )
        .expect("predict ok");
        // Reference is (0..16, 0..16). Predicted Y[j][i] = (i + j) & 0xFF.
        for j in 0..16usize {
            for i in 0..16usize {
                assert_eq!(py[j * 16 + i], ((i + j) & 0xFF) as i32);
            }
        }
    }

    #[test]
    fn ibc_predict_luma_only_when_chroma_absent() {
        let pic = make_pic_with_gradient(32, 32);
        let mv_l = mv_sixteenth_from_integer(0, -16);
        let mv_c = MotionVector { x: 0, y: 0 };
        let mut py = vec![0i32; 16 * 16];
        let mut pcb = vec![999i32; 8 * 8];
        let mut pcr = vec![999i32; 8 * 8];
        predict_ibc_block(
            &pic, 0, 16, 16, 16, mv_l, mv_c, false, &mut py, &mut pcb, &mut pcr,
        )
        .expect("predict ok");
        // pcb/pcr untouched (sentinel preserved).
        assert_eq!(pcb[0], 999);
        assert_eq!(pcr[0], 999);
    }

    #[test]
    fn ibc_predict_rejects_buffer_size_mismatch() {
        let pic = make_pic_with_gradient(32, 32);
        let mv_l = mv_sixteenth_from_integer(0, -16);
        let mv_c = MotionVector { x: 0, y: 0 };
        let mut py = vec![0i32; 8]; // wrong size
        let mut pcb = vec![0i32; 64];
        let mut pcr = vec![0i32; 64];
        let r = predict_ibc_block(
            &pic, 0, 16, 16, 16, mv_l, mv_c, true, &mut py, &mut pcb, &mut pcr,
        );
        assert!(r.is_err());
    }

    #[test]
    fn ibc_predict_rejects_fractional_luma_mv() {
        let pic = make_pic_with_gradient(32, 32);
        // mvL low-nibble non-zero → fractional.
        let mv_l = MotionVector { x: -257, y: -128 };
        let mv_c = derive_ibc_chroma_mv(mv_l, 1);
        let mut py = vec![0i32; 16 * 16];
        let mut pcb = vec![0i32; 64];
        let mut pcr = vec![0i32; 64];
        let r = predict_ibc_block(
            &pic, 16, 16, 16, 16, mv_l, mv_c, true, &mut py, &mut pcb, &mut pcr,
        );
        assert!(r.is_err());
    }

    // -------- end-to-end: derive → validate → predict --------

    #[test]
    fn ibc_pipeline_end_to_end_left_neighbour() {
        // 32×32 pic; current CB at (16, 0), 16×16; BV mvd = (-16, 0).
        let pic = make_pic_with_gradient(32, 32);
        let mvd = MotionVector { x: -16, y: 0 };
        let mv_l = derive_ibc_luma_mv(mvd);
        validate_ibc_constraints(mv_l, 16, 0, 16, 16, 5).expect("conformant BV");
        let mv_c = derive_ibc_chroma_mv(mv_l, pic.chroma_format_idc);
        let mut py = vec![0i32; 256];
        let mut pcb = vec![0i32; 64];
        let mut pcr = vec![0i32; 64];
        predict_ibc_block(
            &pic, 16, 0, 16, 16, mv_l, mv_c, true, &mut py, &mut pcb, &mut pcr,
        )
        .expect("predict ok");
        // Spot-check a corner: predicted Y[0][0] should be source (0,0) = 0.
        assert_eq!(py[0], 0);
        // Y[15][15] should be source (15, 15) = (15+15)&0xFF = 30.
        assert_eq!(py[15 * 16 + 15], 30);
    }

    // -------- decode_ibc_cu (§8.6.1 steps 1-3) --------

    #[test]
    fn ibc_decode_cu_chains_derive_validate_predict() {
        // Same pic + BV as the end-to-end test; verifies that the
        // pipeline returns the same (mvL, mvC) pair the individual
        // derivers do, AND fills the prediction buffer identically.
        let pic = make_pic_with_gradient(32, 32);
        let mvd = MotionVector { x: -16, y: 0 };
        let mut py = vec![0i32; 256];
        let mut pcb = vec![0i32; 64];
        let mut pcr = vec![0i32; 64];
        let (mv_l, mv_c) = decode_ibc_cu(
            &pic, 16, 0, 16, 16, mvd, 5, true, &mut py, &mut pcb, &mut pcr,
        )
        .expect("pipeline ok");
        assert_eq!(mv_l, derive_ibc_luma_mv(mvd));
        assert_eq!(mv_c, derive_ibc_chroma_mv(mv_l, pic.chroma_format_idc));
        // Corner spot-checks (mirroring the end-to-end test above).
        assert_eq!(py[0], 0);
        assert_eq!(py[15 * 16 + 15], 30);
    }

    #[test]
    fn ibc_decode_cu_rejects_non_conformant_bv() {
        // Same pic; BV (-8, 0) overlaps current CU → cond_left + cond_above
        // both false → validate_ibc_constraints rejects.
        let pic = make_pic_with_gradient(32, 32);
        let mvd = MotionVector { x: -8, y: 0 };
        let mut py = vec![0i32; 256];
        let mut pcb = vec![0i32; 64];
        let mut pcr = vec![0i32; 64];
        let r = decode_ibc_cu(
            &pic, 16, 16, 16, 16, mvd, 5, true, &mut py, &mut pcb, &mut pcr,
        );
        assert!(r.is_err(), "expected rejection, got {r:?}");
        // No samples touched on the early-out path.
        assert_eq!(py[0], 0);
    }

    #[test]
    fn ibc_decode_cu_luma_only_skips_chroma_buffers() {
        let pic = make_pic_with_gradient(32, 32);
        let mvd = MotionVector { x: -16, y: 0 };
        let mut py = vec![0i32; 256];
        let mut pcb = vec![999i32; 64]; // sentinel
        let mut pcr = vec![999i32; 64]; // sentinel
        let (mv_l, mv_c) = decode_ibc_cu(
            &pic, 16, 0, 16, 16, mvd, 5, false, &mut py, &mut pcb, &mut pcr,
        )
        .expect("pipeline ok");
        // Chroma MV is zero when chroma is not present.
        assert_eq!(mv_c, MotionVector { x: 0, y: 0 });
        // Sentinel preserved — chroma buffer never touched.
        assert_eq!(pcb[0], 999);
        assert_eq!(pcr[0], 999);
        // Luma still derived + predicted.
        assert_eq!(mv_l, derive_ibc_luma_mv(mvd));
        assert_eq!(py[0], 0);
    }

    // -------- is_ibc_allowed_for_size (§7.4.5 isIbcAllowed) --------

    #[test]
    fn ibc_size_gate_blocks_when_flag_off() {
        assert!(!is_ibc_allowed_for_size(false, 6, 4, 4));
    }

    #[test]
    fn ibc_size_gate_accepts_equal_to_limit() {
        // log2_max_ibc_cand_size = 6 → max block = 64. A 64×64 CB is fine.
        assert!(is_ibc_allowed_for_size(true, 6, 6, 6));
    }

    #[test]
    fn ibc_size_gate_rejects_larger_than_limit() {
        // 128×128 CB exceeds the 64-sample limit.
        assert!(!is_ibc_allowed_for_size(true, 6, 7, 7));
    }

    #[test]
    fn ibc_size_gate_independent_per_axis() {
        // 64-wide × 32-tall under a 64-sample limit is allowed (both ≤).
        assert!(is_ibc_allowed_for_size(true, 6, 6, 5));
        // But 128-wide × 32-tall is rejected purely on the width side.
        assert!(!is_ibc_allowed_for_size(true, 6, 7, 5));
    }
}
