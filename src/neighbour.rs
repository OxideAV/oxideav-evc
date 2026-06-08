//! EVC neighbouring-block availability derivations (ISO/IEC
//! 23094-1:2020 §6.4).
//!
//! Round 242 landed the **§6.4.2 `availLR` (left/right) derivation**
//! plus the four named return values `LR_00`, `LR_10`, `LR_01`,
//! `LR_11` from the section's closing paragraph.
//!
//! Round 258 adds the two companion availability derivations in
//! §6.4.3 (motion-vector candidate availability) and §6.4.4 (ALF
//! availability). Both take a luma location and a small predicate
//! bundle, return a single `availableN` boolean per the spec's
//! "if any of the conditions hold, FALSE; otherwise TRUE" bullet
//! lists.
//!
//! ## §6.4.3 vs §6.4.4 vs §6.4.1
//!
//! The three "single-block availability" derivations share five
//! common falsification conditions but differ on whether a
//! tile-boundary check applies:
//!
//! | Predicate                                | §6.4.1 | §6.4.3 | §6.4.4 |
//! |------------------------------------------|--------|--------|--------|
//! | different tile than current block        | yes    | yes    | no     |
//! | `xNbY < 0`                               | yes    | yes    | yes    |
//! | `yNbY < 0`                               | yes    | yes    | yes    |
//! | `xNbY >= pic_width_in_luma_samples`      | yes    | yes    | yes    |
//! | `yNbY >= pic_height_in_luma_samples`     | yes    | yes    | yes    |
//! | `IsCoded[xNbY][yNbY] == FALSE`           | yes    | yes    | yes    |
//! | neighbour coded in intra / IBC mode      | -      | yes    | yes    |
//!
//! §6.4.3 differs from §6.4.1 by adding the "intra/IBC" disqualifier
//! (since the MV candidate from an intra-coded neighbour is not a
//! real motion vector). §6.4.4 is §6.4.3 minus the tile-boundary
//! check — the ALF filter is designed to reach across tile
//! boundaries when the §7.4.5 `alf_loop_filter_across_tiles_enabled_flag`
//! permits it, so the §6.4.4 derivation deliberately omits the
//! tile-different predicate. (The flag itself is consulted by the
//! ALF caller before invoking §6.4.4, not inside it.)
//!
//! ## Spec scope
//!
//! §6.4.2 turns two per-neighbour availability booleans
//! (`availableL`, `availableR`) — each the result of a §6.4.1
//! invocation at the left and right luma locations of the current
//! block — into a single packed `availLR` token used by every
//! `sps_suco_flag = 1`-aware merge / AMVP / partitioning derivation
//! (§8.5.2.4.5.2 DefaultRefIdxLX, §8.5.2.3.2 spatial merging,
//! §8.5.3.5 affine CP MVP, etc.).
//!
//! The §6.4.2 input semantics are:
//!
//! ```text
//! xNbL = xCurr − 1                                  (text bullet 1)
//! yNbL = yCurr                                      (text bullet 1)
//! xNbR = xCurr + nCbW                               (text bullet 2)
//! yNbR = yCurr                                      (text bullet 2)
//! availableL = invoke §6.4.1 on (xNbL, yNbL)        (text bullet 3)
//! availableR = invoke §6.4.1 on (xNbR, yNbR)        (text bullet 4)
//! availLR = availableL + availableR * 2             (eq. 23)
//! ```
//!
//! and the named constants (closing paragraph of §6.4.2):
//!
//! | `availLR` | label   | left | right |
//! |-----------|---------|------|-------|
//! | 0         | `LR_00` | F    | F     |
//! | 1         | `LR_10` | T    | F     |
//! | 2         | `LR_01` | F    | T     |
//! | 3         | `LR_11` | T    | T     |
//!
//! Note the spec's `LR_LR` ordering: the **first** digit names the
//! **left** neighbour and the **second** digit names the **right**
//! neighbour. `LR_10` is "left available, right not available"; the
//! 1-tuple is read left-to-right, not as a bit-position weight.
//!
//! The closing paragraph adds an invariant:
//!
//! > *"When `sps_suco_flag` is equal to 0, `availLR` is always
//! > smaller than 2 (`availLR` is equal to `LR_10` or `availLR` is
//! > equal to `LR_00`)."*
//!
//! i.e. with SUCO disabled the right neighbour is structurally
//! unreachable and `availLR ∈ { LR_00, LR_10 }`. We expose this as
//! `AvailLr::is_suco_consistent(sps_suco_flag)`.
//!
//! ## Wiring stance
//!
//! Same opt-in posture as the round-218 / 223 / 229 / 232 / 237
//! helper rollouts: pure functions + a typed token (`AvailLr`), no
//! behaviour change to existing decoder paths. Callers that already
//! derive their own left/right neighbour availability (e.g. the
//! AMVP builder in `inter.rs`) keep doing so; future §8.5.2.4.5.2
//! and §8.5.3.5 derivations will consume `AvailLr` directly once
//! they land.
//!
//! `§6.4.1` (neighbouring-block availability) is **not** wrapped in
//! this round: its bullet list mixes tile-boundary lookup, the
//! `IsCoded[][]` raster, and "coded in intra / IBC mode" predicates
//! that the caller already has on hand. `§6.4.2` takes the two
//! booleans as inputs, exactly as the spec invokes §6.4.1 inline.
//!
//! For the same reason, [`derive_mv_candidate_availability`] (§6.4.3)
//! and [`derive_alf_availability`] (§6.4.4) take their per-bullet
//! predicates as boolean / location-pair inputs, leaving the
//! `IsCoded[][]` raster and the "coded in intra / IBC mode" lookup
//! on the caller. They differ structurally only in whether they
//! consult the tile boundary; see the table above.
//!
//! All section / clause numbers refer to **ISO/IEC 23094-1:2020(E)**.

/// Packed `availLR` token returned by [`derive_avail_lr`] — the
/// §6.4.2 eq. (23) value `availableL + availableR * 2` with the four
/// named tokens from the section's closing paragraph.
///
/// The discriminant is the spec's numeric value; pattern-match on
/// the named variants for readability or compare `as u8` to test
/// the eq. (23) integer directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AvailLr {
    /// `availableL = false`, `availableR = false`. Eq. (23) yields 0.
    Lr00 = 0,
    /// `availableL = true`, `availableR = false`. Eq. (23) yields 1.
    /// This is the `sps_suco_flag == 0` "default" shape — every
    /// block has at most a left neighbour, never a right neighbour.
    Lr10 = 1,
    /// `availableL = false`, `availableR = true`. Eq. (23) yields 2.
    /// Reachable only when `sps_suco_flag == 1` (the spec's split
    /// unit coding order can flip a CU's right neighbour into
    /// "already coded" before its left).
    Lr01 = 2,
    /// `availableL = true`, `availableR = true`. Eq. (23) yields 3.
    /// Reachable only when `sps_suco_flag == 1`.
    Lr11 = 3,
}

impl AvailLr {
    /// `availableL` projection (the eq. (23) low bit).
    #[inline]
    pub fn available_l(self) -> bool {
        matches!(self, Self::Lr10 | Self::Lr11)
    }

    /// `availableR` projection (the eq. (23) high bit).
    #[inline]
    pub fn available_r(self) -> bool {
        matches!(self, Self::Lr01 | Self::Lr11)
    }

    /// Raw `availLR` integer per eq. (23), in range `0..=3`. Equal to
    /// `availableL as u8 + (availableR as u8) * 2`.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Reconstruct from the eq. (23) integer. Returns `None` if
    /// `value > 3` (the spec never produces such a token).
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Lr00),
            1 => Some(Self::Lr10),
            2 => Some(Self::Lr01),
            3 => Some(Self::Lr11),
            _ => None,
        }
    }

    /// `True` when this `availLR` value is consistent with the
    /// `sps_suco_flag` setting per the closing paragraph of §6.4.2.
    ///
    /// * `sps_suco_flag == 0` ⇒ `availLR < 2` (only `Lr00` / `Lr10`
    ///   are reachable).
    /// * `sps_suco_flag == 1` ⇒ all four tokens are reachable.
    ///
    /// Non-zero `sps_suco_flag` values other than `1` are treated as
    /// "SUCO enabled" — the spec defines the flag as a single bit
    /// (§7.4.3.1) and any clamping happens at parse time.
    #[inline]
    pub fn is_suco_consistent(self, sps_suco_flag: u8) -> bool {
        if sps_suco_flag == 0 {
            self.as_u8() < 2
        } else {
            true
        }
    }
}

/// §6.4.2 derivation process — eq. (23).
///
/// Inputs are the two §6.4.1 outputs at the left
/// (`xCurr − 1, yCurr`) and right (`xCurr + nCbW, yCurr`) luma
/// locations. Output is the packed `availLR` token.
///
/// The caller is responsible for invoking §6.4.1 at those locations
/// — that derivation needs the tile map, the `IsCoded[][]` raster
/// and the "intra / IBC mode" predicates, all of which already live
/// on the slice walker.
#[inline]
pub fn derive_avail_lr(available_l: bool, available_r: bool) -> AvailLr {
    // Eq. (23): availLR = availableL + availableR * 2.
    let raw = available_l as u8 + (available_r as u8) * 2;
    // `raw` is in 0..=3 by construction; `expect` documents the
    // invariant.
    AvailLr::from_u8(raw).expect("availableL + availableR * 2 is in 0..=3")
}

/// §6.4.3 derivation process — *Derivation process for neighbouring
/// block motion vector candidate availability.*
///
/// Returns `availableN` for the luma location `(x_nb_y, y_nb_y)`
/// covered by a neighbouring block. The spec sets `availableN =
/// FALSE` when any of the seven bullets in §6.4.3 holds, otherwise
/// `TRUE`. The bullets are evaluated in declaration order.
///
/// The five "geometric" bullets — tile-different, two negative-index
/// bounds, two greater-than-or-equal picture-extent bounds — are
/// resolved here from caller-supplied location + picture dimensions.
/// The two "raster" bullets — `IsCoded[][] == FALSE` and "the
/// neighbouring block is coded in intra or intra block copy mode" —
/// are taken as already-looked-up booleans. That mirrors the
/// §6.4.2 contract: §6.4.3 sits between the slice walker and the
/// merge / AMVP / affine candidate builders, all of which have the
/// `IsCoded[][]` raster and the per-block prediction-mode flag on
/// hand.
///
/// Inputs:
/// * `x_nb_y`, `y_nb_y` — luma location of the neighbouring block,
///   in signed coordinates so the spec's `< 0` bullets are
///   expressible.
/// * `pic_width_in_luma_samples`, `pic_height_in_luma_samples` —
///   §7.4.3.1 picture extents (in luma samples).
/// * `neighbour_in_different_tile` — caller's resolution of the
///   §6.4.3 first bullet (the neighbouring block sits in a different
///   tile than the current block).
/// * `is_coded` — caller's lookup of `IsCoded[x_nb_y][y_nb_y]`.
/// * `neighbour_is_intra_or_ibc` — caller's lookup of the
///   neighbour's prediction-mode flag against `MODE_INTRA` and
///   `MODE_IBC` from §7.4.9.4.
///
/// Output is the single `availableN` boolean.
///
/// # Examples
///
/// ```
/// use oxideav_evc::neighbour::derive_mv_candidate_availability;
/// // Eight-sample-wide picture, top-left CTB: a neighbour at
/// // (-1, 0) is out-of-bounds → unavailable.
/// assert!(!derive_mv_candidate_availability(
///     -1, 0, 8, 8, false, true, false,
/// ));
/// ```
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn derive_mv_candidate_availability(
    x_nb_y: i32,
    y_nb_y: i32,
    pic_width_in_luma_samples: u32,
    pic_height_in_luma_samples: u32,
    neighbour_in_different_tile: bool,
    is_coded: bool,
    neighbour_is_intra_or_ibc: bool,
) -> bool {
    // §6.4.3 first bullet: different tile than the current block.
    if neighbour_in_different_tile {
        return false;
    }
    // §6.4.3 bullets 2-3: x_nb_y < 0, y_nb_y < 0.
    if x_nb_y < 0 || y_nb_y < 0 {
        return false;
    }
    // §6.4.3 bullets 4-5: x_nb_y >= pic_width_in_luma_samples,
    // y_nb_y >= pic_height_in_luma_samples. Compared in u32 after
    // the negative-index bullets cleared.
    if (x_nb_y as u32) >= pic_width_in_luma_samples || (y_nb_y as u32) >= pic_height_in_luma_samples
    {
        return false;
    }
    // §6.4.3 bullet 6: IsCoded[ xNbY ][ yNbY ] == FALSE.
    if !is_coded {
        return false;
    }
    // §6.4.3 bullet 7: the neighbouring block is coded in intra or
    // intra block copy mode.
    if neighbour_is_intra_or_ibc {
        return false;
    }
    true
}

/// §6.4.4 derivation process — *Derivation process for ALF
/// neighbouring block availability.*
///
/// Returns `availableN` for the luma location `(x_nb_y, y_nb_y)`
/// covered by a neighbouring block. Structurally §6.4.4 is §6.4.3
/// minus the tile-boundary bullet: ALF deliberately reaches across
/// tile boundaries when the §7.4.5
/// `alf_loop_filter_across_tiles_enabled_flag` permits it, so the
/// §6.4.4 derivation never disqualifies a neighbour for sitting in
/// a different tile. The flag itself is consulted by the ALF
/// caller before invoking §6.4.4; this function evaluates only the
/// six bullets the spec lists in §6.4.4.
///
/// Inputs and outputs mirror [`derive_mv_candidate_availability`]
/// modulo the dropped `neighbour_in_different_tile` parameter.
///
/// # Examples
///
/// ```
/// use oxideav_evc::neighbour::derive_alf_availability;
/// // Coded inter-neighbour inside the picture → available.
/// assert!(derive_alf_availability(4, 4, 16, 16, true, false));
/// // Same neighbour but intra-coded → unavailable per §6.4.4.
/// assert!(!derive_alf_availability(4, 4, 16, 16, true, true));
/// ```
#[inline]
pub fn derive_alf_availability(
    x_nb_y: i32,
    y_nb_y: i32,
    pic_width_in_luma_samples: u32,
    pic_height_in_luma_samples: u32,
    is_coded: bool,
    neighbour_is_intra_or_ibc: bool,
) -> bool {
    // §6.4.4 bullets 1-2: x_nb_y < 0, y_nb_y < 0.
    if x_nb_y < 0 || y_nb_y < 0 {
        return false;
    }
    // §6.4.4 bullets 3-4: picture-extent bounds.
    if (x_nb_y as u32) >= pic_width_in_luma_samples || (y_nb_y as u32) >= pic_height_in_luma_samples
    {
        return false;
    }
    // §6.4.4 bullet 5: IsCoded[ xNbY ][ yNbY ] == FALSE.
    if !is_coded {
        return false;
    }
    // §6.4.4 bullet 6: intra / IBC neighbour.
    if neighbour_is_intra_or_ibc {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- eq. (23) full truth table ------------------------------

    /// Eq. (23) `0 + 0 * 2 == 0` ⇒ `LR_00`.
    #[test]
    fn round242_eq23_both_unavailable_is_lr00() {
        let lr = derive_avail_lr(false, false);
        assert_eq!(lr, AvailLr::Lr00);
        assert_eq!(lr.as_u8(), 0);
    }

    /// Eq. (23) `1 + 0 * 2 == 1` ⇒ `LR_10`. Confirms the spec's
    /// LR-label convention: the first digit names the left neighbour.
    #[test]
    fn round242_eq23_left_only_is_lr10() {
        let lr = derive_avail_lr(true, false);
        assert_eq!(lr, AvailLr::Lr10);
        assert_eq!(lr.as_u8(), 1);
    }

    /// Eq. (23) `0 + 1 * 2 == 2` ⇒ `LR_01`. Confirms the right-only
    /// path resolves the higher numeric value, since `availableR` is
    /// weighted by `* 2`.
    #[test]
    fn round242_eq23_right_only_is_lr01() {
        let lr = derive_avail_lr(false, true);
        assert_eq!(lr, AvailLr::Lr01);
        assert_eq!(lr.as_u8(), 2);
    }

    /// Eq. (23) `1 + 1 * 2 == 3` ⇒ `LR_11`.
    #[test]
    fn round242_eq23_both_available_is_lr11() {
        let lr = derive_avail_lr(true, true);
        assert_eq!(lr, AvailLr::Lr11);
        assert_eq!(lr.as_u8(), 3);
    }

    // -------- projection invariants ----------------------------------

    /// `available_l` / `available_r` reproduce the eq. (23) inputs
    /// for every truth-table row.
    #[test]
    fn round242_projections_invert_derivation() {
        for l in [false, true] {
            for r in [false, true] {
                let lr = derive_avail_lr(l, r);
                assert_eq!(lr.available_l(), l, "l={l}, r={r}");
                assert_eq!(lr.available_r(), r, "l={l}, r={r}");
            }
        }
    }

    /// The packed integer matches the eq. (23) formula for every row.
    #[test]
    fn round242_as_u8_matches_eq23_formula() {
        for l in [false, true] {
            for r in [false, true] {
                let lr = derive_avail_lr(l, r);
                let expected = l as u8 + (r as u8) * 2;
                assert_eq!(lr.as_u8(), expected);
            }
        }
    }

    // -------- from_u8 round-trip + out-of-range --------------------

    /// `from_u8` recovers the same variant for every legal value.
    #[test]
    fn round242_from_u8_round_trip() {
        for variant in [AvailLr::Lr00, AvailLr::Lr10, AvailLr::Lr01, AvailLr::Lr11] {
            assert_eq!(AvailLr::from_u8(variant.as_u8()), Some(variant));
        }
    }

    /// Values outside the eq. (23) range surface `None` — the spec
    /// never produces them.
    #[test]
    fn round242_from_u8_rejects_out_of_range() {
        for v in 4u8..=255 {
            assert!(AvailLr::from_u8(v).is_none(), "v={v}");
        }
    }

    // -------- closing-paragraph invariant -----------------------------

    /// Closing paragraph: when `sps_suco_flag == 0`, only `LR_00`
    /// and `LR_10` are reachable.
    #[test]
    fn round242_suco_off_admits_only_lr00_and_lr10() {
        assert!(AvailLr::Lr00.is_suco_consistent(0));
        assert!(AvailLr::Lr10.is_suco_consistent(0));
        assert!(!AvailLr::Lr01.is_suco_consistent(0));
        assert!(!AvailLr::Lr11.is_suco_consistent(0));
    }

    /// Closing paragraph: when `sps_suco_flag == 1`, every token is
    /// reachable.
    #[test]
    fn round242_suco_on_admits_every_token() {
        for variant in [AvailLr::Lr00, AvailLr::Lr10, AvailLr::Lr01, AvailLr::Lr11] {
            assert!(variant.is_suco_consistent(1), "variant={variant:?}");
        }
    }

    // -------- documentary regression -------------------------------

    /// The four `LR_xx` discriminants are exactly the eq. (23)
    /// integer for the corresponding `(availableL, availableR)`
    /// pair. Pinned as a regression in case anyone reorders the
    /// `repr(u8)` variants.
    #[test]
    fn round242_discriminants_match_spec_table() {
        assert_eq!(AvailLr::Lr00 as u8, 0);
        assert_eq!(AvailLr::Lr10 as u8, 1);
        assert_eq!(AvailLr::Lr01 as u8, 2);
        assert_eq!(AvailLr::Lr11 as u8, 3);
    }

    // ============================================================
    // §6.4.3 — neighbouring block MV candidate availability
    // ============================================================

    /// All-good interior neighbour: not in another tile, in-bounds,
    /// coded, inter-mode → `availableN = TRUE`.
    #[test]
    fn round258_eq643_all_good_interior_is_available() {
        assert!(derive_mv_candidate_availability(
            4, 4, 16, 16, false, true, false,
        ));
    }

    /// Bullet 1: neighbour in a different tile disqualifies.
    #[test]
    fn round258_eq643_different_tile_disqualifies() {
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, true, true, false,
        ));
    }

    /// Bullets 2-3: negative x or negative y disqualifies.
    #[test]
    fn round258_eq643_negative_coords_disqualify() {
        // x < 0
        assert!(!derive_mv_candidate_availability(
            -1, 4, 16, 16, false, true, false,
        ));
        // y < 0
        assert!(!derive_mv_candidate_availability(
            4, -1, 16, 16, false, true, false,
        ));
        // both
        assert!(!derive_mv_candidate_availability(
            -3, -5, 16, 16, false, true, false,
        ));
    }

    /// Bullets 4-5: ≥-extent disqualifies (and equality is the
    /// inclusive boundary, not exclusive — the spec uses `≥`).
    #[test]
    fn round258_eq643_oob_picture_extent_disqualifies() {
        // x exactly at width → unavailable
        assert!(!derive_mv_candidate_availability(
            16, 4, 16, 16, false, true, false,
        ));
        // x beyond width
        assert!(!derive_mv_candidate_availability(
            32, 4, 16, 16, false, true, false,
        ));
        // y exactly at height → unavailable
        assert!(!derive_mv_candidate_availability(
            4, 16, 16, 16, false, true, false,
        ));
        // y beyond height
        assert!(!derive_mv_candidate_availability(
            4, 99, 16, 16, false, true, false,
        ));
    }

    /// Bullet 6: `IsCoded[][] == FALSE` disqualifies.
    #[test]
    fn round258_eq643_uncoded_neighbour_disqualifies() {
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, false, false, false,
        ));
    }

    /// Bullet 7: intra or IBC neighbour disqualifies the MV
    /// candidate — the §6.4.3 motivation, since intra-coded blocks
    /// have no real motion vector.
    #[test]
    fn round258_eq643_intra_or_ibc_neighbour_disqualifies() {
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, false, true, true,
        ));
    }

    /// All seven bullets check independently — flip any one of them
    /// to "disqualifying" on top of an otherwise-good neighbour and
    /// the result becomes `FALSE`.
    #[test]
    fn round258_eq643_each_bullet_independently_disqualifies() {
        // Baseline good neighbour.
        let base = || derive_mv_candidate_availability(4, 4, 16, 16, false, true, false);
        assert!(base());

        // Flip bullet 1.
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, true, true, false,
        ));
        // Flip bullet 2 (x < 0).
        assert!(!derive_mv_candidate_availability(
            -1, 4, 16, 16, false, true, false,
        ));
        // Flip bullet 3 (y < 0).
        assert!(!derive_mv_candidate_availability(
            4, -1, 16, 16, false, true, false,
        ));
        // Flip bullet 4 (x >= pic_width).
        assert!(!derive_mv_candidate_availability(
            16, 4, 16, 16, false, true, false,
        ));
        // Flip bullet 5 (y >= pic_height).
        assert!(!derive_mv_candidate_availability(
            4, 16, 16, 16, false, true, false,
        ));
        // Flip bullet 6 (is_coded == false).
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, false, false, false,
        ));
        // Flip bullet 7 (intra/IBC neighbour).
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, false, true, true,
        ));
    }

    /// Top-left CTB (x=0, y=0) is itself in-bounds when the picture
    /// is non-empty. Pins the zero-coordinate edge of the §6.4.3
    /// negative-index bullets.
    #[test]
    fn round258_eq643_origin_is_in_bounds() {
        assert!(derive_mv_candidate_availability(
            0, 0, 1, 1, false, true, false,
        ));
    }

    // ============================================================
    // §6.4.4 — ALF neighbouring block availability
    // ============================================================

    /// All-good interior ALF neighbour → available.
    #[test]
    fn round258_eq644_all_good_interior_is_available() {
        assert!(derive_alf_availability(4, 4, 16, 16, true, false));
    }

    /// §6.4.4 does NOT include the tile-different bullet — a
    /// neighbour in another tile is still available so far as ALF
    /// is concerned (the §7.4.5 cross-tile flag is consulted by the
    /// caller, not §6.4.4).
    ///
    /// This is the defining structural difference between §6.4.3
    /// and §6.4.4. Pinned explicitly because flipping it would
    /// silently change ALF behaviour at tile seams.
    #[test]
    fn round258_eq644_does_not_consult_tile_boundary() {
        // §6.4.4 has no `neighbour_in_different_tile` input by
        // construction. The only way to express the "same neighbour
        // sits in a different tile" case is to invoke §6.4.4 with
        // its natural inputs and confirm the result is independent
        // of that bit — which is shown by §6.4.3 returning FALSE
        // (bullet 1) for the same inputs that §6.4.4 returns TRUE
        // for.
        assert!(derive_alf_availability(4, 4, 16, 16, true, false));
        assert!(!derive_mv_candidate_availability(
            4, 4, 16, 16, true, true, false,
        ));
    }

    /// Bullets 1-2: negative x or negative y disqualifies.
    #[test]
    fn round258_eq644_negative_coords_disqualify() {
        assert!(!derive_alf_availability(-1, 4, 16, 16, true, false));
        assert!(!derive_alf_availability(4, -1, 16, 16, true, false));
    }

    /// Bullets 3-4: ≥-extent disqualifies (inclusive boundary).
    #[test]
    fn round258_eq644_oob_picture_extent_disqualifies() {
        assert!(!derive_alf_availability(16, 4, 16, 16, true, false));
        assert!(!derive_alf_availability(4, 16, 16, 16, true, false));
    }

    /// Bullet 5: uncoded neighbour disqualifies.
    #[test]
    fn round258_eq644_uncoded_neighbour_disqualifies() {
        assert!(!derive_alf_availability(4, 4, 16, 16, false, false));
    }

    /// Bullet 6: intra / IBC neighbour disqualifies.
    #[test]
    fn round258_eq644_intra_or_ibc_neighbour_disqualifies() {
        assert!(!derive_alf_availability(4, 4, 16, 16, true, true));
    }

    /// All six bullets check independently.
    #[test]
    fn round258_eq644_each_bullet_independently_disqualifies() {
        // Baseline good neighbour.
        assert!(derive_alf_availability(4, 4, 16, 16, true, false));

        // Flip bullet 1.
        assert!(!derive_alf_availability(-1, 4, 16, 16, true, false));
        // Flip bullet 2.
        assert!(!derive_alf_availability(4, -1, 16, 16, true, false));
        // Flip bullet 3.
        assert!(!derive_alf_availability(16, 4, 16, 16, true, false));
        // Flip bullet 4.
        assert!(!derive_alf_availability(4, 16, 16, 16, true, false));
        // Flip bullet 5.
        assert!(!derive_alf_availability(4, 4, 16, 16, false, false));
        // Flip bullet 6.
        assert!(!derive_alf_availability(4, 4, 16, 16, true, true));
    }

    /// §6.4.4 origin in-bounds for a 1×1 picture, matching the
    /// §6.4.3 origin pin.
    #[test]
    fn round258_eq644_origin_is_in_bounds() {
        assert!(derive_alf_availability(0, 0, 1, 1, true, false));
    }

    // ============================================================
    // §6.4.3 vs §6.4.4 structural contrast
    // ============================================================

    /// §6.4.3 and §6.4.4 must agree whenever the §6.4.3 tile-bullet
    /// input is FALSE. Pinned over a small sweep of representative
    /// (location, coded, intra) tuples.
    #[test]
    fn round258_eq643_and_eq644_agree_when_same_tile() {
        let cases: &[(i32, i32, u32, u32, bool, bool)] = &[
            // origin, coded inter
            (0, 0, 16, 16, true, false),
            // interior, coded inter
            (4, 4, 16, 16, true, false),
            // interior, intra → both FALSE
            (4, 4, 16, 16, true, true),
            // interior, uncoded → both FALSE
            (4, 4, 16, 16, false, false),
            // OOB x → both FALSE
            (32, 4, 16, 16, true, false),
            // OOB y → both FALSE
            (4, 32, 16, 16, true, false),
            // x at width (inclusive) → both FALSE
            (16, 0, 16, 16, true, false),
            // y at height (inclusive) → both FALSE
            (0, 16, 16, 16, true, false),
            // negative x → both FALSE
            (-1, 4, 16, 16, true, false),
            // negative y → both FALSE
            (4, -1, 16, 16, true, false),
        ];
        for &(x, y, w, h, coded, intra) in cases {
            let mv = derive_mv_candidate_availability(x, y, w, h, false, coded, intra);
            let alf = derive_alf_availability(x, y, w, h, coded, intra);
            assert_eq!(
                mv, alf,
                "§6.4.3 / §6.4.4 disagreement at (x={x},y={y},w={w},h={h},coded={coded},intra={intra})",
            );
        }
    }

    /// §6.4.3 returns FALSE and §6.4.4 returns TRUE when the only
    /// failing bullet on §6.4.3 is bullet-1 (different tile). Pins
    /// the one structural input difference.
    #[test]
    fn round258_eq643_and_eq644_diverge_only_on_tile_bullet() {
        // Otherwise-good neighbour with `neighbour_in_different_tile = true`.
        let mv = derive_mv_candidate_availability(4, 4, 16, 16, true, true, false);
        let alf = derive_alf_availability(4, 4, 16, 16, true, false);
        assert!(!mv);
        assert!(alf);
    }
}
