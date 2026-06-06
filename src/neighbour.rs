//! EVC neighbouring-block availability derivations (ISO/IEC
//! 23094-1:2020 §6.4).
//!
//! Round 242 lands the **§6.4.2 `availLR` (left/right) derivation**
//! plus the four named return values `LR_00`, `LR_10`, `LR_01`,
//! `LR_11` from the section's closing paragraph.
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
}
