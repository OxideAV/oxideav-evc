//! EVC intra **reference-sample construction + substitution** process
//! (ISO/IEC 23094-1:2020 §8.4.4.1 / §8.4.4.2).
//!
//! §8.4.4.1 *General* gathers the `p[ x ][ y ]` neighbourhood that the
//! per-mode kernels (§8.4.4.3 .. §8.4.4.10) consume: the top row
//! `p[ x ][ −1 ]`, the left column `p[ −1 ][ y ]`, the `p[ −1 ][ −1 ]`
//! corner and — on the `sps_suco_flag == 1` path — the right column
//! `p[ nCbW ][ y ]`. Each location is checked for availability with the
//! §6.4.1 neighbour-availability derivation (plus the
//! `constrained_intra_pred_flag` predicate) and either copied from the
//! post-reconstruction-filtered picture or marked *not available for
//! intra prediction*.
//!
//! §8.4.4.2 *Reference sample substitution* then fills every
//! not-available sample. The fill rule is profile-dependent:
//!
//! * `sps_eipd_flag == 0` — every hole takes the mid-level constant
//!   `1 << (bitDepth − 1)` (Baseline). The corner is filled first, then
//!   the top row left-to-right, then the left column top-to-bottom.
//! * `sps_eipd_flag == 1` — the corner takes `1 << (bitDepth − 1)` only
//!   when itself unavailable; every other hole copies its predecessor
//!   along the scan (`p[ x − 1 ][ y ]` for the top row, `p[ −1 ][ y − 1 ]`
//!   for the left column, `p[ nCbW ][ y − 1 ]` for the SUCO right column).
//!   This is the directional "copy-from-neighbour" fill of the EIPD set.
//!
//! The process is expressed over two caller-supplied closures so the
//! module stays pure (mirroring the `inter` / `merge` / `tmvp` purity
//! contract): an *availability* closure that resolves the §6.4.1
//! `availableN` (already folded with `constrained_intra_pred_flag`) at a
//! component-domain location, and a *sample* closure that returns the
//! reconstructed value at an available component-domain location. The
//! decoder wires its `IsCoded` raster, tile map and recon plane into
//! those two closures.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use crate::eipd::EipdRefSamples;

/// Outcome of constructing the §8.4.4.1 neighbourhood: the filled
/// [`EipdRefSamples`] together with whether any sample needed the
/// §8.4.4.2 substitution (useful for tests / diagnostics).
pub struct ConstructedRefs {
    /// The reference-sample neighbourhood, post-substitution and ready
    /// for the §8.4.4.3 .. §8.4.4.10 kernels.
    pub refs: EipdRefSamples,
    /// `true` when at least one location was *not available* and had to
    /// be filled by §8.4.4.2.
    pub substituted: bool,
}

/// §8.4.4.1 reference-sample construction followed by the §8.4.4.2
/// substitution.
///
/// Inputs (mirroring the §8.4.4.1 process inputs):
/// * `n_cb_w` / `n_cb_h` — coding-block width / height `nCbW` / `nCbH`.
/// * `bit_depth` — `BitDepthY` (cIdx 0) or `BitDepthC` (chroma), used by
///   the §8.4.4.2 `1 << (bitDepth − 1)` fill.
/// * `sps_eipd_flag` — selects the §8.4.4.2 mid-level (0) vs
///   copy-predecessor (1) substitution branch.
/// * `sps_suco_flag` — when 1, the right column `p[ nCbW ][ y ]` is also
///   constructed and substituted.
/// * `available` — closure `(x, y) -> bool` giving §6.4.1 `availableN`
///   at component-domain offset `(x, y)` relative to the block's
///   top-left (already folded with the `constrained_intra_pred_flag`
///   predicate of §8.4.4.1: the caller returns `false` when the
///   neighbour is non-intra and `constrained_intra_pred_flag == 1`).
/// * `sample` — closure `(x, y) -> i32` returning the reconstructed
///   sample at the available offset `(x, y)`. Only invoked for offsets
///   for which `available` returned `true`.
///
/// The offsets passed to the closures match the spec's `p[ x ][ y ]`
/// indices directly (`x`, `y` may be `−1`; the corner is `(−1, −1)`).
#[allow(clippy::too_many_arguments)]
pub fn construct_eipd_refs<A, S>(
    n_cb_w: usize,
    n_cb_h: usize,
    bit_depth: u32,
    sps_eipd_flag: bool,
    sps_suco_flag: bool,
    mut available: A,
    mut sample: S,
) -> ConstructedRefs
where
    A: FnMut(i32, i32) -> bool,
    S: FnMut(i32, i32) -> i32,
{
    let span = (n_cb_w + n_cb_h) as i32; // nCbW + nCbH
    let n_cb_w_i = n_cb_w as i32;

    let mut refs = EipdRefSamples::unavailable(n_cb_w, n_cb_h, bit_depth);

    // Availability masks, indexed with the same +1 origin as the
    // EipdRefSamples buffers (slot 0 == the `−1` element).
    // `top_avail[k]`   ⇔ p[ k − 1 ][ −1 ]
    // `left_avail[k]`  ⇔ p[ −1 ][ k − 1 ]
    // `right_avail[k]` ⇔ p[ nCbW ][ k − 1 ]
    let mask_len = 1 + n_cb_w + n_cb_h;
    let mut top_avail = vec![false; mask_len];
    let mut left_avail = vec![false; mask_len];
    let mut right_avail = vec![false; mask_len];

    // ---- §8.4.4.1 construction ----

    // Corner p[ −1 ][ −1 ].
    let corner_avail = available(-1, -1);
    if corner_avail {
        refs.set_top_left(sample(-1, -1));
    }
    top_avail[0] = corner_avail;
    left_avail[0] = corner_avail;

    // Top row p[ x ][ −1 ], x = 0 .. nCbW + nCbH − 1.
    for x in 0..span {
        let av = available(x, -1);
        if av {
            refs.set_top(x, sample(x, -1));
        }
        top_avail[(x + 1) as usize] = av;
    }

    // Left column p[ −1 ][ y ], y = 0 .. nCbH + nCbW − 1.
    for y in 0..span {
        let av = available(-1, y);
        if av {
            refs.set_left(y, sample(-1, y));
        }
        left_avail[(y + 1) as usize] = av;
    }

    // SUCO right column p[ nCbW ][ y ], y = 0 .. nCbH + nCbW − 1
    // (corner slot `−1` is unused on this column — §8.4.4.2 step 3 starts
    // at y = 0).
    if sps_suco_flag {
        for y in 0..span {
            let av = available(n_cb_w_i, y);
            if av {
                refs.set_right(y, sample(n_cb_w_i, y));
            }
            right_avail[(y + 1) as usize] = av;
        }
    }

    // ---- §8.4.4.2 substitution ----

    let any_missing = top_avail[..=(span as usize)].iter().any(|a| !a)
        || left_avail[1..=(span as usize)].iter().any(|a| !a)
        || (sps_suco_flag && right_avail[1..=(span as usize)].iter().any(|a| !a));

    if any_missing {
        if sps_eipd_flag {
            substitute_eipd(
                &mut refs,
                &top_avail,
                &left_avail,
                &right_avail,
                span,
                n_cb_w_i,
                bit_depth,
                sps_suco_flag,
            );
        } else {
            substitute_baseline(&mut refs, &top_avail, &left_avail, span, bit_depth);
        }
    }

    ConstructedRefs {
        refs,
        substituted: any_missing,
    }
}

/// §8.4.4.2 `sps_eipd_flag == 0` substitution: every hole takes the
/// mid-level constant `1 << (bitDepth − 1)`.
fn substitute_baseline(
    refs: &mut EipdRefSamples,
    top_avail: &[bool],
    left_avail: &[bool],
    span: i32,
    bit_depth: u32,
) {
    let mid = 1i32 << (bit_depth - 1);

    // Corner first.
    if !top_avail[0] {
        refs.set_top_left(mid);
    }
    // Top row x = 0 .. nCbW + nCbH − 1.
    for x in 0..span {
        if !top_avail[(x + 1) as usize] {
            refs.set_top(x, mid);
        }
    }
    // Left column y = 0 .. nCbH + nCbW − 1.
    for y in 0..span {
        if !left_avail[(y + 1) as usize] {
            refs.set_left(y, mid);
        }
    }
}

/// §8.4.4.2 `sps_eipd_flag == 1` substitution: the corner takes the
/// mid-level constant when unavailable; every other hole copies its
/// scan predecessor.
#[allow(clippy::too_many_arguments)]
fn substitute_eipd(
    refs: &mut EipdRefSamples,
    top_avail: &[bool],
    left_avail: &[bool],
    right_avail: &[bool],
    span: i32,
    n_cb_w: i32,
    bit_depth: u32,
    sps_suco_flag: bool,
) {
    let mid = 1i32 << (bit_depth - 1);

    // Corner p[ −1 ][ −1 ] → mid-level when unavailable.
    if !top_avail[0] {
        refs.set_top_left(mid);
    }

    // Step 1: top row, x = 0 .. nCbW + nCbH − 1, copy p[ x − 1 ][ −1 ].
    // x = 0's predecessor is the corner p[ −1 ][ −1 ].
    for x in 0..span {
        if !top_avail[(x + 1) as usize] {
            let prev = if x == 0 {
                refs.top_left()
            } else {
                refs.top(x - 1)
            };
            refs.set_top(x, prev);
        }
    }

    // Step 2: left column, y = 0 .. nCbH + nCbW − 1, copy p[ −1 ][ y − 1 ].
    // y = 0's predecessor is the corner p[ −1 ][ −1 ].
    for y in 0..span {
        if !left_avail[(y + 1) as usize] {
            let prev = if y == 0 {
                refs.top_left()
            } else {
                refs.left(y - 1)
            };
            refs.set_left(y, prev);
        }
    }

    // Step 3 (SUCO only): right column, y = 0 .. nCbH + nCbW − 1, copy
    // p[ nCbW ][ y − 1 ]. y = 0's predecessor is p[ nCbW ][ −1 ], which
    // for the right column is taken from the top row at x = nCbW (the
    // §8.4.4.1 construction marks `p[ nCbW ][ −1 ]` as part of the top
    // row span, since nCbW ≤ nCbW + nCbH − 1).
    if sps_suco_flag {
        for y in 0..span {
            if !right_avail[(y + 1) as usize] {
                let prev = if y == 0 {
                    // p[ nCbW ][ −1 ] lives in the top row at x = nCbW.
                    refs.top(n_cb_w)
                } else {
                    refs.right(y - 1)
                };
                refs.set_right(y, prev);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All neighbours available → the construction copies samples through
    /// verbatim and never substitutes.
    #[test]
    fn all_available_copies_through() {
        let n = 4usize;
        let out = construct_eipd_refs(
            n,
            n,
            8,
            true,
            false,
            |_x, _y| true,
            // Encode the location into the sample so we can verify routing.
            |x, y| (x + 1) * 100 + (y + 1),
        );
        assert!(!out.substituted);
        let r = &out.refs;
        // sample(x, y) = (x+1)*100 + (y+1).
        assert_eq!(r.top_left(), 0); // p[-1][-1] = 0*100 + 0
        assert_eq!(r.top(0), 100); // p[0][-1] = 1*100 + 0
        assert_eq!(r.top(3), 400); // p[3][-1] = 4*100 + 0
        assert_eq!(r.left(0), 1); // p[-1][0] = 0*100 + 1
        assert_eq!(r.left(5), 6); // p[-1][5] = 0*100 + 6 (span = 8)
    }

    /// `sps_eipd_flag == 0`: every hole becomes the mid-level constant.
    #[test]
    fn baseline_fills_midlevel() {
        let n = 4usize;
        let out = construct_eipd_refs(
            n,
            n,
            8,
            false,
            false,
            |_x, _y| false, // nothing available
            |_x, _y| unreachable!("no sample is available"),
        );
        assert!(out.substituted);
        let mid = 1i32 << 7;
        let r = &out.refs;
        assert_eq!(r.top_left(), mid);
        assert_eq!(r.top(0), mid);
        assert_eq!(r.top(7), mid);
        assert_eq!(r.left(0), mid);
        assert_eq!(r.left(7), mid);
    }

    /// `sps_eipd_flag == 1`: the corner is mid-level; the top row copies
    /// forward from the corner along the scan.
    #[test]
    fn eipd_copies_predecessor_top_row() {
        let n = 4usize;
        // Make only the corner unavailable; the rest of the top row also
        // unavailable so the copy chains from the mid-level corner.
        let out = construct_eipd_refs(
            n,
            n,
            8,
            true,
            false,
            |x, y| {
                // left column available (so it's not the focus), top row
                // entirely unavailable including corner.
                x == -1 && y >= 0
            },
            |x, y| {
                assert!(x == -1 && y >= 0);
                y + 10
            },
        );
        assert!(out.substituted);
        let mid = 1i32 << 7;
        let r = &out.refs;
        // Corner unavailable → mid-level.
        assert_eq!(r.top_left(), mid);
        // Whole top row unavailable → each copies its predecessor, all
        // ending up at the mid-level corner.
        for x in 0..(2 * n as i32) {
            assert_eq!(r.top(x), mid, "top[{x}]");
        }
        // Left column was available → carried through.
        assert_eq!(r.left(0), 10);
        assert_eq!(r.left(3), 13);
    }

    /// `sps_eipd_flag == 1`: a hole in the middle of an otherwise
    /// available top row copies the immediately preceding sample.
    #[test]
    fn eipd_hole_copies_left_neighbour() {
        let n = 4usize;
        let out = construct_eipd_refs(
            n,
            n,
            8,
            true,
            false,
            // top row available except x == 2; corner available.
            |x, y| (y == -1 && x != 2) || (x == -1),
            |x, y| {
                if y == -1 {
                    // corner (x == -1) and top row.
                    x + 50
                } else {
                    100
                }
            },
        );
        assert!(out.substituted);
        let r = &out.refs;
        // p[1][-1] available = 51, p[2][-1] hole copies p[1][-1] = 51.
        assert_eq!(r.top(1), 51);
        assert_eq!(r.top(2), 51);
        assert_eq!(r.top(3), 53);
    }

    /// SUCO path constructs and substitutes the right column.
    #[test]
    fn suco_right_column() {
        let n = 4usize;
        let out = construct_eipd_refs(
            n,
            n,
            8,
            true,
            true,
            // Everything available except the right column → it copies.
            |x, y| x != n as i32 || y < 0,
            |x, y| {
                if x == n as i32 {
                    // p[nCbW][-1] (the y<0 available one): lives in top row.
                    200
                } else {
                    (x + 1) * 1000 + (y + 1)
                }
            },
        );
        assert!(out.substituted);
        let r = &out.refs;
        // p[nCbW][-1] is part of the top row at x = nCbW = 4 → value 200.
        assert_eq!(r.top(4), 200);
        // Right column y=0 copies p[nCbW][-1] = top(4) = 200.
        assert_eq!(r.right(0), 200);
        // Subsequent right-column holes chain forward → still 200.
        assert_eq!(r.right(3), 200);
    }
}
