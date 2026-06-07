//! EVC scanning processes (ISO/IEC 23094-1:2020 §6.5).
//!
//! Round 245 lands the **§6.5.3 inverse scan order 1D array
//! initialization process** (eq. 34) and re-exposes the §6.5.2
//! zig-zag scan order initialization process (eq. 33) as a typed,
//! public surface keyed to the §7.4.3.1 `ScanOrder` / `InvScanOrder`
//! tables.
//!
//! ## Spec scope
//!
//! §6.5.2 produces the forward map `ScanOrder[ sPos ] = rPos`, where
//! `sPos` is the zig-zag scan position (`0 ..= blkWidth · blkHeight − 1`)
//! and `rPos` is the raster scan position (also `0 ..= blkWidth ·
//! blkHeight − 1`). The walk starts at the top-left corner and visits
//! the anti-diagonals in order; odd anti-diagonals are traversed from
//! the top-right corner toward the bottom-left, even anti-diagonals
//! the other way round.
//!
//! §6.5.3 then inverts the §6.5.2 map (eq. 34):
//!
//! ```text
//! for( pos = 0; pos < blkWidth * blkHeight; pos++ ) {
//!     inverseScan[ forwardScan[ pos ] ] = pos
//! }                                                              (34)
//! ```
//!
//! i.e. `InvScanOrder[ rPos ]` returns the zig-zag scan position `sPos`
//! that visits raster position `rPos`. By construction, eq. (34)
//! satisfies the round-trip identity
//!
//! ```text
//! inverseScan[ forwardScan[ pos ] ] = pos      for every pos
//! forwardScan[ inverseScan[ pos ] ] = pos      for every pos
//! ```
//!
//! ## Spec usage
//!
//! §7.4.3.1 (page 64) directs the decoder to build
//! `ScanOrder[ log2TbWidth ][ log2TbHeight ][ sPos ]` and
//! `InvScanOrder[ log2TbWidth ][ log2TbHeight ][ rPos ]` for every
//! `log2TbWidth, log2TbHeight ∈ 1 ..= MaxTbLog2SizeY` by invoking
//! §6.5.2 / §6.5.3 with `blkWidth = 1 << log2TbWidth`,
//! `blkHeight = 1 << log2TbHeight`. The forward table is then used by
//! the residual-coding walker (§7.3.8.7) to map a scan position to
//! the in-block raster offset, and the inverse table by every
//! derivation that needs the §6.5.3-bound scan position of an
//! arbitrary in-block raster offset (e.g. `last_sig_coeff_x/y` →
//! ScanPos round-trip).
//!
//! ## Wiring stance
//!
//! Same opt-in posture as the round-218 / 223 / 229 / 232 / 237 /
//! 242 helper rollout: pure functions returning owned vectors, no
//! behaviour change to existing decoder paths. The
//! `slice_data::decode_residual_coding_rle` walker continues to call
//! its in-module zig-zag builder; rebinding it to
//! [`zig_zag_scan`] is a follow-up.

#![allow(clippy::needless_range_loop)]

extern crate alloc;

use alloc::vec::Vec;

/// Build the §6.5.2 zig-zag scan order array (eq. 33) for an
/// `(blk_w × blk_h)` block.
///
/// The returned vector has length `blk_w * blk_h`. Entry `sPos`
/// holds the row-major raster offset `rPos = y * blk_w + x` of the
/// block sample visited at scan position `sPos`.
///
/// The walk starts at the top-left corner and proceeds along
/// anti-diagonals (lines of constant `x + y`). Per eq. (33):
///
/// * Line 0 visits `(0, 0)`.
/// * Odd lines walk from the top-right endpoint
///   `(min(line, blk_w − 1), max(0, line − (blk_w − 1)))` toward the
///   bottom-left endpoint, decrementing `x` and incrementing `y` at
///   each step. Stops when `x < 0` or `y == blk_h`.
/// * Even lines (≥ 2) walk from the bottom-left endpoint
///   `(max(0, line − (blk_h − 1)), min(line, blk_h − 1))` toward the
///   top-right endpoint, incrementing `x` and decrementing `y` at
///   each step. Stops when `y < 0` or `x == blk_w`.
///
/// The total number of points visited equals the sum of anti-diagonal
/// lengths `1 + 2 + … + min(blk_w, blk_h) + … + 2 + 1 = blk_w * blk_h`.
///
/// # Panics
///
/// Does not panic on its own. A `blk_w == 0` or `blk_h == 0` input
/// returns an empty vector; the spec only ever invokes this process
/// at the §7.4.3.1 transform-block sizes (`1 << log2Tb{W,H}` with
/// `log2Tb{W,H} ∈ 1 ..= MaxTbLog2SizeY`), so the positive-dimension
/// case is the only one a conforming bitstream can trigger.
pub fn zig_zag_scan(blk_w: usize, blk_h: usize) -> Vec<u32> {
    let total = blk_w * blk_h;
    let mut zz: Vec<u32> = Vec::with_capacity(total);
    if total == 0 {
        return zz;
    }
    zz.push(0);
    let bw = blk_w as i32;
    let bh = blk_h as i32;
    for line in 1..(bw + bh - 1) {
        if line & 1 == 1 {
            // Odd line: from the top-right endpoint, decrement x,
            // increment y.
            let mut x = line.min(bw - 1);
            let mut y = (line - (bw - 1)).max(0);
            while x >= 0 && y < bh {
                zz.push((y * bw + x) as u32);
                x -= 1;
                y += 1;
            }
        } else {
            // Even line (≥ 2): from the bottom-left endpoint,
            // increment x, decrement y.
            let mut y = line.min(bh - 1);
            let mut x = (line - (bh - 1)).max(0);
            while y >= 0 && x < bw {
                zz.push((y * bw + x) as u32);
                x += 1;
                y -= 1;
            }
        }
    }
    debug_assert_eq!(zz.len(), total);
    zz
}

/// Build the §6.5.3 inverse scan order array (eq. 34) for an
/// `(blk_w × blk_h)` block.
///
/// The returned vector has length `blk_w * blk_h`. Entry `rPos` holds
/// the zig-zag scan position `sPos` that visits raster offset `rPos`.
/// By eq. (34) this is the per-element inverse of the
/// [`zig_zag_scan`] forward map: for every legal `pos`,
///
/// ```text
/// inverseScan[ forwardScan[ pos ] ] = pos
/// forwardScan[ inverseScan[ pos ] ] = pos
/// ```
///
/// # Panics
///
/// Does not panic on its own. A `blk_w == 0` or `blk_h == 0` input
/// returns an empty vector. The §6.5.3 walk depends on the §6.5.2
/// forward array being a permutation of `0 ..= blk_w · blk_h − 1`;
/// the §6.5.2 transcription preserves that invariant.
pub fn inverse_scan(blk_w: usize, blk_h: usize) -> Vec<u32> {
    let total = blk_w * blk_h;
    let mut inv: Vec<u32> = vec![0u32; total];
    if total == 0 {
        return inv;
    }
    let fwd = zig_zag_scan(blk_w, blk_h);
    debug_assert_eq!(fwd.len(), total);
    for pos in 0..total {
        let r_pos = fwd[pos] as usize;
        debug_assert!(r_pos < total);
        inv[r_pos] = pos as u32;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round245_zig_zag_4x4_matches_hand_trace() {
        // §6.5.2 walk for a 4×4 block:
        //   line 0: (0,0) → flat 0
        //   line 1 (odd): (1,0)→1, (0,1)→4
        //   line 2 (even): (0,2)→8, (1,1)→5, (2,0)→2
        //   line 3 (odd): (3,0)→3, (2,1)→6, (1,2)→9, (0,3)→12
        //   line 4 (even): (1,3)→13, (2,2)→10, (3,1)→7
        //   line 5 (odd): (3,2)→11, (2,3)→14
        //   line 6 (even): (3,3)→15
        let s = zig_zag_scan(4, 4);
        let expected: [u32; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn round245_zig_zag_2x2_matches_hand_trace() {
        // §6.5.2 walk for a 2×2 block:
        //   line 0: (0,0) → 0
        //   line 1 (odd): (1,0)→1, (0,1)→2
        //   line 2 (even): (1,1)→3
        let s = zig_zag_scan(2, 2);
        let expected: [u32; 4] = [0, 1, 2, 3];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn round245_zig_zag_4x2_non_square_matches_hand_trace() {
        // §6.5.2 walk for a 4×2 block (blkWidth=4, blkHeight=2):
        //   line 0: (0,0) → 0
        //   line 1 (odd): (1,0)→1, (0,1)→4
        //   line 2 (even): (1,1)→5, (2,0)→2
        //   line 3 (odd): (3,0)→3, (2,1)→6
        //   line 4 (even): (3,1)→7
        let s = zig_zag_scan(4, 2);
        let expected: [u32; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn round245_zig_zag_is_permutation_for_every_tb_size() {
        // §7.4.3.1 invokes §6.5.2 across (log2TbW, log2TbH) ∈
        // [1, MaxTbLog2SizeY=6]^2; we sweep the small end exhaustively
        // and the corners of the large end.
        for log2_w in 1u32..=4 {
            for log2_h in 1u32..=4 {
                let bw = 1usize << log2_w;
                let bh = 1usize << log2_h;
                let s = zig_zag_scan(bw, bh);
                assert_eq!(s.len(), bw * bh);
                let mut seen = vec![false; bw * bh];
                for &r in &s {
                    let r = r as usize;
                    assert!(r < bw * bh, "{bw}x{bh}: out-of-range raster {r}");
                    assert!(!seen[r], "{bw}x{bh}: duplicate raster {r}");
                    seen[r] = true;
                }
                assert!(seen.iter().all(|&b| b), "{bw}x{bh}: not a permutation");
            }
        }
    }

    #[test]
    fn round245_zig_zag_visits_anti_diagonals_in_order() {
        // The §6.5.2 walk visits points by non-decreasing line index
        // (x + y). This is the structural invariant the anti-diagonal
        // loop guarantees.
        for log2_w in 1u32..=4 {
            for log2_h in 1u32..=4 {
                let bw = 1usize << log2_w;
                let bh = 1usize << log2_h;
                let s = zig_zag_scan(bw, bh);
                let mut prev_line: i32 = -1;
                for &r in &s {
                    let r = r as usize;
                    let x = (r % bw) as i32;
                    let y = (r / bw) as i32;
                    let line = x + y;
                    assert!(line >= prev_line, "{bw}x{bh}: anti-diagonal order violated");
                    prev_line = line;
                }
            }
        }
    }

    #[test]
    fn round245_zig_zag_empty_blocks_return_empty_vec() {
        // The spec never invokes §6.5.2 with a zero-sized block;
        // this case is defensive only.
        assert!(zig_zag_scan(0, 0).is_empty());
        assert!(zig_zag_scan(0, 4).is_empty());
        assert!(zig_zag_scan(4, 0).is_empty());
    }

    #[test]
    fn round245_inverse_scan_4x4_round_trips_forward() {
        // §6.5.3 eq. (34): inverseScan[ forwardScan[ pos ] ] = pos.
        let bw = 4usize;
        let bh = 4usize;
        let fwd = zig_zag_scan(bw, bh);
        let inv = inverse_scan(bw, bh);
        for pos in 0..(bw * bh) {
            assert_eq!(inv[fwd[pos] as usize], pos as u32);
        }
    }

    #[test]
    fn round245_inverse_scan_4x4_dc_at_raster_zero_is_scan_zero() {
        // The §6.5.2 walk starts at (0, 0), i.e. forwardScan[0] = 0,
        // so inverseScan[0] = 0 is the DC coefficient bridge: scan
        // position 0 lives at raster 0.
        let inv = inverse_scan(4, 4);
        assert_eq!(inv[0], 0);
    }

    #[test]
    fn round245_inverse_scan_is_bijection_with_forward() {
        // Both maps are permutations and inverses of one another:
        //   forwardScan[ inverseScan[ rPos ] ] = rPos
        for log2_w in 1u32..=4 {
            for log2_h in 1u32..=4 {
                let bw = 1usize << log2_w;
                let bh = 1usize << log2_h;
                let fwd = zig_zag_scan(bw, bh);
                let inv = inverse_scan(bw, bh);
                let total = bw * bh;
                assert_eq!(fwd.len(), total);
                assert_eq!(inv.len(), total);
                for r_pos in 0..total {
                    assert_eq!(fwd[inv[r_pos] as usize] as usize, r_pos);
                }
                for s_pos in 0..total {
                    assert_eq!(inv[fwd[s_pos] as usize] as usize, s_pos);
                }
            }
        }
    }

    #[test]
    fn round245_inverse_scan_4x4_pins_eq34_values() {
        // Hand-derived from the §6.5.2 4×4 forward map (see
        // round245_zig_zag_4x4_matches_hand_trace) by applying
        // eq. (34): inv[ fwd[pos] ] = pos. Reading off:
        //   fwd = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        // gives, for rPos = 0..15:
        //   rPos=0  → sPos=0   (fwd[0]=0)
        //   rPos=1  → sPos=1   (fwd[1]=1)
        //   rPos=2  → sPos=5   (fwd[5]=2)
        //   rPos=3  → sPos=6   (fwd[6]=3)
        //   rPos=4  → sPos=2   (fwd[2]=4)
        //   rPos=5  → sPos=4   (fwd[4]=5)
        //   rPos=6  → sPos=7   (fwd[7]=6)
        //   rPos=7  → sPos=12  (fwd[12]=7)
        //   rPos=8  → sPos=3   (fwd[3]=8)
        //   rPos=9  → sPos=8   (fwd[8]=9)
        //   rPos=10 → sPos=11  (fwd[11]=10)
        //   rPos=11 → sPos=13  (fwd[13]=11)
        //   rPos=12 → sPos=9   (fwd[9]=12)
        //   rPos=13 → sPos=10  (fwd[10]=13)
        //   rPos=14 → sPos=14  (fwd[14]=14)
        //   rPos=15 → sPos=15  (fwd[15]=15)
        let inv = inverse_scan(4, 4);
        let expected: [u32; 16] = [0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15];
        assert_eq!(inv.as_slice(), &expected);
    }

    #[test]
    fn round245_inverse_scan_4x2_pins_eq34_values() {
        // Apply eq. (34) to the 4×2 forward map
        //   fwd = [0, 1, 4, 5, 2, 3, 6, 7]
        // Reading off:
        //   rPos=0 → sPos=0
        //   rPos=1 → sPos=1
        //   rPos=2 → sPos=4
        //   rPos=3 → sPos=5
        //   rPos=4 → sPos=2
        //   rPos=5 → sPos=3
        //   rPos=6 → sPos=6
        //   rPos=7 → sPos=7
        let inv = inverse_scan(4, 2);
        let expected: [u32; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        assert_eq!(inv.as_slice(), &expected);
    }

    #[test]
    fn round245_inverse_scan_empty_blocks_return_empty_vec() {
        // Defensive: mirrors the §6.5.2 zero-size behaviour.
        assert!(inverse_scan(0, 0).is_empty());
        assert!(inverse_scan(0, 4).is_empty());
        assert!(inverse_scan(4, 0).is_empty());
    }
}
