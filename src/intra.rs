//! EVC intra prediction (ISO/IEC 23094-1 §8.4.4).
//!
//! Round-3 scope: Baseline profile with `sps_eipd_flag = 0` and
//! `sps_suco_flag = 0`. Five intra modes are defined per Table 13:
//!
//! | mode | name        |
//! |------|-------------|
//! | 0    | INTRA_DC    |
//! | 1    | INTRA_HOR   |
//! | 2    | INTRA_VER   |
//! | 3    | INTRA_UL    |
//! | 4    | INTRA_UR    |
//!
//! All five are implemented per §8.4.4.3 / .4 / .5 / .6 / .7. Reference
//! sample availability and substitution follows §8.4.4.2 (sps_eipd_flag=0
//! path: missing samples are filled with `1 << (bitDepth − 1)`).
//!
//! The 33-direction angular set (modes 5..) belongs to the
//! `sps_eipd_flag = 1` path (Main profile, §8.4.4.10) and is **not**
//! included in round 3.

/// Baseline-profile intra prediction modes (§7.4.6.4 / Table 13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntraMode {
    /// `INTRA_DC` — average of top + left reference rows (§8.4.4.3).
    Dc,
    /// `INTRA_HOR` — horizontal copy from left neighbour (§8.4.4.4).
    Hor,
    /// `INTRA_VER` — vertical copy from top neighbour (§8.4.4.5).
    Ver,
    /// `INTRA_UL` — upper-left diagonal copy (§8.4.4.6).
    Ul,
    /// `INTRA_UR` — upper-right diagonal blend (§8.4.4.7).
    Ur,
}

impl IntraMode {
    /// Map the syntax-element value (`IntraPredModeY` ∈ 0..=4 for
    /// `sps_eipd_flag = 0`) to the enum.
    pub fn from_baseline_idx(idx: u32) -> Option<Self> {
        match idx {
            0 => Some(Self::Dc),
            1 => Some(Self::Hor),
            2 => Some(Self::Ver),
            3 => Some(Self::Ul),
            4 => Some(Self::Ur),
            _ => None,
        }
    }
}

/// Reference samples around a coding block for intra prediction.
///
/// Layout per §8.4.4.1 (sps_suco_flag=0 path):
///
/// * `top_left = p[-1][-1]`
/// * `top[i] = p[i][-1]` for `i ∈ 0..nCbW + nCbH`
/// * `left[j] = p[-1][j]` for `j ∈ 0..nCbW + nCbH`
///
/// This matches the (`nCbW * 2 + nCbH * 2 + 1`) sample neighbourhood
/// referenced by the prediction modes. Missing samples are pre-filled by
/// the caller using the sps_eipd_flag=0 substitution rule (§8.4.4.2):
/// any "not available" sample becomes `1 << (bitDepth − 1)`.
pub struct RefSamples {
    pub top_left: i32,
    pub top: Vec<i32>,
    pub left: Vec<i32>,
}

impl RefSamples {
    /// Build a fresh reference-sample neighbourhood of the right shape,
    /// pre-filled with the `1 << (bitDepth − 1)` "not available"
    /// substitute per §8.4.4.2 sps_eipd_flag=0 path. Use [`Self::set_top`]
    /// / [`Self::set_left`] / [`Self::set_top_left`] to overlay real
    /// available samples before invoking [`predict`].
    pub fn unavailable(n_cb_w: usize, n_cb_h: usize, bit_depth: u32) -> Self {
        let fill = 1i32 << (bit_depth - 1);
        let span = n_cb_w + n_cb_h;
        Self {
            top_left: fill,
            top: vec![fill; span],
            left: vec![fill; span],
        }
    }
}

/// Run the §8.4.4 intra prediction process for the given mode and write
/// `predSamples[x][y]` into `dst` (row-major, `n_cb_w` columns, stride
/// `n_cb_w`). Caller-supplied reference samples are consumed
/// element-wise.
///
/// Available reference span in sps_eipd_flag=0 mode is
/// `nCbW + nCbH` for both `top` and `left`.
pub fn predict(
    mode: IntraMode,
    refs: &RefSamples,
    n_cb_w: usize,
    n_cb_h: usize,
    bit_depth: u32,
    dst: &mut [i32],
) {
    debug_assert_eq!(dst.len(), n_cb_w * n_cb_h);
    debug_assert!(refs.top.len() >= n_cb_w + n_cb_h);
    debug_assert!(refs.left.len() >= n_cb_w + n_cb_h);
    let max_val = (1i32 << bit_depth) - 1;
    match mode {
        IntraMode::Dc => predict_dc(refs, n_cb_w, n_cb_h, dst, max_val),
        IntraMode::Hor => predict_hor(refs, n_cb_w, n_cb_h, dst),
        IntraMode::Ver => predict_ver(refs, n_cb_w, n_cb_h, dst),
        IntraMode::Ul => predict_ul(refs, n_cb_w, n_cb_h, dst),
        IntraMode::Ur => predict_ur(refs, n_cb_w, n_cb_h, dst),
    }
}

fn predict_dc(refs: &RefSamples, n_cb_w: usize, n_cb_h: usize, dst: &mut [i32], _max: i32) {
    // §8.4.4.3 sps_eipd_flag=0 (eq. 285):
    //   pred = (Σtop + Σleft + nCbW) >> (Log2(nCbW) + 1)
    // Constraint: nCbW must be a power of two (Baseline tree always is).
    let log2_w = (n_cb_w as u32).trailing_zeros();
    let mut sum: i32 = 0;
    for x in 0..n_cb_w {
        sum += refs.top[x];
    }
    for y in 0..n_cb_h {
        sum += refs.left[y];
    }
    let dc = (sum + n_cb_w as i32) >> (log2_w + 1);
    for v in dst.iter_mut() {
        *v = dc;
    }
}

fn predict_hor(refs: &RefSamples, n_cb_w: usize, n_cb_h: usize, dst: &mut [i32]) {
    // §8.4.4.4 sps_eipd_flag=0 (eq. 289): predSamples[x][y] = p[-1][y].
    for y in 0..n_cb_h {
        let v = refs.left[y];
        for x in 0..n_cb_w {
            dst[y * n_cb_w + x] = v;
        }
    }
}

fn predict_ver(refs: &RefSamples, n_cb_w: usize, n_cb_h: usize, dst: &mut [i32]) {
    // §8.4.4.5 (eq. 293): predSamples[x][y] = p[x][-1].
    for y in 0..n_cb_h {
        for x in 0..n_cb_w {
            dst[y * n_cb_w + x] = refs.top[x];
        }
    }
}

fn predict_ul(refs: &RefSamples, n_cb_w: usize, n_cb_h: usize, dst: &mut [i32]) {
    // §8.4.4.6 (eq. 294/295):
    //   y > x → p[-1][y - x - 1]
    //   else → p[x - y - 1][-1]
    // Both branches must read from the top_left when the offset would land
    // at -1 — which is what (y - x - 1) == -1 implies, i.e. y == x.
    // The spec says "y > x" so when y == x we hit the `else` branch with
    // x - y - 1 == -1 → top-left. We model `p[-1][-1] = top_left`.
    for y in 0..n_cb_h {
        for x in 0..n_cb_w {
            let v = if y > x {
                let off = y - x - 1;
                refs.left[off]
            } else if x == y {
                refs.top_left
            } else {
                let off = x - y - 1;
                refs.top[off]
            };
            dst[y * n_cb_w + x] = v;
        }
    }
}

fn predict_ur(refs: &RefSamples, n_cb_w: usize, n_cb_h: usize, dst: &mut [i32]) {
    // §8.4.4.7 (eq. 296):
    //   pred[x][y] = (predSamples[x+y+1][-1] + predSamples[-1][x+y+1]) >> 1
    // Read past the (nCbW + nCbH - 1) edge ⇒ clamp to the last available
    // position; the substitution process guarantees the buffer has at
    // least nCbW + nCbH entries for both top and left.
    for y in 0..n_cb_h {
        for x in 0..n_cb_w {
            let off = x + y + 1;
            let top = refs.top[off.min(refs.top.len() - 1)];
            let left = refs.left[off.min(refs.left.len() - 1)];
            dst[y * n_cb_w + x] = (top + left) >> 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With every reference sample at 128 (bit_depth=8 default fill), all
    /// five modes yield 128 for every sample.
    #[test]
    fn dc_with_default_refs_yields_128() {
        let refs = RefSamples::unavailable(8, 8, 8);
        let mut dst = vec![0i32; 64];
        predict(IntraMode::Dc, &refs, 8, 8, 8, &mut dst);
        assert!(
            dst.iter().all(|&v| v == 128),
            "DC must be 128 from default refs"
        );
    }

    #[test]
    fn hor_with_default_refs_yields_128() {
        let refs = RefSamples::unavailable(8, 8, 8);
        let mut dst = vec![0i32; 64];
        predict(IntraMode::Hor, &refs, 8, 8, 8, &mut dst);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn ver_with_default_refs_yields_128() {
        let refs = RefSamples::unavailable(8, 8, 8);
        let mut dst = vec![0i32; 64];
        predict(IntraMode::Ver, &refs, 8, 8, 8, &mut dst);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn ul_with_default_refs_yields_128() {
        let refs = RefSamples::unavailable(8, 8, 8);
        let mut dst = vec![0i32; 64];
        predict(IntraMode::Ul, &refs, 8, 8, 8, &mut dst);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn ur_with_default_refs_yields_128() {
        let refs = RefSamples::unavailable(8, 8, 8);
        let mut dst = vec![0i32; 64];
        predict(IntraMode::Ur, &refs, 8, 8, 8, &mut dst);
        assert!(dst.iter().all(|&v| v == 128));
    }

    /// Vertical mode with a custom top row must reproduce that row in
    /// every output row.
    #[test]
    fn ver_replicates_top_row() {
        let mut refs = RefSamples::unavailable(4, 4, 8);
        for (i, slot) in refs.top.iter_mut().enumerate().take(4) {
            *slot = (10 + i as i32) * 8;
        }
        let mut dst = vec![0i32; 16];
        predict(IntraMode::Ver, &refs, 4, 4, 8, &mut dst);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    dst[y * 4 + x],
                    refs.top[x],
                    "ver: row {y} col {x} must mirror top[{x}]"
                );
            }
        }
    }

    /// Horizontal mode with a custom left column must replicate that
    /// column across every row.
    #[test]
    fn hor_replicates_left_column() {
        let mut refs = RefSamples::unavailable(4, 4, 8);
        for (j, slot) in refs.left.iter_mut().enumerate().take(4) {
            *slot = (20 + j as i32) * 4;
        }
        let mut dst = vec![0i32; 16];
        predict(IntraMode::Hor, &refs, 4, 4, 8, &mut dst);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], refs.left[y]);
            }
        }
    }

    /// `from_baseline_idx` accepts 0..=4 and rejects everything else.
    #[test]
    fn baseline_mode_index_maps() {
        assert_eq!(IntraMode::from_baseline_idx(0), Some(IntraMode::Dc));
        assert_eq!(IntraMode::from_baseline_idx(1), Some(IntraMode::Hor));
        assert_eq!(IntraMode::from_baseline_idx(2), Some(IntraMode::Ver));
        assert_eq!(IntraMode::from_baseline_idx(3), Some(IntraMode::Ul));
        assert_eq!(IntraMode::from_baseline_idx(4), Some(IntraMode::Ur));
        assert_eq!(IntraMode::from_baseline_idx(5), None);
    }

    /// DC mode with a known top + left row sums to the expected value.
    #[test]
    fn dc_with_uniform_refs_returns_average() {
        // top = all 100, left = all 200, nCbW = nCbH = 4.
        // sum = 4*100 + 4*200 = 1200
        // dc = (1200 + 4) >> 3 = 1204 >> 3 = 150 (integer division)
        let mut refs = RefSamples::unavailable(4, 4, 8);
        for s in refs.top.iter_mut().take(4) {
            *s = 100;
        }
        for s in refs.left.iter_mut().take(4) {
            *s = 200;
        }
        let mut dst = vec![0i32; 16];
        predict(IntraMode::Dc, &refs, 4, 4, 8, &mut dst);
        assert_eq!(dst[0], 150);
        assert!(dst.iter().all(|&v| v == 150));
    }
}
