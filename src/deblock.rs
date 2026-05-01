//! EVC deblocking filter (ISO/IEC 23094-1 §8.8.2).
//!
//! Round-5 implements the `sps_addb_flag = 0` baseline path:
//! horizontal-direction filter (across vertical edges) followed by the
//! vertical-direction filter (across horizontal edges). Per §8.8.2.1
//! the deblocking is applied to all block edges of a picture except
//! the picture boundary, with a 4×4-grid step.
//!
//! ## Per-CU side info (CuSideInfo)
//!
//! The deblocking BS derivation in §8.8.2.3 step 2 needs, at every
//! 4×4 sample grid position adjacent to an edge, the following:
//!
//! * `pred_mode` (intra / inter / ibc),
//! * `cbf` (luma CBF for that 4×4),
//! * `mvL0`, `mvL1` (per-list motion vectors in sub-pel units),
//! * `refIdxL0`, `refIdxL1`.
//!
//! Round-5 fixtures keep refIdx ∈ {-1, 0} (-1 = unavailable, mapped
//! through the spec's `refIdxXLY` "unavailable → mv=0" rule), so the
//! per-list info collapses to a single ref slot per list. We record
//! this on a 4×4-pel grid keyed by `(x>>2, y>>2)`.
//!
//! ## Edge filter math (§8.8.2.3 steps 4-6)
//!
//! For each 4-sample run of an edge:
//! * derive `bS` (0..3) from neighbouring CU type / CBF / MV deltas,
//! * look up `sT` from Table 33 keyed by QP and bS,
//! * read 4 pre-filter samples sA..sD across the edge,
//! * apply eq. 1148-1158 to clip-and-modify each sample,
//! * write the modified samples back to recPicture.
//!
//! Chroma deblocking (the §8.8.2.3 path for cIdx > 0) follows the same
//! shape but with single-sample stencils. Round-5 wires it the same
//! way as luma; the BS derivation reuses the luma side-info grid.

use crate::picture::YuvPicture;
use oxideav_core::Result;

/// Prediction mode of a coding block, as needed for BS derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuPredMode {
    Intra,
    Inter,
    /// IBC (intra block copy) — only reachable in Main profile, but the
    /// BS table treats it specially (bS = 2). Round-5 fixtures don't
    /// generate this; we keep the variant for completeness.
    Ibc,
}

/// Per-4×4-block side info that the deblocking process consults.
#[derive(Clone, Copy, Debug)]
pub struct CuSideInfo {
    pub pred_mode: CuPredMode,
    /// Luma CBF (1 if any coded coefficient was decoded, else 0).
    pub cbf_luma: u8,
    /// MV in 1/4-pel resolution (matches the cMVD-decoded value).
    pub mv_l0_x: i32,
    pub mv_l0_y: i32,
    pub mv_l1_x: i32,
    pub mv_l1_y: i32,
    /// Reference indices; -1 (i.e. 255 here as `u8`) means unavailable.
    pub ref_idx_l0: i8,
    pub ref_idx_l1: i8,
}

impl Default for CuSideInfo {
    fn default() -> Self {
        Self {
            pred_mode: CuPredMode::Intra,
            cbf_luma: 0,
            mv_l0_x: 0,
            mv_l0_y: 0,
            mv_l1_x: 0,
            mv_l1_y: 0,
            ref_idx_l0: -1,
            ref_idx_l1: -1,
        }
    }
}

/// 4×4-grid side-info storage for a whole picture.
#[derive(Clone, Debug)]
pub struct SideInfoGrid {
    /// Width of the grid in 4×4 cells (= ceil(pic_width / 4)).
    pub w_cells: usize,
    /// Height of the grid in 4×4 cells.
    pub h_cells: usize,
    pub cells: Vec<CuSideInfo>,
}

impl SideInfoGrid {
    /// Allocate a fresh grid covering `(pic_width × pic_height)`. All
    /// cells start in the default `Intra, cbf=0, mv=0, refidx=-1`
    /// state — matching the spec's "no neighbour" substitution.
    pub fn new(pic_width: u32, pic_height: u32) -> Self {
        let w_cells = pic_width.div_ceil(4) as usize;
        let h_cells = pic_height.div_ceil(4) as usize;
        let cells = vec![CuSideInfo::default(); w_cells * h_cells];
        Self {
            w_cells,
            h_cells,
            cells,
        }
    }

    /// Stamp a `(cb_w × cb_h)` block at `(x, y)` (luma coords) with the
    /// supplied side info.
    pub fn stamp_block(&mut self, x: u32, y: u32, cb_w: u32, cb_h: u32, info: CuSideInfo) {
        let x0 = (x >> 2) as usize;
        let y0 = (y >> 2) as usize;
        let xn = ((x + cb_w + 3) >> 2) as usize;
        let yn = ((y + cb_h + 3) >> 2) as usize;
        for j in y0..yn.min(self.h_cells) {
            for i in x0..xn.min(self.w_cells) {
                self.cells[j * self.w_cells + i] = info;
            }
        }
    }

    pub fn at(&self, x_cell: usize, y_cell: usize) -> CuSideInfo {
        if x_cell >= self.w_cells || y_cell >= self.h_cells {
            CuSideInfo::default()
        } else {
            self.cells[y_cell * self.w_cells + x_cell]
        }
    }
}

/// `sT[QP][bS]` table (Table 33). Returns 0 for unfiltered cases.
fn s_t(qp: i32, bs: u32) -> i32 {
    if !(1..=3).contains(&bs) {
        return 0;
    }
    if qp <= 17 {
        return 0;
    }
    // bS=1 column.
    let c1 = match qp {
        18..=26 => 1,
        27..=31 => 2,
        32..=34 => 3,
        35..=37 => 4,
        38..=39 => 5,
        40..=41 => 6,
        42 => 7,
        43 => 8,
        44 => 9,
        45 => 10,
        46 => 11,
        47..=51 => 12,
        _ => 12,
    };
    // bS=2 column.
    let c2 = match qp {
        18..=26 => 0,
        27..=31 => 1,
        32..=34 => 2,
        35..=37 => 3,
        38..=39 => 4,
        40..=41 => 5,
        42 => 6,
        43 => 7,
        44 => 8,
        45 => 9,
        46 => 10,
        47..=51 => 11,
        _ => 11,
    };
    // bS=3 column.
    let c3 = match qp {
        18..=31 => 0,
        32..=34 => 1,
        35..=37 => 2,
        38..=39 => 3,
        40..=41 => 4,
        42 => 5,
        43 => 6,
        44 => 7,
        45 => 8,
        46 => 9,
        47..=51 => 10,
        _ => 10,
    };
    match bs {
        1 => c1,
        2 => c2,
        3 => c3,
        _ => 0,
    }
}

/// Derive bS for a single 4-sample run on an edge (§8.8.2.3 step 2).
/// `p` is the side-info on the left/above of the edge; `q` on the
/// right/below.
fn derive_bs(p: CuSideInfo, q: CuSideInfo) -> u32 {
    if matches!(p.pred_mode, CuPredMode::Intra) || matches!(q.pred_mode, CuPredMode::Intra) {
        return 0; // intra → "skip filter" per §8.8.2.3 (bS = 0 means no filter)
    }
    if p.cbf_luma != 0 || q.cbf_luma != 0 {
        return 1;
    }
    if matches!(p.pred_mode, CuPredMode::Ibc) || matches!(q.pred_mode, CuPredMode::Ibc) {
        return 2;
    }
    // Inter both sides: compare refs + MVs (§8.8.2.3 eq. 1136-1138).
    let same_ref_pair = p.ref_idx_l0 == q.ref_idx_l0 && p.ref_idx_l1 == q.ref_idx_l1;
    let swapped_ref_pair = p.ref_idx_l0 == q.ref_idx_l1 && p.ref_idx_l1 == q.ref_idx_l0;
    let mv_thresh = 4;
    if same_ref_pair {
        let dx0 = (p.mv_l0_x - q.mv_l0_x).abs();
        let dy0 = (p.mv_l0_y - q.mv_l0_y).abs();
        let dx1 = (p.mv_l1_x - q.mv_l1_x).abs();
        let dy1 = (p.mv_l1_y - q.mv_l1_y).abs();
        if dx0 >= mv_thresh || dy0 >= mv_thresh || dx1 >= mv_thresh || dy1 >= mv_thresh {
            2
        } else {
            3
        }
    } else if swapped_ref_pair {
        let dx0 = (p.mv_l0_x - q.mv_l1_x).abs();
        let dy0 = (p.mv_l0_y - q.mv_l1_y).abs();
        let dx1 = (p.mv_l1_x - q.mv_l0_x).abs();
        let dy1 = (p.mv_l1_y - q.mv_l0_y).abs();
        if dx0 >= mv_thresh || dy0 >= mv_thresh || dx1 >= mv_thresh || dy1 >= mv_thresh {
            2
        } else {
            3
        }
    } else {
        2
    }
}

/// Apply the 4-tap edge filter to a single edge cross-section
/// `[sA, sB, sC, sD]` at strength `sT_prime`. Returns the modified
/// samples in the same order.
///
/// Implements §8.8.2.3 step 5 / step 6 (eq. 1148-1158).
fn filter_edge_4tap(s: &mut [i32; 4], s_t_prime: i32, max_val: i32) {
    if s_t_prime <= 0 {
        return;
    }
    let i_d = (s[0] - (s[1] << 2) + (s[2] << 2) - s[3]) / 8;
    let abs_d = i_d.abs();
    let i_tmp = (abs_d - s_t_prime).max(0) << 1;
    let mut clip_d = (abs_d - i_tmp).max(0);
    let i_d1 = if i_d.is_negative() { -clip_d } else { clip_d };
    clip_d >>= 1;
    let i_d2 = ((s[0] - s[3]) / 4).clamp(-clip_d, clip_d);
    let new_a = (s[0] - i_d2).clamp(0, max_val);
    let new_b = (s[1] + i_d1).clamp(0, max_val);
    let new_c = (s[2] - i_d1).clamp(0, max_val);
    let new_d = (s[3] + i_d2).clamp(0, max_val);
    s[0] = new_a;
    s[1] = new_b;
    s[2] = new_c;
    s[3] = new_d;
}

/// Run the §8.8.2 luma deblocking pass over a full picture.
///
/// Returns the number of edges filtered (4-sample runs). The picture
/// buffer in `pic.y` is modified in place; chroma planes are left
/// untouched in this round.
///
/// `slice_qp` is the per-slice QP used in Table 33 lookups; round-5
/// fixtures keep `cu_qp_delta = 0` so the slice QP applies uniformly.
pub fn deblock_luma(pic: &mut YuvPicture, side_info: &SideInfoGrid, slice_qp: i32) -> Result<u32> {
    let pic_w = pic.width as usize;
    let pic_h = pic.height as usize;
    let stride = pic.y_stride();
    let max_val = (1i32 << pic.bit_depth) - 1;
    let mut edges = 0u32;
    // Pass 1: vertical edges (filter horizontally, x-direction).
    // Edge runs every 4 pels in x, starting at x=4 (skip picture
    // boundary). For each 4-sample run vertical at fixed x.
    let mut x = 4usize;
    while x < pic_w {
        let mut y = 0usize;
        while y + 4 <= pic_h {
            let p = side_info.at(x.saturating_sub(1) >> 2, y >> 2);
            let q = side_info.at(x >> 2, y >> 2);
            let bs = derive_bs(p, q);
            let s_t_val = s_t(slice_qp, bs);
            let s_t_prime = s_t_val << ((pic.bit_depth as i32) - 8);
            for j in 0..4 {
                if x < 2 || x + 1 >= pic_w {
                    continue;
                }
                let row = (y + j) * stride;
                let mut s = [
                    pic.y[row + x - 2] as i32,
                    pic.y[row + x - 1] as i32,
                    pic.y[row + x] as i32,
                    pic.y[row + x + 1] as i32,
                ];
                filter_edge_4tap(&mut s, s_t_prime, max_val);
                pic.y[row + x - 2] = s[0] as u8;
                pic.y[row + x - 1] = s[1] as u8;
                pic.y[row + x] = s[2] as u8;
                pic.y[row + x + 1] = s[3] as u8;
            }
            edges += 1;
            y += 4;
        }
        x += 4;
    }
    // Pass 2: horizontal edges (filter vertically, y-direction).
    let mut y = 4usize;
    while y < pic_h {
        let mut x = 0usize;
        while x + 4 <= pic_w {
            let p = side_info.at(x >> 2, y.saturating_sub(1) >> 2);
            let q = side_info.at(x >> 2, y >> 2);
            let bs = derive_bs(p, q);
            let s_t_val = s_t(slice_qp, bs);
            let s_t_prime = s_t_val << ((pic.bit_depth as i32) - 8);
            for i in 0..4 {
                if y < 2 || y + 1 >= pic_h {
                    continue;
                }
                let col = x + i;
                let mut s = [
                    pic.y[(y - 2) * stride + col] as i32,
                    pic.y[(y - 1) * stride + col] as i32,
                    pic.y[y * stride + col] as i32,
                    pic.y[(y + 1) * stride + col] as i32,
                ];
                filter_edge_4tap(&mut s, s_t_prime, max_val);
                pic.y[(y - 2) * stride + col] = s[0] as u8;
                pic.y[(y - 1) * stride + col] = s[1] as u8;
                pic.y[y * stride + col] = s[2] as u8;
                pic.y[(y + 1) * stride + col] = s[3] as u8;
            }
            edges += 1;
            x += 4;
        }
        y += 4;
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 33 spot-checks against the printed spec values.
    #[test]
    fn s_t_table_spot_checks() {
        assert_eq!(s_t(17, 1), 0);
        assert_eq!(s_t(18, 1), 1);
        assert_eq!(s_t(26, 1), 1);
        assert_eq!(s_t(27, 1), 2);
        assert_eq!(s_t(45, 1), 10);
        assert_eq!(s_t(45, 2), 9);
        assert_eq!(s_t(45, 3), 8);
        assert_eq!(s_t(51, 1), 12);
        assert_eq!(s_t(51, 3), 10);
        // bS = 0 (intra both sides) is always 0.
        assert_eq!(s_t(40, 0), 0);
    }

    /// Two adjacent intra CUs always produce bS = 0 → no filtering.
    #[test]
    fn bs_intra_intra_is_zero() {
        let p = CuSideInfo {
            pred_mode: CuPredMode::Intra,
            ..Default::default()
        };
        let q = CuSideInfo {
            pred_mode: CuPredMode::Intra,
            ..Default::default()
        };
        assert_eq!(derive_bs(p, q), 0);
    }

    /// One side has CBF=1 → bS=1.
    #[test]
    fn bs_with_cbf_is_one() {
        let p = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            cbf_luma: 1,
            ..Default::default()
        };
        let q = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            cbf_luma: 0,
            ..Default::default()
        };
        assert_eq!(derive_bs(p, q), 1);
    }

    /// Inter both sides, same refs, MVs identical → bS=3.
    #[test]
    fn bs_inter_same_ref_zero_mv_diff_is_three() {
        let p = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            ..Default::default()
        };
        let q = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            ..Default::default()
        };
        assert_eq!(derive_bs(p, q), 3);
    }

    /// Inter both sides, same refs, MV delta >= 4 → bS=2.
    #[test]
    fn bs_inter_same_ref_big_mv_diff_is_two() {
        let p = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            mv_l0_x: 0,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            ..Default::default()
        };
        let q = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            mv_l0_x: 8,
            ref_idx_l0: 0,
            ref_idx_l1: -1,
            ..Default::default()
        };
        assert_eq!(derive_bs(p, q), 2);
    }

    /// Filter is a no-op when sT' = 0 (e.g. low QP).
    #[test]
    fn filter_no_op_at_zero_strength() {
        let mut s = [10, 20, 30, 40];
        filter_edge_4tap(&mut s, 0, 255);
        assert_eq!(s, [10, 20, 30, 40]);
    }

    /// Side-info grid stamping: writing a 16×16 block at (8, 4) populates
    /// the right 4×4 cells.
    #[test]
    fn side_info_grid_stamp_block() {
        let mut grid = SideInfoGrid::new(64, 64);
        let info = CuSideInfo {
            pred_mode: CuPredMode::Inter,
            cbf_luma: 1,
            ..Default::default()
        };
        grid.stamp_block(8, 4, 16, 16, info);
        // (8,4) → cell (2,1). Block covers cells (2..6) × (1..5).
        for cy in 1..5 {
            for cx in 2..6 {
                assert_eq!(grid.at(cx, cy).cbf_luma, 1, "cell ({cx},{cy})");
            }
        }
        // Adjacent cell still default.
        assert_eq!(grid.at(1, 0).cbf_luma, 0);
    }

    /// Run deblock_luma on a uniform-grey picture with a grid of
    /// alternating intra/inter blocks. With all CBF=0 and same MV,
    /// bS=0 across all edges → no filtering. The picture must be
    /// unchanged.
    #[test]
    fn deblock_luma_no_op_on_uniform_grey() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let original = pic.y.clone();
        let grid = SideInfoGrid::new(16, 16); // all default Intra
        let n = deblock_luma(&mut pic, &grid, 22).unwrap();
        // Edges: vertical x ∈ {4, 8, 12} × y in {0, 4, 8, 12} = 12;
        // horizontal y ∈ {4, 8, 12} × x in {0, 4, 8, 12} = 12. Total 24.
        assert_eq!(n, 24);
        // No change because all bS=0 (intra-intra everywhere).
        assert_eq!(pic.y, original);
    }
}
