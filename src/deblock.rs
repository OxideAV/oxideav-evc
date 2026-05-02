//! EVC deblocking filter (ISO/IEC 23094-1 §8.8.2).
//!
//! Round-5 added the `sps_addb_flag = 0` baseline path for **luma**:
//! horizontal-direction filter (across vertical edges) followed by the
//! vertical-direction filter (across horizontal edges). Per §8.8.2.1
//! the deblocking is applied to all block edges of a picture except
//! the picture boundary, with a 4×4-grid step.
//!
//! Round-6 extends the same shape to **chroma** (Cb + Cr) per
//! eq. 1167-1213, sharing the luma side-info grid for BS derivation
//! and using the 2-tap stencil (only sB and sC modified) per
//! eq. 1208/1209/1212/1213.
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
//! shape but with a 2-sample stencil (only sB and sC are modified —
//! eq. 1203-1213). The BS derivation reuses the luma side-info grid
//! (cbf_luma is the trigger per spec step 2 for chroma too). The
//! Table 33 lookup uses `clippedQpC = Clip3(0, 51, QpY + cb_qp_offset)`
//! for cIdx=1, or with cr_qp_offset for cIdx=2 (eq. 1194). Round-6
//! wires the chroma path; pps_cb_qp_offset / pps_cr_qp_offset are
//! always 0 in the Baseline PPS so only slice_cb/cr_qp_offset matter.

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

/// Apply the 2-tap chroma edge filter to the central pair `[sB, sC]` of
/// the cross-section `[sA, sB, sC, sD]`. Implements §8.8.2.3 chroma
/// path eq. 1203-1213 — only sB and sC are modified.
fn filter_chroma_edge_2tap(s: &mut [i32; 4], s_t_prime: i32, max_val: i32) {
    if s_t_prime <= 0 {
        return;
    }
    let i_d = (s[0] - (s[1] << 2) + (s[2] << 2) - s[3]) / 8;
    let abs_d = i_d.abs();
    let i_tmp = (abs_d - s_t_prime).max(0) << 1;
    let clip_d = (abs_d - i_tmp).max(0);
    let i_d1 = if i_d.is_negative() { -clip_d } else { clip_d };
    let new_b = (s[1] + i_d1).clamp(0, max_val);
    let new_c = (s[2] - i_d1).clamp(0, max_val);
    s[1] = new_b;
    s[2] = new_c;
}

/// Map `chroma_format_idc` to the (SubWidthC, SubHeightC) factors
/// (Table 2). Returns (1, 1) for monochrome (caller short-circuits) and
/// for 4:4:4. Round-6 supports 4:2:0 chroma deblock end-to-end; 4:2:2
/// and 4:4:4 use the same stencil with adjusted edge spacing.
fn chroma_subsampling(chroma_format_idc: u32) -> (usize, usize) {
    match chroma_format_idc {
        1 => (2, 2), // 4:2:0
        2 => (2, 1), // 4:2:2
        3 => (1, 1), // 4:4:4
        _ => (1, 1),
    }
}

/// Run the §8.8.2 chroma deblocking pass over a single chroma plane
/// (`c_idx ∈ {1, 2}`). The BS derivation reads the luma 4×4 side-info
/// grid (cbf_luma is the spec's chroma-edge trigger per §8.8.2.3 step 2);
/// the Table 33 lookup uses
/// `qp_c = Clip3(0, 51, slice_qp + chroma_qp_offset)` (eq. 1194).
/// Returns the number of 4-sample chroma edges filtered.
///
/// `chroma_qp_offset` is `slice_cb_qp_offset` for `c_idx=1` and
/// `slice_cr_qp_offset` for `c_idx=2`. The Baseline PPS does not carry
/// `pps_cb_qp_offset`/`pps_cr_qp_offset` (they're always 0), so passing
/// only the slice-level offset is sufficient.
pub fn deblock_chroma(
    pic: &mut YuvPicture,
    side_info: &SideInfoGrid,
    slice_qp: i32,
    chroma_qp_offset: i32,
    c_idx: u32,
) -> Result<u32> {
    if pic.chroma_format_idc == 0 {
        return Ok(0);
    }
    if c_idx != 1 && c_idx != 2 {
        return Ok(0);
    }
    let (sub_w, sub_h) = chroma_subsampling(pic.chroma_format_idc);
    let plane_w = pic.width as usize / sub_w;
    let plane_h = pic.height as usize / sub_h;
    let stride = pic.c_stride();
    let max_val = (1i32 << pic.bit_depth) - 1;
    // qp_c = Clip3(0, 51, slice_qp + chroma_qp_offset). Round-6 has no
    // ChromaQpTable mapping (sps_chroma_qp_table_present_flag=0 in
    // Baseline fixtures), so this is the direct identity.
    let qp_c = (slice_qp + chroma_qp_offset).clamp(0, 51);
    let mut edges = 0u32;

    // Luma 4×4 grid stride is 4 luma pels; chroma stride is 4 / sub_w
    // chroma pels in x, 4 / sub_h chroma pels in y. The 4-sample chroma
    // run covers (4 * sub_w) luma pels in y for vertical-edge filtering
    // (i.e. 2 luma 4×4 rows for 4:2:0).

    // Pass 1: vertical edges in chroma (filter horizontally, x direction).
    // Edge positions in chroma: xC = 4/sub_w, 8/sub_w, ... Each maps to
    // luma x = xC * sub_w, which is a multiple of 4 (the luma cell
    // boundary). Skip the picture boundary at xC = 0.
    let chroma_step_x = 4 / sub_w; // 2 for 4:2:0
    let mut xc = chroma_step_x;
    while xc < plane_w {
        let x_luma = xc * sub_w;
        let mut yc = 0usize;
        while yc + 4 <= plane_h {
            // The chroma 4-sample run covers 4*sub_h luma rows. We sample
            // BS once per luma 4×4 cell crossed (so 1 cell for 4:2:2,
            // and 2 cells for 4:2:0 — but the spec applies a single bS
            // value per (xDi, yDj) chroma 4×4 block, so we use the
            // luma cell at the chroma run's start).
            let y_luma = yc * sub_h;
            let p = side_info.at(x_luma.saturating_sub(1) >> 2, y_luma >> 2);
            let q = side_info.at(x_luma >> 2, y_luma >> 2);
            let bs = derive_bs(p, q);
            let s_t_val = s_t(qp_c, bs);
            let s_t_prime = s_t_val << ((pic.bit_depth as i32) - 8);
            // Inner loop: 4 chroma samples down the edge (sample by
            // sample so we re-fetch BS for each luma cell crossed).
            for j in 0..4 {
                if xc < 2 || xc + 1 >= plane_w {
                    continue;
                }
                let yj_l = (yc + j) * sub_h;
                let p_j = side_info.at(x_luma.saturating_sub(1) >> 2, yj_l >> 2);
                let q_j = side_info.at(x_luma >> 2, yj_l >> 2);
                let bs_j = derive_bs(p_j, q_j);
                let st_j = s_t(qp_c, bs_j);
                let stp_j = st_j << ((pic.bit_depth as i32) - 8);
                let stp_use = if stp_j != 0 { stp_j } else { s_t_prime };
                if stp_use <= 0 {
                    continue;
                }
                let plane = if c_idx == 1 { &mut pic.cb } else { &mut pic.cr };
                let row = (yc + j) * stride;
                let mut s = [
                    plane[row + xc - 2] as i32,
                    plane[row + xc - 1] as i32,
                    plane[row + xc] as i32,
                    plane[row + xc + 1] as i32,
                ];
                filter_chroma_edge_2tap(&mut s, stp_use, max_val);
                plane[row + xc - 1] = s[1] as u8;
                plane[row + xc] = s[2] as u8;
            }
            edges += 1;
            yc += 4;
        }
        xc += chroma_step_x;
    }

    // Pass 2: horizontal edges in chroma (filter vertically).
    let chroma_step_y = 4 / sub_h; // 2 for 4:2:0
    let mut yc = chroma_step_y;
    while yc < plane_h {
        let y_luma = yc * sub_h;
        let mut xc = 0usize;
        while xc + 4 <= plane_w {
            let x_luma = xc * sub_w;
            let p = side_info.at(x_luma >> 2, y_luma.saturating_sub(1) >> 2);
            let q = side_info.at(x_luma >> 2, y_luma >> 2);
            let bs = derive_bs(p, q);
            let s_t_val = s_t(qp_c, bs);
            let s_t_prime = s_t_val << ((pic.bit_depth as i32) - 8);
            for i in 0..4 {
                if yc < 2 || yc + 1 >= plane_h {
                    continue;
                }
                let xi_l = (xc + i) * sub_w;
                let p_i = side_info.at(xi_l >> 2, y_luma.saturating_sub(1) >> 2);
                let q_i = side_info.at(xi_l >> 2, y_luma >> 2);
                let bs_i = derive_bs(p_i, q_i);
                let st_i = s_t(qp_c, bs_i);
                let stp_i = st_i << ((pic.bit_depth as i32) - 8);
                let stp_use = if stp_i != 0 { stp_i } else { s_t_prime };
                if stp_use <= 0 {
                    continue;
                }
                let plane = if c_idx == 1 { &mut pic.cb } else { &mut pic.cr };
                let col = xc + i;
                let mut s = [
                    plane[(yc - 2) * stride + col] as i32,
                    plane[(yc - 1) * stride + col] as i32,
                    plane[yc * stride + col] as i32,
                    plane[(yc + 1) * stride + col] as i32,
                ];
                filter_chroma_edge_2tap(&mut s, stp_use, max_val);
                plane[(yc - 1) * stride + col] = s[1] as u8;
                plane[yc * stride + col] = s[2] as u8;
            }
            edges += 1;
            xc += 4;
        }
        yc += chroma_step_y;
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

    /// Chroma 2-tap filter is a no-op when sT' = 0.
    #[test]
    fn chroma_filter_no_op_at_zero_strength() {
        let mut s = [10, 20, 30, 40];
        filter_chroma_edge_2tap(&mut s, 0, 255);
        assert_eq!(s, [10, 20, 30, 40]);
    }

    /// Chroma 2-tap filter only mutates sB and sC; sA and sD stay put.
    /// Hand-computed reference for [0, 100, 200, 255], sT' = 12:
    ///   iD   = (0 - 400 + 800 - 255) / 8 = 145 / 8 = 18
    ///   absD = 18; iTmp = max(0, (18-12)<<1) = 12; clipD = max(0, 18-12) = 6
    ///   iD1 (positive iD path) = +6
    ///   sB' = 100 + 6 = 106; sC' = 200 - 6 = 194.
    #[test]
    fn chroma_filter_modifies_only_inner_pair() {
        let mut s = [0, 100, 200, 255];
        filter_chroma_edge_2tap(&mut s, 12, 255);
        assert_eq!(s, [0, 106, 194, 255], "only sB and sC mutate");
    }

    /// 4:2:0 / 4:2:2 / 4:4:4 sub-sampling factors per Table 2.
    #[test]
    fn chroma_subsampling_table2() {
        assert_eq!(chroma_subsampling(1), (2, 2));
        assert_eq!(chroma_subsampling(2), (2, 1));
        assert_eq!(chroma_subsampling(3), (1, 1));
    }

    /// Run deblock_chroma on a uniform-grey 16×16 4:2:0 picture (chroma
    /// plane = 8×8). All cells default to Intra → bS=0 → no chroma
    /// filtering. Picture unchanged.
    #[test]
    fn deblock_chroma_no_op_on_uniform_grey() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let cb_orig = pic.cb.clone();
        let cr_orig = pic.cr.clone();
        let grid = SideInfoGrid::new(16, 16); // all Intra
                                              // 8×8 chroma plane, edges at xC=2,4,6 (3 verticals) × 2 row runs
                                              // (yC=0,4) = 6 vertical; same horizontal → 12 total.
        let n_cb = deblock_chroma(&mut pic, &grid, 32, 0, 1).unwrap();
        let n_cr = deblock_chroma(&mut pic, &grid, 32, 0, 2).unwrap();
        assert_eq!(n_cb, 12);
        assert_eq!(n_cr, 12);
        assert_eq!(pic.cb, cb_orig);
        assert_eq!(pic.cr, cr_orig);
    }

    /// Chroma deblock applies filtering when an inter CU with
    /// cbf_luma=1 sits next to an inter CU with cbf_luma=0 (bS=1 per
    /// step 2). With QP=27, sT'=2 for bS=1. The chroma 2-tap filter
    /// only kicks in when absD <= sT' (the edge is "small enough" to
    /// be a blocking artefact rather than a real picture edge), so we
    /// construct a small ramp across the edge.
    #[test]
    fn deblock_chroma_smooths_inter_cbf_edge() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        // Small ramp across xC=4: 100, 102, 104, 106 || 108, 110, 112, 114.
        // At the edge (sA=104, sB=106, sC=108, sD=110):
        //   iD = (104 - 424 + 432 - 110) / 8 = 2 / 8 = 0
        // absD=0 → clipD=0 → no change. Need a sharper local kink.
        // Use [100, 105, 110, 110] for far / sB / sC / far (rows j):
        //   sA=105, sB=110, sC=110, sD=105
        //   iD = (105 - 440 + 440 - 105) / 8 = 0 → no filter.
        // Pick values that produce small absD <= sT' = 2:
        // sA=100, sB=110, sC=120, sD=130:
        //   iD = (100 - 440 + 480 - 130) / 8 = 10 / 8 = 1, absD=1
        //   iTmp = max(0, (1-2)<<1) = 0, clipD = 1, iD1 = +1
        //   sB' = 111, sC' = 119.
        for j in 0..8 {
            pic.cb[j * 8 + 2] = 100;
            pic.cb[j * 8 + 3] = 110;
            pic.cb[j * 8 + 4] = 120;
            pic.cb[j * 8 + 5] = 130;
        }
        let cb_before = pic.cb.clone();
        let mut grid = SideInfoGrid::new(16, 16);
        // Stamp the left luma block (0..8, 0..16) as Inter cbf=1, the
        // right luma block (8..16, 0..16) as Inter cbf=0. The chroma
        // BS lookup at the chroma-x=4 edge (luma x=8) sees one side
        // with cbf=1 → bS=1 per spec step 2.
        grid.stamp_block(
            0,
            0,
            8,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                cbf_luma: 1,
                ref_idx_l0: 0,
                ..Default::default()
            },
        );
        grid.stamp_block(
            8,
            0,
            8,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                cbf_luma: 0,
                ref_idx_l0: 0,
                ..Default::default()
            },
        );
        let n = deblock_chroma(&mut pic, &grid, 27, 0, 1).unwrap();
        assert!(n > 0);
        // The step at xC=3..4 (sB and sC) must be smoothed by the spec
        // formula: sB' = 111, sC' = 119. Far-side samples (xC=2 and 5)
        // remain because chroma is 2-tap.
        for j in 0..8 {
            assert_eq!(pic.cb[j * 8 + 2], cb_before[j * 8 + 2], "sA unchanged");
            assert_eq!(pic.cb[j * 8 + 5], cb_before[j * 8 + 5], "sD unchanged");
            assert_eq!(pic.cb[j * 8 + 3], 111, "sB at row {j}");
            assert_eq!(pic.cb[j * 8 + 4], 119, "sC at row {j}");
        }
    }

    /// Chroma deblock honours `chroma_qp_offset`: a positive offset
    /// raises QP_C above the slice QP, which raises sT' and (for a
    /// small-step edge) turns filtering on where it would otherwise be
    /// off. Picking slice_qp=17 (sT'=0 for bS=1 per Table 33) shuts the
    /// filter; offset=+10 (QP_C=27 → sT'=2) re-enables it on the same
    /// ramp used by `deblock_chroma_smooths_inter_cbf_edge`.
    #[test]
    fn deblock_chroma_respects_qp_offset() {
        // Helper to seed a small-ramp Cb plane (sB=110, sC=120 across
        // the chroma edge at xC=3..4).
        let seed = |pic: &mut YuvPicture| {
            for j in 0..8 {
                pic.cb[j * 8 + 2] = 100;
                pic.cb[j * 8 + 3] = 110;
                pic.cb[j * 8 + 4] = 120;
                pic.cb[j * 8 + 5] = 130;
            }
        };
        let mut grid = SideInfoGrid::new(16, 16);
        grid.stamp_block(
            0,
            0,
            8,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                cbf_luma: 1,
                ref_idx_l0: 0,
                ..Default::default()
            },
        );
        grid.stamp_block(
            8,
            0,
            8,
            16,
            CuSideInfo {
                pred_mode: CuPredMode::Inter,
                cbf_luma: 0,
                ref_idx_l0: 0,
                ..Default::default()
            },
        );

        // Without offset: slice_qp=17 → sT'=0 → no filter.
        let mut pic_a = YuvPicture::new(16, 16, 1, 8).unwrap();
        seed(&mut pic_a);
        let pic_a_before = pic_a.cb.clone();
        deblock_chroma(&mut pic_a, &grid, 17, 0, 1).unwrap();
        assert_eq!(pic_a.cb, pic_a_before, "QP=17, offset=0 → no filter");

        // With offset: slice_qp=17 + offset=+10 → QP_C=27 → sT'=2 → filter.
        let mut pic_b = YuvPicture::new(16, 16, 1, 8).unwrap();
        seed(&mut pic_b);
        let pic_b_before = pic_b.cb.clone();
        deblock_chroma(&mut pic_b, &grid, 17, 10, 1).unwrap();
        assert_ne!(
            pic_b.cb, pic_b_before,
            "QP=17, offset=+10 → QP_C=27 → filter must run"
        );
        // Spot-check sB/sC modified at row 0.
        assert_eq!(pic_b.cb[3], 111);
        assert_eq!(pic_b.cb[4], 119);
    }
}
