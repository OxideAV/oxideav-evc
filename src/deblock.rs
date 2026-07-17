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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// §8.5.1 eqs. 396/397 — the DMVR refinement delta
    /// `refMvLX − ( mvLX << 2 )` in 1/16-pel units (`+dMvL0` on list 0,
    /// `−dMvL0` on list 1); 0 for every non-DMVR cell, which makes the
    /// stored refined MV collapse to eq. 400 (`refMvLX = mvLX << 2`).
    /// Per the §8.5.1 NOTE the collocated (§8.5.2.3.4) readers
    /// reconstruct `refMvLX` from `mv + delta`, while the spatial-MVP
    /// and deblocking readers keep consuming the unrefined `mv_lX`.
    pub ref_mv_delta_l0_x: i32,
    pub ref_mv_delta_l0_y: i32,
    pub ref_mv_delta_l1_x: i32,
    pub ref_mv_delta_l1_y: i32,
    /// `MotionModelIdc[ x ][ y ]` (§7.4.9.4): 0 = translational (or not
    /// inter), 1 = 4-parameter affine, 2 = 6-parameter affine. Non-zero
    /// only on cells stamped by an affine CU; gates the §8.5.3.2/.5
    /// inherited-candidate availability.
    pub motion_model_idc: u8,
    /// `CbPosX/CbPosY[ x ][ y ]` — the covering coding block's top-left
    /// luma position, and `CbWidth/CbHeight` as log2. Written by the
    /// affine stamp so the §8.5.3.3 neighbour projection can locate the
    /// neighbour CU's corner cells (and the §8.5.3.2 step-4 pruning can
    /// compare covering CUs). Zero (unused) on non-affine cells.
    pub cu_x0: u16,
    pub cu_y0: u16,
    pub cu_log2_w: u8,
    pub cu_log2_h: u8,
    /// `cu_skip_flag[ x ][ y ]` — 1 on cells covered by a skip-coded CU.
    /// Feeds the §9.3.4.2.4 Table 96 `cu_skip_flag` neighbour ctxInc
    /// under `sps_cm_init_flag == 1`.
    pub cu_skip: u8,
    /// `IntraPredModeY[ x ][ y ]` — the covering intra CU's luma mode
    /// (Table 15 numbering under `sps_eipd_flag == 1`, the Baseline
    /// 5-mode index otherwise). Meaningful only on `Intra` cells; feeds
    /// the §8.4.2 neighbour candidates and the §8.4.3 co-located luma
    /// mode a chroma CU inherits.
    pub intra_luma_mode: u8,
    /// The covering CU's §8.7.1 `QpY` (the eq. 1042 chain value the
    /// residual was scaled with). Feeds the §8.8.3.5 per-edge `qPav`.
    pub qp_y: u8,
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
            ref_mv_delta_l0_x: 0,
            ref_mv_delta_l0_y: 0,
            ref_mv_delta_l1_x: 0,
            ref_mv_delta_l1_y: 0,
            motion_model_idc: 0,
            cu_x0: 0,
            cu_y0: 0,
            cu_log2_w: 0,
            cu_log2_h: 0,
            cu_skip: 0,
            intra_luma_mode: 0,
            qp_y: 0,
        }
    }
}

/// 4×4-grid side-info storage for a whole picture.
///
/// Besides driving the deblocking-filter BS derivation, the grid doubles
/// as the picture's per-4×4 motion field: the §8.5.2.4 grid AMVP, the
/// §8.5.2.3.2 spatial merge neighbours and — once the picture is retained
/// in the DPB — the §8.5.2.3.3 collocated (`ColPic`) motion store all
/// read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SideInfoGrid {
    /// Width of the grid in 4×4 cells (= ceil(pic_width / 4)).
    pub w_cells: usize,
    /// Height of the grid in 4×4 cells.
    pub h_cells: usize,
    pub cells: Vec<CuSideInfo>,
    /// Luma rectangle of the tile currently being decoded (§6.4.1 /
    /// §6.4.3 first bullet: a neighbouring block in a different tile is
    /// unavailable). Set by the tiled slice walkers per tile segment and
    /// cleared once the walk finishes; `None` (the single-tile default)
    /// disables the tile test — behaviour identical to the historical
    /// walkers.
    pub cur_tile: Option<crate::tiles::TileRect>,
    /// Interior tile boundaries exempt from deblocking (§8.8.2 /
    /// §8.8.3: "the edges that coincide with tile boundaries when
    /// loop_filter_across_tiles_enabled_flag is equal to 0"). `None`
    /// (the default) exempts nothing — single tile, or across-tiles
    /// filtering enabled.
    pub tile_bounds: Option<crate::tiles::TileBounds>,
}

impl Default for SideInfoGrid {
    /// A zero-cell grid — the "no motion field available" placeholder
    /// (every lookup returns the default all-invalid cell).
    fn default() -> Self {
        Self {
            w_cells: 0,
            h_cells: 0,
            cells: Vec::new(),
            cur_tile: None,
            tile_bounds: None,
        }
    }
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
            cur_tile: None,
            tile_bounds: None,
        }
    }

    /// §6.4.1 / §6.4.3 first bullet — whether the neighbouring luma
    /// location `(x, y)` lies in a different tile than the block
    /// currently being decoded. `false` whenever no current-tile
    /// rectangle is armed (single-tile decode).
    #[inline]
    pub fn neighbour_in_other_tile(&self, x: i64, y: i64) -> bool {
        match &self.cur_tile {
            Some(r) => !r.contains(x, y),
            None => false,
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

    /// Overwrite only the §8.5.1 DMVR refined-MV deltas (eqs. 396/397)
    /// of every 4×4 cell the block covers — the whole-CU stamp has
    /// already written the unrefined motion, and the deltas retrofit
    /// the per-subblock `refMvLX` for the collocated readers.
    pub fn stamp_ref_mv_delta(
        &mut self,
        x: u32,
        y: u32,
        cb_w: u32,
        cb_h: u32,
        d_l0: (i32, i32),
        d_l1: (i32, i32),
    ) {
        let x0 = (x >> 2) as usize;
        let y0 = (y >> 2) as usize;
        let xn = ((x + cb_w + 3) >> 2) as usize;
        let yn = ((y + cb_h + 3) >> 2) as usize;
        for j in y0..yn.min(self.h_cells) {
            for i in x0..xn.min(self.w_cells) {
                let cell = &mut self.cells[j * self.w_cells + i];
                cell.ref_mv_delta_l0_x = d_l0.0;
                cell.ref_mv_delta_l0_y = d_l0.1;
                cell.ref_mv_delta_l1_x = d_l1.0;
                cell.ref_mv_delta_l1_y = d_l1.1;
            }
        }
    }

    /// Mutable access to one 4×4 cell (bounds must hold). Used by the
    /// walkers for post-stamp annotations (e.g. the `cu_skip` mark).
    pub fn at_mut(&mut self, x_cell: usize, y_cell: usize) -> &mut CuSideInfo {
        let idx = y_cell * self.w_cells + x_cell;
        &mut self.cells[idx]
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
        // §8.8.2.1: edges that coincide with tile boundaries are exempt
        // when loop_filter_across_tiles_enabled_flag == 0.
        if let Some(tb) = &side_info.tile_bounds {
            if tb.is_col_boundary(x as u32) {
                x += 4;
                continue;
            }
        }
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
                pic.y[row + x - 2] = s[0] as u16;
                pic.y[row + x - 1] = s[1] as u16;
                pic.y[row + x] = s[2] as u16;
                pic.y[row + x + 1] = s[3] as u16;
            }
            edges += 1;
            y += 4;
        }
        x += 4;
    }
    // Pass 2: horizontal edges (filter vertically, y-direction).
    let mut y = 4usize;
    while y < pic_h {
        // §8.8.2.1 tile-boundary edge exemption (horizontal edges).
        if let Some(tb) = &side_info.tile_bounds {
            if tb.is_row_boundary(y as u32) {
                y += 4;
                continue;
            }
        }
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
                pic.y[(y - 2) * stride + col] = s[0] as u16;
                pic.y[(y - 1) * stride + col] = s[1] as u16;
                pic.y[y * stride + col] = s[2] as u16;
                pic.y[(y + 1) * stride + col] = s[3] as u16;
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
        // §8.8.2.1: tile-boundary edges (compared in the luma domain —
        // tile boundaries are CTB-aligned) are exempt when
        // loop_filter_across_tiles_enabled_flag == 0.
        if let Some(tb) = &side_info.tile_bounds {
            if tb.is_col_boundary(x_luma as u32) {
                xc += chroma_step_x;
                continue;
            }
        }
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
                plane[row + xc - 1] = s[1] as u16;
                plane[row + xc] = s[2] as u16;
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
        // §8.8.2.1 tile-boundary edge exemption (horizontal edges).
        if let Some(tb) = &side_info.tile_bounds {
            if tb.is_row_boundary(y_luma as u32) {
                yc += chroma_step_y;
                continue;
            }
        }
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
                plane[(yc - 1) * stride + col] = s[1] as u16;
                plane[yc * stride + col] = s[2] as u16;
            }
            edges += 1;
            xc += 4;
        }
        yc += chroma_step_y;
    }
    Ok(edges)
}

// =====================================================================
// §8.8.3 — Advanced deblocking filter (`sps_addb_flag == 1`).
// =====================================================================

/// Table 34 — α′ per indexA (0..51).
pub const ADDB_ALPHA_PRIME: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

/// Table 34 — β′ per indexB (0..51).
pub const ADDB_BETA_PRIME: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Table 35 — t′C0 per (bS − 1) and indexA.
pub const ADDB_T_C0: [[i32; 52]; 3] = [
    // bS = 1
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
    ],
    // bS = 2
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
    ],
    // bS = 3
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
    ],
];

/// The slice-level §8.8.3.5 threshold offsets (eqs. 86/87:
/// `FilterOffsetA = slice_alpha_offset`, `FilterOffsetB =
/// slice_beta_offset`, both −12..=12, inferred 0 when absent).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddbOffsets {
    pub filter_offset_a: i32,
    pub filter_offset_b: i32,
}

/// §8.8.3.4 — boundary filtering strength between the CUs covering the
/// luma positions P and Q (probed from the side-info grid).
fn addb_bs(grid: &SideInfoGrid, xp: u32, yp: u32, xq: u32, yq: u32, ctb_log2_size_y: u32) -> u32 {
    let p = grid.at((xp >> 2) as usize, (yp >> 2) as usize);
    let q = grid.at((xq >> 2) as usize, (yq >> 2) as usize);
    let p_intra = p.pred_mode == CuPredMode::Intra;
    let q_intra = q.pred_mode == CuPredMode::Intra;
    // Intra across a CTU boundary → 4.
    let different_ctu = (xp >> ctb_log2_size_y) != (xq >> ctb_log2_size_y)
        || (yp >> ctb_log2_size_y) != (yq >> ctb_log2_size_y);
    if (p_intra || q_intra) && different_ctu {
        return 4;
    }
    // Intra or IBC on either side → 3.
    if p_intra || q_intra || p.pred_mode == CuPredMode::Ibc || q.pred_mode == CuPredMode::Ibc {
        return 3;
    }
    // Coded residual on either side → 2. (The `ats_cu_inter_flag == 1`
    // arm shares this strength; ATS-inter is not yet decoded by the
    // walkers so no cell carries it.)
    if p.cbf_luma != 0 || q.cbf_luma != 0 {
        return 2;
    }
    // eqs. 1218-1232: reference/motion comparison at 1/4-pel (>= 4 ==
    // one integer sample).
    let (r0l0, r0l1) = (q.ref_idx_l0, q.ref_idx_l1);
    let (r1l0, r1l1) = (p.ref_idx_l0, p.ref_idx_l1);
    let mv = |r: i8, x: i32| if r != -1 { x } else { 0 };
    let (m0l0x, m0l0y) = (mv(r0l0, q.mv_l0_x), mv(r0l0, q.mv_l0_y));
    let (m0l1x, m0l1y) = (mv(r0l1, q.mv_l1_x), mv(r0l1, q.mv_l1_y));
    let (m1l0x, m1l0y) = (mv(r1l0, p.mv_l0_x), mv(r1l0, p.mv_l0_y));
    let (m1l1x, m1l1y) = (mv(r1l1, p.mv_l1_x), mv(r1l1, p.mv_l1_y));
    let ge4 = |a: i32, b: i32| (a - b).abs() >= 4;
    if (r0l0 == r1l0 && r0l1 == r1l1) || (r0l0 == r1l1 && r0l1 == r1l0) {
        if r0l0 == r0l1 {
            // eq. 1230: all four cross pairings.
            (ge4(m0l0x, m1l0x)
                || ge4(m0l0y, m1l0y)
                || ge4(m0l1x, m1l1x)
                || ge4(m0l1y, m1l1y)
                || ge4(m0l0x, m1l1x)
                || ge4(m0l0y, m1l1y)
                || ge4(m0l1x, m1l0x)
                || ge4(m0l1y, m1l0y)) as u32
        } else if r0l0 == r1l0 && r0l1 == r1l1 {
            // eq. 1231.
            (ge4(m0l0x, m1l0x) || ge4(m0l0y, m1l0y) || ge4(m0l1x, m1l1x) || ge4(m0l1y, m1l1y))
                as u32
        } else {
            // eq. 1232 (crossed lists).
            (ge4(m0l0x, m1l1x) || ge4(m0l0y, m1l1y) || ge4(m0l1x, m1l0x) || ge4(m0l1y, m1l0y))
                as u32
        }
    } else {
        1
    }
}

/// §8.8.3.5 — thresholds for one edge sample set. Returns
/// `(filter_samples_flag, index_a, alpha, beta)`.
#[allow(clippy::too_many_arguments)]
fn addb_thresholds(
    p0: i32,
    p1: i32,
    q0: i32,
    q1: i32,
    qp_p: i32,
    qp_q: i32,
    bit_depth: u32,
    offsets: AddbOffsets,
    bs: u32,
) -> (bool, usize, i32, i32) {
    let qp_av = (qp_p + qp_q + 1) >> 1; // eq. 1233
    let index_a = (qp_av + offsets.filter_offset_a).clamp(0, 51) as usize; // eq. 1234
    let index_b = (qp_av + offsets.filter_offset_b).clamp(0, 51) as usize; // eq. 1235
    let scale = 1i32 << (bit_depth - 8);
    let alpha = ADDB_ALPHA_PRIME[index_a] * scale; // eq. 1236
    let beta = ADDB_BETA_PRIME[index_b] * scale; // eq. 1237
                                                 // eq. 1238.
    let filter =
        bs != 0 && (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta;
    (filter, index_a, alpha, beta)
}

/// §8.8.3.6 — weak filter (bS < 4) over one 4-sample line. `p` and `q`
/// each hold `p0..p2` / `q0..q2`; filtered in place.
#[allow(clippy::too_many_arguments)]
fn addb_filter_weak(
    p: &mut [i32; 3],
    q: &mut [i32; 3],
    chroma_style: bool,
    bs: u32,
    beta: i32,
    index_a: usize,
    bit_depth: u32,
    max_val: i32,
) {
    let t_c0_prime = ADDB_T_C0[(bs - 1).min(2) as usize][index_a];
    let shift = bit_depth.saturating_sub(9);
    let t_c = t_c0_prime * (1 << shift); // eq. 1239
    let ap = (p[2] - p[0]).abs(); // eq. 1240
    let aq = (q[2] - q[0]).abs(); // eq. 1241
    let t_c0 = if !chroma_style {
        let tc_inc_p = (ap < beta) as i32; // eq. 1242
        let tc_inc_q = (aq < beta) as i32; // eq. 1243
        (t_c0_prime + tc_inc_p + tc_inc_q) * (1 << shift) // eq. 1244
    } else {
        (t_c0_prime + 1) * (1 << shift) // eq. 1245
    };
    // eqs. 1246-1248.
    let delta = ((((q[0] - p[0]) << 2) + (p[1] - q[1]) + 4) >> 3).clamp(-t_c0, t_c0);
    let p0_new = p[0] + delta;
    let q0_new = q[0] - delta;
    // eqs. 1249-1252.
    let p1_new = if !chroma_style && ap < beta {
        p[1] + (((p[2] + p[0] + q[0]) * 3 - (p[1] << 3) - q[1]) >> 4).clamp(-t_c, t_c)
    } else {
        p[1]
    };
    let q1_new = if !chroma_style && aq < beta {
        q[1] + (((q[2] + q[0] + p[0]) * 3 - (q[1] << 3) - p[1]) >> 4).clamp(-t_c, t_c)
    } else {
        q[1]
    };
    // eqs. 1255-1260 (p2/q2 unchanged, eqs. 1253/1254).
    p[0] = p0_new.clamp(0, max_val);
    p[1] = p1_new.clamp(0, max_val);
    q[0] = q0_new.clamp(0, max_val);
    q[1] = q1_new.clamp(0, max_val);
}

/// §8.8.3.7 — strong filter (bS == 4) over one 4-sample line. `p` and
/// `q` hold `p0..p3` / `q0..q3`; the first three of each side are
/// filtered in place.
fn addb_filter_strong(
    p: &mut [i32; 4],
    q: &mut [i32; 4],
    chroma_style: bool,
    alpha: i32,
    beta: i32,
    max_val: i32,
) {
    let ap = (p[2] - p[0]).abs();
    let aq = (q[2] - q[0]).abs();
    let strong_gate = (p[0] - q[0]).abs() < ((alpha >> 2) + 2);
    let (p0n, p1n, p2n) = if !chroma_style && ap < beta && strong_gate {
        // eqs. 1262-1264.
        (
            (p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3,
            (p[2] + p[1] + p[0] + q[0] + 2) >> 2,
            (2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3,
        )
    } else {
        // eqs. 1265-1267.
        ((2 * p[1] + p[0] + q[1] + 2) >> 2, p[1], p[2])
    };
    let (q0n, q1n, q2n) = if !chroma_style && aq < beta && strong_gate {
        // eqs. 1269-1271.
        (
            (p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3,
            (p[0] + q[0] + q[1] + q[2] + 2) >> 2,
            (2 * q[3] + 3 * q[2] + q[1] + q[0] + p[0] + 4) >> 3,
        )
    } else {
        // eqs. 1272-1274.
        ((2 * q[1] + q[0] + p[1] + 2) >> 2, q[1], q[2])
    };
    // eqs. 1275-1280.
    p[0] = p0n.clamp(0, max_val);
    p[1] = p1n.clamp(0, max_val);
    p[2] = p2n.clamp(0, max_val);
    q[0] = q0n.clamp(0, max_val);
    q[1] = q1n.clamp(0, max_val);
    q[2] = q2n.clamp(0, max_val);
}

/// §8.8.3.3 — filter one coding-block boundary edge (the left edge for
/// EDGE_VER, the top edge for EDGE_HOR) of the CB at luma `(x_cb,
/// y_cb)` with the given **luma** log2 dimensions, on plane `c_idx`.
/// Returns the number of filtered sample lines.
#[allow(clippy::too_many_arguments)]
fn addb_filter_cb_edge(
    pic: &mut crate::picture::YuvPicture,
    grid: &SideInfoGrid,
    offsets: AddbOffsets,
    ctb_log2_size_y: u32,
    x_cb: u32,
    y_cb: u32,
    log2_cb_width: u32,
    log2_cb_height: u32,
    c_idx: u32,
    vertical: bool,
    cb_qp_offset: i32,
    cr_qp_offset: i32,
) -> u32 {
    // 4:2:0 only (ChromaArrayType == 1): chroma runs the
    // chromaStyleFilteringFlag == 1 branch.
    let chroma_style = c_idx != 0;
    let sub = if c_idx == 0 { 0u32 } else { 1 };
    let (width, height) = if c_idx == 0 {
        (pic.width, pic.height)
    } else {
        (pic.width.div_ceil(2), pic.height.div_ceil(2))
    };
    let stride = if c_idx == 0 {
        pic.y_stride()
    } else {
        pic.c_stride()
    };
    let bit_depth = pic.bit_depth;
    let max_val = (1i32 << bit_depth) - 1;
    // Component-domain block geometry (log2CbWidth/Height are the
    // component dims per §8.8.3.2's cIdx remapping).
    let log2_w = log2_cb_width - sub;
    let log2_h = log2_cb_height - sub;
    let x_cb_c = x_cb >> sub;
    let y_cb_c = y_cb >> sub;
    // Picture-boundary edges are never filtered.
    if (vertical && x_cb_c == 0) || (!vertical && y_cb_c == 0) {
        return 0;
    }
    // Edge-set geometry: luma iterates 4-sample groups with d = 0..3;
    // 4:2:0 chroma-style iterates 2-sample groups with d = 0..1.
    let (steps, group, taps) = if !chroma_style {
        if vertical {
            (1u32 << (log2_h - 2), 4u32, 4usize)
        } else {
            (1u32 << (log2_w - 2), 4u32, 4usize)
        }
    } else if vertical {
        (1u32 << (log2_h.max(1) - 1), 2u32, 2usize)
    } else {
        (1u32 << (log2_w.max(1) - 1), 2u32, 2usize)
    };
    let mut filtered = 0u32;
    for step in 0..steps {
        // bS at luma granularity (§8.8.3.3 step 3: derived from the
        // luma positions; the chroma-style branch reuses the same
        // §8.8.3.4 derivation over the covering luma cells).
        let d_luma = (step * group) << sub;
        let (xp_l, yp_l, xq_l, yq_l) = if vertical {
            (x_cb - 1, y_cb + d_luma, x_cb, y_cb + d_luma)
        } else {
            (x_cb + d_luma, y_cb - 1, x_cb + d_luma, y_cb)
        };
        if xq_l >= pic.width || yq_l >= pic.height {
            continue;
        }
        let bs = addb_bs(grid, xp_l, yp_l, xq_l, yq_l, ctb_log2_size_y);
        if bs == 0 {
            continue;
        }
        // Per-CU QP (§8.8.3.5): chroma edges take the co-located luma
        // CU's QpY plus the slice chroma offset (the crate's chroma-QP
        // simplification, mirroring the §8.8.2 path).
        let qp_off = match c_idx {
            1 => cb_qp_offset,
            2 => cr_qp_offset,
            _ => 0,
        };
        let qp_p =
            (grid.at((xp_l >> 2) as usize, (yp_l >> 2) as usize).qp_y as i32 + qp_off).clamp(0, 51);
        let qp_q =
            (grid.at((xq_l >> 2) as usize, (yq_l >> 2) as usize).qp_y as i32 + qp_off).clamp(0, 51);
        for d in 0..group {
            // Component-domain line coordinates.
            let (line_x, line_y) = if vertical {
                (x_cb_c, y_cb_c + step * group + d)
            } else {
                (x_cb_c + step * group + d, y_cb_c)
            };
            if line_x >= width || line_y >= height {
                continue;
            }
            // Gather p0..p3 / q0..q3 across the edge.
            let plane: &mut Vec<u16> = match c_idx {
                0 => &mut pic.y,
                1 => &mut pic.cb,
                _ => &mut pic.cr,
            };
            let fetch = |plane: &Vec<u16>, k: i64| -> Option<i32> {
                let (x, y) = if vertical {
                    (line_x as i64 + k, line_y as i64)
                } else {
                    (line_x as i64, line_y as i64 + k)
                };
                if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                    return None;
                }
                Some(plane[y as usize * stride + x as usize] as i32)
            };
            // p_k at −1−k, q_k at +k relative to the edge.
            let mut pv = [0i32; 4];
            let mut qv = [0i32; 4];
            let mut ok = true;
            for k in 0..taps {
                match (fetch(plane, -1 - k as i64), fetch(plane, k as i64)) {
                    (Some(a), Some(b)) => {
                        pv[k] = a;
                        qv[k] = b;
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            // For the 2-tap chroma gather, mirror the outermost sample
            // so the p2/q2 reads of the filter maths stay in bounds
            // (chroma-style filtering never *writes* p1/p2).
            if taps == 2 {
                pv[2] = pv[1];
                qv[2] = qv[1];
                pv[3] = pv[2];
                qv[3] = qv[2];
            }
            let (filter, index_a, alpha, beta) = addb_thresholds(
                pv[0], pv[1], qv[0], qv[1], qp_p, qp_q, bit_depth, offsets, bs,
            );
            if !filter {
                continue;
            }
            if bs < 4 {
                let mut p3 = [pv[0], pv[1], pv[2]];
                let mut q3 = [qv[0], qv[1], qv[2]];
                addb_filter_weak(
                    &mut p3,
                    &mut q3,
                    chroma_style,
                    bs,
                    beta,
                    index_a,
                    bit_depth,
                    max_val,
                );
                pv[0] = p3[0];
                pv[1] = p3[1];
                pv[2] = p3[2];
                qv[0] = q3[0];
                qv[1] = q3[1];
                qv[2] = q3[2];
            } else {
                let mut p4 = pv;
                let mut q4 = qv;
                addb_filter_strong(&mut p4, &mut q4, chroma_style, alpha, beta, max_val);
                pv = p4;
                qv = q4;
            }
            // Write back p0..p2 / q0..q2 (chroma-style writes p0/q0
            // only beyond which the maths left p1/p2 untouched — the
            // mirrored gather guarantees the copies are no-ops).
            let write_taps = if taps == 2 { 1 } else { 3 };
            for k in 0..write_taps {
                let (px, py) = if vertical {
                    (line_x as i64 - 1 - k as i64, line_y as i64)
                } else {
                    (line_x as i64, line_y as i64 - 1 - k as i64)
                };
                let (qx, qy) = if vertical {
                    (line_x as i64 + k as i64, line_y as i64)
                } else {
                    (line_x as i64, line_y as i64 + k as i64)
                };
                if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                    plane[py as usize * stride + px as usize] = pv[k] as u16;
                }
                if qx >= 0 && qy >= 0 && (qx as u32) < width && (qy as u32) < height {
                    plane[qy as usize * stride + qx as usize] = qv[k] as u16;
                }
            }
            filtered += 1;
        }
    }
    filtered
}

/// §8.8.3.1/.2 — run the advanced deblocking filter over a whole
/// reconstructed picture: vertical edges first, then horizontal, on a
/// per-coding-block basis (the CB set is recovered from the side-info
/// grid's covering-CU geometry), with the §8.8.3.2 `splitTH` split for
/// blocks wider/taller than the maximum transform size. Only edges on
/// the 8×8 luma grid are filtered (§8.8.3.1). Returns the number of
/// filtered sample lines.
#[allow(clippy::too_many_arguments)]
pub fn addb_deblock_picture(
    pic: &mut crate::picture::YuvPicture,
    grid: &SideInfoGrid,
    offsets: AddbOffsets,
    ctb_log2_size_y: u32,
    max_tb_log2_size_y: u32,
    cb_qp_offset: i32,
    cr_qp_offset: i32,
) -> u32 {
    let chroma = pic.chroma_format_idc != 0;
    let mut filtered = 0u32;
    for vertical in [true, false] {
        // Enumerate CUs: a cell is a CU origin iff its recorded covering
        // geometry starts at that cell.
        for yc in 0..grid.h_cells {
            for xc in 0..grid.w_cells {
                let cell = grid.at(xc, yc);
                if cell.cu_log2_w == 0 {
                    continue; // never decoded (out-of-picture padding)
                }
                let x0 = (xc as u32) << 2;
                let y0 = (yc as u32) << 2;
                if cell.cu_x0 as u32 != x0 || cell.cu_y0 as u32 != y0 {
                    continue; // interior cell
                }
                let log2_w = cell.cu_log2_w as u32;
                let log2_h = cell.cu_log2_h as u32;
                // §8.8.3.1: only 8×8-luma-grid edges are filtered.
                let on_grid = if vertical { x0 % 8 == 0 } else { y0 % 8 == 0 };
                if !on_grid {
                    continue;
                }
                let mut planes: Vec<u32> = vec![0];
                if chroma {
                    planes.push(1);
                    planes.push(2);
                }
                for c_idx in planes {
                    // §8.8.3.2 splitTH: split over-wide/tall blocks into
                    // two boundary invocations.
                    let split_th = if c_idx == 0 {
                        max_tb_log2_size_y
                    } else {
                        max_tb_log2_size_y - 1
                    };
                    let mut calls: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(2);
                    if vertical && log2_w > split_th {
                        calls.push((x0, y0, log2_w >> 1, log2_h));
                        calls.push((x0 + (1 << split_th), y0, log2_w >> 1, log2_h));
                    } else if !vertical && log2_h > split_th {
                        calls.push((x0, y0, log2_w, log2_h >> 1));
                        calls.push((x0, y0 + (1 << split_th), log2_w, log2_h >> 1));
                    } else {
                        calls.push((x0, y0, log2_w, log2_h));
                    }
                    for (cx, cy, lw, lh) in calls {
                        // §8.8.3.1: edges that correspond to tile
                        // boundaries are exempt when
                        // loop_filter_across_tiles_enabled_flag == 0
                        // (coordinates are luma-domain CB positions).
                        if let Some(tb) = &grid.tile_bounds {
                            if (vertical && tb.is_col_boundary(cx))
                                || (!vertical && tb.is_row_boundary(cy))
                            {
                                continue;
                            }
                        }
                        filtered += addb_filter_cb_edge(
                            pic,
                            grid,
                            offsets,
                            ctb_log2_size_y,
                            cx,
                            cy,
                            lw,
                            lh,
                            c_idx,
                            vertical,
                            cb_qp_offset,
                            cr_qp_offset,
                        );
                    }
                }
            }
        }
    }
    filtered
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

    /// Round 416 — §8.8.2.1 tile-boundary edge exemption: with
    /// `tile_bounds` armed at the luma column x = 8, the vertical edge
    /// between two residual-carrying inter CUs is NOT filtered, while
    /// the same picture without the exemption smooths it. Non-boundary
    /// edges behave identically in both runs.
    #[test]
    fn round416_deblock_luma_skips_tile_boundary_edges() {
        let mk_pic = || {
            let mut pic = YuvPicture::new(16, 8, 1, 8).unwrap();
            for j in 0..8usize {
                for i in 0..16usize {
                    pic.y[j * 16 + i] = if i < 8 { 100 } else { 106 };
                }
            }
            for j in 0..4usize {
                for i in 0..8usize {
                    pic.cb[j * 8 + i] = if i < 4 { 90 } else { 96 };
                }
            }
            pic
        };
        let cu = |x0: u16| CuSideInfo {
            pred_mode: CuPredMode::Inter,
            cbf_luma: 1,
            cu_x0: x0,
            cu_y0: 0,
            cu_log2_w: 3,
            cu_log2_h: 3,
            ref_idx_l0: 0,
            qp_y: 32,
            ..Default::default()
        };
        let mut grid = SideInfoGrid::new(16, 8);
        grid.stamp_block(0, 0, 8, 8, cu(0));
        grid.stamp_block(8, 0, 8, 8, cu(8));

        // Unexempt control: the x = 8 step is smoothed.
        let mut pic_open = mk_pic();
        deblock_luma(&mut pic_open, &grid, 32).unwrap();
        assert_ne!(
            pic_open.y[7], 100,
            "control must filter the x = 8 luma edge"
        );

        // Tile boundary at x = 8 with across-tiles filtering off.
        grid.tile_bounds = Some(crate::tiles::TileBounds {
            col_bd: vec![8],
            row_bd: vec![],
        });
        let mut pic_tiled = mk_pic();
        deblock_luma(&mut pic_tiled, &grid, 32).unwrap();
        for j in 0..8usize {
            for i in 6..10usize {
                let want = if i < 8 { 100 } else { 106 };
                assert_eq!(
                    pic_tiled.y[j * 16 + i],
                    want,
                    "tile-boundary luma edge must stay unfiltered at ({i}, {j})"
                );
            }
        }
        // Chroma: the boundary maps to chroma x = 4; it too stays
        // unfiltered under the exemption.
        let mut pic_c = mk_pic();
        deblock_chroma(&mut pic_c, &grid, 32, 0, 1).unwrap();
        for j in 0..4usize {
            for i in 3..5usize {
                let want = if i < 4 { 90 } else { 96 };
                assert_eq!(
                    pic_c.cb[j * 8 + i],
                    want,
                    "tile-boundary chroma edge must stay unfiltered at ({i}, {j})"
                );
            }
        }
    }

    /// Round 416 — §8.8.3.1 (ADDB) tile-boundary edge exemption: the
    /// left edge of a CB whose x0 coincides with an armed tile-column
    /// boundary is skipped; without the exemption the same edge
    /// filters.
    #[test]
    fn round416_addb_skips_tile_boundary_edges() {
        let mk_pic = || {
            let mut pic = YuvPicture::new(16, 8, 1, 8).unwrap();
            for j in 0..8usize {
                for i in 0..16usize {
                    pic.y[j * 16 + i] = if i < 8 { 100 } else { 106 };
                }
            }
            pic
        };
        let cu = |x0: u16| CuSideInfo {
            pred_mode: CuPredMode::Inter,
            cbf_luma: 1,
            cu_x0: x0,
            cu_y0: 0,
            cu_log2_w: 3,
            cu_log2_h: 3,
            ref_idx_l0: 0,
            qp_y: 40,
            ..Default::default()
        };
        let mut grid = SideInfoGrid::new(16, 8);
        grid.stamp_block(0, 0, 8, 8, cu(0));
        grid.stamp_block(8, 0, 8, 8, cu(8));
        let offsets = AddbOffsets {
            filter_offset_a: 0,
            filter_offset_b: 0,
        };

        let mut pic_open = mk_pic();
        let filtered_open = addb_deblock_picture(&mut pic_open, &grid, offsets, 5, 5, 0, 0);
        assert!(filtered_open > 0);
        assert_ne!(
            pic_open.y[7], 100,
            "control must filter the x = 8 ADDB edge"
        );

        grid.tile_bounds = Some(crate::tiles::TileBounds {
            col_bd: vec![8],
            row_bd: vec![],
        });
        let mut pic_tiled = mk_pic();
        addb_deblock_picture(&mut pic_tiled, &grid, offsets, 5, 5, 0, 0);
        for j in 0..8usize {
            for i in 5..11usize {
                let want = if i < 8 { 100 } else { 106 };
                assert_eq!(
                    pic_tiled.y[j * 16 + i],
                    want,
                    "ADDB tile-boundary edge must stay unfiltered at ({i}, {j})"
                );
            }
        }
    }

    /// Round 416 — the grid's §6.4.1 tile probe: `None` (single tile)
    /// never fires; an armed rectangle marks everything outside it as
    /// other-tile.
    #[test]
    fn round416_grid_neighbour_in_other_tile() {
        let mut grid = SideInfoGrid::new(64, 64);
        assert!(!grid.neighbour_in_other_tile(-1, 0));
        assert!(!grid.neighbour_in_other_tile(63, 63));
        grid.cur_tile = Some(crate::tiles::TileRect {
            x0: 32,
            y0: 0,
            x1: 64,
            y1: 64,
        });
        assert!(grid.neighbour_in_other_tile(31, 0)); // left tile
        assert!(!grid.neighbour_in_other_tile(32, 0)); // inside
        assert!(grid.neighbour_in_other_tile(-1, 0)); // out of picture
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

    // =================================================================
    // Round 397: §8.8.3 advanced deblocking units.
    // =================================================================

    fn addb_cell(pred: CuPredMode, cbf: u8, mv: (i32, i32), r0: i8, qp: u8) -> CuSideInfo {
        CuSideInfo {
            pred_mode: pred,
            cbf_luma: cbf,
            mv_l0_x: mv.0,
            mv_l0_y: mv.1,
            ref_idx_l0: r0,
            ref_idx_l1: -1,
            cu_x0: 0,
            cu_y0: 0,
            cu_log2_w: 3,
            cu_log2_h: 3,
            qp_y: qp,
            ..Default::default()
        }
    }

    /// §8.8.3.4 — the boundary-strength cascade: intra across a CTU
    /// boundary → 4; intra or IBC → 3; coded residual → 2; ≥ 1-pel MV
    /// difference → 1; matching motion → 0.
    #[test]
    fn round397_addb_bs_cascade() {
        let mut g = SideInfoGrid::new(128, 64);
        // P at (56..64, 0..8) intra; Q at (64..72, 0..8) intra — the
        // 64-boundary is a CTU boundary at CtbLog2SizeY = 6.
        g.stamp_block(56, 0, 8, 8, addb_cell(CuPredMode::Intra, 0, (0, 0), -1, 30));
        g.stamp_block(64, 0, 8, 8, addb_cell(CuPredMode::Intra, 0, (0, 0), -1, 30));
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 6), 4, "intra across CTU");
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 7), 3, "same CTU → 3");
        // IBC neighbour → 3.
        g.stamp_block(64, 0, 8, 8, addb_cell(CuPredMode::Ibc, 0, (0, 0), -1, 30));
        g.stamp_block(56, 0, 8, 8, addb_cell(CuPredMode::Inter, 0, (0, 0), 0, 30));
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 6), 3);
        // Coded inter residual → 2.
        g.stamp_block(64, 0, 8, 8, addb_cell(CuPredMode::Inter, 1, (0, 0), 0, 30));
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 6), 2);
        // Motion difference ≥ 4 quarter-pel → 1 (eq. 1231 shape:
        // refIdx0L0 == refIdx1L0 = 0, both L1 = −1).
        g.stamp_block(64, 0, 8, 8, addb_cell(CuPredMode::Inter, 0, (4, 0), 0, 30));
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 6), 1);
        // Matching motion → 0.
        g.stamp_block(64, 0, 8, 8, addb_cell(CuPredMode::Inter, 0, (0, 0), 0, 30));
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 6), 0);
        // Different reference indices → 1.
        g.stamp_block(64, 0, 8, 8, addb_cell(CuPredMode::Inter, 0, (0, 0), 1, 30));
        assert_eq!(addb_bs(&g, 63, 0, 64, 0, 6), 1);
    }

    /// Table 34/35 spot values + the eq. 1233-1238 threshold math.
    #[test]
    fn round397_addb_thresholds_tables() {
        assert_eq!(ADDB_ALPHA_PRIME[16], 4);
        assert_eq!(ADDB_ALPHA_PRIME[38], 63);
        assert_eq!(ADDB_ALPHA_PRIME[51], 255);
        assert_eq!(ADDB_BETA_PRIME[16], 2);
        assert_eq!(ADDB_BETA_PRIME[51], 18);
        assert_eq!(ADDB_T_C0[0][51], 13);
        assert_eq!(ADDB_T_C0[1][51], 17);
        assert_eq!(ADDB_T_C0[2][51], 25);
        // qPav = (40 + 40 + 1) >> 1 = 40 → α = 80, β = 13 at 8-bit.
        let (filter, index_a, alpha, beta) =
            addb_thresholds(60, 60, 68, 68, 40, 40, 8, AddbOffsets::default(), 2);
        assert_eq!(index_a, 40);
        assert_eq!(alpha, 80);
        assert_eq!(beta, 13);
        assert!(filter, "|p0−q0| = 8 < 80, flat sides");
        // FilterOffsetA shifts indexA (eq. 1234).
        let (_, index_a2, _, _) = addb_thresholds(
            60,
            60,
            68,
            68,
            40,
            40,
            8,
            AddbOffsets {
                filter_offset_a: -12,
                filter_offset_b: 0,
            },
            2,
        );
        assert_eq!(index_a2, 28);
        // bS = 0 never filters (eq. 1238).
        let (f0, _, _, _) = addb_thresholds(60, 60, 68, 68, 40, 40, 8, AddbOffsets::default(), 0);
        assert!(!f0);
    }

    /// §8.8.3.6 weak filter, hand-traced: flat 60|68 step at
    /// indexA = 40, bS = 2 → t′C0 = 5, tC0 = 7, Δ = 4;
    /// p′1 = 61, q′1 = 67 (the eq. 1249/1251 quarter-step).
    #[test]
    fn round397_addb_weak_filter_hand_trace() {
        let mut p = [60, 60, 60];
        let mut q = [68, 68, 68];
        addb_filter_weak(&mut p, &mut q, false, 2, 13, 40, 8, 255);
        assert_eq!(p[0], 63, "Δ = ((8 << 2) + (p1 − q1) + 4) >> 3 = 3");
        assert_eq!(q[0], 65);
        assert_eq!(p[1], 61);
        assert_eq!(q[1], 67);
        assert_eq!(p[2], 60, "p2 never written");
        assert_eq!(q[2], 68);
        // chromaStyle: p1/q1 stay, tC0 = (t′C0 + 1).
        let mut pc = [60, 60, 60];
        let mut qc = [68, 68, 68];
        addb_filter_weak(&mut pc, &mut qc, true, 2, 13, 40, 8, 255);
        assert_eq!(pc[1], 60);
        assert_eq!(qc[1], 68);
        assert_eq!(pc[0], 63);
    }

    /// §8.8.3.7 strong filter, hand-traced on a flat 100|140 step with
    /// α = 255, β = 18: the eqs. 1262-1264 / 1269-1271 3-tap smoothing.
    #[test]
    fn round397_addb_strong_filter_hand_trace() {
        let mut p = [100, 100, 100, 100];
        let mut q = [140, 140, 140, 140];
        addb_filter_strong(&mut p, &mut q, false, 255, 18, 255);
        assert_eq!(p[0], (100 + 200 + 200 + 280 + 140 + 4) >> 3); // 115
        assert_eq!(p[1], (100 + 100 + 100 + 140 + 2) >> 2); // 110
        assert_eq!(p[2], (200 + 300 + 100 + 100 + 140 + 4) >> 3); // 105
        assert_eq!(q[0], (100 + 200 + 280 + 280 + 140 + 4) >> 3); // 125
        assert_eq!(q[1], (100 + 140 + 140 + 140 + 2) >> 2); // 130
        assert_eq!(q[2], (280 + 420 + 140 + 140 + 100 + 4) >> 3); // 135
                                                                  // Strong-gate failure (|p0 − q0| ≥ (α >> 2) + 2) falls back to
                                                                  // the eqs. 1265/1272 1-tap smoothing.
        let mut p2 = [100, 100, 100, 100];
        let mut q2 = [200, 200, 200, 200];
        addb_filter_strong(&mut p2, &mut q2, false, 100, 18, 255);
        assert_eq!(p2[0], (200 + 100 + 200 + 2) >> 2); // eq. 1265
        assert_eq!(p2[1], 100);
        assert_eq!(q2[0], (400 + 200 + 100 + 2) >> 2); // eq. 1272
    }
}
