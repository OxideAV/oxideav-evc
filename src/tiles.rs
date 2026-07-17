//! Picture tile layout for the pixel decoders (ISO/IEC 23094-1 §6.5.1
//! / §6.3.2 / §6.4.1).
//!
//! The §6.5.1 derivations themselves (`ColWidth`/`RowHeight`/`ColBd`/
//! `RowBd`/`CtbAddrRsToTs`/`CtbAddrTsToRs`/`TileId`, eqs. 24-30) live
//! in [`crate::pps`]; this module packages their *luma-sample-domain*
//! consequences for the reconstruction pipeline:
//!
//! * [`TileRect`] — the half-open luma rectangle one tile covers. The
//!   §6.4.1 / §6.4.3 first bullet ("the neighbouring block is contained
//!   in a different tile than the current block") reduces, for a
//!   decoder positioned inside one tile, to a containment test against
//!   the current tile's rectangle: EVC tiles are rectangular unions of
//!   CTBs, so a luma location is in the same tile iff it lies inside
//!   the rectangle.
//! * [`PicTileLayout`] — the whole-picture tile grid in luma samples:
//!   the eq. 26/27 `ColBd[]`/`RowBd[]` boundaries scaled by
//!   `CtbLog2SizeY` and clamped to the picture extent, plus the §7.4.4.2
//!   `loop_filter_across_tiles_enabled_flag`. This is what the in-loop
//!   filters consult: §8.8.2/§8.8.3 exempt "the edges that coincide
//!   with tile boundaries when loop_filter_across_tiles_enabled_flag is
//!   equal to 0", and §8.8.4.5/.6 switch the per-CTB availability
//!   derivation between §6.4.1 (across-tiles disabled) and §6.4.4
//!   (across-tiles enabled).
//!
//! For a `single_tile_in_pic_flag == 1` picture the layout degenerates
//! to one tile covering the picture, every containment test passes, and
//! no edge coincides with an interior tile boundary — byte-identical
//! behaviour to the historical single-tile decoder paths.

use oxideav_core::{Error, Result};

/// Half-open luma-sample rectangle `[x0, x1) × [y0, y1)` covered by one
/// tile (§6.3.2: a tile is a rectangular region of CTUs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRect {
    /// Left luma column (inclusive).
    pub x0: u32,
    /// Top luma row (inclusive).
    pub y0: u32,
    /// Right luma column (exclusive), clamped to `pic_width`.
    pub x1: u32,
    /// Bottom luma row (exclusive), clamped to `pic_height`.
    pub y1: u32,
}

impl TileRect {
    /// Whether the (signed) luma location lies inside this tile. The
    /// §6.4.1 first bullet is `!contains(...)` for in-picture
    /// locations; out-of-picture locations are rejected by the §6.4.1
    /// extent bullets regardless.
    #[inline]
    pub fn contains(&self, x: i64, y: i64) -> bool {
        x >= self.x0 as i64 && x < self.x1 as i64 && y >= self.y0 as i64 && y < self.y1 as i64
    }

    /// The same rectangle in a sub-sampled component domain
    /// (`sub_w`/`sub_h` ∈ {1, 2} per Table 4). Tile boundaries are
    /// CTB-aligned, so the division is exact for every legal
    /// chroma format.
    #[inline]
    pub fn component(&self, sub_w: u32, sub_h: u32) -> TileRect {
        TileRect {
            x0: self.x0 / sub_w,
            y0: self.y0 / sub_h,
            x1: self.x1.div_ceil(sub_w),
            y1: self.y1.div_ceil(sub_h),
        }
    }
}

/// The picture's tile grid in luma samples plus the §7.4.4.2 loop-filter
/// gating flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PicTileLayout {
    /// Luma-sample x of every tile-column boundary: `ColBd[ i ] <<
    /// CtbLog2SizeY` for `i = 0 ..= num_tile_columns`, with the final
    /// entry clamped to `pic_width`. Strictly increasing.
    pub col_bd_luma: Vec<u32>,
    /// Luma-sample y of every tile-row boundary, final entry clamped to
    /// `pic_height`. Strictly increasing.
    pub row_bd_luma: Vec<u32>,
    /// `pic_width_in_luma_samples`.
    pub pic_width: u32,
    /// `pic_height_in_luma_samples`.
    pub pic_height: u32,
    /// §7.4.4.2 `loop_filter_across_tiles_enabled_flag` (inferred 1
    /// when absent — the single-tile case).
    pub loop_filter_across_tiles_enabled: bool,
}

impl PicTileLayout {
    /// The degenerate one-tile layout covering the whole picture
    /// (`single_tile_in_pic_flag == 1`; the loop-filter flag is
    /// inferred 1 per §7.4.4.2).
    pub fn single_tile(pic_width: u32, pic_height: u32) -> Self {
        Self {
            col_bd_luma: vec![0, pic_width],
            row_bd_luma: vec![0, pic_height],
            pic_width,
            pic_height,
            loop_filter_across_tiles_enabled: true,
        }
    }

    /// Build the layout from the §6.5.1 eq. 26/27 CTB-unit boundary
    /// lists ([`crate::pps::compute_col_bd`] / [`crate::pps::compute_row_bd`]).
    ///
    /// # Errors
    ///
    /// Boundary lists shorter than 2 entries, non-increasing, or whose
    /// final entry falls short of the picture extent are rejected — the
    /// eq. 24-27 derivations always cover the picture exactly, so a
    /// short list means the caller fed inconsistent PPS geometry.
    pub fn from_ctb_bounds(
        col_bd: &[u32],
        row_bd: &[u32],
        ctb_log2_size_y: u32,
        pic_width: u32,
        pic_height: u32,
        loop_filter_across_tiles_enabled: bool,
    ) -> Result<Self> {
        let scale = |bd: &[u32], extent: u32, axis: &str| -> Result<Vec<u32>> {
            if bd.len() < 2 {
                return Err(Error::invalid(format!(
                    "evc tiles: {axis} boundary list needs >= 2 entries (got {})",
                    bd.len()
                )));
            }
            let mut out = Vec::with_capacity(bd.len());
            for (i, &b) in bd.iter().enumerate() {
                let luma = (b << ctb_log2_size_y).min(extent);
                if let Some(&prev) = out.last() {
                    if luma <= prev {
                        return Err(Error::invalid(format!(
                            "evc tiles: {axis} boundary {i} ({luma}) not increasing"
                        )));
                    }
                }
                out.push(luma);
            }
            if *out.last().expect("len >= 2") != extent {
                return Err(Error::invalid(format!(
                    "evc tiles: {axis} boundaries end at {} != picture extent {extent}",
                    out.last().expect("len >= 2")
                )));
            }
            Ok(out)
        };
        Ok(Self {
            col_bd_luma: scale(col_bd, pic_width, "column")?,
            row_bd_luma: scale(row_bd, pic_height, "row")?,
            pic_width,
            pic_height,
            loop_filter_across_tiles_enabled,
        })
    }

    /// Number of tile columns.
    pub fn num_tile_columns(&self) -> usize {
        self.col_bd_luma.len() - 1
    }

    /// Number of tile rows.
    pub fn num_tile_rows(&self) -> usize {
        self.row_bd_luma.len() - 1
    }

    /// Whether the picture holds more than one tile.
    pub fn is_multi_tile(&self) -> bool {
        self.num_tile_columns() > 1 || self.num_tile_rows() > 1
    }

    /// The tile rectangle containing the in-picture luma location
    /// `(x, y)`.
    pub fn tile_rect_at(&self, x: u32, y: u32) -> TileRect {
        let ci = interval_index(&self.col_bd_luma, x);
        let rj = interval_index(&self.row_bd_luma, y);
        TileRect {
            x0: self.col_bd_luma[ci],
            y0: self.row_bd_luma[rj],
            x1: self.col_bd_luma[ci + 1],
            y1: self.row_bd_luma[rj + 1],
        }
    }

    /// Whether a vertical edge at luma column `x` coincides with an
    /// *interior* tile-column boundary (§8.8.2 / §8.8.3 edge
    /// exemption; the picture edge at `x == 0` is excluded separately
    /// by the filters).
    pub fn is_interior_col_boundary(&self, x: u32) -> bool {
        x != 0 && x != self.pic_width && self.col_bd_luma.contains(&x)
    }

    /// Whether a horizontal edge at luma row `y` coincides with an
    /// interior tile-row boundary.
    pub fn is_interior_row_boundary(&self, y: u32) -> bool {
        y != 0 && y != self.pic_height && self.row_bd_luma.contains(&y)
    }
}

/// Index `i` such that `bd[i] <= v < bd[i+1]`, clamping to the final
/// interval for `v` beyond the last boundary (out-of-picture callers
/// are filtered by the §6.4.1 extent bullets before the tile test).
fn interval_index(bd: &[u32], v: u32) -> usize {
    debug_assert!(bd.len() >= 2);
    let mut idx = 0;
    for (i, &b) in bd.iter().enumerate().skip(1) {
        if v >= b {
            idx = i;
        } else {
            break;
        }
    }
    idx.min(bd.len() - 2)
}

/// Interior tile boundaries consulted by the loop filters when
/// `loop_filter_across_tiles_enabled_flag == 0` — the §8.8.2 / §8.8.3
/// "edges that coincide with tile boundaries" exemption set. Carried on
/// [`crate::deblock::SideInfoGrid`] so the deblocking entry points need
/// no signature change; `None` (the default) means no exempt edges
/// (single tile, or across-tiles filtering enabled).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileBounds {
    /// Luma-sample x of every interior tile-column boundary.
    pub col_bd: Vec<u32>,
    /// Luma-sample y of every interior tile-row boundary.
    pub row_bd: Vec<u32>,
}

impl TileBounds {
    /// The §8.8.2/§8.8.3 exemption set for a layout: interior
    /// boundaries only, and only when across-tiles filtering is
    /// disabled (otherwise no edge is exempt → `None`).
    pub fn for_loop_filters(layout: &PicTileLayout) -> Option<Self> {
        if layout.loop_filter_across_tiles_enabled || !layout.is_multi_tile() {
            return None;
        }
        Some(Self {
            col_bd: layout.col_bd_luma[1..layout.col_bd_luma.len() - 1].to_vec(),
            row_bd: layout.row_bd_luma[1..layout.row_bd_luma.len() - 1].to_vec(),
        })
    }

    /// Whether a vertical edge at luma column `x` is exempt.
    #[inline]
    pub fn is_col_boundary(&self, x: u32) -> bool {
        self.col_bd.contains(&x)
    }

    /// Whether a horizontal edge at luma row `y` is exempt.
    #[inline]
    pub fn is_row_boundary(&self, y: u32) -> bool {
        self.row_bd.contains(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 tile grid over a 128×128 picture (CTB 64): boundary lists
    /// scale by CtbLog2SizeY and the four rects tile the picture.
    #[test]
    fn round416_layout_2x2_rects() {
        let l = PicTileLayout::from_ctb_bounds(&[0, 1, 2], &[0, 1, 2], 6, 128, 128, false).unwrap();
        assert_eq!(l.num_tile_columns(), 2);
        assert_eq!(l.num_tile_rows(), 2);
        assert!(l.is_multi_tile());
        assert_eq!(
            l.tile_rect_at(0, 0),
            TileRect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 64
            }
        );
        assert_eq!(
            l.tile_rect_at(64, 0),
            TileRect {
                x0: 64,
                y0: 0,
                x1: 128,
                y1: 64
            }
        );
        assert_eq!(
            l.tile_rect_at(127, 127),
            TileRect {
                x0: 64,
                y0: 64,
                x1: 128,
                y1: 128
            }
        );
    }

    /// The final boundary clamps to the picture extent when the last
    /// tile column/row extends past a non-CTB-multiple picture edge.
    #[test]
    fn round416_layout_clamps_to_picture_extent() {
        // 96×80 picture, CTB 64 → 2×2 CTB grid, tile split 1|1 each way.
        let l = PicTileLayout::from_ctb_bounds(&[0, 1, 2], &[0, 1, 2], 6, 96, 80, true).unwrap();
        assert_eq!(l.col_bd_luma, vec![0, 64, 96]);
        assert_eq!(l.row_bd_luma, vec![0, 64, 80]);
        assert_eq!(
            l.tile_rect_at(95, 79),
            TileRect {
                x0: 64,
                y0: 64,
                x1: 96,
                y1: 80
            }
        );
    }

    /// §6.4.1 tile-containment reduction: locations inside the rect are
    /// same-tile, everything else is the "different tile" bullet.
    #[test]
    fn round416_tile_rect_contains() {
        let r = TileRect {
            x0: 64,
            y0: 0,
            x1: 128,
            y1: 64,
        };
        assert!(r.contains(64, 0));
        assert!(r.contains(127, 63));
        assert!(!r.contains(63, 0)); // left neighbour tile
        assert!(!r.contains(128, 0)); // right of the rect
        assert!(!r.contains(64, 64)); // below the rect
        assert!(!r.contains(-1, 0)); // out of picture
    }

    /// Component-domain scaling for 4:2:0 halves both axes exactly
    /// (tile boundaries are CTB-aligned).
    #[test]
    fn round416_tile_rect_component_scaling() {
        let r = TileRect {
            x0: 64,
            y0: 64,
            x1: 96,
            y1: 80,
        };
        assert_eq!(
            r.component(2, 2),
            TileRect {
                x0: 32,
                y0: 32,
                x1: 48,
                y1: 40
            }
        );
        // 4:4:4 (sub 1) is the identity.
        assert_eq!(r.component(1, 1), r);
    }

    /// The single-tile layout: one rect, no interior boundaries, no
    /// loop-filter exemptions.
    #[test]
    fn round416_single_tile_layout_degenerates() {
        let l = PicTileLayout::single_tile(64, 48);
        assert!(!l.is_multi_tile());
        assert!(l.loop_filter_across_tiles_enabled);
        assert_eq!(
            l.tile_rect_at(10, 10),
            TileRect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 48
            }
        );
        assert!(!l.is_interior_col_boundary(0));
        assert!(!l.is_interior_col_boundary(64));
        assert_eq!(TileBounds::for_loop_filters(&l), None);
    }

    /// The loop-filter exemption set: interior boundaries only, and
    /// only when across-tiles filtering is off.
    #[test]
    fn round416_tile_bounds_for_loop_filters() {
        let off =
            PicTileLayout::from_ctb_bounds(&[0, 1, 2], &[0, 1, 2], 6, 128, 128, false).unwrap();
        let tb = TileBounds::for_loop_filters(&off).unwrap();
        assert_eq!(tb.col_bd, vec![64]);
        assert_eq!(tb.row_bd, vec![64]);
        assert!(tb.is_col_boundary(64));
        assert!(!tb.is_col_boundary(0));
        assert!(!tb.is_col_boundary(128));

        let on = PicTileLayout::from_ctb_bounds(&[0, 1, 2], &[0, 1, 2], 6, 128, 128, true).unwrap();
        assert_eq!(TileBounds::for_loop_filters(&on), None);
    }

    /// Inconsistent boundary lists are rejected.
    #[test]
    fn round416_layout_rejects_bad_bounds() {
        assert!(PicTileLayout::from_ctb_bounds(&[0], &[0, 1], 6, 64, 64, true).is_err());
        assert!(PicTileLayout::from_ctb_bounds(&[0, 1, 1], &[0, 1], 6, 128, 64, true).is_err());
        // Final boundary short of the picture extent.
        assert!(PicTileLayout::from_ctb_bounds(&[0, 1], &[0, 1], 6, 128, 64, true).is_err());
    }

    /// `interval_index` picks the containing half-open interval.
    #[test]
    fn round416_interval_index() {
        let bd = [0u32, 64, 128];
        assert_eq!(interval_index(&bd, 0), 0);
        assert_eq!(interval_index(&bd, 63), 0);
        assert_eq!(interval_index(&bd, 64), 1);
        assert_eq!(interval_index(&bd, 127), 1);
        // Beyond the last boundary clamps to the final interval.
        assert_eq!(interval_index(&bd, 500), 1);
    }
}
