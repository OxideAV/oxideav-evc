//! EVC picture parameter set parser (ISO/IEC 23094-1 §7.3.2.2).
//!
//! The PPS carries the per-picture tile geometry, deblocking enable, and
//! the cu_qp_delta gating. Round-1 surfaces every field but the explicit
//! tile-id payload (which is skipped past with a strict sanity bound to
//! keep header allocations small).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Hard upper bound on per-picture tile counts. EVC profiles cap this far
/// below 256, but a wider bound here keeps experimental encoders working.
const MAX_TILES_PER_DIM: u32 = 256;

#[derive(Clone, Debug)]
pub struct Pps {
    pub pps_pic_parameter_set_id: u32,
    pub pps_seq_parameter_set_id: u32,
    pub num_ref_idx_default_active_minus1: [u32; 2],
    pub additional_lt_poc_lsb_len: u32,
    pub rpl1_idx_present_flag: bool,
    pub single_tile_in_pic_flag: bool,

    pub num_tile_columns_minus1: u32,
    pub num_tile_rows_minus1: u32,
    pub uniform_tile_spacing_flag: bool,
    pub tile_column_width_minus1: Vec<u32>,
    pub tile_row_height_minus1: Vec<u32>,
    pub loop_filter_across_tiles_enabled_flag: bool,
    pub tile_offset_len_minus1: u32,

    pub tile_id_len_minus1: u32,
    pub explicit_tile_id_flag: bool,
    /// Indices stored as flat row-major; row count = num_tile_rows+1,
    /// col count = num_tile_cols+1.
    pub tile_id_val: Vec<u32>,

    pub pic_dra_enabled_flag: bool,
    pub pic_dra_aps_id: u8,

    pub arbitrary_slice_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub log2_cu_qp_delta_area_minus6: u32,
}

/// One step of the §6.5.1 eq. (30) outer tile-enumeration loop.
///
/// Carries the `tileIdx` counter and the `(tile_row_j, tile_col_i)`
/// pair so the consumer can dispatch into either the implicit
/// `TileId[…] = tileIdx` branch or — once the per-tile-id ordering
/// ambiguity in §6.5.1 (see [`Pps::tile_grid_coords`]) is resolved
/// upstream — the explicit `tile_id_val[…]` lookup branch.
///
/// The field order matches the spec's local variable names: `j`
/// ranges over rows (outer), `i` ranges over columns (inner), and
/// `tile_idx` advances on the inner-loop step.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileGridCoord {
    /// `tileIdx`, the linear count from the §6.5.1 eq. (30) outer
    /// loop. Ranges from `0` to `NumTilesInPic - 1` inclusive.
    pub tile_idx: u32,
    /// `j`, the tile-row index. Ranges from `0` to
    /// `num_tile_rows_minus1` inclusive.
    pub tile_row_j: u32,
    /// `i`, the tile-column index. Ranges from `0` to
    /// `num_tile_columns_minus1` inclusive.
    pub tile_col_i: u32,
}

impl Pps {
    /// `num_tile_columns_minus1 + 1` from §7.4.3.2 — the number of
    /// tile columns in the picture. Always at least `1`.
    pub fn num_tile_columns(&self) -> u32 {
        self.num_tile_columns_minus1 + 1
    }

    /// `num_tile_rows_minus1 + 1` from §7.4.3.2 — the number of
    /// tile rows in the picture. Always at least `1`.
    pub fn num_tile_rows(&self) -> u32 {
        self.num_tile_rows_minus1 + 1
    }

    /// `NumTilesInPic` from §6.5.1, the product of
    /// [`num_tile_rows`](Self::num_tile_rows) and
    /// [`num_tile_columns`](Self::num_tile_columns).
    ///
    /// The PPS parser's `MAX_TILES_PER_DIM` sanity bound (256 per
    /// axis) keeps the product inside `u32` without overflow.
    pub fn num_tiles_in_pic(&self) -> u32 {
        self.num_tile_rows() * self.num_tile_columns()
    }

    /// Iterator over the §6.5.1 eq. (30) outer tile-enumeration
    /// loop in raster-tile order.
    ///
    /// Yields one [`TileGridCoord`] per tile, advancing `i`
    /// (columns) within each `j` (row) step, with `tileIdx`
    /// counting linearly across both. Total length is
    /// [`num_tiles_in_pic`](Self::num_tiles_in_pic).
    ///
    /// The spec's eq. (30) inner two loops walk every CTB inside
    /// a tile to populate `TileId[ ctbAddrTs ]`; this helper
    /// captures only the outer loop. The per-CTB walk is the
    /// caller's responsibility once `ColBd[ ]` / `RowBd[ ]` /
    /// `CtbAddrRsToTs[ ]` are available.
    ///
    /// **Spec note (deferred):** the explicit-id branch of
    /// eq. (30) reads `tile_id_val[ i ][ j ]`, but the field's
    /// own prose definition (§7.4.3.2) binds the first index to
    /// the row and the second to the column. Until the docs
    /// collaborator clarifies which ordering rules, this
    /// iterator deliberately stops at `(tile_idx, j, i)` and
    /// does **not** surface `tile_id`.
    pub fn tile_grid_coords(&self) -> TileGridCoordIter {
        TileGridCoordIter {
            num_rows: self.num_tile_rows(),
            num_cols: self.num_tile_columns(),
            next_idx: 0,
        }
    }

    /// §6.5.1 eq. (24) — `ColWidth[ i ]` derivation against this PPS.
    ///
    /// Dispatches into [`compute_col_widths`] with this PPS's
    /// [`uniform_tile_spacing_flag`], [`num_tile_columns_minus1`],
    /// and [`tile_column_width_minus1`] slice. The
    /// `PicWidthInCtbsY` input comes from §7.4.3.1
    /// (`Ceil( pic_width_in_luma_samples / ( 1 << CtbLog2SizeY ) )`)
    /// and lives on the SPS side, so it stays an explicit caller
    /// argument.
    ///
    /// Output length is [`num_tile_columns`](Self::num_tile_columns)
    /// (always at least `1`).
    ///
    /// [`uniform_tile_spacing_flag`]: Self::uniform_tile_spacing_flag
    /// [`num_tile_columns_minus1`]: Self::num_tile_columns_minus1
    /// [`tile_column_width_minus1`]: Self::tile_column_width_minus1
    pub fn col_widths(&self, pic_width_in_ctbs_y: u32) -> Vec<u32> {
        compute_col_widths(
            self.uniform_tile_spacing_flag,
            self.num_tile_columns_minus1,
            &self.tile_column_width_minus1,
            pic_width_in_ctbs_y,
        )
    }

    /// §6.5.1 eq. (25) — `RowHeight[ j ]` derivation against this PPS.
    ///
    /// Symmetric to [`Self::col_widths`]. Dispatches into
    /// [`compute_row_heights`] with this PPS's
    /// [`uniform_tile_spacing_flag`], [`num_tile_rows_minus1`], and
    /// [`tile_row_height_minus1`] slice.
    ///
    /// Output length is [`num_tile_rows`](Self::num_tile_rows)
    /// (always at least `1`).
    ///
    /// [`uniform_tile_spacing_flag`]: Self::uniform_tile_spacing_flag
    /// [`num_tile_rows_minus1`]: Self::num_tile_rows_minus1
    /// [`tile_row_height_minus1`]: Self::tile_row_height_minus1
    pub fn row_heights(&self, pic_height_in_ctbs_y: u32) -> Vec<u32> {
        compute_row_heights(
            self.uniform_tile_spacing_flag,
            self.num_tile_rows_minus1,
            &self.tile_row_height_minus1,
            pic_height_in_ctbs_y,
        )
    }
}

/// Iterator returned by [`Pps::tile_grid_coords`]; see that method
/// for the full spec reference and the iteration order.
#[derive(Clone, Debug)]
pub struct TileGridCoordIter {
    num_rows: u32,
    num_cols: u32,
    next_idx: u32,
}

impl Iterator for TileGridCoordIter {
    type Item = TileGridCoord;

    fn next(&mut self) -> Option<TileGridCoord> {
        let total = self.num_rows.checked_mul(self.num_cols)?;
        if self.next_idx >= total {
            return None;
        }
        let tile_idx = self.next_idx;
        let tile_row_j = tile_idx / self.num_cols;
        let tile_col_i = tile_idx % self.num_cols;
        self.next_idx = tile_idx + 1;
        Some(TileGridCoord {
            tile_idx,
            tile_row_j,
            tile_col_i,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = (self.num_rows as usize).saturating_mul(self.num_cols as usize);
        let remaining = total.saturating_sub(self.next_idx as usize);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TileGridCoordIter {}

/// §6.5.1 eq. (24) — `ColWidth[ i ]` derivation.
///
/// Returns the per-tile-column width list in units of CTBs. Output
/// length is `num_tile_columns_minus1 + 1`, and the entries sum to
/// `pic_width_in_ctbs_y` by construction (uniform and explicit
/// branches alike).
///
/// Inputs are taken as primitive parameters so the helper composes
/// cleanly with both [`Pps::col_widths`] (the instance dispatch)
/// and any future §7.4.3.1-driven derivation that needs to compute
/// `ColWidth[ ]` from an alternative source.
///
/// # Spec text (eq. 24)
///
/// ```text
/// if( uniform_tile_spacing_flag )
///     for( i = 0; i <= num_tile_columns_minus1; i++ )
///         ColWidth[ i ] = ( ( i + 1 ) * PicWidthInCtbsY ) /
///                                  ( num_tile_columns_minus1 + 1 )
///                       - (   i       * PicWidthInCtbsY ) /
///                                  ( num_tile_columns_minus1 + 1 )
/// else {
///     ColWidth[ num_tile_columns_minus1 ] = PicWidthInCtbsY
///     for( i = 0; i < num_tile_columns_minus1; i++ ) {
///         ColWidth[ i ] = tile_column_width_minus1[ i ] + 1
///         ColWidth[ num_tile_columns_minus1 ] -= ColWidth[ i ]
///     }
/// }
/// ```
///
/// # Inputs
///
/// * `uniform_tile_spacing_flag` — §7.4.3.2 PPS flag; `true` picks
///   the integer-division branch, `false` the explicit-widths
///   branch.
/// * `num_tile_columns_minus1` — §7.4.3.2 PPS field. The output
///   has `num_tile_columns_minus1 + 1` entries.
/// * `tile_column_width_minus1` — §7.4.3.2 PPS list, ignored when
///   `uniform_tile_spacing_flag` is `true`. Length-`n` slice where
///   `n = num_tile_columns_minus1`; the last column width is the
///   spec's "remainder" assignment in eq. (24).
/// * `pic_width_in_ctbs_y` — `PicWidthInCtbsY` from §7.4.3.1,
///   `Ceil( pic_width_in_luma_samples / ( 1 << CtbLog2SizeY ) )`.
///
/// # Returned vector
///
/// Length `num_tile_columns_minus1 + 1`. Entry `i` is `ColWidth[ i ]`.
/// In the explicit branch with a short / oversized
/// `tile_column_width_minus1` slice, the helper saturates entries
/// past the slice end at `0` and reports the residual at the
/// last column; callers should validate the input lengths against
/// the §7.3.2.2 PPS parser's contract before invoking.
///
/// # Caveats
///
/// The spec writes `ColWidth[ num_tile_columns_minus1 ] -=
/// ColWidth[ i ]` inside the per-`i` loop, so the running residual
/// can underflow if the encoder mis-specifies
/// `tile_column_width_minus1[ ]`. The helper uses
/// [`u32::saturating_sub`] for the residual update so a malformed
/// stream produces a clamped `0` instead of panicking; the caller
/// is responsible for treating the resulting `ColWidth` list as
/// suspect if the explicit widths overflow `PicWidthInCtbsY`.
pub fn compute_col_widths(
    uniform_tile_spacing_flag: bool,
    num_tile_columns_minus1: u32,
    tile_column_width_minus1: &[u32],
    pic_width_in_ctbs_y: u32,
) -> Vec<u32> {
    let n = (num_tile_columns_minus1 as usize) + 1;
    let mut col_width: Vec<u32> = Vec::with_capacity(n);
    if uniform_tile_spacing_flag {
        // Uniform branch: ColWidth[ i ] = floor(((i+1)*W) / n)
        //                              - floor((  i  *W) / n).
        for i in 0..(n as u32) {
            let lo = (i.saturating_mul(pic_width_in_ctbs_y))
                .checked_div(n as u32)
                .unwrap_or(0);
            let hi = (i.saturating_add(1).saturating_mul(pic_width_in_ctbs_y))
                .checked_div(n as u32)
                .unwrap_or(0);
            col_width.push(hi - lo);
        }
    } else {
        // Explicit branch: prime the residual with PicWidthInCtbsY,
        // subtract each explicit width, then store the residual at
        // num_tile_columns_minus1.
        let mut residual: u32 = pic_width_in_ctbs_y;
        for i in 0..num_tile_columns_minus1 as usize {
            let w = tile_column_width_minus1
                .get(i)
                .map(|m| m.saturating_add(1))
                .unwrap_or(0);
            col_width.push(w);
            residual = residual.saturating_sub(w);
        }
        col_width.push(residual);
    }
    col_width
}

/// §6.5.1 eq. (25) — `RowHeight[ j ]` derivation.
///
/// Returns the per-tile-row height list in units of CTBs. Output
/// length is `num_tile_rows_minus1 + 1`, and the entries sum to
/// `pic_height_in_ctbs_y` by construction. Symmetric to
/// [`compute_col_widths`] (eq. 24).
///
/// # Spec text (eq. 25)
///
/// ```text
/// if( uniform_tile_spacing_flag )
///     for( j = 0; j <= num_tile_rows_minus1; j++ )
///         RowHeight[ j ] = ( ( j + 1 ) * PicHeightInCtbsY ) /
///                                    ( num_tile_rows_minus1 + 1 )
///                        - (   j       * PicHeightInCtbsY ) /
///                                    ( num_tile_rows_minus1 + 1 )
/// else {
///     RowHeight[ num_tile_rows_minus1 ] = PicHeightInCtbsY
///     for( j = 0; j < num_tile_rows_minus1; j++ ) {
///         RowHeight[ j ] = tile_row_height_minus1[ j ] + 1
///         RowHeight[ num_tile_rows_minus1 ] -= RowHeight[ j ]
///     }
/// }
/// ```
pub fn compute_row_heights(
    uniform_tile_spacing_flag: bool,
    num_tile_rows_minus1: u32,
    tile_row_height_minus1: &[u32],
    pic_height_in_ctbs_y: u32,
) -> Vec<u32> {
    let n = (num_tile_rows_minus1 as usize) + 1;
    let mut row_height: Vec<u32> = Vec::with_capacity(n);
    if uniform_tile_spacing_flag {
        for j in 0..(n as u32) {
            let lo = (j.saturating_mul(pic_height_in_ctbs_y))
                .checked_div(n as u32)
                .unwrap_or(0);
            let hi = (j.saturating_add(1).saturating_mul(pic_height_in_ctbs_y))
                .checked_div(n as u32)
                .unwrap_or(0);
            row_height.push(hi - lo);
        }
    } else {
        let mut residual: u32 = pic_height_in_ctbs_y;
        for j in 0..num_tile_rows_minus1 as usize {
            let h = tile_row_height_minus1
                .get(j)
                .map(|m| m.saturating_add(1))
                .unwrap_or(0);
            row_height.push(h);
            residual = residual.saturating_sub(h);
        }
        row_height.push(residual);
    }
    row_height
}

pub fn parse(rbsp: &[u8]) -> Result<Pps> {
    let mut br = BitReader::new(rbsp);
    let pps_pic_parameter_set_id = br.ue()?;
    let pps_seq_parameter_set_id = br.ue()?;
    if pps_pic_parameter_set_id > 63 {
        return Err(Error::invalid(format!(
            "evc pps: pps_pic_parameter_set_id {pps_pic_parameter_set_id} > 63"
        )));
    }

    let mut num_ref_idx_default_active_minus1 = [0u32; 2];
    for slot in &mut num_ref_idx_default_active_minus1 {
        *slot = br.ue()?;
    }
    let additional_lt_poc_lsb_len = br.ue()?;
    let rpl1_idx_present_flag = br.u1()? != 0;
    let single_tile_in_pic_flag = br.u1()? != 0;

    let mut num_tile_columns_minus1 = 0;
    let mut num_tile_rows_minus1 = 0;
    let mut uniform_tile_spacing_flag = true;
    let mut tile_column_width_minus1 = Vec::new();
    let mut tile_row_height_minus1 = Vec::new();
    let mut loop_filter_across_tiles_enabled_flag = false;
    let mut tile_offset_len_minus1 = 0;
    if !single_tile_in_pic_flag {
        num_tile_columns_minus1 = br.ue()?;
        num_tile_rows_minus1 = br.ue()?;
        if num_tile_columns_minus1 >= MAX_TILES_PER_DIM || num_tile_rows_minus1 >= MAX_TILES_PER_DIM
        {
            return Err(Error::invalid(format!(
                "evc pps: tile geometry {}x{} exceeds sanity bound",
                num_tile_columns_minus1 + 1,
                num_tile_rows_minus1 + 1
            )));
        }
        uniform_tile_spacing_flag = br.u1()? != 0;
        if !uniform_tile_spacing_flag {
            tile_column_width_minus1.reserve_exact(num_tile_columns_minus1 as usize);
            for _ in 0..num_tile_columns_minus1 {
                tile_column_width_minus1.push(br.ue()?);
            }
            tile_row_height_minus1.reserve_exact(num_tile_rows_minus1 as usize);
            for _ in 0..num_tile_rows_minus1 {
                tile_row_height_minus1.push(br.ue()?);
            }
        }
        loop_filter_across_tiles_enabled_flag = br.u1()? != 0;
        tile_offset_len_minus1 = br.ue()?;
    }

    let tile_id_len_minus1 = br.ue()?;
    if tile_id_len_minus1 > 31 {
        return Err(Error::invalid(format!(
            "evc pps: tile_id_len_minus1 {tile_id_len_minus1} > 31"
        )));
    }
    let explicit_tile_id_flag = br.u1()? != 0;
    let mut tile_id_val = Vec::new();
    if explicit_tile_id_flag {
        let n_rows = (num_tile_rows_minus1 + 1) as usize;
        let n_cols = (num_tile_columns_minus1 + 1) as usize;
        let total = n_rows * n_cols;
        if total > (MAX_TILES_PER_DIM * MAX_TILES_PER_DIM) as usize {
            return Err(Error::invalid("evc pps: explicit tile-id table too large"));
        }
        tile_id_val.reserve_exact(total);
        let bits = tile_id_len_minus1 + 1;
        for _ in 0..total {
            tile_id_val.push(br.u(bits)?);
        }
    }

    let pic_dra_enabled_flag = br.u1()? != 0;
    let mut pic_dra_aps_id = 0;
    if pic_dra_enabled_flag {
        pic_dra_aps_id = br.u(5)? as u8;
    }
    let arbitrary_slice_present_flag = br.u1()? != 0;
    let constrained_intra_pred_flag = br.u1()? != 0;
    let cu_qp_delta_enabled_flag = br.u1()? != 0;
    let mut log2_cu_qp_delta_area_minus6 = 0;
    if cu_qp_delta_enabled_flag {
        log2_cu_qp_delta_area_minus6 = br.ue()?;
    }

    Ok(Pps {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        num_ref_idx_default_active_minus1,
        additional_lt_poc_lsb_len,
        rpl1_idx_present_flag,
        single_tile_in_pic_flag,
        num_tile_columns_minus1,
        num_tile_rows_minus1,
        uniform_tile_spacing_flag,
        tile_column_width_minus1,
        tile_row_height_minus1,
        loop_filter_across_tiles_enabled_flag,
        tile_offset_len_minus1,
        tile_id_len_minus1,
        explicit_tile_id_flag,
        tile_id_val,
        pic_dra_enabled_flag,
        pic_dra_aps_id,
        arbitrary_slice_present_flag,
        constrained_intra_pred_flag,
        cu_qp_delta_enabled_flag,
        log2_cu_qp_delta_area_minus6,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    fn emit_minimal_pps() -> Vec<u8> {
        let mut e = BitEmitter::new();
        e.ue(0); // pps_pic_parameter_set_id
        e.ue(0); // pps_seq_parameter_set_id
        e.ue(0); // num_ref_idx_default_active_minus1[0]
        e.ue(0); // num_ref_idx_default_active_minus1[1]
        e.ue(0); // additional_lt_poc_lsb_len
        e.u(1, 0); // rpl1_idx_present_flag
        e.u(1, 1); // single_tile_in_pic_flag = 1
        e.ue(0); // tile_id_len_minus1
        e.u(1, 0); // explicit_tile_id_flag
        e.u(1, 0); // pic_dra_enabled_flag
        e.u(1, 0); // arbitrary_slice_present_flag
        e.u(1, 0); // constrained_intra_pred_flag
        e.u(1, 1); // cu_qp_delta_enabled_flag = 1
        e.ue(0); // log2_cu_qp_delta_area_minus6
        e.finish_with_trailing_bits();
        e.into_bytes()
    }

    #[test]
    fn parse_minimal_pps() {
        let rbsp = emit_minimal_pps();
        let pps = parse(&rbsp).unwrap();
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert!(pps.single_tile_in_pic_flag);
        assert!(pps.cu_qp_delta_enabled_flag);
        assert_eq!(pps.log2_cu_qp_delta_area_minus6, 0);
        assert!(!pps.constrained_intra_pred_flag);
    }

    #[test]
    fn parse_pps_with_two_tile_columns() {
        let mut e = BitEmitter::new();
        e.ue(1); // pps id
        e.ue(0);
        e.ue(0);
        e.ue(0);
        e.ue(0);
        e.u(1, 0); // rpl1_idx_present_flag
        e.u(1, 0); // single_tile_in_pic_flag = 0
        e.ue(1); // num_tile_columns_minus1
        e.ue(0); // num_tile_rows_minus1
        e.u(1, 1); // uniform_tile_spacing_flag = 1
        e.u(1, 1); // loop_filter_across_tiles_enabled_flag
        e.ue(2); // tile_offset_len_minus1
        e.ue(3); // tile_id_len_minus1 (4 bits / id)
        e.u(1, 0); // explicit_tile_id_flag
        e.u(1, 0); // pic_dra_enabled_flag
        e.u(1, 0); // arbitrary_slice_present_flag
        e.u(1, 0); // constrained_intra_pred_flag
        e.u(1, 0); // cu_qp_delta_enabled_flag
        e.finish_with_trailing_bits();
        let rbsp = e.into_bytes();
        let pps = parse(&rbsp).unwrap();
        assert_eq!(pps.num_tile_columns_minus1, 1);
        assert_eq!(pps.num_tile_rows_minus1, 0);
        assert!(pps.uniform_tile_spacing_flag);
        assert!(pps.loop_filter_across_tiles_enabled_flag);
        assert_eq!(pps.tile_id_len_minus1, 3);
    }

    #[test]
    fn rejects_huge_tile_geometry() {
        let mut e = BitEmitter::new();
        e.ue(0);
        e.ue(0);
        e.ue(0);
        e.ue(0);
        e.ue(0);
        e.u(1, 0);
        e.u(1, 0); // single_tile_in_pic_flag = 0
        e.ue(MAX_TILES_PER_DIM); // num_tile_columns_minus1 over the bound
        e.ue(0);
        e.finish_with_trailing_bits();
        let err = parse(&e.into_bytes()).unwrap_err();
        assert!(format!("{err}").contains("sanity bound"));
    }

    fn pps_with_grid(rows_minus1: u32, cols_minus1: u32) -> Pps {
        Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            num_ref_idx_default_active_minus1: [0, 0],
            additional_lt_poc_lsb_len: 0,
            rpl1_idx_present_flag: false,
            single_tile_in_pic_flag: rows_minus1 == 0 && cols_minus1 == 0,
            num_tile_columns_minus1: cols_minus1,
            num_tile_rows_minus1: rows_minus1,
            uniform_tile_spacing_flag: true,
            tile_column_width_minus1: Vec::new(),
            tile_row_height_minus1: Vec::new(),
            loop_filter_across_tiles_enabled_flag: false,
            tile_offset_len_minus1: 0,
            tile_id_len_minus1: 0,
            explicit_tile_id_flag: false,
            tile_id_val: Vec::new(),
            pic_dra_enabled_flag: false,
            pic_dra_aps_id: 0,
            arbitrary_slice_present_flag: false,
            constrained_intra_pred_flag: false,
            cu_qp_delta_enabled_flag: false,
            log2_cu_qp_delta_area_minus6: 0,
        }
    }

    #[test]
    fn round237_num_tiles_single_tile_picture() {
        // §7.4.3.2: single_tile_in_pic_flag = 1 leaves both
        // num_tile_columns_minus1 and num_tile_rows_minus1 at their
        // inferred 0. NumTilesInPic is therefore 1.
        let pps = pps_with_grid(0, 0);
        assert_eq!(pps.num_tile_rows(), 1);
        assert_eq!(pps.num_tile_columns(), 1);
        assert_eq!(pps.num_tiles_in_pic(), 1);
    }

    #[test]
    fn round237_num_tiles_two_by_one_grid() {
        // num_tile_columns_minus1 = 1, num_tile_rows_minus1 = 0
        // → 2 columns × 1 row = 2 tiles.
        let pps = pps_with_grid(0, 1);
        assert_eq!(pps.num_tile_columns(), 2);
        assert_eq!(pps.num_tile_rows(), 1);
        assert_eq!(pps.num_tiles_in_pic(), 2);
    }

    #[test]
    fn round237_num_tiles_three_by_two_grid() {
        // 3 columns × 2 rows = 6 tiles.
        let pps = pps_with_grid(1, 2);
        assert_eq!(pps.num_tile_columns(), 3);
        assert_eq!(pps.num_tile_rows(), 2);
        assert_eq!(pps.num_tiles_in_pic(), 6);
    }

    #[test]
    fn round237_tile_grid_coords_single_tile() {
        // Single-tile picture: one iteration yielding (0, 0, 0).
        let pps = pps_with_grid(0, 0);
        let coords: Vec<_> = pps.tile_grid_coords().collect();
        assert_eq!(coords.len(), 1);
        assert_eq!(
            coords[0],
            TileGridCoord {
                tile_idx: 0,
                tile_row_j: 0,
                tile_col_i: 0,
            }
        );
    }

    #[test]
    fn round237_tile_grid_coords_two_by_one_order() {
        // 2 columns × 1 row: tile_idx advances over the inner
        // (column) loop. j stays 0; i counts 0, 1.
        let pps = pps_with_grid(0, 1);
        let coords: Vec<_> = pps.tile_grid_coords().collect();
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0].tile_idx, 0);
        assert_eq!(coords[0].tile_row_j, 0);
        assert_eq!(coords[0].tile_col_i, 0);
        assert_eq!(coords[1].tile_idx, 1);
        assert_eq!(coords[1].tile_row_j, 0);
        assert_eq!(coords[1].tile_col_i, 1);
    }

    #[test]
    fn round237_tile_grid_coords_two_by_two_raster_order() {
        // 2 columns × 2 rows. eq. (30) outer-loop order is
        // (j=0,i=0), (j=0,i=1), (j=1,i=0), (j=1,i=1), and
        // tile_idx counts linearly across the joined loop.
        let pps = pps_with_grid(1, 1);
        let coords: Vec<_> = pps.tile_grid_coords().collect();
        assert_eq!(coords.len(), 4);
        let expected = [(0u32, 0u32, 0u32), (1, 0, 1), (2, 1, 0), (3, 1, 1)];
        for (got, want) in coords.iter().zip(expected.iter()) {
            assert_eq!(got.tile_idx, want.0);
            assert_eq!(got.tile_row_j, want.1);
            assert_eq!(got.tile_col_i, want.2);
        }
    }

    #[test]
    fn round237_tile_grid_coords_three_by_two_full_walk() {
        // 3 columns × 2 rows → 6 tiles. Row-major order from
        // eq. (30): (j,i) cycles (0,0),(0,1),(0,2),(1,0),(1,1),(1,2).
        let pps = pps_with_grid(1, 2);
        let coords: Vec<_> = pps.tile_grid_coords().collect();
        assert_eq!(coords.len(), 6);
        let mut seen = Vec::new();
        for c in &coords {
            seen.push((c.tile_row_j, c.tile_col_i));
        }
        assert_eq!(seen, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
        for (k, c) in coords.iter().enumerate() {
            assert_eq!(c.tile_idx, k as u32);
        }
    }

    #[test]
    fn round237_tile_grid_iterator_exhausts_to_none() {
        // After every tile is yielded, .next() must return None
        // and stay None on a second call.
        let pps = pps_with_grid(0, 1);
        let mut iter = pps.tile_grid_coords();
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn round237_tile_grid_iterator_size_hint() {
        // size_hint must shrink with consumption and stay tight
        // for ExactSizeIterator correctness.
        let pps = pps_with_grid(1, 2); // 6 tiles
        let mut iter = pps.tile_grid_coords();
        assert_eq!(iter.size_hint(), (6, Some(6)));
        assert_eq!(iter.len(), 6);
        let _ = iter.next();
        let _ = iter.next();
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.len(), 4);
        for _ in 0..4 {
            let _ = iter.next();
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.len(), 0);
    }

    fn pps_with_explicit_cols(cols_minus1: u32, widths_minus1: &[u32]) -> Pps {
        // Helper: build a PPS with uniform_tile_spacing_flag = false
        // and the §7.4.3.2 tile_column_width_minus1 list populated.
        Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            num_ref_idx_default_active_minus1: [0, 0],
            additional_lt_poc_lsb_len: 0,
            rpl1_idx_present_flag: false,
            single_tile_in_pic_flag: false,
            num_tile_columns_minus1: cols_minus1,
            num_tile_rows_minus1: 0,
            uniform_tile_spacing_flag: false,
            tile_column_width_minus1: widths_minus1.to_vec(),
            tile_row_height_minus1: Vec::new(),
            loop_filter_across_tiles_enabled_flag: false,
            tile_offset_len_minus1: 0,
            tile_id_len_minus1: 0,
            explicit_tile_id_flag: false,
            tile_id_val: Vec::new(),
            pic_dra_enabled_flag: false,
            pic_dra_aps_id: 0,
            arbitrary_slice_present_flag: false,
            constrained_intra_pred_flag: false,
            cu_qp_delta_enabled_flag: false,
            log2_cu_qp_delta_area_minus6: 0,
        }
    }

    fn pps_with_explicit_rows(rows_minus1: u32, heights_minus1: &[u32]) -> Pps {
        // Helper: §6.5.1 eq. (25) explicit-row variant of the
        // above. Symmetric.
        Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            num_ref_idx_default_active_minus1: [0, 0],
            additional_lt_poc_lsb_len: 0,
            rpl1_idx_present_flag: false,
            single_tile_in_pic_flag: false,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: rows_minus1,
            uniform_tile_spacing_flag: false,
            tile_column_width_minus1: Vec::new(),
            tile_row_height_minus1: heights_minus1.to_vec(),
            loop_filter_across_tiles_enabled_flag: false,
            tile_offset_len_minus1: 0,
            tile_id_len_minus1: 0,
            explicit_tile_id_flag: false,
            tile_id_val: Vec::new(),
            pic_dra_enabled_flag: false,
            pic_dra_aps_id: 0,
            arbitrary_slice_present_flag: false,
            constrained_intra_pred_flag: false,
            cu_qp_delta_enabled_flag: false,
            log2_cu_qp_delta_area_minus6: 0,
        }
    }

    #[test]
    fn round249_col_widths_single_tile_returns_full_picture() {
        // §6.5.1 eq. (24) uniform branch with n = 1: single tile
        // spans the entire PicWidthInCtbsY (here, 10).
        let pps = pps_with_grid(0, 0);
        let widths = pps.col_widths(10);
        assert_eq!(widths.as_slice(), &[10]);
    }

    #[test]
    fn round249_col_widths_uniform_two_columns_even_split() {
        // §6.5.1 eq. (24) uniform branch, n = 2, PicWidthInCtbsY = 10.
        //   ColWidth[ 0 ] = ((0+1)*10)/2 - (0*10)/2 = 5 - 0 = 5
        //   ColWidth[ 1 ] = ((1+1)*10)/2 - (1*10)/2 = 10 - 5 = 5
        let pps = pps_with_grid(0, 1);
        let widths = pps.col_widths(10);
        assert_eq!(widths.as_slice(), &[5, 5]);
    }

    #[test]
    fn round249_col_widths_uniform_three_columns_floor_division() {
        // §6.5.1 eq. (24) uniform branch, n = 3, PicWidthInCtbsY = 10.
        //   ColWidth[ 0 ] = ( 1*10)/3 - 0   = 3 - 0 = 3
        //   ColWidth[ 1 ] = ( 2*10)/3 - 3   = 6 - 3 = 3
        //   ColWidth[ 2 ] = ( 3*10)/3 - 6   = 10 - 6 = 4
        // Sum is 10. The closed form is structurally exact-cover.
        let pps = pps_with_grid(0, 2);
        let widths = pps.col_widths(10);
        assert_eq!(widths.as_slice(), &[3, 3, 4]);
        assert_eq!(widths.iter().sum::<u32>(), 10);
    }

    #[test]
    fn round249_col_widths_uniform_covers_pic_width_exactly() {
        // §6.5.1 eq. (24) uniform-branch invariant: sum of
        // ColWidth[] over the column count is PicWidthInCtbsY for
        // every (n, W) the spec accepts. Sweep a representative
        // grid.
        for cols_minus1 in 0u32..=8 {
            for w in [1u32, 2, 5, 10, 17, 32, 64, 100] {
                let pps = pps_with_grid(0, cols_minus1);
                let widths = pps.col_widths(w);
                assert_eq!(widths.len(), (cols_minus1 + 1) as usize);
                assert_eq!(widths.iter().sum::<u32>(), w);
            }
        }
    }

    #[test]
    fn round249_col_widths_explicit_branch_pins_eq24_remainder() {
        // §6.5.1 eq. (24) explicit branch: ColWidth[i] =
        //     tile_column_width_minus1[i] + 1 for i < n-1, and
        // ColWidth[n-1] = PicWidthInCtbsY - sum-of-others.
        //
        // n = 3, tile_column_width_minus1 = [2, 0],
        // PicWidthInCtbsY = 10. Walk: ColWidth[0] = 3, ColWidth[1]
        // = 1, residual starts at 10, → 7 → 6, so ColWidth[2] = 6.
        let pps = pps_with_explicit_cols(2, &[2, 0]);
        let widths = pps.col_widths(10);
        assert_eq!(widths.as_slice(), &[3, 1, 6]);
        assert_eq!(widths.iter().sum::<u32>(), 10);
    }

    #[test]
    fn round249_col_widths_explicit_branch_two_cols_residual() {
        // §6.5.1 eq. (24) explicit branch, n = 2,
        // tile_column_width_minus1 = [3], PicWidthInCtbsY = 10.
        //   ColWidth[ 0 ] = 4
        //   ColWidth[ 1 ] = 10 − 4 = 6
        let pps = pps_with_explicit_cols(1, &[3]);
        let widths = pps.col_widths(10);
        assert_eq!(widths.as_slice(), &[4, 6]);
    }

    #[test]
    fn round249_col_widths_explicit_overflow_saturates_residual() {
        // §6.5.1 eq. (24) explicit branch with malformed input:
        // explicit widths sum past PicWidthInCtbsY. The spec's
        // running subtraction would underflow on the residual;
        // we clamp at 0 via saturating_sub so the function never
        // panics on a non-conforming stream.
        //   n = 2, tile_column_width_minus1 = [99], PicWidthInCtbsY = 5
        //   ColWidth[ 0 ] = 100, residual = 5.saturating_sub(100) = 0
        //   ColWidth[ 1 ] = 0
        let pps = pps_with_explicit_cols(1, &[99]);
        let widths = pps.col_widths(5);
        assert_eq!(widths.as_slice(), &[100, 0]);
    }

    #[test]
    fn round249_row_heights_uniform_two_rows_even_split() {
        // §6.5.1 eq. (25) uniform branch, n = 2, PicHeightInCtbsY = 8.
        //   RowHeight[ 0 ] = ((0+1)*8)/2 - 0 = 4
        //   RowHeight[ 1 ] = ((1+1)*8)/2 - 4 = 4
        let pps = pps_with_grid(1, 0);
        let heights = pps.row_heights(8);
        assert_eq!(heights.as_slice(), &[4, 4]);
    }

    #[test]
    fn round249_row_heights_uniform_covers_pic_height_exactly() {
        // §6.5.1 eq. (25) uniform-branch invariant mirror of the
        // col_widths sweep. Sum over RowHeight[] is
        // PicHeightInCtbsY for every (n, H) the spec accepts.
        for rows_minus1 in 0u32..=8 {
            for h in [1u32, 2, 5, 10, 17, 32, 64, 100] {
                let pps = pps_with_grid(rows_minus1, 0);
                let heights = pps.row_heights(h);
                assert_eq!(heights.len(), (rows_minus1 + 1) as usize);
                assert_eq!(heights.iter().sum::<u32>(), h);
            }
        }
    }

    #[test]
    fn round249_row_heights_explicit_branch_pins_eq25_remainder() {
        // §6.5.1 eq. (25) explicit branch: symmetric to eq. (24).
        // n = 3, tile_row_height_minus1 = [0, 4], PicHeightInCtbsY = 10.
        //   RowHeight[ 0 ] = 1, residual = 10 → 9
        //   RowHeight[ 1 ] = 5, residual = 9 → 4
        //   RowHeight[ 2 ] = 4
        let pps = pps_with_explicit_rows(2, &[0, 4]);
        let heights = pps.row_heights(10);
        assert_eq!(heights.as_slice(), &[1, 5, 4]);
        assert_eq!(heights.iter().sum::<u32>(), 10);
    }

    #[test]
    fn round249_compute_col_widths_uniform_matches_pps_dispatch() {
        // The free function and the Pps::col_widths instance method
        // must agree on every input — they are the same derivation.
        for cols_minus1 in 0u32..=4 {
            for w in [1u32, 7, 10, 33] {
                let pps = pps_with_grid(0, cols_minus1);
                let direct = compute_col_widths(true, cols_minus1, &[], w);
                let dispatched = pps.col_widths(w);
                assert_eq!(direct, dispatched);
            }
        }
    }

    #[test]
    fn round249_compute_row_heights_uniform_matches_pps_dispatch() {
        // Mirror of the col_widths uniform-agreement sweep.
        for rows_minus1 in 0u32..=4 {
            for h in [1u32, 7, 10, 33] {
                let pps = pps_with_grid(rows_minus1, 0);
                let direct = compute_row_heights(true, rows_minus1, &[], h);
                let dispatched = pps.row_heights(h);
                assert_eq!(direct, dispatched);
            }
        }
    }

    #[test]
    fn round249_col_widths_zero_pic_width_returns_all_zeros() {
        // §6.5.1 eq. (24) with PicWidthInCtbsY = 0 produces a
        // valid all-zero list of length n. The spec never accepts
        // a zero-CTB-width picture, but the helper's behaviour
        // matters for defensive callers.
        let pps = pps_with_grid(0, 2);
        let widths = pps.col_widths(0);
        assert_eq!(widths.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn round237_tile_idx_matches_row_col_packing() {
        // The §6.5.1 eq. (30) outer loop is equivalent to
        //    tile_idx = j * num_cols + i
        // for j ∈ [0, num_rows) and i ∈ [0, num_cols). Round-trip
        // every coord through the packing identity.
        let pps = pps_with_grid(2, 3); // 3 rows × 4 cols = 12
        let num_cols = pps.num_tile_columns();
        for c in pps.tile_grid_coords() {
            assert_eq!(c.tile_idx, c.tile_row_j * num_cols + c.tile_col_i);
        }
    }
}
