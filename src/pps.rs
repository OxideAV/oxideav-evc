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
}
