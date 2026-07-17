//! Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1).
//!
//! Round-10 status: a working **Baseline-profile IDR + P + B** decoder
//! with residual coding (RLE + dequant + IDCT for nTbS up to 64),
//! deblocking (§8.8.2 luma + chroma path), Main-profile CABAC init
//! tables (Tables 40-90, §9.3.4.2 ctxInc helpers, eq. 1425/1426),
//! full **reference-picture-list parsing** (§7.3.7 / §7.4.8) for
//! non-IDR slices (`sps_rpl_flag = 1`), the **HMVP candidate list**
//! (§8.5.2.7 / §8.5.2.4.4) wired through the inter pipeline, the
//! round-9 **multi-ref DPB + POC reordering**, and the round-10
//! **spatial-neighbour MV grid AMVP** (§8.5.2.4) +
//! **LTRP RPL → DPB resolution** (§8.3.2 / §8.3.5) +
//! **`flush()` drain** of the output queue.
//!
//! The crate decomposes into:
//!
//! * [`bitreader`] — MSB-first bit reader (§9.2 helpers).
//! * [`nal`] — 2-byte NAL header (§7.3.1.2) + length-prefixed framing.
//! * [`sps`] / [`pps`] / [`aps`] — parameter-set parsers (§7.3.2.x).
//! * [`slice_header`] — `slice_header()` parse (§7.3.4).
//! * [`cabac`] — full CABAC parsing process (§9.3): arithmetic decoding
//!   engine (regular + bypass + terminate) plus the FL / U / TR / EGk
//!   binarization helpers. The Baseline `sps_cm_init_flag == 0` path uses
//!   a single ctxTable=0 / ctxIdx=0 context.
//! * [`cabac_init`] — Main-profile (`sps_cm_init_flag == 1`) initValue
//!   tables (Tables 40-90 of §9.3.5) + the §9.3.2.2 init pipeline
//!   ([`cabac_init::init_main_profile_contexts`]) + the §9.3.4.2 per-
//!   syntax-element ctxInc helpers (`btt_split_flag`,
//!   `last_sig_coeff_x/y_prefix`, `sig_coeff_flag`,
//!   `coeff_abs_level_greaterA/B_flag`, etc.).
//! * [`slice_data`] — `slice_data()` walker plus the round-3 IDR pixel
//!   pipeline ([`slice_data::decode_baseline_idr_slice`]) **and** the
//!   round-4 inter pipeline ([`slice_data::decode_baseline_inter_slice`]).
//! * [`intra`] — intra prediction (§8.4.4) for the Baseline 5-mode set
//!   (DC, HOR, VER, UL, UR; `sps_eipd_flag == 0` path).
//! * [`inter`] — round-4 inter prediction (§8.5): MV resolution + 8-tap
//!   luma + 4-tap chroma sub-pel interpolation (Tables 25 / 27 — Baseline
//!   subset only) + AMVP candidate construction + default-weighted bipred.
//! * [`transform`] — inverse DCT-II for nTbS ∈ {2, 4, 8, 16, 32, 64}
//!   (eq. 1062-1076). The 64-point matrix is built from the closed-form
//!   `M[m][n] = round(64·√2·cos(π·m·(2n+1)/128))` (m≥1, M[0][n]=64),
//!   verified against every printed entry of eq. 1072 / 1074.
//! * [`dequant`] — scaling + transform + final renorm (§8.7.2 / §8.7.3 /
//!   §8.7.4) for the `sps_iqt_flag == 0` Baseline path.
//! * [`picture`] — yuv420p 8-bit picture buffer + per-CU intra
//!   reconstruct glue (§8.7.5).
//! * [`decoder`] — registered decoder factory returning a working
//!   `Decoder` for Baseline IDR + P/B bitstreams (8-bit 4:2:0, no
//!   residuals, single reference).
//! * [`rpl`] — round-8 `ref_pic_list_struct()` parser (§7.3.7 /
//!   §7.4.8). Handles STRP + LTRP entries with the per-entry
//!   `delta_poc_st` / `strp_entry_sign_flag` / `poc_lsb_lt` shape.
//! * [`hmvp`] — round-8 history-based MV prediction (§8.5.2.7 LRU
//!   update + §8.5.2.4.4 derive_default_mv walk). 23-entry list with
//!   per-CTU-row reset.
//!
//! Round-8 deliberate omissions (pending follow-up rounds):
//!
//! * 10-bit support,
//! * advanced deblocking (`sps_addb_flag = 1` — round-6 supports the
//!   `sps_addb_flag = 0` baseline filter only, now for both luma and
//!   chroma; addb is a Main-profile feature),
//! * multi-reference DPB + reference list reordering (round 8 parses
//!   the RPL and the slice-side `num_ref_idx_active_minus1[]`, but the
//!   inter pipeline still keys off the previous picture only),
//! * **Main-profile decode** — round 7 lands the CABAC infrastructure
//!   (Tables 40-90 + ctxInc helpers); round 8 lands the RPL parse path
//!   and the HMVP candidate list. The actual Main-profile syntax
//!   decode (BTT / SUCO / ADMVP / EIPD / IBC / ATS / ADCC / ALF / DRA
//!   / AMVR / MMVD / affine / DMVR) still bubbles up
//!   `Error::Unsupported`.
//!
//! All section / clause numbers refer to **ISO/IEC 23094-1:2020(E)** at
//! `docs/video/evc/ISO_IEC_23094-1-EVC-2020.pdf`. Every module is
//! spec-only — clauses, equations, and table numbers cite the
//! Recommendation directly.

// Internal syntax/recon plumbing — kept `pub` so tests and fixture tooling
// can drive each stage directly, but `#[doc(hidden)]` so the documented API
// (and cargo-semver-checks) tracks only the stable surface: `probe` +
// `EvcFileInfo` + `CODEC_ID_STR`, `register`, and `decoder::make_decoder`.
#[doc(hidden)]
pub mod adcc;
#[doc(hidden)]
pub mod affine;
#[doc(hidden)]
pub mod affine_cand;
#[doc(hidden)]
pub mod affine_syntax;
#[doc(hidden)]
pub mod alf;
#[doc(hidden)]
pub mod alf_tables;
#[doc(hidden)]
pub mod amvr_syntax;
#[doc(hidden)]
pub mod aps;
#[doc(hidden)]
pub mod ats;
#[doc(hidden)]
pub mod bitreader;
#[doc(hidden)]
pub mod cabac;
#[doc(hidden)]
pub mod cabac_init;
#[doc(hidden)]
pub mod deblock;
pub mod decoder;
#[doc(hidden)]
pub mod dequant;
#[doc(hidden)]
pub mod dmvr;
#[doc(hidden)]
pub mod dra;
#[doc(hidden)]
pub mod eipd;
#[doc(hidden)]
pub mod eipd_mode;
#[doc(hidden)]
pub mod eipd_ref;
#[doc(hidden)]
pub mod eipd_syntax;
#[doc(hidden)]
pub mod hmvp;
#[doc(hidden)]
pub mod htdf;
#[doc(hidden)]
pub mod ibc;
#[doc(hidden)]
pub mod inter;
#[doc(hidden)]
pub mod inter_cu_syntax;
#[doc(hidden)]
pub mod intra;
#[doc(hidden)]
pub mod merge;
#[doc(hidden)]
pub mod mmvd;
#[doc(hidden)]
pub mod mmvd_syntax;
#[doc(hidden)]
pub mod nal;
#[doc(hidden)]
pub mod neighbour;
#[doc(hidden)]
pub mod picture;
#[doc(hidden)]
pub mod pps;
#[doc(hidden)]
pub mod rpl;
#[doc(hidden)]
pub mod scan;
#[doc(hidden)]
pub mod slice_data;
#[doc(hidden)]
pub mod slice_header;
#[doc(hidden)]
pub mod split;
#[doc(hidden)]
pub mod sps;
#[doc(hidden)]
pub mod tiles;
#[doc(hidden)]
pub mod tmvp;
#[doc(hidden)]
pub mod transform;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry};

/// Public codec id string. Matches the aggregator feature name `evc`.
pub const CODEC_ID_STR: &str = "evc";

/// Summary info recoverable from a bare EVC SPS — the public return type
/// of [`probe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvcFileInfo {
    pub width: u32,
    pub height: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub chroma_format_idc: u32,
}

/// Probe a buffer for an EVC bitstream and return summary info from the
/// first parseable SPS. Accepts either Annex B raw-bitstream framing
/// (`u(32)` length prefix, the canonical case per ISO/IEC 23094-1 Annex B)
/// or the tolerant `0x000001` / `0x00000001` start-code scanner.
///
/// Returns `None` when no SPS is found; bubbles up `Some(_)` for the
/// first SPS that parses cleanly.
pub fn probe(input: &[u8]) -> Option<EvcFileInfo> {
    // Try length-prefixed first — that's what Annex B specifies.
    if let Ok(nals) = nal::iter_length_prefixed(input) {
        for nal_ref in nals {
            if let Some(info) = info_from_nal(&nal_ref) {
                return Some(info);
            }
        }
    }
    // Fall back to the tolerant Annex-B-style scanner for ad-hoc files.
    for nal_ref in nal::iter_annex_b(input) {
        if let Some(info) = info_from_nal(&nal_ref) {
            return Some(info);
        }
    }
    None
}

fn info_from_nal(nal_ref: &nal::NalRef<'_>) -> Option<EvcFileInfo> {
    if nal_ref.header.nal_unit_type != nal::NalUnitType::Sps {
        return None;
    }
    let sps = sps::parse(nal_ref.rbsp()).ok()?;
    Some(EvcFileInfo {
        width: sps.pic_width_in_luma_samples,
        height: sps.pic_height_in_luma_samples,
        profile_idc: sps.profile_idc,
        level_idc: sps.level_idc,
        bit_depth_luma: sps.bit_depth_y(),
        bit_depth_chroma: sps.bit_depth_c(),
        chroma_format_idc: sps.chroma_format_idc,
    })
}

/// Walk an IDR slice's `slice_data()` end-to-end given the active SPS and
/// PPS. The slice's RBSP is split into the slice header + the
/// (byte-aligned) slice-data payload; this helper invokes the slice
/// header parser then drives [`slice_data::walk_baseline_idr_slice`]
/// across the rest of the RBSP. Returns the [`slice_data::SliceWalkStats`]
/// once the engine terminates cleanly.
///
/// **Round-2 scope**: Baseline-profile IDR slices only. Errors out on any
/// SPS toolset combination not yet supported by the walker (see
/// [`slice_data::walk_baseline_idr_slice`] for the constraint set).
// Internal round-2 harness over hidden parser types — not a stable API.
#[doc(hidden)]
pub fn walk_idr_slice(
    sps: &sps::Sps,
    pps: &pps::Pps,
    slice_nal_rbsp: &[u8],
) -> oxideav_core::Result<slice_data::SliceWalkStats> {
    use oxideav_core::Error;
    // Build the slice-parse context from SPS + PPS.
    let ctx = slice_header::SliceParseContext {
        single_tile_in_pic_flag: pps.single_tile_in_pic_flag,
        arbitrary_slice_present_flag: pps.arbitrary_slice_present_flag,
        tile_id_len_minus1: pps.tile_id_len_minus1,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        sps_pocs_flag: sps.sps_pocs_flag,
        sps_rpl_flag: sps.sps_rpl_flag,
        sps_alf_flag: sps.sps_alf_flag,
        sps_mmvd_flag: sps.sps_mmvd_flag,
        sps_admvp_flag: sps.sps_admvp_flag,
        sps_addb_flag: sps.sps_addb_flag,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        chroma_array_type: sps.chroma_array_type(),
        num_ref_pic_lists_in_sps_l0: sps.num_ref_pic_lists_in_sps_l0,
        num_ref_pic_lists_in_sps_l1: sps.num_ref_pic_lists_in_sps_l1,
        rpl1_idx_present_flag: pps.rpl1_idx_present_flag,
        long_term_ref_pics_flag: sps.long_term_ref_pics_flag,
        additional_lt_poc_lsb_len: pps.additional_lt_poc_lsb_len,
    };
    // Round-2 walker requires the Baseline profile constraint set (Annex
    // A.3.2). Refuse anything else cleanly.
    // sps_alf_flag and sps_dra_flag are no longer gated here (round-11
    // handles them as post-filter passes).
    if sps.sps_btt_flag
        || sps.sps_suco_flag
        || sps.sps_admvp_flag
        || sps.sps_eipd_flag
        || sps.sps_addb_flag
        || sps.sps_dquant_flag
        || sps.sps_ats_flag
        || sps.sps_adcc_flag
        || sps.sps_cm_init_flag
        || sps.sps_amvr_flag
        || sps.sps_mmvd_flag
        || sps.sps_affine_flag
        || sps.sps_dmvr_flag
        || sps.sps_hmvp_flag
    {
        return Err(Error::unsupported(
            "evc walk_idr_slice: round-2 only supports Baseline-profile toolset",
        ));
    }
    if !pps.single_tile_in_pic_flag {
        return Err(Error::unsupported(
            "evc walk_idr_slice: round-2 requires single_tile_in_pic_flag == 1",
        ));
    }
    // Parse the slice header to find where slice_data() begins. We use a
    // bit-counting BitReader to determine how many bits the header
    // consumed.
    let mut hdr_br = crate::bitreader::BitReader::new(slice_nal_rbsp);
    parse_slice_header_consume(&mut hdr_br, nal::NalUnitType::Idr, &ctx)?;
    // Align to the next byte boundary (slice_data() starts byte-aligned).
    hdr_br.align_to_byte();
    let consumed_bits = hdr_br.bit_position();
    if consumed_bits % 8 != 0 {
        return Err(Error::invalid(
            "evc walk_idr_slice: slice header not byte-aligned after parse",
        ));
    }
    let consumed_bytes = (consumed_bits / 8) as usize;
    if consumed_bytes >= slice_nal_rbsp.len() {
        return Err(Error::invalid(
            "evc walk_idr_slice: no slice_data bytes after header",
        ));
    }
    let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
    // Build SliceWalkInputs from SPS / PPS. Round 90 surfaces the IBC
    // SPS gates so the coding_unit() walker can apply §7.4.5
    // `isIbcAllowed` per-CU.
    let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
    let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
    let max_tb_log2_size_y = 6; // eq. 51: MaxTbLog2SizeY = 6 (a constant)
    let log2_max_ibc_cand_size = sps.log2_max_ibc_cand_size().unwrap_or(0);
    let inputs = slice_data::SliceWalkInputs {
        pic_width: sps.pic_width_in_luma_samples,
        pic_height: sps.pic_height_in_luma_samples,
        ctb_log2_size_y,
        min_cb_log2_size_y,
        max_tb_log2_size_y,
        chroma_format_idc: sps.chroma_format_idc,
        cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
        sps_adcc_flag: sps.sps_adcc_flag,
        sps_eipd_flag: sps.sps_eipd_flag,
        sps_dquant_flag: sps.sps_dquant_flag,
        cu_qp_delta_area: pps.log2_cu_qp_delta_area_minus6 + 6,
        sps_ibc_flag: sps.sps_ibc_flag,
        log2_max_ibc_cand_size,
        // This entry point uses a minimal header parse that doesn't yet
        // surface the §7.3.4 ALF map fields; default them off (no
        // per-CTU ALF map signalled) so `coding_tree_unit()` reads no
        // `alf_ctb_*` bins. `decode_non_idr` threads the real values.
        slice_alf_enabled_flag: false,
        slice_alf_map_flag: false,
        slice_chroma_alf_enabled_flag: false,
        slice_alf_chroma_map_flag: false,
        slice_chroma2_alf_enabled_flag: false,
        slice_alf_chroma2_map_flag: false,
        // This stats-only walker keeps the Baseline gate above
        // (sps_btt_flag == 0), so the default (BTT/SUCO-off) tree gates
        // are exact.
        tree_gates: slice_data::CodingTreeGates::default(),
    };
    slice_data::walk_baseline_idr_slice(slice_data_bytes, inputs)
}

/// **Round-3** end-to-end decode of a Baseline-profile IDR slice.
///
/// Mirrors [`walk_idr_slice`] but invokes the pixel-emission pipeline:
/// returns a freshly-reconstructed [`picture::YuvPicture`] populated by
/// per-CU intra prediction and the spec's picture-construction step
/// (§8.7.5). Round-3 fixtures must use `cbf_luma == cbf_cb == cbf_cr ==
/// 0` for every CU; non-zero CBFs trigger `Error::Unsupported` (round-4
/// scope).
// Internal round-3 harness over hidden parser types — not a stable API.
#[doc(hidden)]
pub fn decode_idr_slice(
    sps: &sps::Sps,
    pps: &pps::Pps,
    slice_nal_rbsp: &[u8],
) -> oxideav_core::Result<(picture::YuvPicture, slice_data::SliceDecodeStats)> {
    use oxideav_core::Error;
    // Round 384: sps_admvp_flag no longer gates the IDR path — the ADMVP
    // toolset only alters inter coding-unit syntax (§7.3.8.4) and the
    // slice-header temporal-MVP group (P/B only); an intra slice decodes
    // identically under either value.
    // Round 391: sps_btt_flag and sps_suco_flag are lifted — the pixel
    // walker decodes the full §7.3.8.3 split_unit() syntax (BTT split
    // group + split_unit_coding_order_flag) via the threaded
    // `CodingTreeGates`.
    // Round 397: sps_cm_init_flag is lifted — the walkers initialise the
    // §9.3.2.2 Main-profile context tables at the slice QP and route
    // every regular bin through the §9.3.4.2.1
    // `ctxIdx = ctxIdxOffset + ctxInc` selection (`CtxSel`).
    // Round 397: sps_dquant_flag is lifted — the walkers thread the
    // §7.3.8.3 cuQpDeltaCode marks, the §7.3.8.5 code/latch presence
    // gate and the §8.7.1 eq. 1042 QpY chain.
    // Round 397: sps_eipd_flag is lifted — the coding_unit() walker
    // decodes the §7.3.8.4 MPM/PIMS/rem-mode group + the chroma mode
    // and reconstructs through the §8.4.4 EIPD kernels.
    // Round 397: sps_adcc_flag is lifted — residual_coding() routes to
    // the §7.3.8.8 residual_coding_adv() layer.
    // Round 397: sps_addb_flag is lifted — deblocking dispatches to the
    // §8.8.3 advanced filter.
    // Round 404: sps_ats_flag is lifted on the IDR path — the intra
    // `transform_unit()` reads the §7.3.8.5 `ats_cu_intra_flag` group and
    // the §8.7.4.2 DST-VII / DCT-VIII kernels drive the luma inverse
    // transform (an IDR slice has no inter CUs, so ATS-intra fully covers
    // it). `CodingTreeGates::from_sps` carries `sps_ats_flag` to the walker.
    // sps_alf_flag and sps_dra_flag are handled by the round-11 post-filter
    // pipeline and no longer gate this function.
    if !pps.single_tile_in_pic_flag {
        return Err(Error::unsupported(
            "evc decode_idr_slice: round-3 requires single_tile_in_pic_flag == 1",
        ));
    }
    let mut hdr_br = crate::bitreader::BitReader::new(slice_nal_rbsp);
    let slice_qp = parse_slice_header_for_decode(&mut hdr_br, sps, pps)?;
    hdr_br.align_to_byte();
    let consumed_bits = hdr_br.bit_position();
    if consumed_bits % 8 != 0 {
        return Err(Error::invalid(
            "evc decode_idr_slice: slice header not byte-aligned after parse",
        ));
    }
    let consumed_bytes = (consumed_bits / 8) as usize;
    if consumed_bytes >= slice_nal_rbsp.len() {
        return Err(Error::invalid(
            "evc decode_idr_slice: no slice_data bytes after header",
        ));
    }
    let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
    let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
    let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
    let max_tb_log2_size_y = 6; // eq. 51: MaxTbLog2SizeY = 6 (a constant)
    let log2_max_ibc_cand_size = sps.log2_max_ibc_cand_size().unwrap_or(0);
    let walk = slice_data::SliceWalkInputs {
        pic_width: sps.pic_width_in_luma_samples,
        pic_height: sps.pic_height_in_luma_samples,
        ctb_log2_size_y,
        min_cb_log2_size_y,
        max_tb_log2_size_y,
        chroma_format_idc: sps.chroma_format_idc,
        cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
        sps_adcc_flag: sps.sps_adcc_flag,
        sps_eipd_flag: sps.sps_eipd_flag,
        sps_dquant_flag: sps.sps_dquant_flag,
        cu_qp_delta_area: pps.log2_cu_qp_delta_area_minus6 + 6,
        sps_ibc_flag: sps.sps_ibc_flag,
        log2_max_ibc_cand_size,
        // Minimal-header entry point: ALF map fields default off.
        slice_alf_enabled_flag: false,
        slice_alf_map_flag: false,
        slice_chroma_alf_enabled_flag: false,
        slice_alf_chroma_map_flag: false,
        slice_chroma2_alf_enabled_flag: false,
        slice_alf_chroma2_map_flag: false,
        // Round 391: thread the real §7.3.8.3 BTT/SUCO gates — the
        // pixel walker decodes the Main-profile coding-tree syntax.
        tree_gates: slice_data::CodingTreeGates::from_sps(sps),
    };
    let decode = slice_data::SliceDecodeInputs {
        slice_qp,
        bit_depth_luma: sps.bit_depth_y(),
        bit_depth_chroma: sps.bit_depth_c(),
        enable_deblock: false, // round-3 fixtures keep deblock off
        sps_addb_flag: sps.sps_addb_flag,
        filter_offset_a: 0,
        filter_offset_b: 0,
        slice_cb_qp_offset: 0,
        slice_cr_qp_offset: 0,
        sps_ibc_flag: sps.sps_ibc_flag,
        log2_max_ibc_cand_size,
        sps_htdf_flag: sps.sps_htdf_flag,
    };
    slice_data::decode_baseline_idr_slice(slice_data_bytes, walk, decode)
}

/// The resolved tile plumbing of one slice: the §7.3.8.1 CTU walk
/// order, the picture tile layout, and the parsed §7.3.4 entry-point
/// offsets ([`slice_header::compute_tile_subset_byte_ranges`] turns
/// them into §7.4.5 eq. 88/89 byte subsets once the slice-data length
/// is known).
#[doc(hidden)]
pub struct SliceTilePlumbing {
    pub order: slice_data::SliceTileWalkOrder,
    pub layout: tiles::PicTileLayout,
    pub entry_point_offsets: Vec<u32>,
}

impl SliceTilePlumbing {
    /// The §7.4.5 eq. 88/89 per-tile byte subsets over a slice-data
    /// buffer of `slice_data_len` bytes: one whole-buffer range for a
    /// single-tile slice, the entry-point-derived split otherwise.
    pub fn subset_ranges(
        &self,
        slice_data_len: usize,
    ) -> oxideav_core::Result<Vec<core::ops::Range<usize>>> {
        if self.order.segments.len() <= 1 {
            return Ok(core::iter::once(0..slice_data_len).collect());
        }
        slice_header::compute_tile_subset_byte_ranges(&self.entry_point_offsets, slice_data_len)
    }
}

/// Resolve a parsed slice header's tile plumbing against the active
/// SPS/PPS: the §6.5.1 eq. 24-32 derivations feed
/// [`slice_data::resolve_slice_tile_walk_order`] and the luma-domain
/// [`tiles::PicTileLayout`], and — for a multi-tile slice — the §7.3.4
/// `entry_point_offset_minus1[ ]` loop is consumed off `br` (which must
/// sit just past `slice_cr_qp_offset`, i.e. right after
/// [`slice_header::parse_consume`]).
#[doc(hidden)]
pub fn resolve_slice_tiles(
    sps: &sps::Sps,
    pps: &pps::Pps,
    header: &slice_header::SliceHeader,
    br: &mut bitreader::BitReader,
) -> oxideav_core::Result<SliceTilePlumbing> {
    let ctb_log2 = sps.log2_ctu_size_minus5 + 5;
    let ctb = 1u32 << ctb_log2;
    let pw = sps.pic_width_in_luma_samples;
    let ph = sps.pic_height_in_luma_samples;
    let pw_ctbs = pw.div_ceil(ctb);
    let ph_ctbs = ph.div_ceil(ctb);
    let col_bd = pps.col_bd(pw_ctbs);
    let row_bd = pps.row_bd(ph_ctbs);
    let layout = tiles::PicTileLayout::from_ctb_bounds(
        &col_bd,
        &row_bd,
        ctb_log2,
        pw,
        ph,
        pps.loop_filter_across_tiles_enabled_flag,
    )?;
    let maps = pps.tile_index_maps(pw_ctbs, ph_ctbs);
    let num_tiles_in_pic = pps.num_tiles_in_pic();
    let slice_tile_idx =
        header.slice_tile_indices(&maps, pps.num_tile_columns_minus1, num_tiles_in_pic)?;
    let num_ctus_in_tile = pps.num_ctus_in_tile(pw_ctbs, ph_ctbs);
    let ts_to_rs = pps.ctb_addr_ts_to_rs(pw_ctbs, ph_ctbs);
    let order = slice_data::resolve_slice_tile_walk_order(
        &slice_tile_idx,
        &maps.first_ctb_addr_ts,
        &num_ctus_in_tile,
        &ts_to_rs,
    )?;
    let entry_point_offsets = header.parse_entry_points(
        br,
        &maps,
        pps.num_tile_columns_minus1,
        num_tiles_in_pic,
        pps.tile_offset_len_minus1,
    )?;
    Ok(SliceTilePlumbing {
        order,
        layout,
        entry_point_offsets,
    })
}

/// Everything the registered decoder needs from a fully-parsed IDR
/// slice: the reconstructed picture + stats, the slice-header ALF
/// routing (per-CTU map enables + APS ids) and the resolved tile
/// layout for the §8.8.4.5/.6 post-filter availability.
#[doc(hidden)]
pub struct IdrDecodeResult {
    pub pic: picture::YuvPicture,
    pub stats: slice_data::SliceDecodeStats,
    pub chroma_cb_enabled: bool,
    pub chroma_cr_enabled: bool,
    pub alf_luma_aps_id: Option<u8>,
    pub alf_chroma_aps_id: Option<u8>,
    pub alf_chroma2_aps_id: Option<u8>,
    pub layout: tiles::PicTileLayout,
}

/// Full-header IDR decode: the §7.3.4 slice-header parse (tile fields,
/// ALF block, deblocking controls — superseding the round-3 minimal
/// parse that ignored them), the §7.4.5 tile plumbing, and the
/// §7.3.8.1 (possibly multi-tile) pixel walk. `slice_deblocking_filter_flag`
/// and the §8.8.3 alpha/beta offsets are honoured on the IDR path.
#[doc(hidden)]
pub fn decode_idr_slice_full(
    sps: &sps::Sps,
    pps: &pps::Pps,
    slice_nal_rbsp: &[u8],
) -> oxideav_core::Result<IdrDecodeResult> {
    use oxideav_core::Error;
    let ctx = slice_header::SliceParseContext {
        single_tile_in_pic_flag: pps.single_tile_in_pic_flag,
        arbitrary_slice_present_flag: pps.arbitrary_slice_present_flag,
        tile_id_len_minus1: pps.tile_id_len_minus1,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        sps_pocs_flag: sps.sps_pocs_flag,
        sps_rpl_flag: sps.sps_rpl_flag,
        sps_alf_flag: sps.sps_alf_flag,
        sps_mmvd_flag: sps.sps_mmvd_flag,
        sps_admvp_flag: sps.sps_admvp_flag,
        sps_addb_flag: sps.sps_addb_flag,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        chroma_array_type: sps.chroma_array_type(),
        num_ref_pic_lists_in_sps_l0: sps.num_ref_pic_lists_in_sps_l0,
        num_ref_pic_lists_in_sps_l1: sps.num_ref_pic_lists_in_sps_l1,
        rpl1_idx_present_flag: pps.rpl1_idx_present_flag,
        long_term_ref_pics_flag: sps.long_term_ref_pics_flag,
        additional_lt_poc_lsb_len: pps.additional_lt_poc_lsb_len,
    };
    let mut br = bitreader::BitReader::new(slice_nal_rbsp);
    let header = slice_header::parse_consume(&mut br, nal::NalUnitType::Idr, &ctx)?;
    let plumbing = resolve_slice_tiles(sps, pps, &header, &mut br)?;
    br.align_to_byte();
    let consumed_bits = br.bit_position();
    if consumed_bits % 8 != 0 {
        return Err(Error::invalid(
            "evc decode_idr_slice_full: slice header not byte-aligned after parse",
        ));
    }
    let consumed_bytes = (consumed_bits / 8) as usize;
    if consumed_bytes >= slice_nal_rbsp.len() {
        return Err(Error::invalid(
            "evc decode_idr_slice_full: no slice_data bytes after header",
        ));
    }
    let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
    let subset_ranges = plumbing.subset_ranges(slice_data_bytes.len())?;

    let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
    let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
    let max_tb_log2_size_y = 6; // eq. 51: MaxTbLog2SizeY = 6 (a constant)
    let log2_max_ibc_cand_size = sps.log2_max_ibc_cand_size().unwrap_or(0);
    let walk = slice_data::SliceWalkInputs {
        pic_width: sps.pic_width_in_luma_samples,
        pic_height: sps.pic_height_in_luma_samples,
        ctb_log2_size_y,
        min_cb_log2_size_y,
        max_tb_log2_size_y,
        chroma_format_idc: sps.chroma_format_idc,
        cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
        sps_adcc_flag: sps.sps_adcc_flag,
        sps_eipd_flag: sps.sps_eipd_flag,
        sps_dquant_flag: sps.sps_dquant_flag,
        cu_qp_delta_area: pps.log2_cu_qp_delta_area_minus6 + 6,
        sps_ibc_flag: sps.sps_ibc_flag,
        log2_max_ibc_cand_size,
        // §7.3.8.2: the slice-header ALF map controls thread through so
        // the CTU walker decodes the per-CTU `alf_ctb_*` map (the
        // round-3 minimal parse defaulted these off on IDR).
        slice_alf_enabled_flag: header.slice_alf_enabled_flag,
        slice_alf_map_flag: header.slice_alf_map_flag,
        slice_chroma_alf_enabled_flag: header.slice_chroma_alf_enabled_flag,
        slice_alf_chroma_map_flag: header.slice_alf_chroma_map_flag,
        slice_chroma2_alf_enabled_flag: header.slice_chroma2_alf_enabled_flag,
        slice_alf_chroma2_map_flag: header.slice_alf_chroma2_map_flag,
        tree_gates: slice_data::CodingTreeGates::from_sps(sps),
    };
    let decode = slice_data::SliceDecodeInputs {
        slice_qp: header.slice_qp as i32,
        bit_depth_luma: sps.bit_depth_y(),
        bit_depth_chroma: sps.bit_depth_c(),
        enable_deblock: header.slice_deblocking_filter_flag,
        sps_addb_flag: sps.sps_addb_flag,
        filter_offset_a: header.slice_alpha_offset,
        filter_offset_b: header.slice_beta_offset,
        slice_cb_qp_offset: header.slice_cb_qp_offset,
        slice_cr_qp_offset: header.slice_cr_qp_offset,
        sps_ibc_flag: sps.sps_ibc_flag,
        log2_max_ibc_cand_size,
        sps_htdf_flag: sps.sps_htdf_flag,
    };
    let (pic, stats) = slice_data::decode_baseline_idr_slice_tiled(
        slice_data_bytes,
        walk,
        decode,
        &plumbing.order,
        &subset_ranges,
        &plumbing.layout,
    )?;
    Ok(IdrDecodeResult {
        pic,
        stats,
        chroma_cb_enabled: header.slice_chroma_alf_enabled_flag,
        chroma_cr_enabled: header.slice_chroma2_alf_enabled_flag,
        alf_luma_aps_id: header.slice_alf_luma_aps_id,
        alf_chroma_aps_id: header.slice_alf_chroma_aps_id,
        alf_chroma2_aps_id: header.slice_alf_chroma2_aps_id,
        layout: plumbing.layout,
    })
}

/// Helper for [`decode_idr_slice`]: parse the Baseline IDR slice header
/// and recover `slice_qp` for downstream dequant. Round-3 supports the
/// minimal set: `slice_pps_id`, `slice_type` (must be 2), trailing
/// flags, `slice_qp ∈ 0..=51`, `slice_cb_qp_offset`, `slice_cr_qp_offset`.
fn parse_slice_header_for_decode(
    br: &mut crate::bitreader::BitReader,
    _sps: &sps::Sps,
    _pps: &pps::Pps,
) -> oxideav_core::Result<i32> {
    use oxideav_core::Error;
    let _slice_pps_id = br.ue()?;
    let slice_type = br.ue()?;
    if slice_type != 2 {
        return Err(Error::invalid(format!(
            "evc decode_idr_slice: IDR slice_type must be 2 (got {slice_type})"
        )));
    }
    let _no_output = br.u1()?;
    let _slice_deblocking_filter_flag = br.u1()?;
    let slice_qp = br.u(6)?;
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc decode_idr_slice: slice_qp {slice_qp} > 51"
        )));
    }
    let _slice_cb_qp_offset = br.se()?;
    let _slice_cr_qp_offset = br.se()?;
    Ok(slice_qp as i32)
}

/// Internal helper that re-runs the slice header parse on a *borrowed*
/// BitReader so the caller can recover the header bit position. Mirrors
/// [`slice_header::parse`] but takes a mutable reference. Round-3 may
/// fold this into the public `slice_header::parse` once the active-PS
/// tracker lands.
fn parse_slice_header_consume(
    br: &mut crate::bitreader::BitReader,
    _nal_unit_type: nal::NalUnitType,
    _ctx: &slice_header::SliceParseContext,
) -> oxideav_core::Result<()> {
    use oxideav_core::Error;
    let _slice_pps_id = br.ue()?;
    // Baseline + IDR + single_tile_in_pic_flag = no tile fields.
    let slice_type = br.ue()?;
    if slice_type != 2 {
        return Err(Error::invalid(format!(
            "evc walk_idr_slice: IDR slice_type must be 2 (got {slice_type})"
        )));
    }
    let _no_output = br.u1()?;
    // sps_mmvd_flag, sps_alf_flag = 0 in Baseline → skip.
    // IDR + sps_pocs_flag is irrelevant (POC LSB not in IDR header).
    // sps_rpl_flag = 0 → no RPL fields.
    // not P/B → no ref_idx fields.
    let _slice_deblocking_filter_flag = br.u1()?;
    // sps_addb_flag = 0 → no alpha/beta offsets.
    let slice_qp = br.u(6)?;
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc walk_idr_slice: slice_qp {slice_qp} > 51"
        )));
    }
    let _slice_cb_qp_offset = br.se()?;
    let _slice_cr_qp_offset = br.se()?;
    Ok(())
}

/// Register the EVC implementation (currently parser-only) with a codec
/// registry. The registered decoder factory returns an unsupported-error
/// decoder per the round-1 deliverable.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("evc_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(sps::MAX_DIMENSION, sps::MAX_DIMENSION);
    // ISOBMFF sample-description FourCCs registered for EVC by
    // ISO/IEC 14496-15 (clauses 12 / 13):
    //   `evc1` — track-stored EVC
    //   `evcC` — EVCDecoderConfigurationRecord box code (atom name, kept
    //   for completeness so callers carrying a 4-cc atom can locate us).
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tags([
                CodecTag::fourcc(b"evc1"),
                CodecTag::fourcc(b"evcC"),
                CodecTag::fourcc(b"EVC1"),
            ]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::tests::BitEmitter;

    /// Build a single-NAL length-prefixed EVC bitstream containing a
    /// minimal 320x240 4:2:0 8-bit SPS, suitable for exercising `probe`.
    fn build_minimal_sps_stream() -> Vec<u8> {
        // SPS body
        let mut body = BitEmitter::new();
        body.ue(0); // sps_seq_parameter_set_id
        body.u(8, 1); // profile_idc
        body.u(8, 30); // level_idc
        body.u(32, 0); // toolset_idc_h
        body.u(32, 0); // toolset_idc_l
        body.ue(1); // chroma_format_idc
        body.ue(320); // pic_width_in_luma_samples
        body.ue(240); // pic_height_in_luma_samples
        body.ue(0); // bit_depth_luma_minus8
        body.ue(0); // bit_depth_chroma_minus8
        for _ in 0..13 {
            body.u(1, 0);
        }
        body.ue(1); // log2_sub_gop_length
        body.ue(1); // max_num_tid0_ref_pics
        body.u(1, 0); // picture_cropping_flag
        body.u(1, 0); // chroma_qp_table_present_flag
        body.u(1, 0); // vui_parameters_present_flag
        body.finish_with_trailing_bits();
        let sps_rbsp = body.into_bytes();

        // 2-byte NAL header for SPS (NUT=24)
        let nut_plus1: u16 = 24 + 1;
        let mut hdr_word: u16 = 0;
        hdr_word |= (nut_plus1 & 0x3F) << 9;
        // tid 0, reserved 0, ext 0 — leaves the lower bits zero
        let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];

        // Build the length-prefixed envelope: [u32 len BE][NAL]
        let nal_len = (hdr.len() + sps_rbsp.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&nal_len.to_be_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&sps_rbsp);
        out
    }

    #[test]
    fn probe_minimal_sps_stream() {
        let bs = build_minimal_sps_stream();
        let info = probe(&bs).expect("probe must recover SPS dimensions");
        assert_eq!(info.width, 320);
        assert_eq!(info.height, 240);
        assert_eq!(info.bit_depth_luma, 8);
        assert_eq!(info.bit_depth_chroma, 8);
        assert_eq!(info.chroma_format_idc, 1);
        assert_eq!(info.profile_idc, 1);
        assert_eq!(info.level_idc, 30);
    }

    #[test]
    fn probe_returns_none_on_empty() {
        assert!(probe(&[]).is_none());
    }

    #[test]
    fn probe_returns_none_on_pps_only() {
        // PPS NAL with empty body — should not satisfy probe.
        let nut_plus1: u16 = 25 + 1;
        let mut hdr_word: u16 = 0;
        hdr_word |= (nut_plus1 & 0x3F) << 9;
        let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps id
        pps_body.ue(0); // sps id
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.u(1, 0);
        pps_body.u(1, 1); // single_tile_in_pic_flag
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.finish_with_trailing_bits();
        let body = pps_body.into_bytes();
        let nal_len = (hdr.len() + body.len()) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(&nal_len.to_be_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&body);
        assert!(probe(&out).is_none());
    }

    #[test]
    fn register_creates_factory() {
        let mut reg = CodecRegistry::default();
        register(&mut reg);
        // The registry must know the "evc" codec id and have a decoder
        // implementation registered.
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn make_decoder_handles_empty_packet() {
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), Vec::new());
        // Empty packet → no NALs → ok, no frame emitted.
        dec.send_packet(&pkt).unwrap();
        let err = dec.receive_frame().unwrap_err();
        // Iterating with no input must surface NeedMore (not Eof).
        assert!(matches!(err, oxideav_core::Error::NeedMore));
    }

    /// End-to-end Baseline-IDR walk: build an SPS + PPS in-memory,
    /// hand-craft a slice header for an IDR slice, append a CABAC-encoded
    /// `slice_data()` payload, then run [`walk_idr_slice`] across it
    /// and verify every bin is consumed cleanly through the
    /// `end_of_tile_one_bit` terminate decision.
    ///
    /// This is the round-2 deliverable's "real (small) EVC IDR slice has
    /// its bins consumed cleanly" milestone. The slice covers a 64×64
    /// picture (a single 64×64 CTU explicitly split via `split_cu_flag = 1`
    /// into four 32×32 leaves; each leaf then declines to split via
    /// `split_cu_flag = 0` since the walker permits more recursion under
    /// the Baseline default `MinCbLog2SizeY = 2`). Each leaf goes through
    /// the dual-tree luma + chroma `coding_unit()` pair with no CBFs set.
    ///
    /// Bin sequence (21 regular bins + terminate):
    /// * 1 × `split_cu_flag = 1` (CTB)
    /// * 4 × `split_cu_flag = 0` (each 32×32 child)
    /// * 4 × (intra_pred_mode + cbf_luma + cbf_cb + cbf_cr) = 16
    #[test]
    fn end_to_end_walk_baseline_idr_slice() {
        use crate::cabac::CabacEncoder;
        // Build a Baseline-profile SPS for 64×64 4:2:0 8-bit. Default
        // toolset (sps_btt = 0) leaves the parser at CtbLog2SizeY = 6
        // (64×64 CTU) and MinCbLog2SizeY = 2 (4×4 minimum CB).
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc = Baseline (0)
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc = 1 (4:2:0)
        sps_body.ue(64); // pic_width
        sps_body.ue(64); // pic_height
        sps_body.ue(0); // bit_depth_luma_minus8
        sps_body.ue(0); // bit_depth_chroma_minus8
        for _ in 0..13 {
            sps_body.u(1, 0);
        }
        sps_body.ue(1); // log2_sub_gop_length
        sps_body.ue(1); // max_num_tid0_ref_pics
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        let sps_rbsp = sps_body.into_bytes();
        let sps = sps::parse(&sps_rbsp).expect("SPS parses");

        // Baseline PPS: cu_qp_delta_enabled = 0 so transform_unit doesn't
        // emit cu_qp_delta_abs.
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps_id
        pps_body.ue(0); // sps_id
        pps_body.ue(0); // num_ref_idx_default_active_minus1[0]
        pps_body.ue(0); // num_ref_idx_default_active_minus1[1]
        pps_body.ue(0); // additional_lt_poc_lsb_len
        pps_body.u(1, 0); // rpl1_idx_present_flag
        pps_body.u(1, 1); // single_tile_in_pic_flag
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0); // explicit_tile_id_flag
        pps_body.u(1, 0); // pic_dra_enabled_flag
        pps_body.u(1, 0); // arbitrary_slice_present_flag
        pps_body.u(1, 0); // constrained_intra_pred_flag
        pps_body.u(1, 0); // cu_qp_delta_enabled_flag = 0
        pps_body.finish_with_trailing_bits();
        let pps_rbsp = pps_body.into_bytes();
        let pps = pps::parse(&pps_rbsp).expect("PPS parses");

        // Slice header (Baseline IDR with single_tile_in_pic_flag = 1).
        let mut hdr = BitEmitter::new();
        hdr.ue(0); // slice_pps_id
        hdr.ue(2); // slice_type = I
        hdr.u(1, 0); // no_output_of_prior_pics_flag
        hdr.u(1, 0); // slice_deblocking_filter_flag
        hdr.u(6, 22); // slice_qp = 22
        hdr.ue(0); // slice_cb_qp_offset (se: 0)
        hdr.ue(0); // slice_cr_qp_offset
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();

        // CABAC-encoded slice_data:
        //   1× split_cu_flag = 1 at the CTB
        //   4× split_cu_flag = 0 at each 32×32 child
        //   4× (intra_pred_mode + cbf_luma + cbf_cb + cbf_cr)
        //   then terminate(true).
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // CTB split = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split = 0
            enc.encode_decision(0, 0, 0); // intra_pred_mode
            enc.encode_decision(0, 0, 0); // cbf_luma
            enc.encode_decision(0, 0, 0); // cbf_cb
            enc.encode_decision(0, 0, 0); // cbf_cr
        }
        enc.encode_terminate(true);
        let slice_data_bytes = enc.finish();
        slice_rbsp.extend_from_slice(&slice_data_bytes);

        let stats = walk_idr_slice(&sps, &pps, &slice_rbsp).expect("walk succeeds");
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.split_cu_flag_bins, 5, "1 CTB split + 4 child splits");
        assert_eq!(stats.coding_units, 8, "4 leaves × (luma + chroma)");
        assert_eq!(stats.intra_pred_mode_bins, 4);
        assert_eq!(stats.cbf_luma_bins, 4);
        assert_eq!(stats.cbf_chroma_bins, 8);
    }

    /// Confirm the integration helper rejects a non-Baseline SPS up front
    /// rather than handing the walker a bitstream it cannot parse.
    #[test]
    fn walk_idr_slice_rejects_main_profile_sps() {
        // SPS with sps_btt_flag = 1 — outside Baseline.
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0);
        sps_body.u(8, 1); // profile_idc = 1 (Main)
        sps_body.u(8, 41);
        sps_body.u(32, 1); // toolset_idc_h non-zero
        sps_body.u(32, 1);
        sps_body.ue(1);
        sps_body.ue(1920);
        sps_body.ue(1080);
        sps_body.ue(2);
        sps_body.ue(2);
        sps_body.u(1, 1); // sps_btt_flag = 1
        sps_body.ue(1);
        sps_body.ue(0);
        sps_body.ue(2);
        sps_body.ue(1);
        sps_body.ue(0);
        sps_body.u(1, 0); // sps_suco
        sps_body.u(1, 0); // sps_admvp
        sps_body.u(1, 0); // sps_eipd
        sps_body.u(1, 0); // sps_cm_init
        sps_body.u(1, 0); // sps_iqt
        sps_body.u(1, 0); // sps_addb
        sps_body.u(1, 0); // sps_alf
        sps_body.u(1, 0); // sps_htdf
        sps_body.u(1, 0); // sps_rpl
        sps_body.u(1, 0); // sps_pocs
        sps_body.u(1, 0); // sps_dquant
        sps_body.u(1, 0); // sps_dra
        sps_body.ue(1);
        sps_body.ue(1);
        sps_body.u(1, 0);
        sps_body.u(1, 0);
        sps_body.u(1, 0);
        sps_body.finish_with_trailing_bits();
        let sps_rbsp = sps_body.into_bytes();
        let sps = sps::parse(&sps_rbsp).expect("SPS parses");

        // Minimal PPS.
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.u(1, 0);
        pps_body.u(1, 1);
        pps_body.ue(0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.u(1, 0);
        pps_body.finish_with_trailing_bits();
        let pps_rbsp = pps_body.into_bytes();
        let pps = pps::parse(&pps_rbsp).unwrap();

        let res = walk_idr_slice(&sps, &pps, &[0u8; 16]);
        assert!(res.is_err());
        let err_text = format!("{}", res.unwrap_err());
        assert!(
            err_text.contains("Baseline-profile toolset"),
            "got: {err_text}"
        );
    }

    /// Build a minimal Baseline SPS for an `(width × height)` 4:2:0 8-bit
    /// picture. CTB size defaults to 64×64 (`log2_ctu_size_minus5 = 0` →
    /// `CtbLog2SizeY = 5` since `sps_btt_flag = 0` keeps the spec's
    /// default of `1` per §7.4.3.1, putting CTB at 32×32 — round-3
    /// fixtures keep below 64×64 to dodge the unimplemented `nTbS = 64`
    /// transform path).
    fn build_baseline_sps_rbsp(width: u32, height: u32) -> Vec<u8> {
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc Baseline
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc 4:2:0
        sps_body.ue(width);
        sps_body.ue(height);
        sps_body.ue(0); // bit_depth_luma_minus8
        sps_body.ue(0); // bit_depth_chroma_minus8
        for _ in 0..13 {
            sps_body.u(1, 0);
        }
        sps_body.ue(1); // log2_sub_gop_length
        sps_body.ue(1); // max_num_tid0_ref_pics
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        sps_body.into_bytes()
    }

    /// Round-8 helper: build a Baseline-shaped SPS that turns on
    /// `sps_rpl_flag = 1` and `sps_pocs_flag = 1` so non-IDR slice
    /// headers exercise the RPL parsing path. `num_ref_pic_lists_in_sps`
    /// is set to 0 for both lists so the slice header MUST carry an
    /// inline `ref_pic_list_struct()`.
    fn build_rpl_sps_rbsp(width: u32, height: u32) -> Vec<u8> {
        let mut sps_body = sps::tests::BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc 4:2:0
        sps_body.ue(width);
        sps_body.ue(height);
        sps_body.ue(0); // bit_depth_luma_minus8
        sps_body.ue(0); // bit_depth_chroma_minus8
                        // 13 toolset bit-flags up to (and including) sps_cm_init/sps_iqt
                        // — all default to 0 except sps_rpl_flag and sps_pocs_flag.
        sps_body.u(1, 0); // sps_btt
        sps_body.u(1, 0); // sps_suco
        sps_body.u(1, 0); // sps_admvp
        sps_body.u(1, 0); // sps_eipd
        sps_body.u(1, 0); // sps_cm_init
        sps_body.u(1, 0); // sps_iqt
        sps_body.u(1, 0); // sps_addb
        sps_body.u(1, 0); // sps_alf
        sps_body.u(1, 0); // sps_htdf
        sps_body.u(1, 1); // sps_rpl  ← enable
        sps_body.u(1, 1); // sps_pocs ← enable
        sps_body.u(1, 0); // sps_dquant
        sps_body.u(1, 0); // sps_dra
                          // sps_pocs_flag=1 → log2_max_pic_order_cnt_lsb_minus4
        sps_body.ue(4); // log2_max_pic_order_cnt_lsb_minus4 = 4 → 8 bits
                        // sps_rpl_flag=1 path:
                        //   sps_max_dec_pic_buffering_minus1, long_term_ref_pics_flag,
                        //   rpl1_same_as_rpl0_flag, num_ref_pic_lists_in_sps[0],
                        //   (num_ref_pic_lists_in_sps[1] when not same).
        sps_body.ue(1); // sps_max_dec_pic_buffering_minus1
        sps_body.u(1, 0); // long_term_ref_pics_flag
        sps_body.u(1, 0); // rpl1_same_as_rpl0_flag (need both lists explicitly)
        sps_body.ue(0); // num_ref_pic_lists_in_sps[0] = 0 (force inline RPL)
        sps_body.ue(0); // num_ref_pic_lists_in_sps[1] = 0
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        sps_body.into_bytes()
    }

    /// Build a minimal Baseline PPS for `cu_qp_delta_enabled_flag = 0`.
    fn build_baseline_pps_rbsp() -> Vec<u8> {
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps_id
        pps_body.ue(0); // sps_id
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.ue(0);
        pps_body.u(1, 0); // rpl1_idx_present_flag
        pps_body.u(1, 1); // single_tile_in_pic_flag
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0); // explicit_tile_id_flag
        pps_body.u(1, 0); // pic_dra_enabled_flag
        pps_body.u(1, 0); // arbitrary_slice_present_flag
        pps_body.u(1, 0); // constrained_intra_pred_flag
        pps_body.u(1, 0); // cu_qp_delta_enabled_flag = 0
        pps_body.finish_with_trailing_bits();
        pps_body.into_bytes()
    }

    /// **Round-3 end-to-end pixel decode.** Build a Baseline IDR slice
    /// covering a 32×32 picture (one 32×32 CTU split into four 16×16
    /// leaves; each leaf carries `intra_pred_mode = 0` (= INTRA_DC) and
    /// `cbf_luma = cbf_cb = cbf_cr = 0`). Decode through
    /// [`decode_idr_slice`] and verify the reconstructed Y plane is
    /// uniformly 128 (the bit-depth substitution value for "no
    /// neighbours" is 128, DC-prediction of all-128 references is 128,
    /// and a zero residual leaves it at 128).
    #[test]
    fn round3_end_to_end_decode_grey_idr() {
        use crate::cabac::CabacEncoder;
        // 64×64 picture (1 CTU at the default CTB log2 = 6) — keeps the
        // walker on the simple "CTB == picture" path, avoiding implicit
        // boundary splits.
        let sps = sps::parse(&build_baseline_sps_rbsp(64, 64)).unwrap();
        let pps = pps::parse(&build_baseline_pps_rbsp()).unwrap();
        // Slice header for an IDR with deblocking off, slice_qp = 22.
        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2); // I slice
        hdr.u(1, 0);
        hdr.u(1, 0); // slice_deblocking_filter_flag = 0
        hdr.u(6, 22);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();
        // Encode CABAC: one CTB split (32 → four 16x16 leaves), each
        // leaf intra_pred_mode = 0 (one "0" bin), cbf_luma = cbf_cb =
        // cbf_cr = 0 (no residual_coding fires).
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // CTB split_cu_flag = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split_cu_flag = 0 (leaf)
                                          // dual-tree luma CU: intra_pred_mode + cbf_luma
            enc.encode_decision(0, 0, 0); // intra_pred_mode_idx = 0 (one "0" bin)
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
                                          // dual-tree chroma CU: cbf_cb + cbf_cr (no intra_pred_mode_idx
                                          // for chroma in sps_eipd_flag=0 dual-tree path)
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        slice_rbsp.extend_from_slice(&enc.finish());

        let (pic, stats) = decode_idr_slice(&sps, &pps, &slice_rbsp).unwrap();
        // Picture geometry checks.
        assert_eq!(pic.width, 64);
        assert_eq!(pic.height, 64);
        assert_eq!(pic.y.len(), 64 * 64);
        assert_eq!(pic.cb.len(), 32 * 32);
        assert_eq!(pic.cr.len(), 32 * 32);
        // Stats: 1 CTU, 5 split_cu_flag bins (1 CTB + 4 children), 8
        // coding_units (luma+chroma per leaf), 4 intra_pred_mode bins,
        // 4 cbf_luma bins, 8 cbf_chroma bins.
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.split_cu_flag_bins, 5);
        assert_eq!(stats.coding_units, 8);
        assert_eq!(stats.intra_pred_mode_bins, 4);
        assert_eq!(stats.cbf_luma_bins, 4);
        assert_eq!(stats.cbf_chroma_bins, 8);
        // Pixel content: every Y/Cb/Cr sample should be 128 (the IDR-with-
        // no-neighbours INTRA_DC prediction value at bit-depth 8). Since
        // the picture buffer was pre-filled with 128 and the prediction
        // computes 128 from all-128 references with zero residual, the
        // result is uniform 128.
        let mismatched_y = pic.y.iter().filter(|&&v| v != 128).count();
        assert_eq!(
            mismatched_y, 0,
            "Y plane should be uniform 128 (got {mismatched_y} non-128 samples)"
        );
        let mismatched_cb = pic.cb.iter().filter(|&&v| v != 128).count();
        let mismatched_cr = pic.cr.iter().filter(|&&v| v != 128).count();
        assert_eq!(mismatched_cb, 0);
        assert_eq!(mismatched_cr, 0);
        // PSNR check vs hand-computed reference (uniform 128): PSNR is
        // infinite at MSE = 0 — we only assert MSE = 0.
        let mse: f64 = pic
            .y
            .iter()
            .map(|&v| (v as f64 - 128.0).powi(2))
            .sum::<f64>()
            / pic.y.len() as f64;
        assert_eq!(mse, 0.0);
    }

    /// eq. 51 regression: `MaxTbLog2SizeY` is the constant 6, not a
    /// CTU-derived cap. A Baseline SPS with `sps_btt_flag = 0` infers
    /// `log2_ctu_size_minus5 = 1` (64×64 CTU); an unsplit 64×64 intra CB
    /// is then a **single** 64×64 TB, so a lone DC coefficient must lift
    /// the whole 64×64 luma plane uniformly. (The historical
    /// `min(CtbLog2SizeY, 5)` cap parsed the same bins for a DC-only
    /// block but reconstructed only the top-left 32×32 — the bottom-right
    /// sample is the discriminator.)
    #[test]
    fn eq51_max_tb_is_constant_6_single_64x64_tb() {
        use crate::cabac::CabacEncoder;
        let sps = sps::parse(&build_baseline_sps_rbsp(64, 64)).unwrap();
        assert_eq!(sps.log2_ctu_size_minus5, 1, "inferred 64×64 CTU");
        let pps = pps::parse(&build_baseline_pps_rbsp()).unwrap();
        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2); // I slice
        hdr.u(1, 0);
        hdr.u(1, 0); // deblocking off
        hdr.u(6, 51);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 0); // split_cu_flag = 0 → unsplit 64×64 CB
                                      // luma CU: DC mode + one DC coefficient.
        enc.encode_decision(0, 0, 0); // intra_pred_mode = 0 (DC)
        enc.encode_decision(0, 0, 1); // cbf_luma = 1
        enc.encode_decision(0, 0, 0); // coeff_zero_run = 0
        for _ in 0..63 {
            enc.encode_decision(0, 0, 1); // coeff_abs_level_minus1 = 63
        }
        enc.encode_decision(0, 0, 0); // U terminator → level 64
        enc.encode_bypass(0); // sign = +
        enc.encode_decision(0, 0, 1); // coeff_last_flag = 1
                                      // chroma CU: no residual.
        enc.encode_decision(0, 0, 0); // cbf_cb = 0
        enc.encode_decision(0, 0, 0); // cbf_cr = 0
        enc.encode_terminate(true);
        slice_rbsp.extend_from_slice(&enc.finish());

        let (pic, stats) = decode_idr_slice(&sps, &pps, &slice_rbsp).unwrap();
        assert_eq!(stats.ctus, 1);
        assert_eq!(stats.cbf_luma_bins, 1, "single TU → one cbf_luma bin");
        // The inverse transform of the lone coefficient spans the whole
        // 64×64 TB: samples beyond the 33rd row/column move too. (Under
        // the historical 32×32 cap the reconstruct only wrote the
        // top-left 32×32, leaving these at the 128 prefill.)
        assert_ne!(pic.y[0], 128, "residual must shift the plane");
        assert_ne!(
            pic.y[33], 128,
            "row 0, col 33 lies outside a 32×32 TB — the 64×64 TB must cover it"
        );
        assert_ne!(
            pic.y[33 * 64],
            128,
            "row 33, col 0 lies outside a 32×32 TB — the 64×64 TB must cover it"
        );
        // Chroma untouched.
        assert!(pic.cb.iter().all(|&v| v == 128));
        assert!(pic.cr.iter().all(|&v| v == 128));
    }

    /// End-to-end pixel decode through `make_decoder` (the registered
    /// codec factory). Wraps the Baseline IDR slice from above into a
    /// length-prefixed NAL stream containing SPS + PPS + IDR, sends it
    /// to the registered decoder, and pulls a `Frame::Video` out of
    /// `receive_frame`. Verifies the pixel plane shape and content.
    #[test]
    fn round3_make_decoder_decodes_idr_to_grey_frame() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};

        let sps_rbsp = build_baseline_sps_rbsp(64, 64);
        let pps_rbsp = build_baseline_pps_rbsp();

        // Slice header (IDR, slice_qp = 22, deblocking off).
        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2);
        hdr.u(1, 0);
        hdr.u(1, 0);
        hdr.u(6, 22);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut idr_rbsp = hdr.into_bytes();
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // CTB split = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split = 0
            enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&enc.finish());

        // Build NAL headers + length-prefixed envelope.
        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp)); // SPS NUT = 24
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp)); // PPS NUT = 25
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp)); // IDR NUT = 1

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let video = match frame {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };
        assert_eq!(video.planes.len(), 3);
        assert_eq!(video.planes[0].stride, 64);
        assert_eq!(video.planes[0].data.len(), 64 * 64);
        assert!(video.planes[0].data.iter().all(|&v| v == 128));
        assert_eq!(video.planes[1].stride, 32);
        assert_eq!(video.planes[1].data.len(), 32 * 32);
        assert!(video.planes[1].data.iter().all(|&v| v == 128));
        assert_eq!(video.planes[2].data.len(), 32 * 32);
    }

    /// **Round-4 end-to-end fixture through `make_decoder`.** Push an
    /// SPS + PPS + IDR + P-slice sequence into the registered EVC
    /// decoder. The IDR is a uniform grey 32×32 picture; the P-slice is
    /// a single 32×32 leaf with `cu_skip_flag = 1` and `mvp_idx_l0 = 3`
    /// (zero MV). Both frames must come out of `receive_frame`, with the
    /// P frame being a verbatim copy of the IDR (since zero MV + zero
    /// residual ≡ identity).
    #[test]
    fn round4_make_decoder_decodes_idr_plus_p_to_two_frames() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let sps_rbsp = build_baseline_sps_rbsp(32, 32);
        let pps_rbsp = build_baseline_pps_rbsp();

        // IDR: single 32×32 leaf at log2 = 5 (no split).
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0);
        idr_hdr.ue(2); // I slice
        idr_hdr.u(1, 0); // no_output
        idr_hdr.u(1, 0); // slice_deblocking_filter_flag
        idr_hdr.u(6, 22); // slice_qp
        idr_hdr.ue(0);
        idr_hdr.ue(0);
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        let mut idr_enc = CabacEncoder::new();
        // 32x32 single leaf at log2 = 5 (no split needed since
        // log2 == ctb_log2_size_y → no split_cu_flag emitted in walker
        // path? Actually log2 > min so split is emitted).
        // With ctb_log2 = 5 and min_cb_log2 = 4, the CTU is 32×32, and
        // we want a single 32x32 leaf → split_cu_flag = 0.
        idr_enc.encode_decision(0, 0, 0); // split_cu_flag = 0 at CTB
                                          // dual-tree luma CU (32x32):
        idr_enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
        idr_enc.encode_decision(0, 0, 0); // cbf_luma
                                          // dual-tree chroma CU:
        idr_enc.encode_decision(0, 0, 0); // cbf_cb
        idr_enc.encode_decision(0, 0, 0); // cbf_cr
        idr_enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&idr_enc.finish());

        // P slice header: slice_type = 1 (P), no override, deblock off.
        let mut p_hdr = BitEmitter::new();
        p_hdr.ue(0); // slice_pps_id
        p_hdr.ue(1); // slice_type = P
                     // Not IDR + sps_pocs_flag = 0 → no POC LSB.
                     // Not IDR + slice_type=P → ref-idx + admvp branch.
        p_hdr.u(1, 0); // num_ref_idx_active_override_flag = 0
                       // sps_admvp_flag = 0 → skip temporal_mvp.
        p_hdr.u(1, 0); // slice_deblocking_filter_flag
        p_hdr.u(6, 22); // slice_qp
        p_hdr.ue(0);
        p_hdr.ue(0);
        while p_hdr.bit_position() % 8 != 0 {
            p_hdr.u(1, 0);
        }
        let mut p_rbsp = p_hdr.into_bytes();
        let mut p_enc = CabacEncoder::new();
        p_enc.encode_decision(0, 0, 0); // split_cu_flag = 0 at CTB
                                        // Single-tree inter CU:
        p_enc.encode_decision(0, 0, 1); // cu_skip_flag = 1
        for _ in 0..3 {
            p_enc.encode_decision(0, 0, 1); // mvp_idx_l0 = 3 prefix (3 ones)
        }
        p_enc.encode_decision(0, 0, 0); // cbf_luma
        p_enc.encode_decision(0, 0, 0); // cbf_cb
        p_enc.encode_decision(0, 0, 0); // cbf_cr
        p_enc.encode_terminate(true);
        p_rbsp.extend_from_slice(&p_enc.finish());

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp)); // SPS
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp)); // PPS
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp)); // IDR
        bs.extend_from_slice(&nal_envelope(0, &p_rbsp)); // NonIDR

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        // Pull the two frames.
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        let v0 = match f0 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v1 = match f1 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        // Both should be uniform 128 (the IDR is grey, the P slice is a
        // zero-MV, zero-residual copy of the IDR → still grey).
        assert!(v0.planes[0].data.iter().all(|&v| v == 128));
        assert!(v1.planes[0].data.iter().all(|&v| v == 128));
        assert_eq!(
            v0.planes[0].data, v1.planes[0].data,
            "P frame must equal IDR"
        );
        assert_eq!(v0.planes[1].data, v1.planes[1].data);
        assert_eq!(v0.planes[2].data, v1.planes[2].data);
        // PSNR Y = ∞ (MSE=0). The acceptance bar is ≥ 30 dB.
        let mse: f64 = v0.planes[0]
            .data
            .iter()
            .zip(v1.planes[0].data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / v0.planes[0].data.len() as f64;
        assert_eq!(mse, 0.0, "PSNR must be infinite for identical frames");
    }

    /// **Round-8 RPL non-IDR fixture.** Same shape as the round-4 IDR+P
    /// fixture but with an SPS that turns on `sps_rpl_flag = 1` and
    /// `sps_pocs_flag = 1`, so the P slice header carries:
    ///
    ///   * `slice_pic_order_cnt_lsb` (8 bits per `log2_max_poc_lsb=8`),
    ///   * a per-list inline `ref_pic_list_struct()` (one STRP entry each
    ///     because `num_ref_pic_lists_in_sps[i] = 0`).
    ///
    /// The decoder must walk these fields cleanly via the canonical
    /// slice_header parser, then drive the inter pipeline as before.
    /// Both the IDR and the P frame come back as uniform-128 — PSNR Y =
    /// ∞ (MSE = 0), well above the 30 dB acceptance bar.
    #[test]
    fn round8_rpl_non_idr_decodes_to_two_frames() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let sps_rbsp = build_rpl_sps_rbsp(32, 32);
        let pps_rbsp = build_baseline_pps_rbsp();

        // IDR slice (slice_type=I, no RPL fields per §7.3.4 IDR branch).
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0);
        idr_hdr.ue(2); // I slice
        idr_hdr.u(1, 0); // no_output
        idr_hdr.u(1, 0); // slice_deblocking_filter_flag
        idr_hdr.u(6, 22); // slice_qp
        idr_hdr.ue(0);
        idr_hdr.ue(0);
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        let mut idr_enc = CabacEncoder::new();
        idr_enc.encode_decision(0, 0, 0); // split_cu_flag = 0
        idr_enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
        idr_enc.encode_decision(0, 0, 0); // cbf_luma
        idr_enc.encode_decision(0, 0, 0); // cbf_cb
        idr_enc.encode_decision(0, 0, 0); // cbf_cr
        idr_enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&idr_enc.finish());

        // P slice header with inline RPL on both lists.
        let mut p_hdr = BitEmitter::new();
        p_hdr.ue(0); // slice_pps_id
        p_hdr.ue(1); // slice_type = P
                     // sps_pocs_flag = 1 + non-IDR → POC LSB (8 bits).
        p_hdr.u(8, 1);
        // sps_rpl_flag = 1 path — for both i = 0 and i = 1:
        //   ref_pic_list_sps_flag[i] omitted because
        //     num_ref_pic_lists_in_sps[i] == 0 → not signalled.
        //   The inline `ref_pic_list_struct()` follows directly.
        // RPL L0 inline: 1 STRP, delta=1, sign=0 (negative). Round-9
        // resolves the ref POC as `slice_poc + signed_delta`, so
        // sign=0 → ref POC = 1 + (-1) = 0 → matches the IDR.
        p_hdr.ue(1); // num_strp_entries
        p_hdr.ue(1); // delta_poc_st = 1
        p_hdr.u(1, 0); // sign negative → -1
                       // RPL L1 inline: same shape (delta=1, sign=0).
        p_hdr.ue(1);
        p_hdr.ue(1);
        p_hdr.u(1, 0);
        // P slice → ref_idx + admvp branch:
        p_hdr.u(1, 0); // num_ref_idx_active_override_flag
                       // sps_admvp_flag = 0 → no temporal_mvp.
        p_hdr.u(1, 0); // slice_deblocking_filter_flag
        p_hdr.u(6, 22); // slice_qp
        p_hdr.ue(0);
        p_hdr.ue(0);
        while p_hdr.bit_position() % 8 != 0 {
            p_hdr.u(1, 0);
        }
        let mut p_rbsp = p_hdr.into_bytes();
        let mut p_enc = CabacEncoder::new();
        p_enc.encode_decision(0, 0, 0); // split_cu_flag = 0
                                        // Single-tree inter CU:
        p_enc.encode_decision(0, 0, 1); // cu_skip_flag = 1
        for _ in 0..3 {
            p_enc.encode_decision(0, 0, 1); // mvp_idx_l0 = 3 prefix
        }
        p_enc.encode_decision(0, 0, 0); // cbf_luma
        p_enc.encode_decision(0, 0, 0); // cbf_cb
        p_enc.encode_decision(0, 0, 0); // cbf_cr
        p_enc.encode_terminate(true);
        p_rbsp.extend_from_slice(&p_enc.finish());

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp));
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp));
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp));
        bs.extend_from_slice(&nal_envelope(0, &p_rbsp));

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        let v0 = match f0 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v1 = match f1 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        // Both are uniform 128 (zero-MV inter copy of the all-128 IDR).
        assert!(v0.planes[0].data.iter().all(|&v| v == 128));
        assert!(v1.planes[0].data.iter().all(|&v| v == 128));
        assert_eq!(v0.planes[0].data, v1.planes[0].data);
        let mse: f64 = v0.planes[0]
            .data
            .iter()
            .zip(v1.planes[0].data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / v0.planes[0].data.len() as f64;
        assert_eq!(mse, 0.0, "RPL P-slice must be bit-identical to the IDR");
    }

    /// **Round-9 multi-frame DPB + POC fixture.** Decode an IDR + two P
    /// slices (POC 0, 1, 2) where each P slice references the
    /// previously-decoded frame via an inline RPL. The DPB must keep
    /// the IDR + the first P alive long enough for the second P's RPL
    /// (`delta_poc_st = 1, sign = 0` → `cur - 1`) to resolve. All three
    /// frames come back as uniform-128 (the IDR is grey + the P slices
    /// are zero-MV identity copies of the previous frame).
    #[test]
    fn round9_three_frame_idr_p_p_with_dpb() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let sps_rbsp = build_rpl_sps_rbsp(32, 32);
        let pps_rbsp = build_baseline_pps_rbsp();

        // IDR slice (POC 0).
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0);
        idr_hdr.ue(2); // I slice
        idr_hdr.u(1, 0); // no_output
        idr_hdr.u(1, 0); // slice_deblocking_filter_flag
        idr_hdr.u(6, 22); // slice_qp
        idr_hdr.ue(0);
        idr_hdr.ue(0);
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        let mut idr_enc = CabacEncoder::new();
        idr_enc.encode_decision(0, 0, 0); // split_cu_flag = 0
        idr_enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
        idr_enc.encode_decision(0, 0, 0); // cbf_luma
        idr_enc.encode_decision(0, 0, 0); // cbf_cb
        idr_enc.encode_decision(0, 0, 0); // cbf_cr
        idr_enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&idr_enc.finish());

        // Helper to build a P slice referencing POC = cur_poc - 1.
        fn build_p_slice(cur_poc_lsb: u32) -> Vec<u8> {
            use crate::cabac::CabacEncoder;
            let mut p_hdr = BitEmitter::new();
            p_hdr.ue(0); // slice_pps_id
            p_hdr.ue(1); // slice_type = P
            p_hdr.u(8, cur_poc_lsb); // POC LSB
                                     // RPL L0 inline: 1 STRP, delta=1, sign=0 → ref = cur - 1.
            p_hdr.ue(1);
            p_hdr.ue(1);
            p_hdr.u(1, 0);
            // RPL L1 inline: same shape.
            p_hdr.ue(1);
            p_hdr.ue(1);
            p_hdr.u(1, 0);
            p_hdr.u(1, 0); // num_ref_idx_active_override_flag
            p_hdr.u(1, 0); // slice_deblocking_filter_flag
            p_hdr.u(6, 22); // slice_qp
            p_hdr.ue(0);
            p_hdr.ue(0);
            while p_hdr.bit_position() % 8 != 0 {
                p_hdr.u(1, 0);
            }
            let mut p_rbsp = p_hdr.into_bytes();
            let mut p_enc = CabacEncoder::new();
            p_enc.encode_decision(0, 0, 0); // split_cu_flag = 0
            p_enc.encode_decision(0, 0, 1); // cu_skip_flag = 1
            for _ in 0..3 {
                p_enc.encode_decision(0, 0, 1); // mvp_idx_l0 = 3
            }
            p_enc.encode_decision(0, 0, 0); // cbf_luma
            p_enc.encode_decision(0, 0, 0); // cbf_cb
            p_enc.encode_decision(0, 0, 0); // cbf_cr
            p_enc.encode_terminate(true);
            p_rbsp.extend_from_slice(&p_enc.finish());
            p_rbsp
        }
        let p1_rbsp = build_p_slice(1);
        let p2_rbsp = build_p_slice(2);

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp));
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp));
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp));
        bs.extend_from_slice(&nal_envelope(0, &p1_rbsp));
        bs.extend_from_slice(&nal_envelope(0, &p2_rbsp));

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        let f2 = dec.receive_frame().unwrap();
        let v0 = match f0 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v1 = match f1 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v2 = match f2 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        assert!(v0.planes[0].data.iter().all(|&v| v == 128));
        assert!(v1.planes[0].data.iter().all(|&v| v == 128));
        assert!(v2.planes[0].data.iter().all(|&v| v == 128));
        assert_eq!(v0.planes[0].data, v1.planes[0].data);
        assert_eq!(v1.planes[0].data, v2.planes[0].data);
    }

    /// **Round-10 flush() drain end-to-end.** Decodes a 2-frame
    /// IDR + P bitstream and verifies that calling `flush()` after
    /// receiving every frame is idempotent — no duplicate frames
    /// surface, and `receive_frame` returns `NeedMore` once the queue
    /// is drained.
    #[test]
    fn round10_flush_after_receive_is_idempotent() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let sps_rbsp = build_rpl_sps_rbsp(32, 32);
        let pps_rbsp = build_baseline_pps_rbsp();
        // IDR slice (POC 0) — uniform 128.
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0);
        idr_hdr.ue(2);
        idr_hdr.u(1, 0);
        idr_hdr.u(1, 0);
        idr_hdr.u(6, 22);
        idr_hdr.ue(0);
        idr_hdr.ue(0);
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        let mut idr_enc = CabacEncoder::new();
        idr_enc.encode_decision(0, 0, 0);
        idr_enc.encode_decision(0, 0, 0);
        idr_enc.encode_decision(0, 0, 0);
        idr_enc.encode_decision(0, 0, 0);
        idr_enc.encode_decision(0, 0, 0);
        idr_enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&idr_enc.finish());
        // P slice (POC 1).
        let mut p_hdr = BitEmitter::new();
        p_hdr.ue(0);
        p_hdr.ue(1);
        p_hdr.u(8, 1);
        p_hdr.ue(1);
        p_hdr.ue(1);
        p_hdr.u(1, 0);
        p_hdr.ue(1);
        p_hdr.ue(1);
        p_hdr.u(1, 0);
        p_hdr.u(1, 0);
        p_hdr.u(1, 0);
        p_hdr.u(6, 22);
        p_hdr.ue(0);
        p_hdr.ue(0);
        while p_hdr.bit_position() % 8 != 0 {
            p_hdr.u(1, 0);
        }
        let mut p_rbsp = p_hdr.into_bytes();
        let mut p_enc = CabacEncoder::new();
        p_enc.encode_decision(0, 0, 0);
        p_enc.encode_decision(0, 0, 1);
        for _ in 0..3 {
            p_enc.encode_decision(0, 0, 1);
        }
        p_enc.encode_decision(0, 0, 0);
        p_enc.encode_decision(0, 0, 0);
        p_enc.encode_decision(0, 0, 0);
        p_enc.encode_terminate(true);
        p_rbsp.extend_from_slice(&p_enc.finish());

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp));
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp));
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp));
        bs.extend_from_slice(&nal_envelope(0, &p_rbsp));

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        // Drain both frames.
        let _f0 = dec.receive_frame().unwrap();
        let _f1 = dec.receive_frame().unwrap();
        // Now flush() should be a no-op (every DPB entry has output_emitted = true).
        dec.flush().unwrap();
        let next = dec.receive_frame();
        assert!(matches!(next, Err(oxideav_core::Error::NeedMore)));
    }

    /// Round 384: an SPS with `sps_admvp_flag = 1` (affine + hmvp on,
    /// amvr/dmvr/mmvd off) built like [`build_rpl_sps_rbsp`] but with the
    /// §7.3.2.1 admvp-nested flag group present.
    fn build_admvp_sps_rbsp(width: u32, height: u32) -> Vec<u8> {
        let mut sps_body = sps::tests::BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 1); // profile_idc (Main)
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc 4:2:0
        sps_body.ue(width);
        sps_body.ue(height);
        sps_body.ue(0); // bit_depth_luma_minus8
        sps_body.ue(0); // bit_depth_chroma_minus8
        sps_body.u(1, 0); // sps_btt
        sps_body.u(1, 0); // sps_suco
        sps_body.u(1, 1); // sps_admvp ← ON
        sps_body.u(1, 1); //   sps_affine
        sps_body.u(1, 0); //   sps_amvr
        sps_body.u(1, 0); //   sps_dmvr
        sps_body.u(1, 0); //   sps_mmvd
        sps_body.u(1, 1); //   sps_hmvp
        sps_body.u(1, 0); // sps_eipd
        sps_body.u(1, 0); // sps_cm_init
        sps_body.u(1, 0); // sps_iqt
        sps_body.u(1, 0); // sps_addb
        sps_body.u(1, 0); // sps_alf
        sps_body.u(1, 0); // sps_htdf
        sps_body.u(1, 1); // sps_rpl  ← enable
        sps_body.u(1, 1); // sps_pocs ← enable
        sps_body.u(1, 0); // sps_dquant
        sps_body.u(1, 0); // sps_dra
        sps_body.ue(4); // log2_max_pic_order_cnt_lsb_minus4 = 4 → 8 bits
        sps_body.ue(1); // sps_max_dec_pic_buffering_minus1
        sps_body.u(1, 0); // long_term_ref_pics_flag
        sps_body.u(1, 0); // rpl1_same_as_rpl0_flag
        sps_body.ue(0); // num_ref_pic_lists_in_sps[0]
        sps_body.ue(0); // num_ref_pic_lists_in_sps[1]
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        sps_body.into_bytes()
    }

    /// **Round 384 decoder-level ADMVP e2e.** The `sps_admvp_flag == 1`
    /// gate is lifted: a Main-toolset (admvp + affine + hmvp) IDR + P
    /// stream decodes through the public decoder. The P slice's single
    /// CU is a cu_skip merge CU routed through `read_cu_skip_main`
    /// (`affine_flag = 0` bin present because sps_affine is on;
    /// `merge_idx = 0` selects the §8.5.2.3.8 zero-MV fill — no spatial
    /// neighbours, no usable collocated motion from the intra IDR), so
    /// the frame is a zero-MV copy of the grey IDR.
    #[test]
    fn round384_admvp_gate_lifted_idr_p_e2e() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let sps_rbsp = build_admvp_sps_rbsp(32, 32);
        let pps_rbsp = build_baseline_pps_rbsp();

        // IDR slice (POC 0) — grey.
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0);
        idr_hdr.ue(2); // I slice
        idr_hdr.u(1, 0); // no_output_of_prior_pics_flag
        idr_hdr.u(1, 0); // slice_deblocking_filter_flag
        idr_hdr.u(6, 22); // slice_qp
        idr_hdr.ue(0);
        idr_hdr.ue(0);
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        let mut idr_enc = CabacEncoder::new();
        idr_enc.encode_decision(0, 0, 0); // split_cu_flag = 0
        idr_enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
        idr_enc.encode_decision(0, 0, 0); // cbf_luma
        idr_enc.encode_decision(0, 0, 0); // cbf_cb
        idr_enc.encode_decision(0, 0, 0); // cbf_cr
        idr_enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&idr_enc.finish());

        // P slice (POC 1): inline RPL both lists, admvp slice header
        // (temporal_mvp_assigned_flag = 0).
        let mut p_hdr = BitEmitter::new();
        p_hdr.ue(0); // slice_pps_id
        p_hdr.ue(1); // slice_type = P
        p_hdr.u(8, 1); // POC LSB
        p_hdr.ue(1); // RPL L0: num_strp_entries
        p_hdr.ue(1); //   delta_poc_st = 1
        p_hdr.u(1, 0); //   sign → −1 → ref POC 0
        p_hdr.ue(1); // RPL L1: same
        p_hdr.ue(1);
        p_hdr.u(1, 0);
        p_hdr.u(1, 0); // num_ref_idx_active_override_flag
        p_hdr.u(1, 0); // temporal_mvp_assigned_flag (sps_admvp = 1)
        p_hdr.u(1, 0); // slice_deblocking_filter_flag
        p_hdr.u(6, 22); // slice_qp
        p_hdr.ue(0);
        p_hdr.ue(0);
        while p_hdr.bit_position() % 8 != 0 {
            p_hdr.u(1, 0);
        }
        let mut p_rbsp = p_hdr.into_bytes();
        let mut p_enc = CabacEncoder::new();
        p_enc.encode_decision(0, 0, 0); // split_cu_flag = 0
        p_enc.encode_decision(0, 0, 1); // cu_skip_flag = 1
                                        // read_cu_skip_main: sps_mmvd off → no mmvd bin;
                                        // affine gate on (32×32) → affine_flag bin.
        p_enc.encode_decision(0, 0, 0); // affine_flag = 0
        p_enc.encode_decision(0, 0, 0); // merge_idx = 0
        p_enc.encode_decision(0, 0, 0); // cbf_luma
        p_enc.encode_decision(0, 0, 0); // cbf_cb
        p_enc.encode_decision(0, 0, 0); // cbf_cr
        p_enc.encode_terminate(true);
        p_rbsp.extend_from_slice(&p_enc.finish());

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp));
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp));
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp));
        bs.extend_from_slice(&nal_envelope(0, &p_rbsp));

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        let v0 = match f0 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v1 = match f1 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        assert!(v0.planes[0].data.iter().all(|&v| v == 128));
        assert!(
            v1.planes[0].data.iter().all(|&v| v == 128),
            "admvp cu_skip zero-MV merge must copy the grey IDR"
        );
    }

    /// Round 391: `decode_idr_slice` end-to-end with a **Main-shaped
    /// SPS** (`sps_btt_flag = 1`, 32×32 CTU, monochrome): the §7.3.8.3
    /// BTT split syntax parses out of the real SPS fields, the slice
    /// decodes a BT_HOR root → leaf + TT_VER subtree, and the all-VER
    /// no-residual leaves reconstruct uniform mid-grey.
    #[test]
    fn round391_decode_idr_slice_btt_sps_end_to_end() {
        use crate::cabac::CabacEncoder;
        use crate::cabac_init::MainCtxTable;
        // SPS: btt on (CtbLog2SizeY = 5, MinCbLog2SizeY = 2, all diff
        // fields 0), suco off, monochrome 32x32.
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 1); // profile_idc (Main)
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(0); // chroma_format_idc 0 (monochrome)
        sps_body.ue(32); // width
        sps_body.ue(32); // height
        sps_body.ue(0); // bit_depth_luma_minus8
        sps_body.ue(0); // bit_depth_chroma_minus8
        sps_body.u(1, 1); // sps_btt_flag = 1
        sps_body.ue(0); // log2_ctu_size_minus5 → CtbLog2SizeY = 5
        sps_body.ue(0); // log2_min_cb_size_minus2 → MinCbLog2SizeY = 2
        sps_body.ue(0); // log2_diff_ctu_max_14_cb_size
        sps_body.ue(0); // log2_diff_ctu_max_tt_cb_size
        sps_body.ue(0); // log2_diff_min_cb_min_tt_cb_size_minus2
        for _ in 0..12 {
            sps_body.u(1, 0); // suco..dra tool flags all 0
        }
        sps_body.ue(1); // log2_sub_gop_length
        sps_body.ue(1); // max_num_tid0_ref_pics
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        let sps = sps::parse(&sps_body.into_bytes()).unwrap();
        assert!(sps.sps_btt_flag);
        let pps = pps::parse(&build_baseline_pps_rbsp()).unwrap();

        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2); // I slice
        hdr.u(1, 0);
        hdr.u(1, 0); // slice_deblocking_filter_flag = 0
        hdr.u(6, 22);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();

        let t_flag = MainCtxTable::BttSplitFlag as usize;
        let t_dir = MainCtxTable::BttSplitDir as usize;
        let t_type = MainCtxTable::BttSplitType as usize;
        let mut enc = CabacEncoder::new();
        let leaf = |enc: &mut CabacEncoder| {
            enc.encode_decision(0, 0, 1); // intra_pred_mode ...
            enc.encode_decision(0, 0, 1); //   = 2 (INTRA_VER)
            enc.encode_decision(0, 0, 0); //   U terminator
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
        };
        // CTU 32x32 → BT_HOR; top 32x16 leaf; bottom 32x16 → TT_VER
        // with three leaves.
        enc.encode_decision(t_flag, 0, 1);
        enc.encode_decision(t_dir, 0, 0);
        enc.encode_decision(t_type, 0, 0);
        enc.encode_decision(t_flag, 0, 0);
        leaf(&mut enc);
        enc.encode_decision(t_flag, 0, 1);
        enc.encode_decision(t_dir, 0, 1);
        enc.encode_decision(t_type, 0, 1);
        for _ in 0..3 {
            enc.encode_decision(t_flag, 0, 0);
            leaf(&mut enc);
        }
        enc.encode_terminate(true);
        slice_rbsp.extend_from_slice(&enc.finish());

        let (pic, stats) = decode_idr_slice(&sps, &pps, &slice_rbsp).unwrap();
        assert_eq!(stats.split_cu_flag_bins, 0);
        assert_eq!(stats.tree.btt.flag_bins, 6);
        assert_eq!(stats.tree.btt.dir_bins, 2);
        assert_eq!(stats.tree.btt.type_bins, 2);
        assert_eq!(stats.coding_units, 8);
        assert!(pic.y.iter().all(|&v| v == 128));
    }

    /// Round 391: `decode_idr_slice` end-to-end with a **10-bit SPS**
    /// (`bit_depth_luma_minus8 = 2`): the reconstructed picture carries
    /// `bit_depth = 10` and its DC + positive-residual samples exceed
    /// the 8-bit range.
    #[test]
    fn round391_decode_idr_slice_10bit_sps_end_to_end() {
        use crate::cabac::CabacEncoder;
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc 4:2:0
        sps_body.ue(32); // width
        sps_body.ue(32); // height
        sps_body.ue(2); // bit_depth_luma_minus8 = 2 → 10-bit
        sps_body.ue(2); // bit_depth_chroma_minus8 = 2
        for _ in 0..13 {
            sps_body.u(1, 0); // all tool flags 0 (Baseline)
        }
        sps_body.ue(1); // log2_sub_gop_length
        sps_body.ue(1); // max_num_tid0_ref_pics
        sps_body.u(1, 0); // picture_cropping_flag
        sps_body.u(1, 0); // chroma_qp_table_present_flag
        sps_body.u(1, 0); // vui_parameters_present_flag
        sps_body.finish_with_trailing_bits();
        let sps = sps::parse(&sps_body.into_bytes()).unwrap();
        assert_eq!(sps.bit_depth_y(), 10);
        let pps = pps::parse(&build_baseline_pps_rbsp()).unwrap();

        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2); // I slice
        hdr.u(1, 0);
        hdr.u(1, 0);
        hdr.u(6, 30); // slice_qp = 30
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut slice_rbsp = hdr.into_bytes();

        // One 32x32 CTU: split_cu_flag = 1 → four 16x16 dual-tree leaf
        // pairs; the first luma CU carries a DC level of +30, the rest
        // are residual-free.
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // split_cu_flag = 1
        for leaf_idx in 0..4 {
            enc.encode_decision(0, 0, 0); // child split_cu_flag = 0
            enc.encode_decision(0, 0, 0); // intra_pred_mode = 0 (DC)
            if leaf_idx == 0 {
                enc.encode_decision(0, 0, 1); // cbf_luma = 1
                enc.encode_decision(0, 0, 0); // coeff_zero_run = 0
                for _ in 0..29 {
                    enc.encode_decision(0, 0, 1); // coeff_abs_level_minus1 = 29
                }
                enc.encode_decision(0, 0, 0); // U terminator → level 30
                enc.encode_bypass(0); // sign = +
                enc.encode_decision(0, 0, 1); // coeff_last_flag = 1
            } else {
                enc.encode_decision(0, 0, 0); // cbf_luma = 0
            }
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        slice_rbsp.extend_from_slice(&enc.finish());

        let (pic, stats) = decode_idr_slice(&sps, &pps, &slice_rbsp).unwrap();
        assert_eq!(pic.bit_depth, 10);
        assert_eq!(stats.coeff_runs, 1);
        // First 16x16 leaf: DC 512 + non-negative residual (position-
        // dependent magnitudes under the literal eq. 1062 basis; the
        // smallest round to zero) — at least one sample must exceed the
        // 8-bit-impossible 512 and none may drop below it.
        assert!(
            (0..16).any(|j| (0..16).any(|i| pic.y[j * 32 + i] > 512)),
            "10-bit residual leaf must lift samples above the mid-level"
        );
        assert!((0..16).all(|j| (0..16).all(|i| pic.y[j * 32 + i] >= 512)));
        // Residual-free leaves stay at the 10-bit mid-level.
        assert!((0..16).all(|j| (16..32).all(|i| pic.y[j * 32 + i] == 512)));
        assert!(pic.cb.iter().all(|&v| v == 512));
        assert!(pic.cr.iter().all(|&v| v == 512));
    }

    /// Round 416 — a 2×1-tile PPS: 128×64 picture, CTU 64, two 64×64
    /// tiles, `loop_filter_across_tiles_enabled_flag = 0`,
    /// `tile_offset_len_minus1 = 15` (16-bit entry points),
    /// `tile_id_len_minus1 = 0`.
    fn build_two_tile_pps_rbsp() -> Vec<u8> {
        let mut pps_body = BitEmitter::new();
        pps_body.ue(0); // pps_id
        pps_body.ue(0); // sps_id
        pps_body.ue(0); // num_ref_idx_default_active_minus1[0]
        pps_body.ue(0); // num_ref_idx_default_active_minus1[1]
        pps_body.ue(0); // additional_lt_poc_lsb_len
        pps_body.u(1, 0); // rpl1_idx_present_flag
        pps_body.u(1, 0); // single_tile_in_pic_flag = 0
        pps_body.ue(1); // num_tile_columns_minus1 = 1
        pps_body.ue(0); // num_tile_rows_minus1 = 0
        pps_body.u(1, 1); // uniform_tile_spacing_flag
        pps_body.u(1, 0); // loop_filter_across_tiles_enabled_flag = 0
        pps_body.ue(15); // tile_offset_len_minus1 (16-bit entry points)
        pps_body.ue(0); // tile_id_len_minus1
        pps_body.u(1, 0); // explicit_tile_id_flag
        pps_body.u(1, 0); // pic_dra_enabled_flag
        pps_body.u(1, 0); // arbitrary_slice_present_flag
        pps_body.u(1, 0); // constrained_intra_pred_flag
        pps_body.u(1, 0); // cu_qp_delta_enabled_flag
        pps_body.finish_with_trailing_bits();
        pps_body.into_bytes()
    }

    /// Round 416 — one Baseline 64×64 intra CTU as its own §7.4.5 tile
    /// subset: unsplit DC CU, a single luma DC level, quiet chroma.
    fn encode_tile_ctu_subset(abs_minus1: u32) -> Vec<u8> {
        use crate::cabac::CabacEncoder;
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 0); // split_cu_flag = 0 (64×64 leaf)
        enc.encode_decision(0, 0, 0); // intra_pred_mode = 0 (DC)
        enc.encode_decision(0, 0, 1); // cbf_luma = 1
        enc.encode_decision(0, 0, 0); // coeff_zero_run = 0
        for _ in 0..abs_minus1 {
            enc.encode_decision(0, 0, 1);
        }
        enc.encode_decision(0, 0, 0); // U terminator
        enc.encode_bypass(0); // sign +
        enc.encode_decision(0, 0, 1); // coeff_last_flag
        enc.encode_decision(0, 0, 0); // cbf_cb = 0
        enc.encode_decision(0, 0, 0); // cbf_cr = 0
        enc.encode_terminate(true); // end_of_tile_one_bit
        enc.finish()
    }

    /// Round 416 — the full two-tile NAL bitstream (SPS, a 2×1-tile
    /// PPS, a both-tiles IDR and a both-tiles skip-P) plus the two IDR
    /// tile subsets for stitch controls.
    fn build_two_tile_bitstream() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use crate::cabac::CabacEncoder;
        let sps_rbsp = build_baseline_sps_rbsp(128, 64);
        let pps_rbsp = build_two_tile_pps_rbsp();

        // --- IDR slice: both tiles, distinct DC levels per tile. ---
        let subset_a = encode_tile_ctu_subset(149);
        let subset_b = encode_tile_ctu_subset(49);
        let mut idr_hdr = BitEmitter::new();
        idr_hdr.ue(0); // slice_pps_id
        idr_hdr.u(1, 0); // single_tile_in_slice_flag = 0
        idr_hdr.u(1, 0); // first_tile_id (1 bit)
        idr_hdr.u(1, 1); // last_tile_id (1 bit) — arbitrary flag absent
        idr_hdr.ue(2); // slice_type = I
        idr_hdr.u(1, 0); // no_output_of_prior_pics_flag
        idr_hdr.u(1, 0); // slice_deblocking_filter_flag
        idr_hdr.u(6, 51); // slice_qp (51 → visible DC shift)
        idr_hdr.ue(0); // slice_cb_qp_offset
        idr_hdr.ue(0); // slice_cr_qp_offset
                       // §7.3.4 entry point: NumTilesInSlice − 1 = 1 offset, 16 bits.
        idr_hdr.u(16, subset_a.len() as u32 - 1); // entry_point_offset_minus1[0]
        while idr_hdr.bit_position() % 8 != 0 {
            idr_hdr.u(1, 0);
        }
        let mut idr_rbsp = idr_hdr.into_bytes();
        idr_rbsp.extend_from_slice(&subset_a);
        idr_rbsp.extend_from_slice(&subset_b);

        // --- P slice: both tiles, one 64×64 zero-slot skip CU each. ---
        let encode_skip_tile = || -> Vec<u8> {
            let mut enc = CabacEncoder::new();
            enc.encode_decision(0, 0, 0); // split_cu_flag = 0
            enc.encode_decision(0, 0, 1); // cu_skip_flag = 1
            for _ in 0..3 {
                enc.encode_decision(0, 0, 1); // mvp_idx = 3 (zero slot)
            }
            enc.encode_decision(0, 0, 0); // cbf_luma
            enc.encode_decision(0, 0, 0); // cbf_cb
            enc.encode_decision(0, 0, 0); // cbf_cr
            enc.encode_terminate(true);
            enc.finish()
        };
        let p_subset_a = encode_skip_tile();
        let p_subset_b = encode_skip_tile();
        let mut p_hdr = BitEmitter::new();
        p_hdr.ue(0); // slice_pps_id
        p_hdr.u(1, 0); // single_tile_in_slice_flag = 0
        p_hdr.u(1, 0); // first_tile_id
        p_hdr.u(1, 1); // last_tile_id
        p_hdr.ue(1); // slice_type = P
        p_hdr.u(1, 0); // num_ref_idx_active_override_flag
        p_hdr.u(1, 0); // slice_deblocking_filter_flag
        p_hdr.u(6, 51); // slice_qp
        p_hdr.ue(0); // cb offset
        p_hdr.ue(0); // cr offset
        p_hdr.u(16, p_subset_a.len() as u32 - 1); // entry point
        while p_hdr.bit_position() % 8 != 0 {
            p_hdr.u(1, 0);
        }
        let mut p_rbsp = p_hdr.into_bytes();
        p_rbsp.extend_from_slice(&p_subset_a);
        p_rbsp.extend_from_slice(&p_subset_b);

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp)); // SPS
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp)); // PPS
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp)); // IDR
        bs.extend_from_slice(&nal_envelope(0, &p_rbsp)); // NonIDR
        (bs, subset_a, subset_b)
    }

    /// Round 416 — end-to-end **multi-tile** decode through the
    /// registered decoder: SPS + a 2×1-tile PPS + an IDR slice spanning
    /// both tiles (tile fields + a 16-bit entry point in the header,
    /// two CABAC subsets) + a P slice of per-tile zero-MV skip CUs.
    /// The IDR frame must stitch the two standalone single-tile decodes
    /// bit-exactly (§6.4.1 keeps tile 1's intra refs off tile 0), and
    /// the P frame must equal the IDR frame.
    #[test]
    fn round416_make_decoder_decodes_two_tile_idr_plus_p() {
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let (bs, subset_a, subset_b) = build_two_tile_bitstream();
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let f0 = dec.receive_frame().unwrap();
        let f1 = dec.receive_frame().unwrap();
        let v0 = match f0 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        let v1 = match f1 {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("not video"),
        };
        // Standalone controls: each tile subset decoded as its own
        // 64×64 single-tile picture must stitch the IDR frame exactly.
        let walk = slice_data::SliceWalkInputs {
            pic_width: 64,
            pic_height: 64,
            ctb_log2_size_y: 6,
            min_cb_log2_size_y: 2,
            max_tb_log2_size_y: 6,
            chroma_format_idc: 1,
            cu_qp_delta_enabled: false,
            ..Default::default()
        };
        let decode_in = slice_data::SliceDecodeInputs {
            slice_qp: 51,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            enable_deblock: false,
            ..Default::default()
        };
        let (pic_a, _) = slice_data::decode_baseline_idr_slice(&subset_a, walk, decode_in).unwrap();
        let (pic_b, _) = slice_data::decode_baseline_idr_slice(&subset_b, walk, decode_in).unwrap();
        assert_ne!(pic_a.y[0], pic_b.y[0], "tiles must differ");
        let y0 = &v0.planes[0].data;
        for j in 0..64usize {
            for i in 0..64usize {
                assert_eq!(
                    y0[j * 128 + i] as u16,
                    pic_a.y[j * 64 + i],
                    "tile 0 mismatch at ({i}, {j})"
                );
                assert_eq!(
                    y0[j * 128 + 64 + i] as u16,
                    pic_b.y[j * 64 + i],
                    "tile 1 mismatch at ({i}, {j})"
                );
            }
        }
        // The P frame zero-MV-copies the IDR per tile.
        assert_eq!(v0.planes[0].data, v1.planes[0].data);
        assert_eq!(v0.planes[1].data, v1.planes[1].data);
        assert_eq!(v0.planes[2].data, v1.planes[2].data);
    }

    /// Round 416 — mutation gate over the multi-tile stream: every
    /// single-byte corruption of the two-tile IDR+P bitstream (two XOR
    /// patterns per position — a low-bit flip and a full invert) must
    /// come back as a clean `Err` or a decoded frame, never a panic,
    /// out-of-bounds access or runaway allocation. Exercises the tile
    /// plumbing's error paths: PPS tile geometry, slice-header tile
    /// ids, the entry-point subsets and the per-tile CABAC terminates.
    #[test]
    fn round416_two_tile_stream_mutation_never_panics() {
        use oxideav_core::{CodecParameters, Packet, TimeBase};
        let (bs, _, _) = build_two_tile_bitstream();
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        for pos in 0..bs.len() {
            for pat in [0x01u8, 0xFF] {
                let mut mutated = bs.clone();
                mutated[pos] ^= pat;
                let mut dec = decoder::make_decoder(&params).unwrap();
                let pkt = Packet::new(0, TimeBase::new(1, 90_000), mutated).with_pts(0);
                // Every outcome except a panic is acceptable: a clean
                // parse error from send_packet, an empty frame queue,
                // or a (differently-)decoded picture.
                if dec.send_packet(&pkt).is_ok() {
                    while dec.receive_frame().is_ok() {}
                }
            }
        }
    }

    /// Round 391: full NAL-stream 10-bit decode through the registered
    /// decoder factory. SPS (`bit_depth_luma_minus8 = 2`) + PPS + IDR
    /// wrapped as length-prefixed NALs; `receive_frame` must emit
    /// little-endian 16-bit planes (Yuv420P10Le-family layout) whose
    /// residual-free samples sit at the 10-bit mid-level 512.
    #[test]
    fn round391_make_decoder_decodes_10bit_idr_to_le16_frame() {
        use crate::cabac::CabacEncoder;
        use oxideav_core::{CodecParameters, Packet, TimeBase};

        // 10-bit 32x32 4:2:0 Baseline SPS.
        let mut sps_body = BitEmitter::new();
        sps_body.ue(0); // sps_id
        sps_body.u(8, 0); // profile_idc
        sps_body.u(8, 30); // level_idc
        sps_body.u(32, 0); // toolset_idc_h
        sps_body.u(32, 0); // toolset_idc_l
        sps_body.ue(1); // chroma_format_idc 4:2:0
        sps_body.ue(32);
        sps_body.ue(32);
        sps_body.ue(2); // bit_depth_luma_minus8 = 2
        sps_body.ue(2); // bit_depth_chroma_minus8 = 2
        for _ in 0..13 {
            sps_body.u(1, 0);
        }
        sps_body.ue(1); // log2_sub_gop_length
        sps_body.ue(1); // max_num_tid0_ref_pics
        sps_body.u(1, 0);
        sps_body.u(1, 0);
        sps_body.u(1, 0);
        sps_body.finish_with_trailing_bits();
        let sps_rbsp = sps_body.into_bytes();
        let pps_rbsp = build_baseline_pps_rbsp();

        let mut hdr = BitEmitter::new();
        hdr.ue(0);
        hdr.ue(2);
        hdr.u(1, 0);
        hdr.u(1, 0);
        hdr.u(6, 22);
        hdr.ue(0);
        hdr.ue(0);
        while hdr.bit_position() % 8 != 0 {
            hdr.u(1, 0);
        }
        let mut idr_rbsp = hdr.into_bytes();
        // One 32x32 CTU: quad split into four residual-free DC leaves.
        let mut enc = CabacEncoder::new();
        enc.encode_decision(0, 0, 1); // split_cu_flag = 1
        for _ in 0..4 {
            enc.encode_decision(0, 0, 0); // child split = 0
            enc.encode_decision(0, 0, 0); // intra_pred_mode = 0
            enc.encode_decision(0, 0, 0); // cbf_luma = 0
            enc.encode_decision(0, 0, 0); // cbf_cb = 0
            enc.encode_decision(0, 0, 0); // cbf_cr = 0
        }
        enc.encode_terminate(true);
        idr_rbsp.extend_from_slice(&enc.finish());

        fn nal_envelope(nut: u8, rbsp: &[u8]) -> Vec<u8> {
            let nut_plus1: u16 = (nut as u16) + 1;
            let mut hdr_word: u16 = 0;
            hdr_word |= (nut_plus1 & 0x3F) << 9;
            let hdr = [(hdr_word >> 8) as u8, (hdr_word & 0xFF) as u8];
            let nal_len = (hdr.len() + rbsp.len()) as u32;
            let mut out = Vec::new();
            out.extend_from_slice(&nal_len.to_be_bytes());
            out.extend_from_slice(&hdr);
            out.extend_from_slice(rbsp);
            out
        }
        let mut bs = Vec::new();
        bs.extend_from_slice(&nal_envelope(24, &sps_rbsp));
        bs.extend_from_slice(&nal_envelope(25, &pps_rbsp));
        bs.extend_from_slice(&nal_envelope(1, &idr_rbsp));

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = decoder::make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), bs).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let video = match frame {
            oxideav_core::Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };
        assert_eq!(video.planes.len(), 3);
        // Byte stride = 2 × width; every sample is LE 512 (0x00, 0x02).
        assert_eq!(video.planes[0].stride, 64);
        assert_eq!(video.planes[0].data.len(), 32 * 32 * 2);
        assert!(video.planes[0]
            .data
            .chunks_exact(2)
            .all(|c| u16::from_le_bytes([c[0], c[1]]) == 512));
        assert_eq!(video.planes[1].stride, 32);
        assert_eq!(video.planes[1].data.len(), 16 * 16 * 2);
        assert!(video.planes[1]
            .data
            .chunks_exact(2)
            .all(|c| u16::from_le_bytes([c[0], c[1]]) == 512));
    }
}
