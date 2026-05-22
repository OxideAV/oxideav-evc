//! Registry glue + minimal decode pipeline for the EVC crate.
//!
//! Round-9 status: a working decoder for **Baseline-profile** bitstreams
//! that satisfy:
//!
//! * 8-bit luma + chroma,
//! * `slice_deblocking_filter_flag = 0`,
//! * every CU has `cbf_luma == cbf_cb == cbf_cr == 0` (pure intra/inter
//!   prediction with no residual; round-5 wires real residual decoding),
//! * Inter MVs land on the Baseline 1/4-pel grid (sub-pel phases 4, 8, 12
//!   for luma; 4, 8, 12, 16, 20, 24, 28 for chroma).
//!
//! Round-9 lifts the round-4 single-reference constraint: an in-memory
//! DPB keeps every previously-decoded short-term reference picture
//! indexed by POC, and at each non-IDR slice the decoder builds the
//! L0 / L1 lists by walking the slice's `ref_pic_list_struct()` deltas
//! (relative to `slice_pic_order_cnt_lsb`). Each inter CU's
//! `RefIdxL0` / `RefIdxL1` then resolves to the right DPB entry.
//! Output ordering follows POC, not coding order.
//!
//! Anything else (non-Baseline, 10-bit, deblocked, residuals present,
//! sub-pel outside Baseline grid) bubbles up as `Error::Unsupported`.
//! The decoder consumes length-prefixed NAL units (Annex B raw
//! bitstream framing) per ISO/IEC 23094-1.

use std::collections::VecDeque;

use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::alf::{self, AlfData};
use crate::dra::{self, DraData};
use crate::inter::RefPictureView;
use crate::nal::{iter_length_prefixed, NalUnitType};
use crate::picture::YuvPicture;
use crate::pps::{self, Pps};
use crate::slice_data::{InterDecodeInputs, SliceDecodeInputs, SliceWalkInputs};
use crate::sps::{self, Sps};
use crate::CODEC_ID_STR;

/// Decoded-picture buffer entry (round-9 / round-10). Each slot keeps
/// the reconstructed picture, its POC and the original presentation
/// timestamp so frames can be re-ordered by POC before emission.
#[derive(Clone, Debug)]
struct DpbEntry {
    pic: YuvPicture,
    poc: i32,
    pts: Option<i64>,
    /// `true` while the picture is still needed as a reference. Round-9
    /// keeps every IDR + short-term ref alive until a fresh IDR flushes
    /// the buffer. Long-term references and explicit RPL-driven
    /// removal are deferred — the field is parked here for round-11's
    /// sliding-window unmark step.
    #[allow(dead_code)]
    used_for_reference: bool,
    /// Round-10: whether this DPB entry has already been pushed to the
    /// `out` output queue. Stops `flush()` from re-emitting frames the
    /// caller has already received.
    output_emitted: bool,
}

/// Build the round-3 decoder for the registry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(EvcDecoder::new()))
}

/// Public-but-internal type that callers normally see via the `Decoder`
/// trait. Wraps the per-stream parameter-set cache, the round-9 DPB and
/// the POC-ordered output queue.
pub struct EvcDecoder {
    codec_id: CodecId,
    sps: Option<Sps>,
    pps: Option<Pps>,
    pending_pts: Option<i64>,
    /// Frames waiting to be emitted, sorted by ascending POC.
    out: VecDeque<VideoFrame>,
    /// Parallel POC tracker for `out` so insertion stays sorted even
    /// when the bitstream coding order differs from display order.
    out_pocs: VecDeque<i32>,
    /// Round-9 DPB: every short-term reference picture (IDR + non-IDR)
    /// indexed by POC. Capped at [`MAX_DPB_ENTRIES`]; eviction is by
    /// lowest POC when a fresh picture would overflow.
    dpb: Vec<DpbEntry>,
    /// `PicOrderCntMsb` for the currently-decoded sequence (§8.3.1).
    /// Reset to 0 at every IDR; advanced when a non-IDR's
    /// `slice_pic_order_cnt_lsb` wraps around relative to the previous
    /// picture's POC LSB.
    poc_msb: i32,
    /// `prevPicOrderCntLsb` from §8.3.1: the POC LSB of the most
    /// recently decoded reference picture in coding order.
    prev_poc_lsb: i32,
    /// Round-11: cached ALF data from the most recently parsed ALF APS.
    /// Keyed by `adaptation_parameter_set_id` (5-bit, 0..=31). Only
    /// slot 0 is consulted for now (round-11 per-CTU selection deferred).
    alf_aps: Option<AlfData>,
    /// Round-11: cached DRA data from the most recently parsed DRA APS.
    dra_aps: Option<DraData>,
}

const MAX_DPB_ENTRIES: usize = 16;

impl EvcDecoder {
    pub fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            sps: None,
            pps: None,
            pending_pts: None,
            out: VecDeque::new(),
            out_pocs: VecDeque::new(),
            dpb: Vec::new(),
            poc_msb: 0,
            prev_poc_lsb: 0,
            alf_aps: None,
            dra_aps: None,
        }
    }

    /// Find the DPB entry with the given POC, returning a borrow. None
    /// when no entry matches (caller may signal a malformed bitstream).
    fn dpb_find(&self, poc: i32) -> Option<&DpbEntry> {
        self.dpb.iter().find(|e| e.poc == poc)
    }

    /// Insert a freshly-decoded picture into the DPB, evicting the
    /// lowest-POC reference when at capacity.
    fn dpb_insert(&mut self, entry: DpbEntry) {
        if self.dpb.len() >= MAX_DPB_ENTRIES {
            // Evict the entry with the smallest POC (oldest in display
            // order). Round-9 doesn't track sliding-window LTRP yet so
            // any non-LTRP entry is fair game.
            let mut min_idx = 0;
            for (i, e) in self.dpb.iter().enumerate() {
                if e.poc < self.dpb[min_idx].poc {
                    min_idx = i;
                }
            }
            self.dpb.remove(min_idx);
        }
        self.dpb.push(entry);
    }

    /// Flush all reference pictures (IDR boundary). Output queue is
    /// untouched — already-emitted POCs stay where they are.
    fn dpb_flush(&mut self) {
        self.dpb.clear();
        self.poc_msb = 0;
        self.prev_poc_lsb = 0;
    }

    /// Compute a full POC from a slice's `slice_pic_order_cnt_lsb` per
    /// §8.3.1 (simplified for the Baseline single-layer case): if the
    /// LSB has wrapped relative to `prev_poc_lsb`, advance MSB by
    /// `MaxPicOrderCntLsb`. The formal pred is symmetric (also handles
    /// regress) which we mirror here.
    fn derive_poc(&self, slice_lsb: i32, max_poc_lsb: i32) -> i32 {
        let half = max_poc_lsb >> 1;
        let prev_lsb = self.prev_poc_lsb;
        let mut poc_msb = self.poc_msb;
        if slice_lsb < prev_lsb && (prev_lsb - slice_lsb) >= half {
            poc_msb = poc_msb.wrapping_add(max_poc_lsb);
        } else if slice_lsb > prev_lsb && (slice_lsb - prev_lsb) > half {
            poc_msb = poc_msb.wrapping_sub(max_poc_lsb);
        }
        poc_msb + slice_lsb
    }
}

impl Default for EvcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for EvcDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let nals = iter_length_prefixed(&packet.data)
            .map_err(|e| Error::invalid(format!("evc decoder: NAL framing: {e}")))?;
        for nal in nals {
            match nal.header.nal_unit_type {
                NalUnitType::Sps => {
                    let sps = sps::parse(nal.rbsp())
                        .map_err(|e| Error::invalid(format!("evc decoder: SPS parse: {e}")))?;
                    self.sps = Some(sps);
                }
                NalUnitType::Pps => {
                    let pps = pps::parse(nal.rbsp())
                        .map_err(|e| Error::invalid(format!("evc decoder: PPS parse: {e}")))?;
                    self.pps = Some(pps);
                }
                NalUnitType::Idr => {
                    let sps = self
                        .sps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: IDR slice before SPS"))?
                        .clone();
                    let pps = self
                        .pps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: IDR slice before PPS"))?
                        .clone();
                    let (mut pic, _stats) = crate::decode_idr_slice(&sps, &pps, nal.rbsp())?;
                    // §8.3.1: IDR resets POC to 0, flushes the DPB.
                    self.dpb_flush();
                    // Round-11: ALF + DRA post-filter pass.
                    self.apply_post_filters(&mut pic, &sps);
                    let entry = DpbEntry {
                        pic,
                        poc: 0,
                        pts: packet.pts,
                        used_for_reference: true,
                        output_emitted: false,
                    };
                    self.prev_poc_lsb = 0;
                    self.poc_msb = 0;
                    self.dpb_insert(entry);
                    self.enqueue_for_output(0);
                }
                NalUnitType::NonIdr => {
                    let sps = self
                        .sps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: NonIDR slice before SPS"))?
                        .clone();
                    let pps = self
                        .pps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: NonIDR slice before PPS"))?
                        .clone();
                    let (mut pic, poc) = self.decode_non_idr(&sps, &pps, nal.rbsp())?;
                    // Round-11: ALF + DRA post-filter pass.
                    self.apply_post_filters(&mut pic, &sps);
                    let entry = DpbEntry {
                        pic,
                        poc,
                        pts: packet.pts,
                        used_for_reference: true,
                        output_emitted: false,
                    };
                    self.dpb_insert(entry);
                    self.enqueue_for_output(poc);
                }
                NalUnitType::Aps => {
                    // Round-11: parse the APS and cache ALF / DRA data.
                    // Errors in APS parsing are non-fatal — the filter simply
                    // won't fire for this sequence.
                    let rbsp = nal.rbsp();
                    if let Ok(aps) = crate::aps::parse(rbsp) {
                        if aps.is_alf() && !aps.payload_raw.is_empty() {
                            if let Ok(alf_data) = alf::parse_alf_data(&aps.payload_raw) {
                                self.alf_aps = Some(alf_data);
                            }
                        } else if aps.is_dra() && !aps.payload_raw.is_empty() {
                            if let Ok(dra_data) = dra::parse_dra_data(&aps.payload_raw) {
                                self.dra_aps = Some(dra_data);
                            }
                        }
                    }
                }
                _ => {
                    // SEI / FD / etc. — skipped.
                }
            }
        }
        self.pending_pts = packet.pts;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.out.pop_front() {
            Some(v) => {
                self.out_pocs.pop_front();
                Ok(Frame::Video(v))
            }
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Round-10 drain: every DPB entry that hasn't yet been pushed
        // to the output queue is emitted now, in ascending POC order.
        // Pictures already in `out` (the typical case for low-delay GOPs
        // where decode order == display order) stay in place and are
        // not duplicated. After flush(), the next `receive_frame` calls
        // will pull every reconstructed picture in display order.
        self.drain_dpb_to_output();
        Ok(())
    }
}

impl EvcDecoder {
    /// Push the freshly-decoded picture at `poc` to the output queue
    /// in POC order. Round-9 keeps a parallel `out_pocs` buffer so the
    /// queue stays sorted by POC even when bitstream coding-order
    /// differs from display-order (e.g. B-pyramid GOPs). The frame is
    /// inserted at the unique position where every earlier frame has a
    /// smaller POC and every later frame has a greater POC. Round-10
    /// marks the corresponding DPB entry as `output_emitted = true` so
    /// `flush()` doesn't re-push it.
    fn enqueue_for_output(&mut self, poc: i32) {
        let (frame, pts) = match self.dpb_find(poc) {
            Some(e) => (picture_to_video_frame(&e.pic, e.pts), e.pts),
            None => return,
        };
        let _ = pts; // silence unused
        let pos = self.out_pocs.iter().position(|&p| p > poc);
        match pos {
            Some(i) => {
                self.out.insert(i, frame);
                self.out_pocs.insert(i, poc);
            }
            None => {
                self.out.push_back(frame);
                self.out_pocs.push_back(poc);
            }
        }
        if let Some(entry) = self.dpb.iter_mut().find(|e| e.poc == poc) {
            entry.output_emitted = true;
        }
    }

    /// Round-10 `flush()` drain: emit every DPB entry that hasn't yet
    /// been queued for output, in ascending POC order. Pictures already
    /// in `out` (the typical case for low-delay GOPs) are skipped so
    /// the caller never sees the same frame twice.
    ///
    /// Used by [`Decoder::flush`] but also exposed as a helper for
    /// tests that want to verify the drain path independently.
    fn drain_dpb_to_output(&mut self) {
        // Sort indices by POC so we emit in display order.
        let mut idxs: Vec<usize> = (0..self.dpb.len())
            .filter(|&i| !self.dpb[i].output_emitted)
            .collect();
        idxs.sort_by_key(|&i| self.dpb[i].poc);
        for i in idxs {
            let poc = self.dpb[i].poc;
            self.enqueue_for_output(poc);
        }
    }

    /// Decode a NonIDR slice end-to-end. Returns the reconstructed
    /// picture and its POC. The DPB-resolved L0 / L1 reference lists
    /// are built from the slice's `ref_pic_list_struct()` deltas,
    /// applied to the slice's POC.
    fn decode_non_idr(
        &mut self,
        sps: &Sps,
        pps: &Pps,
        slice_nal_rbsp: &[u8],
    ) -> Result<(YuvPicture, i32)> {
        if sps.sps_btt_flag
            || sps.sps_suco_flag
            || sps.sps_admvp_flag
            || sps.sps_eipd_flag
            || sps.sps_addb_flag
            || sps.sps_dquant_flag
            || sps.sps_ats_flag
            || sps.sps_adcc_flag
            || sps.sps_cm_init_flag
        {
            return Err(Error::unsupported(
                "evc decoder: P/B requires Baseline-profile toolset (round-9 adds DPB + POC)",
            ));
        }
        // Round 95: sps_ibc_flag is no longer part of the unsupported gate.
        // The §7.3.8.4 IBC branch is surfaced inside the per-CU inter
        // walker (`decode_inter_coding_unit`) so non-IDR (P/B) slices with
        // IBC-enabled SPS now decode the `ibc_flag` syntax element + the
        // §8.6.1 IBC pipeline symmetrically to the IDR path lifted in
        // round 90.
        // sps_alf_flag and sps_dra_flag are handled by the round-11 post-filter
        // pipeline — they no longer gate the Unsupported path.
        if !pps.single_tile_in_pic_flag {
            return Err(Error::unsupported(
                "evc decoder: P/B requires single_tile_in_pic_flag == 1",
            ));
        }
        let ctx = crate::slice_header::SliceParseContext {
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
        let header = crate::slice_header::parse(slice_nal_rbsp, NalUnitType::NonIdr, &ctx)?;
        let slice_is_b = match header.slice_type {
            crate::slice_header::SliceType::B => true,
            crate::slice_header::SliceType::P => false,
            crate::slice_header::SliceType::I => {
                return Err(Error::invalid(
                    "evc decoder: NonIDR slice cannot be slice_type == I",
                ));
            }
        };
        let num_ref_idx_active_minus1_l0 = header.num_ref_idx_active_minus1[0];
        let num_ref_idx_active_minus1_l1 = header.num_ref_idx_active_minus1[1];
        let slice_deblocking_filter_flag = header.slice_deblocking_filter_flag;
        let slice_qp = header.slice_qp;
        let slice_cb_qp_offset = header.slice_cb_qp_offset;
        let slice_cr_qp_offset = header.slice_cr_qp_offset;

        // Round-9 POC derivation (§8.3.1).
        let max_poc_lsb = if sps.sps_pocs_flag {
            1i32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
        } else {
            1
        };
        let slice_poc_lsb = header.slice_pic_order_cnt_lsb as i32;
        let poc = if sps.sps_pocs_flag {
            self.derive_poc(slice_poc_lsb, max_poc_lsb)
        } else {
            // No POC signalling — fall back to coding-order as POC.
            self.dpb.iter().map(|e| e.poc).max().unwrap_or(0) + 1
        };

        // Round-9 reference-list construction. Per §8.3.5: for each list
        // i ∈ {0, 1}, walk the RPL entries (either inline or SPS-resident);
        // each STRP entry contributes a POC delta (positive or negative)
        // applied to the current slice's POC; LTRPs would resolve via
        // poc_lsb_lt (deferred). The resulting POC is matched against
        // the DPB; a missing match makes the slice undecodable.
        let n_active_l0 = (num_ref_idx_active_minus1_l0 + 1) as usize;
        let n_active_l1 = if slice_is_b {
            (num_ref_idx_active_minus1_l1 + 1) as usize
        } else {
            0
        };
        let pocs_l0 = if sps.sps_rpl_flag {
            let rpl_l0 = self.resolve_slice_rpl(&header, sps, 0)?;
            self.build_ref_pocs(&rpl_l0, poc, n_active_l0, max_poc_lsb)?
        } else {
            // §8.3.5 round-9 implicit fallback: with no per-slice or
            // SPS RPL signalling, use the highest-POC decoded picture
            // as the single L0 entry (low-delay coding-order GOP).
            self.implicit_ref_pocs(n_active_l0)?
        };
        let pocs_l1 = if slice_is_b {
            if sps.sps_rpl_flag {
                let rpl_l1 = self.resolve_slice_rpl(&header, sps, 1)?;
                self.build_ref_pocs(&rpl_l1, poc, n_active_l1, max_poc_lsb)?
            } else {
                self.implicit_ref_pocs(n_active_l1)?
            }
        } else {
            Vec::new()
        };

        // Re-run the parse on a counting BitReader to discover where the
        // slice header ends so slice_data can pick up at the next byte.
        let mut br = crate::bitreader::BitReader::new(slice_nal_rbsp);
        crate::slice_header::parse_consume(&mut br, NalUnitType::NonIdr, &ctx)?;
        br.align_to_byte();
        let consumed_bits = br.bit_position();
        if consumed_bits % 8 != 0 {
            return Err(Error::invalid("evc decoder: slice header not byte-aligned"));
        }
        let consumed_bytes = (consumed_bits / 8) as usize;
        if consumed_bytes >= slice_nal_rbsp.len() {
            return Err(Error::invalid(
                "evc decoder: no slice_data bytes after NonIDR header",
            ));
        }
        let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
        let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
        let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
        let max_tb_log2_size_y = ctb_log2_size_y.min(5);
        let log2_max_ibc_cand_size = sps.log2_max_ibc_cand_size().unwrap_or(0);
        let walk = SliceWalkInputs {
            pic_width: sps.pic_width_in_luma_samples,
            pic_height: sps.pic_height_in_luma_samples,
            ctb_log2_size_y,
            min_cb_log2_size_y,
            max_tb_log2_size_y,
            chroma_format_idc: sps.chroma_format_idc,
            cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
            sps_ibc_flag: sps.sps_ibc_flag,
            log2_max_ibc_cand_size,
        };
        let decode = SliceDecodeInputs {
            slice_qp: slice_qp as i32,
            bit_depth_luma: sps.bit_depth_y(),
            bit_depth_chroma: sps.bit_depth_c(),
            enable_deblock: slice_deblocking_filter_flag,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            sps_ibc_flag: sps.sps_ibc_flag,
            log2_max_ibc_cand_size,
        };
        // Resolve POCs to DPB views. This consumes immutable borrows
        // of `self.dpb`, so we collect into an owned Vec<RefPictureView>
        // first.
        let ref_list_l0 = self.dpb_views_for_pocs(&pocs_l0)?;
        let ref_list_l1 = if slice_is_b {
            self.dpb_views_for_pocs(&pocs_l1)?
        } else {
            Vec::new()
        };
        let inputs = InterDecodeInputs {
            walk,
            decode,
            slice_is_b,
            num_ref_idx_active_minus1_l0,
            num_ref_idx_active_minus1_l1,
            ref_list_l0: &ref_list_l0,
            ref_list_l1: &ref_list_l1,
        };
        let (pic, _stats) =
            crate::slice_data::decode_baseline_inter_slice(slice_data_bytes, inputs)?;

        // Update §8.3.1 prev-LSB tracker for the next non-IDR slice.
        if sps.sps_pocs_flag {
            // Walk poc back into msb / lsb decomposition.
            let lsb = ((poc % max_poc_lsb) + max_poc_lsb) % max_poc_lsb;
            self.poc_msb = poc - lsb;
            self.prev_poc_lsb = lsb;
        }

        Ok((pic, poc))
    }

    /// Resolve the slice's RPL for list `list_x` (0 or 1) into a concrete
    /// `RefPicListStruct`. Either inline (`slice_rpl[list_x] = Some`) or
    /// indirected through the SPS via `ref_pic_list_idx[list_x]`.
    fn resolve_slice_rpl(
        &self,
        header: &crate::slice_header::SliceHeader,
        sps: &Sps,
        list_x: usize,
    ) -> Result<crate::rpl::RefPicListStruct> {
        if let Some(ref rpl) = header.slice_rpl[list_x] {
            return Ok(rpl.clone());
        }
        // SPS-resident RPL: `ref_pic_list_idx[list_x]` selects an entry.
        let idx = header.ref_pic_list_idx[list_x] as usize;
        let sps_rpls = if list_x == 0 {
            &sps.ref_pic_list_structs_l0
        } else {
            &sps.ref_pic_list_structs_l1
        };
        sps_rpls.get(idx).cloned().ok_or_else(|| {
            Error::invalid(format!(
                "evc decoder: ref_pic_list_idx[{list_x}] = {idx} but SPS has {} RPLs",
                sps_rpls.len()
            ))
        })
    }

    /// Implicit-RPL fallback for streams with `sps_rpl_flag == 0`:
    /// repeat the highest-POC DPB entry `n_active` times. This matches
    /// the low-delay coding-order GOP (one reference, no reordering)
    /// that round-4 fixtures use.
    fn implicit_ref_pocs(&self, n_active: usize) -> Result<Vec<i32>> {
        let max_poc = self
            .dpb
            .iter()
            .map(|e| e.poc)
            .max()
            .ok_or_else(|| Error::invalid("evc decoder: P/B slice with empty DPB"))?;
        Ok(vec![max_poc; n_active])
    }

    /// Convert a list of resolved POCs into borrowed [`RefPictureView`]s
    /// from the DPB. Bails with `Error::Invalid` if any POC isn't in
    /// the buffer.
    fn dpb_views_for_pocs(&self, pocs: &[i32]) -> Result<Vec<RefPictureView<'_>>> {
        let mut out = Vec::with_capacity(pocs.len());
        for &poc in pocs {
            let entry = self.dpb_find(poc).ok_or_else(|| {
                Error::invalid(format!(
                    "evc decoder: reference picture POC {poc} not in DPB"
                ))
            })?;
            let pic = &entry.pic;
            out.push(RefPictureView {
                y: &pic.y,
                cb: &pic.cb,
                cr: &pic.cr,
                width: pic.width,
                height: pic.height,
                y_stride: pic.y_stride(),
                c_stride: pic.c_stride(),
                chroma_format_idc: pic.chroma_format_idc,
            });
        }
        Ok(out)
    }

    /// Round-11 post-filter pass: apply ALF (§8.9) then DRA (§8.10) to
    /// a decoded picture if the corresponding APS has been cached and the
    /// SPS flags indicate they are active. This is called after deblocking
    /// for every IDR and non-IDR slice.
    fn apply_post_filters(&self, pic: &mut YuvPicture, sps: &Sps) {
        let bd_y = sps.bit_depth_y();
        let bd_c = sps.bit_depth_c();
        if sps.sps_alf_flag {
            if let Some(ref alf_data) = self.alf_aps {
                alf::apply_alf(pic, alf_data, bd_y, bd_c);
            }
        }
        if sps.sps_dra_flag {
            if let Some(ref dra_data) = self.dra_aps {
                dra::apply_dra(pic, dra_data, bd_y, bd_c);
            }
        }
    }
}

impl EvcDecoder {
    /// Walk an `RefPicListStruct` to produce up to `n_active` reference
    /// POCs per §8.3.5. Each STRP entry contributes a running-sum POC
    /// (delta added to the prior entry's POC, with the slice's current
    /// POC seeding the chain). Round-10 also resolves LTRP entries by
    /// matching the entry's `poc_lsb_lt` against `(POC & (max_poc_lsb − 1))`
    /// for every DPB entry; the matching DPB POC is the LTRP's POC.
    ///
    /// LTRP resolution does NOT advance the running STRP delta sum:
    /// per §8.3.2 / §8.3.5 the spec defines the LTRP slot independently
    /// (the POC carried by the LTRP entry is absolute, not relative to
    /// the previous entry).
    fn build_ref_pocs(
        &self,
        rpl: &crate::rpl::RefPicListStruct,
        cur_poc: i32,
        n_active: usize,
        max_poc_lsb: i32,
    ) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(n_active);
        let mut running = cur_poc;
        for entry in rpl.entries.iter().take(n_active) {
            match entry {
                crate::rpl::RefPicListEntry::Strp { .. } => {
                    // STRP delta-POC is signed per §7.4.8 eq. 124.
                    let delta = entry.signed_delta_poc().expect("STRP carries signed delta");
                    running += delta;
                    out.push(running);
                }
                crate::rpl::RefPicListEntry::Ltrp { poc_lsb_lt } => {
                    // §8.3.2 LTRP marking: match the entry's poc_lsb_lt
                    // against (poc & mask) for each DPB entry. The first
                    // matching long-term-marked entry (or any entry if
                    // none have been explicitly marked LT yet) is the
                    // resolved POC. round-10 keeps every IDR + non-IDR
                    // picture marked as `used_for_reference == true`,
                    // so we accept any matching POC.
                    let mask = max_poc_lsb - 1;
                    let lsb = (*poc_lsb_lt as i32) & mask;
                    let matched =
                        self.dpb
                            .iter()
                            .find(|e| (e.poc & mask) == lsb)
                            .ok_or_else(|| {
                                Error::invalid(format!(
                                    "evc decoder: LTRP poc_lsb_lt {lsb} not in DPB"
                                ))
                            })?;
                    out.push(matched.poc);
                }
            }
        }
        if out.len() < n_active {
            return Err(Error::invalid(format!(
                "evc decoder: RPL has {} entries but n_active = {n_active}",
                out.len()
            )));
        }
        Ok(out)
    }
}

fn picture_to_video_frame(pic: &crate::picture::YuvPicture, pts: Option<i64>) -> VideoFrame {
    let y_stride = pic.y_stride();
    let c_stride = pic.c_stride();
    VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: y_stride,
                data: pic.y.clone(),
            },
            VideoPlane {
                stride: c_stride,
                data: pic.cb.clone(),
            },
            VideoPlane {
                stride: c_stride,
                data: pic.cr.clone(),
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `derive_poc` mirrors §8.3.1 wrap detection: when the new LSB is
    /// less than the previous LSB by more than half of `MaxPicOrderCntLsb`,
    /// the MSB advances by `MaxPicOrderCntLsb`.
    #[test]
    fn derive_poc_wraps_on_lsb_rollover() {
        let mut dec = EvcDecoder::new();
        // 8-bit POC LSB → MaxPicOrderCntLsb = 256.
        let max = 256;
        // Start at POC 100.
        dec.poc_msb = 0;
        dec.prev_poc_lsb = 100;
        // Next slice LSB = 110 → no wrap. POC = 110.
        assert_eq!(dec.derive_poc(110, max), 110);
        // Next slice LSB = 5 (after prev=200) → wrap. POC = 256 + 5 = 261.
        dec.prev_poc_lsb = 200;
        assert_eq!(dec.derive_poc(5, max), 261);
        // Next slice LSB = 200 (after prev=5) → reverse wrap. POC = -56.
        dec.prev_poc_lsb = 5;
        assert_eq!(dec.derive_poc(200, max), -56);
    }

    /// DPB inserts up to MAX_DPB_ENTRIES; eviction picks the lowest POC.
    #[test]
    fn dpb_evicts_lowest_poc_at_capacity() {
        use crate::picture::YuvPicture;
        let mut dec = EvcDecoder::new();
        for i in 0..MAX_DPB_ENTRIES + 1 {
            let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
            dec.dpb_insert(DpbEntry {
                pic,
                poc: i as i32,
                pts: None,
                used_for_reference: true,
                output_emitted: false,
            });
        }
        assert_eq!(dec.dpb.len(), MAX_DPB_ENTRIES);
        // POC 0 was evicted; POC 1..MAX_DPB_ENTRIES still present.
        assert!(dec.dpb_find(0).is_none());
        assert!(dec.dpb_find(1).is_some());
        assert!(dec.dpb_find(MAX_DPB_ENTRIES as i32).is_some());
    }

    /// `dpb_flush` clears every entry + resets POC tracker.
    #[test]
    fn dpb_flush_clears_all() {
        use crate::picture::YuvPicture;
        let mut dec = EvcDecoder::new();
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        dec.dpb_insert(DpbEntry {
            pic,
            poc: 5,
            pts: None,
            used_for_reference: true,
            output_emitted: false,
        });
        dec.poc_msb = 256;
        dec.prev_poc_lsb = 100;
        dec.dpb_flush();
        assert!(dec.dpb.is_empty());
        assert_eq!(dec.poc_msb, 0);
        assert_eq!(dec.prev_poc_lsb, 0);
    }

    /// **Round-10 flush() drain.** A DPB entry that was never pushed
    /// to the output queue (synthesised here by direct insertion) is
    /// emitted by `flush()` in POC order without duplicating entries
    /// that were already enqueued.
    #[test]
    fn round10_flush_drains_unemitted_dpb_entries_in_poc_order() {
        use crate::picture::YuvPicture;
        let mut dec = EvcDecoder::new();
        // Insert three DPB entries directly. POC 1 is marked emitted
        // (simulating "already in out queue from a prior receive_frame").
        // POCs 0 and 2 are NOT yet in `out`. After flush(), the out
        // queue holds POCs {0, 2} in ascending POC order.
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        dec.dpb_insert(DpbEntry {
            pic: pic.clone(),
            poc: 1,
            pts: None,
            used_for_reference: true,
            output_emitted: true,
        });
        dec.dpb_insert(DpbEntry {
            pic: pic.clone(),
            poc: 0,
            pts: None,
            used_for_reference: true,
            output_emitted: false,
        });
        dec.dpb_insert(DpbEntry {
            pic,
            poc: 2,
            pts: None,
            used_for_reference: true,
            output_emitted: false,
        });
        assert!(dec.out.is_empty());
        dec.drain_dpb_to_output();
        // Expect POCs 0 and 2 in the out queue, sorted ascending.
        let pocs: Vec<i32> = dec.out_pocs.iter().copied().collect();
        assert_eq!(pocs, vec![0, 2]);
        // Calling flush again is idempotent: no duplicate emission.
        dec.drain_dpb_to_output();
        let pocs: Vec<i32> = dec.out_pocs.iter().copied().collect();
        assert_eq!(pocs, vec![0, 2]);
    }

    /// **Round-10 LTRP RPL resolution.** An LTRP entry's `poc_lsb_lt`
    /// resolves to the matching DPB entry's POC (matched against
    /// `poc & (max_poc_lsb − 1)`).
    #[test]
    fn round10_ltrp_rpl_resolves_against_dpb() {
        use crate::picture::YuvPicture;
        use crate::rpl::{RefPicListEntry, RefPicListStruct};
        let mut dec = EvcDecoder::new();
        // 8-bit POC LSB → max_poc_lsb = 256. Insert DPB entries at
        // POCs 0, 1, 5. We'll request an LTRP with poc_lsb_lt = 5
        // (matches POC 5 directly because 5 & 0xFF = 5).
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        for &p in &[0, 1, 5] {
            dec.dpb_insert(DpbEntry {
                pic: pic.clone(),
                poc: p,
                pts: None,
                used_for_reference: true,
                output_emitted: true,
            });
        }
        let rpl = RefPicListStruct {
            num_strp_entries: 0,
            num_ltrp_entries: 1,
            entries: vec![RefPicListEntry::Ltrp { poc_lsb_lt: 5 }],
        };
        let pocs = dec.build_ref_pocs(&rpl, /* cur_poc */ 6, 1, 256).unwrap();
        assert_eq!(pocs, vec![5]);
    }

    /// LTRP entry with no DPB match is rejected as invalid bitstream.
    #[test]
    fn round10_ltrp_missing_dpb_entry_is_invalid() {
        use crate::rpl::{RefPicListEntry, RefPicListStruct};
        let dec = EvcDecoder::new(); // empty DPB
        let rpl = RefPicListStruct {
            num_strp_entries: 0,
            num_ltrp_entries: 1,
            entries: vec![RefPicListEntry::Ltrp { poc_lsb_lt: 7 }],
        };
        let err = dec.build_ref_pocs(&rpl, 10, 1, 256).unwrap_err();
        assert!(format!("{err}").contains("LTRP"));
    }

    /// Mixed STRP + LTRP RPL: STRP delta chain advances normally
    /// while LTRP slot pulls the absolute POC out of the DPB.
    #[test]
    fn round10_mixed_strp_and_ltrp_resolve() {
        use crate::picture::YuvPicture;
        use crate::rpl::{RefPicListEntry, RefPicListStruct};
        let mut dec = EvcDecoder::new();
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        // DPB: POCs 0, 4, 7. cur_poc = 8. RPL = [STRP delta=-1, LTRP poc_lsb_lt=4].
        for &p in &[0, 4, 7] {
            dec.dpb_insert(DpbEntry {
                pic: pic.clone(),
                poc: p,
                pts: None,
                used_for_reference: true,
                output_emitted: true,
            });
        }
        let rpl = RefPicListStruct {
            num_strp_entries: 1,
            num_ltrp_entries: 1,
            entries: vec![
                RefPicListEntry::Strp {
                    delta_poc_st: 1,
                    sign: false,
                }, // delta = -1 → 8 + (-1) = 7
                RefPicListEntry::Ltrp { poc_lsb_lt: 4 }, // → POC 4
            ],
        };
        let pocs = dec.build_ref_pocs(&rpl, 8, 2, 256).unwrap();
        assert_eq!(pocs, vec![7, 4]);
    }
}
