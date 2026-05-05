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

use crate::inter::RefPictureView;
use crate::nal::{iter_length_prefixed, NalUnitType};
use crate::picture::YuvPicture;
use crate::pps::{self, Pps};
use crate::slice_data::{InterDecodeInputs, SliceDecodeInputs, SliceWalkInputs};
use crate::sps::{self, Sps};
use crate::CODEC_ID_STR;

/// Decoded-picture buffer entry (round-9). Each slot keeps the
/// reconstructed picture, its POC and the original presentation
/// timestamp so frames can be re-ordered by POC before emission.
#[derive(Clone, Debug)]
struct DpbEntry {
    pic: YuvPicture,
    poc: i32,
    pts: Option<i64>,
    /// `true` while the picture is still needed as a reference. Round-9
    /// keeps every IDR + short-term ref alive until a fresh IDR flushes
    /// the buffer. Long-term references and explicit RPL-driven
    /// removal are deferred — the field is parked here for round-10's
    /// sliding-window unmark step.
    #[allow(dead_code)]
    used_for_reference: bool,
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
                    let (pic, _stats) = crate::decode_idr_slice(&sps, &pps, nal.rbsp())?;
                    // §8.3.1: IDR resets POC to 0, flushes the DPB.
                    self.dpb_flush();
                    let entry = DpbEntry {
                        pic,
                        poc: 0,
                        pts: packet.pts,
                        used_for_reference: true,
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
                    let (pic, poc) = self.decode_non_idr(&sps, &pps, nal.rbsp())?;
                    let entry = DpbEntry {
                        pic,
                        poc,
                        pts: packet.pts,
                        used_for_reference: true,
                    };
                    self.dpb_insert(entry);
                    self.enqueue_for_output(poc);
                }
                _ => {
                    // APS / SEI / FD / etc. — skipped for round 3.
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
        // Drain remaining DPB entries to the output queue in POC order.
        // Round-9: only entries that haven't already been emitted are
        // pushed; the output queue itself is emitted by `receive_frame`.
        Ok(())
    }
}

impl EvcDecoder {
    /// Push the freshly-decoded picture at `poc` to the output queue
    /// in POC order. Round-9 keeps a parallel `out_pocs` buffer so the
    /// queue stays sorted by POC even when bitstream coding-order
    /// differs from display-order (e.g. B-pyramid GOPs). The frame is
    /// inserted at the unique position where every earlier frame has a
    /// smaller POC and every later frame has a greater POC.
    fn enqueue_for_output(&mut self, poc: i32) {
        let entry = match self.dpb_find(poc) {
            Some(e) => e.clone(),
            None => return,
        };
        let frame = picture_to_video_frame(&entry.pic, entry.pts);
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
            || sps.sps_alf_flag
            || sps.sps_addb_flag
            || sps.sps_dquant_flag
            || sps.sps_ats_flag
            || sps.sps_ibc_flag
            || sps.sps_dra_flag
            || sps.sps_adcc_flag
            || sps.sps_cm_init_flag
        {
            return Err(Error::unsupported(
                "evc decoder: P/B requires Baseline-profile toolset (round-9 adds DPB + POC)",
            ));
        }
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
            build_ref_pocs(&rpl_l0, poc, n_active_l0)?
        } else {
            // §8.3.5 round-9 implicit fallback: with no per-slice or
            // SPS RPL signalling, use the highest-POC decoded picture
            // as the single L0 entry (low-delay coding-order GOP).
            self.implicit_ref_pocs(n_active_l0)?
        };
        let pocs_l1 = if slice_is_b {
            if sps.sps_rpl_flag {
                let rpl_l1 = self.resolve_slice_rpl(&header, sps, 1)?;
                build_ref_pocs(&rpl_l1, poc, n_active_l1)?
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
        let walk = SliceWalkInputs {
            pic_width: sps.pic_width_in_luma_samples,
            pic_height: sps.pic_height_in_luma_samples,
            ctb_log2_size_y,
            min_cb_log2_size_y,
            max_tb_log2_size_y,
            chroma_format_idc: sps.chroma_format_idc,
            cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
        };
        let decode = SliceDecodeInputs {
            slice_qp: slice_qp as i32,
            bit_depth_luma: sps.bit_depth_y(),
            bit_depth_chroma: sps.bit_depth_c(),
            enable_deblock: slice_deblocking_filter_flag,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
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
}

/// Walk an `RefPicListStruct` to produce up to `n_active` reference POCs
/// per §8.3.5. Each STRP entry contributes the running-sum POC. LTRPs
/// would carry an absolute POC (poc_lsb_lt) which round-9 doesn't yet
/// resolve — they're skipped with an invalid-bitstream error if the
/// active count would require them.
fn build_ref_pocs(
    rpl: &crate::rpl::RefPicListStruct,
    cur_poc: i32,
    n_active: usize,
) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(n_active);
    let mut running = cur_poc;
    for entry in rpl.entries.iter().take(n_active) {
        // STRP delta-POC is signed; entry.signed_delta_poc applies the
        // sign bit per §7.4.8 eq. 124. LTRP entries return None.
        let delta = entry.signed_delta_poc().ok_or_else(|| {
            Error::unsupported(
                "evc decoder: round-9 long-term-reference resolution not implemented",
            )
        })?;
        running += delta;
        out.push(running);
    }
    if out.len() < n_active {
        return Err(Error::invalid(format!(
            "evc decoder: RPL has {} entries but n_active = {n_active}",
            out.len()
        )));
    }
    Ok(out)
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
        });
        dec.poc_msb = 256;
        dec.prev_poc_lsb = 100;
        dec.dpb_flush();
        assert!(dec.dpb.is_empty());
        assert_eq!(dec.poc_msb, 0);
        assert_eq!(dec.prev_poc_lsb, 0);
    }
}
