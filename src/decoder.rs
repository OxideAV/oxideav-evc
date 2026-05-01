//! Registry glue + minimal decode pipeline for the EVC crate.
//!
//! Round-4 status: a working decoder for **Baseline-profile** bitstreams
//! that satisfy:
//!
//! * 8-bit luma + chroma,
//! * `slice_deblocking_filter_flag = 0`,
//! * every CU has `cbf_luma == cbf_cb == cbf_cr == 0` (pure intra/inter
//!   prediction with no residual; round-5 wires real residual decoding),
//! * P / B slices use a single reference picture per list (Baseline
//!   round-4 fixtures keep `num_ref_idx_active_minus1_l? == 0`),
//! * Inter MVs land on the Baseline 1/4-pel grid (sub-pel phases 4, 8, 12
//!   for luma; 4, 8, 12, 16, 20, 24, 28 for chroma).
//!
//! Anything else (non-Baseline, 10-bit, deblocked, residuals present,
//! multi-ref, sub-pel outside Baseline grid) bubbles up as
//! `Error::Unsupported`. The decoder consumes length-prefixed NAL units
//! (Annex B raw bitstream framing) per ISO/IEC 23094-1.

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

/// Build the round-3 decoder for the registry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(EvcDecoder::new()))
}

/// Public-but-internal type that callers normally see via the `Decoder`
/// trait. Wraps the per-stream parameter-set cache + an output queue.
pub struct EvcDecoder {
    codec_id: CodecId,
    sps: Option<Sps>,
    pps: Option<Pps>,
    pending_pts: Option<i64>,
    out: VecDeque<VideoFrame>,
    /// Last decoded picture — kept around so the next P / B slice can
    /// reference it. Round-4 only supports a single L0 / L1 reference.
    last_pic: Option<YuvPicture>,
}

impl EvcDecoder {
    pub fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            sps: None,
            pps: None,
            pending_pts: None,
            out: VecDeque::new(),
            last_pic: None,
        }
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
                        .ok_or_else(|| Error::invalid("evc decoder: IDR slice before SPS"))?;
                    let pps = self
                        .pps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: IDR slice before PPS"))?;
                    let (pic, _stats) = crate::decode_idr_slice(sps, pps, nal.rbsp())?;
                    let frame = picture_to_video_frame(&pic, packet.pts);
                    self.out.push_back(frame);
                    self.last_pic = Some(pic);
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
                    let last = self.last_pic.as_ref().ok_or_else(|| {
                        Error::invalid("evc decoder: P/B slice without a prior IDR reference")
                    })?;
                    let pic = decode_non_idr_via_inter(&sps, &pps, nal.rbsp(), last)?;
                    let frame = picture_to_video_frame(&pic, packet.pts);
                    self.out.push_back(frame);
                    self.last_pic = Some(pic);
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
            Some(v) => Ok(Frame::Video(v)),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Decode a NonIDR (P or B) slice into a fresh [`YuvPicture`] using the
/// previous picture as the L0 reference. Round-4 supports a single ref
/// per list and uses the same picture for L0 and L1 in B slices (no
/// multi-frame reordering yet).
fn decode_non_idr_via_inter(
    sps: &Sps,
    pps: &Pps,
    slice_nal_rbsp: &[u8],
    last: &YuvPicture,
) -> Result<YuvPicture> {
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
            "evc decoder: round-4 P/B requires Baseline-profile toolset",
        ));
    }
    if !pps.single_tile_in_pic_flag {
        return Err(Error::unsupported(
            "evc decoder: round-4 P/B requires single_tile_in_pic_flag == 1",
        ));
    }
    // Parse the slice header just enough to recover slice_type, slice_qp,
    // and the byte alignment for slice_data().
    let mut br = crate::bitreader::BitReader::new(slice_nal_rbsp);
    let _slice_pps_id = br.ue()?;
    let slice_type = br.ue()?; // 0=B, 1=P, 2=I
    let slice_is_b = match slice_type {
        0 => true,
        1 => false,
        other => {
            return Err(Error::invalid(format!(
                "evc decoder: NonIDR slice_type must be 0 (B) or 1 (P), got {other}"
            )))
        }
    };
    // sps_pocs_flag = 0 (Baseline), sps_rpl_flag = 0, sps_alf_flag = 0,
    // sps_mmvd_flag = 0 → no extra fields in this region.
    // We must read the P/B-only ref_idx fields per slice_header parser.
    // For Baseline single-ref: num_ref_idx_active_override_flag = 0 →
    // skip the explicit overrides.
    let num_ref_idx_active_override_flag = br.u1()?;
    let mut num_ref_idx_active_minus1_l0 = 0u32;
    let mut num_ref_idx_active_minus1_l1 = 0u32;
    if num_ref_idx_active_override_flag != 0 {
        num_ref_idx_active_minus1_l0 = br.ue()?;
        if slice_is_b {
            num_ref_idx_active_minus1_l1 = br.ue()?;
        }
    }
    // sps_admvp_flag = 0 → no temporal_mvp_assigned_flag.
    let slice_deblocking_filter_flag = br.u1()? != 0;
    let slice_qp = br.u(6)?;
    if slice_qp > 51 {
        return Err(Error::invalid(format!(
            "evc decoder: NonIDR slice_qp {slice_qp} > 51"
        )));
    }
    let _slice_cb_qp_offset = br.se()?;
    let _slice_cr_qp_offset = br.se()?;
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
    };
    let ref_view = RefPictureView {
        y: &last.y,
        cb: &last.cb,
        cr: &last.cr,
        width: last.width,
        height: last.height,
        y_stride: last.y_stride(),
        c_stride: last.c_stride(),
        chroma_format_idc: last.chroma_format_idc,
    };
    let inputs = InterDecodeInputs {
        walk,
        decode,
        slice_is_b,
        num_ref_idx_active_minus1_l0,
        num_ref_idx_active_minus1_l1,
        ref_l0: ref_view,
        ref_l1: if slice_is_b { Some(ref_view) } else { None },
    };
    let (pic, _stats) = crate::slice_data::decode_baseline_inter_slice(slice_data_bytes, inputs)?;
    Ok(pic)
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
