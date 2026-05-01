//! Registry glue + minimal decode pipeline for the EVC crate.
//!
//! Round-3 status: a working decoder for **Baseline-profile IDR-only**
//! bitstreams that satisfy:
//!
//! * 8-bit luma + chroma,
//! * `slice_deblocking_filter_flag = 0`,
//! * every CU has `cbf_luma == cbf_cb == cbf_cr == 0` (pure intra
//!   prediction with no residual; round-4 wires real residual decoding).
//!
//! Anything else (P/B slices, non-Baseline, 10-bit, deblocked, residuals
//! present) bubbles up as `Error::Unsupported`. The decoder consumes
//! length-prefixed NAL units (Annex B raw bitstream framing) per
//! ISO/IEC 23094-1.

use std::collections::VecDeque;

use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::nal::{iter_length_prefixed, NalUnitType};
use crate::pps::{self, Pps};
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
}

impl EvcDecoder {
    pub fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            sps: None,
            pps: None,
            pending_pts: None,
            out: VecDeque::new(),
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
                }
                NalUnitType::NonIdr => {
                    return Err(Error::unsupported(
                        "evc decoder: round-3 supports IDR only (P / B slice arrived)",
                    ));
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
