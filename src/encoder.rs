//! Registry glue for the EVC **encoder** (round 429): the
//! [`oxideav_core::Encoder`]-trait wrapper over the Baseline intra
//! pipeline (`headers_enc` + `slice_enc`).
//!
//! Every input frame becomes one self-contained key access unit —
//! `[SPS][PPS][IDR slice]` in the Annex B length-prefixed framing — so
//! any packet decodes standalone through [`crate::decoder::make_decoder`].
//! All-intra only for now (low-delay P is the next encoder milestone).
//!
//! ## Validation posture
//!
//! No external EVC validator binary is staged in `docs/video/evc/`, so
//! the encoder's correctness gates are (a) re-parse-exactness through
//! this crate's own §7.3 parsers and (b) byte-exact
//! encode→decode == encoder-reconstruction round trips against this
//! crate's decoder — which is itself the conformance-validated side of
//! the crate. Cross-implementation decode remains an open follow-up
//! until a validator lands in docs.
//!
//! Options (via [`CodecParameters::options`]):
//!
//! * `qp` — Baseline slice QP, `0..=51` (default 30).

use std::collections::VecDeque;

use oxideav_core::format::PixelFormat;
use oxideav_core::{CodecId, CodecParameters, Encoder, Error, Frame, Packet, Result, TimeBase};

use crate::headers_enc::{
    append_length_prefixed_nal, write_idr_slice_header, write_pps_rbsp, write_sps_rbsp,
    EncSequenceConfig,
};
use crate::nal::NalUnitType;
use crate::picture::YuvPicture;
use crate::slice_enc::{encode_idr_slice_data, EncStats};
use crate::CODEC_ID_STR;

/// Default slice QP when the caller doesn't pass the `qp` option.
pub const DEFAULT_QP: i32 = 30;

/// Encode one 8-bit 4:2:0 source picture into a complete key access
/// unit (`[SPS][PPS][IDR]`, length-prefixed). Returns the bitstream,
/// the reconstruction the decoder will reproduce byte-exactly, and the
/// slice-encoder statistics. This is the whole-frame entry point the
/// registry encoder wraps; tests and fixture tooling can call it
/// directly.
pub fn encode_idr_access_unit(
    src: &YuvPicture,
    slice_qp: i32,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    if src.bit_depth != 8 {
        return Err(Error::unsupported(
            "evc encoder: 8-bit sources only in this round (recon chain is depth-ready)",
        ));
    }
    let (payload, recon, stats) = encode_idr_slice_data(src, slice_qp)?;
    let mut slice_rbsp = write_idr_slice_header(slice_qp as u32)?;
    slice_rbsp.extend_from_slice(&payload);

    let cfg = EncSequenceConfig {
        width: src.width,
        height: src.height,
        level_idc: 51, // generous cap; no external constraint checking yet
    };
    let mut out = Vec::new();
    append_length_prefixed_nal(&mut out, NalUnitType::Sps, &write_sps_rbsp(&cfg)?);
    append_length_prefixed_nal(&mut out, NalUnitType::Pps, &write_pps_rbsp()?);
    append_length_prefixed_nal(&mut out, NalUnitType::Idr, &slice_rbsp);
    Ok((out, recon, stats))
}

/// Build the registered encoder — the [`crate::register`] factory and
/// the historical direct constructor of the workspace dual-API
/// convention.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("evc encoder: CodecParameters.width required"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("evc encoder: CodecParameters.height required"))?;
    if width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0 {
        return Err(Error::unsupported(format!(
            "evc encoder: dimensions {width}x{height} must be non-zero multiples of 4"
        )));
    }
    if let Some(pf) = params.pixel_format {
        if pf != PixelFormat::Yuv420P {
            return Err(Error::unsupported(format!(
                "evc encoder: pixel format {pf:?} unsupported (Yuv420P only)"
            )));
        }
    }
    let qp = match params.options.get("qp") {
        None => DEFAULT_QP,
        Some(s) => s
            .parse::<i32>()
            .ok()
            .filter(|q| (0..=51).contains(q))
            .ok_or_else(|| Error::invalid(format!("evc encoder: qp option {s:?} not in 0..=51")))?,
    };
    let mut out_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    out_params.width = Some(width);
    out_params.height = Some(height);
    out_params.pixel_format = Some(PixelFormat::Yuv420P);
    Ok(Box::new(EvcEncoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        out_params,
        width,
        height,
        qp,
        queue: VecDeque::new(),
    }))
}

/// The registered Baseline intra encoder. Stateless across frames
/// (every frame is an IDR access unit), so `flush` has nothing to
/// drain beyond the packet queue.
// Internal: reach it through [`make_encoder`] / the registry instead.
#[doc(hidden)]
pub struct EvcEncoder {
    codec_id: CodecId,
    out_params: CodecParameters,
    width: u32,
    height: u32,
    qp: i32,
    queue: VecDeque<Packet>,
}

impl Encoder for EvcEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("evc encoder: expected a video frame")),
        };
        let src = video_frame_to_picture(v, self.width, self.height)?;
        let (data, _recon, _stats) = encode_idr_access_unit(&src, self.qp)?;
        let mut pkt = Packet::new(0, TimeBase::new(1, 90_000), data);
        pkt.pts = v.pts;
        pkt.dts = v.pts; // intra-only: decode order == display order
        pkt.flags.keyframe = true;
        self.queue.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // No lookahead: every frame was emitted eagerly.
        Ok(())
    }
}

/// Convert an 8-bit `Yuv420P` [`oxideav_core::VideoFrame`] into the
/// crate's u16 picture buffer, honouring per-plane strides.
fn video_frame_to_picture(
    v: &oxideav_core::VideoFrame,
    width: u32,
    height: u32,
) -> Result<YuvPicture> {
    let planes = v.image_planes();
    if planes.len() < 3 {
        return Err(Error::invalid(format!(
            "evc encoder: expected 3 image planes (Yuv420P), got {}",
            planes.len()
        )));
    }
    let mut pic = YuvPicture::new(width, height, 1, 8)?;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let copy = |dst: &mut [u16],
                dst_w: usize,
                dst_h: usize,
                plane: &oxideav_core::VideoPlane|
     -> Result<()> {
        if plane.stride < dst_w || plane.data.len() < plane.stride * dst_h {
            return Err(Error::invalid(format!(
                "evc encoder: plane too small (stride {}, len {}, need {dst_w}x{dst_h})",
                plane.stride,
                plane.data.len()
            )));
        }
        for y in 0..dst_h {
            let row = &plane.data[y * plane.stride..y * plane.stride + dst_w];
            for (d, &s) in dst[y * dst_w..(y + 1) * dst_w].iter_mut().zip(row.iter()) {
                *d = s as u16;
            }
        }
        Ok(())
    };
    copy(&mut pic.y, width as usize, height as usize, &planes[0])?;
    copy(&mut pic.cb, cw, ch, &planes[1])?;
    copy(&mut pic.cr, cw, ch, &planes[2])?;
    Ok(pic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::frame::{VideoFrame, VideoPlane};

    fn synth_frame(w: usize, h: usize) -> VideoFrame {
        let mut y = vec![0u8; w * h];
        for (i, v) in y.iter_mut().enumerate() {
            let (x, yy) = (i % w, i / w);
            *v = (30 + (x * 2 + yy * 3) % 190) as u8;
        }
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let cb: Vec<u8> = (0..cw * ch).map(|i| (90 + (i % 70)) as u8).collect();
        let cr: Vec<u8> = (0..cw * ch).map(|i| (150 + (i % 60)) as u8).collect();
        VideoFrame {
            pts: Some(42),
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: cb,
                },
                VideoPlane {
                    stride: cw,
                    data: cr,
                },
            ],
        }
    }

    fn params(w: u32, h: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        p.width = Some(w);
        p.height = Some(h);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        p
    }

    /// Whole-loop pin through the *registry* surfaces: make_encoder →
    /// send_frame/receive_packet → make_decoder → the decoded frame is
    /// byte-identical to the encoder's own reconstruction.
    #[test]
    fn registry_encode_decode_round_trip_exact() {
        let (w, h) = (96u32, 64u32);
        let mut enc = make_encoder(&params(w, h)).unwrap();
        let frame = synth_frame(w as usize, h as usize);
        enc.send_frame(&Frame::Video(frame.clone())).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert!(pkt.flags.keyframe);
        assert_eq!(pkt.pts, Some(42));

        // Reference recon from the direct entry point.
        let src = video_frame_to_picture(&frame, w, h).unwrap();
        let (stream, recon, _stats) = encode_idr_access_unit(&src, DEFAULT_QP).unwrap();
        assert_eq!(pkt.data, stream, "registry and direct paths must agree");

        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        dec.send_packet(&pkt).unwrap();
        let out = dec.receive_frame().unwrap();
        let vf = match out {
            Frame::Video(vf) => vf,
            other => panic!("expected video frame, got {other:?}"),
        };
        let y8: Vec<u8> = recon.y.iter().map(|&v| v as u8).collect();
        let cb8: Vec<u8> = recon.cb.iter().map(|&v| v as u8).collect();
        let cr8: Vec<u8> = recon.cr.iter().map(|&v| v as u8).collect();
        assert_eq!(vf.planes[0].data, y8, "decoded luma != encoder recon");
        assert_eq!(vf.planes[1].data, cb8, "decoded cb != encoder recon");
        assert_eq!(vf.planes[2].data, cr8, "decoded cr != encoder recon");
    }

    /// The emitted access unit re-parses exactly: NAL iteration finds
    /// SPS/PPS/IDR, probe() recovers the geometry, and the SPS/PPS
    /// parse to the encoder's declared configuration.
    #[test]
    fn access_unit_reparses_exactly() {
        let src = video_frame_to_picture(&synth_frame(64, 48), 64, 48).unwrap();
        let (stream, _recon, _stats) = encode_idr_access_unit(&src, 22).unwrap();
        let nals = crate::nal::iter_length_prefixed(&stream).unwrap();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(nals[1].header.nal_unit_type, NalUnitType::Pps);
        assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Idr);
        let info = crate::probe(&stream).unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        assert_eq!(info.profile_idc, 0);
        assert_eq!(info.bit_depth_luma, 8);
        let sps = crate::sps::parse(nals[0].rbsp()).unwrap();
        let pps = crate::pps::parse(nals[1].rbsp()).unwrap();
        assert!(pps.single_tile_in_pic_flag);
        assert!(!sps.sps_btt_flag);
    }

    /// Multi-frame session: three frames in, three keyframe packets
    /// out, each independently decodable (IDR flushes the DPB).
    #[test]
    fn multi_frame_session() {
        let (w, h) = (64u32, 64u32);
        let mut enc = make_encoder(&params(w, h)).unwrap();
        for i in 0..3i64 {
            let mut f = synth_frame(w as usize, h as usize);
            f.pts = Some(i * 3600);
            enc.send_frame(&Frame::Video(f)).unwrap();
        }
        enc.flush().unwrap();
        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        for i in 0..3i64 {
            let pkt = enc.receive_packet().unwrap();
            assert_eq!(pkt.pts, Some(i * 3600));
            assert!(pkt.flags.keyframe);
            dec.send_packet(&pkt).unwrap();
            let f = dec.receive_frame().unwrap();
            assert!(matches!(f, Frame::Video(_)));
        }
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
    }

    /// Option handling: explicit qp is honoured (lower qp → bigger
    /// packet), bad qp is refused, missing dims are refused.
    #[test]
    fn options_and_validation() {
        let (w, h) = (64u32, 64u32);
        let frame = synth_frame(w as usize, h as usize);
        let mut p_low = params(w, h);
        p_low.options.insert("qp", "4");
        let mut p_high = params(w, h);
        p_high.options.insert("qp", "48");
        let mut enc_low = make_encoder(&p_low).unwrap();
        let mut enc_high = make_encoder(&p_high).unwrap();
        enc_low.send_frame(&Frame::Video(frame.clone())).unwrap();
        enc_high.send_frame(&Frame::Video(frame)).unwrap();
        let low = enc_low.receive_packet().unwrap();
        let high = enc_high.receive_packet().unwrap();
        assert!(
            low.data.len() > high.data.len(),
            "qp 4 ({}) must out-size qp 48 ({})",
            low.data.len(),
            high.data.len()
        );

        let mut bad = params(w, h);
        bad.options.insert("qp", "52");
        assert!(make_encoder(&bad).is_err());
        let mut nodims = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        nodims.pixel_format = Some(PixelFormat::Yuv420P);
        assert!(make_encoder(&nodims).is_err());
        assert!(make_encoder(&params(66, 64)).is_err());
    }

    /// Output params advertise the stream a muxer needs.
    #[test]
    fn output_params_shape() {
        let enc = make_encoder(&params(128, 96)).unwrap();
        let p = enc.output_params();
        assert_eq!(p.codec_id.as_str(), CODEC_ID_STR);
        assert_eq!(p.width, Some(128));
        assert_eq!(p.height, Some(96));
        assert_eq!(p.pixel_format, Some(PixelFormat::Yuv420P));
    }
}
