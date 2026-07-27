//! Registry glue for the EVC **encoder** (round 429 intra bootstrap,
//! round 431 context modelling + low-delay P): the
//! [`oxideav_core::Encoder`]-trait wrapper over the intra
//! (`headers_enc` + `slice_enc`) and P (`slice_enc_p`) pipelines.
//!
//! Every GOP opens with a self-contained key access unit —
//! `[SPS][PPS][IDR slice]` in the Annex B length-prefixed framing — so
//! that packet decodes standalone through
//! [`crate::decoder::make_decoder`]; with `gop > 1` the following
//! frames are single-NAL low-delay P access units, each referencing
//! the previous reconstruction (decode order == display order).
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
//! * `qp` — slice QP, `0..=51` (default 30).
//! * `deblock` — `1`/`true` signals `slice_deblocking_filter_flag = 1`
//!   and the decoder runs the §8.8.2 filter; the encoder tracks the
//!   filtered reconstruction so output stays byte-exact (default off).
//! * `cm_init` — `sps_cm_init_flag` context modelling (§9.3.2.2 /
//!   §9.3.4.2), default **on**: per-syntax-element CABAC contexts, the
//!   round-431 rate win at identical reconstruction. The SPS then
//!   signals `profile_idc = 1` (Main) because Annex A.3.2 pins
//!   `sps_cm_init_flag == 0` in the Baseline profile. `cm_init=0`
//!   restores the Baseline-profile single-context stream.
//! * `gop` — GOP length (default 1 = the historical all-intra shape):
//!   frame indices `0, gop, 2·gop, …` are IDR access units, the rest
//!   low-delay P pictures (skip / explicit-MV inter / intra mode
//!   ladder per CU against the previous reconstruction).

use std::collections::VecDeque;

use oxideav_core::format::PixelFormat;
use oxideav_core::{CodecId, CodecParameters, Encoder, Error, Frame, Packet, Result, TimeBase};

use crate::headers_enc::{
    append_length_prefixed_nal, write_idr_slice_header, write_p_slice_header, write_pps_rbsp,
    write_sps_rbsp, EncSequenceConfig,
};
use crate::nal::NalUnitType;
use crate::picture::YuvPicture;
use crate::slice_enc::EncStats;
use crate::slice_enc_p::PEncStats;
use crate::CODEC_ID_STR;

/// Default slice QP when the caller doesn't pass the `qp` option.
pub const DEFAULT_QP: i32 = 30;

/// Encode one 4:2:0 source picture (any 8..=16 bit depth the recon
/// chain supports) into a complete key access unit
/// (`[SPS][PPS][IDR]`, length-prefixed). Returns the bitstream,
/// the reconstruction the decoder will reproduce byte-exactly, and the
/// slice-encoder statistics. This is the whole-frame entry point the
/// registry encoder wraps; tests and fixture tooling can call it
/// directly.
pub fn encode_idr_access_unit(
    src: &YuvPicture,
    slice_qp: i32,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_access_unit_with(src, slice_qp, false)
}

/// [`encode_idr_access_unit`] with `slice_deblocking_filter_flag`
/// control: with `deblock` the returned recon is the §8.8.2-filtered
/// picture the decoder emits (see
/// [`crate::slice_enc::encode_idr_slice_data_with`]).
pub fn encode_idr_access_unit_with(
    src: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    encode_idr_access_unit_opts(src, slice_qp, deblock, false)
}

/// [`encode_idr_access_unit_with`] plus the round-431 `sps_cm_init_flag`
/// entropy selection: with `cm_init` the slice payload runs the
/// §9.3.2.2 per-syntax-element context modelling and the SPS signals
/// `profile_idc = 1` (Main) with the Table A.6 cm_init toolset bit —
/// Annex A.3.2 bars the tool from the Baseline profile. The registered
/// encoder defaults this ON (it is a pure rate win at identical
/// reconstruction); `cm_init = false` keeps the historical
/// Baseline-profile single-context stream.
pub fn encode_idr_access_unit_opts(
    src: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
) -> Result<(Vec<u8>, YuvPicture, EncStats)> {
    let (payload, recon, stats) =
        crate::slice_enc::encode_idr_slice_data_opts(src, slice_qp, deblock, cm_init)?;
    let mut slice_rbsp = write_idr_slice_header(slice_qp as u32, deblock)?;
    slice_rbsp.extend_from_slice(&payload);

    let cfg = EncSequenceConfig {
        width: src.width,
        height: src.height,
        level_idc: 51, // generous cap; no external constraint checking yet
        bit_depth: src.bit_depth,
        cm_init,
    };
    let mut out = Vec::new();
    append_length_prefixed_nal(&mut out, NalUnitType::Sps, &write_sps_rbsp(&cfg)?);
    append_length_prefixed_nal(&mut out, NalUnitType::Pps, &write_pps_rbsp()?);
    append_length_prefixed_nal(&mut out, NalUnitType::Idr, &slice_rbsp);
    Ok((out, recon, stats))
}

/// Encode one **P** access unit (round 431) — a single NonIDR NAL
/// referencing `ref_pic`, the previous picture exactly as the decoder
/// holds it in the DPB (post-§8.8.2 when `deblock` is on). The SPS/PPS
/// travel with the GOP's opening IDR access unit; the decoder's
/// `sps_pocs_flag == 0` fallback derives POC as coding order and its
/// implicit-RPL fallback resolves L0[0] to the highest-POC DPB entry —
/// i.e. this picture's `ref_pic`. Returns the bitstream, the
/// reconstruction the decoder reproduces byte-exactly (the caller's
/// next reference), and the P-slice statistics.
pub fn encode_p_access_unit_opts(
    src: &YuvPicture,
    ref_pic: &YuvPicture,
    slice_qp: i32,
    deblock: bool,
    cm_init: bool,
) -> Result<(Vec<u8>, YuvPicture, PEncStats)> {
    let (payload, recon, stats) =
        crate::slice_enc_p::encode_p_slice_data(src, ref_pic, slice_qp, deblock, cm_init)?;
    let mut slice_rbsp = write_p_slice_header(slice_qp as u32, deblock)?;
    slice_rbsp.extend_from_slice(&payload);
    let mut out = Vec::new();
    append_length_prefixed_nal(&mut out, NalUnitType::NonIdr, &slice_rbsp);
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
    let bit_depth = match params.pixel_format {
        None | Some(PixelFormat::Yuv420P) => 8,
        Some(PixelFormat::Yuv420P10Le) => 10,
        Some(PixelFormat::Yuv420P12Le) => 12,
        Some(pf) => {
            return Err(Error::unsupported(format!(
                "evc encoder: pixel format {pf:?} unsupported \
                 (Yuv420P / Yuv420P10Le / Yuv420P12Le)"
            )))
        }
    };
    let parse_bool = |name: &str, default: bool| -> Result<bool> {
        match params.options.get(name) {
            None => Ok(default),
            Some("1") | Some("true") => Ok(true),
            Some("0") | Some("false") => Ok(false),
            Some(other) => Err(Error::invalid(format!(
                "evc encoder: {name} option {other:?} not a boolean"
            ))),
        }
    };
    let deblock = parse_bool("deblock", false)?;
    // Round 431: `sps_cm_init_flag` context modelling — default ON (the
    // rate win; identical reconstruction). `cm_init=0` selects the
    // historical Baseline-profile single-context stream.
    let cm_init = parse_bool("cm_init", true)?;
    let qp = match params.options.get("qp") {
        None => DEFAULT_QP,
        Some(s) => s
            .parse::<i32>()
            .ok()
            .filter(|q| (0..=51).contains(q))
            .ok_or_else(|| Error::invalid(format!("evc encoder: qp option {s:?} not in 0..=51")))?,
    };
    // Round 431: GOP length — every gop-th frame opens with an IDR
    // access unit, the rest are low-delay P pictures referencing the
    // previous reconstruction. Default 1 keeps the historical
    // all-intra shape.
    let gop = match params.options.get("gop") {
        None => 1u32,
        Some(s) => s
            .parse::<u32>()
            .ok()
            .filter(|&g| g >= 1)
            .ok_or_else(|| Error::invalid(format!("evc encoder: gop option {s:?} not >= 1")))?,
    };
    let mut out_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    out_params.width = Some(width);
    out_params.height = Some(height);
    out_params.pixel_format = Some(params.pixel_format.unwrap_or(PixelFormat::Yuv420P));
    Ok(Box::new(EvcEncoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        out_params,
        width,
        height,
        bit_depth,
        qp,
        deblock,
        cm_init,
        gop,
        frame_idx: 0,
        ref_recon: None,
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
    bit_depth: u32,
    qp: i32,
    deblock: bool,
    cm_init: bool,
    /// GOP length: frame indices `0, gop, 2·gop, …` are IDR access
    /// units, the rest low-delay P pictures.
    gop: u32,
    frame_idx: u64,
    /// The previous reconstruction — the decoder's DPB picture the next
    /// P frame references (post-deblock when the option is on).
    ref_recon: Option<YuvPicture>,
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
        let src = video_frame_to_picture(v, self.width, self.height, self.bit_depth)?;
        let is_idr = self.gop <= 1 || self.frame_idx % (self.gop as u64) == 0;
        let (data, recon) = if is_idr {
            let (data, recon, _stats) =
                encode_idr_access_unit_opts(&src, self.qp, self.deblock, self.cm_init)?;
            (data, recon)
        } else {
            let refp = self
                .ref_recon
                .as_ref()
                .expect("non-IDR frame always follows a reconstructed frame");
            let (data, recon, _stats) =
                encode_p_access_unit_opts(&src, refp, self.qp, self.deblock, self.cm_init)?;
            (data, recon)
        };
        self.ref_recon = Some(recon);
        self.frame_idx += 1;
        let mut pkt = Packet::new(0, TimeBase::new(1, 90_000), data);
        pkt.pts = v.pts;
        pkt.dts = v.pts; // low delay: decode order == display order
        pkt.flags.keyframe = is_idr;
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

/// Convert a planar 4:2:0 [`oxideav_core::VideoFrame`] into the
/// crate's u16 picture buffer, honouring per-plane strides. 8-bit
/// frames carry one byte per sample; deeper frames two little-endian
/// bytes per sample (the `Yuv420P10Le`-family layout, strides in
/// bytes) — the same conventions the decoder emits.
fn video_frame_to_picture(
    v: &oxideav_core::VideoFrame,
    width: u32,
    height: u32,
    bit_depth: u32,
) -> Result<YuvPicture> {
    let planes = v.image_planes();
    if planes.len() < 3 {
        return Err(Error::invalid(format!(
            "evc encoder: expected 3 image planes (4:2:0 planar), got {}",
            planes.len()
        )));
    }
    let mut pic = YuvPicture::new(width, height, 1, bit_depth)?;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let bytes_per = if bit_depth > 8 { 2usize } else { 1 };
    let max_val = ((1u32 << bit_depth) - 1) as u16;
    let copy = |dst: &mut [u16],
                dst_w: usize,
                dst_h: usize,
                plane: &oxideav_core::VideoPlane|
     -> Result<()> {
        let row_bytes = dst_w * bytes_per;
        if plane.stride < row_bytes || plane.data.len() < plane.stride * dst_h {
            return Err(Error::invalid(format!(
                "evc encoder: plane too small (stride {}, len {}, need {dst_w}x{dst_h}x{bytes_per})",
                plane.stride,
                plane.data.len()
            )));
        }
        for y in 0..dst_h {
            let row = &plane.data[y * plane.stride..y * plane.stride + row_bytes];
            for (x, d) in dst[y * dst_w..(y + 1) * dst_w].iter_mut().enumerate() {
                let s = if bytes_per == 2 {
                    u16::from_le_bytes([row[2 * x], row[2 * x + 1]])
                } else {
                    row[x] as u16
                };
                if s > max_val {
                    return Err(Error::invalid(format!(
                        "evc encoder: sample {s} exceeds {bit_depth}-bit range"
                    )));
                }
                *d = s;
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

        // Reference recon from the direct entry point (the registry
        // defaults `cm_init` on).
        let src = video_frame_to_picture(&frame, w, h, 8).unwrap();
        let (stream, recon, _stats) =
            encode_idr_access_unit_opts(&src, DEFAULT_QP, false, true).unwrap();
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

    /// The `cm_init` access unit re-parses exactly: Main profile
    /// (Annex A.3.2 bars cm_init from Baseline), the Table A.6 binIdx-14
    /// toolset bit, `sps_cm_init_flag = 1`, every other tool off — and
    /// the whole loop stays recon-exact through the registered decoder.
    #[test]
    fn cm_init_access_unit_reparses_and_round_trips() {
        let src = video_frame_to_picture(&synth_frame(64, 48), 64, 48, 8).unwrap();
        let (stream, recon, _stats) = encode_idr_access_unit_opts(&src, 22, false, true).unwrap();
        let nals = crate::nal::iter_length_prefixed(&stream).unwrap();
        let sps = crate::sps::parse(nals[0].rbsp()).unwrap();
        assert_eq!(sps.profile_idc, 1, "cm_init stream must declare Main");
        assert_eq!(sps.toolset_idc_h, 0x4000, "Table A.6 binIdx 14 only");
        assert_eq!(sps.toolset_idc_l, 0);
        assert!(sps.sps_cm_init_flag);
        assert!(!sps.sps_adcc_flag, "conditional sps_adcc_flag written 0");
        assert!(!sps.sps_btt_flag);
        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), stream);
        dec.send_packet(&pkt).unwrap();
        let vf = match dec.receive_frame().unwrap() {
            Frame::Video(vf) => vf,
            other => panic!("expected video frame, got {other:?}"),
        };
        let y8: Vec<u8> = recon.y.iter().map(|&v| v as u8).collect();
        assert_eq!(vf.planes[0].data, y8, "cm_init decode != recon");
    }

    /// The emitted Baseline (`cm_init=0`) access unit re-parses exactly:
    /// NAL iteration finds SPS/PPS/IDR, probe() recovers the geometry,
    /// and the SPS/PPS parse to the encoder's declared configuration.
    #[test]
    fn access_unit_reparses_exactly() {
        let src = video_frame_to_picture(&synth_frame(64, 48), 64, 48, 8).unwrap();
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
        let mut bad_cm = params(w, h);
        bad_cm.options.insert("cm_init", "sometimes");
        assert!(make_encoder(&bad_cm).is_err());
        let mut bad_gop = params(w, h);
        bad_gop.options.insert("gop", "0");
        assert!(make_encoder(&bad_gop).is_err());
        let mut bad_gop2 = params(w, h);
        bad_gop2.options.insert("gop", "-3");
        assert!(make_encoder(&bad_gop2).is_err());
        // cm_init=0 keeps the historical Baseline-profile stream.
        let mut p_base = params(w, h);
        p_base.options.insert("cm_init", "0");
        let mut enc_base = make_encoder(&p_base).unwrap();
        enc_base
            .send_frame(&Frame::Video(synth_frame(w as usize, h as usize)))
            .unwrap();
        let pkt_base = enc_base.receive_packet().unwrap();
        assert_eq!(crate::probe(&pkt_base.data).unwrap().profile_idc, 0);
        let mut nodims = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        nodims.pixel_format = Some(PixelFormat::Yuv420P);
        assert!(make_encoder(&nodims).is_err());
        assert!(make_encoder(&params(66, 64)).is_err());
    }

    /// 10-bit input (`Yuv420P10Le`, two LE bytes per sample): the whole
    /// registry loop must round-trip recon-exact, with the decoder
    /// emitting the matching 16-bit-LE planes.
    #[test]
    fn ten_bit_round_trip_exact() {
        let (w, h) = (64u32, 48u32);
        let (cw, chh) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        let pack16 = |vals: &[u16]| -> Vec<u8> {
            let mut out = Vec::with_capacity(vals.len() * 2);
            for &v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        };
        let y: Vec<u16> = (0..(w * h) as usize)
            .map(|i| (((i % w as usize) * 9 + (i / w as usize) * 5) % 1024) as u16)
            .collect();
        let cb: Vec<u16> = (0..cw * chh).map(|i| (300 + (i % 400)) as u16).collect();
        let cr: Vec<u16> = (0..cw * chh).map(|i| (600 + (i % 300)) as u16).collect();
        let frame = VideoFrame {
            pts: Some(7),
            planes: vec![
                VideoPlane {
                    stride: w as usize * 2,
                    data: pack16(&y),
                },
                VideoPlane {
                    stride: cw * 2,
                    data: pack16(&cb),
                },
                VideoPlane {
                    stride: cw * 2,
                    data: pack16(&cr),
                },
            ],
        };
        let mut p = params(w, h);
        p.pixel_format = Some(PixelFormat::Yuv420P10Le);
        p.options.insert("qp", "20");
        let mut enc = make_encoder(&p).unwrap();
        enc.send_frame(&Frame::Video(frame.clone())).unwrap();
        let pkt = enc.receive_packet().unwrap();
        let info = crate::probe(&pkt.data).unwrap();
        assert_eq!(info.bit_depth_luma, 10);

        let src = video_frame_to_picture(&frame, w, h, 10).unwrap();
        let (stream, recon, _stats) = encode_idr_access_unit_opts(&src, 20, false, true).unwrap();
        assert_eq!(pkt.data, stream);

        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        dec.send_packet(&pkt).unwrap();
        let vf = match dec.receive_frame().unwrap() {
            Frame::Video(vf) => vf,
            other => panic!("expected video frame, got {other:?}"),
        };
        assert_eq!(vf.planes[0].data, pack16(&recon.y), "10-bit luma");
        assert_eq!(vf.planes[1].data, pack16(&recon.cb), "10-bit cb");
        assert_eq!(vf.planes[2].data, pack16(&recon.cr), "10-bit cr");
        // And the recon is a faithful 10-bit picture, not an 8-bit one.
        assert!(recon.y.iter().any(|&v| v > 255));
    }

    /// Single-byte mutation gate over the encoder's own access unit:
    /// every low-bit flip and full inversion of every stream byte must
    /// decode to either a clean error or a frame — never a panic.
    #[test]
    fn mutation_gate_over_encoded_stream() {
        let src = video_frame_to_picture(&synth_frame(48, 48), 48, 48, 8).unwrap();
        // The registry-default shape (cm_init on) — the widest context
        // surface the decoder can be driven across by a bit flip.
        let (stream, _recon, _stats) = encode_idr_access_unit_opts(&src, 42, false, true).unwrap();
        for i in 0..stream.len() {
            for mask in [0x01u8, 0xFF] {
                let mut mutated = stream.clone();
                mutated[i] ^= mask;
                let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
                let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
                let pkt = Packet::new(0, TimeBase::new(1, 90_000), mutated);
                if dec.send_packet(&pkt).is_ok() {
                    let _ = dec.receive_frame();
                }
            }
        }
    }

    /// 12-bit input (`Yuv420P12Le`) closes the depth matrix: registry
    /// loop recon-exact, probe reports depth 12.
    #[test]
    fn twelve_bit_round_trip_exact() {
        let (w, h) = (32u32, 32u32);
        let (cw, chh) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        let pack16 = |vals: &[u16]| -> Vec<u8> {
            let mut out = Vec::with_capacity(vals.len() * 2);
            for &v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        };
        let y: Vec<u16> = (0..(w * h) as usize)
            .map(|i| ((i * 37) % 4096) as u16)
            .collect();
        let cb: Vec<u16> = (0..cw * chh)
            .map(|i| (1000 + i * 3 % 2000) as u16)
            .collect();
        let cr: Vec<u16> = (0..cw * chh).map(|i| (2500 + i % 1500) as u16).collect();
        let frame = VideoFrame {
            pts: Some(1),
            planes: vec![
                VideoPlane {
                    stride: w as usize * 2,
                    data: pack16(&y),
                },
                VideoPlane {
                    stride: cw * 2,
                    data: pack16(&cb),
                },
                VideoPlane {
                    stride: cw * 2,
                    data: pack16(&cr),
                },
            ],
        };
        let mut p = params(w, h);
        p.pixel_format = Some(PixelFormat::Yuv420P12Le);
        let mut enc = make_encoder(&p).unwrap();
        enc.send_frame(&Frame::Video(frame.clone())).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(crate::probe(&pkt.data).unwrap().bit_depth_luma, 12);
        let src = video_frame_to_picture(&frame, w, h, 12).unwrap();
        let (stream, recon, _stats) =
            encode_idr_access_unit_opts(&src, DEFAULT_QP, false, true).unwrap();
        assert_eq!(pkt.data, stream);
        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        dec.send_packet(&pkt).unwrap();
        let vf = match dec.receive_frame().unwrap() {
            Frame::Video(vf) => vf,
            other => panic!("expected video frame, got {other:?}"),
        };
        assert_eq!(vf.planes[0].data, pack16(&recon.y), "12-bit luma");
        assert_eq!(vf.planes[1].data, pack16(&recon.cb), "12-bit cb");
        assert_eq!(vf.planes[2].data, pack16(&recon.cr), "12-bit cr");
        assert!(
            recon.y.iter().any(|&v| v > 1023),
            "true 12-bit range in play"
        );
    }

    /// Deblocking-on encode: with the `deblock` option the slice signals
    /// `slice_deblocking_filter_flag = 1` and the decoder runs the
    /// §8.8.2 post-pass. On an **all-intra** picture that pass is
    /// normatively a no-op: §8.8.2.3 step 2 sets `bS = 0` whenever p0
    /// or q0 lies in an intra-coded CU, and Table 33 defines `sT` only
    /// for `bS ∈ {1, 2, 3}` (step 5 filters only when `sT′ > 0`) — the
    /// Baseline filter touches inter/IBC/cbf edges exclusively. So the
    /// pin is: flag round-trips, the decoder's filtered output still
    /// equals the encoder recon byte-exactly, and that recon equals the
    /// unfiltered encode (the intra no-op made explicit). The plumbing
    /// becomes load-bearing the moment the P encoder lands.
    #[test]
    fn deblock_on_round_trip_exact() {
        let (w, h) = (64u32, 64u32);
        let frame = synth_frame(w as usize, h as usize);
        let src = video_frame_to_picture(&frame, w, h, 8).unwrap();
        let qp = 45;
        let (stream_db, recon_db, _) = encode_idr_access_unit_opts(&src, qp, true, true).unwrap();
        let (stream_no, recon_no, _) = encode_idr_access_unit_opts(&src, qp, false, true).unwrap();
        assert_eq!(
            recon_db.y, recon_no.y,
            "§8.8.2.3: intra edges are bS 0 — the pass must be a no-op on I pictures"
        );
        assert_ne!(stream_db, stream_no, "the header flag bit must differ");
        assert_eq!(stream_db.len(), stream_no.len());

        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), stream_db);
        dec.send_packet(&pkt).unwrap();
        let vf = match dec.receive_frame().unwrap() {
            Frame::Video(vf) => vf,
            other => panic!("expected video frame, got {other:?}"),
        };
        let y8: Vec<u8> = recon_db.y.iter().map(|&v| v as u8).collect();
        let cb8: Vec<u8> = recon_db.cb.iter().map(|&v| v as u8).collect();
        let cr8: Vec<u8> = recon_db.cr.iter().map(|&v| v as u8).collect();
        assert_eq!(vf.planes[0].data, y8, "deblocked luma mismatch");
        assert_eq!(vf.planes[1].data, cb8, "deblocked cb mismatch");
        assert_eq!(vf.planes[2].data, cr8, "deblocked cr mismatch");

        // And through the registry option surface.
        let mut p = params(w, h);
        p.options.insert("deblock", "1");
        p.options.insert("qp", "45");
        let mut enc = make_encoder(&p).unwrap();
        enc.send_frame(&Frame::Video(frame)).unwrap();
        let pkt2 = enc.receive_packet().unwrap();
        assert_eq!(pkt2.data, pkt.data);
        let mut bad = params(w, h);
        bad.options.insert("deblock", "maybe");
        assert!(make_encoder(&bad).is_err());
    }

    /// Round 431 — the registry GOP loop: `gop=4` over 8 frames yields
    /// the IDR/P/P/P cadence (keyframe flags + NAL shapes), and every
    /// decoded frame is byte-identical to the encoder's own
    /// reconstruction chain (P frames predicting from the previous
    /// recon exactly as the decoder's DPB serves it).
    #[test]
    fn registry_gop_p_frames_round_trip_exact() {
        let (w, h) = (64u32, 48u32);
        let mut p = params(w, h);
        p.options.insert("gop", "4");
        p.options.insert("qp", "30");
        let mut enc = make_encoder(&p).unwrap();
        // Moving scene: shift a bright block per frame.
        let mk = |t: usize| -> VideoFrame {
            let mut f = synth_frame(w as usize, h as usize);
            for yy in 0..h as usize {
                for xx in 0..w as usize {
                    let bx = 4 + 3 * t;
                    let by = 4 + 2 * t;
                    if xx >= bx && xx < bx + 10 && yy >= by && yy < by + 8 {
                        f.planes[0].data[yy * w as usize + xx] = 230;
                    }
                }
            }
            f.pts = Some(t as i64 * 3000);
            f
        };
        // Reference chain through the direct entry points.
        let mut ref_recon: Option<crate::picture::YuvPicture> = None;
        let mut expected: Vec<(bool, Vec<u8>)> = Vec::new();
        for t in 0..8usize {
            let src = video_frame_to_picture(&mk(t), w, h, 8).unwrap();
            let is_idr = t % 4 == 0;
            let recon = if is_idr {
                let (_d, r, _s) = encode_idr_access_unit_opts(&src, 30, false, true).unwrap();
                r
            } else {
                let (_d, r, _s) =
                    encode_p_access_unit_opts(&src, ref_recon.as_ref().unwrap(), 30, false, true)
                        .unwrap();
                r
            };
            let y8: Vec<u8> = recon.y.iter().map(|&v| v as u8).collect();
            expected.push((is_idr, y8));
            ref_recon = Some(recon);
        }
        let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
        for (t, (exp_idr, exp_y)) in expected.iter().enumerate() {
            enc.send_frame(&Frame::Video(mk(t))).unwrap();
            let pkt = enc.receive_packet().unwrap();
            assert_eq!(pkt.flags.keyframe, *exp_idr, "frame {t} keyframe flag");
            let nals = crate::nal::iter_length_prefixed(&pkt.data).unwrap();
            if *exp_idr {
                assert_eq!(nals.len(), 3, "IDR AU carries SPS+PPS+IDR");
                assert_eq!(nals[2].header.nal_unit_type, NalUnitType::Idr);
            } else {
                assert_eq!(nals.len(), 1, "P AU is a single NonIDR NAL");
                assert_eq!(nals[0].header.nal_unit_type, NalUnitType::NonIdr);
            }
            dec.send_packet(&pkt).unwrap();
            let vf = match dec.receive_frame().unwrap() {
                Frame::Video(vf) => vf,
                other => panic!("expected video frame, got {other:?}"),
            };
            assert_eq!(&vf.planes[0].data, exp_y, "frame {t}: decode != recon");
        }
    }

    /// Round 431 — GOP rate sanity: with mostly-static content the P
    /// frames must cost a small fraction of the IDR frames, and the
    /// whole-GOP byte count must undercut the all-intra encode of the
    /// same frames.
    #[test]
    fn registry_gop_beats_all_intra_on_static_content() {
        let (w, h) = (96u32, 64u32);
        let frame = synth_frame(w as usize, h as usize);
        let run = |gop: &str| -> usize {
            let mut p = params(w, h);
            p.options.insert("gop", gop);
            let mut enc = make_encoder(&p).unwrap();
            let mut total = 0usize;
            for t in 0..6i64 {
                let mut f = frame.clone();
                f.pts = Some(t);
                enc.send_frame(&Frame::Video(f)).unwrap();
                total += enc.receive_packet().unwrap().data.len();
            }
            total
        };
        let all_intra = run("1");
        let gop6 = run("6");
        assert!(
            gop6 * 3 < all_intra,
            "static GOP ({gop6}) must be well under a third of all-intra ({all_intra})"
        );
    }

    /// Round 431 — single-byte mutation gate over a P access unit: feed
    /// the intact IDR AU, then every low-bit flip / inversion of every
    /// P-AU byte, into a fresh decoder session. Clean error or frame —
    /// never a panic.
    #[test]
    fn mutation_gate_over_p_access_unit() {
        let (w, h) = (48u32, 48u32);
        let f0 = synth_frame(w as usize, h as usize);
        let mut f1 = synth_frame(w as usize, h as usize);
        for (i, v) in f1.planes[0].data.iter_mut().enumerate() {
            if i % 37 == 0 {
                *v = v.wrapping_add(40);
            }
        }
        let src0 = video_frame_to_picture(&f0, w, h, 8).unwrap();
        let src1 = video_frame_to_picture(&f1, w, h, 8).unwrap();
        let (idr_au, recon0, _s) = encode_idr_access_unit_opts(&src0, 38, false, true).unwrap();
        let (p_au, _recon1, _s) =
            encode_p_access_unit_opts(&src1, &recon0, 38, false, true).unwrap();
        for i in 0..p_au.len() {
            for mask in [0x01u8, 0xFF] {
                let mut mutated = p_au.clone();
                mutated[i] ^= mask;
                let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
                let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
                let pkt0 = Packet::new(0, TimeBase::new(1, 90_000), idr_au.clone());
                dec.send_packet(&pkt0).unwrap();
                let _ = dec.receive_frame();
                let pkt1 = Packet::new(0, TimeBase::new(1, 90_000), mutated);
                if dec.send_packet(&pkt1).is_ok() {
                    let _ = dec.receive_frame();
                }
            }
        }
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

    /// Determinism: encoding the same frame twice (fresh encoder each
    /// time) must produce byte-identical access units — no hidden
    /// state, no non-deterministic RD tie-breaks. Both entropy shapes.
    #[test]
    fn encode_is_deterministic() {
        let src = video_frame_to_picture(&synth_frame(100, 60), 100, 60, 8).unwrap();
        for &cm in &[false, true] {
            let (a, _, _) = encode_idr_access_unit_opts(&src, 33, false, cm).unwrap();
            let (b, _, _) = encode_idr_access_unit_opts(&src, 33, false, cm).unwrap();
            assert_eq!(a, b, "cm_init {cm}");
        }
    }

    /// QCIF-class rate/PSNR characterization across the QP grid, on
    /// **both** entropy shapes: every point must round-trip recon-exact
    /// through the registered decoder, PSNR must be monotone
    /// non-increasing and rate monotone non-increasing in QP, the
    /// reconstruction must be identical across shapes (the entropy
    /// layer is lossless), and the round-431 `cm_init` shape must
    /// out-compress the Baseline collapse at every QP. Run with
    /// `--nocapture` to read the measured curves + the rate movement.
    #[test]
    fn rate_psnr_curve_qcif() {
        let (w, h) = (176u32, 144u32);
        let mut f = synth_frame(w as usize, h as usize);
        // Add a diagonal feature field so the frame isn't trivially flat.
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                if (xx + yy) % 23 == 0 {
                    f.planes[0].data[yy * w as usize + xx] = 235;
                }
            }
        }
        let src = video_frame_to_picture(&f, w, h, 8).unwrap();
        let pixels = (w * h) as f64;
        for &cm_init in &[false, true] {
            let mut prev_psnr = f64::INFINITY;
            let mut prev_len = usize::MAX;
            for &qp in &[4i32, 10, 16, 22, 28, 34, 40, 46, 51] {
                let (stream, recon, _stats) =
                    encode_idr_access_unit_opts(&src, qp, false, cm_init).unwrap();
                // Recon-exactness through the registered decoder.
                let dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
                let mut dec = crate::decoder::make_decoder(&dparams).unwrap();
                let pkt = Packet::new(0, TimeBase::new(1, 90_000), stream.clone());
                dec.send_packet(&pkt).unwrap();
                let vf = match dec.receive_frame().unwrap() {
                    Frame::Video(vf) => vf,
                    other => panic!("expected video frame, got {other:?}"),
                };
                let y8: Vec<u8> = recon.y.iter().map(|&v| v as u8).collect();
                assert_eq!(
                    vf.planes[0].data, y8,
                    "qp {qp} cm{cm_init}: decode != recon"
                );
                // The entropy layer must not touch the reconstruction.
                let (other_stream, other_recon, _) =
                    encode_idr_access_unit_opts(&src, qp, false, !cm_init).unwrap();
                assert_eq!(recon.y, other_recon.y, "qp {qp}: recon differs by shape");
                if cm_init {
                    let saved = 100.0 * (1.0 - stream.len() as f64 / other_stream.len() as f64);
                    assert!(
                        stream.len() < other_stream.len(),
                        "qp {qp}: cm_init {} must beat baseline {}",
                        stream.len(),
                        other_stream.len()
                    );
                    eprintln!(
                        "qcif qp {qp:2}: cm_init saves {saved:5.1}% \
                         ({} vs {} bytes)",
                        stream.len(),
                        other_stream.len()
                    );
                }
                // Curve shape.
                let mse: f64 = src
                    .y
                    .iter()
                    .zip(recon.y.iter())
                    .map(|(&a, &b)| {
                        let d = a as f64 - b as f64;
                        d * d
                    })
                    .sum::<f64>()
                    / pixels;
                let psnr = if mse == 0.0 {
                    99.0
                } else {
                    10.0 * (255.0f64 * 255.0 / mse).log10()
                };
                let bpp = stream.len() as f64 * 8.0 / pixels;
                eprintln!(
                    "qcif cm{} qp {qp:2}: {:6} bytes  {bpp:5.3} bpp  {psnr:5.2} dB",
                    u8::from(cm_init),
                    stream.len()
                );
                // Monotone in the meaningful range — above ~60 dB the
                // points are all effectively lossless (sub-1-LSB noise)
                // and their ordering is quantization-rounding luck.
                assert!(
                    psnr.min(60.0) <= prev_psnr.min(60.0) + 0.01,
                    "qp {qp}: PSNR rose ({psnr:.2} after {prev_psnr:.2})"
                );
                assert!(stream.len() <= prev_len, "qp {qp}: rate rose");
                if qp == 4 {
                    assert!(psnr >= 46.0, "qp 4 PSNR {psnr:.2}");
                }
                if qp == 51 {
                    assert!(bpp <= 4.0, "qp 51 bpp {bpp:.3}");
                }
                prev_psnr = psnr;
                prev_len = stream.len();
            }
        }
    }
}
