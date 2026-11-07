//! Core 0.1.43 `Frame::StreamParams` contract on a spliced stream: two
//! independently encoded sequences of different geometry / bit depth
//! fed through one decoder. A new SPS at an IDR may change the output
//! shape; the decoder must announce it exactly once, immediately before
//! the first frame of the new shape, with `output_params()` and the
//! generation counter already moved.

use oxideav_core::format::PixelFormat;
use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet};

fn frame(w: usize, h: usize, bytes_per_sample: usize, t: u32) -> VideoFrame {
    let plane = |pw: usize, ph: usize, base: u32| {
        let mut data = Vec::with_capacity(pw * ph * bytes_per_sample);
        for y in 0..ph {
            for x in 0..pw {
                let v = (base + (x as u32 * 3 + y as u32 * 5 + t * 7) % 90) as u16;
                if bytes_per_sample == 1 {
                    data.push(v as u8);
                } else {
                    data.extend_from_slice(&(v << 2).to_le_bytes());
                }
            }
        }
        VideoPlane {
            stride: pw * bytes_per_sample,
            data,
        }
    };
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    VideoFrame {
        pts: Some(i64::from(t)),
        planes: vec![plane(w, h, 60), plane(cw, ch, 100), plane(cw, ch, 120)],
    }
}

/// Encode `n` frames at `w`x`h` and return the packets.
fn encode(w: u32, h: u32, pf: PixelFormat, n: u32, t0: u32) -> Vec<Packet> {
    let mut p = CodecParameters::video(CodecId::new("evc"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.options.insert("qp", "40");
    p.options.insert("gop", "2");
    let mut enc = oxideav_evc::encoder::make_encoder(&p).expect("encoder");
    let bps = if pf == PixelFormat::Yuv420P { 1 } else { 2 };
    let mut out = Vec::new();
    for t in t0..t0 + n {
        enc.send_frame(&Frame::Video(frame(w as usize, h as usize, bps, t)))
            .expect("send_frame");
        while let Ok(pkt) = enc.receive_packet() {
            out.push(pkt);
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        out.push(pkt);
    }
    out
}

type Shape = (Option<u32>, Option<u32>, Option<PixelFormat>);

fn shape(p: &CodecParameters) -> Shape {
    (p.width, p.height, p.pixel_format)
}

#[derive(Debug, PartialEq)]
enum Ev {
    Pic(u32, u32),
    Params(u32, u32, Option<PixelFormat>),
}

#[test]
fn spliced_sequences_announce_each_shape_change_once() {
    let segments = [
        (64u32, 48u32, PixelFormat::Yuv420P),
        (96, 64, PixelFormat::Yuv420P),
        (96, 64, PixelFormat::Yuv420P10Le),
        (62, 46, PixelFormat::Yuv420P10Le), // cropped geometry
    ];
    let mut packets = Vec::new();
    for (i, &(w, h, pf)) in segments.iter().enumerate() {
        packets.extend(encode(w, h, pf, 3, 3 * i as u32));
    }
    for drain_late in [false, true] {
        let mut dec =
            oxideav_evc::decoder::make_decoder(&CodecParameters::video(CodecId::new("evc")))
                .expect("decoder");
        assert!(dec.output_params().is_none(), "nothing parsed yet");
        let mut events = Vec::new();
        let mut gen_seen = dec.output_params_generation();
        let mut cur: Option<Shape> = None;
        let mut pull = |dec: &mut Box<dyn oxideav_core::Decoder>, events: &mut Vec<Ev>| loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    let p = dec.output_params().expect("shape known at a frame");
                    let (w, h) = (p.width.unwrap(), p.height.unwrap());
                    let bps = if p.pixel_format == Some(PixelFormat::Yuv420P) {
                        1
                    } else {
                        2
                    };
                    assert_eq!(vf.planes.len(), 3);
                    assert_eq!(
                        vf.planes[0].stride,
                        w as usize * bps,
                        "frame matches params"
                    );
                    assert_eq!(vf.planes[0].data.len(), (w * h) as usize * bps);
                    if let Some(c) = cur {
                        assert_eq!(c, shape(&p), "no silent change between frames");
                    }
                    assert_eq!(dec.output_params_generation(), gen_seen, "no silent bump");
                    cur = Some(shape(&p));
                    events.push(Ev::Pic(w, h));
                }
                Ok(Frame::StreamParams(sp)) => {
                    let now = dec.output_params().expect("params");
                    assert_eq!(
                        shape(&sp),
                        shape(&now),
                        "event payload equals output_params()"
                    );
                    assert_ne!(Some(shape(&now)), cur, "never an event without a change");
                    let g = dec.output_params_generation();
                    assert_eq!(g, gen_seen.wrapping_add(1), "one bump per event");
                    gen_seen = g;
                    cur = Some(shape(&now));
                    events.push(Ev::Params(
                        now.width.unwrap(),
                        now.height.unwrap(),
                        now.pixel_format,
                    ));
                }
                Ok(_) => panic!("unexpected frame kind"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("decode: {e}"),
            }
        };
        for pkt in &packets {
            dec.send_packet(pkt).expect("send_packet");
            if !drain_late {
                pull(&mut dec, &mut events);
            }
        }
        dec.flush().expect("flush");
        pull(&mut dec, &mut events);
        let mut want = Vec::new();
        for (i, &(w, h, pf)) in segments.iter().enumerate() {
            if i > 0 {
                want.push(Ev::Params(w, h, Some(pf)));
            }
            for _ in 0..3 {
                want.push(Ev::Pic(w, h));
            }
        }
        assert_eq!(events, want, "drain_late={drain_late}");
        assert_eq!(dec.output_params_generation(), 3);
    }
}

#[test]
fn output_params_known_from_the_sps_before_the_first_frame() {
    let packets = encode(62, 46, PixelFormat::Yuv420P10Le, 1, 0);
    let mut dec =
        oxideav_evc::decoder::make_decoder(&CodecParameters::video(CodecId::new("evc"))).unwrap();
    dec.send_packet(&packets[0]).unwrap();
    let p = dec
        .output_params()
        .expect("known after the key access unit");
    assert_eq!((p.width, p.height), (Some(62), Some(46)), "cropped size");
    assert_eq!(p.pixel_format, Some(PixelFormat::Yuv420P10Le));
    assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv420P10Le));
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    assert_eq!(
        dec.output_params_generation(),
        0,
        "the first shape is no change"
    );
}

#[test]
fn encoder_refuses_a_stream_params_event() {
    let mut p = CodecParameters::video(CodecId::new("evc"));
    p.width = Some(64);
    p.height = Some(48);
    let mut enc = oxideav_evc::encoder::make_encoder(&p).unwrap();
    let mut np = p.clone();
    np.width = Some(96);
    let err = enc
        .send_frame(&Frame::StreamParams(Box::new(np)))
        .expect_err("cannot re-configure");
    assert!(matches!(err, Error::Unsupported(_)), "{err}");
}
