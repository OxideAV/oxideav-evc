//! Whole-encoder round trip on fuzzed content: an IDR + inter GOP at a
//! fuzzed geometry / QP / bit depth / tool set is encoded through the
//! registered encoder (exact-cost decide pass, RDOQ trellis, rate model)
//! and every access unit decoded through the registered decoder, which
//! must reproduce the encoder's reconstruction sample for sample and
//! never panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::format::PixelFormat;
use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::{CodecId, CodecParameters, Encoder, Frame};

fn plane(bytes: &[u8], cursor: &mut usize, n: usize, bytes_per: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * bytes_per);
    for _ in 0..n * bytes_per {
        let b = if bytes.is_empty() {
            0x80
        } else {
            bytes[*cursor % bytes.len()]
        };
        *cursor += 1;
        out.push(b);
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let w = 8 * (1 + u32::from(data[0] % 8)); // 8..=64
    let h = 8 * (1 + u32::from(data[1] % 8));
    let qp = i32::from(data[2] % 52);
    let flags = data[3];
    let cm_init = flags & 1 != 0;
    let deblock = flags & 2 != 0;
    let b_pictures = flags & 4 != 0;
    let refs = 1 + u32::from((flags >> 3) & 1);
    let ten_bit = flags & 0x10 != 0;
    let frames = 2 + usize::from((flags >> 5) & 1);
    let (pf, bytes_per, max_val) = if ten_bit {
        (PixelFormat::Yuv420P10Le, 2usize, 1023u16)
    } else {
        (PixelFormat::Yuv420P, 1, 255)
    };

    let mut p = CodecParameters::video(CodecId::new("evc"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.options.insert("qp", qp.to_string());
    p.options.insert("gop", frames.to_string());
    p.options.insert("refs", refs.to_string());
    p.options.insert("cm_init", if cm_init { "1" } else { "0" });
    p.options.insert("deblock", if deblock { "1" } else { "0" });
    p.options.insert("b", if b_pictures { "1" } else { "0" });
    let mut enc = oxideav_evc::encoder::make_evc_encoder(&p).expect("encoder");
    let mut dec = oxideav_evc::decoder::make_decoder(&CodecParameters::video(CodecId::new("evc")))
        .expect("decoder");

    let payload = &data[4..];
    let mut cursor = 0usize;
    let (cw, ch) = ((w as usize).div_ceil(2), (h as usize).div_ceil(2));
    for t in 0..frames {
        let mut y = plane(payload, &mut cursor, (w * h) as usize, bytes_per);
        let mut cb = plane(payload, &mut cursor, cw * ch, bytes_per);
        let mut cr = plane(payload, &mut cursor, cw * ch, bytes_per);
        if ten_bit {
            for v in [&mut y, &mut cb, &mut cr] {
                for px in v.chunks_mut(2) {
                    let s = u16::from_le_bytes([px[0], px[1]]).min(max_val);
                    px.copy_from_slice(&s.to_le_bytes());
                }
            }
        }
        let frame = VideoFrame {
            pts: Some(t as i64),
            planes: vec![
                VideoPlane {
                    stride: w as usize * bytes_per,
                    data: y,
                },
                VideoPlane {
                    stride: cw * bytes_per,
                    data: cb,
                },
                VideoPlane {
                    stride: cw * bytes_per,
                    data: cr,
                },
            ],
        };
        enc.send_frame(&Frame::Video(frame)).expect("send_frame");
        let pkt = enc.receive_packet().expect("packet");
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Video(vf) = dec.receive_frame().expect("frame") else {
            panic!("expected a video frame")
        };
        let recon = enc.last_recon().expect("recon");
        let planes = [&recon.y, &recon.cb, &recon.cr];
        for (c, want) in planes.iter().enumerate() {
            let got = &vf.planes[c];
            let (pw, ph) = if c == 0 {
                (w as usize, h as usize)
            } else {
                (cw, ch)
            };
            let stride = if c == 0 {
                recon.y_stride()
            } else {
                recon.c_stride()
            };
            for yy in 0..ph {
                for xx in 0..pw {
                    let want_v = want[yy * stride + xx];
                    let got_v = if ten_bit {
                        let i = yy * got.stride + 2 * xx;
                        u16::from_le_bytes([got.data[i], got.data[i + 1]])
                    } else {
                        u16::from(got.data[yy * got.stride + xx])
                    };
                    assert_eq!(got_v, want_v, "frame {t} plane {c} ({xx},{yy}): decode != recon");
                }
            }
        }
    }
});
