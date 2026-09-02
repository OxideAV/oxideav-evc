//! Rate/PSNR curve of the registered encoder over the crate's synthetic
//! encode corpus — the measurement tool behind the README's encoder
//! numbers and the per-tool BD-rate deltas in the CHANGELOG.
//!
//! ```text
//! cargo run --release --example rd_curve -- [k=v ...]
//! ```
//!
//! Every `k=v` argument is passed to the encoder as an option (`gop`,
//! `refs`, `b`, `qp`, `bitrate`, `fps`, `deblock`, `cm_init`, …);
//! `qps=22,27,32,37` overrides the QP ladder, `frames=N` the sequence
//! length, `sizes=176x144,96x64` the corpus geometries. One CSV row per
//! (size, QP): `size,qp,bytes,psnr_y,psnr_u,psnr_v`, PSNR over every
//! decoded frame of the sequence against the source (the stream is
//! decoded through the registered decoder, so the numbers are the
//! decoder's output, not the encoder's belief).

use std::collections::BTreeMap;

use oxideav_core::format::PixelFormat;
use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::{CodecId, CodecParameters, Frame};

/// Deterministic moving scene with independent per-frame noise: a
/// diagonal gradient, a checker texture, a bright square translating
/// (3, 2) per frame, a noise band at the bottom, and ±5 noise everywhere.
fn scene(w: usize, h: usize, t: u32) -> VideoFrame {
    let mut y = vec![0u8; w * h];
    let mut s = 0x2545_F491u32 ^ t.wrapping_mul(0x9E37_79B9);
    let (bx, by) = ((8 + 3 * t) as usize, (6 + 2 * t) as usize);
    for yy in 0..h {
        for xx in 0..w {
            let mut v = 40 + ((xx as i32 + 2 * yy as i32) % 140);
            if xx >= w / 2 && yy < h / 2 && ((xx / 3) + (yy / 3)) % 2 == 0 {
                v += 35;
            }
            if xx >= bx && xx < bx + 12 && yy >= by && yy < by + 10 {
                v = 220;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            if yy >= h - h / 8 {
                v += ((s >> 20) % 41) as i32 - 20;
            }
            v += ((s >> 24) % 11) as i32 - 5;
            y[yy * w + xx] = v.clamp(0, 255) as u8;
        }
    }
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let cb: Vec<u8> = (0..cw * ch)
        .map(|i| (110 + (i % cw + i / cw + t as usize) % 40) as u8)
        .collect();
    let cr: Vec<u8> = (0..cw * ch)
        .map(|i| (130 + (2 * (i % cw) + i / cw) % 50) as u8)
        .collect();
    VideoFrame {
        pts: Some(t as i64 * 3000),
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

fn psnr(sse: f64, n: f64) -> f64 {
    if sse == 0.0 {
        return 99.0;
    }
    10.0 * (255.0 * 255.0 * n / sse).log10()
}

fn main() {
    let mut opts: BTreeMap<String, String> = BTreeMap::new();
    let mut qps = vec![22i32, 27, 32, 37];
    let mut frames = 8u32;
    let mut sizes = vec![(176u32, 144u32), (96, 64), (64, 48)];
    for arg in std::env::args().skip(1) {
        let Some((k, v)) = arg.split_once('=') else {
            eprintln!("ignoring argument {arg:?} (expected k=v)");
            continue;
        };
        match k {
            "qps" => qps = v.split(',').map(|q| q.parse().expect("qp")).collect(),
            "frames" => frames = v.parse().expect("frames"),
            "sizes" => {
                sizes = v
                    .split(',')
                    .map(|s| {
                        let (w, h) = s.split_once('x').expect("WxH");
                        (w.parse().expect("w"), h.parse().expect("h"))
                    })
                    .collect()
            }
            _ => {
                opts.insert(k.to_string(), v.to_string());
            }
        }
    }
    println!("size,qp,bytes,psnr_y,psnr_u,psnr_v");
    for &(w, h) in &sizes {
        for &qp in &qps {
            let mut p = CodecParameters::video(CodecId::new("evc"));
            p.width = Some(w);
            p.height = Some(h);
            p.pixel_format = Some(PixelFormat::Yuv420P);
            p.options.insert("qp", qp.to_string());
            for (k, v) in &opts {
                p.options.insert(k.clone(), v.clone());
            }
            let mut enc = oxideav_evc::encoder::make_encoder(&p).expect("encoder");
            let mut dec =
                oxideav_evc::decoder::make_decoder(&CodecParameters::video(CodecId::new("evc")))
                    .expect("decoder");
            let mut bytes = 0usize;
            let mut sse = [0f64; 3];
            let mut n = [0f64; 3];
            for t in 0..frames {
                let src = scene(w as usize, h as usize, t);
                enc.send_frame(&Frame::Video(src.clone()))
                    .expect("send_frame");
                let pkt = enc.receive_packet().expect("packet");
                bytes += pkt.data.len();
                dec.send_packet(&pkt).expect("send_packet");
                let Frame::Video(vf) = dec.receive_frame().expect("frame") else {
                    panic!("expected video")
                };
                for c in 0..3 {
                    let (pw, ph) = if c == 0 {
                        (w as usize, h as usize)
                    } else {
                        ((w as usize).div_ceil(2), (h as usize).div_ceil(2))
                    };
                    let a = &src.planes[c];
                    let b = &vf.planes[c];
                    for yy in 0..ph {
                        for xx in 0..pw {
                            let d = a.data[yy * a.stride + xx] as f64
                                - b.data[yy * b.stride + xx] as f64;
                            sse[c] += d * d;
                        }
                    }
                    n[c] += (pw * ph) as f64;
                }
            }
            println!(
                "{w}x{h},{qp},{bytes},{:.3},{:.3},{:.3}",
                psnr(sse[0], n[0]),
                psnr(sse[1], n[1]),
                psnr(sse[2], n[2])
            );
        }
    }
}
