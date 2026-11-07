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
//! length, `sizes=176x144,96x64` the corpus geometries, `scene=mixed`
//! the second corpus (see [`scene_mixed`]), `out=<prefix>` writes each
//! (size, QP) stream to `<prefix>-<w>x<h>-q<qp>.evc` (the length-prefixed
//! NAL framing, for black-box tools). One CSV row per (size, QP):
//! `size,qp,bytes,psnr_y,psnr_u,psnr_v,ssim_y`, PSNR and luma SSIM over
//! every decoded frame of the sequence against the source (the stream
//! is decoded through the registered decoder, so the numbers are the
//! decoder's output, not the encoder's belief). SSIM is the mean over
//! 8×8 windows on a 4-sample grid with `C1 = (0.01·255)²`,
//! `C2 = (0.03·255)²`.

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

/// The second corpus (round 483): content whose local activity varies
/// — a near-flat sky gradient with ±1 noise on top, a high-contrast
/// texture field (±30 grain over a coarse pattern) on the bottom left,
/// a hard-edged block pattern on the bottom right, and a mid-grey disc
/// translating (2, 1) per frame across the sky.
fn scene_mixed(w: usize, h: usize, t: u32) -> VideoFrame {
    let mut y = vec![0u8; w * h];
    let mut s = 0x1234_5679u32 ^ t.wrapping_mul(0x85EB_CA6B);
    let (cx, cy) = ((w / 4 + 2 * t as usize) as i32, (h / 4 + t as usize) as i32);
    let r2 = ((h / 6) * (h / 6)) as i32;
    for yy in 0..h {
        for xx in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (s >> 24) as i32; // 0..=255
            let v = if yy < h / 2 {
                let mut v = 90 + (yy as i32 * 60) / (h as i32 / 2) + (n % 3) - 1;
                let (dx, dy) = (xx as i32 - cx, yy as i32 - cy);
                if dx * dx + dy * dy < r2 {
                    v = 150 + (n % 3) - 1;
                }
                v
            } else if xx < w / 2 {
                let base = if ((xx / 6) + (yy / 6)) % 2 == 0 {
                    80
                } else {
                    120
                };
                base + (n % 61) - 30
            } else {
                let block = ((xx / 8) * 7 + (yy / 8) * 3) % 4;
                [40, 200, 110, 160][block] + (n % 5) - 2
            };
            y[yy * w + xx] = v.clamp(0, 255) as u8;
        }
    }
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let cb: Vec<u8> = (0..cw * ch)
        .map(|i| (120 + ((i % cw) * 20 / cw) as u32 + ((i / cw) * 10 / ch) as u32) as u8)
        .collect();
    let cr: Vec<u8> = (0..cw * ch)
        .map(|i| (128 + ((i / cw) * 24 / ch) as u32) as u8 - ((i % cw) * 8 / cw) as u8)
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

/// Mean luma SSIM over 8×8 windows on a 4-sample grid: returns the
/// window sum and count.
fn ssim_windows(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> (f64, f64) {
    let c1 = (0.01f64 * 255.0).powi(2);
    let c2 = (0.03f64 * 255.0).powi(2);
    let (mut sum, mut n) = (0f64, 0f64);
    let mut y = 0;
    while y + 8 <= h {
        let mut x = 0;
        while x + 8 <= w {
            let (mut ma, mut mb, mut va, mut vb, mut cov) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for yy in y..y + 8 {
                for xx in x..x + 8 {
                    let p = f64::from(a[yy * sa + xx]);
                    let q = f64::from(b[yy * sb + xx]);
                    ma += p;
                    mb += q;
                    va += p * p;
                    vb += q * q;
                    cov += p * q;
                }
            }
            let k = 64.0;
            ma /= k;
            mb /= k;
            va = va / k - ma * ma;
            vb = vb / k - mb * mb;
            cov = cov / k - ma * mb;
            sum += ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            n += 1.0;
            x += 4;
        }
        y += 4;
    }
    (sum, n)
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
    let mut mixed = false;
    let mut out: Option<String> = None;
    for arg in std::env::args().skip(1) {
        let Some((k, v)) = arg.split_once('=') else {
            eprintln!("ignoring argument {arg:?} (expected k=v)");
            continue;
        };
        match k {
            "qps" => qps = v.split(',').map(|q| q.parse().expect("qp")).collect(),
            "frames" => frames = v.parse().expect("frames"),
            "scene" => mixed = v == "mixed",
            "out" => out = Some(v.to_string()),
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
    println!("size,qp,bytes,psnr_y,psnr_u,psnr_v,ssim_y");
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
            // Send everything, flush, decode everything, flush: the
            // hierarchical shapes reorder, so pair by pts.
            let sources: Vec<VideoFrame> = (0..frames)
                .map(|t| {
                    if mixed {
                        scene_mixed(w as usize, h as usize, t)
                    } else {
                        scene(w as usize, h as usize, t)
                    }
                })
                .collect();
            let (mut ssim_sum, mut ssim_n) = (0f64, 0f64);
            let mut packets = Vec::new();
            for src in &sources {
                enc.send_frame(&Frame::Video(src.clone()))
                    .expect("send_frame");
                while let Ok(pkt) = enc.receive_packet() {
                    packets.push(pkt);
                }
            }
            enc.flush().expect("flush");
            while let Ok(pkt) = enc.receive_packet() {
                packets.push(pkt);
            }
            if let Some(prefix) = &out {
                let all: Vec<u8> = packets.iter().flat_map(|p| p.data.iter().copied()).collect();
                std::fs::write(format!("{prefix}-{w}x{h}-q{qp}.evc"), all).expect("write stream");
            }
            for pkt in &packets {
                bytes += pkt.data.len();
                dec.send_packet(pkt).expect("send_packet");
            }
            dec.flush().expect("decoder flush");
            let mut decoded = 0usize;
            while let Ok(Frame::Video(vf)) = dec.receive_frame() {
                decoded += 1;
                let t = (vf.pts.expect("pts") / 3000) as usize;
                let src = &sources[t];
                let (a, b) = (&src.planes[0], &vf.planes[0]);
                let (win_sum, win_n) =
                    ssim_windows(&a.data, a.stride, &b.data, b.stride, w as usize, h as usize);
                ssim_sum += win_sum;
                ssim_n += win_n;
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
            assert_eq!(decoded, frames as usize, "every picture decodes");
            println!(
                "{w}x{h},{qp},{bytes},{:.3},{:.3},{:.3},{:.5}",
                psnr(sse[0], n[0]),
                psnr(sse[1], n[1]),
                psnr(sse[2], n[2]),
                ssim_sum / ssim_n
            );
        }
    }
}
