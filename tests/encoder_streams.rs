//! MD5-pinned encoder stream fixtures (round 452).
//!
//! Each case drives the registered encoder over a deterministic
//! synthetic scene, then (a) pins the RFC 1321 MD5 of the concatenated
//! access units — the encoder's exact output on every platform (its RD
//! costs are evaluated without transcendental calls) — and (b) decodes
//! the stream through the registered decoder and pins the MD5 of every
//! output plane, sample-exact against the encoder's own
//! reconstruction chain (the in-tree unit tests assert that equality
//! directly; here the digests freeze it as a fixture).
//!
//! Re-pin only for a deliberate encoder change, and say why in the
//! CHANGELOG.

mod md5;

use oxideav_core::format::PixelFormat;
use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::{CodecId, CodecParameters, Frame};

/// Deterministic moving scene with independent per-frame noise: a
/// diagonal gradient, a bright square translating (3, 2) per frame, a
/// noise band at the bottom, and ±5 noise everywhere.
fn scene(w: usize, h: usize, t: u32) -> VideoFrame {
    let mut y = vec![0u8; w * h];
    let mut s = 0x2545_F491u32 ^ t.wrapping_mul(0x9E37_79B9);
    let (bx, by) = ((8 + 3 * t) as usize, (6 + 2 * t) as usize);
    for yy in 0..h {
        for xx in 0..w {
            let mut v = 40 + ((xx as i32 + 2 * yy as i32) % 140);
            if xx >= bx && xx < bx + 12 && yy >= by && yy < by + 10 {
                v = 220;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
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

struct Case {
    name: &'static str,
    w: u32,
    h: u32,
    frames: u32,
    options: &'static [(&'static str, &'static str)],
    /// MD5 of the concatenated access units.
    stream_md5: &'static str,
    /// MD5 over every decoded plane of every frame, in output order.
    output_md5: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "p_multi_ref",
        w: 64,
        h: 48,
        frames: 6,
        options: &[("gop", "6"), ("refs", "3"), ("qp", "24")],
        stream_md5: "79ec391406643e93cb4201fc239e2c5b",
        output_md5: "9ffa4477ba727601f9dace68946bc350",
    },
    Case {
        name: "b_low_delay",
        w: 64,
        h: 48,
        frames: 6,
        options: &[("gop", "6"), ("refs", "2"), ("b", "1"), ("qp", "24")],
        stream_md5: "49275b6cf75149e6c36f1cd590fcb309",
        output_md5: "55d667f2b4e459a7a8d8c0573547a20c",
    },
    Case {
        name: "b_low_delay_baseline_deblock",
        w: 64,
        h: 48,
        frames: 5,
        options: &[
            ("gop", "5"),
            ("refs", "2"),
            ("b", "1"),
            ("qp", "30"),
            ("cm_init", "0"),
            ("deblock", "1"),
        ],
        stream_md5: "cbc0d5eb74e0448960d83f7457f1cf78",
        output_md5: "9585661b68e2fe365c8f39a6351ebeef",
    },
];

fn run_case(c: &Case) -> (String, String, usize) {
    let mut p = CodecParameters::video(CodecId::new("evc"));
    p.width = Some(c.w);
    p.height = Some(c.h);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    for (k, v) in c.options {
        p.options.insert(*k, *v);
    }
    let mut enc = oxideav_evc::encoder::make_encoder(&p).expect("encoder");
    let mut dec = oxideav_evc::decoder::make_decoder(&CodecParameters::video(CodecId::new("evc")))
        .expect("decoder");
    let mut stream = Vec::new();
    let mut output = Vec::new();
    for t in 0..c.frames {
        enc.send_frame(&Frame::Video(scene(c.w as usize, c.h as usize, t)))
            .expect("send_frame");
        let pkt = enc.receive_packet().expect("packet");
        stream.extend_from_slice(&pkt.data);
        dec.send_packet(&pkt).expect("send_packet");
        match dec.receive_frame().expect("frame") {
            Frame::Video(vf) => {
                assert_eq!(vf.planes.len(), 3);
                for pl in &vf.planes {
                    output.extend_from_slice(&pl.data);
                }
            }
            other => panic!("expected video, got {other:?}"),
        }
    }
    (md5::hex(&stream), md5::hex(&output), stream.len())
}

#[test]
fn encoder_stream_fixtures() {
    let mut failures = Vec::new();
    for c in CASES {
        let (s, o, len) = run_case(c);
        eprintln!("fixture {}: {len} bytes, stream {s}, output {o}", c.name);
        if s != c.stream_md5 || o != c.output_md5 {
            failures.push(format!(
                "{}: stream {s} (pinned {}), output {o} (pinned {})",
                c.name, c.stream_md5, c.output_md5
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "fixture divergence:\n{}",
        failures.join("\n")
    );
}
