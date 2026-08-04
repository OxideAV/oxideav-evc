//! ISO/IEC 23094-4 conformance-bitstream gate.
//!
//! Drives the registered decoder over the official EVC conformance
//! streams (the electronic attachments to ISO/IEC 23094-4, staged in the
//! private docs repo) and checks every decoded picture, per plane,
//! against the published `.opl` MD5 lists.
//!
//! The corpus is **data, not source** — it lives outside this repo and
//! the test is gated on its presence: when the staging directory is
//! absent (e.g. on CI, where only this crate is checked out) the test
//! prints a skip note and passes. Point `EVC_CONFORMANCE_DIR` at a
//! `conformance/` staging root to override the default
//! `../../docs/video/evc/conformance` (relative to the crate).
//!
//! `EVC_CONF_FILTER=<substring>[,<substring>…]` restricts the run to
//! matching stream names — the local triage loop.
//!
//! Reference-hash conventions honoured (from the corpus' own `.txt`
//! sheets): the `.opl` rows are in output (display) order; hashes are
//! over each plane's tightly-packed samples at either the stream's
//! native bit depth (1 byte/sample at 8-bit, 2-byte little-endian
//! otherwise) or — some submitters ran their decode at
//! `--output_bit_depth 10` — over the 8-bit samples up-shifted by 2 into
//! 2-byte little-endian. Both encodings are accepted; a plane fails only
//! when neither matches.

mod md5;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use oxideav_core::{Decoder, Packet, TimeBase};
use oxideav_evc::decoder::EvcDecoder;
use oxideav_evc::nal::{iter_length_prefixed, NalUnitType};

/// One `.opl` row: output-order picture dimensions + per-plane hashes.
struct OplRow {
    poc: i64,
    width: usize,
    height: usize,
    md5: [String; 3],
}

fn parse_opl(path: &Path) -> Result<Vec<OplRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() {
            continue;
        }
        if f.len() < 6 {
            return Err(format!(
                "{path:?}:{}: short row ({} fields)",
                ln + 1,
                f.len()
            ));
        }
        rows.push(OplRow {
            poc: f[0]
                .parse::<i64>()
                .map_err(|e| format!("{path:?}:{}: poc: {e}", ln + 1))?,
            width: f[1]
                .parse::<usize>()
                .map_err(|e| format!("{path:?}:{}: w: {e}", ln + 1))?,
            height: f[2]
                .parse::<usize>()
                .map_err(|e| format!("{path:?}:{}: h: {e}", ln + 1))?,
            md5: [
                f[3].to_lowercase(),
                f[4].to_lowercase(),
                f[5].to_lowercase(),
            ],
        });
    }
    Ok(rows)
}

/// A decoded output picture, flattened to per-plane byte vectors exactly
/// as emitted by the decoder (1 byte/sample for 8-bit, 2-byte LE
/// otherwise).
struct OutPic {
    planes: Vec<Vec<u8>>,
    strides: Vec<usize>,
}

/// Decode a whole length-prefixed elementary stream, returning the
/// output pictures in emission order. NALs are fed one per packet; the
/// queue is drained at every IDR boundary (a coded-video-sequence flush)
/// and at end of stream, so display order is respected inside each CVS.
fn decode_stream(bytes: &[u8]) -> Result<Vec<OutPic>, String> {
    let nals = iter_length_prefixed(bytes).map_err(|e| format!("framing: {e}"))?;
    let mut dec = EvcDecoder::new();
    let mut out = Vec::new();
    let tb = TimeBase::new(1, 90_000);
    let mut any_vcl = false;
    for nal in &nals {
        if nal.header.nal_unit_type == NalUnitType::Idr && any_vcl {
            dec.flush().map_err(|e| format!("flush: {e}"))?;
            drain(&mut dec, &mut out);
        }
        if nal.header.nal_unit_type.is_vcl() {
            any_vcl = true;
        }
        let mut data = Vec::with_capacity(4 + nal.raw.len());
        data.extend_from_slice(&(nal.raw.len() as u32).to_be_bytes());
        data.extend_from_slice(nal.raw);
        let pkt = Packet::new(0, tb, data);
        dec.send_packet(&pkt)
            .map_err(|e| format!("send_packet ({:?}): {e}", nal.header.nal_unit_type))?;
    }
    dec.flush().map_err(|e| format!("flush: {e}"))?;
    drain(&mut dec, &mut out);
    Ok(out)
}

fn drain(dec: &mut EvcDecoder, out: &mut Vec<OutPic>) {
    while let Ok(oxideav_core::Frame::Video(v)) = dec.receive_frame() {
        out.push(OutPic {
            strides: v.planes.iter().map(|p| p.stride).collect(),
            planes: v.planes.into_iter().map(|p| p.data).collect(),
        });
    }
}

/// The two hash encodings accepted per plane (native, and 8-bit
/// up-shifted to 10-bit 2-byte LE).
fn plane_hashes(data: &[u8], stride: usize, width: usize) -> Vec<String> {
    let mut hashes = Vec::new();
    // Native bytes: 8-bit when stride == width in samples, 16-bit LE
    // when stride is 2x the sample count per row.
    hashes.push(md5::hex(data));
    if stride == width {
        // 8-bit native; also offer the `--output_bit_depth 10` shape.
        let mut wide = Vec::with_capacity(data.len() * 2);
        for &b in data {
            wide.extend_from_slice(&((b as u16) << 2).to_le_bytes());
        }
        hashes.push(md5::hex(&wide));
    }
    hashes
}

struct StreamResult {
    name: String,
    pics: usize,
    detail: Option<String>,
}

fn check_stream(dir: &Path, name: &str) -> StreamResult {
    let bit = dir.join(format!("{name}.bit"));
    let opl = dir.join(format!("{name}.opl"));
    let rows = match parse_opl(&opl) {
        Ok(r) => r,
        Err(e) => {
            return StreamResult {
                name: name.into(),
                pics: 0,
                detail: Some(format!("opl parse: {e}")),
            }
        }
    };
    let bytes = match std::fs::read(&bit) {
        Ok(b) => b,
        Err(e) => {
            return StreamResult {
                name: name.into(),
                pics: 0,
                detail: Some(format!("read .bit: {e}")),
            }
        }
    };
    let pics = match decode_stream(&bytes) {
        Ok(p) => p,
        Err(e) => {
            return StreamResult {
                name: name.into(),
                pics: 0,
                detail: Some(format!("decode: {e}")),
            }
        }
    };
    if pics.len() != rows.len() {
        return StreamResult {
            name: name.into(),
            pics: pics.len(),
            detail: Some(format!(
                "picture count: decoded {} vs opl {}",
                pics.len(),
                rows.len()
            )),
        };
    }
    let mut divergences = String::new();
    let mut bad = 0usize;
    for (i, (pic, row)) in pics.iter().zip(rows.iter()).enumerate() {
        let dims = [
            (row.width, row.height),
            (row.width / 2, row.height / 2),
            (row.width / 2, row.height / 2),
        ];
        for (c, name_c) in ["Y", "Cb", "Cr"].iter().enumerate() {
            let (w, h) = dims[c];
            let plane = &pic.planes[c];
            let stride = pic.strides[c];
            let sample_bytes = if stride == w { 1 } else { 2 };
            if stride != w * sample_bytes || plane.len() != w * h * sample_bytes {
                bad += 1;
                if bad <= 4 {
                    let _ = writeln!(
                        divergences,
                        "  pic {i} (poc {}) {name_c}: geometry {}x{} stride {} len {} vs opl {w}x{h}",
                        row.poc, w, h, stride, plane.len()
                    );
                }
                continue;
            }
            let cands = plane_hashes(plane, stride, w);
            if !cands.iter().any(|hh| *hh == row.md5[c]) {
                bad += 1;
                if bad <= 4 {
                    let _ = writeln!(
                        divergences,
                        "  pic {i} (poc {}) {name_c}: got {} want {}",
                        row.poc, cands[0], row.md5[c]
                    );
                }
            }
        }
    }
    if bad > 0 {
        let _ = writeln!(divergences, "  ({bad} diverging planes total)");
        return StreamResult {
            name: name.into(),
            pics: pics.len(),
            detail: Some(divergences),
        };
    }
    StreamResult {
        name: name.into(),
        pics: pics.len(),
        detail: None,
    }
}

fn conformance_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("EVC_CONFORMANCE_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/evc/conformance");
    p.is_dir().then_some(p)
}

fn run_set(subdir: &str) {
    let Some(root) = conformance_root() else {
        eprintln!("conformance: staging dir absent; skipping (docs-gated test)");
        return;
    };
    let dir = root.join(subdir);
    if !dir.is_dir() {
        eprintln!("conformance: {dir:?} absent; skipping");
        return;
    }
    let filter: Vec<String> = std::env::var("EVC_CONF_FILTER")
        .map(|f| f.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "bit").then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    if !filter.is_empty() {
        names.retain(|n| filter.iter().any(|f| n.contains(f.as_str())));
    }
    let mut failed = Vec::new();
    for name in &names {
        let r = check_stream(&dir, name);
        match &r.detail {
            None => eprintln!("conformance {subdir}/{name}: PASS ({} pics)", r.pics),
            Some(d) => {
                eprintln!("conformance {subdir}/{name}: FAIL\n{d}");
                failed.push(r.name);
            }
        }
    }
    eprintln!(
        "conformance {subdir}: {}/{} streams pass",
        names.len() - failed.len(),
        names.len()
    );
    assert!(
        failed.is_empty(),
        "conformance {subdir}: {} of {} streams diverge: {failed:?}",
        failed.len(),
        names.len()
    );
}

/// ISO/IEC 23094-4 1st-edition set (ETM 7.3-generated).
#[test]
fn conformance_ed1() {
    run_set("streams");
}

/// ISO/IEC 23094-4 Amd 1 set (ETM 8.0-generated).
#[test]
fn conformance_amd1() {
    run_set("streams-amd1");
}
