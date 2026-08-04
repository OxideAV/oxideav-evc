//! Minimal MD5 (RFC 1321) for checking decoded planes against the
//! conformance corpus' published hashes. Hashing only — not part of the
//! codec surface. The per-step sine constants are computed at run time
//! from their defining formula `K[i] = floor(|sin(i + 1)| * 2^32)`, so
//! no transcribed table is carried.

/// Per-round left-rotate amounts.
const S: [[u32; 4]; 4] = [
    [7, 12, 17, 22],
    [5, 9, 14, 20],
    [4, 11, 16, 23],
    [6, 10, 15, 21],
];

fn k(i: usize) -> u32 {
    (((i as f64 + 1.0).sin().abs()) * 4294967296.0) as u64 as u32
}

pub fn digest(data: &[u8]) -> [u8; 16] {
    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];

    // Padded message: data || 0x80 || zeros || bit-length (64-bit LE).
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut tail = vec![0x80u8];
    while (data.len() + tail.len()) % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&bit_len.to_le_bytes());

    let mut process = |block: &[u8]| {
        let mut m = [0u32; 16];
        for (j, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                block[4 * j],
                block[4 * j + 1],
                block[4 * j + 2],
                block[4 * j + 3],
            ]);
        }
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let round = i / 16;
            let (f, g) = match round {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(k(i))
                    .wrapping_add(m[g])
                    .rotate_left(S[round][i % 4]),
            );
            a = tmp;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    };

    let mut chunks = data.chunks_exact(64);
    for block in &mut chunks {
        process(block);
    }
    let mut last = chunks.remainder().to_vec();
    last.extend_from_slice(&tail);
    for block in last.chunks_exact(64) {
        process(block);
    }

    let mut out = [0u8; 16];
    for (i, s) in state.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&s.to_le_bytes());
    }
    out
}

pub fn hex(data: &[u8]) -> String {
    digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn rfc1321_vectors() {
    assert_eq!(hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
    assert_eq!(hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(hex(b"message digest"), "f96b697d7cb7938d525a2f31aaf161d0");
    assert_eq!(
        hex(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
        "d174ab98d277d9f5a5611c2c9f419d9f"
    );
    // A >64-byte message exercises the multi-block path.
    assert_eq!(
        hex(b"12345678901234567890123456789012345678901234567890123456789012345678901234567890"),
        "57edf4a22be3c955ac49da2e2107b67a"
    );
}
