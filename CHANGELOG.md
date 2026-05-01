# Changelog

## Round 5 — residual coding, deblocking, and 64-point IDCT

- Wire `residual_coding_rle()` per §7.3.8.7 through the IDR + inter
  decode pipelines: levels → §8.7.2/§8.7.3 dequant → §8.7.4 inverse
  transform → add to predictor → clip and store. Both intra (DC) and
  inter (skip + explicit) CUs handle non-zero CBF.
- Add the 64-point inverse DCT-II matrix. The spec's eq. 1071-1076 has
  m/n labelling typos that blocked transcription in round 3; we build
  the matrix from the closed-form
  `M[m][n] = round(64·√2·cos(π·m·(2n+1)/128))` (m≥1, M[0][n] = 64),
  verified against every printed entry of eq. 1072 / 1074.
- Implement the §8.8.2 luma deblocking filter (`sps_addb_flag = 0`):
  Table 33 strength lookup, BS derivation per §8.8.2.3 step 2, and the
  4-tap edge filter per eq. 1148-1158. Per-4×4-grid side info
  (PredMode + CBF + MV + RefIdx) is stamped during CU decode.
- Bumps the test count from 109 → 126; the new fixtures cover the
  zig-zag scan order, `decode_residual_coding_rle` round-trip, the
  64×64 IDCT (zero + DC-only), the IDR + residual pipeline, the
  inter + residual pipeline, the IDR + 64×64 CTU acceptance test, and
  the deblocking filter (table values, BS rules, no-op end-to-end).
- README + crate-level docs refreshed to reflect round-5 status.

## Round 4 — Baseline P + B inter prediction

- Inter slice decode for Baseline P and B (single reference per list,
  `cbf_luma = cbf_cb = cbf_cr = 0`), including AMVP candidate
  construction and default-weighted bipred.

## Round 3 — IDR pixel pipeline

- End-to-end IDR slice decode with intra prediction (DC / HOR / VER /
  UL / UR), `sps_iqt_flag = 0` dequant, and inverse DCT for nTbS ∈
  {2, 4, 8, 16, 32}.

## Round 2 — slice walker + CABAC

- `slice_data()` walker for Baseline IDR slices: parses every `ae(v)`
  syntax element, drives the CABAC engine through end_of_tile_one_bit.

## Round 1 — parameter sets + probe

- SPS / PPS / APS / slice header parse; `probe()` recovers picture
  dimensions and bit depth from a length-prefixed bitstream.
