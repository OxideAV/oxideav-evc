# Changelog

## Round 6 — chroma deblocking

- Extend §8.8.2 deblocking to the chroma planes per eq. 1167-1213. The
  chroma path uses the 2-tap stencil (only `sB` and `sC` are modified —
  eq. 1208/1209/1212/1213) instead of luma's 4-tap stencil, and looks
  up Table 33 with `qp_c = Clip3(0, 51, slice_qp + slice_cb_qp_offset)`
  for Cb (with `slice_cr_qp_offset` for Cr — eq. 1194).
- The BS derivation reuses the luma 4×4 side-info grid (cbf_luma is
  the spec's chroma-edge trigger per §8.8.2.3 step 2). Edge spacing
  scales by `(SubWidthC, SubHeightC)` from Table 2: chroma edges land
  every 2 chroma samples for 4:2:0 (= every 4 luma samples = the luma
  cell boundary). 4:2:2 and 4:4:4 use the same code path with
  Table 2 sub-sampling factors.
- `SliceDecodeInputs` gains `slice_cb_qp_offset` and `slice_cr_qp_offset`
  so the IDR + inter pipelines forward the per-slice chroma offsets
  (range −12..=12). The Baseline PPS does not encode
  `pps_cb_qp_offset` / `pps_cr_qp_offset` (always 0), so slice-level
  offsets are sufficient.
- Bumps the test count from 126 → 132. New fixtures cover the chroma
  2-tap reference values, Table 2 sub-sampling factors, the chroma
  no-op pass on uniform grey, the small-step inter+cbf edge smoothing
  (`sB`/`sC` mutate, `sA`/`sD` stay), and the `chroma_qp_offset`
  switching the filter on / off at `slice_qp = 17`.
- The pre-existing `idr_decode_with_deblock_enabled_no_op` fixture now
  asserts 960 edges (480 luma + 240 Cb + 240 Cr) and verifies all
  three planes stay grey.

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
