# Changelog

## [0.0.1](https://github.com/OxideAV/oxideav-evc/compare/v0.0.0...v0.0.1) - 2026-05-03

### Other

- rewrite mat_64_dc_row_is_64s loop to use iter().take()

## Round 7 — Main-profile CABAC initialization tables

- New [`cabac_init`](src/cabac_init.rs) module transcribes every
  `initValue` from ISO/IEC 23094-1:2020 §9.3.5 Tables 40-90 as
  `pub const &[u16]` arrays and exposes them through the
  [`MainCtxTable`] enum (discriminant = spec table number). Covers
  every context-coded Main-profile syntax element: `alf_ctb_flag`,
  `split_cu_flag`, `btt_split_flag` / `_dir` / `_type`,
  `split_unit_coding_order_flag` (SUCO), `pred_mode_constraint_type_flag`,
  `cu_skip_flag`, `mvp_idx_l0/l1`, `merge_idx`, `mmvd_flag` /
  `_group_idx` / `_merge_idx` / `_distance_idx` / `_direction_idx`,
  `affine_flag` / `_merge_idx` / `_mode_flag` / `_mvp_flag` /
  `_mvd_flag_l0` / `_mvd_flag_l1`, `pred_mode_flag`, `intra_pred_mode`,
  `intra_luma_pred_mpm_flag` / `_idx`, `intra_chroma_pred_mode`,
  `ibc_flag`, `amvr_idx`, `direct_mode_flag`, `inter_pred_idc`,
  `merge_mode_flag`, `bi_pred_idx`, `ref_idx_l0/l1`, `abs_mvd_l0/l1`,
  `cbf_all` / `_luma` / `_cb` / `_cr`, `cu_qp_delta_abs`, `ats_hor/ver_mode`,
  `ats_cu_inter_flag` / `_quad_flag` / `_horizontal_flag` / `_pos_flag`,
  `coeff_zero_run`, `coeff_abs_level_minus1`, `coeff_last_flag`,
  `last_sig_coeff_x/y_prefix`, `sig_coeff_flag`, and
  `coeff_abs_level_greaterA/B_flag`.
- `init_main_profile_contexts(engine, init_type, slice_qp)` walks every
  Main-profile table, derives `(valState, valMps)` per §9.3.2.2
  eq. 1425/1426, and installs the result at `(ctxTable, ctxIdx)`
  in the engine. Table 39's initType=0 (I) vs initType=1 (P/B)
  ctxIdx split is captured in `MainCtxTable::init_type_range`.
- ctxInc derivation helpers (§9.3.4.2.2 through 9.3.4.2.12) for
  every per-syntax-element `ctxInc` rule that the Main-profile decode
  pipeline will need:
  `ctx_inc_coeff_zero_run` (eq. 1434/1435),
  `ctx_inc_suco_flag` (eq. 1436/1437),
  `ctx_inc_neighbour_sum` (eq. 1438 + Table 96),
  `ctx_inc_btt_split_flag` (eq. 1440 + Table 97 `ctxSetIdx`),
  `ctx_inc_last_sig_coeff_prefix` (eq. 1441),
  `ctx_inc_sig_coeff_flag` (eq. 1447/1451),
  `ctx_inc_coeff_abs_level_greater_a/b` (eq. 1457/1458 + 1464/1465),
  `rice_param_coeff_abs_level_remaining` (Table 98),
  `ctx_inc_ats_cu_inter_flag` (eq. 1472), and
  `ctx_inc_ats_cu_inter_horizontal_flag` (eq. 1473).
- `CabacEngine::set_context()` exposes context installation for the
  Main-profile init path; `MAX_CTX_TABLES` bumped from 64 to 91 so
  the engine's per-table vector can index by the spec's Table-N
  number directly.
- The Baseline `(ctx_table=0, ctx_idx=0)` slot remains untouched, so
  no existing call site changes. Decode of the Main-profile syntax
  elements themselves still bubbles up `Error::Unsupported`; this
  round just lands the tables and helpers so subsequent rounds can
  wire individual tools (BTT split, ALF, DRA, AMVR, …) without
  initValue churn.
- Bumps the test count from 132 → 167. The 35 new tests cover spot
  checks against the printed Tables 42, 43, 44, 45, 47, 53, 55, 67,
  84, 85, 87, 88, 89, 90, the Table 39 `init_type_range` for
  representative tables, every ctxInc helper (with Table 97 lookups,
  the ats_cu_inter_horizontal aspect-ratio cases, the Table 98
  Rice-parameter, and the corner-case sigCtx clip at xc+yc<2), and
  the end-to-end `init_main_profile_contexts` derivation at slice
  QP 22 (P slice) and QP 32 (I slice) — verifying the output
  `(valState, valMps)` against hand-computed eq. 1425/1426 results
  — plus an idempotency test that calling init twice yields the
  same state for every Main-profile table entry.

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
