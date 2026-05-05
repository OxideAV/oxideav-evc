# Changelog

## [Unreleased]

### Round 11 — ALF adaptive loop filter + DRA dynamic range adjustment

#### Added
- `alf` module (`src/alf.rs`): `parse_alf_data` parses §7.3.5
  `alf_data()` APS-type-0 payload. Luma: up to 25 filter sets, each
  with 12 abs-coded 6-bit symmetric tap coefficients and DC offset
  (eq. 1264: `c[12] = 128 − 2·Σ|c[0..11]|`). Chroma: up to 4
  alternates, each with 6 abs-coded taps + DC. `apply_alf_luma` /
  `apply_alf_chroma` clone the source plane, run a boundary-clamped
  convolution with the 7×7 luma diamond (12 tap pairs) or 5×5 chroma
  diamond (6 tap pairs), and write results in-place. `apply_alf` is
  the one-call entry point (luma filter[0] then chroma alt[0]).
- `dra` module (`src/dra.rs`): `parse_dra_data` parses §7.3.6
  `dra_data()` APS-type-1 payload. `build_luma_lut` produces a
  256-entry piecewise-linear LUT from up to 16 segments; first scale
  is 11-bit unsigned Q8.3; subsequent scales are 12-bit signed deltas.
  `apply_dra` maps every Y sample through the LUT and offsets Cb/Cr by
  the segment-0 `chroma_qp_offset`. `find_segment` is a linear scan
  from the tail for boundary matching.
- `EvcDecoder::alf_aps` / `EvcDecoder::dra_aps`: decoder caches the
  most-recent parsed `AlfData` and `DraData` from `NalUnitType::Aps`
  NAL units.
- `EvcDecoder::apply_post_filters`: runs ALF then DRA on every decoded
  `YuvPicture` (IDR and non-IDR) when the SPS gates are set and APS
  data is available.

#### Changed
- `sps_alf_flag = 1` no longer returns `Error::Unsupported` in
  `walk_idr_slice`, `decode_idr_slice`, or `decode_non_idr`; instead
  the ALF post-filter pass is applied when APS data is present.
- `sps_dra_flag = 1` no longer returns `Error::Unsupported` in
  `walk_idr_slice`, `decode_idr_slice`, or `decode_non_idr`; instead
  the DRA post-filter pass is applied when APS data is present.
- 226 unit tests pass (was 205); 21 new tests cover ALF and DRA parse
  + application paths.

### Round 10 — spatial-neighbour MV grid AMVP + LTRP RPL resolution + flush() drain

#### Added
- `baseline_amvp_select_with_grid_and_hmvp` (`slice_data`): the
  Baseline §8.5.2.4 AMVP `mvpList[]` is now built from the per-4×4
  `SideInfoGrid` at the spec's three spatial probe positions:
  - `mvpList[0]` ← MV at `(xCb − 1, yCb + nCbH − 1)` (left column,
    bottom-most cell of the CU).
  - `mvpList[1]` ← MV at `(xCb + nCbW − 1, yCb − 1)` (above row,
    right-most cell of the CU).
  - `mvpList[2]` ← MV at `(xCb + nCbW, yCb − 1)` (above-right corner).
  - `mvpList[3]` ← temporal slot (still zero MV — §8.5.2.5 collocated
    is parked for a follow-up round).
  Each spatial probe is gated on `(pred_mode == Inter && ref_idx_l* ==
  cur_ref_idx_lx)` per §8.5.2.4.3 — an in-picture neighbour with a
  different reference is unavailable. When any spatial slot would
  land on the spec's `(1, 1)` substitution AND the round-8
  `HmvpCandList` holds a valid candidate, `derive_default_mv` is
  consulted (§8.5.2.4.4 fallback unchanged from round 9).
- `spatial_neighbour_mv` helper: probes the side-info grid at luma
  coordinates `(x, y)` for an inter neighbour with a matching
  ref-idx on `list_x`. Returns `Some(mv)` only when the cell is in
  picture, `pred_mode == Inter`, and `ref_idx_l* == cur_ref_idx_lx`.
- `EvcDecoder::build_ref_pocs` (was a free function `build_ref_pocs`):
  promoted to a method so it can resolve LTRP entries against the
  DPB. `RefPicListEntry::Ltrp { poc_lsb_lt }` is matched against
  `(poc & (max_poc_lsb − 1))` for every DPB slot; the matching POC
  becomes the LTRP slot's resolved reference. STRP entries continue
  to advance the running delta-POC chain unchanged. The round-9
  `Error::Unsupported` gate on LTRP is gone.
- `EvcDecoder::drain_dpb_to_output`: pushes every DPB entry whose
  `output_emitted == false` to the `out` queue in ascending POC order.
- `DpbEntry::output_emitted` field: tracks whether the entry has been
  pushed to the output queue. `enqueue_for_output` flips it to `true`
  after the picture is queued; `drain_dpb_to_output` skips entries
  that already have `output_emitted == true` so flush() is idempotent.

#### Changed
- `Decoder::flush()`: was a no-op; now drains unemitted DPB entries
  to the output queue in POC order. Pictures already in `out`
  (low-delay GOPs) stay in place — flush is idempotent and a no-op
  when every DPB entry is already emitted.
- `decode_inter_coding_unit` cu_skip + explicit-MV paths now call
  `baseline_amvp_select_with_grid_and_hmvp` (passing `side_info`,
  `x0`, `y0`, `n_cb_w`, `n_cb_h`, `ref_idx`, `list_x`) instead of
  the round-9 `baseline_amvp_select_with_hmvp` (which always faked
  the spatial slots as unavailable). The round-9 helper is kept
  `#[cfg(test)]` for direct unit tests of the (1, 1) → HMVP
  fallback path in isolation.

#### Tests
- 205 unit tests pass (up from 196). 9 new tests cover round-10:
  - `round10_spatial_neighbour_left_drives_amvp_slot_0` — grid lookup
    for slot 0 (left position).
  - `round10_spatial_neighbour_above_right_drives_slot_2` — grid
    lookup for slot 2 (above-right).
  - `round10_spatial_neighbour_ref_idx_mismatch_is_unavailable` —
    strict ref-idx match per §8.5.2.4.3.
  - `round10_spatial_amvp_falls_through_to_hmvp` — empty grid +
    non-empty HMVP still hits the §8.5.2.4.4 fallback.
  - `round10_ltrp_rpl_resolves_against_dpb` — LTRP `poc_lsb_lt` →
    DPB POC.
  - `round10_ltrp_missing_dpb_entry_is_invalid` — LTRP with no
    matching DPB POC is rejected as invalid bitstream.
  - `round10_mixed_strp_and_ltrp_resolve` — STRP delta chain plus
    LTRP slot in one RPL walk.
  - `round10_flush_drains_unemitted_dpb_entries_in_poc_order` —
    direct `drain_dpb_to_output` test.
  - `round10_flush_after_receive_is_idempotent` — end-to-end
    IDR + P decode through `Decoder::flush()`; no duplicate frames.

#### Deferred to round 11
- ALF (§8.9 adaptive loop filter) — APS-driven coefficient sets are
  parsed in round-7's `cabac_init`, but the filter tap-pass is not
  yet implemented (still surfaces `sps_alf_flag = 1` SPS as
  `Error::Unsupported`).
- DRA (§8.10 dynamic range adjustment) — APS-driven mapping tables
  are parsed but the range-mapping post-pass is not yet implemented.
- §8.5.2.5 temporal AMVP (collocated-picture) — slot `mvpList[3]` is
  still zero. Needs the temporal MV buffer to be threaded through
  the DPB.
- §8.3.2 sliding-window unmark of LTRPs — the `output_emitted` field
  exists but no automatic LTRP eviction happens; the DPB only flushes
  on IDR.

## [0.0.1](https://github.com/OxideAV/oxideav-evc/compare/v0.0.0...v0.0.1) - 2026-05-05

### Other

- multi-reference DPB + HMVP-as-AMVP fallback + POC reordering
- fix doc_lazy_continuation in round8 fixture comment
- RPL non-IDR parse + HMVP candidate list
- rewrite mat_64_dc_row_is_64s loop to use iter().take()

### Round 9 — multi-reference DPB + HMVP-as-AMVP fallback + POC reordering

#### Added
- DPB inside `EvcDecoder`: every decoded picture is held in a
  16-slot `Vec<DpbEntry>` indexed by POC. IDR slices flush the buffer;
  non-IDR slices append at the resolved POC. Eviction at capacity
  drops the lowest POC. New `dpb_find()` / `dpb_insert()` /
  `dpb_flush()` helpers.
- POC derivation per §8.3.1 (`derive_poc`): wraps `PicOrderCntMsb`
  forward when the new `slice_pic_order_cnt_lsb` falls more than
  half-`MaxPicOrderCntLsb` below the previous LSB, and backward in the
  symmetric direction. Round-9 routes every non-IDR slice through it.
- POC-ordered output queue (`out_pocs` parallel buffer): the decoder
  inserts each freshly-decoded picture into the `out` queue at its
  sorted POC position so callers see frames in display order even when
  bitstream coding order differs.
- `InterDecodeInputs::ref_list_l0` / `ref_list_l1` slice surfaces
  (was single-entry `ref_l0` / `ref_l1`), plus `ref_l0(idx)` /
  `ref_l1(idx)` resolver helpers. Slice entry validates that each
  list holds at least `num_ref_idx_active_minus1[i] + 1` entries.
- `baseline_amvp_select_with_hmvp()`: when the §8.5.2.4.3 spatial
  AMVP slot would land on the spec's `(1, 1)` substitution AND the
  HMVP candidate list is non-empty, the predictor falls through to
  `hmvp.derive_default_mv(ref_idx, list_x)`. Wires the round-8
  HMVP infrastructure into actual MV selection per §8.5.2.4.4.
- `decoder::decode_non_idr` end-to-end: parses the slice header,
  derives POC, walks the slice's `ref_pic_list_struct()` to build
  per-list POC arrays (delta-POC chained from the current POC),
  resolves each POC against the DPB, and threads the resulting
  `RefPictureView` slices through the inter pipeline.
- `implicit_ref_pocs()` fallback for streams with `sps_rpl_flag = 0`:
  uses the highest-POC DPB entry as the single reference, preserving
  round-4 behaviour without requiring RPL signalling.

#### Changed
- `InterDecodeInputs` ref fields are now `&'b [RefPictureView<'a>]`
  (was `RefPictureView` / `Option<RefPictureView>`). Tests passing a
  single reference wrap it in a single-element array. The
  round-4-era `num_ref_idx_active_minus1_l? > 0` `Error::Unsupported`
  gate is gone — multi-reference bitstreams now decode end-to-end.
- `apply_inter_prediction` resolves L0 / L1 ref via `ref_l0(ref_idx)`
  on the inputs struct instead of always reading `ref_l0[0]`. Each
  CU's per-list `ref_idx_l*` is honoured.
- `decode_inter_coding_unit` cu_skip + explicit-MV paths use
  `baseline_amvp_select_with_hmvp` instead of the round-4 stubbed
  `baseline_amvp_select`. The HMVP fallback is queried with the
  resolved `ref_idx_l*` so the §8.5.2.4.4 ref-idx-match rule fires
  on real CUs.
- Decoder NonIDR routing replaced: the old `decode_non_idr_via_inter`
  free function is gone, replaced by the `EvcDecoder::decode_non_idr`
  method (needs `&mut self` for DPB updates + POC tracker).
- Round-8 fixture `round8_rpl_non_idr_decodes_to_two_frames` now
  signs its inline `delta_poc_st` as negative (sign=0) so the ref
  POC = 1 + (-1) = 0 resolves to the IDR; the previous fixture's
  positive sign pointed at a future POC and only worked because
  round 8 ignored the delta entirely.

#### Tests
- 196 unit tests pass (up from 187). 9 new tests cover the round-9
  pipeline:
  - `round9_hmvp_fallback_overrides_unavailable_neighbour` — direct
    helper test verifying the AMVP `(1, 1)` slot is replaced with
    the HMVP entry when ref-idx matches.
  - `round9_hmvp_fallback_noop_on_empty_list` — verifies the
    fallback is silent when HMVP is fresh.
  - `round9_multiref_dpb_two_entry_l0` — pipeline acceptance for a
    P slice with `num_ref_idx_active_minus1_l0 = 1` (two L0 refs).
  - `round9_rejects_empty_ref_list_l0` and
    `round9_rejects_oversized_active_count` — DPB validation tests.
  - `derive_poc_wraps_on_lsb_rollover` — §8.3.1 POC derivation.
  - `dpb_evicts_lowest_poc_at_capacity` — DPB eviction.
  - `dpb_flush_clears_all` — DPB reset on IDR.
  - `round9_three_frame_idr_p_p_with_dpb` — end-to-end fixture
    decoding IDR + P (POC 1) + P (POC 2) where the second P
    references the first via inline RPL `delta_poc_st = 1, sign = 0`.

### Round 8 — RPL non-IDR + HMVP infrastructure

#### Added
- New [`rpl`](src/rpl.rs) module: `parse_ref_pic_list_struct()` walks
  every entry of an EVC `ref_pic_list_struct(listIdx, rplsIdx,
  ltrpFlag)` (§7.3.7 / §7.4.8). STRP entries surface a signed
  `delta_poc_st` via `RefPicListEntry::signed_delta_poc()`
  (§7.4.8 eq. 124); LTRP entries surface the fixed-length
  `poc_lsb_lt`. Caps the per-list entry count at
  `MAX_REF_PIC_LIST_ENTRIES = 64` to bound allocations.
- New [`hmvp`](src/hmvp.rs) module: 23-entry `HmvpCandList` with
  the §8.5.2.7 LRU update process (left-shift-on-full, no-op on
  invalid candidates) and `derive_default_mv()` per §8.5.2.4.4
  (last-4-tail walk, ref-idx-match preferred, fallback to most-recent
  valid candidate). `reset()` clears the list at CTU-row left
  boundaries per §7.3.8.2.
- `Sps` exposes `num_ref_pic_lists_in_sps_l0/l1` plus per-list
  `Vec<RefPicListStruct>`. The previous skip-only RPL handling is
  replaced with the real parser; `rpl1_same_as_rpl0_flag = 1`
  inherits list 1 from list 0 per §7.4.3.1.
- `SliceHeader` exposes `ref_pic_list_sps_flag[2]`,
  `ref_pic_list_idx[2]`, `slice_rpl[2]: Option<RefPicListStruct>`
  and `slice_rpls_idx[2]` (§7.4.5 eq. 83).
- `slice_header::parse_consume()` parses off an existing BitReader so
  callers can recover the slice-data byte offset.
- `InterDecodeStats::hmvp_cand_count_final` exposes
  `NumHmvpCand` at slice end for fixture verification.

#### Changed
- `slice_header::parse` now consumes the non-IDR `sps_rpl_flag` branch
  end-to-end: `ref_pic_list_sps_flag[i]` (with the §7.4.5 inference
  rules for `i == 1` when `rpl1_idx_present_flag = 0`),
  `ref_pic_list_idx[i]` (sized by `Ceil(Log2(n_in_sps))`), inline
  `ref_pic_list_struct()` for slices that supply their own RPL, and
  the per-LTRP `additional_poc_lsb_present_flag` /
  `additional_poc_lsb_val` loop. The prior `Error::Unsupported` gate
  is gone.
- `decoder::decode_non_idr_via_inter` re-routes through the canonical
  `slice_header::parse_consume`, so any production stream with
  `sps_rpl_flag = 1` decodes through a single tested code path.
- `decode_baseline_inter_slice` threads an `HmvpCandList` through
  every inter CU, resets it on CTU-row boundaries, and updates it
  after each inter CU per §8.5.2.7.
- `SliceParseContext` adds `num_ref_pic_lists_in_sps_l0/l1`,
  `rpl1_idx_present_flag`, `long_term_ref_pics_flag`,
  `additional_lt_poc_lsb_len` (passed from SPS + PPS by the lib-level
  helper).

#### Tests
- 187 unit tests pass (up from 167). 20 new tests cover the RPL
  parser (STRP-only, mixed STRP+LTRP, zero-delta no-sign-bit,
  empty list, oversized-count rejection on both STRP and LTRP),
  HMVP (LRU shift-on-full, exact-refIdx walk in tail, fallback
  to most-recent valid, last-4 walk bound, L1-field selection,
  invalid candidate dropped, reset semantics), slice-header RPL
  paths (inline RPL on both lists, SPS-pointer with
  `rpl1_idx_present_flag = 0` inferring list 1, `ceil_log2`
  helper), and the round-8 end-to-end fixture
  `round8_rpl_non_idr_decodes_to_two_frames` driving the
  registered decoder through an IDR + P with `sps_rpl_flag = 1`
  and inline RPL.

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
