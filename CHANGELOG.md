# Changelog

## [Unreleased]

### Round 113 — per-CTB ALF apply-masking (§8.9 / §7.3.8.2)

#### Added
- `alf.rs`: `AlfCtbMap` — the resolved per-CTU `alf_ctb_*` applicability
  decoded by `coding_tree_unit()` (§7.3.8.2), one triplet per CTU in raster
  order plus `CtbLog2SizeY` / `PicWidthInCtbsY` / `PicHeightInCtbsY`. Carries
  the present-or-inferred on/off state used by the §8.9 apply.
- `alf.rs`: `apply_alf_luma_masked` implements the §8.9 luma loop (lines
  18059-18074): for each CTU at `(rx, ry)`, the coding tree block luma type
  filtering is invoked **only when `alf_ctb_flag[rx][ry]` is 1**. The
  §8.9 `blkWidth` / `blkHeight` picture-edge clamp falls out of intersecting
  the per-CTB write range with the picture extent; CTBs whose flag is 0 keep
  their reconstructed (pre-ALF) samples, matching the spec's "init
  alfPicture to recPicture, overwrite filtered CTBs" semantics. The filter
  reads from a whole-plane pre-filter snapshot so a filtered CTB's edge
  samples still tap unfiltered neighbours.
- `alf.rs`: `apply_alf_with_map` — map-driven entry point. Masks the luma
  apply per CTB; for `ChromaArrayType` 1..2 (Baseline 4:2:0/4:2:2) the chroma
  plane is filtered as a whole only when the slice-level
  `sliceChromaAlfEnabledFlag` / `sliceChroma2AlfEnabledFlag` holds (the
  §7.3.8.2 per-CTB chroma map flags are inferred 0 in that case).
- `slice_data.rs`: the IDR + inter CTU loops now record each CTU's resolved
  `AlfCtbFlags` into a new `alf_ctb_map: AlfCtbMap` on `SliceDecodeStats` /
  `InterDecodeStats` (both lose their `Copy` derive — no call site relied on
  it). The round-107 `let _alf = ...` discards become `map.set(...)`.
- `decoder.rs`: `decode_non_idr` returns a new `NonIdrDecodeResult` carrying
  the picture, POC, the decoded `alf_ctb_map`, and the slice-level chroma ALF
  enables. `apply_post_filters` consults the map: when at least one luma CTB
  is on (or chroma is enabled) it uses `apply_alf_with_map`; an all-off map
  (the minimal-header IDR path that doesn't thread the slice ALF enables)
  falls back to the whole-plane `apply_alf`, preserving round-11 behaviour.

#### Tests
- 300 pass (was 292). 8 new tests:
  - `alf::masked_luma_apply_only_touches_flagged_ctbs`: two-CTB picture,
    left on / right off ⇒ only the left CTB is filtered.
  - `alf::masked_luma_apply_clamps_at_picture_edge`: a partial bottom-right
    CTB filters only its in-picture 8×8 region (§8.9 blkWidth/blkHeight).
  - `alf::masked_apply_all_off_map_leaves_picture_unchanged`.
  - `alf::apply_alf_with_map_matches_whole_plane_when_all_on`: an all-on map
    reproduces the whole-plane `apply_alf` bit-for-bit.
  - `alf::apply_alf_with_map_chroma_gated_by_slice_enable`: Cb enabled /
    Cr disabled ⇒ only Cb filtered.
  - `alf::alf_ctb_map_new_sizes_to_ctb_grid`: ceil-div CTB grid sizing.
  - `slice_data::round113_idr_decode_populates_alf_ctb_map`: the IDR decode
    threads the decoded per-CTU flag into `stats.alf_ctb_map`.
  - `slice_data::round113_idr_two_ctb_map_drives_masked_alf_apply`:
    end-to-end decode of a 64×32 IDR with the left CTB coded ALF-on and the
    right ALF-off, then the masked apply filters only the left CTB.

#### Notes
- This round wires the round-107 decoded ALF applicability map into the §8.9
  apply (the documented round-107 follow-up: previously `apply_alf` filtered
  whole planes regardless of the per-CTB flags). Still deferred: the
  per-CTU 25-filter-set selection (`alf_luma_filter_idx`, §8.9.6) — filter
  set 0 is always applied — and the §8.8.4.3 ALF transpose / filter-index
  classification (the apply uses the fixed diamond tap layout).
- Clean-room from ISO/IEC 23094-1:2020 (PDF + extracted text in
  `docs/video/evc/`). No xeve, xevd, ETM reference, or libavcodec evcdec
  consulted; no web access.

### Round 107 — `coding_tree_unit()` ALF applicability map (§7.3.8.2)

#### Added
- `slice_header.rs`: the §7.3.4 ALF slice-header block now surfaces the
  per-CTU map controls instead of parsing-and-discarding them. New
  `SliceHeader` fields: `slice_alf_map_flag`, `slice_alf_chroma_idc`,
  the §7.4.5-derived `slice_chroma_alf_enabled_flag` /
  `slice_chroma2_alf_enabled_flag`, and the `ChromaArrayType == 3`-only
  `slice_alf_chroma_map_flag` / `slice_alf_chroma2_map_flag`. The parser
  also consumes the previously-skipped `ChromaArrayType == 3` branch
  (re-signalled `slice_alf_chroma_idc` when ALF is luma-disabled, the
  per-component chroma APS ids, and the chroma map flags).
- `slice_data.rs`: `decode_coding_tree_unit_alf` implements the
  §7.3.8.2 lines 2626-2631 `coding_tree_unit()` ALF prefix. It decodes
  the 0-3 `alf_ctb_flag` / `alf_ctb_chroma_flag` / `alf_ctb_chroma2_flag`
  bins (each FL `cMax = 1`, ae(v), Table 40, ctxInc 0 under
  `sps_cm_init_flag == 0`) gated exactly as the spec syntax
  (`slice_alf_enabled_flag && slice_alf_map_flag` for luma, etc.), and
  applies the §7.4.9.2 not-present inference (each flag → the matching
  slice-level enable). Returns an `AlfCtbFlags` triplet. For Baseline
  4:2:0 (ChromaArrayType 1) the chroma map flags are inferred 0, so only
  the luma bin can appear — but the full triplet is decoded for
  spec-completeness on the ChromaArrayType-3 path.
- The helper is wired into all three CTU loops — `walk_baseline_idr_slice`,
  `decode_baseline_idr_slice`, and `decode_baseline_inter_slice` — at the
  top of each per-CTU iteration, before `split_unit()`. New
  `SliceWalkInputs` fields thread the slice-header map controls; the
  `decode_non_idr` path in `decoder.rs` populates them from the parsed
  `SliceHeader`. The two `lib.rs` minimal-header entry points default
  them off.
- New `AlfCtbStats` (`luma_bins` / `chroma_cb_bins` / `chroma_cr_bins` /
  `luma_on_ctus`) embedded into `SliceWalkStats`, `SliceDecodeStats`, and
  `InterDecodeStats` so fixtures can assert the §7.3.8.2 presence gating.

#### Tests
- 292 pass (was 284). 8 new tests:
  - `round107_ctu_alf_no_map_consumes_no_bins`: no map signalled ⇒ zero
    bins, luma inferred from (off) `slice_alf_enabled_flag`.
  - `round107_ctu_alf_map_without_enable_infers_off`: map flag on but
    enable off ⇒ no luma bin (presence is enable && map).
  - `round107_ctu_alf_luma_map_reads_one_bin`: two CTBs, coded 1 then 0,
    one bin each; `luma_on_ctus` tracks only the set one.
  - `round107_ctu_alf_chroma3_reads_three_bins`: ChromaArrayType-3 path
    decodes luma + Cb + Cr bins, each resolving independently.
  - `round107_idr_decode_reads_alf_ctb_flag_bin` /
    `..._without_alf_map_reads_no_alf_bins`: end-to-end 32×32 monochrome
    IDR decode with/without the luma map, asserting the consumed
    `alf_ctb_flag` bin count and unchanged grey reconstruction.
  - `round107_alf_map_fields_baseline_chroma1` /
    `round107_alf_map_chroma_idc_zero`: slice-header parse surfaces the
    new fields + derived enables with correct bit alignment.

#### Notes
- This round lands the ALF *applicability-map parse + per-CTU on/off
  decision*. Actually masking the §8.9 ALF apply per-CTB by the decoded
  map (today's `apply_alf` filters whole planes) is a follow-up: the
  per-CTB `AlfCtbFlags` are decoded and counted but not yet consulted by
  the post-filter pass.
- Clean-room from ISO/IEC 23094-1:2020 (PDF + extracted text in
  `docs/video/evc/`). No xeve, xevd, ETM reference, or libavcodec evcdec
  consulted; no web access.

### Round 103 — IBC-branch `cu_qp_delta` wiring (§7.3.8.5)

#### Added
- `decode_ibc_branch` (`src/slice_data.rs`, IDR path): the IBC
  `transform_unit()` now decodes the §7.3.8.5 `cu_qp_delta_abs` /
  `cu_qp_delta_sign_flag` syntax elements after the (inferred-1)
  `cbf_luma` and before the luma residual, instead of hard-coding
  `cu_qp = slice_qp`. The §7.3.8.5 line 3073 presence condition is
  mode-independent, so a MODE_IBC CU reads the element exactly as the
  intra single-tree (round-3) and regular-inter (round-100) paths.
  Under Baseline's `sps_dquant_flag == 0` the guard collapses to
  `cu_qp_delta_enabled_flag && (cbf_luma || cbf_cb || cbf_cr)`; the IBC
  DUAL_TREE_LUMA branch infers `cbf_luma = 1` with no chroma cbf, so
  the condition reduces to `cu_qp_delta_enabled_flag`.
  `cu_qp_delta_abs` is U-binarized with ctxInc 0 (Table 95) under
  Table 78 init; `cu_qp_delta_sign_flag` is bypass-coded for a
  non-zero magnitude. The derived QP (eq. 148, clamped to `[0, 51]`)
  is threaded into `apply_ibc_branch_predict_and_reconstruct` via a
  new `cu_qp` parameter that drives `scale_and_inverse_transform`.
- `decode_inter_ibc_branch` (`src/slice_data.rs`, non-IDR P/B path):
  symmetric wiring after the single-tree `cbf_luma` / `cbf_cb` /
  `cbf_cr` bins. The resolved per-CU QP is threaded into
  `apply_inter_ibc_branch_predict_and_reconstruct` via a new `cu_qp`
  parameter.
- `SliceDecodeStats::cu_qp_delta_abs_bins`: new counter mirroring the
  inter-side `InterDecodeStats` tracker. Incremented once per IDR-path
  CU (intra single-tree or IBC) that satisfies the §7.3.8.5 presence
  condition. The pre-existing intra `decode_transform_unit` read now
  increments it too (it previously decoded the element with no
  counter).

#### Tests
- 284 pass (was 280). 4 new tests:
  - `round103_idr_ibc_branch_cu_qp_delta_abs_zero_is_single_u_bin`:
    engine-level isolation of the exact prefix the IDR IBC branch reads
    (cbf_luma inferred 1 → no bin; `cu_qp_delta_abs = 0` decodes as a
    single all-regular U "0" terminator with no sign bit). Robust
    against the test-only encoder's `encode_bypass` defer bug.
  - `round103_ibc_cu_qp_delta_signed_magnitude_and_clamp`: eq. 148
    signed-magnitude derivation + `[0, 51]` clamp over the sign and
    saturation corners.
  - `round103_idr_ibc_apply_threads_cu_qp_into_residual_scaling`:
    direct call of `apply_ibc_branch_predict_and_reconstruct` with a
    fixed non-zero DC residual at QP 22 vs QP 40 — the reconstructions
    differ and the higher QP deviates further from the predictor,
    proving the threaded `cu_qp` drives the §8.7.3 scaling.
  - `round103_inter_ibc_apply_threads_cu_qp_into_residual_scaling`:
    same verification for the non-IDR
    `apply_inter_ibc_branch_predict_and_reconstruct` helper.

#### Notes
- A full-slice non-skip CABAC fixture driving `cu_qp_delta` end-to-end
  through either IBC branch is still blocked by the test-only
  `CabacEncoder::encode_bypass` defer bug on the residual
  `coeff_sign_flag` (round-90/95/100 notes). Coverage therefore splits
  between engine-level read isolation and direct-call helper checks,
  exactly as round 100 did. With this round, all four
  `transform_unit()` entry points (intra single-tree, regular inter,
  IDR IBC, non-IDR IBC) decode per-CU `cu_qp_delta`.
- Clean-room from ISO/IEC 23094-1:2020 (PDF + extracted text in
  `docs/video/evc/`). No xeve, xevd, ETM reference, or libavcodec
  evcdec consulted; no web access.

### Round 100 — non-IDR (P/B) inter-CU `cu_qp_delta` wiring (§7.3.8.5)

#### Added
- `decode_inter_coding_unit` (`src/slice_data.rs`): the non-skip
  MODE_INTER transform_unit() path now decodes the §7.3.8.5
  `cu_qp_delta_abs` / `cu_qp_delta_sign_flag` syntax elements instead
  of hard-coding `cu_qp = slice_qp`. The presence condition is
  mode-independent in the spec; under Baseline's `sps_dquant_flag == 0`
  the §7.3.8.5 line 3073 guard collapses to
  `cu_qp_delta_enabled_flag && (cbf_luma || cbf_cb || cbf_cr)`, matching
  the intra single-tree path (round-3 wiring). `cu_qp_delta_abs` is
  U-binarized with ctxInc 0 for every bin (Table 95) under Table 78
  init; `cu_qp_delta_sign_flag` is bypass-coded and only present when
  the magnitude is non-zero. The derived QP follows eq. 148
  (`QpY = slice_qp + cu_qp_delta_abs * (1 - 2 * sign)`), clamped to the
  legal 8-bit-depth range `[0, 51]`, and feeds the existing
  per-component `scale_and_inverse_transform` residual scaling.
- `InterDecodeStats::cu_qp_delta_abs_bins`: new counter mirroring the
  IDR-side `SliceDecodeStats` tracker — one increment per inter CU that
  decodes the syntax element.

#### Tests
- 280 pass (was 277). 3 new tests:
  - `round100_inter_skip_cu_consumes_no_cu_qp_delta_bins`: full-slice
    P-slice cu_skip CU with `cu_qp_delta_enabled = true` — verifies the
    §7.3.8.5 presence condition is false for an inferred-zero-CBF skip
    CU, so `cu_qp_delta_abs_bins == 0` and the reconstruction is the
    exact zero-MV reference copy. All-regular bins (robust against the
    test encoder's bypass defer bug).
  - `round100_inter_cu_qp_delta_abs_zero_decodes_as_single_u_bin`:
    engine-level isolation of the exact transform_unit() prefix the
    inter walker reads (`cbf_luma = 1`, `cu_qp_delta_abs = 0`) — the U
    "0" terminator decodes as 0 with no sign bit.
  - `round100_inter_cu_qp_delta_signed_magnitude_and_clamp`: exercises
    the eq. 148 signed-magnitude derivation + `[0, 51]` clamp the walker
    applies (sign, abs == 0, and both saturation corners).

#### Notes
- A full-slice non-skip MODE_INTER fixture with a non-zero CBF (which
  would drive `cu_qp_delta` end-to-end) is still blocked by the
  test-only `CabacEncoder::encode_bypass` defer bug on the residual
  `coeff_sign_flag` (documented round-90/95). The new tests therefore
  split coverage between a robust all-regular full-slice negative gate
  and engine/arithmetic-level positive checks of the new read, exactly
  as the round-95 IBC active-decode coverage did. The IDR-side IBC
  branch + the inter-IBC branch still hard-code `cu_qp = slice_qp`
  (they reach transform_unit() with the same condition); wiring
  `cu_qp_delta` into those two branches is a symmetric follow-up.

### Round 95 — non-IDR (P/B) IBC `coding_unit()` wiring

#### Added
- `decode_inter_coding_unit` (`src/slice_data.rs`): inside the
  `!cu_skip_flag` branch, after `pred_mode_flag` is consumed, the
  walker now evaluates §7.4.5 `isIbcAllowed` against the inter
  `SliceDecodeInputs::sps_ibc_flag` + `log2_max_ibc_cand_size`. When
  the gate holds, the regular-coded `ibc_flag` (Table 90 → Table 66
  init; under `sps_cm_init_flag = 0` the only ctxIdx is 0) is
  decoded. On `ibc_flag = 1` — and per §7.4.9.5 the IBC bit always
  wins over `pred_mode_flag` for `predModeConstraint =
  PRED_MODE_NO_CONSTRAINT` — the walker takes the §7.3.8.4
  lines 2868–2876 IBC syntax path: two `abs_mvd_l0` EG-0 bypass
  magnitudes (x then y) each with an optional `mvd_l0_sign_flag`
  bypass bit.
- `decode_inter_ibc_branch`: drives the §8.6.1 IBC pipeline inside
  the single-tree inter walker. Reads `cbf_luma` + (optionally)
  `cbf_cb` / `cbf_cr` via the normal Baseline cbf path, decodes per
  component residual coefficients, then hands off to
  `apply_inter_ibc_branch_predict_and_reconstruct`. The helper is
  symmetric to the IDR-side `decode_ibc_branch` (round 90) but
  operates against `InterDecodeStats` / `InterDecodeInputs` and
  single-tree.
- `apply_inter_ibc_branch_predict_and_reconstruct`: pure-compute
  closure (no CABAC engine, no bitstream) of the §8.6.1 step 1-5
  pipeline for the inter path. Calls `ibc::decode_ibc_cu` to predict
  luma + chroma from the current picture's reconstructed region,
  scale + IDCT the per-component residual coefficients, do the
  `clip(pred + res)` reconstruction (§8.7.5 eq. 1091), stamp
  `CuPredMode::Ibc` + the 1/16-pel luma MV into the side-info grid,
  and leave the HMVP candidate list untouched (per §8.5.2.7, IBC
  CUs do not contribute an inter-AMVP candidate). Single-tree
  inter-slice chroma destinations are scaled to chroma-pel
  coordinates before `pic.store_block` (no DUAL_TREE_CHROMA pass to
  compensate, unlike the IDR-side wiring).
- `InterDecodeStats`: new counters mirroring the IDR-side IBC
  trackers — `ibc_flag_bins`, `ibc_cus`, `ibc_abs_mvd_bins`,
  `ibc_mvd_sign_bins`.

#### Changed
- `decoder.rs::decode_non_idr`: lifted `sps_ibc_flag` from the
  Baseline-toolset unsupported gate. P/B slices with IBC-enabled SPS
  now drive through the slice walker symmetrically to the IDR path
  unblocked in round 90.

#### Tests
- 277 pass (was 272). 5 new tests:
  - `round95_inter_decode_without_ibc_flag_consumes_no_ibc_bins`:
    `sps_ibc_flag = 0` ⇒ zero `ibc_flag` / IBC-counter bins on the
    P-slice cu_skip path.
  - `round95_inter_decode_skips_ibc_flag_when_cu_exceeds_cand_size`:
    `sps_ibc_flag = 1` but `log2_max_ibc_cand_size = 1` ⇒ §7.4.5
    size gate suppresses `ibc_flag` emission (verified on the
    cu_skip path which intrinsically skips `ibc_flag` anyway).
  - `round95_inter_ibc_branch_predicts_from_left_neighbour`: direct
    exercise of the pure-compute helper. Pre-stamps a 4×4 luma
    pattern on the left half of an 8×4 monochrome picture, then
    runs the inter IBC helper with BV = (−4, 0) at the right-half
    CU. Verifies the right-half samples are a bit-exact copy of
    the left half, the side-info grid is stamped `CuPredMode::Ibc`,
    and the HMVP list remains empty.
  - `round95_inter_ibc_branch_rejects_non_conformant_bv`: BV (0, 0)
    overlapping the current CU short-circuits with `Error::Invalid`
    before any sample write.
  - `round95_inter_ibc_branch_chroma_residual_roundtrips`: 8×8 luma
    CB at (8, 0) on a 4:2:0 picture with chroma pre-fill on the
    left half + BV (−8, 0). Verifies the chroma destination
    coordinate scaling in `pic.store_block` puts the IBC chroma
    samples at the correct chroma-pel position.

#### Notes
- The crate-private test-only `CabacEncoder` has a pre-existing
  `encode_bypass` defer bug that breaks long mixed regular+bypass
  streams (documented in the round-90 CHANGELOG entry). End-to-end
  CABAC-driven fixtures for the inter IBC path therefore exercise
  the negative gates (`sps_ibc_flag = 0` + cu_skip suppression) and
  the bit-exact reconstruction is covered by the pure-compute
  helper tests.
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`).
  No xeve, xevd, ETM reference, or libavcodec evcdec consulted.

### Round 90 — IBC `coding_unit()` wiring

#### Added
- `walk_coding_unit` (`src/slice_data.rs`): when
  `ibc::is_ibc_allowed_for_size(sps_ibc_flag, log2_max_ibc_cand_size,
  log2CbWidth, log2CbHeight)` holds on the luma / single tree path,
  the walker decodes the regular-coded `ibc_flag` (Table 90 →
  Table 66 init; under `sps_cm_init_flag = 0` the only ctxIdx is 0).
  When the flag is 1, the walker follows the IBC syntax path of
  spec §7.3.8.4 lines 2868–2876: two `abs_mvd_l0` EG-0 bypass
  magnitudes (x then y) each with an optional `mvd_l0_sign_flag`
  bypass bit. The IBC CU's `intra_pred_mode` is suppressed and
  `transform_unit()` runs normally.
- `decode_coding_unit` IBC branch: same gate. On `ibc_flag = 1` the
  decoder calls `decode_signed_mvd` for `mvd_x` / `mvd_y` and hands
  the resolved `MotionVector` to a new `decode_ibc_branch` helper.
- `decode_ibc_branch`: consumes the luma residual via
  `decode_residual_coding_rle` + `scale_and_inverse_transform`, then
  delegates to `apply_ibc_branch_predict_and_reconstruct` (pure
  compute, no CABAC engine) for the §8.6.1 step 1-5 pipeline
  closure: predict luma + chroma via `ibc::decode_ibc_cu`, add the
  luma residual, `clip(pred + res)` per eq. 1091, stamp
  `CuPredMode::Ibc` + 1/16-pel luma MV into the side-info grid.
- `apply_ibc_branch_predict_and_reconstruct`: pure-compute helper
  that takes pre-decoded `(mvd, luma_residual_levels)` and runs the
  full §8.6.1 reconstruction. Bypasses the CABAC engine entirely,
  enabling bit-exact direct-call tests.
- `luma_cell_is_ibc(side_info, x, y)` predicate: probes the
  side-info grid at the matching luma cell for `CuPredMode::Ibc`.
  Used by the dual-tree-chroma `decode_coding_unit` pass to skip
  `intra_reconstruct_cb` (the chroma samples were already placed by
  `ibc::decode_ibc_cu`'s §8.6.3 step).
- `add_chroma_residual_to_block`: chroma equivalent of
  `intra_reconstruct_cb` minus the prediction step — adds a residual
  on top of already-placed chroma samples and clips to bit depth.
- `SliceWalkInputs` / `SliceDecodeInputs`: new `sps_ibc_flag` +
  `log2_max_ibc_cand_size` fields, with `Default` impls on both
  structs.
- `SliceWalkStats` / `SliceDecodeStats`: new counters
  `ibc_flag_bins`, `ibc_cus`, `ibc_abs_mvd_bins`, `ibc_mvd_sign_bins`.

#### Changed
- `walk_idr_slice` / `decode_idr_slice` (`src/lib.rs`): SPS-level
  `sps_ibc_flag = 1` gate **lifted**. Both entry points plumb
  `sps_ibc_flag` + `log2_max_ibc_cand_size` into the new
  `SliceWalkInputs` / `SliceDecodeInputs` fields.
- `decoder.rs` per-slice context build for non-IDR: plumbs
  `sps_ibc_flag` + `log2_max_ibc_cand_size` (the non-IDR P/B gate
  itself remains pending — see Followups).
- `decode_transform_unit`: new `luma_cu_is_ibc` parameter routes the
  chroma reconstruction past the intra-DC step when the matching
  luma CU was IBC, falling through to `add_chroma_residual_to_block`
  for non-zero `cbf_cb` / `cbf_cr`.

#### Tests
- 272 unit tests pass (up from 265). 7 new tests (all in
  `src/slice_data.rs`):
  - `round90_egk0_bypass_roundtrip`: EG-0 bypass values 0..=31
    round-trip through the test-only CABAC encoder/decoder pair.
  - `round90_idr_decode_without_ibc_flag_consumes_no_ibc_bins`: SPS
    gate off → no `ibc_flag` bin consumed; round-3 grey IDR fixture
    decodes to uniform 128.
  - `round90_idr_decode_skips_ibc_flag_when_cu_exceeds_cand_size`:
    `log2_max_ibc_cand_size = 1` < `log2CbWidth = 2` → walker
    suppresses `ibc_flag` per §7.4.5's size bullet.
  - `round90_ibc_branch_predicts_from_left_neighbour`: direct
    `apply_ibc_branch_predict_and_reconstruct` test — pre-populates
    a 4×4 luma block at (0,0) with a known 16-value gradient, calls
    the helper with BV = (−4, 0) at (4, 0), verifies the right-half
    samples are a bit-exact copy of the left-half pattern. Confirms
    `CuPredMode::Ibc` side-info stamp + `mv_l0_x = −64` (1/16-pel
    grid).
  - `round90_ibc_branch_rejects_non_conformant_bv`: BV = (0, 0)
    overlaps the current CU → returns `Error::Invalid` from
    `validate_ibc_constraints`; no samples written; side-info stays
    `CuPredMode::Intra`.
  - `round90_luma_cell_is_ibc_probe`: side-info grid probe helper
    correctly reports IBC-stamped cells and defaults to `false`
    elsewhere (including out-of-picture coordinates).
  - `round90_add_chroma_residual_clips_to_bit_depth`: chroma
    residual addition clips to `[0, 255]` for 8-bit.

#### Notes
- The crate-private test-only `CabacEncoder` has a pre-existing
  `encode_bypass` defer bug that desynchronises long mixed
  regular+bypass streams (~5+ consecutive bypass calls in the
  middle range produce a bit ordering the decoder reads
  inconsistently). End-to-end CABAC fixtures for IBC are therefore
  tested through the pure-compute
  `apply_ibc_branch_predict_and_reconstruct` helper for round 90.
  Fixing the encoder is deferred to a separate round (it doesn't
  affect production decode paths — the test-only encoder is gated
  on `#[cfg(test)]`).
- `decode_non_idr` (P/B slices) still gates on `sps_ibc_flag = 0`.
  Wiring IBC into the inter `coding_unit()` path requires the same
  `isIbcAllowed` probe inside `decode_inter_coding_unit` plus an
  IBC branch invocation when the IBC flag fires; deferred to a
  follow-up round.
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`).
  No xeve, xevd, ETM reference, libavcodec evcdec consulted.

### Round 74 — IBC pipeline composition + MV rounding helper

#### Added
- `ibc::decode_ibc_cu(cur_pic, xCb, yCb, nCbWL, nCbHL, mvd, ctbLog2SizeY,
  chroma_present, pred_y, pred_cb, pred_cr)` — chains §8.6.1 steps 1
  through 3 (derive luma MV → validate bitstream-conformance constraints
  → derive chroma MV → predict block) in a single call. Returns the
  resolved `(mvL, mvC)` pair so the caller can stamp the per-4×4 side-
  info grid and run an HMVP update. Steps 4 (residual decode, §8.5.6.1)
  and 5 (picture-construction prior to in-loop filtering, §8.7.5) stay
  in their existing modules — they are shared with the inter pipeline.
- `ibc::is_ibc_allowed_for_size(sps_ibc_flag, log2MaxIbcCandSize,
  log2CbWidth, log2CbHeight)` — the structural part of the §7.4.5
  `isIbcAllowed` predicate. The dual-tree / `predModeConstraint`
  bullet stays caller-side (the walker tracks the tree-type enum
  internally).
- `Sps::log2_max_ibc_cand_size()` / `Sps::max_ibc_cand_size()` — eq. 70
  derived variables (`log2MaxIbcCandSize = 2 + log2_max_ibc_cand_size_minus2`
  and the corresponding `1 << log2MaxIbcCandSize`). Both return `None`
  when `sps_ibc_flag == 0` so callers don't read an undefined field.
- `inter::round_motion_vector(mv, rightShift, leftShift)` — §8.5.3.10
  eq. 907-909 standalone helper. Computes
  `((mv[k] + offset − (mv[k] >= 0)) >> rightShift) << leftShift`
  with `offset = (rightShift == 0) ? 0 : 1 << (rightShift − 1)`. The
  arithmetic right shift preserves the sign for negative MVs (rounding
  toward negative infinity), and the `− (mv >= 0)` term gives the
  spec's asymmetric rounding direction (round toward negative infinity
  on positives too). Needed by the §8.5.3 affine derivation paths
  (eq. 911, 918, 953, 962, 1023) and by AMVR resolution scaling.

#### Tests
- 265 unit tests pass (up from 251). 14 new tests cover:
  - `ibc::decode_ibc_cu` (3 tests): pipeline matches individual
    derivers + predictor on a 32×32 gradient; non-conformant BV
    short-circuits before any sample read; luma-only path leaves
    sentinel chroma buffers untouched.
  - `ibc::is_ibc_allowed_for_size` (4 tests): flag-off rejects;
    equal-to-limit accepts; larger-than-limit rejects; per-axis
    independence (a 64-wide × 32-tall under a 64-sample limit
    accepts; a 128-wide × 32-tall under the same limit rejects on
    width alone).
  - `Sps::log2_max_ibc_cand_size` (2 tests): IBC-disabled returns
    `None`; IBC-enabled returns `2 + minus2` across the full spec
    range 0..=4 of `log2_max_ibc_cand_size_minus2`.
  - `inter::round_motion_vector` (5 tests): `right_shift = 0`
    short-circuits the offset; `right_shift = 2` rounds positive
    components toward negative infinity (7 → 2, 4 → 1); negative
    components also round toward negative infinity (−4 → −1, −7 → −2,
    −8 → −2); combined right-then-left shift recovers MV resolution
    (11 → 12 at rs=2, ls=2); zero MV at `right_shift = 1` is identity
    (offset cancels exactly).

#### Notes
- The `sps_ibc_flag = 1` SPS-level gate in `walk_idr_slice`,
  `decode_idr_slice`, and `decode_non_idr` is **still unchanged**.
  Lifting it needs the CABAC walker to emit `ibc_flag` in
  `coding_unit()`, parse the IBC `abs_mvd_l0` / `mvd_l0_sign_flag`
  (binariser already supports both), and route through
  `decode_ibc_cu`. The §8.6.1 step-4 residual decode currently lives
  in the inter slice walker; sharing it for IBC needs a small
  refactor.
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`).
  No xeve, xevd, ETM reference, libavcodec evcdec consulted.

### Round 73 — IBC (Intra Block Copy) primitive scaffold

#### Added
- `ibc` module (`src/ibc.rs`): clean-room transcription of ISO/IEC
  23094-1 §8.6 (decoding process for IBC-coded coding units). Four
  pure-function primitives:
  - `derive_ibc_luma_mv(mvd)` — §8.6.2.1 eq. 1025-1039. Folds the
    parsed `MvdL0` into a signed-16-bit `mvL` via the spec's
    `(mvp + mvd + 2^16) % 2^16` modular wrap (mvp = 0 for IBC), then
    shifts left by 4 to land on the 1/16-pel grid the §8.5.4.3
    interpolator expects.
  - `derive_ibc_chroma_mv(mvL, chroma_format_idc)` — §8.6.2.2 eq.
    1040-1041. Computes `mvC[k] = (mvL[k] >> (3 + SubXC)) * 32`,
    handling all four `chroma_format_idc` values (monochrome → zero
    chroma MV; 4:2:0 / 4:2:2 / 4:4:4 → spec sub-sampling).
  - `validate_ibc_constraints(mvL, xCb, yCb, nCbW, nCbH, ctbLog2SizeY)`
    — §8.6.2.1 bitstream-conformance rules. Enforces:
    - the "at least one of mvL[0]+nCbW ≤ 0, mvL[1]+nCbH ≤ 0" guard
      (reference block lies strictly above-or-left of current CB);
    - eq. 1035/1036 — same-CTU-row constraint on yRefTL and yRefBL;
    - eq. 1037 — xRefTL must be in current or left CTU column;
    - eq. 1038 — xRefTR cannot cross into the right CTU;
    - fractional-pel BVs are rejected (eq. 1039 guarantees mvL low
      nibble is zero);
    - negative reference-picture coordinates are rejected.
    Returns `Err(Error::Invalid)` with a per-condition diagnostic on
    non-conformant BVs. The `sps_suco_flag = 1` extra rules are
    deferred (suco_flag isn't supported anywhere else in the decoder
    either).
  - `predict_ibc_block(cur_pic, xCb, yCb, nCbW, nCbH, mvL, mvC,
    chroma_present, pred_y, pred_cb, pred_cr)` — §8.6.3. Integer-pel
    rectangular copy from the current picture's already-reconstructed
    region into the luma + chroma prediction buffers. Since eq. 1039
    makes the BV land on an integer sample, this collapses the spec's
    "invoke §8.5.4.3.1 fractional sample interpolation" step into a
    direct memcpy (a clean-room observation, not from a reference
    decoder). Picture-edge clamping is per the standard
    `Clip3(0, picW − 1, …)` convention.
- 25 unit tests (`ibc::tests::*`) covering:
  - Luma MV derivation: zero MVD, negative MVD, positive MVD, 16-bit
    wrap;
  - Chroma MV derivation: monochrome (returns zero), 4:2:0 (halves x
    and y), 4:2:2 (halves x only), 4:4:4 (no halving), negative-sign
    preservation through arithmetic shift;
  - Constraint validation: above-only reference (ok), left-only (ok),
    overlapping reference (rejected), cross-CTU-row (rejected),
    left-neighbour CTU (allowed), two CTUs to the left (rejected),
    fractional BV (rejected), zero CB dims (rejected), bad
    `ctbLog2SizeY` (rejected: 4 and 8), negative reference origin
    (rejected);
  - Block prediction: copy of an above block (gradient pattern
    verified per-pixel), copy of a left block, luma-only path when
    `chroma_present = false` (chroma buffers untouched), buffer-size
    mismatch (rejected), fractional luma MV (rejected);
  - Pipeline integration: `derive → validate → derive_chroma → predict`
    end-to-end on a 32×32 gradient picture with the left-neighbour BV.

#### Changed
- 251 unit tests pass (was 226 before round 73). The `sps_ibc_flag = 1`
  SPS-level gate in `walk_idr_slice`, `decode_idr_slice` and
  `decode_non_idr` is **unchanged** — the next round will lift it
  once the CABAC walker emits IBC CUs and the §8.6.1 5-step
  decoding pipeline is wired in `coding_unit()`. Round 73 only lands
  the §8.6.2 / §8.6.3 primitives with unit-test coverage; the new
  module is `pub` so external integration tests can drive it
  directly.

#### Notes
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`).
  No xeve, xevd, ETM reference, libavcodec evcdec consulted.

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
