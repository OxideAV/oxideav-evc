# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Baseline + Main profiles. Zero C dependencies, zero FFI,
zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round-120 status

Round 120 closes the documented round-117 follow-up by landing the
**spec-faithful §7.3.5 `alf_data()` parser + the §8.9.4 `AlfCoeffL`
class-to-filter derivation (eq. 96-104) + the §8.8.4.2 classified per-CTB
luma apply (eq. 1281-1288)**. Until now `parse_alf_data` consumed a
simplified u(6) abs + u(1) sign payload, and the apply applied
`luma_filters[0]` uniformly across every flagged CTB. The new parser
handles the full §7.3.5 syntax — `alf_luma_type_flag` (eq. 90/91
`coefPosMap`), `alf_luma_num_filters_signalled_minus1`,
`alf_luma_coeff_delta_idx[ ]` (u(v) with `Ceil(Log2(NumSignalledFilter))`),
`alf_luma_fixed_filter_usage_pattern` (uek(v), k=0) +
`alf_luma_fixed_filter_usage_flag[ ]` + `alf_luma_fixed_filter_set_idx[ ]`
(u(4)), `alf_luma_coeff_delta_flag` + `alf_luma_coeff_delta_prediction_flag`,
`alf_luma_min_eg_order_minus1` (ue) + `alf_luma_eg_order_increase_flag[ ]`
(eq. 92/93 `expGoOrderY` chain), `alf_luma_coeff_flag[ ]`, and per-tap
`alf_luma_coeff_delta_abs[ ][ ]` (uek(v) with order picked by eq.
`golombOrderIdxY[ ]`) + `alf_luma_coeff_delta_sign_flag[ ][ ]`. Chroma
mirrors the uek(v) signalling per §7.3.5 second half (eq. 109/110 instead
of the round-11 mis-scaled DC). The new `derive_alf_coeff_l` then
assembles all 25 per-class `AlfCoeffL[ filtIdx ][ 0..12 ]` arrays per
eq. 96-104: starts from the 64-row [`alf_tables::ALF_FIX_FILT_COEFF`] (eq.
102, transcribed verbatim) using [`alf_tables::ALF_CLASS_TO_FILT_MAP`] (eq.
103) when `alf_luma_fixed_filter_usage_flag[ filtIdx ] == 1`, adds the
per-class delta `filterCoefficients[ alf_luma_coeff_delta_idx[ filtIdx ] ][
coefPosMap[ j ] − 1 ]` for every `coefPosMap[ j ] > 0` (eq. 100), and
computes position 12 per eq. 104. The new
`apply_alf_luma_classified` / `apply_alf_luma_classified_masked` close
the §8.8.4.2 loop: for every sample of every flagged CTB, the round-117
`derive_alf_classification` picks `filtIdx[ x ][ y ]` + `transposeIdx`,
the `AlfCoeffL[ filtIdx ][ ]` row is permuted per eq. 1282-1285, and the
sum is reduced per eq. 1286/1287 (the centre tap multiplies
`recPicture[ x, y ]`, scaled by `(sum + 256) >> 9` — the spec's exact
scaling, not the round-11 `(sum + 64) >> 7` approximation). The decoder
threads this through `apply_post_filters` so masked-luma ALF now uses the
classified path; the minimal-header IDR (all-off map) falls back to the
whole-plane `apply_alf` (now also spec-scaled). 321 unit tests pass (was
309): full §7.3.5 syntax round-trip with the new `BitEmitter::uek` writer
(`alf_luma_coeff_delta_idx`, fixed-filter pattern 0/1/2,
`alf_luma_coeff_delta_prediction_flag` cumulative-sum check, multi-EG-order
chain, chroma EG progression), eq. 96-104 derivation against the spec
tables (pattern-1 every-class fixed-filter seed reproduces
[`alf_tables::ALF_FIX_FILT_COEFF`] under [`alf_tables::ALF_CLASS_TO_FILT_MAP`]),
eq. 104 DC invariant verified on a multi-class config, classified apply
on uniform / vertical-edge / partial-edge-CTB / two-CTB-mask pictures, and
the spec tables' dimensions / spot-checked entries. Suggested workspace-
README row delta: EVC now decodes the full §7.3.5 `alf_data()` and derives
the §8.9.4 per-class `AlfCoeffL` + applies the §8.8.4.2 classified luma
filter per CTB (lacks: per-CTU ALF filter-set selection §8.9.6,
Main-profile toolset — BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-117 status

Round 117 lands the §8.8.4.3 **ALF transpose + classification filter-index
derivation** — the per-luma-sample gradient classification that selects, for
each sample of a coding tree block, which of the 25 ALF filter classes
(`filtIdx`) and which of the 4 coefficient permutations (`transposeIdx`)
apply. Until now the §8.9 apply used filter set 0 uniformly across every
flagged CTB; §8.8.4.3 is the input to the §8.8.4.2 per-sample filter
selection. A new pure `derive_alf_classification` faithfully transcribes
eq. 1289-1320: per-position Laplacian gradients `filtH/V/D0/D1` over the
[−2, blk+1] halo, per-4×4-subblock window sums (`i, j = −2..5`), the
direction-strength branch (`dir1/dir2/dirS` via the eq. 1310-1314 diagonal-
vs-HV cross-product test, computed in `i64` to avoid overflow), the `varTab`
activity quantisation with the eq. 1316 `BitDepthY − 2` shift, and the final
`transposeTable`-driven `transposeIdx` (eq. 1317-1318) + `filtIdx` with the
eq. 1320 direction offset. `transpose_luma_coeffs` applies the §8.8.4.2
eq. 1282-1285 13-tap permutation a sample's `transposeIdx` requests. Both are
syntax-free building blocks: classification is unit-tested directly against
an independently-coded reference re-derivation (exhaustive sample cross-check
on a pseudo-random plane, full + edge CTBs) plus targeted degenerate / range
/ bit-depth cases. 309 unit tests pass (was 300). Wiring the classified
per-sample selection into the apply needs the full §8.9.4
`AlfCoeffL[ ][ filtIdx ][ ]` derivation (eq. 96-104 +
`alf_luma_coeff_delta_idx` + fixed filters), which the round-11 simplified
`alf_data()` parser does not yet capture — that's the documented follow-up.
Suggested workspace-README row delta: EVC now derives the §8.8.4.3 per-sample
ALF classification (`filtIdx` / `transposeIdx`) + the §8.8.4.2 coefficient
transpose (lacks: §8.9.4 `AlfCoeffL` class-to-filter derivation wiring,
per-CTU ALF filter-set selection §8.9.6, Main-profile toolset —
BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-113 status

Round 113 wires the round-107 **per-CTU ALF applicability map** into the
**§8.9 apply** — the documented round-107 follow-up. Until now `apply_alf`
filtered whole planes regardless of the per-CTB `alf_ctb_flag`; the §8.9
luma loop (lines 18059-18074) actually invokes the coding-tree-block luma
filtering process *only* for CTBs whose `alf_ctb_flag[rx][ry]` is 1. A new
`AlfCtbMap` records the resolved (present-or-inferred) per-CTU triplet during
the IDR + inter CTU loops; `apply_alf_luma_masked` filters only the flagged
luma CTBs, with the §8.9 `blkWidth` / `blkHeight` picture-edge clamp and a
whole-plane pre-filter snapshot so a filtered CTB's edge taps still read
unfiltered neighbours. `apply_alf_with_map` adds the §8.9 chroma path for
`ChromaArrayType` 1..2 (the plane is filtered when the slice-level chroma
enable holds; the per-CTB chroma map flags are inferred 0 in Baseline
4:2:0). The decoder threads the map (and the chroma enables) out of
`decode_non_idr` via a new `NonIdrDecodeResult` and into `apply_post_filters`;
an all-off map (the minimal-header IDR path) falls back to the whole-plane
apply, so existing behaviour is preserved. `SliceDecodeStats` /
`InterDecodeStats` gain an `alf_ctb_map` field (both drop their `Copy`
derive). 300 unit tests pass (was 292): masked apply touches only flagged
CTBs, clamps a partial edge CTB, an all-on map reproduces the whole-plane
result bit-for-bit, chroma is gated by the slice enable, and an end-to-end
64×32 IDR with one CTB on / one off filters only the flagged CTB. Suggested
workspace-README row delta: EVC now masks the §8.9 ALF luma apply per coding
tree block by the decoded §7.3.8.2 map (lacks: per-CTU ALF filter-set
selection §8.9.6 / ALF classification §8.8.4.3, Main-profile toolset —
BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-107 status

Round 107 lands the §7.3.8.2 `coding_tree_unit()` **adaptive-loop-filter
applicability map** — the per-CTU `alf_ctb_flag` / `alf_ctb_chroma_flag`
/ `alf_ctb_chroma2_flag` syntax that, until now, every CTU loop skipped
(rounds ≤103 recursed straight into `split_unit()`). The slice-header
parser (§7.3.4) now surfaces the map controls it used to parse-and-drop —
`slice_alf_map_flag`, `slice_alf_chroma_idc` with its §7.4.5-derived
`sliceChromaAlfEnabledFlag` / `sliceChroma2AlfEnabledFlag`, and the
`ChromaArrayType == 3`-only chroma map flags. A new
`decode_coding_tree_unit_alf` helper decodes the 0-3 ae(v) flags (FL
`cMax = 1`, Table 40, ctxInc 0 under `sps_cm_init_flag == 0`) gated
exactly as the spec syntax and applies the §7.4.9.2 not-present
inference, returning an `AlfCtbFlags` triplet. It is wired into all three
CTU loops (IDR walk, IDR decode, P/B inter decode) before `split_unit()`;
the `decode_non_idr` path threads the real slice-header values through new
`SliceWalkInputs` fields. For Baseline 4:2:0 only the luma bin can appear
(chroma map flags are inferred 0); the full triplet is decoded for
spec-completeness on the ChromaArrayType-3 path. New `AlfCtbStats`
counters (`luma_bins` / `chroma_cb_bins` / `chroma_cr_bins` /
`luma_on_ctus`) thread into all three stats structs. 292 unit tests pass
(was 284): engine-level isolation of the present/absent/inferred gating,
the ChromaArrayType-3 three-bin path, end-to-end IDR decode with and
without the luma map, and slice-header parse of the new fields. Masking
the §8.9 ALF apply per-CTB by the decoded map (today `apply_alf` filters
whole planes) is a documented follow-up. Suggested workspace-README row
delta: EVC now decodes the §7.3.8.2 per-CTU ALF applicability map in
every CTU loop (lacks: per-CTB ALF apply-masking, Main-profile toolset —
BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-103 status

Round 103 extends the §7.3.8.5 `cu_qp_delta` wiring to the **two IBC
branches**. Round 100 wired the element into the regular (non-IBC)
inter `coding_unit()` path, but the IDR-side `decode_ibc_branch` and
the non-IDR `decode_inter_ibc_branch` still hard-coded
`cu_qp = slice_qp`. The §7.3.8.5 line 3073 presence condition is
mode-independent, so a MODE_IBC CU now reads `cu_qp_delta_abs` (U
binarization, ctxInc 0 per Table 95 / Table 78 init) + the bypass
`cu_qp_delta_sign_flag` right after the cbf bins and before the
residual, applies eq. 148 clamped to `[0, 51]`, and threads the
derived per-CU QP into the residual scaling via a new `cu_qp` parameter
on both `apply_*_ibc_branch_predict_and_reconstruct` helpers. The
IDR-path `SliceDecodeStats` gains a `cu_qp_delta_abs_bins` counter
(mirroring the inter-side one) that the intra single-tree path now also
increments. 284 unit tests pass (was 280): engine-level isolation of
the new read (single all-regular U "0" bin), the eq. 148
signed-magnitude + clamp arithmetic, and two direct-call helper checks
that a fixed non-zero residual reconstructs differently at QP 22 vs
QP 40 (the full-slice non-skip CBF path is still blocked by the
test-only encoder's `encode_bypass` defer bug). With this round all
four `transform_unit()` entry points (intra single-tree, regular inter,
IDR IBC, non-IDR IBC) decode per-CU `cu_qp_delta`. Suggested
workspace-README row delta: EVC now decodes per-CU `cu_qp_delta` on
every Baseline `transform_unit()` path including both IBC branches
(lacks: Main-profile toolset — BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-100 status

Round 100 wires the §7.3.8.5 `cu_qp_delta` syntax into the **non-skip
MODE_INTER** `coding_unit()` path. Previously P/B inter CUs ignored
`cu_qp_delta_enabled_flag` and always reconstructed at the slice QP;
the inter `transform_unit()` walker now decodes `cu_qp_delta_abs` (U
binarization, ctxInc 0 per Table 95 / Table 78 init) + the bypass
`cu_qp_delta_sign_flag` after the cbf bins, applies eq. 148
(`QpY = slice_qp + cu_qp_delta_abs * (1 - 2 * sign)`) clamped to
`[0, 51]`, and feeds the derived per-CU QP into the residual scaling —
symmetric to the intra single-tree path. New
`InterDecodeStats::cu_qp_delta_abs_bins` counter. 280 unit tests pass
(was 277): a robust all-regular full-slice negative-gate test (skip CU
emits zero `cu_qp_delta` bins) plus engine/arithmetic-level positive
checks of the new read (the full-slice non-skip CBF path is still
blocked by the test-only encoder's `encode_bypass` defer bug on the
residual `coeff_sign_flag`). Suggested workspace-README row delta: EVC
now decodes per-CU `cu_qp_delta` on both the intra and the regular
inter paths (lacks: full inter-IBC-branch `cu_qp_delta`, Main-profile
toolset).

## Round-95 status

Working **Baseline-profile** decoder for IDR + P + B slices with full
residual coding, luma + chroma deblocking, the 64-point IDCT, the
**Main-profile CABAC initialization tables** (Tables 40-90) and
§9.3.4.2 ctxInc derivation helpers, the round-8 RPL parser + HMVP
infrastructure, the round-9 multi-ref DPB + POC reordering, the
round-10 spec-compliance additions, the round-11 post-filter
pipeline, the round-73/74 **IBC primitives**, the round-90
**IDR-slice IBC `coding_unit()` wiring**, and round-95's **non-IDR
(P/B) IBC `coding_unit()` wiring**: the `decode_inter_coding_unit`
path now decodes the regular-coded `ibc_flag` under §7.4.5
`isIbcAllowed`, the §7.3.8.4 lines 2868–2876 IBC syntax (two
`abs_mvd_l0` EG-0 magnitudes + optional `mvd_l0_sign_flag` per
component), and routes through a new `decode_inter_ibc_branch` →
`apply_inter_ibc_branch_predict_and_reconstruct` pair that closes
the §8.6.1 1-5 pipeline (single-tree luma + chroma reconstruction,
side-info grid stamp, HMVP-no-op gate). The SPS-level `sps_ibc_flag
= 1` gate is now **lifted on both IDR and non-IDR (P/B) paths**.

## Round-95 deltas vs round 90

- **`decode_inter_coding_unit` IBC branch** (`src/slice_data.rs`):
  inside the `!cu_skip_flag` branch, after the `pred_mode_flag` is
  consumed, when `is_ibc_allowed_for_size(decode.sps_ibc_flag,
  decode.log2_max_ibc_cand_size, log2CbWidth, log2CbHeight)` holds,
  the walker decodes the regular-coded `ibc_flag` bin (Table 90 →
  Table 66 init; `sps_cm_init_flag = 0` ⇒ single ctxIdx 0). On
  `ibc_flag = 1` the walker parses `abs_mvd_l0[0/1]` (EG-0 bypass)
  + optional `mvd_l0_sign_flag` per component, then hands the
  resolved `MotionVector` to **`decode_inter_ibc_branch`**. Per
  §7.4.9.5 the IBC flag always wins over `pred_mode_flag` when
  `predModeConstraint = PRED_MODE_NO_CONSTRAINT`.
- **`decode_inter_ibc_branch`**: drives the §8.6.1 IBC pipeline for
  the single-tree P/B CU. Reads `cbf_luma` + (optionally) `cbf_cb`
  / `cbf_cr` on the standard Baseline cbf path, decodes per-component
  residual coefficients via `decode_residual_coding_rle`, then hands
  off to `apply_inter_ibc_branch_predict_and_reconstruct`.
- **`apply_inter_ibc_branch_predict_and_reconstruct`**: pure-compute
  closure of the §8.6.1 step 1-5 pipeline. Calls `ibc::decode_ibc_cu`
  for luma + chroma prediction, scales + IDCT the residuals,
  `clip(pred + res)` per eq. 1091, stamps `CuPredMode::Ibc` into
  the side-info grid. Single-tree inter slices have no
  DUAL_TREE_CHROMA pass to compensate, so the helper scales the
  luma `(x0, y0)` to chroma-pel coordinates before `pic.store_block`
  on the chroma planes (`x_c = x0 / SubWidthC`, similarly for y).
  The §8.5.2.7 HMVP update is a no-op for IBC CUs (both ref_idx
  slots remain −1, so `HmvpCandList::update`'s validity gate drops
  the candidate by construction — matching the spec intent that
  IBC CUs do not contribute an inter-AMVP candidate).
- **`InterDecodeStats`**: new IBC counters — `ibc_flag_bins`,
  `ibc_cus`, `ibc_abs_mvd_bins`, `ibc_mvd_sign_bins` — mirroring
  the round-90 IDR-side trackers.
- **SPS gate lifted on non-IDR path**: `decoder.rs::decode_non_idr`
  no longer returns `Error::Unsupported` for `sps_ibc_flag = 1`
  streams. The remaining Baseline-toolset gate (`sps_btt_flag`,
  `sps_admvp_flag`, `sps_eipd_flag`, etc.) is unchanged.
- **277 unit tests pass** (was 272). 5 new tests:
  - `round95_inter_decode_without_ibc_flag_consumes_no_ibc_bins`:
    `sps_ibc_flag = 0` ⇒ zero `ibc_flag` / IBC-counter bins on the
    P-slice cu_skip path.
  - `round95_inter_decode_skips_ibc_flag_when_cu_exceeds_cand_size`:
    `sps_ibc_flag = 1` but `log2_max_ibc_cand_size = 1` ⇒ §7.4.5
    size gate suppresses `ibc_flag` emission.
  - `round95_inter_ibc_branch_predicts_from_left_neighbour`: direct
    exercise of the pure-compute helper. Pre-stamps a 4×4 luma
    pattern on the left half of an 8×4 monochrome picture, then
    runs the inter IBC helper with BV = (−4, 0) at the right-half
    CU. Verifies the right-half samples are a bit-exact copy of
    the left half, the side-info grid is stamped `CuPredMode::Ibc`
    with `mv_l0_x = −64` (1/16-pel grid), and the HMVP list
    remains empty.
  - `round95_inter_ibc_branch_rejects_non_conformant_bv`: BV (0, 0)
    overlapping the current CU short-circuits with `Error::Invalid`
    from `validate_ibc_constraints` before any sample is written.
  - `round95_inter_ibc_branch_chroma_residual_roundtrips`: 8×8 luma
    CB on a 4:2:0 picture verifies the chroma-pel coordinate
    scaling in `pic.store_block` puts the IBC chroma samples at
    the correct chroma destination.

### Round 95 follow-ups

- The CabacEncoder `encode_bypass` defer bug (documented in the
  round-90 follow-ups) still blocks end-to-end CABAC fixtures for
  the inter IBC active-decode path. Bit-exact reconstruction is
  covered by the pure-compute helper tests; revisiting the encoder
  to enable a full CABAC-driven inter IBC fixture remains scoped
  for a future round.

## Round-90 status

Working **Baseline-profile** decoder for IDR + P + B slices with full
residual coding, luma + chroma deblocking, the 64-point IDCT, the
**Main-profile CABAC initialization tables** (Tables 40-90) and
§9.3.4.2 ctxInc derivation helpers, the round-8 RPL parser + HMVP
infrastructure, the round-9 multi-ref DPB + POC reordering, the
round-10 spec-compliance additions, the round-11 post-filter
pipeline, the round-73/74 **IBC primitives** (§8.6 derivation +
validation + integer-pel block copy + §8.6.1 `decode_ibc_cu` pipeline
composition + §7.4.5 `isIbcAllowed` gate + SPS `log2MaxIbcCandSize`
helpers), and the round-90 **IBC `coding_unit()` wiring**: the per-CU
walker / decoder both surface the §7.3.8.4 IBC branch (regular-coded
`ibc_flag` gated on `isIbcAllowed`; EG-0 `abs_mvd_l0` + optional
`mvd_l0_sign_flag` per component; route through `decode_ibc_cu` for
the §8.6.1 step 1-3 prediction; `cbf_luma` inferred 1 per spec
§7.4.9.5 line 6065-6066 in DUAL_TREE_LUMA → residual decode + IDCT +
`clip(pred + res)` picture construction; `CuPredMode::Ibc` stamped on
the side-info grid so the matching DUAL_TREE_CHROMA `coding_unit()`
skips its intra reconstruction and leaves the IBC-placed chroma
samples in place). The SPS-level `sps_ibc_flag = 1` gate in
`walk_idr_slice` and `decode_idr_slice` is **lifted**.

## Round-90 deltas vs round 74

- **`walk_coding_unit` IBC branch** (`src/slice_data.rs`): when
  `is_luma_tree` is true AND `ibc::is_ibc_allowed_for_size(...)`
  holds, the parse-only walker decodes the regular-coded `ibc_flag`
  bin (Table 90 → Table 66 init; under `sps_cm_init_flag = 0` the
  only ctxIdx is 0). When the flag is 1, the walker follows the IBC
  syntax path of spec lines 2868–2876: two `abs_mvd_l0` EG-0 bypass
  magnitudes (x then y) each with an optional `mvd_l0_sign_flag`
  bypass bit, then falls through to `transform_unit()`. Otherwise
  the existing intra `intra_pred_mode` U-binarised read runs.
- **`decode_coding_unit` IBC branch**: same gate logic; on `ibc_flag
  = 1` the decoder runs `decode_signed_mvd` for `mvd_x` / `mvd_y`
  and hands the resolved `MotionVector` to the new
  `decode_ibc_branch` helper.
- **`decode_ibc_branch`**: consumes the luma residual via the
  existing `decode_residual_coding_rle` + `scale_and_inverse_transform`
  chain, then hands off to **`apply_ibc_branch_predict_and_reconstruct`**
  (pure compute, no CABAC) for the §8.6.1 5-step pipeline closure:
  predict luma + chroma into temporary buffers via `ibc::decode_ibc_cu`,
  add the luma residual, `clip(pred + res)` per eq. 1091, write the
  CU back to the picture buffer, stamp the side-info grid with
  `CuPredMode::Ibc` + the 1/16-pel luma MV (so the deblocking pass
  treats IBC edges with BS = 2 per Table 33).
- **`decode_transform_unit` chroma gate**: a new `luma_cu_is_ibc`
  parameter — derived per-CU by probing the side-info grid at the
  matching luma cell via `luma_cell_is_ibc` — suppresses the
  `intra_reconstruct_cb` chroma reconstruction. When non-zero chroma
  CBFs are signalled, `add_chroma_residual_to_block` adds the
  residual on top of the already-placed IBC chroma samples instead.
- **`SliceWalkInputs` / `SliceDecodeInputs`**: new `sps_ibc_flag` +
  `log2_max_ibc_cand_size` fields plumbed from SPS through both
  walk / decode entry points (`walk_idr_slice` / `decode_idr_slice`
  in `lib.rs`, `decode_non_idr` in `decoder.rs`). `Default` impls on
  both structs so legacy test sites get the no-IBC defaults.
- **`SliceWalkStats` / `SliceDecodeStats`**: new counters
  `ibc_flag_bins`, `ibc_cus`, `ibc_abs_mvd_bins`, `ibc_mvd_sign_bins`
  for fixture verification.
- **SPS gate lifted**: `walk_idr_slice` and `decode_idr_slice` no
  longer return `Error::Unsupported` on `sps_ibc_flag = 1` streams
  (gate removed from both lists in `lib.rs`; the `decoder.rs`
  non-IDR path still gates pending a P/B IBC wire-through in a
  follow-up round).
- **272 unit tests pass** (was 265). 7 new tests (all in
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
    `apply_ibc_branch_predict_and_reconstruct` test — pre-populates a
    4×4 luma block at (0,0) with a known gradient, calls the helper
    with BV = (−4, 0) at (4, 0), verifies the right-half samples are
    a bit-exact copy of the left-half. Confirms `CuPredMode::Ibc`
    side-info stamp + `mv_l0_x = −64` (1/16-pel grid).
  - `round90_ibc_branch_rejects_non_conformant_bv`: BV = (0, 0)
    overlaps the current CU → returns `Error::Invalid` from
    `validate_ibc_constraints`; no samples written; side-info stays
    `CuPredMode::Intra`.
  - `round90_luma_cell_is_ibc_probe`: side-info grid probe helper
    correctly reports IBC-stamped cells and defaults to `false`
    elsewhere (including out-of-picture coordinates).
  - `round90_add_chroma_residual_clips_to_bit_depth`: chroma
    residual addition clips to `[0, 255]` for 8-bit.

### Round 90 follow-ups

- The crate-private test-only `CabacEncoder` has a pre-existing
  `encode_bypass` defer bug that desynchronises long mixed
  regular+bypass streams (the `outstanding`-bit accumulation in
  `encode_bypass`'s middle-range path is consistent with the
  H.264 reference encoder, but observed behaviour after ~5
  consecutive bypass calls is misaligned with the decoder). End-to-end
  CABAC fixtures for IBC are therefore tested via the pure-compute
  `apply_ibc_branch_predict_and_reconstruct` helper for round 90.
  Fix-up scope for a future round: revisit `encode_bypass` against
  the M-coder spec; once corrected, re-add the full `coding_unit()`
  IBC round-trip fixture.
- `decode_non_idr` (P/B slices) still gates on `sps_ibc_flag = 0`.
  The `cu_skip_flag` / `pred_mode_flag` interactions of the IBC
  syntax in inter slices are unchanged from spec §7.3.8.4 — wiring
  them requires the same `isIbcAllowed` probe inside
  `decode_inter_coding_unit` plus a similar `decode_ibc_branch`
  invocation when the IBC flag fires.

## Round-74 deltas vs round 73

- **`ibc::decode_ibc_cu`** (`src/ibc.rs`): one-call wrapper that
  chains §8.6.1 steps 1-3: `derive_ibc_luma_mv` → `validate_ibc_constraints`
  → `derive_ibc_chroma_mv` (when chroma is present) → `predict_ibc_block`.
  Returns the resolved `(mvL, mvC)` pair so the caller can stamp the
  side-info grid and update HMVP. Step 4 (residual) and step 5
  (picture reconstruction §8.7.5) stay in their existing modules — they
  are shared with the inter pipeline.
- **`ibc::is_ibc_allowed_for_size`**: the structural part of the
  §7.4.5 `isIbcAllowed` predicate — `sps_ibc_flag == 1` AND both
  `log2CbWidth, log2CbHeight ≤ log2MaxIbcCandSize`. The `treeType` /
  `predModeConstraint` rules stay caller-side because they depend on
  the dual-tree state the CABAC walker tracks.
- **`Sps::log2_max_ibc_cand_size` / `Sps::max_ibc_cand_size`**: eq. 70
  derived variables. Return `None` when IBC is disabled at the SPS
  level so the caller can short-circuit before consulting the cand
  size.
- **`inter::round_motion_vector`** (`src/inter.rs`): §8.5.3.10 eq.
  907-909 standalone helper. `((mv + offset − (mv >= 0)) >> rightShift)
  << leftShift` with `offset = (rightShift == 0) ? 0 : 1 << (rightShift − 1)`.
  Needed by the §8.5.3 affine derivation paths (eq. 911, 918, 953, 962,
  1023) and by AMVR resolution scaling.
- 14 new unit tests (now 265 total, was 251): IBC pipeline end-to-end
  (chains match individual derivers; non-conformant BV short-circuits
  before any sample read; luma-only path leaves chroma buffers
  untouched), `isIbcAllowed` size gate (flag-off, equal-to-limit
  acceptance, larger-than-limit rejection, per-axis independence),
  SPS `log2_max_ibc_cand_size` (disabled → None; enabled across the
  full spec range 0..=4 of `log2_max_ibc_cand_size_minus2`), and
  §8.5.3.10 rounding (zero-right-shift no-offset, positive vs negative
  rounding direction at right_shift=2, combined right+left shift, and
  the right_shift=1 zero-MV round-trip).
- The `sps_ibc_flag = 1` SPS-level gate in `walk_idr_slice`,
  `decode_idr_slice`, and `decode_non_idr` is **still unchanged** —
  lifting it needs the CABAC walker to emit `ibc_flag` in
  `coding_unit()`, parse `abs_mvd_l0` / `mvd_l0_sign_flag` for the
  IBC case (binariser already supports both), then wire
  `decode_ibc_cu` into the per-CU path. The §8.6.1 step-4 residual
  decode currently lives in the inter slice walker; sharing it for
  IBC needs a small refactor of the slice-data residual code.

## Round-73 status

## Round-73 deltas vs round 11

- **`ibc` module** (`src/ibc.rs`): clean-room transcription of
  ISO/IEC 23094-1 §8.6 (decoding process for IBC-coded CUs).
  Four pure-function primitives:
  - `derive_ibc_luma_mv(mvd)` — §8.6.2.1 eq. 1025-1039. Folds `MvdL0`
    into a signed-16-bit `mvL` via `(mvp + mvd + 2^16) % 2^16` (mvp=0
    for IBC), then `<< 4` to land on the 1/16-pel grid.
  - `derive_ibc_chroma_mv(mvL, chroma_format_idc)` — §8.6.2.2 eq.
    1040-1041. Handles monochrome / 4:2:0 / 4:2:2 / 4:4:4.
  - `validate_ibc_constraints(mvL, xCb, yCb, nCbW, nCbH,
    ctbLog2SizeY)` — §8.6.2.1 bitstream conformance: above-or-left
    guard, eq. 1035/1036 same-CTU-row, eq. 1037/1038 current-or-left
    CTU column, fractional-pel rejection, negative-coords rejection.
    Returns `Err(Error::Invalid)` on any violation. `sps_suco_flag=1`
    extra rules deferred (suco isn't supported elsewhere either).
  - `predict_ibc_block(...)` — §8.6.3 integer-pel rectangular copy
    from the current picture's already-reconstructed region into
    luma + chroma prediction buffers. eq. 1039's `<< 4` guarantees
    integer-pel landing, so the spec's "invoke §8.5.4.3.1 fractional
    sample interpolation" step collapses to a memcpy (clean-room
    observation).
- 25 new tests cover: luma MV derivation (zero/±/wrap), chroma MV
  derivation (monochrome/4:2:0/4:2:2/4:4:4/sign), constraint
  validation (above-only, left-only, overlapping, cross-CTU-row,
  left-neighbour CTU ok, two-CTUs-left rejected, fractional BV
  rejected, zero dims, bad CtbLog2SizeY, negative ref origin), block
  prediction (above block, left block, luma-only, buffer mismatch,
  fractional MV rejected), and the `derive → validate →
  derive_chroma → predict` pipeline end-to-end on a 32×32 gradient.
- 251 unit tests pass (was 226 in round 11).
- The `sps_ibc_flag = 1` SPS-level gate is **unchanged** — round 73
  only lands the §8.6.2 / §8.6.3 primitives. Lifting the gate
  requires the next round to (a) emit `ibc_flag` from CABAC in
  `coding_unit()` (§7.3.8.4 + §9.3.4.2.4 ctxInc — helper
  `ctx_inc_neighbour_sum` already exists), (b) parse `abs_mvd_l0` +
  `mvd_l0_sign_flag` for the IBC case (also already in the binariser),
  and (c) wire the §8.6.1 5-step pipeline (deriveMV → deriveChromaMV
  → predict → residual → reconstruct).

## Round-11 status

Working **Baseline-profile** decoder for IDR + P + B slices with full
residual coding, luma + chroma deblocking, the 64-point IDCT, the
**Main-profile CABAC initialization tables** (Tables 40-90) and
§9.3.4.2 ctxInc derivation helpers, the round-8 RPL parser + HMVP
infrastructure, the round-9 multi-ref DPB + POC reordering, the
round-10 spec-compliance additions, plus the round-11 post-filter
pipeline:

* **ALF (Adaptive Loop Filter)** (§8.9): APS-type 0 payload is now
  fully parsed into `AlfData` (up to 25 luma filter sets × 13 taps,
  up to 4 chroma alternates × 7 taps). When `sps_alf_flag = 1` and an
  ALF APS has been received, `apply_alf` applies luma filter[0] (7×7
  diamond, 12 symmetric tap pairs + DC offset per eq. 1263) and chroma
  filter[0] (5×5 diamond, 6 symmetric pairs + DC per eq. 1290) as a
  post-deblocking in-loop pass. `sps_alf_flag = 1` streams no longer
  return `Error::Unsupported`.
* **DRA (Dynamic Range Adjustment)** (§8.10): APS-type 1 payload is
  fully parsed into `DraData` (up to 16 piecewise-linear segments with
  Q8.3 scale + chroma QP offset per segment). `build_luma_lut` produces
  a 256-entry mapping table; `apply_dra` maps every Y sample through the
  LUT and offsets Cb/Cr by the segment-0 chroma offset. `sps_dra_flag =
  1` streams no longer return `Error::Unsupported`.
* **Spatial-neighbour MV grid AMVP** (§8.5.2.4): the per-CU `mvpList[]`
  now sources its left / above / above-right slots from the per-4×4
  `SideInfoGrid` instead of always falling back to the spec's
  `(1, 1)` substitution. The strict-match gate of §8.5.2.4.3 — the
  neighbour is only available when its `pred_mode == Inter` AND its
  `ref_idx_l*` matches the current CU's `cur_ref_idx_lx` — is honoured.
  When all three spatial slots resolve to `(1, 1)`, the round-9 HMVP
  fallback (§8.5.2.4.4) still fires. The temporal slot stays at zero
  MV (the §8.5.2.5 collocated path is parked for a follow-up round).
* **LTRP entries in slice RPL → DPB resolution** (§8.3.2 / §8.3.5):
  `RefPicListEntry::Ltrp { poc_lsb_lt }` no longer surfaces as
  `Error::Unsupported`. Instead the entry is matched against
  `(poc & (max_poc_lsb − 1))` for every DPB slot; the matching POC
  becomes the LTRP slot's resolved reference. STRP entries continue
  to advance the running delta-POC chain unchanged. A mixed
  STRP + LTRP RPL works end-to-end.
* **`flush()` drain** (output queue): every DPB entry that hasn't yet
  been pushed to the output queue is now emitted by `Decoder::flush()`,
  in ascending POC order. Pictures already in `out` (the typical case
  for low-delay GOPs where decode order == display order) stay in
  place — flush is idempotent. A new `output_emitted: bool` field on
  the DPB entry tracks the per-picture emission state.

## Round-9 deltas vs round 8

* **Multi-reference DPB** (§8.3) — `EvcDecoder` now carries a
  16-slot decoded-picture buffer indexed by POC. Every freshly-decoded
  picture is held in the DPB; IDR slices flush the buffer (and reset
  POC to 0); non-IDR slices are inserted at the resolved POC.
  Eviction at capacity drops the lowest POC. The
  `InterDecodeInputs::ref_list_l0` / `ref_list_l1` slice surfaces
  expose every active reference at once; `apply_inter_prediction`
  resolves each CU's per-list `ref_idx_l*` against the right slot.
  The round-4 single-reference gate (`Error::Unsupported` when
  `num_ref_idx_active_minus1_l? > 0`) is gone.
* **HMVP-driven AMVP fallback** (§8.5.2.4.4) — the round-8 HMVP
  candidate list now drives MV selection, not just shadows it. When
  the §8.5.2.4.3 spatial AMVP slot would land on the spec's `(1, 1)`
  substitution AND HMVP holds at least one valid candidate, the
  predictor falls through to `hmvp.derive_default_mv(ref_idx, list_x)`.
  The fallback fires on every inter CU after the first one in a
  CTU row and significantly improves correctness on real Main-profile
  content (the round-4 stub always returned `(1, 1)` or zero).
* **POC reordering** (§8.3.1) — `derive_poc()` implements the spec's
  wrap-detection on `slice_pic_order_cnt_lsb` (forward wrap when the
  new LSB falls more than half-`MaxPicOrderCntLsb` below the previous
  LSB; backward in the symmetric case). The decoder's output queue is
  POC-sorted via a parallel `out_pocs` buffer so callers see frames in
  display order even when bitstream coding order differs (e.g.
  B-pyramid GOPs).

Per-slice reference-list construction walks the slice's
`ref_pic_list_struct()` and applies each STRP entry's signed
`delta_poc_st` (per §7.4.8 eq. 124) to the current slice's POC,
chaining entry-to-entry per §8.3.5; the resolved POCs are looked up in
the DPB. Streams with `sps_rpl_flag = 0` keep the round-4 implicit
fallback (highest-POC DPB entry as the single reference).

The remaining Baseline constraints (future rounds will lift them):

- 8-bit luma + chroma (4:2:0).
- `sps_addb_flag = 0` (deblocking uses the §8.8.2 baseline filter; the
  Main-profile advanced deblocking §8.8.3 is parked).
- Sub-pel MV phases restricted to the Baseline 1/4-pel grid for luma
  (Table 25 phases 4, 8, 12) and 1/8-pel grid for chroma (Table 27
  phases 4, 8, 12, 16, 20, 24, 28).
- ALF applies filter set 0 only; per-CTU filter-set selection (the
  `alf_ctb_flag` signalling loop per §8.9.2.4) is deferred.
- DRA applies segment-0 chroma offset uniformly; per-pixel segment
  lookup for chroma is deferred.
- The §8.5.2.5 temporal AMVP (collocated-picture) candidate is still
  zero — round-10 only wires the spatial neighbours.

Anything outside the Baseline toolset (BTT, SUCO, ADMVP, EIPD, IBC, ATS,
ADCC, AMVR, MMVD, affine, DMVR, …) bubbles up as `Error::Unsupported` —
but the CABAC contexts, the RPL parse path (with LTRP resolution),
the HMVP candidate list (production-wired into AMVP), the
spatial-neighbour AMVP grid, the POC-indexed DPB with `flush()` drain,
and the POC-sorted output queue are all ready.

## Round-11 deltas vs round 10

- **`alf` module** (`src/alf.rs`): `parse_alf_data` parses the §7.3.5
  `alf_data()` payload (APS type 0). Luma: up to 25 filter sets, each
  with 12 abs-coded 6-bit symmetric tap coefficients and a derived DC
  offset (eq. 1264: `c[12] = 128 − 2·Σ|c[0..11]|`). Chroma: up to 4
  alternates, each with 6 abs-coded taps + derived DC. `apply_alf_luma`
  / `apply_alf_chroma` clone the source plane, run the convolution with
  boundary-clamped reads, and write results back in-place. `apply_alf`
  is the one-call entry point (luma filter[0] then chroma alt[0]).
- **`dra` module** (`src/dra.rs`): `parse_dra_data` parses the §7.3.6
  `dra_data()` payload (APS type 1). `build_luma_lut` produces a
  256-entry piecewise-linear LUT from up to 16 segments with Q8.3 scale
  values (first scale is 11-bit unsigned; subsequent scales are 12-bit
  signed deltas). `apply_dra` maps every Y sample through the LUT and
  offsets Cb/Cr by the segment-0 `chroma_qp_offset`.
- **APS → post-filter wiring** (`decoder.rs`): `EvcDecoder` caches the
  most-recent parsed `AlfData` and `DraData` from `NalUnitType::Aps`
  NAL units. The new `apply_post_filters` method runs ALF then DRA on
  every decoded `YuvPicture` (IDR and non-IDR) when the SPS gates are
  set and APS data is available. `sps_alf_flag = 1` and `sps_dra_flag =
  1` no longer return `Error::Unsupported` anywhere in the decoder.
- 226 unit tests pass (was 205 before round 11); 21 new tests cover:
  ALF APS parse (identity, explicit coefficients, chroma-only, error
  cases, negative coefficients), ALF filter application (luma + chroma
  with DC), DRA APS parse (not-present, single-range, two-ranges, error
  cases), DRA LUT construction (identity, not-present identity, 2× scale
  clip), and DRA application (noop, identity preserves, chroma offset
  shift, `find_segment` index).

## Round-10 deltas vs round 9

- **Spatial-neighbour MV grid AMVP** (§8.5.2.4): new helper
  `baseline_amvp_select_with_grid_and_hmvp` consults the per-4×4
  `SideInfoGrid` at the spec's left `(xCb − 1, yCb + nCbH − 1)`,
  above `(xCb + nCbW − 1, yCb − 1)` and above-right
  `(xCb + nCbW, yCb − 1)` positions. The strict ref-idx-match gate of
  §8.5.2.4.3 is honoured; mismatched refIdx is treated as unavailable.
  When all spatial slots resolve to `(1, 1)`, the round-9 HMVP
  fallback fires. `decode_inter_coding_unit` now threads CU
  `(x0, y0, n_cb_w, n_cb_h)` through to the AMVP selection so the
  grid can be probed.
- **LTRP entries in slice RPL → DPB resolution**: `EvcDecoder::build_ref_pocs`
  now resolves both STRP (signed delta-POC chain per eq. 124) and
  LTRP (`poc_lsb_lt` matched against `(POC & (max_poc_lsb − 1))`)
  entries in a single walk. Mixed STRP + LTRP RPLs work end-to-end.
  The round-9 `Error::Unsupported` gate on LTRP is gone.
- **`flush()` drain**: `Decoder::flush()` now emits every DPB entry
  that hasn't yet been pushed to the output queue, in ascending POC
  order. Pictures already in `out` (low-delay GOPs) stay in place —
  flush is idempotent and a no-op when every DPB entry is already
  emitted. New `output_emitted: bool` field on `DpbEntry` tracks the
  per-picture state.
- 205 unit tests pass (was 196); 9 new tests cover round-10:
  4 spatial-neighbour AMVP cases (left, above-right, refIdx mismatch,
  HMVP fallback), 3 LTRP RPL cases (resolve, missing DPB entry,
  mixed STRP + LTRP), 1 `drain_dpb_to_output` direct, 1 end-to-end
  flush()-after-receive idempotence.

## Round-9 deltas vs round 8

- **Multi-reference DPB**: `EvcDecoder` carries a 16-slot DPB
  (`Vec<DpbEntry>`) keyed by POC. IDR flushes; non-IDR appends.
  `InterDecodeInputs::ref_list_l0` / `ref_list_l1` are now
  `&[RefPictureView]` slices; `apply_inter_prediction` resolves each
  CU's `ref_idx_l*` against the right slot. The round-4 single-ref
  gate is removed.
- **HMVP-as-AMVP**: `baseline_amvp_select_with_hmvp` consults
  `hmvp.derive_default_mv()` whenever the §8.5.2.4.3 spatial slot
  resolves to the `(1, 1)` substitution and HMVP holds a valid
  candidate. Round-8 only built the HMVP list; round-9 makes it
  drive MV selection.
- **POC + reordering**: `derive_poc()` implements §8.3.1 wrap
  detection. Output queue is POC-sorted via a parallel `out_pocs`
  buffer so display order is preserved even when coding order isn't.
- **Per-slice RPL → DPB resolution**: `decode_non_idr` walks the
  slice's `ref_pic_list_struct()` deltas (chained from `cur_poc` per
  §8.3.5), looks each POC up in the DPB, and packs the results into
  the inter pipeline's `ref_list_l0` / `ref_list_l1`.
- **Implicit fallback**: streams with `sps_rpl_flag = 0` still work —
  the highest-POC DPB entry becomes the single L0 ref. Preserves
  round-4 fixture behaviour.
- 196 unit tests pass (was 187); 9 new tests cover the round-9
  pipeline (HMVP fallback direct + no-op, multi-ref DPB acceptance,
  empty/oversized validation, POC wrap, DPB eviction, DPB flush,
  three-frame IDR + P + P end-to-end).

## Round-8 deltas vs round 7

- **`rpl` module** (§7.3.7 / §7.4.8): `parse_ref_pic_list_struct` walks
  STRP + LTRP entries with full per-entry data (`delta_poc_st`,
  `strp_entry_sign_flag`, `poc_lsb_lt` of
  `log2_max_pic_order_cnt_lsb_minus4 + 4` bits). `RefPicListEntry`
  exposes `signed_delta_poc()` mirroring §7.4.8 eq. 124.
- **SPS RPL surfacing**: `Sps` now carries
  `num_ref_pic_lists_in_sps_l0/l1` plus the per-list
  `Vec<RefPicListStruct>`. The previous "skip + sanity-bound" path is
  replaced with the real parser, with `rpl1_same_as_rpl0_flag = 1`
  inferring list 1 from list 0.
- **Slice-header non-IDR RPL**: the prior `Error::Unsupported` path on
  `sps_rpl_flag = 1` is gone. `slice_header::parse` now consumes
  `ref_pic_list_sps_flag[i]` (with the `i == 1` inference rules from
  §7.4.5), `ref_pic_list_idx[i]` (sized by `Ceil(Log2(n_in_sps))`),
  any inline `ref_pic_list_struct()`, and the optional
  `additional_poc_lsb_present_flag` / `additional_poc_lsb_val` loop
  per LTRP entry. Surfaces `slice_rpls_idx[i]` (eq. 83). New
  `parse_consume` API lets the decoder reuse the parser to recover
  the slice-data byte offset.
- **`hmvp` module** (§8.5.2.7, §8.5.2.4.4): 23-entry LRU candidate
  list with `update()` / `reset()` / `derive_default_mv()`. The list
  is reset at every CTU-row left boundary, updated after every inter
  CU, and ready to be consulted as a fallback when a neighbour-based
  AMVP candidate would be unavailable. `InterDecodeStats` now exposes
  `hmvp_cand_count_final` for fixture verification.
- **Decoder NonIDR rewire**: the inline mini-parser inside
  `decoder.rs` is replaced with a call into the canonical
  `slice_header::parse_consume`, so RPL-bearing P/B slices decode
  through a single tested code path. New end-to-end fixture
  `round8_rpl_non_idr_decodes_to_two_frames` round-trips an
  IDR + P pair through the registered decoder factory with
  `sps_rpl_flag = 1` and inline RPL on both lists; PSNR Y = ∞
  (MSE = 0) on the zero-MV identity copy.
- 187 unit tests pass (was 167 in round 7); the 20 new tests cover
  the RPL parser (STRP-only, mixed STRP+LTRP, zero-delta no-sign-bit,
  oversized-count rejection), HMVP (LRU shift-on-full, ref-idx-match
  walk, fallback to most-recent, last-4 walk bound, L1-field
  selection, reset semantics), the slice-header RPL paths (inline
  + SPS-pointer with `rpl1_idx_present_flag = 0` inference, ceil_log2
  helper), and the round-8 end-to-end decode fixture.

## Round-7 deltas vs round 6

- **Main-profile CABAC init tables**: every initValue from
  ISO/IEC 23094-1 §9.3.5 Tables 40-90 (BTT split, SUCO, cu_skip,
  AMVR, MMVD, affine, ALF, DRA, ATS, ADCC, IBC, mvp_idx, merge_idx,
  ref_idx, abs_mvd, intra_pred_mode + MPM, inter_pred_idc,
  cbf_all/luma/cb/cr, cu_qp_delta_abs, sig_coeff_flag,
  coeff_abs_level_greaterA/B, last_sig_coeff_x/y_prefix, …) lives in
  the new [`cabac_init`](src/cabac_init.rs) module as `pub const`
  arrays keyed by the [`MainCtxTable`] enum (discriminant = spec
  table number).
- **§9.3.2.2 init pipeline** (eq. 1425/1426) wired through
  [`init_main_profile_contexts`](src/cabac_init.rs), which walks every
  Main-profile table and installs the per-(table, ctxIdx)
  `(valState, valMps)` derived from the slice QP. Table 39's
  initType=0 (I) vs initType=1 (P/B) split is encoded in
  [`MainCtxTable::init_type_range`].
- **§9.3.4.2.2-12 ctxInc helpers**: pure-function ports of the
  per-syntax-element ctxInc derivations for `coeff_zero_run` /
  `coeff_abs_level_minus1`, `split_unit_coding_order_flag`, the
  neighbour-block sum used by `affine_flag` / `cu_skip_flag` /
  `pred_mode_flag` / `ibc_flag`, `btt_split_flag` (with Table 97
  `ctxSetIdx`), `last_sig_coeff_{x,y}_prefix`, `sig_coeff_flag`,
  `coeff_abs_level_greaterA/B_flag`, `coeff_abs_level_remaining`
  Rice-parameter (Table 98), `ats_cu_inter_flag`, and
  `ats_cu_inter_horizontal_flag`.

## Round-6 deltas vs round 5

- **Chroma deblocking**: §8.8.2 chroma path per eq. 1167-1213. The
  2-tap stencil (only `sB` and `sC` mutate — eq. 1208/1209/1212/1213)
  runs against both Cb and Cr planes after the luma pass. Table 33 is
  keyed by `qp_c = Clip3(0, 51, slice_qp + slice_cb_qp_offset)` for Cb
  (with `slice_cr_qp_offset` for Cr — eq. 1194). Edge spacing scales
  via Table 2 sub-sampling factors (4:2:0 → every 2 chroma samples).

## Round-5 deltas vs round 4

- **Residual coding**: `residual_coding_rle()` per §7.3.8.7 now drives
  per-position `TransCoeffLevel` reconstruction. Each non-zero CBF
  triggers the dequant + inverse-transform + add-to-predictor pipeline
  (intra and inter both supported).
- **64×64 IDCT**: the spec's eq. 1071-1076 has m/n indexing typos that
  blocked transcription in round 3. Round 5 builds the matrix from
  the closed form `M[m][n] = round(64·√2·cos(π·m·(2n+1)/128))` for m≥1
  (M[0][n] = 64), verified against every printed entry of eq. 1072 /
  eq. 1074.
- **Deblocking**: §8.8.2 luma path with the BS table from Table 33,
  the 4-tap edge filter from eq. 1148-1158, and per-4×4-grid side-info
  tracking (PredMode + CBF + MV + RefIdx) populated as CUs decode.

167 unit tests cover the CABAC engine, NAL / parameter-set parsing, the
`slice_data()` walker, the IDR pixel pipeline, the inter-prediction
primitives, the residual / dequant / IDCT chain, the 64-point IDCT
construction, the deblocking filter math for both luma (4-tap) and
chroma (2-tap) — Table 33 spot checks, BS rules, edge smoothing
fixtures, the `chroma_qp_offset` switch, the end-to-end no-op pass on
a uniform-grey IDR — and the round-7 Main-profile CABAC infrastructure
(Tables 40-90 spot checks, Table 39 init-type ranges, the §9.3.4.2.x
ctxInc helpers, and end-to-end `init_main_profile_contexts` derivation
at QP 22 / 32).
