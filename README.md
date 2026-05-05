# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Baseline + Main profiles. Zero C dependencies, zero FFI,
zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round-10 status

Working **Baseline-profile** decoder for IDR + P + B slices with full
residual coding, luma + chroma deblocking, the 64-point IDCT, the
**Main-profile CABAC initialization tables** (Tables 40-90) and
§9.3.4.2 ctxInc derivation helpers, the round-8 RPL parser + HMVP
infrastructure, the round-9 multi-ref DPB + POC reordering, plus the
round-10 spec-compliance additions:

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
- **ALF** (§8.9 adaptive loop filter) — APS-driven coefficient sets are
  parsed but the filter tap-pass is not yet implemented.
- **DRA** (§8.10 dynamic range adjustment) — APS-driven mapping tables
  are parsed but the range-mapping post-pass is not yet implemented.
- The §8.5.2.5 temporal AMVP (collocated-picture) candidate is still
  zero — round-10 only wires the spatial neighbours.

Anything outside the Baseline toolset (BTT, SUCO, ADMVP, EIPD, IBC, ATS,
ADCC, AMVR, MMVD, affine, DMVR, …) bubbles up as `Error::Unsupported` —
but the CABAC contexts, the RPL parse path (with LTRP resolution),
the HMVP candidate list (production-wired into AMVP), the
spatial-neighbour AMVP grid, the POC-indexed DPB with `flush()` drain,
and the POC-sorted output queue are all ready.

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
