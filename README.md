# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Baseline + Main profiles. Zero C dependencies, zero FFI,
zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round-8 status

Working **Baseline-profile** decoder for IDR + P + B slices with full
residual coding, luma + chroma deblocking, the 64-point IDCT, the
**Main-profile CABAC initialization tables** (Tables 40-90) and
§9.3.4.2 ctxInc derivation helpers, plus the round-8 additions:

* **Reference picture list parsing** (§7.3.7 / §7.4.8) — the new
  [`rpl`](src/rpl.rs) module fully parses every
  `ref_pic_list_struct(listIdx, rplsIdx, ltrpFlag)` entry (STRP +
  LTRP), including the `delta_poc_st` / `strp_entry_sign_flag` /
  `poc_lsb_lt` per-entry shape. The SPS parser populates
  `num_ref_pic_lists_in_sps[i]` and the per-list candidate arrays
  (no longer "skip"-only); the slice-header parser walks the non-IDR
  RPL branch (`sps_rpl_flag == 1`), handles both the SPS-pointer
  (`ref_pic_list_sps_flag = 1`) and inline-RPL paths, derives
  `SliceRplsIdx[i]` per eq. 83, and consumes the per-LTRP
  `additional_poc_lsb_present_flag` / `additional_poc_lsb_val` loop.
  Round-8 unblocks decode of any non-IDR slice that carries an RPL
  (the prior gate was `Error::Unsupported`).
* **History-based MV prediction** (§8.5.2.7 + §8.5.2.4.4) — the new
  [`hmvp`](src/hmvp.rs) module implements the 23-entry HMVP candidate
  list with the spec's left-shift-on-full LRU update, per-CTU-row
  reset (`if (xCtb == xFirstCtb) NumHmvpCand = 0`), and the §8.5.2.4.4
  `derive_default_mv()` walk that prefers an exact-refIdx match in
  the last 4 entries before falling back to the most-recent valid
  candidate. The list is threaded through
  `decode_baseline_inter_slice` and updated after every inter CU; the
  final `NumHmvpCand` is surfaced via `InterDecodeStats`.

The decoder's NonIDR path now routes through
`crate::slice_header::parse_consume`, so any production stream with
`sps_rpl_flag = 1` (with or without `sps_pocs_flag = 1`) decodes
through the canonical parser. The Baseline residual / inter / intra /
deblock pipeline is unchanged.

The remaining Baseline constraints (future rounds will lift them):

- 8-bit luma + chroma (4:2:0).
- `sps_addb_flag = 0` (deblocking uses the §8.8.2 baseline filter; the
  Main-profile advanced deblocking §8.8.3 is parked).
- Single reference picture per list (`num_ref_idx_active_minus1_l? = 0`).
- Sub-pel MV phases restricted to the Baseline 1/4-pel grid for luma
  (Table 25 phases 4, 8, 12) and 1/8-pel grid for chroma (Table 27
  phases 4, 8, 12, 16, 20, 24, 28).

Anything outside the Baseline toolset (BTT, SUCO, ADMVP, EIPD, IBC, ATS,
ADCC, ALF, DRA, AMVR, MMVD, affine, DMVR, …) bubbles up as
`Error::Unsupported` — but the CABAC contexts, the RPL parse path and
the HMVP candidate list are now ready.

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
