# Changelog

## [Unreleased]

### Other

- **Tile-axis corpus extension + mutation gate.** A 2×2 four-tile IDR stitch fixture (the bottom-right tile gets neither left nor top references from its decoded neighbours) joins the 2×1 fixtures, and a decoder-level mutation gate runs every single-byte corruption of the two-tile IDR+P bitstream (low-bit flip + full invert per position) through `make_decoder` — every outcome must be a clean error or a decoded frame, never a panic; this sweeps the tile plumbing's error paths (PPS tile geometry, slice-header tile ids, entry-point subsets, per-tile CABAC terminates).
- **DRA apply at every bit depth — the 8-bit-only gate is lifted.** `build_luma_lut_vec` generalises the §8.9 luma mapping table to one entry per code-space value at the picture's bit depth (`8..=16`), scaling the §7.3.6 10-bit-domain `dra_range_l[]` boundaries by `1 << (BitDepthY − 10)` into the code space (zero shift at <= 10 bits — byte-identical to the historical 256-entry table over the shared domain); `build_ranges_array` (the chroma co-located-segment lookup) scales identically. `apply_dra` routes through the full LUT and the decoder's `pic.bit_depth == 8` skip is removed, closing the last "Not yet supported" README item. Fixtures: 8-bit vec/array equivalence, 10-bit full-coverage + clip, the 12-bit boundary shift (512 → 2048), and a 10-bit end-to-end `apply_dra` (luma 300 → 600, chroma mid + segment offset).
- **Multi-tile decode through the registered decoder — `single_tile_in_pic_flag` lifted.** `resolve_slice_tiles` turns the active PPS's §6.5.1 eq. 24-32 derivations + the parsed slice-header tile fields into the §7.3.8.1 walk order, the luma-domain tile layout, and the §7.3.4 `entry_point_offset_minus1[]` loop (consumed between the header and the byte alignment); §7.4.5 eq. 88/89 byte subsets feed the tiled walkers on **both** slice paths. The IDR path is upgraded from the round-3 minimal header parse to the **full §7.3.4 parse**: tile fields, the ALF slice block (the per-CTU `alf_ctb_*` map + the round-126 APS-id routing now work on IDR pictures), and the deblocking controls (`slice_deblocking_filter_flag` + the §8.8.3 alpha/beta offsets — previously ignored, IDR pictures never deblocked). The post-filter pass receives the resolved layout + the P/B motion grid for the §8.8.4.5/.6 availability. e2e fixture: SPS + 2×1-tile PPS + a both-tiles IDR (16-bit entry point, two CABAC subsets) + a both-tiles skip-P through `make_decoder` — the IDR frame must stitch the two standalone single-tile decodes bit-exactly and the P frame must equal it.
- **§8.8.4.5/.6 per-CTB ALF input-sample derivation.** `derive_alf_input` builds the spec's padded `recPictureOut` for every filtered CTB: interior copy (eq. 1324/1349) plus 3-sample edge strips that either copy the plane (edge available) or **mirror across the block boundary** (eqs. 1325-1348/1350-1373). Edge availability follows the printed selector: `loop_filter_across_tiles_enabled_flag == 0` → §6.4.1 (the tile bullet against the picture layout), `== 1` → §6.4.4 (an intra/IBC-coded neighbour is unavailable — probed from the decoded motion grid; with no grid the picture is all-intra and every CTB edge mirrors, the IDR shape). The §8.8.4.3 classification now runs **over the padded input** (per the §8.8.4.2 invocation order), and new `apply_alf_luma_availability` / `apply_alf_chroma_availability` filter eqs. 1281-1288/1321-1323 over per-CTB pads built from one pre-ALF snapshot (a filtered CTB never feeds a later CTB's taps). The registered decoder's map-driven ALF paths route through these; the historical clamp-based utilities remain for the legacy whole-plane fallback. Two literal-text resolutions are documented in-code: the chroma top strip's stray "`and loop_filter_across_tiles_enabled_flag is equal to FALSE`" conjunct (eq. 1362 — would leave `(availableT == 0, flag == 1)` unassigned; resolved as mirror-when-unavailable like §8.8.4.5 and the other three strips) and §8.8.4.6's `SubWidthC`-for-both-axes probe scaling (resolved with `SubHeightC` on y). Fixtures: strip-exact mirror/copy pins, the §6.4.4 intra-bullet selector, and a two-CTB apply where an armed tile boundary filters each CTB as an island while an all-inter grid leaks neighbour samples.
- **§8.8.2.1/§8.8.3.1 tile-boundary deblocking exemption.** When `loop_filter_across_tiles_enabled_flag == 0`, both deblocking flavours skip "the edges that coincide with tile boundaries": the §8.8.2 luma vertical/horizontal edge loops and the chroma loops (compared in the luma domain — tile boundaries are CTB-aligned), and the §8.8.3 ADDB per-CB edge dispatch (a CB whose x0/y0 lands on an armed boundary skips that direction). Driven by the `TileBounds` set the tiled walkers stage on the side-info grid; `None` (single tile / across-tiles enabled) filters everything as before. Fixtures: a two-CU small-step edge that the open control smooths but the armed boundary leaves bit-exact (luma + chroma + ADDB).
- **§7.3.8.1/§9.3.1 multi-tile P/B pixel walker.** `decode_baseline_inter_slice_tiled` mirrors the IDR restructure on the inter walker: per-tile fresh CABAC engine + §9.3.2.2 initType-1 re-init over the §7.4.5 subset, per-tile `QpY_PREV` restart, and the §7.3.8.2 lines-2622-2625 `NumHmvpCand = 0` reset keyed on the tile's own `xFirstCtb` (the history list never crosses a CTB row *or* a tile — the historical `xCtb == 0` test only matched a picture-origin tile). Inter *sample* prediction deliberately stays unconstrained: §8.5.4 MC reads reference pictures across tile boundaries; only current-picture neighbour derivations carry the §6.4 tile bullet. The untiled entry point becomes a single-segment wrapper. Fixture: a two-tile P slice where tile 1's left-slot (`mvp_idx = 0`) skip CUs must take the §8.5.2.4 `(1, 1)` quarter-pel unavailable-substitution instead of inheriting tile 0's zero MV across the boundary (retained motion field + sub-pel-vs-copy pixels pinned; a single-tile control with identical CU syntax plain-copies).
- **§7.3.8.1/§9.3.1 multi-tile IDR pixel walker.** New `decode_baseline_idr_slice_tiled`: the pixel-emission counterpart of the round-292 structural `walk_baseline_idr_slice_tiled` — the outer loop runs once per `SliceTileIdx[]` segment in tile-scan CTU order, each tile decoding from its own §7.4.5 eq. 88/89 byte subset with a **fresh CABAC engine + §9.3.2.2 context re-init** (§9.3.1 item 2) and a **per-tile `QpY_PREV = slice_qp` restart** (§8.7.1), terminating on its own `end_of_tile_one_bit`; the layout arms `cur_tile` for the §6.4.1/§6.4.3 probes and `TileBounds` for the deblocking exemptions. `decode_baseline_idr_slice` is now a thin wrapper (one segment, whole-RBSP subset, single-tile layout) — behaviour pinned byte-identical by a wrapper-equivalence fixture. Fixtures: a two-tile 64×32 IDR whose tiled decode must **stitch the two standalone 32×32 single-tile decodes bit-exactly** (each tile decodes as an independent picture — the tile-1 left column may not leak tile 0's samples), plus malformed-plumbing rejections (subset/segment count mismatch, empty order, out-of-RBSP subset, out-of-grid CTB address).
- **§6.4.1/§6.4.3 tile bullet threaded through every decode-time neighbour probe.** New `tiles` module: `TileRect` (a tile's half-open luma rectangle — the "different tile" bullet reduces to a containment test since EVC tiles are rectangular), `PicTileLayout` (the §6.5.1 eq. 26/27 boundaries scaled to luma samples + the §7.4.4.2 loop-filter flag) and `TileBounds` (the §8.8.2/§8.8.3 tile-boundary edge-exemption set). `SideInfoGrid` carries `cur_tile` (armed per tile segment by the tiled walkers) + `tile_bounds`, and every §6.4.1/§6.4.3-shaped probe now folds the tile test: the §9.3.4.2.4/.5 ctxInc neighbour sums (`ctx_inc_neighbour_cells`, `btt_num_smaller`), the §8.4.2 EIPD neighbour modes + SUCO right-column probe, the Baseline AMVP / §8.5.2.3 merge / §8.5.3.3 affine grid lookups, the Baseline + EIPD intra reference fetches (`fetch_intra_refs_in_tile` / `fetch_eipd_refs_in_tile` + the `_in_tile` reconstruct bridges, §8.4.4.2 substitution for cross-tile holes), the §8.7.6.2 HTDF border availability (`apply_htdf_luma_in_tile`), and the §8.6.1 IBC reference-corner conformance (`validate_ibc_constraints_in_tile` — a BV crossing a tile boundary is non-conformant). Also fixes the §7.4.4.2 inference: an absent `loop_filter_across_tiles_enabled_flag` is inferred **1** (was 0). `None`/unset everywhere reproduces the single-tile behaviour byte-identically; the tiled walkers land next.

- **Internal plumbing marked `#[doc(hidden)]`.** The syntax/recon modules (parsers, CABAC, prediction, transform, filters) plus the round-2/3 slice harnesses and `decoder::EvcDecoder` are hidden from the documented API — they stay `pub` for tests, but the stable surface is `probe` + `EvcFileInfo` + `CODEC_ID_STR`, `register`, and `decoder::make_decoder`. Attributes and comments only; no semantic or signature changes.
- **U binarization bounded by Table 91 `cMax`, not a fixed 64 bins.** The CABAC U reader's historical 64-bin CPU cap mis-rejected conformant streams once eq. 51 enabled 64×64 TBs: `coeff_zero_run` is legal up to `(1 << (log2W + log2H)) − 1 = 4095` and `coeff_abs_level_minus1` up to the ±32768 TransCoeffLevel window. New `decode_u_regular_capped(c_max, …)` (§9.3.3.2 always terminates, so a legal bin string is ≤ `c_max` ones + a 0; past-`c_max` is an invalid-stream error) now drives both RLE readers on both walkers; the small-range elements keep the 64-bin wrapper. Fixtures: a cabac-level cMax round-trip/reject pair, and a 64×64-TB decode with a 100-position zero run + level 200 that the old cap refused.
- **Errata #238 (a) — TB-split completed on the P/B intra path.** `decode_inter_intra_cu` (a `pred_mode_flag == 1` CU inside a P/B slice) now loops the single-tree `transform_unit()` body over the same §7.3.8.4 tiling (per-TU cbf_luma/cbf_cb/cbf_cr, eq. 1042 cu_qp chain, ATS-intra group; CB-sized residual assembly, whole-CB prediction/reconstruction + the §8.4.1 HTDF step at the final chain QP). Every walker path the erratum touches is now tiled; the P/B IBC branch keeps its explicit `CB ≤ MaxTb` refusal (IBC CBs are bounded by `log2MaxIbcCandSize`). Fixture: a 64×64 intra CU in a P slice at `MaxTb = 32`, bottom-right-only residual, quadrant-exact.
- **HTDF wired into decode (§8.7.6 to pixels).** The Hadamard Transform Domain Filter — previously a complete pure module with a `picture::apply_htdf_luma` bridge that nothing invoked (streams with `sps_htdf_flag = 1` decoded with unfiltered luma) — now runs at its two spec invocation points: after every **intra** luma CB reconstruction (§8.4.1 line 6812, NOT gated on cbf_luma; the §8.7.6.1 applicability gates + Qp′Y eq. 1043 do the rest — IDR path and the P/B `decode_inter_intra_cu` path both), and after **inter** CU reconstruction when `cbf_luma == 1` (§8.5 CU-reconstruction step 7, incl. the DMVR arm; eq. 1106 square-≥32 predicate threaded). §8.6 defines no HTDF step, so IBC CUs are excluded. `sps_htdf_flag` threads through `SliceDecodeInputs` from all decoder entry points; new `htdf_blocks` counters in both stats structs. 3 fixtures: an intra decode whose filtered output equals the direct bridge applied to the unfiltered decode (and differs from it), the intra cbf-0 case still invoking the filter (identity on the flat field, counter = 1), and the inter cbf gate (fires on cbf 1 + bridge-exact, suppressed on cbf 0).
- **TB-split coverage at spec-exact geometry + per-TU QP chain.** Two more fixtures: a **128×128 CTB** (`CtbLog2SizeY = 7`, the §7.4.3.1 maximum) tiling an unsplit intra CB into four **64×64** TBs at the eq. 51 MaxTb — the only four-call geometry a conformant stream can produce, bottom-right lift confined to (64, 64)..(127, 127) — and a per-TU `cu_qp_delta` chain proof: under the Baseline cbf-gated presence form each cbf-carrying TU reads its own delta into the §8.7.1 eq. 1042 chain (delta 0 then +21), so identical levels in the top-right and bottom-right TUs reconstruct at QP 30 vs 51 and must differ (quiet TUs read no delta; exactly 2 `cu_qp_delta_abs` reads).
- **Errata #238 (a) — §7.3.8.4 transform-unit tiling on the inter (P/B) path.** `decode_inter_cu_residual_and_reconstruct_motion` now loops the whole per-TU body (cbf_luma/cbf_cb/cbf_cr group, cu_qp_delta presence + eq. 1042 chain, ATS-inter group — never present on a TB-split CB since an allowed orientation needs `log2Cb ≤ MaxTb` — and per-component `residual_coding()`) over the same `transform_unit_offsets` tiling, scattering each TB residual into CB-sized §8.5.6.2/.3 resSamples buffers consumed by the whole-CB motion compensation; the side-info stamp records the aggregate cbf. Fixture: a 64×64 zero-MV merge CU at `MaxTb = 32` with distinct luma residuals in the top-right and bottom-right TUs — quadrant-exact placement over the MC copy, 4 cbf_luma + 8 chroma bins.
- **Errata #238 (a) — §7.3.8.4 transform-unit tiling implemented on the intra walkers.** A CB whose width/height exceeds `MaxTbSizeY` now decodes the spec's up-to-four `transform_unit()` invocations (each with its own cbf group, cu_qp_delta presence check against the eq. 1042 chain, ATS group and per-component `residual_coding()`), with the bottom-right call placed at the erratum-resolved `(x0 + 2^log2TbWidth, y0 + 2^log2TbHeight)` — the printed `logTbWidth` (spec line 3046) is a dropped-`2` typo assigned nowhere. Per §8.4.1/§8.4.5 the per-TU residuals scatter into a CB-sized `resSamples` buffer (eq. 386, new `transform_unit_offsets` + `scatter_tb_residual` helpers) and intra prediction + §8.7.5 picture construction run once over the whole CB. Both the stats-only Baseline bin walker and the pixel walker (luma and chroma dual trees) are covered; previously a TB-split CB decoded only the top-left TU and desynced. 3 fixtures: the offset-enumeration unit (all five split shapes), a 64×64 CB at `MaxTb = 32` with distinct residuals in the top-right and bottom-right TUs (quadrant-exact placement + per-TU independence), and a chroma-tree split with a bottom-right-only Cb residual (16×16 chroma TB at the `>> 1` offset).
- **eq. 51 — `MaxTbLog2SizeY` is the constant 6, not a CTU-derived cap.** All three decoder entry points (the registered-decoder non-IDR path, `decode_idr_slice`, and the stats-only `walk_idr_slice`) historically derived `max_tb_log2_size_y = min(CtbLog2SizeY, 5 or 6)`; §7.3.2.2 eq. 51 fixes it at 6 unconditionally, so an unsplit 64×64 CB is a **single** 64×64 TB (the old `min(·, 5)` cap treated it as a TB-split CB and reconstructed only the top-left 32×32 quadrant). New end-to-end regression `eq51_max_tb_is_constant_6_single_64x64_tb`: a Baseline SPS with `sps_btt_flag = 0` infers a 64×64 CTU; a lone luma coefficient on the unsplit CB must move samples beyond the 33rd row/column of the CTU.
- **Errata #238 (b)/(c) reconciliation** — the two settled Clause-7 identifier typos are anchored against the freshly filed in-repo errata entries: eq. 148's bare `cu_qp_delta_sign` resolves to `cu_qp_delta_sign_flag` (`CuQpDelta = abs * (1 − 2 * flag)`, inferred 0 when absent — the walkers already implemented this; now cited + pinned by `errata_238b_eq148_sign_flag_multiplier`), and the eq. 74 `startQP` seed reads resolve to the line-4003 `startQp` assignment (`16` / `−QpBdOffsetC` — already implemented; now cited + pinned by the negative-seed `errata_238c_start_qp_seed_negative_qp_bd_offset` at QpBdOffsetC = 12, which separates the resolved reading from a zero-seed misread).
- **ATS coverage deepening** — two more end-to-end fixtures: a 16×16 P-slice ATS-inter **quad** sub-block transform (all four allow flags true → `ats_cu_inter_quad_flag` + `ats_cu_inter_horizontal_flag` present, quad/horizontal/pos = 1 → a 4×16 width-split sub-block, exact tallies + clean reconstruct), and an ATS-intra decode on a MODE_INTRA CU **inside a P slice** (`pred_mode_flag == 1` → `decode_inter_intra_cu`, initType 1 context, `ats_hor_mode`/`ats_ver_mode` = 0/1 → trTypes (1, 2)).
- **Errata #213(b) conformance guard** — a `residual_coding_adv()` fixture that pins `last_sig_coeff_y_suffix` to `last_sig_coeff_y_prefix` (not `last_sig_coeff_x_prefix`, the Table-91 `cMax` copy-paste typo): a 16×16 block with `last_x = 12` (x_prefix 7 → 2-bin x_suffix) and `last_y = 5` (y_prefix 4 → 1-bin y_suffix) makes the two suffix widths differ, so a typo-following decoder would over-read one bypass bin and desync the sig walk. The clean decode to `(12, 5)` + terminating flush + exact 3-suffix-bin tally confirms the crate sizes the y-suffix from `y_prefix` per §7.4.
- **ATS on the P/B path — the last gated Main-profile tool is lifted.** Intra CUs inside a P/B slice (`decode_inter_intra_cu`) now read the §7.3.8.5 `ats_cu_intra_flag` group (initType 1) exactly like the IDR path, and inter (MODE_INTER) CUs read the **ATS-inter (sub-block transform)** group in the residual reconstruct: `read_ats_inter` (gated on `sps_ats_flag`, some cbf, and `AllowAtsInter::any()` with MinTbLog2SizeY = 2 / eq. 52 and the walker's MaxTb) resolves the §7.3.8.5 flags + the lines-3103-3130 sub-block geometry (`TrafoX0/Y0`, reduced `TrafoLog2Width/Height`); the luma residual decodes at the sub-block size, transforms through the §8.7.4.1 Table-31 `(trTypeHor, trTypeVer)` kernel pair (new `AtsInter::tr_types`), and scatters into the zero-filled full-CB residual buffer at the sub-block offset (`scatter_subblock`); the 4:2:0 chroma residual mirrors the sub-block (`TrafoLog2 − 1` at `Trafo{X,Y}0 >> 1`, always trType 0). **`sps_ats_flag` is lifted from the `decode_non_idr` gate** — with the IDR lift below, ATS is no longer gated anywhere and a full Main-profile stream decodes end-to-end. 4 fixtures: an 8×8 P-slice ATS-inter sub-block transform (flag + horizontal + pos, no quad on 8×8, exact tallies + clean reconstruct), a `scatter_subblock` offset-placement unit, a Table-31 `tr_types` unit, and the shared dequant kernel-divergence proof.
- **ATS-intra (`sps_ats_flag == 1`) wired to pixels on the IDR path** — the §7.3.8.5 `ats_cu_intra_flag` group (spec lines 3080-3087) is now read by `decode_transform_unit` after the cu_qp block and before `residual_coding()`, on any intra (MODE_INTRA) luma tree whose CU satisfies the presence predicate `sps_ats_flag && log2CbW ≤ 5 && log2CbH ≤ 5 && cbf_luma` (IBC CUs and the standalone chroma tree are excluded per `CuPredMode == MODE_INTRA`). When `ats_cu_intra_flag == 1` the Table-30 `(trTypeHor, trTypeVer) = (1 + ats_hor_mode, 1 + ats_ver_mode)` decision drives the luma inverse transform through the §8.7.4.2 DST-VII / DCT-VIII kernels via the new `dequant::scale_and_inverse_transform_ats` (scaling eq. 1059 → the trType-selected transform eq. 1062 → final renorm eq. 1055); `(0, 0)` is byte-for-byte the DCT-II path. `sps_ats_flag` is carried to the walker through `CodingTreeGates` (`from_sps`) and **lifted from the `decode_idr_slice` gate** — an IDR slice has no inter CUs, so ATS-intra fully covers it. 3 fixtures: an end-to-end ATS-intra IDR decode (`ats_hor_mode = 1` / `ats_ver_mode = 0` → DCT-VIII/DST-VII, exact bin tallies + clean CABAC termination), an `ats_cu_intra_flag == 0` regression that stays byte-identical to the DCT-II Baseline reconstruction, and a dequant-level proof that the ATS kernel pair diverges from DCT-II while `(0, 0)` matches it.
- **ADDB (`sps_addb_flag == 1`) — the §8.8.3 advanced deblocking filter** on both walkers' post-reconstruction pass: the §8.8.3.4 boundary-strength cascade (intra across a CTU boundary → 4; intra/IBC → 3; coded residual → 2; the eqs. 1218-1232 reference/motion comparisons at 1-pel granularity → 1/0) over the side-info grid, the §8.8.3.5 per-edge thresholds (`qPav` from the **per-CU eq. 1042 QpY** — every CU stamp now records its `qp_y` — with the eq. 86/87 `slice_alpha_offset`/`slice_beta_offset` already parsed by the slice header, Table 34 α′/β′, eq. 1238 `filterSamplesFlag`), the §8.8.3.6 weak filter (Table 35 t′C0, the eqs. 1242-1245 tC0 forms, the eqs. 1246-1252 Δ/quarter-step updates) and the §8.8.3.7 strong filter (the eqs. 1261/1268 gates, eqs. 1262-1274 3-tap smoothing), driven per coding block (recovered from the grid's covering-CU geometry) with the §8.8.3.2 `splitTH` split, vertical edges before horizontal, 8×8-luma-grid edges only, luma + 4:2:0 chroma-style (`chromaStyleFilteringFlag == 1`, 2-sample groups, `p0/q0`-only writes). `sps_addb_flag` lifted from both decoder gates; the deblock dispatch selects §8.8.3 vs §8.8.2 per SPS. 5 fixtures: the bS cascade, Table 34/35 spot values + threshold math, hand-traced weak + strong filters, and an IDR e2e where ADDB shrinks a residual CU's edge step while leaving CU cores untouched.
- **ADCC (`sps_adcc_flag == 1`) — the §7.3.8.8 `residual_coding_adv()` layer** (`adcc` module) replacing run-length coding at the shared §7.3.8.6 `residual_coding()` dispatch (`decode_residual_block`, both walkers, all TU paths incl. the IBC branches). The full syntax: `last_sig_coeff_{x,y}_prefix` (TR `cMax = (log2TrafoSize << 1) − 1`, Tables 87/88 at the §9.3.4.2.6 eq. 1441 ctxOffset/ctxShift ctxInc) + FL bypass suffixes composing eqs. 149-152; the reverse 16-coefficient-group walk over the §6.5.2 zig-zag with `sig_coeff_flag` (Table 89, the §9.3.4.2.7 eqs. 1442-1451 five-neighbour stencil over the partially-decoded levels), `coeff_abs_level_greaterA/B_flag` (shared Table 90 space, §9.3.4.2.8/.9 stencils, ctxInc 0 at the last position), the escape `coeff_abs_level_remaining` (§9.3.3.8 all-bypass: TR prefix over `numBinRem << cRiceParam` (Table 94) + `k = cRiceParam + 1` EGk suffix, Rice parameter per §9.3.4.2.10 eqs. 1466-1471 + Table 98, the §7.3.8.8 `baseLevel`/`countFirstBCoef` schedule), and the MSB-first `coeff_signs_group` (one bypass bin per non-zero coefficient — the coherent reading of the §7.3.8.8 `<< (32 − numNZ)` walk; the printed FL `cMax = (1 << 16) − 1` input would discard leading signs for `numNZ < 16`). Works under both `sps_cm_init_flag` states (Baseline collapse vs Table 39 offsets). `sps_adcc_flag` lifted from both decoder gates. 6 fixtures: engine-level DC / cm-init escape-path / prefix-suffix composition units, plus IDR luma (cm-init), IDR chroma escape-path and P/B inter e2e decodes with exact per-element tallies.
- **EIPD (`sps_eipd_flag == 1`) wired into the coding-unit leaves** — both walkers select the §7.3.8.4 intra syntax on the SPS gate: luma trees read the MPM/PIMS/rem-mode group (`eipd_syntax::resolve_eipd_luma_mode`) resolved through the §8.4.2 three-list derivation over real grid neighbours (`(xCb − 1, yCb)`, `(xCb, yCb − 1)`, `(xCb + nCbW, yCb)`; valid iff decoded and MODE_INTRA, per the new per-cell `intra_luma_mode` stamp), chroma-carrying trees read `intra_chroma_pred_mode` resolved per §8.4.3 against the co-located luma mode; reconstruction routes through `intra_reconstruct_cb_eipd` (the §8.4.4.1/.2 reference construction + §8.4.4.3-.10 kernels), with the SUCO right-column availability probed from the decoded-cell grid. The per-CU mode is now `CuIntraMode::{Baseline, Eipd}` through the transform-unit path; P/B MODE_INTRA CUs (SINGLE_TREE and the local-dual-tree chroma) take the same fork. `sps_eipd_flag` lifted from both decoder gates. Fixtures: an IDR MPM-group CU with a DC residual through the EIPD kernel (+ DM chroma), a two-CTU derivation chain where the second CTU selects INTRA_VER from the §8.4.2 lists against its stamped DC neighbour and column-exactly copies the first CTU's bottom row, and a P-slice `pred_mode_flag == 1` EIPD group read.
- **DQUANT (`sps_dquant_flag == 1`) through both pixel walkers** — the §7.3.8.3 dquant block (spec lines 2660-2677) threads `cuQpDeltaCode` through the `split_unit()` recursion: an unsplit block with `log2W + log2H ≥ cuQpDeltaArea` (eq. 76) marks its subtree code 1 (or 2 above MaxTb), a split crossing the area boundary (`== area`, or `== area + 1` with a ternary split) marks code 2, each mark resetting the `isCuQpDeltaCoded` latch. The §7.3.8.5 presence gate takes its full two-arm form — `((!sps_dquant_flag || (code == 1 && !coded)) && cbf) || (code == 2 && !coded)` — so a code-2 area reads `cu_qp_delta_abs` on its first TU even with all CBFs zero, and later TUs in the area are latch-suppressed. The decoded delta folds into the §8.7.1 eq. 1042 chain (`QpY = (QpY_PREV + CuQpDelta + 52) % 52`, `QpY_PREV` threaded per slice via the new `QpState` — replacing the historical `slice_qp + delta` [0, 51] clamp with the spec's modular wrap) on every TU path (intra, inter, both IBC branches, and — new — MODE_INTRA CUs on P/B slices, which previously never read the element). `sps_dquant_flag` lifted from both decoder gates. 3 fixtures: code-1 per-leaf areas with the chroma-CU latch, a code-2 CTU reading a cbf-less delta once, and the P/B code-2 latch over admvp skip CUs.
- **`sps_cm_init_flag == 1` on the P/B walker** — the inter pixel walker initialises the initType-1 (second-half Table 39 ranges) contexts and routes its reads through the P/B `ctxIdxOffset`: `cu_skip_flag` (Table 47 at the §9.3.4.2.4 eq. 1438 neighbour sum over the new per-cell `cu_skip` mark), `pred_mode_flag` (Table 61, numCtx 3 neighbour sum), `ibc_flag` (Table 66 neighbour sum, both the unconstrained and INTRA_IBC-constrained paths), the Baseline `mvp_idx` / `inter_pred_idc` pairs (Tables 48/69 per-bin ctxInc), `ref_idx` (Table 72 — first two prefix bins regular, later bins **bypass** per Table 95, via `decode_ref_idx_tr`), and `abs_mvd` (EG0 with a **regular bin0** on Table 73 via the new `CabacEngine::decode_eg0_first_regular`; the Baseline all-bypass read is preserved). Every Main-profile syntax module (`eipd_syntax::EipdCtx` — now carrying the slice `initType` via `for_slice` — plus `ats`, `amvr_syntax`, `affine_syntax`, `mmvd_syntax`, `inter_cu_syntax`) adds the §9.3.4.2.1 offset to its per-bin ctxInc, so the admvp merge/MMVD/affine/explicit-AMVP readers land on the initialised P/B contexts. `sps_cm_init_flag` lifted from the `decode_non_idr` gate — a full Main-CABAC P/B stream now decodes through the public decoder. e2e fixture: a cm-init P-slice admvp skip CU decodes on the second-half contexts (split 41/1, skip 47/2, merge_idx 49/5, cbf 75-77/1) and zero-MV-copies the reference; a unit test pins the Table 39 offsets (merge_idx 5, cu_skip 2, amvr 4, btt 15, coeff_zero_run 24).
- **§9.3.2.2 + §9.3.4.2.1 `sps_cm_init_flag == 1` context machinery through the pixel walkers** — the IDR and P/B slice walkers initialise every Main-profile context table (Tables 40-90) from the printed initValues at the slice QP with the §9.3.2.2 `initType` (0 for I, 1 for P/B) and route every regular bin through the new `cabac_init::CtxSel` selector: `ctxIdx = ctxIdxOffset(initType) + ctxInc` on the element's own table (§9.3.4.2.1), with the Baseline `sps_cm_init_flag == 0` collapse preserved byte-identically. Wired elements: `split_cu_flag` (Table 41), the BTT group (Tables 42-44 — `decode_btt_split` now takes the selector and the real §9.3.4.2.5 eq. 1439 `numSmaller`, probed from the side-info grid via the new `btt_num_smaller`; every luma CU stamp now records its covering-CB geometry to feed it), SUCO (Table 45), `pred_mode_constraint_type_flag` (Table 46), `intra_pred_mode` (Table 62, bin0/rest ctxInc split), `ibc_flag` (Table 66 at the §9.3.4.2.4 eq. 1438 neighbour sum via `ctx_inc_neighbour_cells`), `cbf_luma`/`cbf_cb`/`cbf_cr` (Tables 75-77), `cu_qp_delta_abs` (Table 78), and the §7.3.8.7 run-length residual (Tables 84/85 at the §9.3.4.2.2 eq. 1434/1435 ctxInc driven by the now-tracked `PrevLevel` chain, `coeff_last_flag` Table 86 at `cIdx == 0 ? 0 : 1`). `sps_cm_init_flag` is lifted from the `decode_idr_slice` gate; the in-test `CabacEncoder` gains the mirror `init_main_profile` so fixtures encode from identical context state. e2e fixture: a cm-init BTT IDR (BT_HOR → 2× BT_VER → four 16×16 leaves, luma + Cb DC residuals on the last leaf, a real `numSmaller = 1` btt_split_flag read) decodes to pixels with exact presence-gating tallies.
- **§7.3.8.3 BTT + SUCO CABAC coding-tree walk to pixels** — both the IDR and P/B slice walkers decode the full Main-profile `split_unit()` decision syntax: the `btt_split_flag`/`dir`/`type` group (Tables 42-44, picture-boundary implicit splits), `split_unit_coding_order_flag` (Table 45, §7.4.9.3 `allowSplitUnitCodingOrder`, mirrored child orders proven by pixel placement), and the recursion geometry (`quad_split_children` / `split_unit_children` with ctDepth/splitUnitOrder threading). `sps_btt_flag` / `sps_suco_flag` lifted from both decoder gates.
- **§7.4.9.3 mode-constraint machinery on P/B slices** — `pred_mode_constraint_type_flag` (Table 46) decoded when `needSignalPredModeConstraintTypeFlag` holds (64-sample BT / 128-sample TT splits, admvp, 4:2:0), resolving per eq. 126; INTER-constrained subtrees suppress `pred_mode_flag` and feed the §7.4.8.3 `allowSplit*` INTER carve-outs; INTRA_IBC-constrained subtrees decode as a local dual tree (luma-only intra/IBC CUs + one `DUAL_TREE_CHROMA` CU at `isTreeSplitPoint`); the §7.3.8.3 leaf overrides (I slice, admvp 4×4) share the same derivation.
- **isIbcAllowed presence fix (§7.4.9.4)** — on the unconstrained P/B path the `ibc_flag` bin is read only when `pred_mode_flag == 1`; round 95 read it for MODE_INTER CUs too, which would desync any conforming stream with `sps_ibc_flag = 1` and a small non-skip inter CU.
- **10-bit decode through the whole reconstruction chain** — `YuvPicture` stores u16 samples (BitDepth 8..=16, §7.4.3.1); the 8-bit-only gates are lifted from both walkers (every dequant/transform/intra/interpolation/deblock/HTDF/ALF stage was already bit-depth-parameterized); `picture_to_video_frame` emits 8-bit planes byte-per-sample and >8-bit planes as little-endian 16-bit (Yuv420P10Le-family layout). The §8.9 DRA apply remains 8-bit-code-space and is gated off for >8-bit pictures.

- **§8.5.1 DMVR to pixels** — the r378 §8.5.5 search core is now invoked from the CU path: the §8.5.5.2 Table-29 bilinear interpolation (`bilinear_pred_plane`, eqs. 989/990 + 999-1017), the eqs.-387-390 subblock partition, the `dmvrAppliedFlag` modification cascade (regular merge/skip bi-pred with `mmvd_flag == 0`, opposite-side equidistant references, ≥8×8, `sps_dmvr_flag` threaded through `InterToolGates` from the SPS), and `apply_dmvr_inter_prediction` (per-subblock refinement → eqs.-395-399 refined `refMvLX` → padded luma + §8.5.2.6 chroma MC). Per the §8.5.1 NOTE the per-subblock deltas are stored on the grid (`ref_mv_delta_*`, default 0 = eq. 400) and the collocated readers reconstruct `refMvLX = mv + delta` while spatial-MVP/deblocking keep the unrefined `mv`. e2e: a ±1-column clamped-ramp fixture recovers `dMvL0 = (16, 0)` on all four 16×16 subblocks of a 32×32 CU and reconstructs the un-shifted content bit-exactly; same-side-POC and sps-flag-off negatives fall back to the plain average.
- **eqs. 923/924 + 932/933 subblock reference-padding clamp** — `interpolate_luma_block_main_padded` / `interpolate_chroma_block_main_padded` clip every reference fetch into the `PadAnchor`-anchored window (±3 luma / ±1 chroma around the *unrefined*-MV subblock, the §8.5.4.3.1 `mvOffset` inversion) after the eqs.-921/922 picture clip.
- **§8.5.3.2/§8.5.3.5 inherited affine candidates from a per-CU CPMV store** — `CuSideInfo` grows `MotionModelIdc` + covering-CU geometry (written by the affine subblock stamp); `affine_neighbour_from_grid` resolves the §8.5.3.3 `NeighbourAffineSource` from the stored corner cells; the affine-merge path fills the five step-3 inherited slots (LR_10 order) with the step-4 same-covering-CU pruning, and the affine-MVP path resolves the refIdx-matched A/B/C groups — inherited candidates now precede the constructed/zero fills.
- **Whole-picture P/B integration fixtures** over the in-crate CABAC encoder: a two-CU slice where a skip affine-merge CU inherits the explicit-affine CU's model through A1 (stamped field cross-checked against an independent §8.5.3.3 projection), and a two-picture chain where picture 1's DMVR deltas feed picture 2's §8.5.2.3.3 temporal merge candidate with the refined `refMvLX`.

- §8.5.2.4 **`sps_admvp_flag == 1` luma MVP derivation** (eqs. 619-646) replacing the r381 grid-slot-0 approximation on the explicit-AMVP path. `admvp_explicit_mvp` consults the single `amvr_idx`-selected neighbour (0→A1, 1→B1, 2→B0, 3→A0, 4→B2 at the §8.5.2.4.1 `availLR`-shaped positions), POC-rescales its MV when the neighbour references a different picture (`distScaleFactor = (targetPocDiff << 5) / currPocDiff`, eqs. 619-638), falls back to the §8.5.2.4.5.2 `DefaultRefIdx` + §8.5.2.4.2 `DefaultMv` cascade (A1 refIdx-matched → B1 refIdx-matched → A1 any-valid → B1 any-valid → the §8.5.2.4.4 HMVP walk) with the eqs.-639-642 rescale, and applies the eqs.-645/646 AMVR-grid predictor rounding (previously never applied). **Spec-conformance fix**: `merge_mode_flag` is inferred **0** when absent (`amvr_idx != 0`) per its §7.4 semantics — r381 inferred 1; an AMVR-shifted CU is therefore an explicit-MVD CU, which is exactly why §8.5.2.4 indexes its MVP neighbour by `amvr_idx` (the wrong-premise test was rewritten). 3 new tests (amvr-indexed neighbour selection, POC rescale halving, default-cascade fallback).
- **`sps_admvp_flag` lifted from the decoder's unsupported gates** — the deepest r384 integration step: a Main-toolset (admvp + affine/amvr/mmvd/hmvp) stream now decodes end-to-end through the public decoder. `decode_non_idr` threads `InterToolGates` from the SPS (`sps_admvp/amvr/mmvd/affine_flag`) + the slice header's `mmvd_group_enable_flag` into every P/B slice, so each CU routes through the §7.3.8.4 Main-profile drivers (cu_skip merge tree, merge-mode, MMVD, affine merge, explicit-AMVP, explicit-affine) with the TMVP/POC context already wired. `decode_idr_slice` drops `sps_admvp_flag` from its gate too (the ADMVP toolset does not alter intra-slice syntax). New decoder-level e2e: an `sps_admvp_flag = 1` SPS (affine + hmvp nested flags) + grey IDR + P slice whose cu_skip merge CU (with the sps_affine-gated `affine_flag` bin) zero-MV-copies the IDR through `make_decoder`/`send_packet`/`receive_frame`. Baseline streams are byte-identical (all-false gates reproduce the historical inline path).
- §7.3.8.4 **explicit-affine sub-tree** (spec lines 2941-2980) decoded + reconstructed — the last missing branch of the ADMVP inter coding-unit tree. `read_explicit_amvp` gains the spec-line-2941 gate (`sps_affine_flag && log2CbWidth ≥ 4 && log2CbHeight ≥ 4 && amvr_idx == 0` → `affine_flag`) and, on `affine_flag == 1`, reads `affine_mode_flag` (`vertexNum = 2 + flag`) then per active list `ref_idx_lX` → `affine_mvp_flag_lX` → `affine_mvd_flag_lX` → per-vertex `abs_mvd`/sign pairs, returning an `ExplicitAffineDecision` (`bi_pred_idx` never read on this branch). Reconstruction (`admvp_affine_amvp_motion`): per list, the §8.5.3.6 refIdx-matched corners are resolved from the grid (corner 0 B2→B3→A2, corner 1 B0→B1→C2, corner 2 A1→A0, corner 3 C1→C0), `constructed_mvp_candidate` + the per-corner translational fill + zero tail assemble the two-entry `cpMvpListLX` via `build_affine_mvp_cand_list` (inherited model-based predictors stay unavailable — no per-CU CPMV store), the eq.-867 `affine_mvp_flag` select + §8.5.3.1 per-CP MVD reconstruction produce the CPMVs, and the §8.5.3.7 subblock field feeds the same per-subblock MC as affine merge. 5 new tests (P-slice affine group bin walk, mvd-flag-0 suppression budget, all four gate corners, zero-MVP e2e reference-copy through `decode_baseline_inter_slice`, constructed-predictor + MVD field gradient).
- §8.5.3 **affine-merge CUs now reconstruct through the CPMV sub-block motion field** — replacing the r381 translational fallback. `admvp_affine_merge_motion` assembles the §8.5.3.2 `affineMergeCandList` per CU: corners resolved from the per-4×4 grid via `resolve_affine_corners` (with the corner-3 collocated fallback fed by the threaded `ColPic` and an exact per-cell eq.-502 POC context), the §8.5.3.4 constructed candidates Const1..6, and the step-9 zero-CPMV tail (inherited model-based neighbours stay unavailable — the grid carries no per-CU CPMV store; documented deferral). The selected candidate's per-list CPMVs drive `affine_subblock_mvs` (§8.5.3.7) into a dense 1/16-pel field, and the new `apply_affine_inter_prediction` motion-compensates **per subblock** with the full-phase Main-profile filters, combining lists per eq. 988 and adding the residual. The reconstruction tail is now motion-generic (`CuMotion::Translational` / `CuMotion::Affine`): affine CUs stamp each subblock's own field vector (1/16→1/4-pel §8.5.3.10 rounding) into the side-info grid and update HMVP with the §8.5.2.7 `affine_center_mv`. Tables 24/26 (the `sps_admvp_flag == 1` full 1/16-/1/32-pel interpolation coefficients) are transcribed alongside `interpolate_luma_block_main` / `interpolate_chroma_block_main`, and translational MC now selects Table 24/26 vs 25/27 on `sps_admvp_flag` per §8.5.4.3. 3 new tests (zero-CPMV affine-merge skip CU e2e, constructed-Const1 field gradient `(dX·sizeSb) >> 5` exactness, cu_skip affine bin budget).
- **Test-encoder correctness fix**: the in-test `CabacEncoder` is rewritten as an exact carry-propagation arithmetic coder. The previous outstanding-bit construction under-committed the flushed codeword for some patterns (regular sequence `0 1 1 0 0 0 0` mis-decoded its final bin — caught by the new affine e2e), and its bypass path had the long-documented "bypass-tail defer" quirk. The new encoder emits one committed bit per decoder-consumed bit with explicit carry walks and flushes the full 14-bit window at terminate, so **every** regular + bypass sequence round-trips bit-exactly (validated by a 500-sequence randomized round-trip test + the regression pattern).
- §8.5.2.3.3 **collocated temporal merge candidate (TMVP) wired into the ADMVP merge list** — the last unfilled slot of the §8.5.2.3.1 assembly. `InterDecodeInputs` gains `col_pic: Option<ColPicInputs>` (`ColPic`'s per-4×4 motion field + its POC + the POCs of *its* reference lists, so a collocated cell's stored `refIdxLXCol` resolves to the eq.-502 `refPicOfColPic[X]` POC); `admvp_temporal_merge_cand` drives `tmvp_merge_candidate_with_poc` over the threaded grid with the per-cell POC context and strips the L1 half on P slices (the §8.5.2.3.3 list-1 output is B-only), and every ADMVP merge/skip CU now passes the derived candidate into `build_merge_cand_list`'s step-2 slot. Decoder side: `decode_baseline_inter_slice` surfaces the slice's `SideInfoGrid` in its stats; each non-IDR `DpbEntry` retains that motion field plus its reference-list POCs; the slice header now surfaces `col_pic_list_idx` / `col_source_mvp_list_idx` / `col_pic_ref_idx` (§7.3.4, with the P→0 / B→1 inference) and `decode_non_idr` resolves `ColPic = RefPicList[col_pic_list_idx][col_pic_ref_idx]` (§8.3.4) from the DPB. New stat `tmvp_candidates`. 3 new tests: a full e2e P-slice cu_skip CU whose `merge_idx = 0` selects the temporal candidate and motion-compensates at the collocated MV offset, the eq.-503 POC-ratio doubling, and the P-slice L1 strip.
- §8.5.2.3.9 MMVD **wired into the ADMVP CU path with real POC distances** — `InterDecodeInputs` gains `pocs: InterPocs` (`PicOrderCnt(currPic)` + the per-list reference POC arrays, parallel to `ref_list_l0/l1`); the decoder threads the §8.3.1-derived slice POC and the §8.3.5-resolved reference POCs through `decode_non_idr`. `admvp_merge_branch_to_pair` now runs the full `mmvd::mmvd_motion_vector` (group retargeting + POC-asymmetric offset assignment) on every MMVD merge/skip CU, honouring the previously-ignored `mmvd_group_idx`. POC-less synthetic callers (`InterPocs::default`) get a documented equal-distance same-side synthesis under which the clause reduces to the historical symmetric per-list offset, keeping every existing fixture byte-identical. 2 new integration tests (POC-asymmetric bi-pred offset scaling — nearer list takes the offset, farther list the eqs.-599-601 scaled copy; `mmvd_group_idx = 1` dropping L1 on a bi-pred base).
- §8.5.2.3.9 full MMVD motion-vector derivation (`mmvd` module) — the complete eqs. 531-616 process superseding the symmetric-offset placeholder. Stage 1 (group retargeting, `mmvd_group_idx ∈ {1, 2}`): a bi-pred base drops L1 (group 1, eqs. 533-536) or L0 (group 2, eqs. 562-565); a uni-pred B-slice base derives the missing list by POC-ratio scaling (`distScaleFactor = (currPocDiffLX << 5) / currPocDiffLY`, the eq.-543-style Sign/Abs round-half-away-from-zero + `Clip3(±2¹⁵)`), with the eq.-538/555/567/584 mirrored-secondary-reference test (`DiffPicOrderCnt(RefPicListX[1], currPic) == DiffPicOrderCnt(currPic, RefPicListY[refIdxY])` → `refIdx = 1`), group 1 keeping both lists and group 2 dropping the source list; a P-slice base retargets `refIdxL0` (group 1: `!refIdxL0` when 2+ actives, eq. 546; group 2: `refIdxL0 < 2 ? 2 : 1` when 3+ actives, eq. 575) with the fixed ±3 x-nudge (eqs. 547/576) when the target equals the source, else the POC rescale (eqs. 549-553/578-582). Stage 2 (offset assignment, eqs. 591-612): equal `|currPocDiff|` → both lists take `MmvdOffset`; otherwise the nearer list takes it and the farther gets the eqs.-600/601 literal `(dsf·d + 16) >> 5` arithmetic-shift scaling (documented as distinct from the Sign/Abs motion scaling); opposite-side references negate the L1 delta (eqs. 607-610); uni-pred puts the offset on the active list (eqs. 611/612). `MmvdPocs` carries the `DiffPicOrderCnt` context (curr POC + per-list reference POCs) with bounds-checked lookups; zero POC denominators and out-of-range refIdx surface as decode errors. 16 tests (every group×base-shape branch, mirrored-secondary selection, both P retarget flavours, same-side vs straddling negation, both scale directions, error paths).

- §7.3.8.4 Main-profile inter coding-unit **integration** — the slice-walker now selects the `inter_cu_syntax` CU-syntax drivers on `sps_admvp_flag` inside `decode_inter_coding_unit`. `InterToolGates` threads through `InterDecodeInputs` (the all-false default is the historical Baseline `sps_admvp_flag == 0` inline path, so every existing fixture is byte-unchanged). On `sps_admvp_flag == 1`: cu_skip CUs route through `read_cu_skip_main`; non-skip MODE_INTER CUs through `read_inter_cu_mode` (merge-mode) or `read_explicit_amvp` (`merge_mode_flag == 0`). The decoded `CuSkipDecision` / `InterCuModeDecision` / `ExplicitAmvpDecision` is projected into per-CU motion: merge branches assemble the §8.5.2.3 ADMVP `mergeCandList` from the per-4×4 `SideInfoGrid` spatial neighbours (`merge_neighbour_mv_from_grid`) + HMVP merge candidates and run the §8.5.2.3.1 step-6 selection (`admvp_merge_motion_from_grid`); the explicit body adds the §8.5.2.4 grid AMVP predictor to the eq.-145 amvr-shifted MVD; MMVD adds the §8.5.2.3.9 axis-aligned offset (eqs. 133/134) to the base candidate. The shared CBF / residual / motion-compensation tail is factored into `decode_inter_cu_residual_and_reconstruct`, common to the Baseline and Main-profile front-ends. New stats: `admvp_syntax`, `admvp_skip_cus` / `admvp_merge_cus` / `admvp_explicit_cus`. 8 new tests cover the cu_skip regular-merge, non-skip merge-mode, non-skip explicit-AMVP (P uni-pred), B-slice bi-pred zero-fill, real spatial-neighbour merge selection from a populated grid, zero-fill fallback, grid neighbour availability, and MMVD offset application. Deferred on this path: §8.5.2.3.3 collocated temporal merge (needs ColPic motion field + DPB POC), POC-scaled MMVD asymmetry (§8.5.2.3.9 eqs. 531-616), affine CPMV sub-block field (§8.5.3/§8.5.5 — translational fallback for now), and the explicit-affine sub-tree (spec lines 2940-2980).
- §7.3.8.4 Main-profile inter coding-unit syntax driver (`inter_cu_syntax` module) — the picture-level glue that threads the standalone `amvr_syntax` / `mmvd_syntax` / `affine_syntax` readers into the layered §7.3.8.4 mode-gating tree (spec lines 2811-3025). `read_inter_cu_mode` drives the `sps_admvp_flag == 1` non-skip path: `amvr_idx` (else inferred 0) → `merge_mode_flag` (read iff `amvr_idx == 0`, else inferred 1 per line 5827) → merge branch (`mmvd_flag ? MMVD group : affine_flag (size-gated ≥8×8) ? affine_merge_idx : merge_idx`), returning `merge: None` when `merge_mode_flag == 0` so the caller dispatches to the explicit-AMVP driver. `read_explicit_amvp` drives the `merge_mode_flag == 0` body (lines 2912-3025): `inter_pred_idc` (B only) → `bi_pred_idx` (PRED_BI only) → per-list `{ ref_idx? (gated `num_ref_idx_active_minus1>0 && bi_pred_idx==0`), abs_mvd[0]/sign, abs_mvd[1]/sign (gated `bi_pred_idx != 1` for L0 / `!= 2` for L1) }`, capturing `ExplicitAmvpDecision { inter_pred_idc, bi_pred_idx, l0, l1 }`. `read_cu_skip_main` drives the cu_skip merge tree (lines 2811-2832): a skip CU is implicitly a merge CU — `mmvd_flag ? MMVD : affine ? affine_merge_idx : (sps_admvp_flag ? merge_idx : mvp_idx_l0 [+ mvp_idx_l1 for B])`. The shared mmvd/affine/merge_idx fall-through is `read_merge_branch`; the `merge_idx` (TR cMax=(nCbW·nCbH≤32)?3:5, Table 49), `inter_pred_idc` (TR cMax=(!admvp||nCbW+nCbH>12)?2:1, Table 69), `bi_pred_idx` (TR cMax=2, Table 71) and `mvp_idx` (TR cMax=3, Table 48) syntax readers landed alongside in `amvr_syntax`/`inter_cu_syntax`, with `inter::merge_idx_c_max`/`merge_idx_ctx_inc`/`inter_pred_idc_c_max`/`inter_pred_idc_ctx_inc`/`bi_pred_idx_ctx_inc` + `PRED_L0/L1/BI` (Table 8) supplying the §9.3.3 geometry. All collapse to ctx `(0,0)` under Baseline `sps_cm_init_flag == 0`. 28 new tests (each merge/skip/explicit branch, the amvr/merge-mode inference corners, area-gated cMax boundaries, positional ctxInc, bi_pred_idx MVD-suppression gates, ref_idx presence; MVD-value asserts respect the test-encoder's documented bypass-tail defer). The remaining wiring is the slice-walker selecting these drivers on `sps_admvp_flag` and feeding the decoded decisions into the §8.5 MV-reconstruction (`merge`/`affine_cand`/`inter`) already in place.
- §8.5.5 DMVR (decoder-side motion-vector refinement) search core (`dmvr` module): the §8.5.5 bilateral-SAD MV-refinement kernels for the `sps_dmvr_flag == 1` bi-predicted path (`refMvL0 = mvL0 + dMvL0`, `refMvL1 = mvL1 − dMvL0`, eqs. 396/397). `sad_values` (§8.5.5.3 eq. 1018) computes the 9-entry bilateral SAD over the `bC[2][9]` 3×3 integer-offset grid, reading `L0[x+offH0][y+offV0]` against `L1[x−2·bCx+offH1][y−2·bCy+offV1]` (the −2·bC content shift that re-aligns the two predictions). `select_best_idx` (§8.5.5.4) is the array-entry selection with the four-way cross-neighbour quadrant split (`sad[1]`/`sad[7]` horizontal, `sad[3]`/`sad[5]` vertical) + the centre-bias (`sad[4] <= sad[best]` → centre) and diagonal-`idxVal` tie rules. `parametric_refine` (§8.5.5.5 eqs. 1019-1022) adds the per-axis parabolic sub-pel offset `((s_lo−s_hi)<<4) / (2·(s_lo+s_hi−2·sad[4]))` (degenerate axis → 0). `refine_subblock_mv` (§8.5.5.1 eqs. 989-998) is the driver: the `sadVal[4] >= sbW·sbH` refinement gate, the two integer-step passes (8×8-grid `bestIdx/3−1`, `bestIdx%3−1` offsets + the `offsetH`/`offsetV` window shift) and the `fracPelAppliedflag`-gated parametric stage, pure over a caller-supplied `PredPlane` (the `(sbW+4)×(sbH+4)` §8.5.5.2 bilinear-prediction window the decoder's data plane fills). Clean-room note: the §8.5.5.3 SAD window's spec-prose inclusive upper bound `..(nSb + bC)` is read as the sbW×sbH exclusive window (the off-by-one in the prose is resolved by the `(sbW+4)` array dimension). 7 new tests (identical-planes no-refine, DC-offset centre selection, centre-bias + left-neighbour selection, parametric symmetric/biased axes, horizontal-mismatch non-flat cost surface, end-to-end driver flat-surface centre). The §8.5.5.2 bilinear interpolation + the slice-walker invocation are the remaining DMVR wiring.
- §7.3.8.4 AMVR + inter-mode-gating CABAC syntax (`amvr_syntax` module): the §7.3.8.4 non-skip inter-CU mode-gating group (spec lines 2878-2884) that precedes the MMVD / affine / explicit-MVD paths. `read_amvr_idx` reads `amvr_idx` (TR cMax=4, per-bin ctxInc 0,1,2,3 via the existing `inter::amvr_idx_ctx_inc`, Table 67) → the 0..=4 resolution index feeding the §8.5 eq.-145 MVD shift + eqs.-645/646 MVP round already in `inter`. `read_merge_mode_flag` (FL cMax=1, ctxInc 0, Table 70; the `sps_admvp_flag == 1 && amvr_idx == 0` gate, else inferred 1) and `read_direct_mode_flag` (FL cMax=1, ctxInc 0, Table 68; the `sps_admvp_flag == 0` B-slice direct-mode gate) complete the immediate mode-selection group. All share the `sps_cm_init_flag == 0` collapse-to-(0,0) discipline. 6 new tests (amvr_idx values 0/2/3 via the real `AmvrIdx` TR contexts, amvr→§8.5 MVD-shift bridge, both mode flags single-bin, cm_init context routing). 
- §7.3.8.4 MMVD (merge mode with motion-vector difference) CABAC syntax reader (`mmvd_syntax` module): the bitstream-walk companion to the §8.5 MMVD derivation already in `inter` (`mmvd_distance` / `mmvd_sign` / `mmvd_offset` + the per-element ctxInc helpers). `read_mmvd_group` consumes the §7.3.8.4 group (spec lines 2811-2818): `mmvd_flag` (FL cMax=1, ctxInc 0, Table 50); when set, `mmvd_group_idx` (TR cMax=2, per-bin ctxInc 0/1, Table 51, gated by `mmvd_group_idx_present` = `mmvd_group_enable_flag && (log2W+log2H) > 5`, line 2814), `mmvd_merge_idx` (TR cMax=3, per-bin 0/1/2, Table 52), `mmvd_distance_idx` (TR cMax=7, per-bin 0..6, Table 53), `mmvd_direction_idx` (FL cMax=3 = 2 bins, per-bin 0/1, Table 54) — composing the existing `inter::mmvd_*_ctx_inc` helpers + `decode_tr_regular` with the `sps_cm_init_flag == 0` collapse-to-(0,0) discipline shared by `eipd_syntax`/`ats`/`affine_syntax`. Resolves to an `MmvdDecision { flag, group_idx, merge_idx, distance_idx, direction_idx }` the §8.5.2.3.9 MMVD-offset derivation consumes (`distance_idx` → `MmvdDistance` Table 9, `direction_idx` → `MmvdSign` Table 10, `merge_idx` selects the base merge candidate). 6 new tests (group_idx presence predicate, flag-0 single-bin no-MMVD, full group with group_idx present, group_idx absent on small CB, distance_idx TR saturation at cMax=7, decision→§8.5 distance/sign derivation).
- §7.3.8.5 ATS-inter (sub-block transform / SBT, `ats_cu_inter_flag`) syntax + geometry (`ats` module: `AllowAtsInter`, `AtsInter`, `read_ats_inter`): the inter companion to the existing ATS-intra path. `AllowAtsInter::derive` computes the four §6 (lines 6000-6049) `allowAtsInter{Ver,Hor}{Half,Quad}` flags from the CB log2 dimensions + the `[MinTbLog2SizeY, MaxTbLog2SizeY]` window (each flag needs both CB sides `<= MaxTb`; the partitioned side `>= MinTb + (Half?1:2)`). `read_ats_inter` consumes the §7.3.8.5 group (lines 3088-3100) — `ats_cu_inter_flag` (ctxInc per §9.3.4.2.11 eq. 1472 `(log2W+log2H) >= 8 ? 0 : 1`, Table 80), `ats_cu_inter_quad_flag` (ctxInc 0, Table 81, present iff a Half and a Quad orientation coexist), `ats_cu_inter_horizontal_flag` (ctxInc per §9.3.4.2.12 eq. 1473 `(W==H)?0:(W<H?1:2)`, Table 82, present iff both orientations are available for the chosen granularity, else inferred = `allowAtsInterHor{Quad,Half}` per lines 6144/6147), `ats_cu_inter_pos_flag` (ctxInc 0, Table 83) — all FL cMax=1 with the `sps_cm_init_flag == 0` collapse-to-(0,0) discipline. `AtsInter::derive_geometry` resolves the sub-block transform geometry (lines 3103-3127): `TrafoLog2{Width,Height}` reduced by `(quad?2:1)` on the split axis, `TrafoX0`/`TrafoY0` placing the residual sub-block per `pos_flag`. Clean-room note: the spec PDF braces `ats_cu_inter_pos_flag` inside the horizontal-flag presence `if` (line 3098-3101), but pos_flag's "which sub-block carries the residual" meaning is orientation-independent, so it is read unconditionally when the flag is set (documented at the read site as a line-wrap extraction artifact). The transform *type* is unchanged (plain DCT-II per sub-block size), so no new kernel is needed — the existing `inverse_transform` applies over the derived `(trafo_log2_w, trafo_log2_h)`. 8 new tests (allow-flag derivation incl. MinTb/MaxTb boundaries, quad/horizontal presence predicates, geometry for half-horizontal + quad-vertical, eqs.-1472/1473 ctxInc both cm_init branches, end-to-end decode half-horizontal, flag-0 disabled single-bin, single-orientation horizontal inference).
- §8.5.2.3.3 per-cell TMVP POC-distance wiring (`tmvp::PocInputs` / `diff_pic_order_cnt` / `tmvp_merge_candidate_with_poc` / `collocated_cell_from_side_info`): the deferred per-CU TMVP wiring `tmvp`'s module doc named — "only the per-CU wiring of the §8.5.2.3.3 POC distances from the decoded `RefPicList0/1` remains". `diff_pic_order_cnt` is the §8.6.2 eq. 165 `PicOrderCnt(picA) − PicOrderCnt(picB)`. `PocInputs` carries the per-CU POCs (`currPic`, `RefPicList0[0]`, `RefPicList1[0]`, `ColPic`); `PocInputs::derive(col_ref_l0_poc, col_ref_l1_poc)` builds the full `PocContext` for one collocated cell — eq. 501 `currPocDiffLX = DiffPicOrderCnt(currPic, RefPicListX[0])` (per-CU) + eq. 502 `colPocDiffLX = DiffPicOrderCnt(ColPic, refPicOfColPic[X])` (per-cell). The clean-room reading of §8.5.2.3.4 is that `colPocDiff` depends on `refPicOfColPic[X]` — the picture `ColPic`'s stored motion *at that collocated sample* referenced — so it is genuinely per-cell, not a single per-CU constant: `CollocatedCell` extends `CollocatedMv` with the two resolved `refPicOfColPic` POCs, and `tmvp_merge_candidate_with_poc` recomputes the eq.-503 `distScaleFactor` per consulted central/bottom/side position from that cell's POCs (otherwise identical fallback order/gating/grid-snap/small-block-demotion to `tmvp_merge_candidate`). `collocated_cell_from_side_info` bridges the decoder's `SideInfoGrid` plus a `col_ref_poc(list, ref_idx)` ColPic-reference-list lookup into the per-cell type (querying the POC lookup only for valid lists). The remaining wiring is the decoder threading `PicOrderCnt`s into `RefPictureView`/the DPB so the slice-walker can supply `PocInputs` + the ColPic reference-list POC map. 5 new tests (eq.-165 subtraction, eqs.-501/502 derive, per-cell central scaling, per-cell zero-colPocDiff fall-through, cell bridge ref-POC resolution gated on list validity).
- §8.5.3.4 affine corner CPMV resolution incl. the corner-2/3 collocated-MV temporal fallback (`affine_cand::resolve_affine_corners`): the deferred §8.5.3.4 bridge that fills the four `cpMvLXCorner[ 0..3 ]` slots `constructed_merge_candidates` consumes, pure over a §6.4.3 spatial-neighbour lookup (`merge::NeighbourMv`) and a `ColPic` collocated-MV lookup (`tmvp::CollocatedMv`) — the same purity contract as `merge`/`tmvp`. Corner 0 scans `B2 → B3 → A2` (eqs. 764-767), corner 1 `B0 → B1 → C2` (eqs. 768-771), each taking the first §6.4.3-available neighbour's stored `PredFlagLX`/`RefIdxLX`/`MvLX`. Corner 2 (eqs. 772-784) scans `A0 → A1` when `availLR ∈ {LR_10, LR_11}`, otherwise falls back to the §8.5.2.3.4 collocated MV at `(xCb−1, yCb+cbHeight)` (eqs. 776/777); corner 3 (eqs. 785-797) scans `C0 → C1` when `availLR ∈ {LR_01, LR_11}`, otherwise the collocated MV at `(xCb+cbWidth, yCb+cbHeight)` (eqs. 789/790). The collocated fallback (`corner_from_collocated`) applies the §8.5.3.4 same-CTB-row + in-picture gate (`yCb >> CtbLog2SizeY == yCol >> CtbLog2SizeY`, `yCol < pic_height`, the per-corner horizontal test `xCol > 0` / `xCol < pic_width`), 8×8-grid-snaps the position (`(v >> 3) << 3`), runs `tmvp::tmvp_collocated_mv` with `refIdxLXCorner = 0`, and fills `cpMvL0Corner`/`cpMvL1Corner` per eqs. 778-784 / 791-797 — list 1 retained only on a B-slice. This closes the §8.5.3.4 corner-resolution wiring the `affine_cand` candidate-list layer left as a caller responsibility; what remains is the slice-walker supplying the real §6.4.3 grid + `ColPic` motion store. 5 new tests (corner-0/1 first-available scan order, corner-2 collocated fallback on LR_01, corner-3 same-CTB-row gate skip, collocated list-1 B-slice-only retention, resolved-corners → Const1 end-to-end).
- §8.4.4 EIPD end-to-end picture-buffer reconstruct (`picture::intra_reconstruct_cb_eipd`): the EIPD (`sps_eipd_flag == 1`) analogue of `intra_reconstruct_cb`, tying `fetch_eipd_refs` → `eipd::predict_eipd` → §8.7.5 picture-construction (`rec = clip(pred + res)`, eq. 1091) → `store_block` into one call. Derives the §6.4.2 `availLR` the §8.4.4 kernels consume from the simplified causal rule (left available iff `x > 0`, right per the SUCO `right_available`), threads `sps_suco_flag`/`right_available` into the §8.4.4.1 construction, and supports any EIPD mode index 0..32 + any component. This completes the data-plane API parity for the EIPD intra path (the Baseline path already had `intra_reconstruct_cb`); remaining is the slice-walker selecting this vs the Baseline path on `sps_eipd_flag` and supplying the real §6.4.1/§6.4.2 availability. 2 new tests (first-CU INTRA_DC mid-level reconstruct, INTRA_VER top-row copy + residual offset end-to-end).
- §8.7.6 HTDF picture-buffer bridge (`picture::apply_htdf_luma`): the data-plane half of the HTDF wiring — applies the `htdf` module's filter to a luma coding block in place on the `YuvPicture` buffer. Short-circuits (returns false, plane untouched) when the §8.7.6.1 applicability gates fail; otherwise derives the §8.7.6.3 LUT, builds the §8.7.6.2 padded array from the luma plane (the §8.7.6.2 `dx`/`dy` clamp keeps every read in-extent, with a defensive plane-bounds clamp), runs §8.7.6.1 `filter_block` and writes the modified samples back. Border availability uses the in-picture-extent rule (single-slice/tile, `constrained_intra_pred_flag == 0`); threading the real §6.4.1 predicate + invoking from the §8.4.1/§8.6 reconstruction (when `cbf_luma && sps_htdf_flag`) is the remaining slice-walker step. 3 new tests (inapplicable-block no-op, flat-field identity, impulse smoothing without amplification on the plane).
- §8.7.6 Hadamard Transform Domain Filter (HTDF) post-reconstruction filter (`htdf` module, `sps_htdf_flag == 1`): the luma-only post-reconstruction filter the §8.4.1 / §8.6 reconstruction invokes before in-loop deblocking/ALF. `htdf_applies` encodes the four §8.7.6.1 applicability gates (`nCbW·nCbH < 64`, `Max ≥ 128`, `Min ≥ 32 && !intra`, `QpY ≤ 17`). `derive_htdf_lut` (§8.7.6.3) selects `(bLUT, aTHR, tblShift)` from the `setOfLUT[5][16]` (eq. 1108) / `tblThrLog2[5]` (eq. 1110) tables keyed on `qpIdx` (eq. 1106, the inter-square-≥32 vs default branch) with `tblShift = tblThrLog2[qpIdx] − 4` (eq. 1109) and `aTHR = (1 << tblThrLog2[qpIdx]) − (1 << tblShift)` (eq. 1111). `pad_rec_samples` (§8.7.6.2) builds the `(nCbW+2)×(nCbH+2)` replicate-padded array with the `dx`/`dy` border clamp gated on a `BorderAvailability` trait (the §6.4.1 + `constrained_intra_pred_flag` predicate; `InPictureBorder` is the single-slice in-extent rule). `filter_block` (§8.7.6.1) runs the sliding 2×2 Hadamard (eqs. 1093-1097), the eqs.-1098/1099 bit-depth-branched soft-threshold of the three AC coefficients (DC kept), the inverse Hadamard (eqs. 1100-1103), the eq.-1104 overlap accumulation (`+= invHadFilt[i] >> 2`) and the eq.-1105 `Clip1Y((accFlt + 2) >> 2)` rounding. Pure over the reconstructed-sample accessor; the slice-walker wiring (invoke when `cbf_luma && sps_htdf_flag` per §8.4.1 / §8.6 line-8929) is the next step. 5 new tests (applicability gates, qpIdx/LUT selection both branches, flat-field identity, padding border replication both edges, impulse smoothing/no-amplification).
- §8.4.4.1 EIPD picture-buffer reference fetch (`picture::fetch_eipd_refs`): bridges the new `eipd_ref` construction/substitution onto the `YuvPicture` reconstructed-sample buffer, the EIPD analogue of the existing Baseline `fetch_intra_refs`. Gathers the full §8.4.4.1 EIPD neighbourhood — top row `p[x][-1]`, left column `p[-1][y]` (each spanning `nCbW + nCbH`), the `p[-1][-1]` corner, and the SUCO `p[nCbW][y]` right column — by driving `eipd_ref::construct_eipd_refs` with two closures over the picture plane: an availability predicate mirroring `fetch_intra_refs`'s causal rule (top available iff above the block, left iff to its left, both in-picture) extended to the right column (gated on the caller's `right_available` SUCO split-unit-coding-order resolution), and a sample lookup reading the post-reconstruction plane. Returns the substituted `EipdRefSamples` ready for `predict_eipd`. This is the data-plane half of the intra wiring; threading the real §6.4.1 `IsCoded`/tile predicate + `constrained_intra_pred_flag` through the availability closure remains for the slice-walker integration. 3 new tests (first-CU all-grey EIPD substitution, left-neighbour carry-through with EIPD top-row corner-chain fill, SUCO right-column population from the plane).
- §8.4.4.4 EIPD INTRA_HOR right-column blend + §8.4.4.5 INTRA_VER (`eipd::predict_eipd`): corrects the `sps_eipd_flag == 1` INTRA_HOR derivation, which previously routed through `predict_directional`'s pure left-column copy for every `availLR`. Per §8.4.4.4 the EIPD INTRA_HOR is `availLR`-dependent: `LR_11` (both neighbours) runs the eq.-290 per-`x` horizontal blend of `p[−1][y]` and `p[nCbW][y]` weighted by `(nCbW−x)` / `(x+1)` and normalised by `divScaleMult[Log2(nCbW)] >> divScaleShift` (Table 17); `LR_01` (right only) is the eq.-291 pure right-column copy `p[nCbW][y]`; `LR_00`/`LR_10` keep the eq.-292 pure left-column copy. INTRA_HOR (mode 24) and INTRA_VER (mode 12) are now dispatched directly from `predict_eipd` into dedicated `predict_hor_eipd` / `predict_ver_eipd` kernels rather than special-cased inside the directional path (their Table 20 rows are dashes). 3 new tests (LR_01 right-copy, LR_11 constant-column reciprocal-exact blend, LR_11 monotone edge blend between distinct column values).
- §8.4.4.1/.2 intra reference-sample construction + substitution process (`eipd_ref` module): the deferred intra follow-up that gathers the `p[x][y]` neighbourhood the §8.4.4.3–§8.4.4.10 kernels consume and fills every not-available location. `construct_eipd_refs` runs the §8.4.4.1 *General* construction — the top row `p[x][−1]` (x = 0..nCbW+nCbH−1), left column `p[−1][y]`, `p[−1][−1]` corner, and the SUCO `p[nCbW][y]` right column — over two caller-supplied closures (an `availableN` predicate already folded with `constrained_intra_pred_flag`, and a reconstructed-sample lookup), marking each location available/not-available and copying the recon value through when available (the `merge`/`tmvp`/`inter` purity contract). The §8.4.4.2 substitution then fills holes per profile: `sps_eipd_flag == 0` assigns the mid-level constant `1 << (bitDepth−1)` to every hole (corner first, then top row, then left column); `sps_eipd_flag == 1` assigns mid-level only to an unavailable corner and copies the scan predecessor for every other hole (`p[x−1][−1]` along the top row, `p[−1][y−1]` down the left column, and on the SUCO path `p[nCbW][y−1]` down the right column, with the y=0 right-column predecessor taken from the top row at x=nCbW). Returns a `ConstructedRefs` { filled `EipdRefSamples`, `substituted` flag }; pure, no DPB coupling (the decoder wires its `IsCoded` raster, tile map and recon plane into the closures). The remaining intra wiring is threading these closures from the slice walker into the §8.4.4 dispatch. 5 new tests (all-available copy-through routing, baseline mid-level fill, EIPD corner-chain + mid-row-hole predecessor copy, SUCO right-column construction + substitution).
- §8.5.3.2/.4/.5/.6 affine merge/predictor candidate-list assembly (`affine_cand` module): the deferred affine candidate-list layer that feeds the §8.5.3.7 subblock-MV derivation, pure over caller-supplied neighbour/corner sources (the `merge`/`tmvp` purity contract). §8.5.3.4 `constructed_merge_candidates` builds the six corner-combined candidates Const1..Const6 (eqs. 798-835) from the four resolved corner CPMVs — Const1 {0,1,2}, Const2 {0,1,3} with the eq.-807 corner-2 completion `clip(c3+c0−c1)`, Const3 {0,2,3} eq.-813 `clip(c3+c0−c2)`, Const4 {1,2,3} eq.-819 `clip(c1+c2−c3)`, Const5 {0,1} 4-param, Const6 {0,2} with the eqs. 828/829 top-right derivation (`shift = 7 + 2·log2W − log2H`, §8.5.3.10 `rightShift = 7` rounding + eqs. 834/835 16-bit clip) — gating each list X on matching `refIdxLXCorner` + `predFlagLXCorner == 1`. §8.5.3.2 `build_affine_merge_cand_list` assembles `affineMergeCandList` (≤ 5): the inherited model-based neighbours (`availLR`-ordered via `affine_merge_inherited_order`, eqs. 720/721, projected through §8.5.3.3 `inherited_cp_mvs`), then Const1..6, then the step-9 zero-CPMV tail (eqs. 723-733, P-slice L0-only / B-slice bi-pred, `motionModelIdc = 1`); `select_affine_merge_candidate` is the eqs. 735-741 index bridge. §8.5.3.6 `constructed_mvp_candidate` derives the single constructed predictor (eqs. 868-872, the corner-2 completion `clip(c3+c0−c1)` + the 4-param `MotionModelIdc == 1` fallback). §8.5.3.5 `build_affine_mvp_cand_list` assembles `cpMvpListLX` (exactly 2): the inherited A/B/C refIdx-matched predictors (steps 4-6, eqs. 843-854), the constructed predictor (step 7), the per-corner translational fill (step 8, cpIdx 2→0 with the cpIdx-2→3 redirect, eqs. 859-861), then the zero tail (step 9, eqs. 863-866). `reconstruct_affine_amvp_cp_mvs` composes the eq.-867 predictor select with the §8.5.3.1 (eqs. 688-691) per-control-point MVD reconstruction. Neighbour geometry exposed separately: `affine_merge_nb_positions` (eqs. 708-717), `affine_mvp_nb_positions` (eqs. 836-842), `AffineNbName` resolution. The §8.5.3.4 corner-2/3 collocated-MV temporal fallback threads through the same `CornerMv` slots (the caller-side §8.5.2.3.4 lookup wiring deferred). 21 new tests (each Const arithmetic path + refIdx gating, list capping at 5, P/B zero fill, constructed-after-inherited ordering, MVP three-corner / corner-2-completion / 4-param / unavailable cases, MVP zero + corner fills, AMVP select+MVD reconstruction flag-0/1 + 4/6-param, eq.-exact neighbour positions + availLR order split).
- §8.5.3.7–§8.5.3.10 + §8.5.3.1 affine subblock-MV geometric core (`affine` module): the pure §8.5.3 derivation that turns a 2- or 3-CPMV (control point motion vector) set into a dense per-subblock motion field. `affine_model_params` (§8.5.3.9) derives `dX`/`dY`/`mvBaseScaled` (eqs. 897-906) incl. the 4-parameter `dY` rotation identity (`dY[0] = −dX[1]`, `dY[1] = dX[0]`) and the 6-parameter genuine vertical gradient. `affine_subblock_size` (§8.5.3.8) derives `(sizeSbX, sizeSbY)` from Tables 22/23 (`mvWx`/`mvWy` → size, eqs. 879-882), the EIF bounding-box applicability test (eqs. 883-890, `(W+2)·(H+2) > 72` → `clipMV`; the `dY[1] < −512` / slope-bound `eifCanBeApplied` disqualification, eqs. 891-894 floor-to-8), and `numSb{X,Y} = cb{Width,Height} / sizeSb{X,Y}` (eqs. 895/896). `affine_subblock_mvs` (§8.5.3.7) evaluates the affine model at each subblock centre (eqs. 873-876), applies the §8.5.3.10 `rightShift = 5` rounding + eq.-877/878 18-bit clip, and derives the §8.5.2.6 chroma MV (eqs. 676/677, `* 2 / SubWC`); output luma field is 1/16-pel, chroma 1/32-pel. `reconstruct_cp_mv` (§8.5.3.1 eqs. 688-691) is the AMVP-path 16-bit-modular CP reconstruction; `affine_center_mv` (eqs. 696-701) is the §8.5.2.7 HMVP-update centre vector (`rightShift = 7`, `Clip3(−2¹⁵, 2¹⁵−1)`). Pure over the CPMV inputs, no DPB coupling. 11 new tests (translational identity, 4/6-param slopes, subblock shrink, chroma subsampling modes, CP wrap, centre clip).
- §8.5.3.3 inherited affine CPMV derivation from a neighbour (`affine::inherited_cp_mvs` + `NeighbourAffineSource`): projects an affine-coded neighbour's stored corner motion vectors onto the current block's control points (the affine-merge inheritance path). Picks the `isCTUboundary` row (eqs. 744-751; above-CTU-row neighbour samples its bottom edge, ordinary case the top edge), derives the `dHorY`/`dVerY` model-idc split (eqs. 752-755; genuine 6-param vertical gradient vs the 4-param/CTU-boundary rotation identity), projects onto `cpMvLX[0..numCpMv]` (eqs. 756-761), then applies §8.5.3.10 `rightShift = 7` rounding + eq.-762/763 `Clip3(−2¹⁵, 2¹⁵−1)`. Pure over the supplied neighbour corners. 3 new tests (translational inheritance, 6-param distinct CPs, CTU-boundary bottom-edge sampling).
- §7.3.8.4 affine inter-syntax CABAC reads (`affine_syntax` module): the syntax-layer companion to the §8.5.3 geometric core. `read_affine_flag` (FL cMax=1) applies the §9.3.4.2.4 / Table 96 neighbour-derived ctxInc `= Min(Σ condX&&availX, 1)`. `read_affine_merge_idx` (TR cMax=5, cRice=0) reads the prefix with per-bin ctxInc 0..4 (Table 56). `read_affine_mode_flag` (Table 57) yields `numCpMv = 2 + flag` (eqs. 136/137). `read_affine_group` composes them with `affine_mvp_flag_lX` (Table 58) + `affine_mvd_flag_l0`/`l1` (Tables 59/60), read per active list per `inter_pred_idc`, into an `AffineDecision` (`NotAffine` / `Merge{idx}` / `Amvp{mode, l0, l1}`) the derivation consumes. `sps_cm_init_flag == 0` collapses regular bins to `(0,0)` like `eipd_syntax`/`ats`. 9 new tests (ctxInc saturation, TR merge_idx values + saturation, merge + AMVP bi-pred + L0-only group walks, exact per-element bin budgets).
- §8.5.2.3.3–§8.5.2.3.5 temporal (collocated) merge-candidate (TMVP) derivation (`tmvp` module): the §8.5.2.3.3 step-2 sub-derivation that produces the `Option<MergeCand>` `merge::build_merge_cand_list` already accepts in its `temporal` slot. `tmvp_collocated_mv` (§8.5.2.3.4) scales a collocated sample's stored motion by the POC ratio — `distScaleFactor = (currPocDiff << 5) / colPocDiff` (eq. 503), `mvp = Clip3(−32768, 32767, Sign(p)·((Abs(p)+16) >> 5))` (eq. 504, round-half-away-from-zero) — gated on the eq.-after-504 invalid-refIdx / zero-colPocDiff escape (→ availableFlagLXCol = 0), and folds the per-list availability into the joint `availableFlagCol` code (0/1/2/3). `constrain_scaled_mv` (§8.5.2.3.5) clips each scaled vector against the padded reference grid (eqs. 506-511, `picPaddingSize = 144`). `tmvp_merge_candidate` (§8.5.2.3.3) walks the central (eqs. 485/486), bottom (eqs. 489-492, gated on same-CTB-row + in-picture-height), then side (eqs. 495-498, gated on strictly-in-picture-width) collocated positions in order — each 8×8-grid-quantised (`(v >> 3) << 3`) and looked up via a caller-supplied `ColPic` motion-field closure (mirroring the `merge`/`inter` purity contract) — stopping at the first non-zero `availableFlagCol`, applying the eqs. 487/488 small-block (`nCbW+nCbH ≤ 12`) bi-pred→uni demotion, and emitting the `MergeCand` with `refIdxLXCol = 0`. The per-step §8.5.2.3.10 redundancy trim stays in `build_merge_cand_list`'s step-2. Pure spec functions, no DPB coupling (the decoder's reference-picture motion store wires the closure). 12 new tests (equal/double/negative POC scaling, zero-colPocDiff + invalid-refIdx escapes, small-block demotion, both boundary-clip directions, central short-circuit, bottom/side fallback chain, CTB-row gate, all-unavailable → None).
- §8.5.2.3.4 collocated motion-field bridge (`tmvp::collocated_mv_from_side_info`): adapts a decoded picture's `deblock::SideInfoGrid` — the per-4×4-cell motion field stamped during reconstruction — into the `CollocatedMv` a `tmvp_merge_candidate` `col_at` closure returns, so the existing decoder data structure serves as the `ColPic` motion store without a new DPB-level per-picture array. Maps `predFlagLXCol` ⇐ (cell coded `Inter` ∧ `refIdxLX != −1`), `refIdxLXCol`/`mvLXCol` ⇐ the cell's stored `ref_idx_lX`/`mv_lX`; intra / IBC / out-of-grid cells read invalid (both lists off → §8.5.2.3.4 availableFlagCol = 0). The luma `(x,y)` is snapped to the covering 4×4 cell (`>> 2`). 3 new tests (inter-cell mapping, intra/IBC + OOB invalidity, end-to-end SideInfoGrid → §8.5.2.3.3 scaled-candidate recovery).
- §8.5.2.3.1 step-6 merge selection bridge (`merge::select_merge_candidate` → `MergedMotion`): projects a decoded `merge_idx` (or `mmvd_merge_idx`) into the concrete `mvLX[0][0]` / `refIdxLX` / `predFlagLX[0][0]` the §8.5.4 inter sample-prediction process consumes (eqs. 450-453), applying the eq. 453 valid-reference gate (a candidate's inactive-list stale `refIdx == -1` never lights `predFlagLX`). Out-of-range `merge_idx` returns `None` (caller surfaces a decode error). This is the reconstruction bridge from the candidate list to motion compensation. 3 new tests (chosen-candidate projection, OOB rejection, valid-ref gate).
- §8.5.2.3.6 history-based (HMVP) merge-candidate derivation (`HmvpCandList::hmvp_merge_candidates`): walks `HmvpCandList[ NumHmvpCand − hMvpIdx ]` for `hMvpIdx = 3, 7, 11, …` up to `Min( maxNumCheckedHistory, (mLSize == 4) ? 15 : 23 )` with `maxNumCheckedHistory = ( ( ( NumHmvpCand + 1 ) >> 2 ) << 2 ) − 1`, applying the step-1 small-block (`nCbW+nCbH ≤ 12`) bi-pred→uni demotion. Returns the ordered `MergeCand` tail-walk consumed by `merge::build_merge_cand_list`'s §8.5.2.3.1 step-3 HMVP loop (which enforces the per-append §8.5.2.3.10 trim + `numCurrMergeCand == mLSize` stop). Empty when `NumHmvpCand < 4`. 3 new tests (too-few-entries, stride-of-4 from index 3, small-block demotion).
- §8.5.2.3 ADMVP merge-mode candidate-list derivation (`merge` module): the Main-profile `sps_admvp_flag == 1` merge/direct MV-prediction candidate construction, implemented as pure spec functions parameterised on a caller-supplied §6.4.3 neighbour-MV lookup (mirroring the `inter::build_amvp_list_baseline` purity contract). `spatial_neighbour_positions` resolves the five §8.5.2.3.2 neighbour luma locations (A1/B1/B0/A0/B2) across all three §6.4.2 `availLR` branches (LR_11 double-sided fan, LR_01 mirrored, LR_10/LR_00 left-biased). `spatial_merge_candidates` appends them in order with the eqs. 463/464 small-block (`nCbW+nCbH ≤ 12`) bi-pred→uni demotion, the `numCurrMergeCand < mLSize−1` A0/B2 gate, and the §8.5.2.3.10 redundancy trim on B1/B0/A0/B2. `combined_bipred_candidates` implements the §8.5.2.3.7 B-slice combined-bipred pass with the Table 21 `l0CandIdx`/`l1CandIdx` pairing and the `numInputMergeCand·(numInputMergeCand−1)` stop. `zero_mv_candidates` fills the §8.5.2.3.8 tail (P-slice uni-L0 vs B-slice bi-pred zero per the `biPredAllowed` predicate). `build_merge_cand_list` is the §8.5.2.3.1 general assembly — `mLSize = (nCbW·nCbH ≤ 32) ? 4 : 6`, spatial → optional temporal (caller-threaded §8.5.2.3.3 collocated cand) → HMVP merge cands (§8.5.2.3.6, slice-passed) → combined bipred → zero fill, each append running the redundancy trim. Temporal-collocated (§8.5.2.3.3/.4, needs DPB motion field) and HMVP-merge derivation (§8.5.2.3.6, threads `HmvpCandList`) are accepted as caller inputs rather than derived in-module — wiring deferred to the next round. 15 new tests (position geometry per `availLR`, distinct-collect, B1 redundancy trim, small-block demotion, A0/B2 gate, P/B zero fill, combined-bipred pairing, full general-assembly with temporal+HMVP threading and mLSize=4 small-CU).
- §7.3.8.5 ATS-intra (Adaptive Transform Selection, `sps_ats_flag == 1`) syntax + transform-kernel layer (`ats` module + `transform::inverse_transform_ats`): `read_ats_intra` consumes the §7.3.8.5 group — `ats_cu_intra_flag` (FL cMax=1, bypass per Table 95), and when set `ats_hor_mode` / `ats_ver_mode` (FL cMax=1, ctxInc 0, Table 79) — then applies the Table 30 derivation (`trTypeHor = 1 + ats_hor_mode`, `trTypeVer = 1 + ats_ver_mode`; trType 0=DCT-II / 1=DST-VII / 2=DCT-VIII). `ats_intra_flag_present` encodes the line-3080 presence predicate (`sps_ats_flag && log2CbWidth <= 5 && log2CbHeight <= 5 && cbf_luma`). `inverse_transform_ats` runs the §8.7.4.1 trType-parameterized two-stage inverse (column stage `trTypeVer`, row stage `trTypeHor`) with the §8.7.4.3 DST-VII (eqs. 1077/1078) and DCT-VIII (eqs. 1084/1085) 4×4 + 8×8 matrices transcribed from the spec; 16/32 DST/DCT sizes surface `Unsupported` (deferred) rather than panic. `EipdCtx::is_cm_init` exposes `sps_cm_init_flag` so the Table 79 context shares the Baseline collapse-to-`(0,0)` discipline. 11 new tests.
- §7.3.8.4 EIPD intra-mode CABAC syntax reader (`eipd_syntax` module) + §9.3.3.6 TB primitive (`cabac::decode_tb_bypass`): wires the `sps_eipd_flag == 1` per-CU intra-mode bitstream onto the CABAC engine. `read_luma_mode_selector` reads `intra_luma_pred_mpm_flag`/`idx` (FL cMax=1, ctxInc 0, Tables 63/64), `intra_luma_pred_pims_flag` (bypass), `intra_luma_pred_pims_idx` (FL cMax=7 bypass), `intra_luma_pred_rem_mode` (TB cMax=22, all-bypass via the new truncated-binary decode) per Table 95, resolving `Mpm`/`Pims`/`Rem(rem_mode+10)`. `read_intra_chroma_pred_mode` decodes the §9.3.3.7 Table 93 bin string (bin0 ctxInc 0 Table 65, rest bypass) → 0..4. `resolve_eipd_luma_mode` / `resolve_eipd_chroma_mode` compose those reads with the §8.4.2/.3 derivation + selection, producing the concrete `IntraPredModeY` / `IntraPredModeC` that `eipd::predict_eipd` consumes; the `EipdCtx` selector honours `sps_cm_init_flag`. 15 new tests incl. a full syntax→derivation→`predict_eipd` kernel run and the TB codeword bijection / prefix-free / cMax-22 length checks.
- §8.4.4.8/.9/.10 EIPD intra prediction-sample derivation (`eipd` module): the Main-profile `sps_eipd_flag == 1` sample kernels for `INTRA_BI` (bilinear, eqs. 297-311 with the Table 17 `divScaleMult` / Table 18 `weightFactor`), `INTRA_PLN` (planar, eqs. 314-325 with the Table 19 `mult`/`shift`), the `INTRA_DC` aspect-ratio average (eqs. 286-288), and the full 33-direction angular set (§8.4.4.10 Table 20 `dirXYSign`/`divDxy`/`divDyx`, the two-step `iOffset`/`iX`/`iY`/`refPosition` derivation incl. the LR_01/LR_11 vs LR_00/LR_10 quadrant branches, eqs. 326-364, then the 4-tap fractional filter eqs. 365-385). Driven by a new `EipdRefSamples` neighbourhood exposing `p[x][-1]` / `p[-1][y]` / `p[nCbW][y]` at the spec's `-1` origin and an `AvailLr` (eq. 23) availability enum. 6 new tests (flat-field fixed-point across all modes/sizes/LR codes; VER/HOR copies; DC average; directional in-range under steep gradients).
- §8.4.2 EIPD luma intra-mode derivation (`eipd_mode` module): the step-2 candidate A/B/C validity-pruning fold-down, step-3 `candModeList[0..1]` (eqs. 172-175), step-4 `extCandModeList[0..7]` across all six validC×{both-planar, planar+directional, both-directional} branches (the anchor list eqs. 176-182, single-directional fill eqs. 187-206, and the `list[]` dedup-fill loops eqs. 209-278), step-5 `remModeList[0..32]` from `defaultModeList[33]` (proven a 33-mode permutation by test), and the step-6 `ModeSelector` dispatch. 5 new tests.
- §8.4.3 EIPD chroma intra-mode derivation (`eipd_mode::derive_chroma_mode`): `intra_chroma_pred_mode == 0` DM reuse + Table 16 mapping with the `modeIdx` skip rule that bumps the chroma index past any luma-occupied entry. 3 new tests.
- §8.9.8 `tableNum == 0` branch (eqs. 1398-1409, `chroma_scale_joined`): reconciled the joined-chroma-scale pivot-boundary sub-case against the now-staged errata #81 / #130 (`docs/video/evc/evc-errata-and-clarifications.md`). The branch is confirmed to be a guard around eqs. (1400)-(1402) only — the divide + `(1<<9) − x % (1<<9)` complement that misbehave when `scaleDraNorm` lands exactly on a `ScaleQP[]` pivot — falling through to eqs. (1403)-(1409) unchanged, NOT a separate identity output. Published net result: `qpDraFrac = 0`, `qpDraInt = (2*IndexScaleQP − 60) − 1`, `qpDraIntAdj = qpDraFracAdj = 0`, and `draChromaQpShift = ChromaQpTable[cIdx][dra_table_idx] − qp0 − qpDraInt` (eq. 1409) with `qp0` from the clipped eq. 1403 `idx0`; the errata explicitly rejects short-circuiting `draChromaQpShift` to a constant. The round-284 in-tree pin (filed when the errata had no §8.9.8 entry) is promoted to a formal errata-cited lock: code + test comments now cite #81 / #130. 4 new tests lock the published net-result table directly — full-chain byte-equality with the public output, `draChromaQpShift` varies with `dra_table_idx` (not constant), the `qpDraInt` `−1` decrement at two distinct pivots, and eq. 1409 continuity across the pivot (errata evidence #4). No behaviour change; 733 tests pass (was 729).
- §7.4.9.3 SUCO (split-unit-coding-order) availability layer (`split` module): `SucoSizeLimits::derive` computes `MaxSucoLog2Size` (eq. 68, `Min(CtbLog2SizeY − log2_diff_ctu_size_max_suco_cb_size, 6)`) and `MinSucoLog2Size` (eq. 69, `Max(MaxSucoLog2Size − log2_diff_max_suco_min_suco_cb_size, Max(4, MinCbLog2SizeY))`) from the `sps_suco_flag == 1` syntax. `allow_split_unit_coding_order` implements the four-condition §7.4.9.3 `allowSplitUnitCodingOrder` derivation (lines 5505-5521): FALSE when the block's longer side exceeds `MaxSucoLog2Size` or shorter side is below `MinSucoLog2Size`; when it straddles a picture edge; when `log2CbWidth <= log2CbHeight` with `split_cu_flag == 0`; or when `SplitMode` is `SPLIT_BT_HOR`/`SPLIT_TT_HOR`/`NO_SPLIT` with `split_cu_flag == 0`. This is the predicate that — together with `sps_suco_flag` — gates whether `split_unit_coding_order_flag` is signalled (spec line 2685), feeding the existing recursion-geometry layer's mirrored-order reordering. Pure geometry, no CABAC bins. 8 new tests (eqs. 68/69 incl. the `Max(4,MinCb)` floor and the `Min(…,6)` clamp; each of the four FALSE conditions; the quad-split and shape escapes).
- §7.3.8.3 split-unit recursion-geometry layer (`split::split_unit_children` + `split::quad_split_children`): given a resolved `SplitMode` (or a `split_cu_flag == 1` quad split) at a parent block, enumerates the ordered child `split_unit()` invocations exactly as the §7.3.8.3 syntax body (lines 2690-2787) dictates — each `SplitChild` carries the child luma position, log2 dimensions, `ctDepth` (parent + 1 for BT/half children, parent + 2 for the TT quarter children), and `splitUnitOrder`. Implements the spec's picture-boundary child gating (BT_VER right child / BT_HOR bottom child / quad TR/BL/BR gated on `x1 < pic_width` & `y1 < pic_height`; ternary shapes emit all three children unguarded since `allowSplitTt*` already requires the whole parent in-picture) and the `sps_suco_flag` mirrored-order recursion (`splitUnitOrder = 1`, reversed column/quadrant order). Pure geometry, consumes no CABAC bins — the recursion skeleton the future CABAC-driven coding-tree walker drives. 14 new tests (BT/TT both directions, quad, boundary gating per shape, SUCO mirroring).
- §7.3.8.3 CABAC-driven BTT split-syntax reader (`split::decode_btt_split`): consumes the `btt_split_flag` / `btt_split_dir` / `btt_split_type` bins from a `CabacEngine` against Tables 42/43/44, applies the §7.3.8.3 presence gating + §7.4.8.3 inference for absent elements, and resolves the final `SplitMode` via the existing geometry layer. ctxInc wiring per Table 95 / §9.3.4.2.5: `btt_split_flag` uses eq. (1440) `Min(numSmaller,2)+3*ctxSetIdx` under `sps_cm_init_flag==1` (0 otherwise), `btt_split_dir` uses `log2CbWidth−log2CbHeight+2` (clamped 0..=4), `btt_split_type` ctxInc 0 unconditionally. Returns `BttSplit { mode, flag, dir, split_type }` + `BttSplitStats` bin tallies. 7 new round-trip tests (binary/ternary, both directions, errata-corrected TT_HOR, dir/type/flag inference cases).
- §8.9.4 spec-faithful chroma inverse-mapping apply (eqs. 1377-1382): `map_one_chroma_sample` (per-sample magnitude scale: abs-multiply, `>> 9` truncate, restore sign, re-pivot, `Clip1C`) + `apply_chroma_inverse_mapping_u8` (whole-plane apply driven by the §8.9.6/§8.9.7/§8.9.8 `DraChromaDerived` `chromaScale` per co-located pre-mapping luma sample). Supersedes the round-11 per-segment chroma QP-offset approximation for the spec-faithful DRA path. 8 new tests.
- §7.4.8.3 BTT split-geometry layer (`split` module): `allowSplitBtVer`/`BtHor`/`TtVer`/`TtHor` derivations, `btt_split_dir`/`type` signalling + inference predicates, and the `SplitMode` derivation incl. picture-boundary implicit-split rules; §7.3.2.2 BTT size limits (eqs. 43/44/62-67). Documents the §7.4.8.3 `SplitMode`-table horizontal-ternary typo (lines 5459-5469 list SPLIT_TT_VER; recursion-consistent reading is SPLIT_TT_HOR)
- Cargo.toml `description` updated to reflect the working Baseline IDR/P/B 8-bit 4:2:0 pixel decoder (was the stale round-1 "scaffold: pixel decode pending" string)

## [0.0.3](https://github.com/OxideAV/oxideav-evc/compare/v0.0.2...v0.0.3) - 2026-06-15

### Other

- §7.3.8.2 xFirstCtb derivation (coding_tree_unit preamble, lines 2620-2623)
- §7.3.8.2 NumHmvpCand=0 reset (lines 2624-2625) keyed on xFirstCtb
- §7.3.8.1 multi-tile slice_data() walk (walk_baseline_idr_slice_tiled)
- §7.3.8.1 multi-tile CTU-iteration order (resolve_slice_tile_walk_order)
- §7.3.4 entry points + §7.4.5 eq. (88)/(89) tile subsets; errata-#97 reconciliation + §8.9.8 tableNum==0 pins
- §7.4.5 eq. (78)-(82) slice-tile resolution + arbitrary-slice parser fixes
- §6.5.1 eq. (32) TileIdToIdx + FirstCtbAddrTs + luma-sample tile extents
- §6.5.1 eq. (28)-(31) CTB-address conversion + TileId[] (errata #97)
- §6.5.1 ColBd (eq. 26) + RowBd (eq. 27) tile-boundary derivations
- §6.4.1 base neighbouring-block availability derivation
- §6.4.3 MV-candidate + §6.4.4 ALF availability derivations
- §6.5.1 ColWidth (eq. 24) + RowHeight (eq. 25) tile-extent derivations
- §6.5.3 inverse scan order (eq. 34) + §6.5.2 public surface
- §6.4.2 availLR derivation (eq. 23) + LR_xx tokens
- §6.5.1 tile-grid iterator + §7.4.3.2 picture-tile counters
- §8.5.2.3.10 MV prediction redundancy check
- §8.5.2.3.9 entry-process signed POC scaling primitives
- §8.5.2.3.9 bipred MMVD offset distribution (eqs. 591-616)
- §7.4.7 MMVD distance / sign / offset derivation + §9.3.4 ctxInc
- §8.5 AMVR (Adaptive Motion Vector Resolution) helper trio
- derive_dra_chroma_state_for_sps SPS adapter (joined + unjoined dispatch)
- §7.4.3.1 page-67 "Otherwise" identity ChromaQpTable + SPS->table adapter
- §7.4.3.1 SPS-signalled ChromaQpTable (eq. 74) parse + populate

### Round 309 — §7.3.8.2 `xFirstCtb` derivation (coding_tree_unit preamble)

#### Added
- `slice_data::derive_x_first_ctb(ctb_addr_in_rs, ctb_addr_rs_to_ts,
  tile_id, tile_index_maps, ctb_addr_ts_to_rs, pic_width_in_ctbs_y,
  ctb_log2_size_y)` — the §7.3.8.2 `coding_tree_unit( )` opening
  derivation (lines 2620-2623):

  ```text
  tileIndex      = TileIdToIdx[ TileId[ CtbAddrRsToTs[ CtbAddrInRs ] ] ]
  firstCtbAddrRs = CtbAddrTsToRs[ FirstCtbAddrTs[ tileIndex ] ]
  xFirstCtb      = ( firstCtbAddrRs % PicWidthInCtbsY ) << CtbLog2SizeY
  ```

  Round 305 wired the `xCtb == xFirstCtb` `NumHmvpCand = 0` reset by
  passing `xFirstCtb` from the caller (single-tile raster walk
  hard-codes 0; multi-tile walk reads the segment's first CTU). This
  closes the preamble itself by consuming the §6.5.1 maps the spec names
  — `CtbAddrRsToTs[ ]` (eq. 28), `TileId[ ]` (eq. 30), `TileIdToIdx[ ]`
  / `FirstCtbAddrTs[ ]` (eq. 32) and `CtbAddrTsToRs[ ]` (eq. 29), all
  already built in `crate::pps`. The derived `xFirstCtb` equals the
  multi-tile walk's segment shortcut by construction (a segment's first
  raster CTU **is** `CtbAddrTsToRs[ FirstCtbAddrTs[ tileIndex ] ]`),
  pinned by a full-grid cross-check.

#### Notes
- Opt-in posture (same as the round-218 onward helper rollout): a pure
  function returning an owned value, no behaviour change to existing
  decoder paths. Rebinding `walk_baseline_idr_slice_tiled` to derive
  `xFirstCtb` through this helper (rather than the segment shortcut) is
  the natural consumer follow-up.
- Malformed slice/PPS combinations are rejected, not panicked:
  out-of-range `CtbAddrInRs`, out-of-range tile-scan address, a
  `TileId` naming no tile, a `tileIndex` out of `FirstCtbAddrTs`
  range, and `PicWidthInCtbsY == 0`.

#### Tests
- 8 new unit tests (681 total; was 673): single-tile left-column pin;
  3×2-grid multi-tile hand-trace (10 raster addresses → tile-column
  luma edges 0/64/128); `CtbLog2SizeY` 6 scaling; full-grid agreement
  with the `walk_baseline_idr_slice_tiled` segment shortcut; explicit
  sparse-tile-ID resolution through `TileIdToIdx[ ]` (errata #97
  indexing); and three malformed rejections.

### Round 305 — §7.3.8.2 `NumHmvpCand = 0` reset

#### Added
- `SliceWalkStats::hmvp_resets` — count of §7.3.8.2 (lines 2624-2625)
  history-based MV predictor list resets. The reset fires for every CTB
  whose luma column equals its tile's first-CTB column
  (`xCtb == xFirstCtb`, the leftmost CTB of each CTB row within each
  tile), so HMVP candidates never cross a row or tile boundary. For a
  single-tile slice this equals `PicHeightInCtbsY`.

#### Changed
- `slice_data::walk_single_ctu` now takes `x_first_ctb` (the tile's
  first-CTB luma column, §7.3.8.2 line 2623) and bumps `hmvp_resets`
  when `xCtb == x_first_ctb`. The single-tile raster walker passes 0
  (tile anchored at the picture origin); the multi-tile walker derives
  each tile's `x_first_ctb` from its own `firstCtbAddrRs`, so a tile not
  at the picture's left edge resets on its own column. No bitstream
  syntax is consumed by the reset.

### Round 298 — §7.3.8.1 multi-tile `slice_data()` walk

#### Added
- `slice_data::walk_baseline_idr_slice_tiled(rbsp, inputs, order,
  subset_ranges)` — the §7.3.8.1 multi-tile `slice_data()` walk: drives
  the per-CTU CABAC walk off a round-292 `SliceTileWalkOrder`. For each
  tile in `SliceTileIdx[ ]` order it builds a fresh `CabacEngine` over
  that tile's §7.4.5 eq. (88)/(89) coded subset (§9.3.1 engine restart
  at the first CTU of each tile), walks the tile's `CtbAddrInRs` CTUs,
  and consumes `end_of_tile_one_bit`; `byte_alignment( )` between tiles
  is the subset boundary. The single-tile order reduces to the existing
  raster walk bit-for-bit.
- `SliceWalkStats::end_of_tile_bits` / `tile_byte_alignments` — the
  §7.3.8.1 per-tile terminate count and inter-tile alignment count.

#### Changed
- `slice_data::walk_baseline_idr_slice` shares the new internal
  `walk_single_ctu` per-CTU body with the multi-tile walk and records
  `end_of_tile_bits = 1`; behaviour is otherwise unchanged.

### Round 292 — §7.3.8.1 multi-tile CTU-iteration order

#### Added
- `slice_data::resolve_slice_tile_walk_order(slice_tile_idx,
  first_ctb_addr_ts, num_ctus_in_tile, ctb_addr_ts_to_rs)` — the
  §7.3.8.1 `slice_data()` CTU-iteration order (line-2596 syntax table)
  as a pure function. For each slice tile `i`, it walks
  `NumCtusInTile[ SliceTileIdx[ i ] ]` tile-scan CTU addresses from
  `FirstCtbAddrTs[ SliceTileIdx[ i ] ]`, mapping each through
  `CtbAddrTsToRs[ ]` to the raster `CtbAddrInRs` the
  `coding_tree_unit( )` consumes. This is the multi-tile backbone the
  round-2 single-tile raster walker generalizes to.
- `slice_data::SliceTileWalkOrder` — the per-slice result, one
  `SliceTileWalkSegment` per tile (in §7.3.8.1 `i` order) with
  `total_ctus()` and `ctb_addr_in_rs_flat()` views.
- `slice_data::SliceTileWalkSegment` — one tile's contribution: its
  geometric `tile_idx`, `first_ctb_addr_ts`, `num_ctus`, the ordered
  `ctb_addr_in_rs` raster sequence, and `byte_align_after` (true for
  every tile but the last, pinning the §7.3.8.1 `byte_alignment( )`
  that trails each non-final `end_of_tile_one_bit`).
- Malformed slice/PPS combinations are rejected (out-of-range
  `SliceTileIdx[ i ]`; a `FirstCtbAddrTs + NumCtusInTile` overrun of
  `CtbAddrTsToRs[ ]`) rather than panicking.

8 new unit tests (663 total; was 655): single-tile raster identity,
3×2-grid full-picture hand-trace, sub-rectangle two-tile walk,
single-tile-vs-raster cross-check, an end-to-end pin driving the walk
from the round-281 §7.4.5 `SliceTileIdx[ ]` derivation, the two
malformed rejections, and an empty-slice defensive case.

### Round 284 — §7.3.4 entry points + eq. (88)/(89) tile subsets; errata-#97 / §8.9.8 pins

#### Added
- `slice_header::parse_entry_point_offsets(br, num_tiles_in_slice,
  tile_offset_len_minus1)` — the §7.3.4 `entry_point_offset_minus1[ i ]`
  loop at the tail of `slice_header( )` (present when
  `single_tile_in_slice_flag == 0`): `NumTilesInSlice − 1` elements of
  `tile_offset_len_minus1 + 1` bits. Rejects
  `tile_offset_len_minus1 > 31` (§7.4.3.2 range).
- `slice_header::compute_tile_subset_byte_ranges(entry_point_offset_minus1,
  slice_data_len)` — §7.4.5 eq. (88)/(89): the per-tile subset byte
  ranges of the coded slice data, returned half-open
  (`end = lastByte + 1`), with the prefix sum carried in `u64` so
  32-bit offsets cannot wrap. A subset overrunning the data or an
  empty trailing subset errors (each subset must carry one tile's
  coded CTU bits).
- `SliceHeader::parse_entry_points(...)` — dispatch consuming the
  entry-point loop off the reader `parse_consume` leaves positioned
  past `slice_cr_qp_offset`, deriving `NumTilesInSlice` via the
  round-281 eq. (78)/(80) chain; a no-op (no bits read) for
  `single_tile_in_slice_flag == 1` slices.

#### Tests
- **Errata-#97 ↔ §7.4.5 reconciliation:** a §7.4.3.2-conformant
  explicit-tile-ID PPS (sparse IDs strictly increasing along the
  raster flat index `j * cols + i`) drives the full chain — eq. (30)
  `TileId[ ]` → eq. (32) `TileIdToIdx[ ]`/`FirstCtbAddrTs[ ]` →
  eq. (78)/(79) rectangular and eq. (81)/(82) arbitrary slice-tile
  resolution. A transposition guard pins that the rejected
  §7.4.3.2 first-sentence reading (column-major flat-packing) is
  observably different on a non-square grid.
- **Synthetic bitstream walks:** rectangular and arbitrary multi-tile
  slice headers followed by their entry-point loops, parsed
  end-to-end (`parse_consume` + `parse_entry_points`) and resolved to
  eq. (88)/(89) subset ranges; a single-tile-slice no-op pin;
  malformed-range rejections (overrun, empty last subset, zero-length
  data, `u32::MAX` wrap).
- **§8.9.8 `tableNum == 0` pins:** the spec's page-308 branch sentence
  leaves `qpDraFracAdj` / `draChromaQpShift` underivable by literal
  reading and the staged errata file still carries no §8.9.8 entry
  (docs task #1278), so three tests pin the documented in-tree
  reading via its closed form — `chromaScale = (scaleDra *
  QpScale[Clip3(0, 24, shift + 12)] + (1 << 17)) >> 18` with
  `qpDraInt = 2 * IndexScaleQP − 61` — at two `ScaleQP` knots
  (`scaleDraNorm` 724 and 1448, both chroma components), plus a
  branch-seam continuity check one norm step past the knot.
- 15 new unit tests (655 total; was 640).

### Round 281 — §7.4.5 eq. (78)-(82) slice-tile resolution over `TileIdToIdx[ ]`

#### Added
- `slice_header::SliceTileDims` — the §7.4.5 eq. (78) outputs
  `numTileRowsInSlice` / `numTileColumnsInSlice` / `NumTilesInSlice`
  for a rectangular (`arbitrary_slice_flag == 0`) slice.
- `slice_header::compute_slice_tile_dims(first_tile_id, last_tile_id,
  tile_index_maps, num_tile_columns_minus1, num_tiles_in_pic)` —
  §7.4.5 eq. (78) verbatim, including both wrap arms (bottom edge:
  `lastTileIdx < firstTileIdx` adds `NumTilesInPic`; right edge:
  `firstTileColumnIdx > lastTileColumnIdx` adds one tile row).
  `deltaTileIdx` is carried in `i64` because the row-wrap path makes
  it transiently negative. Tile IDs that name no picture tile error.
- `slice_header::compute_slice_tile_indices(...)` — §7.4.5 eq. (79):
  the `SliceTileIdx[ cIdx ]` walk, with the row-loop-head
  `tileIdx % NumTilesInPic` bottom wrap and the inner-row
  `currTileIdx − (num_tile_columns_minus1 + 1)` right wrap.
- `slice_header::compute_num_tiles_in_slice_arbitrary(...)` —
  §7.4.5 eq. (80) (`num_remaining_tiles_in_slice_minus1 + 2`).
- `slice_header::compute_slice_tile_indices_arbitrary(first_tile_id,
  delta_tile_id_minus1, tile_index_maps)` — §7.4.5 eq. (81)/(82): the
  running `sliceTileId[ i ]` chain resolved through
  `TileIdToIdx[ ]`. (The printed eq. (82) reads "liceTileIdx" — a
  dropped leading character of `SliceTileIdx`.)
- `SliceHeader::num_tiles_in_slice` / `SliceHeader::slice_tile_indices`
  — instance dispatch selecting the rectangular (eq. 78+79) or
  arbitrary (eq. 80-82) derivation from the parsed header, consuming
  `pps::TileIndexMaps` (round-278 §6.5.1 eq. 32) + `NumTilesInPic`.
- `SliceHeader::delta_tile_id_minus1` — the §7.4.5
  `delta_tile_id_minus1[ i ]` list is now surfaced (previously parsed
  and discarded).

#### Fixed
- **Arbitrary-slice header mis-parse:** the §7.3.4 delta loop runs
  `i < NumTilesInSlice − 1` with `NumTilesInSlice =
  num_remaining_tiles_in_slice_minus1 + 2` (eq. 80) — i.e.
  `minus1 + 1` entries. The parser previously read only `minus1`
  deltas, shifting every field after the tile block by one ue(v) for
  any `arbitrary_slice_flag == 1` slice.
- `last_tile_id` is now inferred equal to `first_tile_id` when not
  present (§7.4.5), instead of defaulting to 0 — single-tile and
  arbitrary slices previously surfaced a bogus `last_tile_id = 0`.
- `num_remaining_tiles_in_slice_minus1` is now bounded by
  `NumTilesInPic − 1` per §7.4.5 (rejects malformed headers instead
  of reserving unbounded memory).

### Round 278 — §6.5.1 eq. (32) `TileIdToIdx[ ]` + `FirstCtbAddrTs[ ]` + luma-sample tile extents (§6.5.1 complete)

#### Added
- `pps::TileIndexMaps` — the two §6.5.1 eq. (32) outputs.
  `tile_id_to_idx` carries the spec's *set* `TileIdToIdx[ tileId ]`
  as `(tileId, tileIdx)` pairs in tile-scan first-encounter order
  (explicit tile IDs are sparse, so a dense tileId-indexed list would
  be unbounded); `first_ctb_addr_ts` is the list
  `FirstCtbAddrTs[ tileIdx ]` for `tileIdx` in
  `0 ..= NumTilesInPic − 1`. `TileIndexMaps::tile_idx_for_id` is the
  `TileIdToIdx[ tileId ]` lookup (`None` when the ID names no tile).
- `pps::compute_tile_index_maps(tile_id) -> TileIndexMaps` — §6.5.1
  eq. (32). Verbatim port of the single
  `tileStartFlag` / `tileEndFlag` walk over `TileId[ ctbAddrTs ]`:
  each run start records the set entry + the first-CTB address, each
  run end advances `tileIdx`. A malformed `TileId[ ]` with a repeated
  non-contiguous ID follows the loop's assignment semantics verbatim
  (later run overwrites the set entry, still appends its own
  `FirstCtbAddrTs` slot); spec-derived inputs cannot hit that branch
  because eq. (28)-(30) pack each tile contiguously in tile scan.
- `pps::compute_column_width_in_luma_samples(col_widths,
  ctb_log2_size_y) -> Vec<u32>` /
  `pps::compute_row_height_in_luma_samples(row_heights,
  ctb_log2_size_y) -> Vec<u32>` — the §6.5.1 trailing
  `ColumnWidthInLumaSamples[ i ] = ColWidth[ i ] << CtbLog2SizeY` /
  `RowHeightInLumaSamples[ j ] = RowHeight[ j ] << CtbLog2SizeY`
  derivations. The shift saturates to `u32::MAX` on overflow.
- `Pps::tile_index_maps` / `Pps::column_width_in_luma_samples` /
  `Pps::row_height_in_luma_samples` instance methods — dispatch into
  the free functions over this PPS's derived lists. `PicWidthInCtbsY`
  / `PicHeightInCtbsY` / `CtbLog2SizeY` stay explicit caller
  arguments (all derive from §7.4.3.1 against the SPS).

#### Notes
- This closes out §6.5.1 entirely: every list the subclause derives
  (eq. 24-32 plus the two luma-sample lists) now has a pure helper +
  `Pps` dispatch. The slice-header consumers — §7.4.5's
  `ctbAddrInTs = FirstCtbAddrTs[ SliceTileIdx[ i ] ]` walk and the
  §7.4.3.4 `TileIdToIdx[ first_tile_id ]` slice-tile resolution — are
  the natural next arc.
- 12 new unit tests (620 total; was 608): eq. (32) single-tile +
  2×2 `FirstCtbAddrTs` hand-trace + `NumCtusInTile[ ]` prefix-sum
  sweep + explicit-ID set round-trip + `TileId[ FirstCtbAddrTs ]`
  inversion sweep + empty-input defensive + malformed-repeat
  assignment-semantics pin; luma-sample hand-traces at CtbLog2SizeY
  5/6 + picture-coverage sweep + saturation pin; `Pps` dispatch
  agreement including the explicit-tile-ID path.

### Round 273 — §6.5.1 eq. (28)-(31) CTB-address conversion + `TileId[ ]` (errata #97 unblock)

#### Added
- `pps::compute_ctb_addr_rs_to_ts(col_widths, row_heights, col_bd,
  row_bd, pic_width_in_ctbs_y) -> Vec<u32>` — §6.5.1 eq. (28).
  Raster-scan→tile-scan CTB-address conversion. Consumes the eq.
  (24)/(25) extents and eq. (26)/(27) boundaries; output length is
  `PicSizeInCtbsY` and the result is a permutation of
  `0 ..= PicSizeInCtbsY − 1`. All arithmetic is saturating so a
  malformed extent list clamps rather than overflows.
- `pps::compute_ctb_addr_ts_to_rs(ctb_addr_rs_to_ts) -> Vec<u32>` —
  §6.5.1 eq. (29). Inverts the eq. (28) permutation.
- `pps::compute_num_ctus_in_tile(col_widths, row_heights) -> Vec<u32>`
  — §6.5.1 eq. (31). `NumCtusInTile[ tileIdx ] = ColWidth[ i ] *
  RowHeight[ j ]` in raster-tile order (`tileIdx = j * num_cols + i`).
- `pps::compute_tile_id(col_bd, row_bd, ctb_addr_rs_to_ts,
  pic_width_in_ctbs_y, explicit_tile_id) -> Vec<u32>` — §6.5.1 eq.
  (30). Builds `TileId[ ctbAddrTs ]`. Implicit branch assigns the
  linear `tileIdx`; the explicit branch reads `tile_id_val[ i ][ j ]`
  with `i` = column, `j` = row per the in-repo errata #97 (§7.4.3.2's
  first-sentence row/column words are transposed), indexing the flat
  §7.4.3.2 syntax-order table at `j * num_cols + i`.
- `Pps::ctb_addr_rs_to_ts` / `Pps::ctb_addr_ts_to_rs` /
  `Pps::num_ctus_in_tile` / `Pps::tile_id` instance methods —
  derive the four §6.5.1 lists from the parsed PPS and dispatch into
  the free functions. `PicWidthInCtbsY` / `PicHeightInCtbsY` stay
  explicit caller arguments (they derive from §7.4.3.1 against the
  SPS). `Pps::tile_id` selects the explicit branch automatically when
  `explicit_tile_id_flag` is set, feeding the parsed `tile_id_val`.

#### Notes
- The eq. (30) `tile_id_val[ i ][ j ]` index ordering that rounds 237
  / 249 / 270 deliberately deferred is now resolved by the in-repo
  errata `evc-errata-and-clarifications.md` #97: `i` is the column
  index, `j` the row index, matching the eq. (30) loop nest and
  §7.4.3.2's own uniqueness constraint.
- Wiring stance unchanged from the round-218 onward helper rollout:
  pure functions returning owned vectors, no behaviour change to
  existing decoder paths. The slice walker rebinds onto these once the
  per-tile addressing path is threaded through.
- These complete the §6.5.1 CTB-raster-and-tile-scanning chain
  (eq. 24-31); only eq. (32) `TileIdToIdx[ ]` / `FirstCtbAddrTs[ ]`
  remains, a follow-up that consumes `TileId[ ]`.

### Round 270 — §6.5.1 eq. (26) `ColBd[ ]` + eq. (27) `RowBd[ ]` tile-boundary derivations

#### Added
- `pps::compute_col_bd(col_widths) -> Vec<u32>` — §6.5.1 eq. (26).
  Pure module-level prefix-sum producing the `ColBd[ i ]` tile-column
  boundary list (CTB-column index of each tile-column edge) from the
  eq. (24) `ColWidth[ ]` list. Output length is `col_widths.len() + 1`
  per the spec's inclusive `0 ..= num_tile_columns_minus1 + 1` range:
  `ColBd[ 0 ] = 0`, final entry is the picture width in CTBs. The
  running sum uses `u32::saturating_add` so a malformed
  over-specified explicit-tile `ColWidth[ ]` clamps rather than
  overflows.
- `pps::compute_row_bd(row_heights) -> Vec<u32>` — §6.5.1 eq. (27).
  Symmetric prefix-sum producing the `RowBd[ j ]` tile-row boundary
  list from the eq. (25) `RowHeight[ ]` list.
- `Pps::col_bd(pic_width_in_ctbs_y: u32) -> Vec<u32>` — instance
  dispatch that feeds `Self::col_widths(pic_width_in_ctbs_y)` into
  `compute_col_bd`. `PicWidthInCtbsY` stays an explicit argument (it
  derives from §7.4.3.1 against the SPS, not the PPS).
- `Pps::row_bd(pic_height_in_ctbs_y: u32) -> Vec<u32>` — instance
  dispatch into `compute_row_bd`, symmetric.

#### Notes
- These are the next §6.5.1 primitives after round-249's
  `ColWidth[ ]` / `RowHeight[ ]` (eq. 24 / 25) and the inputs the
  eq. (28) `CtbAddrRsToTs[ ]` walk consumes via its
  `tbX >= ColBd[ i ]` / `tbY >= RowBd[ j ]` tile-locating tests.
- The contested eq. (30) `tile_id_val[ i ][ j ]` index ordering (docs
  gap #1470) and the §8.9.8 eq. 1398-1409 `tableNum == 0` branch (docs
  gap #1278) are both avoided: eq. (26) / (27) are pure prefix-sums
  over the extent lists and reach neither.
- Wiring stance unchanged from the round-218 / 223 / 229 / 232 / 237 /
  242 / 245 / 249 / 258 / 263 helper rollout: pure functions
  returning owned vectors, no behaviour change to existing decoder
  paths.

#### Tests
- 11 new unit tests (595 total; was 584):
  - `round270_col_bd_single_tile_is_zero_and_full_width`
  - `round270_col_bd_prefix_sum_matches_hand_trace`
  - `round270_col_bd_length_is_widths_plus_one`
  - `round270_col_bd_final_entry_equals_total_width` (sweep over
    `(cols_minus1, PicWidthInCtbsY)`)
  - `round270_col_bd_is_strictly_monotonic_for_nonempty_tiles`
  - `round270_col_bd_explicit_branch_matches_widths`
  - `round270_col_bd_empty_widths_is_single_zero`
  - `round270_row_bd_prefix_sum_matches_hand_trace`
  - `round270_row_bd_final_entry_equals_total_height`
  - `round270_col_bd_matches_pps_dispatch`
  - `round270_row_bd_matches_pps_dispatch`

### Round 263 — §6.4.1 base neighbouring-block availability derivation

#### Added
- `neighbour::derive_neighbour_availability(x_nb_y, y_nb_y, pic_width_in_luma_samples, pic_height_in_luma_samples, neighbour_in_different_tile, is_coded) -> bool`
  — §6.4.1 single-block availability derivation. Returns
  `availableN` per the spec's six-bullet "if any condition holds,
  FALSE; otherwise TRUE" rule. The five geometric bullets
  (tile-different, two negative-index bounds, two picture-extent
  bounds) are evaluated from the explicit inputs; the one raster
  bullet (`IsCoded[][]` lookup) is taken as an already-looked-up
  boolean, in the same shape established by `derive_mv_candidate_availability`
  (§6.4.3) and `derive_alf_availability` (§6.4.4) in round 258.

#### Notes
- Completes the §6.4 single-block availability quartet alongside
  §6.4.2 (round 242), §6.4.3 (round 258) and §6.4.4 (round 258).
  Structurally: §6.4.1 = six bullets (geometry + `IsCoded[][]`);
  §6.4.3 = §6.4.1 + the intra/IBC neighbour bullet; §6.4.4 = §6.4.3
  − the tile-boundary bullet; §6.4.2 = packed token
  `availableL + availableR * 2` over two §6.4.1 outputs.
- Module-level rationale for the pre-resolved-predicate shape
  matches §6.4.3 / §6.4.4: the slice walker already has the tile
  map and `IsCoded[][]` raster on hand at the §6.4.1 call sites
  (this is exactly how §6.4.2 invokes §6.4.1 inline), so the
  derivation contract takes them as inputs rather than carrying
  the raster around inside the function.
- Wiring stance unchanged from the round-218 / 223 / 229 / 232 /
  237 / 242 / 245 / 249 / 258 helper rollout: pure function, no
  behaviour change to existing decoder paths.
- Round 263 does **not** rebind existing callers (the §6.4.1
  bullets remain inlined wherever the slice walker, the AMVP
  builder, the ALF classifier, etc. need them). A follow-up round
  can rebind them once the §6.4 helper set is exhaustively in
  place.

### Round 258 — §6.4.3 MV-candidate + §6.4.4 ALF neighbouring-block availability derivations

#### Added
- `neighbour::derive_mv_candidate_availability(x_nb_y, y_nb_y, pic_width_in_luma_samples, pic_height_in_luma_samples, neighbour_in_different_tile, is_coded, neighbour_is_intra_or_ibc) -> bool`
  — §6.4.3 single-block availability derivation. Returns
  `availableN` per the spec's seven-bullet "if any condition holds,
  FALSE; otherwise TRUE" rule. The five geometric bullets
  (tile-different, two negative-index bounds, two picture-extent
  bounds) are evaluated from the explicit inputs; the two raster
  bullets (`IsCoded[][]` lookup and intra/IBC prediction-mode flag)
  are taken as already-looked-up booleans, mirroring the §6.4.2
  contract.
- `neighbour::derive_alf_availability(x_nb_y, y_nb_y, pic_width_in_luma_samples, pic_height_in_luma_samples, is_coded, neighbour_is_intra_or_ibc) -> bool`
  — §6.4.4 single-block availability derivation. Structurally
  §6.4.3 minus the tile-boundary bullet: the ALF filter
  deliberately reaches across tile boundaries when
  §7.4.5 `alf_loop_filter_across_tiles_enabled_flag` permits it,
  so §6.4.4 never disqualifies a neighbour for sitting in a
  different tile. The flag itself is consulted by the ALF caller,
  not inside §6.4.4.

#### Notes
- These are the natural rounders for the round-242 §6.4.2 `availLR`
  work. §6.4.1 (the underlying single-block neighbouring
  availability) remains intentionally unwrapped: its bullet list
  mixes tile-boundary lookup, the `IsCoded[][]` raster, and the
  per-block prediction-mode flag that callers already have on
  hand. §6.4.3 and §6.4.4 take the same caller-on-hand booleans as
  inputs, exactly as the spec invokes them inline.
- Wiring stance unchanged from the round-218 / 223 / 229 / 232 /
  237 / 242 / 245 / 249 helper rollout: pure functions, no
  behaviour change to existing decoder paths.
- Round 258 does **not** rebind existing callers (the AMVP builder
  in `inter.rs` and the ALF classifiers in `alf.rs` inline their
  own per-bullet logic today). A follow-up round can rebind them
  once the §6.4 helper set is exhaustively in place.
- Docs gaps #1278 (§8.9.8 eq. 1398-1409 `tableNum == 0` branch)
  and #1470 (§6.5.1 eq. (30) vs §7.4.3.2 `tile_id_val` indexing
  contradiction) remain unresolved upstream; this round does not
  touch either.

#### Tests
- 18 new unit tests (573 total; was 555) plus 2 new doc-tests:
  - §6.4.3 (eight tests + one doc-test):
    - `round258_eq643_all_good_interior_is_available`
    - `round258_eq643_different_tile_disqualifies`
    - `round258_eq643_negative_coords_disqualify`
    - `round258_eq643_oob_picture_extent_disqualifies` (pins both
      the inclusive `>=` boundary and the strict-greater case)
    - `round258_eq643_uncoded_neighbour_disqualifies`
    - `round258_eq643_intra_or_ibc_neighbour_disqualifies`
    - `round258_eq643_each_bullet_independently_disqualifies`
      (baseline + per-bullet flip sweep)
    - `round258_eq643_origin_is_in_bounds`
  - §6.4.4 (eight tests + one doc-test):
    - `round258_eq644_all_good_interior_is_available`
    - `round258_eq644_does_not_consult_tile_boundary` (pins the
      defining §6.4.3-vs-§6.4.4 structural difference)
    - `round258_eq644_negative_coords_disqualify`
    - `round258_eq644_oob_picture_extent_disqualifies`
    - `round258_eq644_uncoded_neighbour_disqualifies`
    - `round258_eq644_intra_or_ibc_neighbour_disqualifies`
    - `round258_eq644_each_bullet_independently_disqualifies`
    - `round258_eq644_origin_is_in_bounds`
  - Structural contrast (two tests):
    - `round258_eq643_and_eq644_agree_when_same_tile` (ten-tuple
      sweep)
    - `round258_eq643_and_eq644_diverge_only_on_tile_bullet`

### Round 249 — §6.5.1 eq. (24) `ColWidth[ ]` + eq. (25) `RowHeight[ ]` tile-extent derivations

#### Added
- `pps::compute_col_widths(uniform_tile_spacing_flag, num_tile_columns_minus1, tile_column_width_minus1, pic_width_in_ctbs_y) -> Vec<u32>`
  — §6.5.1 eq. (24). Pure module-level function returning the
  `ColWidth[ i ]` list of length `num_tile_columns_minus1 + 1` in
  units of CTBs. Selects between the uniform branch (closed-form
  integer-division split of `PicWidthInCtbsY` across `n` columns)
  and the explicit branch (`tile_column_width_minus1[ i ] + 1` for
  `i < n − 1`, residual at column `n − 1`).
- `pps::compute_row_heights(uniform_tile_spacing_flag, num_tile_rows_minus1, tile_row_height_minus1, pic_height_in_ctbs_y) -> Vec<u32>`
  — §6.5.1 eq. (25). Symmetric to eq. (24) for tile rows.
- `Pps::col_widths(pic_width_in_ctbs_y: u32) -> Vec<u32>` —
  instance dispatch into `compute_col_widths` that pulls
  `uniform_tile_spacing_flag`, `num_tile_columns_minus1`, and the
  `tile_column_width_minus1` slice off the parsed PPS. The
  `PicWidthInCtbsY` argument stays explicit (it derives from
  §7.4.3.1 against the SPS, not the PPS).
- `Pps::row_heights(pic_height_in_ctbs_y: u32) -> Vec<u32>` —
  instance dispatch into `compute_row_heights`, symmetric.

#### Notes
- These are the pre-eq.(30) primitives the §6.5.1 `TileId[ ]` walk
  consumes. The contested `tile_id_val` index ordering in eq. (30)
  (docs gap #1470) is **not** touched: eq. (24) / (25) take only
  the uniform-spacing flag, explicit-widths/heights vectors, and
  picture CTB counts as inputs.
- The explicit branch saturates the running residual at `0` via
  `u32::saturating_sub` so a malformed bitstream that
  over-specifies `tile_column_width_minus1[ ]` /
  `tile_row_height_minus1[ ]` produces a clamped list rather than
  panicking. Callers should treat such a list as suspect.
- Wiring stance unchanged from the round-218 / 223 / 229 / 232 /
  237 / 242 / 245 rollout: pure functions returning owned vectors,
  no behaviour change to existing decoder paths.

#### Tests
- 13 new unit tests (555 total; was 542):
  - `round249_col_widths_single_tile_returns_full_picture`
  - `round249_col_widths_uniform_two_columns_even_split`
  - `round249_col_widths_uniform_three_columns_floor_division`
  - `round249_col_widths_uniform_covers_pic_width_exactly` (sweep
    over `(cols_minus1, PicWidthInCtbsY) ∈ [0, 8] × {1, 2, 5, 10,
    17, 32, 64, 100}`)
  - `round249_col_widths_explicit_branch_pins_eq24_remainder`
  - `round249_col_widths_explicit_branch_two_cols_residual`
  - `round249_col_widths_explicit_overflow_saturates_residual`
  - `round249_row_heights_uniform_two_rows_even_split`
  - `round249_row_heights_uniform_covers_pic_height_exactly`
  - `round249_row_heights_explicit_branch_pins_eq25_remainder`
  - `round249_compute_col_widths_uniform_matches_pps_dispatch`
  - `round249_compute_row_heights_uniform_matches_pps_dispatch`
  - `round249_col_widths_zero_pic_width_returns_all_zeros`

### Round 245 — §6.5.3 inverse scan order 1D array (eq. 34) + §6.5.2 public surface

#### Added
- `scan` module — new module hosting §6.5 scanning processes.
- `scan::zig_zag_scan(blk_w: usize, blk_h: usize) -> Vec<u32>` —
  §6.5.2 eq. (33). Builds the forward map
  `ScanOrder[ sPos ] = rPos` for an `(blk_w × blk_h)` transform
  block; entry `sPos` carries the row-major raster offset
  `y * blk_w + x` of the block sample visited at scan position
  `sPos`. The walk proceeds along anti-diagonals (lines of constant
  `x + y`) starting at the top-left corner: odd anti-diagonals run
  from the top-right endpoint toward the bottom-left, even
  anti-diagonals the opposite way.
- `scan::inverse_scan(blk_w: usize, blk_h: usize) -> Vec<u32>` —
  §6.5.3 eq. (34). Builds the inverse map
  `InvScanOrder[ rPos ] = sPos` by inverting the §6.5.2 forward
  permutation. By construction satisfies the two-way round-trip
  identity `inverseScan[ forwardScan[ pos ] ] = pos` and
  `forwardScan[ inverseScan[ pos ] ] = pos` for every legal `pos`.

#### Notes
- §7.4.3.1 (page 64) directs the decoder to build the
  `ScanOrder[ log2TbW ][ log2TbH ][ sPos ]` and
  `InvScanOrder[ log2TbW ][ log2TbH ][ rPos ]` arrays for every
  `(log2TbW, log2TbH) ∈ [1, MaxTbLog2SizeY]^2` by invoking §6.5.2
  / §6.5.3 with `blkWidth = 1 << log2TbW`, `blkHeight = 1 <<
  log2TbH`. The round-245 entry points are the building blocks of
  that population pass; the per-TB-size table cache is a follow-up.
- The `slice_data::decode_residual_coding_rle` walker keeps its
  in-module zig-zag builder for this round; rebinding it to
  `scan::zig_zag_scan` is a follow-up.

#### Tests
- 12 new unit tests (542 total; was 530):
  - `round245_zig_zag_4x4_matches_hand_trace`
  - `round245_zig_zag_2x2_matches_hand_trace`
  - `round245_zig_zag_4x2_non_square_matches_hand_trace`
  - `round245_zig_zag_is_permutation_for_every_tb_size`
  - `round245_zig_zag_visits_anti_diagonals_in_order`
  - `round245_zig_zag_empty_blocks_return_empty_vec`
  - `round245_inverse_scan_4x4_round_trips_forward`
  - `round245_inverse_scan_4x4_dc_at_raster_zero_is_scan_zero`
  - `round245_inverse_scan_is_bijection_with_forward`
  - `round245_inverse_scan_4x4_pins_eq34_values`
  - `round245_inverse_scan_4x2_pins_eq34_values`
  - `round245_inverse_scan_empty_blocks_return_empty_vec`

### Round 242 — §6.4.2 `availLR` derivation (eq. 23) + `LR_xx` tokens

#### Added
- `neighbour` module — new module hosting §6.4 neighbouring-block
  availability derivations.
- `neighbour::AvailLr` — `repr(u8)` enum carrying the §6.4.2
  eq. (23) integer in its discriminant. Four variants `Lr00 = 0`,
  `Lr10 = 1`, `Lr01 = 2`, `Lr11 = 3` matching the section's
  closing-paragraph token table. `Copy + Clone + Debug +
  PartialEq + Eq + Hash`.
- `neighbour::AvailLr::available_l(self) -> bool` /
  `neighbour::AvailLr::available_r(self) -> bool` — projections
  back to the eq. (23) input booleans (low bit and high bit
  respectively).
- `neighbour::AvailLr::as_u8(self) -> u8` /
  `neighbour::AvailLr::from_u8(u8) -> Option<Self>` — the eq. (23)
  integer view. `from_u8` rejects every value greater than 3; the
  spec never produces such a token.
- `neighbour::AvailLr::is_suco_consistent(self, sps_suco_flag: u8) -> bool`
  — the §6.4.2 closing-paragraph invariant: with `sps_suco_flag
  == 0`, `availLR ∈ { LR_00, LR_10 }`; otherwise every token is
  reachable.
- `neighbour::derive_avail_lr(available_l: bool, available_r: bool) -> AvailLr`
  — eq. (23) `availLR = availableL + availableR * 2`. Caller
  invokes §6.4.1 at the spec-mandated left
  (`xCurr − 1, yCurr`) and right (`xCurr + nCbW, yCurr`) luma
  locations and feeds the booleans in.

#### Notes
- §6.4.1 (neighbouring-block availability) is deliberately not
  wrapped this round. Its bullet list mixes tile-boundary lookup,
  the `IsCoded[][]` raster and the "intra / IBC mode" predicate —
  all of which already live on the slice walker. The §6.4.2
  entry point mirrors the spec's invocation pattern: callers
  compute `availableL` / `availableR` via §6.4.1 themselves.

#### Tests
- 11 new unit tests (530 total; was 519):
  - `round242_eq23_both_unavailable_is_lr00`
  - `round242_eq23_left_only_is_lr10`
  - `round242_eq23_right_only_is_lr01`
  - `round242_eq23_both_available_is_lr11`
  - `round242_projections_invert_derivation`
  - `round242_as_u8_matches_eq23_formula`
  - `round242_from_u8_round_trip`
  - `round242_from_u8_rejects_out_of_range`
  - `round242_suco_off_admits_only_lr00_and_lr10`
  - `round242_suco_on_admits_every_token`
  - `round242_discriminants_match_spec_table`

### Round 237 — §6.5.1 tile-grid iterator + §7.4.3.2 picture-tile counters

#### Added
- `pps::TileGridCoord` — typed value carrying the
  `(tile_idx, tile_row_j, tile_col_i)` triple from the §6.5.1 eq. (30)
  outer tile-enumeration loop. `Copy + Clone + Debug + PartialEq + Eq`.
- `pps::TileGridCoordIter` — `ExactSizeIterator` returned by
  `Pps::tile_grid_coords`; `size_hint` and `len` stay tight against
  the remaining-tile count.
- `Pps::num_tile_columns(&self) -> u32` /
  `Pps::num_tile_rows(&self) -> u32` — the
  `num_tile_*_minus1 + 1` adapter pair from §7.4.3.2. Always at
  least `1`, including the inferred `single_tile_in_pic_flag == 1`
  shape.
- `Pps::num_tiles_in_pic(&self) -> u32` — `NumTilesInPic` from §6.5.1.
  Product of the two axis counts; the parser's `MAX_TILES_PER_DIM`
  bound (256 per axis) keeps the product inside `u32`.
- `Pps::tile_grid_coords(&self) -> TileGridCoordIter` — the §6.5.1
  eq. (30) outer-loop iterator. Yields `tile_idx` linearly from
  `0` to `NumTilesInPic - 1`, advancing `i` (columns) within each
  `j` (row), with `tile_row_j * num_tile_columns + tile_col_i ==
  tile_idx` as the row-major packing identity.

#### Notes
- The explicit-id branch of eq. (30) reads `tile_id_val[ i ][ j ]`,
  but the §7.4.3.2 prose for `tile_id_val[ i ][ j ]` binds the first
  dimension to the row and the second to the column. These
  orderings contradict; `Pps::tile_grid_coords` therefore stops at
  the unambiguous `(tile_idx, tile_row_j, tile_col_i)` triple and
  does **not** surface `tile_id`. A docs-collaborator clarification
  will unblock the explicit-id resolution helper.

#### Tests
- 10 new unit tests (519 total; was 509):
  - `round237_num_tiles_single_tile_picture`
  - `round237_num_tiles_two_by_one_grid`
  - `round237_num_tiles_three_by_two_grid`
  - `round237_tile_grid_coords_single_tile`
  - `round237_tile_grid_coords_two_by_one_order`
  - `round237_tile_grid_coords_two_by_two_raster_order`
  - `round237_tile_grid_coords_three_by_two_full_walk`
  - `round237_tile_grid_iterator_exhausts_to_none`
  - `round237_tile_grid_iterator_size_hint`
  - `round237_tile_idx_matches_row_col_packing`

### Round 232 — §8.5.2.3.10 motion vector prediction redundancy check

#### Added
- `inter::MergeCand` — compact §8.5.2.3.x merge-candidate descriptor
  carrying `pred_flag_lX`, `ref_idx_lX`, `mv_lX` for X = 0, 1. The
  per-list refIdx / MV slots only carry meaning when the corresponding
  `pred_flag_lX` is set; the §8.5.2.3.10 predicate explicitly masks
  out inactive-list slots ("corresponding to available reference
  lists" qualifier).
- `inter::merge_cand_matches(a, b) -> bool` — §8.5.2.3.10 matching
  predicate. Compresses the spec's four ordered conditions ("number of
  available reference lists", "same available reference list
  indices", "same valid reference indices", "same motion vectors") to
  a single structural compare on the active-list-restricted
  projection. Reflexive + symmetric on its own; the inactive-list
  fields participate in neither equality test.
- `inter::merge_cand_redundancy_check(merge_cand_list,
  num_curr_merge_cand) -> Result<usize>` — §8.5.2.3.10 trim loop. When
  `numCurrMergeCand > 1`, scans `candIndx` from 0 against the tail
  entry at `numCurrMergeCand - 1`. Stops on first match (decrementing
  the count by 1, per the spec's exit clause), or walks every prior
  entry and returns the input count untouched. Pre-test no-op when
  `numCurrMergeCand ≤ 1` per the spec's outer "When numCurrMergeCand
  is greater than 1" guard. Caller-bug oversized counts surface
  `Error::Unsupported`.

#### Tests
- 16 new unit tests (509 total; was 493):
  - `round232_pred_flag_bitmask_mismatch_blocks_match`
  - `round232_ref_idx_mismatch_blocks_match`
  - `round232_mv_component_mismatch_blocks_match`
  - `round232_inactive_list_fields_are_ignored`
  - `round232_pre_test_no_op_when_count_le_1`
  - `round232_duplicate_tail_drops_count`
  - `round232_new_tail_preserves_count`
  - `round232_first_duplicate_short_circuits_scan`
  - `round232_penultimate_duplicate_decrements`
  - `round232_two_element_duplicate`
  - `round232_two_element_distinct`
  - `round232_bipred_full_match_drops_count`
  - `round232_bipred_l1_only_difference_preserves_count`
  - `round232_oversize_count_errors`
  - `round232_predicate_reflexive`
  - `round232_predicate_symmetric`

### Round 229 — §8.5.2.3.9 entry-process signed POC scaling primitives

#### Added
- `inter::MMVD_P_SAME_TARGET_SHIFT: i32 = 3` — §8.5.2.3.9 eqs. 547 /
  576 P-slice "same target ref" magnitude. Group 1 adds the constant
  to `mMvL0[0]`; group 2 subtracts it.
- `inter::mmvd_signed_dist_scale_factor(poc_diff_num, poc_diff_den)
  -> Result<i32>` — §8.5.2.3.9 eqs. 542 / 551 / 559 / 571 / 580 /
  588 signed POC-diff scaling factor `(num << 5) / den`. Operates on
  the **signed** POC differences directly (unlike round-223's
  `mmvd_dist_scale_factor` which pre-absolutes). Zero denominator
  surfaces `Error::Unsupported`. Widens to `i64` internally to keep
  the left-shift safe at the i32 boundary; the spec's domain is
  small POC differences so the cast back to i32 is lossless in
  practice.
- `inter::mmvd_signed_scale_component(dist_scale_factor, v) -> i32` —
  §8.5.2.3.9 eqs. 543 / 544 / 552 / 553 / 560 / 561 / 572 / 573 /
  581 / 582 / 589 / 590 round-toward-zero scaling of one MV
  component: `Clip3(-32768, 32767, Sign(s*v) * ((Abs(s*v) + 16) >>
  5))`. This is the **symmetric** half-up rounding form the entry
  process uses, distinct from the round-223 bipred-tail form
  `(s*v + 16) >> 5` that rides arithmetic right shift. Both forms
  agree at most operand pairs; we keep both literal to mirror the
  spec arithmetic.
- `inter::mmvd_signed_scale_mv(dist_scale_factor, mv) ->
  MotionVector` — both axes of `mmvd_signed_scale_component` with
  the same `distScaleFactor`, matching the eqs. (543, 544), (552,
  553), … pair structure.

#### Tests
- 11 new unit tests (493 total; was 482):
  - `round229_signed_dist_scale_factor_worked_examples`
  - `round229_signed_dist_scale_factor_rejects_zero_denominator`
  - `round229_signed_dist_scale_factor_zero_numerator`
  - `round229_signed_scale_component_symmetric_in_sign`
  - `round229_signed_scale_component_half_up_threshold`
  - `round229_signed_scale_component_clamps_to_signed16`
  - `round229_signed_scale_mv_applies_to_both_axes`
  - `round229_signed_scale_component_unit_factor_identity`
  - `round229_signed_scale_component_zero_inputs`
  - `round229_p_same_target_shift_constant_pinned`
  - `round229_signed_scale_agreement_at_zero_threshold` —
    documents that the entry-process and bipred-tail forms agree at
    the `|product| < 16` zero-threshold, where both reduce to 0.

#### Notes
- Wiring stance unchanged from rounds 187 / 193 / 195 / 201 / 207 /
  213 / 218 / 223: these helpers are opt-in. Baseline streams set
  `sps_mmvd_flag = 0` so `mmvd_flag` is inferred 0 per §7.4.7 and
  nothing in §8.5.2.3.9 executes. The full §8.5.2.3.9 entry process
  (`mmvd_group_idx ∈ { 1, 2 }` dispatch + `slice_type == B` / P
  sub-branches + `refIdxLX` / `predFlagLX` updates) still depends
  on the merge-candidate-list builder, `NumRefIdxActive[]`, and the
  populated `RefPicListX` arrays. Round 229 closes the arithmetic
  side: the signed POC-scale form is now a tested pure helper ready
  to drop into the eventual entry-process dispatcher.
- No docs gap. Eqs. 542 / 551 / 559 / 571 / 580 / 588 are the
  signed denominator form (zero is reached only by callers that
  already routed through a same-POC-distance fallback). Eqs. 543-
  590 spell the round-toward-zero scaling literally; we mirror it.

### Round 223 — §8.5.2.3.9 bipred MMVD offset distribution (eqs. 591-616)

#### Added
- `inter::mmvd_dist_scale_factor(abs_poc_num, abs_poc_den) -> Result<i32>`
  — §8.5.2.3.9 eq. 599 / 604 POC-difference scaling factor
  `(|num| << 5) / |den|`. Zero / negative denominator and negative
  numerator surface `Error::Unsupported` (the bipred caller always
  passes `Abs(currPocDiffL?)`).
- `inter::mmvd_apply_bipred_offset(mv_l0, mv_l1, mmvd_offset,
  curr_poc_diff_l0, curr_poc_diff_l1, pred_flag_l0, pred_flag_l1) ->
  Result<(MotionVector, MotionVector)>` — the tail of §8.5.2.3.9
  (eqs. 591-616). Resolves the bipred branch's three magnitude cases
  (`Abs(L0) ==/>/< Abs(L1)`), applies the opposite-POC-sign flip
  (eqs. 607-610) to `mMvdL1` only, handles the one-list-active
  "Otherwise" branch (eqs. 611-612), and accumulates `mMvLX += mMvdLX`
  with the shared `wrap16` semantics (eqs. 613-616). Pure helper:
  inputs are already-resolved POC diffs + offset + per-list flags;
  no `Sps` / `Slice` / merge-candidate-list state.

#### Internal
- `inter::clip_mvd_component(v)` — `Clip3(-32768, 32767, v)` for the
  scaled-MV components per eqs. 600 / 601 / 605 / 606.
- `inter::mmvd_scale_component(sf, mv)` — round-half-up POC scaling
  `Clip3(-32768, 32767, (sf * mv + 16) >> 5)` shared by eqs. 600 /
  601 / 605 / 606.

#### Tests
- 14 new unit tests (482 total; was 468):
  - `round223_dist_scale_factor_eq599_eq604_form`
  - `round223_dist_scale_factor_rejects_bad_inputs`
  - `round223_bipred_symmetric_same_magnitude_same_sign`
  - `round223_bipred_symmetric_opposite_sign_flips_l1`
  - `round223_bipred_l1_closer_scales_l0`
  - `round223_bipred_l0_closer_scales_l1`
  - `round223_bipred_l1_closer_opposite_sign_flips_l1_only`
  - `round223_one_list_active_l0_only`
  - `round223_one_list_active_l1_only`
  - `round223_rejects_both_lists_inactive`
  - `round223_accumulation_wraps_into_signed_16bit`
  - `round223_scaled_component_clips_to_signed16`
  - `round223_scale_component_eq600_form`
  - `round223_bipred_symmetric_property_offset_distribution`

#### Notes
- Spec typo recorded: eq. 601 (page 170) reads
  `Clip3( −32768, 32767 ( distScaleFactor * mMvdL1[ 1 ] + 16 ) >> 5 )`
  — the comma between `32767` and the inner expression is missing in
  the typeset PDF. Context-determined: the form is identical to
  eq. 600 on the y component.
- Wiring stance unchanged from rounds 218 / 213: the bipred-MMVD
  distribution helper is opt-in. The §8.5.2.3.9 entry process
  (eqs. 531-590) covers the `mmvd_group_idx ∈ { 1, 2 }` ref-list
  reassignment and a `slice_type == P` sub-branch, both of which need
  the merge-candidate list + populated `RefPicList0 / RefPicList1`
  arrays; threading that is the documented follow-up.

### Round 218 — §7.4.7 MMVD distance / sign / offset derivation + §9.3.4 ctxInc

#### Added
- `inter::MMVD_DISTANCE_IDX_MAX = 7` — §9.3.3 TR cMax for
  `mmvd_distance_idx`.
- `inter::MMVD_DIRECTION_IDX_MAX = 3` — §9.3.3 FL cMax for
  `mmvd_direction_idx`.
- `inter::MMVD_GROUP_IDX_MAX = 2` — §9.3.3 TR cMax for
  `mmvd_group_idx`.
- `inter::MMVD_MERGE_IDX_MAX = 3` — §9.3.3 TR cMax for
  `mmvd_merge_idx`.
- `inter::mmvd_distance(mmvd_distance_idx) -> Result<i32>` — Table 9
  lookup mapping `mmvd_distance_idx ∈ 0..=7` to
  `MmvdDistance ∈ { 1, 2, 4, 8, 16, 32, 64, 128 }`.
- `inter::mmvd_sign(mmvd_direction_idx) -> Result<(i32, i32)>` —
  Table 10 lookup mapping `mmvd_direction_idx ∈ 0..=3` to the
  axis-aligned `(MmvdSign[0], MmvdSign[1])`.
- `inter::mmvd_offset(mmvd_distance_idx, mmvd_direction_idx) ->
  Result<MotionVector>` — eqs. 133 + 134 combined; axis-aligned by
  construction.
- `inter::mmvd_flag_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the single `mmvd_flag` FL bit. Not in
  Table 96.
- `inter::mmvd_group_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `mmvd_group_idx` TR bins (cMax = 2).
- `inter::mmvd_merge_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `mmvd_merge_idx` TR bins (cMax = 3).
- `inter::mmvd_distance_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `mmvd_distance_idx` TR bins (cMax = 7).
- `inter::mmvd_direction_idx_ctx_inc(bin_idx) -> Result<usize>` —
  §9.3.4 positional ctxIdxInc for the `mmvd_direction_idx` FL bins
  (cMax = 3 ⇒ 2-bit FL code, two bin positions).

#### Tests
- 16 new unit tests (468 total; was 452):
  - `round218_mmvd_distance_table9_full_range`
  - `round218_mmvd_distance_rejects_oob_idx`
  - `round218_mmvd_sign_table10_full_range`
  - `round218_mmvd_sign_rejects_oob_idx`
  - `round218_mmvd_offset_eq133_eq134_spot_checks`
  - `round218_mmvd_offset_always_axis_aligned` — Cartesian-product
    8 × 4 axis-alignment property.
  - `round218_mmvd_offset_magnitude_equals_distance` — Cartesian-
    product magnitude property pins Table-10 unit-magnitude rows.
  - `round218_mmvd_offset_propagates_oob_distance_idx`
  - `round218_mmvd_offset_propagates_oob_direction_idx`
  - `round218_mmvd_flag_ctx_inc_positional`
  - `round218_mmvd_group_idx_ctx_inc_positional`
  - `round218_mmvd_merge_idx_ctx_inc_positional`
  - `round218_mmvd_distance_idx_ctx_inc_positional`
  - `round218_mmvd_direction_idx_ctx_inc_positional`
  - `round218_mmvd_worked_chain_dist3_dir2` — `(3, 2) ⇒ (0, 8)`.
  - `round218_mmvd_binarization_cmax_constants_match_spec` — pins
    the four cMax constants.

#### Notes
- Helpers are opt-in: the Baseline pipeline (`sps_mmvd_flag = 0`)
  treats `mmvd_flag` as inferred-0 per §7.4.7 and never reaches these
  helpers. A future Main-profile decode path threads `sps_mmvd_flag`
  from `Sps` + the parsed `mmvd_*_idx` values from the CABAC bitstream
  into them.
- The §8.5.2.3.9 "Derivation process for MMVD motion vector" (eqs.
  531–616) — the consumer that adds `MmvdOffset` to a selected merge
  candidate's `mvL0` / `mvL1` while POC-scaling across L0 / L1 — is
  the documented follow-up: it requires the merge-candidate list
  builder + ref-list threading.

### Round 213 — §8.5 AMVR (Adaptive Motion Vector Resolution) helper trio

#### Added
- `inter::AMVR_IDX_MAX = 4` — TR cMax constant (§9.3.3 binarization
  table) for the `amvr_idx[ x0 ][ y0 ]` syntax element.
- `inter::amvr_apply_to_mvd(mvd_component, amvr_idx) -> Result<i32>` —
  eq. 145: `MvdLX[…] = MvdLX[…] << amvr_idx`. `amvr_idx > 4` surfaces
  `Error::Unsupported` with a §-cited message.
- `inter::amvr_apply_to_mvd_vector(mvd, amvr_idx)` — vector form.
- `inter::amvr_round_mvp(mvp_component, amvr_idx) -> Result<i32>` —
  eq. 645/646: sign-symmetric magnitude round of the AMVP predictor
  (round-half-away-from-zero, symmetric on sign). Distinguishes
  itself from `round_motion_vector` (§8.5.3.10) at `mv = −2,
  amvr_idx = 2` (AMVR ↦ −4 ; affine ↦ 0).
- `inter::amvr_round_mvp_vector(mvp, amvr_idx)` — vector form.
- `inter::amvr_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `amvr_idx` TR bins. `amvr_idx` is
  **not** in Table 96; the ctxInc is purely positional (bin `k` →
  ctx `k`, both `initType` halves of Table 67 cover `0..3`).

#### Tests
- 15 new unit tests (452 total; was 437):
  - `round213_amvr_apply_to_mvd_zero_idx_identity`
  - `round213_amvr_apply_to_mvd_shift_examples`
  - `round213_amvr_apply_to_mvd_vector_both_axes`
  - `round213_amvr_apply_to_mvd_rejects_oob_idx`
  - `round213_amvr_round_mvp_zero_idx_identity`
  - `round213_amvr_round_mvp_sign_symmetric_at_idx2`
  - `round213_amvr_round_mvp_differs_from_affine_round_for_negatives`
    — pins the smoking-gun distinction vs `round_motion_vector` at
    `mv = −2, amvr_idx = 2`.
  - `round213_amvr_round_mvp_half_pel`
  - `round213_amvr_round_mvp_four_pel`
  - `round213_amvr_round_mvp_vector_both_axes`
  - `round213_amvr_round_mvp_rejects_oob_idx`
  - `round213_amvr_idx_ctx_inc_is_positional`
  - `round213_amvr_idx_ctx_inc_rejects_oob_bin`
  - `round213_amvr_baseline_pipeline_identity_at_idx0` — round-trip
    on Baseline pipeline (sps_amvr_flag = 0 ⇒ amvr_idx = 0 ⇒ all
    helpers no-op, `mv = mvp + mvd` unchanged).
  - `round213_amvr_worked_chain_at_idx2` — full Main-profile MV
    reconstruction at integer-pel resolution.

#### Notes
- Helpers are opt-in: the Baseline pipeline (sps_amvr_flag = 0)
  treats them as no-ops (`amvr_idx = 0` is identity for both shift
  and round). A future Main-profile decode path threads
  `sps_amvr_flag` from `Sps` + the parsed `amvr_idx` into them.

### Round 207 — `derive_dra_chroma_state_for_sps` SPS adapter (joined + unjoined dispatch)

#### Added
- `dra::derive_dra_chroma_state_for_sps(syntax, derived, cidx, sps)` —
  SPS-aware adapter that dispatches between the §8.9.7 unjoined path
  (`derived.joined_scale_flag == false` ⇒ `derive_dra_chroma_state`)
  and the §8.9.8 joined path (`derived.joined_scale_flag == true` ⇒
  `derive_dra_chroma_state_joined` with the SPS-active `ChromaQpTable`
  from round-201's `chroma_qp_table_for_sps`). `bit_depth_y` is pulled
  from `sps.bit_depth_y()`. Closes the SPS → §8.9.6 chroma chain at
  one call site, parallelling round-201's `chroma_qp_table_for_sps`
  for the table half. Opt-in helper; direct-invocation callers
  unchanged.

#### Tests
- 8 new unit tests (437 total; was 429):
  - `round207_for_sps_unjoined_matches_direct_unjoined` — unjoined
    path produces byte-identical `DraChromaDerived` to a direct
    `derive_dra_chroma_state` call (10-bit, 4:2:0).
  - `round207_for_sps_joined_matches_direct_joined` — joined path
    matches a direct `derive_dra_chroma_state_joined` invoked with
    the SPS-active `ChromaQpTable`.
  - `round207_for_sps_dispatches_on_joined_scale_flag` — same SPS,
    differing only in `syntax.dra_table_idx`, exercises both
    dispatch branches. Joined chromaScales[i] varies across i (per-
    range luma scales 256/512/1024); unjoined chromaScales[i]
    is constant `= dra_cb_scale_value`.
  - `round207_for_sps_uses_signalled_chroma_qp_table_on_joined_path`
    — when `sps.chroma_qp_table = Some(signalled_identity)`, the
    adapter feeds that table to the joined derive (NOT Table 5).
    Distinguishing assertion: identity ChromaQpTable produces
    materially different `chroma_scales` from the Table-5 default.
  - `round207_for_sps_propagates_zero_cb_scale_error_unjoined` —
    `dra_cb_scale_value = 0` surfaces verbatim from
    `derive_dra_chroma_state` (eq. 1386 guard).
  - `round207_for_sps_propagates_zero_cb_scale_error_joined` —
    same on the joined path (zero scale rejected by
    `derive_dra_chroma_state_joined` before eq. 1386 reciprocation).
  - `round207_for_sps_monochrome_synthesises_identity_on_joined_path`
    — monochrome SPS (`chroma_format_idc == 0`,
    `chroma_qp_table = None`): adapter synthesises the spec-page-67
    "Otherwise" identity table via `chroma_qp_table_for_sps` and the
    joined chain stays positive-definite.
  - `round207_for_sps_threads_bit_depth_y_from_sps` — same syntax /
    derived under 8-bit and 10-bit SPSes produces different §8.9.5
    top sentinels (`1 << 8` vs `1 << 10`).

#### Documented followups
- The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

#### Source
- ISO/IEC 23094-1:2020(E) §8.9.7 + §8.9.8, plus §7.4.3.1 page 67 for
  the round-201 `ChromaQpTable` dispatch this round consumes.

### Round 201 — §7.4.3.1 page-67 "Otherwise" identity `ChromaQpTable` + SPS → table adapter

#### Added
- `dra::default_chroma_qp_table_identity(bit_depth_chroma_minus8)` —
  builds the spec-page-67 "Otherwise" branch (`ChromaArrayType != 1`,
  `chroma_qp_table_present_flag == 0`) identity `ChromaQpTable` with
  `ChromaQpTable[m][qPi] = qPi`. Cb and Cr byte-for-byte equal.
  Indexed by `qPi ∈ [−QpBdOffsetC, 57]` with
  `QpBdOffsetC = 6 * bit_depth_chroma_minus8`; eq. 1403 / 1404
  clamping still applies on out-of-range lookups.
- `dra::chroma_qp_table_for_sps(sps) -> Result<ChromaQpTable>` —
  three-way dispatch that picks `sps.chroma_qp_table.clone()` when
  `Some`, else [`default_chroma_qp_table`] for `chroma_format_idc == 1`,
  else [`default_chroma_qp_table_identity`] for non-4:2:0. Lets
  §8.9.8 callers (including `derive_dra_chroma_state_joined`) consume
  a parsed `Sps` directly without re-implementing the dispatch.

#### Tests
- 12 new unit tests (429 total; was 417): identity-table values + Cb
  ≡ Cr property + 10-bit negative-`qPi` range + out-of-range clamping
  + bit-depth rejection + identity-vs-Table-5 differentiation, plus 6
  adapter tests (`Some` passthrough, 4:2:0 / Table 5, 4:2:0 / Table 6,
  monochrome / identity, 4:2:2 / identity, 4:4:4 / identity), plus an
  end-to-end test that chains the adapter into round-193's
  `derive_dra_chroma_state_joined` on a 10-bit monochrome SPS.

### Round 195 — §7.4.3.1 SPS-signalled `ChromaQpTable` (eq. 74) parse + populate

#### Added
- `dra::SignalledChromaQpTableParams { same_qp_table_for_chroma,
  global_offset_flag, tables }` — SPS chroma-QP-table body, one or
  two pivot-set rows depending on `same_qp_table_for_chroma`.
- `dra::SignalledChromaQpTablePivots { delta_qp_in_val_minus1[],
  delta_qp_out_val[] }` — one chroma component's pivot points.
- `dra::build_signalled_chroma_qp_table(params, bit_depth_chroma_minus8)`
  — eq. 74 + spec page 67–68 fill loops transcribed verbatim. Anchor
  at `qpInVal[0]`, down-fill below, per-segment linear interpolation,
  up-fill above. `same_qp_table_for_chroma == 1` aliases Cr := Cb.
  Rejects empty / mismatched-length / out-of-range pivots.
- `Sps::chroma_qp_table: Option<ChromaQpTable>` — populated when
  `chroma_qp_table_present_flag == 1` and `chroma_format_idc != 0`,
  `None` otherwise.

#### Changed
- `sps::parse` no longer discards `delta_qp_in_val_minus1[]` /
  `delta_qp_out_val[]`; threads them through the eq. 74 derivation.
- `num_points_in_qp_table_minus1[i]` bound tightened from a round-1
  placeholder (`> 64`) to the spec's page-67 bound
  `57 + QpBdOffsetC − (global_offset_flag == 1 ? 16 : 0)`.
- SPS test-helper `BitEmitter` gained a `se(i32)` method (signed 0-th
  order Exp-Golomb encoder, inverse of `BitReader::se`).

#### Documented followups
- `ChromaArrayType == 0` (monochrome) still leaves
  `Sps::chroma_qp_table = None`; spec page 67 "Otherwise" branch
  (`ChromaQpTable[m][qPi] = qPi`) needs a helper + consumer rewiring.
  Round 196 slot.
- `derive_dra_chroma_state_joined` still takes an externally-built
  `ChromaQpTable`; a small adapter that reads `Sps::chroma_qp_table`
  (else falls back to `default_chroma_qp_table`) would close the
  SPS → joined chroma-scale path end-to-end. Round 196 slot.
- Eq. 74 bracketing ambiguity (missing trailing `)`) noted on
  `build_signalled_chroma_qp_table`'s rustdoc; observable only in
  `qpOutVal[]` (informational), not in `ChromaQpTable[]` past the
  first pivot. Docs collaborator confirmation outstanding.

### Round 193 — §8.9.8 joined chroma-scale path (`DraJoinedScaleFlag = 1`) + default `ChromaQpTable` builder

#### Added
- `dra::SCALE_QP` (55 entries) — eq. 1420 transcribed verbatim. One
  extra trailing entry so eq. 1399's `ScaleQP[IndexScaleQP + 1]` is
  in-bounds at the top.
- `dra::QP_SCALE` (25 entries) — eq. 1421 transcribed verbatim.
- `dra::ChromaQpTableEntry { qp_bd_offset_c, table }` — flat-packed
  `ChromaQpTable[qPi]` for `qPi ∈ [−QpBdOffsetC, 57]`. `lookup(qpi)`
  applies eq. 1403 / 1404 `Clip3` clamping.
- `dra::ChromaQpTable { cb, cr }` — both chroma components' tables
  in one struct; `lookup(cidx, qpi)` dispatches by `ChromaIdx`.
- `dra::default_chroma_qp_table(sps_iqt_flag, bit_depth_chroma_minus8)`
  — Table 5 (`sps_iqt_flag = 0`) / Table 6 (`sps_iqt_flag = 1`)
  builder for the `chroma_qp_table_present_flag == 0` path on
  `ChromaArrayType == 1` (4:2:0). Both `cb` and `cr` identical
  byte-for-byte per spec page 67.
- `dra::chroma_scale_joined(luma_scale, dra_cb_scale_value,
  dra_cr_scale_value, cidx, dra_table_idx, chroma_qp_table) -> i64`
  — eq. 1395 → 1419 verbatim. Pure function. Handles both
  `tableNum == 0` and `tableNum != 0` branches; the
  `qpDraFracAdj < 0` fix-up (eq. 1410-1411); the eq. 1416 / 1417
  sign-dependent `draChromaScaleShiftFrac`; and the final
  eq. 1419 `(scaleDra * draChromaScaleShift + (1 << 17)) >> 18`.
- `dra::derive_dra_chroma_state_joined(syntax, derived, cidx,
  bit_depth_y, chroma_qp_table)` — full §8.9.7 derivation over the
  joined path. `chromaScales[i]` is per-range via
  `chroma_scale_joined(lumaScales[i], …)`; eq. 1386 reciprocates;
  eq. 1387 / 1389-1393 / top sentinel match the round-187 unjoined
  chain byte-for-byte (same downstream layout, so
  `chroma_scale_for_luma_sample` accepts joined state without
  modification).

#### Changed
- `dra::derive_dra_chroma_state`'s `Err(Unsupported)` message on a
  joined-flag state now points the caller at
  `derive_dra_chroma_state_joined` (the new round-193 entry) instead
  of saying "round 187 does not thread through".

#### Wiring stance
- The joined entry is independent of `apply_post_filters` / the
  legacy `dra::apply_dra` chroma path. Post-filter pipeline still
  uses the round-148 path so existing fixtures don't shift bit
  positions; the joined entry is opt-in.

#### Documented followups
- SPS-signalled `ChromaQpTable` parsing in `src/sps.rs:340-358` still
  discards `delta_qp_in_val_minus1` / `delta_qp_out_val`. Eq. 74
  pivot-point derivation needed to populate a `ChromaQpTable` on the
  parsed `Sps` so the joined consumer can pick the per-SPS table
  instead of the Table 5/6 default. Round 194 slot.
- `ChromaArrayType == 0` (monochrome) — spec page 67 "Otherwise" says
  `ChromaQpTable[m][qPi] = qPi`. `default_chroma_qp_table` does not
  currently take a `chroma_format_idc` argument; round 194 follow-up.
- Spec text ambiguity at the `tableNum == 0` branch of
  eq. 1398-1409 — literal text sets `qpDraFrac = 0` + `qpDraInt -= 1`
  but leaves `draChromaQpShift` undefined. Round 193 follows the
  parallel-structure reading (`qpDraIntAdj = 0`, eq. 1409 still
  applies); a docs collaborator should verify before any conformance
  fixture exercises the corner.

#### Tests
- 18 new unit tests (407 total; was 389) covering SCALE_QP / QP_SCALE
  literal spot-checks + monotonicity, Table 5 / 6 spot checks (Cb ==
  Cr identity, qPi tail behaviour), `ChromaQpTable` `Clip3` clamping
  at both ends, joined-path `chroma_scale_joined` positivity +
  monotonicity in `lumaScale` + pure-function property, end-to-end
  three-range joined derivation with per-range distinct
  `chroma_scales[]`, defensive `Err` paths on
  `joined_scale_flag == false` (caller routed to wrong entry), zero
  Cb scale, zero `dra_scale_value[i]`, empty-state no-op,
  `chroma_scale_for_luma_sample` against joined state (samples at
  distinct OutRangesC boundaries return distinct chroma scales — the
  signature distinguishing the joined path from the unjoined
  collapse), and a verification that the round-187 entry's error
  hint now mentions `derive_dra_chroma_state_joined`.

## [0.0.2](https://github.com/OxideAV/oxideav-evc/compare/v0.0.1...v0.0.2) - 2026-05-29

### Other

- §8.9.7 chroma DRA derived state + §8.9.6 chromaScale entry point (DraJoinedScaleFlag = 0)
- wire round-151 spec-faithful DRA state into §8.9.3 entry point
- §8.9.3 luma inverse mapping helpers + §7.4.7 InvLumaScales[0] docs gap
- §7.3.6-faithful dra_data() parser + §7.4.7 derivation
- §8.9.5 range-idx helper + per-sample co-located-luma chroma DRA offset
- §8.8.4.4 per-CTB chroma type filtering + eq. 1321 tap fix
- multi-APS cache indexed by adaptation_parameter_set_id
- §7.3.5 alf_data() rewrite + §8.9.4 AlfCoeffL + §8.8.4.2 classified luma apply
- derive §8.8.4.3 ALF transpose + classification filter index
- mask §8.9 ALF luma apply per CTB by the decoded §7.3.8.2 map
- decode §7.3.8.2 coding_tree_unit() ALF applicability map
- wire §7.3.8.5 cu_qp_delta into both IBC transform_unit() branches
- wire §7.3.8.5 cu_qp_delta into the non-skip inter coding_unit() path
- non-IDR (P/B) IBC coding_unit() wiring
- IBC coding_unit() branch wiring
- IBC pipeline composition + §8.5.3.10 MV rounding helper
- IBC primitive scaffold — §8.6 derivation + validation + integer-pel block copy
- ALF adaptive loop filter + DRA dynamic range adjustment
- spatial-neighbour MV grid AMVP + LTRP RPL resolution + flush() drain

### Round 187 — §8.9.7 chroma DRA derived state + §8.9.6 chromaScale entry point (DraJoinedScaleFlag = 0 path)

#### Added
- `dra::DraChromaDerived` — per-APS chroma DRA derived state for a
  single chroma component (`ChromaIdx::Cb` / `ChromaIdx::Cr`). Holds
  `out_ranges_c[]` (§8.9.7 eq. 1387 midpoints + §8.9.5 top sentinel),
  `chroma_scales[]` (eq. 1394), `inv_chroma_scales[]` (eq. 1386),
  `out_scales_c[]` (eq. 1391 / 1393), `out_offsets_c[]` (eq. 1389 /
  1392).
- `dra::ChromaIdx { Cb, Cr }` — explicit Cb/Cr selector with
  `.as_u32()` returning the §8.9.8 `cIdx ∈ {0, 1}`.
- `dra::derive_dra_chroma_state(syntax, derived, cidx, bit_depth_y)`
  — full §8.9.7 derivation (eq. 1386-1393) for a single chroma
  component on the `DraJoinedScaleFlag = 0` path (eq. 1394). Returns
  `Err(Error::Unsupported)` for `DraJoinedScaleFlag = 1` (joined
  table-driven path, deferred to a follow-up round). Defensive
  divide-by-zero guard on `dra_cb_scale_value == 0` /
  `dra_cr_scale_value == 0` (already forbidden by §7.4.7).
- `dra::chroma_scale_for_luma_sample(luma_sample, chroma_derived) ->
  i64` — §8.9.6 entry point: runs §8.9.5 against `out_ranges_c[]`
  with `numRanges = num_ranges_l + 1`, then transcribes eq. 1384
  (`incValue = lumaSample − OutRangesC[rangeIdx]`) and eq. 1385
  (`OutOffsetsC[rangeIdx] + ((OutScalesC[rangeIdx] * incValue +
  (1 << 9)) >> 10)`).
- `dra::DRA_MAX_RANGES_C = DRA_MAX_RANGES_V2 + 2 = 34` — storage
  size for `OutRangesC` including the §8.9.5 one-past-end top
  sentinel placed at `1 << bit_depth_y`.

#### Wiring stance
- The new entry point is independent of the legacy round-148
  `dra::apply_dra` chroma path used by `apply_post_filters`. Callers
  opt in explicitly. The post-filter pipeline stays on the round-148
  path so existing fixtures don't shift bit-positions in this round;
  a subsequent round can decide whether to retire the legacy path or
  thread §8.9.6 alongside it once a fixture with `sps_dra_flag = 1`
  + populated `dra_syntax_aps` is staged.

#### Tests
- 16 new unit tests (389 total; was 373):
  - `round187_chroma_derive_rejects_joined_path` — `dra_table_idx
    != 58` errors out.
  - `round187_chroma_derive_rejects_zero_cb_scale` — eq. 1386
    divide-by-zero defence on Cb path.
  - `round187_chroma_derive_rejects_zero_cr_scale` — eq. 1386
    divide-by-zero defence on Cr path.
  - `round187_chroma_derive_noop_on_empty_state` — `num_ranges == 0`
    returns empty state + `chroma_scale_for_luma_sample == 0`.
  - `round187_chroma_derive_eq1386_inv_chroma_scales_identity` —
    `dra_cb_scale_value = 512` (Q9 1.0) ⇒ `invChromaScales = 512`.
  - `round187_chroma_derive_eq1386_doubled_chroma_scale` —
    `dra_cb_scale_value = 1024` (Q9 2.0) ⇒ `invChromaScales = 256`.
  - `round187_chroma_derive_eq1387_out_ranges_c_midpoints` —
    eq. 1387 midpoints.
  - `round187_chroma_derive_eq1389_eq1391_out_scales_offsets_unjoined`
    — `OutScalesC[i] = 0` (deltaScale collapses to 0 under joined
    = 0); `OutOffsetsC[i] = invChromaScales[i−1]`.
  - `round187_chroma_derive_eq1392_eq1393_top_sentinel` —
    `i = numRangesL` final-pair.
  - `round187_chroma_derive_index_zero_layout_matches_spec` — the
    "respectively" line above eq. 1387 (`OutOffsetsC[0] =
    invChromaScales[0]`).
  - `round187_chroma_derive_cb_cr_distinct` — Cb scale = 256 / Cr
    scale = 1024 derive independently; `OutRangesC` is component-
    agnostic.
  - `round187_chroma_scale_for_sample_eq1384_eq1385_unjoined` —
    §8.9.6 on unjoined path collapses to `invChromaScales[0]`.
  - `round187_chroma_scale_constant_property_unjoined` —
    exhaustive 10-bit sample-space sweep verifying constancy.
  - `round187_chroma_scale_for_sample_handles_out_of_range_high` —
    §8.9.5 top sentinel handles `luma > last_internal_boundary`
    without panic.
  - `round187_chroma_derive_single_range_identity` — edge case
    `num_ranges_minus1 = 0` (empty eq. 1388-1391 loop body).
  - `round187_chroma_idx_as_u32` — Cb = 0, Cr = 1.

#### Documented followup
- §8.9.8 `DraJoinedScaleFlag = 1` (joined chroma-scale via
  `ChromaQpTable`): eq. 1395-1419 + `ScaleQP[54]` / `QpScale[25]`
  tables (eq. 1420 / 1421) + the SPS `ChromaQpTable` (per §7.4.3
  page 78). Implementation needs the `ChromaQpTable` threaded
  through from the SPS into either `DraDerived` or the §8.9.6 entry
  point's signature. Round 187 surfaces the joined path as
  `Err(Error::Unsupported)` so the caller can branch; the table
  transcription + integration is the suggested round-188 EVC slot.

#### Clean-room
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`);
  spec-only sourcing.

### Round 181 — wire round-151 spec-faithful DRA state into a §8.9.3 entry point

#### Added
- `EvcDecoder::apply_luma_inverse_mapping_spec_faithful(pic,
  pps_dra_aps_id) -> Result<bool>`: public entry point that closes the
  round-151 → round-174 → r181 chain. Looks up the
  `(DraSyntax, DraDerived)` pair populated by `parse_dra_syntax` +
  `derive_dra_state` for `pps_dra_aps_id`, runs the §7.4.7 off-by-one
  reconciliation (`dra::fill_inv_luma_scales_range_zero`) on a local
  clone so the cache stays as the literal spec reading, then applies
  `dra::apply_luma_inverse_mapping_u8` over the picture's luma plane.
  Returns `Ok(true)` when the cache slot was populated and the apply
  ran, `Ok(false)` when the slot was empty (clean no-op), and the
  reconciliation's `Err(Error::invalid)` when
  `dra_scale_value[0] == 0`.
- `EvcDecoder::dra_syntax_aps_for_pps(pps_dra_aps_id) -> Option<&
  (DraSyntax, DraDerived)>`: private accessor for the round-151
  cache, mirroring `dra_aps_for_pps`'s shape but strict on the
  `None` `pps_dra_aps_id` path (no `last_dra_aps_id` fallback — §8.9
  always invokes §8.9.3 with an explicit `pic_dra_aps_id`).

#### Wiring stance
- The new entry point is independent of the legacy round-148
  `dra::apply_dra` path used by `apply_post_filters`. Callers opt in
  explicitly. The post-filter pipeline stays on the round-148 path so
  existing fixtures don't shift bit-positions in this round; a
  subsequent round can decide whether to retire the legacy path or
  thread §8.9.3 alongside it once a fixture with `sps_dra_flag = 1`
  + populated `dra_syntax_aps` is staged.

#### Tests
- 6 new unit tests (373 total; was 367):
  - `round181_apply_luma_inv_map_empty_slot_is_noop` — empty cache
    slot returns `Ok(false)` and leaves the picture untouched.
  - `round181_apply_luma_inv_map_none_aps_id_is_noop` — strict
    `None` `pps_dra_aps_id` is a clean no-op.
  - `round181_apply_luma_inv_map_identity_scale_is_identity_on_8bit_codespace`
    — `dra_scale_value[0] = 512` at `dra_descriptor2 = 9` (Q9 1.0)
    yields the identity LUT on `[0, 255]`.
  - `round181_apply_luma_inv_map_doubled_scale_halves_midpoint` —
    `dra_scale_value[0] = 1024` produces a monotone non-decreasing
    mapping with `InvLumaScales[0] = 256`; input 240 maps to 120.
  - `round181_apply_luma_inv_map_zero_scale_value_propagates_error`
    — `dra_scale_value[0] == 0` errors out of the reconciliation;
    the picture stays untouched on the error path.
  - `round181_spec_faithful_path_is_orthogonal_to_legacy_apply_dra`
    — populating both `dra_aps[id]` and `dra_syntax_aps[id]` for
    the same id; the new entry point reads only the round-151
    cache.

#### Clean-room
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`);
  spec-only sourcing.

### Round 174 — §8.9.3 luma inverse mapping helpers + documented §7.4.7 docs gap

#### Added
- `dra::apply_luma_inverse_mapping(plane, derived, bit_depth_y)`: pure
  §8.9.3 luma inverse-mapping apply for a `u16` plane, transcribing
  ISO/IEC 23094-1:2020(E) eq. 1374 (`incrValue = InvLumaScales[apsId][rangeIdx]
  * lumaSample`), eq. 1375 (`mappedSample = (DraOffsets[apsId][rangeIdx]
  + incrValue + (1 << 8)) >> 9`), and eq. 1376 (`invLumaSample =
  Clip1Y(mappedSample)`). The range-index is selected via the round-148
  §8.9.5 `find_range_idx` helper against `derived.out_ranges_l`
  re-materialised as the §8.9.5 `rangesArray`. Covers 8 / 10 / 12-bit
  luma uniformly through the `u16` element type.
- `dra::apply_luma_inverse_mapping_u8(plane, derived)`: 8-bit shortcut
  that builds a 256-entry LUT once via the new
  [`build_inv_luma_lut_8bit`] and applies it to every sample.
- `dra::build_inv_luma_lut_8bit(derived)`: pre-computed
  `[u8; 256]` LUT for the 8-bit fast path. Returns an identity LUT
  when `derived.num_ranges == 0`.
- `dra::fill_inv_luma_scales_range_zero(derived, syntax)`: opt-in
  helper that fills `InvLumaScales[0]` and `DraOffsets[0]` by extending
  eq. 118 / 120 / 121 to `i = 0` — the off-by-one reconciliation of the
  §7.4.7 docs gap below. Rejects `dra_scale_value[0] == 0` as a
  defence-in-depth check.

#### Documented docs gap (§7.4.7 InvLumaScales[0] / DraOffsets[0])
- §7.4.7 (page 86) restricts the `InvLumaScales[apsId][i]` /
  `DraOffsets[apsId][i]` derivation (eq. 117-121) to
  `i ∈ [1, dra_number_ranges_minus1]`, leaving index 0 explicitly
  undefined. But §8.9.3 indexes both arrays with
  `rangeIdx ∈ [0, numOutRangesL − 1]` — including index 0. Under the
  literal spec reading, every sample falling in the lowest segment
  collapses to `(0 + 0*sample + 256) >> 9 = 0` regardless of bit depth
  — a degenerate behaviour the identity-DRA case rules out.
- Round 174 implements both interpretations side-by-side:
  - Default `derive_dra_state` (round 151) leaves index 0 at zero
    (literal spec).
  - `fill_inv_luma_scales_range_zero` reconciles the off-by-one
    (extends eq. 118 / 120 / 121 to `i = 0`).
- No wiring into the `apply_post_filters` pipeline yet. Both helpers
  are surfaced for the docs collaborator + a follow-up round to pick
  the resolution. This makes the gap empirically visible (a new
  `round174_literal_spec_range0_is_degenerate` test pins the
  symptom) without taking a stance on which reading is correct.

#### Tests
- 11 new unit tests (367 total; was 356):
  - `round174_literal_spec_range0_is_degenerate`: confirms the
    `InvLumaScales[0] = DraOffsets[0] = 0` literal-spec reading
    collapses range-0 samples to 0 under eq. 1375.
  - `round174_fill_inv_luma_range0_restores_non_degenerate`: with the
    off-by-one fill applied, range-0 samples produce a monotonic,
    bounded mapping (no flattening).
  - `round174_fill_range0_rejects_zero_scale`: defensive rejection of
    `dra_scale_value[0] == 0`.
  - `round174_fill_range0_noop_when_num_ranges_zero`: empty derived
    state is a no-op.
  - `round174_apply_eq1376_clips_to_bit_depth`: Clip1Y caps at
    `(1 << bit_depth_y) − 1`.
  - `round174_apply_eq1376_clips_negative_to_zero`: Clip1Y at the
    lower bound.
  - `round174_apply_u16_in_place`: end-to-end plane apply for the
    identity-Q18 case (every sample maps to itself).
  - `round174_apply_u8_in_place_matches_lut`: the 8-bit shortcut and
    the LUT path agree bit-for-bit.
  - `round174_apply_noop_when_num_ranges_zero`: both `u8` / `u16`
    helpers are no-ops on an empty derived state.
  - `round174_build_lut_identity_when_num_ranges_zero`: LUT builder
    returns an identity LUT on an empty derived state.
  - `round174_apply_multi_range_segment_dispatch`: a 2-range DRA with
    distinct per-range `InvLumaScales` / `DraOffsets` correctly
    dispatches each sample through eq. 1374-1376 per §8.9.5's
    `rangeIdx`, verified with hand-computed mapped values across the
    split and at the Clip1Y upper bound.

### Round 151 — §7.3.6-faithful `dra_data()` parser + §7.4.7 derivation

#### Added
- `dra::DraSyntax`: every raw bit `dra_data()` writes per ISO/IEC
  23094-1:2020(E) §7.3.6 (page 42) — `dra_descriptor1` u(4),
  `dra_descriptor2` u(4), `dra_number_ranges_minus1` ue(v),
  `dra_equal_ranges_flag` u(1), `dra_global_offset` u(10),
  `dra_delta_range[]` u(10), `dra_scale_value[]` u(numBitsDraScale),
  `dra_cb_scale_value` / `dra_cr_scale_value` u(numBitsDraScale),
  `dra_table_idx` ue(v).
- `dra::DraDerived`: per-APS state §7.4.7 mandates from the parsed
  bits — `num_bits_dra_scale` (eq. 111), `joined_scale_flag` (the
  `dra_table_idx == 58` branch), `in_dra_range[0..=num_ranges]`
  (eq. 112-114), `out_ranges_l[0..=num_ranges]` (eq. 115-116 then
  re-shifted per eq. 122), `inv_luma_scales[]` (eq. 117-119), and
  `dra_offsets[]` (eq. 120-121). Sized for `num_ranges` up to 32,
  matching the spec ceiling on `dra_number_ranges_minus1`.
- `dra::parse_dra_syntax(payload, bit_depth_y)`: §7.3.6 bitstream
  parser that also invokes §7.4.7 derivation. Rejects every §7.4.7
  bitstream-conformance violation with `Error::Invalid` (zero
  `numBitsDraScale`, `dra_number_ranges_minus1 > 31`,
  `dra_global_offset` outside `[1, Min(1023, (1<<BitDepthY) − 1)]`,
  `dra_scale_value` outside `[1, (4 << dra_descriptor2) − 1]`,
  `dra_table_idx > 58`, and `InDraRange[j] > (1 << BitDepthY) − 1`).
- `dra::derive_dra_state(syntax, bit_depth_y)`: standalone §7.4.7
  derivation — surfaced so a re-encoder can compute the derived
  state from a hand-constructed `DraSyntax` without round-tripping
  through the byte parser.
- `EvcDecoder::dra_syntax_aps`: 32-slot parallel cache (indexed by
  `adaptation_parameter_set_id`) of the new `(DraSyntax, DraDerived)`
  pair. Populated by the `send_packet` APS branch whenever an SPS
  has been parsed (so `BitDepthY` is known) — the legacy
  `dra::parse_dra_data` / `dra_aps[]` cache stays populated in
  parallel for round-148 `apply_dra` chroma-offset compatibility.

#### Tests
- 16 new `dra.rs` unit tests cover minimal-payload round-trip,
  `joined_scale_flag` polarity, equal-ranges delta distribution,
  unequal-ranges per-range delta, `BitDepthY ∈ {8, 10, 12}` shift
  arithmetic for `InDraRange[]`, every §7.4.7 conformance rejection
  path (empty payload, zero/overlarge scale_value, zero
  global_offset, overlarge table_idx, overlarge num_ranges_minus1,
  InDraRange overflow), `OutRangesL[]` post-eq.-122 recursion with
  identity Q4.9 scale, and `InvLumaScales[]` eq.-118 with identity
  + halved input scales.
- 1 new `decoder.rs` test verifies the parallel `dra_syntax_aps` cache
  slot stores a `(DraSyntax, DraDerived)` pair and round-trips its
  `dra_descriptor1` / `num_bits_dra_scale` / `joined_scale_flag`
  fields.

#### Notes
- The legacy `dra::DraData` / `dra::parse_dra_data` / `dra::apply_dra`
  chain stays untouched in this round — it does **not** match the
  §7.3.6 wire format (the round-11 parser made up its own
  `dra_descriptor_present_flag` / `dra_range_l[]` /
  `dra_chroma_qp_offset[]` shape). A follow-up round wiring §8.9.3
  luma inverse mapping (eq. 1374-1376) + §8.9.6 chroma scale (eq.
  1384-1385) will route through the new spec-faithful pair and
  retire the legacy types.

### Round 148 — §8.9.5 range-idx helper + per-sample co-located-luma chroma DRA offset

#### Added
- `dra::find_range_idx`: spec-faithful transcription of ISO/IEC
  23094-1:2020(E) §8.9.5 (page 305) eq. 1383 — the piecewise-function
  range-index identification process used by every §8.9.3 / §8.9.4 /
  §8.9.6 chroma scale lookup. Pure function: walks `rangesArray` from
  the bottom, returns the first `rangeIdx` whose upper boundary the
  input sample falls below; falls through to `numRanges − 1` for any
  input ≥ the top boundary, exactly matching the spec's `rangeFound`
  fallback + final `Min(rangeIdx, numRanges − 1)` clamp.
- `dra::build_ranges_array`: synthesises the §8.9.5-compatible
  `rangesArray` (`num_ranges + 1` entries) from the round-11
  `DraData::range_l` table (which stores `num_ranges` lower boundaries
  only) by appending `1 << bit_depth` as the top sentinel so the
  last-segment fall-through matches the spec.

#### Changed
- `dra::apply_dra` now applies the chroma offset **per chroma sample**
  rather than uniformly using segment 0's offset. Per §8.9.2 the
  §8.9.4 chroma DRA process takes `decPictureL[ x * SubWidthC, y *
  SubHeightC ]` (i.e. the **pre-DRA** decoded luma at the co-located
  position) as input; `apply_dra` now snapshots the pre-DRA luma plane
  before in-place LUT rewriting, then for each (x, y) chroma sample
  looks up the co-located luma's range index via §8.9.5 and applies
  the parsed `dra_chroma_qp_offset[ rangeIdx ]` (still a round-11
  simplification of the full §8.9.6 + §8.9.7 + §8.9.8 derivation
  chain, but no longer collapsed to segment 0). Supports 4:2:0, 4:2:2,
  4:4:4 via the per-format `SubWidthC` / `SubHeightC` mapping. Picture
  edges clamp the luma coordinate to `[0, luma_w − 1]` /
  `[0, luma_h − 1]` for the residual sub-sampled chroma rows / cols
  that overhang.

#### Tests
- 8 new unit tests (now 339, was 331): §8.9.5 eq. 1383 three-range
  walk (boundary + below-top + above-top fall-through), single-range
  always-zero, zero-ranges no-op, ranges-array top-sentinel synthesis
  for 8-bit + 10-bit, per-sample chroma offset uses co-located luma's
  segment (3-segment 8×8 4:4:4 with x-varying luma → distinct chroma
  offsets per column), the snapshot-correctness check (post-DRA luma
  re-classified into a sentinel-offset segment fails the test if the
  chroma lookup wrongly reads post-DRA luma), 4:2:0 subsampled-luma
  alignment (chroma row y reads luma row 2y), and the upper clip
  (`+60` over `chroma = 250` → `255` cap).

#### Spec-faithfulness notes
- The full §8.9.6 `chromaScale = OutOffsetsC[ apsId ][ cIdx ][ rangeIdx ]
  + ( OutScalesC[ ... ] * incValue + ( 1 << 9 ) ) >> 10` (eq. 1384-1385)
  with `OutScalesC` / `OutOffsetsC` derived per §8.9.7 + §8.9.8 from
  `invChromaScales` + `OutRangesL` is **still parked** pending the
  §7.3.6-faithful APS parser rewrite (the round-11 parser uses an
  approximate Q8.3 scale + 8-bit chroma-offset layout rather than the
  §7.3.6 `dra_global_offset` + `dra_scale_value` u(v) numBitsDraScale
  layout). Round 148 only lands the §8.9.5 helper + the co-located-
  luma per-sample chroma offset; the full chroma-scale derivation is a
  documented follow-up.

### Round 145 — §8.8.4.4 per-CTB chroma type filtering + eq. 1321 tap-geometry fix

#### Added
- `alf::apply_alf_chroma_masked`: per-CTB chroma type filtering process
  (§8.8.4.4 eq. 1321-1323), gated per-CTU by the `alf_ctb_chroma_flag` /
  `alf_ctb_chroma2_flag` side of the round-113 `AlfCtbMap`. The
  `ChromaArrayType == 3` path (where the per-CTB chroma map flags are
  decoded out of the bitstream) now drives the chroma apply per CTB
  instead of the round-126 whole-plane fallback. Mirrors
  `apply_alf_luma_masked`: pre-filter snapshot read, picture-edge clamp
  via `blkWidth / SubWidthC` / `blkHeight / SubHeightC` (§8.8.4.1 lines
  18105-18107).
- `EvcDecoder::apply_chroma_alf_masked_or_whole_plane` helper
  (`decoder.rs`, free fn): when the slice surfaced a non-empty
  `alf_ctb_chroma*` plane in the map, routes to
  `apply_alf_chroma_masked`; otherwise falls back to the whole-plane
  `apply_alf_chroma` (the `ChromaArrayType ∈ {1, 2}` path where the
  per-CTB chroma flags are always inferred 0). Threaded through both
  the luma-on and luma-off branches of `apply_post_filters`.
- 5 new unit tests (now 331, was 326): eq. 1321 tap correctness on a
  planted gradient via an independently-coded reference (catches
  tap-position permutations), per-CTB masked apply reproduces
  whole-plane apply bit-for-bit when every CTU is flagged, per-CTB
  apply touches only flagged CTUs and leaves the other plane
  untouched, partial-CTU edge clamp (24×24 4:2:0 → 12×12 chroma plane
  with a 4-column-wide right chroma CTU), and monochrome
  (`chroma_format_idc = 0`) is a no-op.

#### Fixed
- `alf::CHROMA_TAPS` / `CHROMA_TAPS_SYM`: round-11 had taps 3, 4, 5
  permuted vs §8.8.4.4 eq. 1321. The previous layout multiplied
  `coef[3]` (which the spec assigns to the `rec[x ∓ 1, y ± 1]`
  diagonal pair) against the `rec[x ± 2, y]` horizontal pair instead;
  similarly `coef[4]` was mapped to the `rec[x ± 1, y]` pair and
  `coef[5]` to the diagonal pair. No existing unit test exercised
  non-DC chroma coefficients, so this had been silent. Corrected in
  the same commit; the round-145 eq. 1321 reference test exhaustively
  pins the geometry going forward.

### Round 126 — Multi-APS cache indexed by `adaptation_parameter_set_id` (§7.4.2.3 / §7.3.4 / §8.9 routing)

#### Added
- `EvcDecoder::alf_aps`: 32-slot ALF APS cache (was `Option<AlfData>`),
  indexed by `adaptation_parameter_set_id` (5-bit, 0..=31). Replaces the
  round-11 last-APS-wins behaviour with the spec's update-by-id
  semantics. New `EvcDecoder::dra_aps` mirrors the same shape for DRA
  APS (selected by the PPS's `pic_dra_aps_id`).
- `EvcDecoder::alf_aps_for_slice(Option<u8>)` and `dra_aps_for_pps(Option<u8>)`:
  routing helpers that resolve the slice / PPS-referenced APS id against
  the new caches. Fall back to the most-recently-stored APS when the
  caller can't surface an id (e.g. the minimal-header IDR fixture path
  that doesn't decode a slice header).
- `SliceHeader::slice_alf_luma_aps_id`, `slice_alf_chroma_aps_id`,
  `slice_alf_chroma2_aps_id`: surfaces the §7.3.4 `slice_alf_*_aps_id`
  fields (formerly parsed into `_` discards). Each is `Option<u8>` —
  `Some(id)` exactly when the corresponding ALF idc / ChromaArrayType
  gating signals the APS id in the bitstream.
- `PostFilterInputs`: bundles the §8.9 + §8.10 inputs to
  `apply_post_filters` (ALF CTB map, chroma enables, three ALF APS ids,
  one DRA APS id) into one struct so the call site stays inside the
  `clippy::too_many_arguments` lint threshold.
- 5 new unit tests (now 326, was 321):
  `round126_alf_aps_ids_chroma444_three_apsids` (ChromaArrayType==3
  slice surfaces three distinct ALF APS ids),
  `round126_alf_aps_ids_unset_when_disabled` (no ids when
  `slice_alf_enabled_flag = 0`),
  `round126_alf_aps_cache_distinct_slots_resolve_independently` (two
  distinct ALF APS payloads at slots 3 and 19 resolve via their slice
  ids, with a fallback to the most-recently-cached slot when the slice
  doesn't surface one),
  `round126_dra_aps_cache_routes_via_pps_id` (DRA mirror of the same),
  `round126_aps_nal_writes_indexed_cache_slot` (APS NAL parse populates
  the cache slot named by its `adaptation_parameter_set_id`, leaving
  other slots intact).

#### Changed
- `EvcDecoder::send_packet` APS branch now writes the parsed ALF / DRA
  payload into the cache slot named by the APS NAL's
  `adaptation_parameter_set_id`, and records the id in
  `last_alf_aps_id` / `last_dra_aps_id` so the back-compat fallback
  has a referent.
- `apply_post_filters` resolves the luma / Cb / Cr ALF APS slots
  independently per the slice's three APS ids, so a 4:4:4 slice
  signalling `slice_alf_chroma_idc == 3` with distinct Cb / Cr APS
  ids pulls each chroma plane from its own cache slot (was: both
  chroma planes used the luma APS's `chroma_filters[0]`). Cb / Cr
  fallback chain: explicit chroma APS id → joint chroma APS id →
  luma APS — matching §7.4.5's "inference of same APS" intent.
- `NonIdrDecodeResult` adds `alf_luma_aps_id` / `alf_chroma_aps_id`
  / `alf_chroma2_aps_id` so the apply pass can route to the right
  slot per slice.

### Round 120 — Spec-faithful §7.3.5 alf_data() parser + §8.9.4 AlfCoeffL + §8.8.4.2 classified luma apply

#### Added
- `alf_tables.rs`: ISO/IEC 23094-1 §8.9.4 eq. 102 (`ALF_FIX_FILT_COEFF`, 64 fixed
  12-tap luma filter sets) and eq. 103 (`ALF_CLASS_TO_FILT_MAP`, 25×16
  class-to-filter mapping) transcribed verbatim from the spec PDF.
- `alf.rs::derive_alf_coeff_l`: §8.9.4 eq. 96-104 derivation. For every class
  `filtIdx = 0..24`, seeds `outCoef[ ]` from `ALF_FIX_FILT_COEFF[
  ALF_CLASS_TO_FILT_MAP[ filtIdx ][ alf_luma_fixed_filter_set_idx[ filtIdx ] ]
  ]` per eq. 98 when the fixed-filter usage flag is 1 (eq. 99 zero otherwise),
  adds the per-class delta `filterCoefficients[ alf_luma_coeff_delta_idx[
  filtIdx ] ][ coefPosMap[ j ] − 1 ]` for every j with `coefPosMap[ j ] > 0`
  (eq. 100), and computes position 12 per eq. 104 (`512 − Σ << 1`).
- `alf.rs::apply_alf_luma_classified` / `apply_alf_luma_classified_masked`:
  §8.8.4.2 eq. 1281-1288 per-sample luma apply. For every sample of every
  flagged CTB selects `f[ j ] = AlfCoeffL[ filtIdx[ x ][ y ] ][ j ]` from the
  per-CTB classification (round-117 `derive_alf_classification`), permutes via
  `transpose_luma_coeffs` per eq. 1282-1285, accumulates `sum = Σ
  filterCoeff[ k ] * (north + south) + filterCoeff[ 12 ] * recPicture[ x, y ]`
  per eq. 1286, then `((sum + 256) >> 9).clamp(0, max)` per eq. 1287/1288.
- `AlfData` fields: `luma_type_flag` (§7.3.5),
  `num_signalled_luma_filters`, `alf_luma_coeff_delta_idx[25]`,
  `alf_luma_fixed_filter_usage_flag[25]`, `alf_luma_fixed_filter_set_idx[25]`.
- `sps::tests::BitEmitter::uek(k, value)`: spec-aligned uek(v) writer used by
  the new ALF tests for round-tripping the §7.3.5 syntax through the parser.
- 12 new unit tests (321 total, was 309) — exercising the §7.3.5 syntax
  (7-tap vs 13-tap `alf_luma_type_flag`, `alf_luma_coeff_delta_idx` routing,
  `alf_luma_fixed_filter_usage_pattern == 1` every-class fixed-filter seed,
  `alf_luma_coeff_delta_prediction_flag` cumulative-sum prediction across
  signalled filters, eq. 104 DC invariant, chroma EG progression), the
  classified apply (uniform-plane identity, per-class routing on a
  vertical edge, CTB-mask gating, partial-edge-CTB clamp), and dimension /
  spot-check of `ALF_FIX_FILT_COEFF` / `ALF_CLASS_TO_FILT_MAP`.

#### Changed
- `parse_alf_data`: replaces the round-11 simplified u(6)/u(1) signalling with
  the full §7.3.5 syntax driven by uek(v) deltas and the §8.9.4 derivation.
- `apply_alf_luma` / `apply_alf_luma_masked` / `apply_alf_chroma`: now apply
  eq. 1286/1287 (spec scaling `(sum + 256) >> 9` with `coef[12]` /
  `coef[6]` multiplying the centre sample) instead of the round-11
  `(sum + 64) >> 7` constant-DC approximation. Coefficient ranges now match
  the spec — `-512..511` for spatial taps, `-1024..1023` for the centre.
- `decoder::apply_post_filters`: masked-luma ALF now invokes
  `apply_alf_luma_classified_masked` (§8.8.4.2 per-sample filter selection)
  in place of the round-113 whole-set-0 fallback. Chroma path unchanged.

### Round 117 — ALF transpose + classification filter-index derivation (§8.8.4.3)

#### Added
- `alf.rs`: `derive_alf_classification` — a clean-room transcription of
  ISO/IEC 23094-1 §8.8.4.3 (eq. 1289-1320). For every luma sample of a coding
  tree block it derives the gradient-classification `filtIdx` (0..24, one of
  the 25 spec filter classes) and the `transposeIdx` (0..3) from local
  horizontal / vertical / diagonal Laplacian activity: per-position gradients
  `filtH/V/D0/D1` over the [−2, blk+1] halo (eq. 1289-1292), per-4×4-subblock
  window sums with `i, j = −2..5` (eq. 1293-1297), per-sample direction
  strength `dir1/dir2/dirS` (eq. 1298-1314), activity quantisation `avgVar`
  via `varTab` (eq. 1315-1316), and finally `transposeIdx`
  (`transposeTable`, eq. 1317-1318) + `filtIdx` with the eq. 1320
  direction-strength offset. Cross products use `i64` to avoid overflow on
  large CTBs / high bit depths. Off-edge reads clamp to the nearest
  in-picture sample (§8.8.4.5 padding behaviour).
- `alf.rs`: `AlfClassification` — the per-sample classification output
  (`blk_width` / `blk_height` + row-major `filt_idx` / `transpose_idx`) with
  `filt_idx_at(x, y)` / `transpose_idx_at(x, y)` accessors.
- `alf.rs`: `transpose_luma_coeffs` — the §8.8.4.2 coefficient permutation
  (eq. 1282-1285) selecting the 13-tap arrangement a sample's `transposeIdx`
  requests; `transposeIdx == 0` (and any out-of-range value) returns the
  taps unchanged.
- `alf.rs`: `NUM_ALF_FILTERS` constant (25, `NumAlfFilters` per §8.9.4.1).
- 9 new unit tests (309 total, was 300): flat-block class-0 (with the
  spec's degenerate transposeIdx = 3), 4×4 subblock-constant resolution,
  exhaustive cross-check against an independently-coded reference derivation
  on a pseudo-random plane (full + edge CTBs), horizontal-edge directional
  classification, adversarial-checkerboard range bounds, the three spec
  transpose permutations + identity + centre/DC invariance, and the eq. 1316
  `BitDepthY − 2` shift scaling activity down at 10-bit.

#### Notes
- `derive_alf_classification` / `transpose_luma_coeffs` are pure, syntax-free
  building blocks. Wiring the per-sample classified filter selection into the
  §8.8.4.2 apply additionally requires the full §8.9.4
  `AlfCoeffL[ ][ filtIdx ][ ]` derivation (eq. 96-104 +
  `alf_luma_coeff_delta_idx` + `coefPosMap` + fixed filters), which the
  round-11 simplified `alf_data()` parser does not yet capture — tracked as a
  follow-up (the parser needs to consume the real §7.3.5 syntax:
  `uek(v)`-coded coeff deltas, the eg-order signalling, and the 25-entry
  `alf_luma_coeff_delta_idx[]` class-to-signalled-filter map).

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
  `docs/video/evc/`); spec-only sourcing, no web access.

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
  `docs/video/evc/`); spec-only sourcing, no web access.

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
  `docs/video/evc/`); spec-only sourcing, no web access.

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
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`);
  spec-only sourcing.

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
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`);
  spec-only sourcing.

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
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`);
  spec-only sourcing.

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
- Clean-room from ISO/IEC 23094-1:2020 (PDF in `docs/video/evc/`);
  spec-only sourcing.

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
