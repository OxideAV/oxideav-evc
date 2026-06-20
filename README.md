# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

A working **Baseline-profile** decoder: IDR + P + B slices, 8-bit 4:2:0,
with residual coding (RLE + dequant + inverse DCT-II for transform sizes
up to 64×64), intra prediction (the 5-mode Baseline set), inter
prediction (8-tap luma / 4-tap chroma sub-pel interpolation, AMVP
candidate construction, default-weighted bipred), deblocking
(`sps_addb_flag == 0` luma + chroma), a multi-reference DPB with POC
reordering, spatial-neighbour MV grid AMVP, the HMVP candidate list, and
reference-picture-list parsing for non-IDR slices.

The crate decomposes into spec-faithful modules: `bitreader`, `nal`,
`sps` / `pps` / `aps`, `slice_header`, `cabac` + `cabac_init`,
`slice_data`, `intra`, `inter`, `eipd` / `eipd_mode` / `eipd_syntax`,
`ats`, `transform`, `dequant`, `deblock`, `hmvp`, `rpl`, `neighbour`,
`picture`, and the registered `decoder`
factory. All clause / equation / table numbers cite ISO/IEC
23094-1:2020(E) directly.

### Main profile

The Main-profile CABAC infrastructure is in place — the §9.3.5 init
tables (Tables 40-90), the §9.3.2.2 init pipeline, and the §9.3.4.2
per-syntax-element ctxInc helpers — together with a large body of pure
§6.5.1 / §7.4.5 tile-geometry and §6.4 neighbour-availability helpers and
the §7.3.8.1 multi-tile `slice_data()` CTU-walk driver. The §7.4.8.3
binary/ternary-tree (BTT) split-geometry layer (`split` module) supplies
the `allowSplit{Bt,Tt}{Ver,Hor}` derivations, the `btt_split_dir`/`type`
signalling + inference predicates, and the `SplitMode` derivation
(including the picture-boundary implicit-split rules), all driven by the
§7.3.2.2 BTT size limits (eqs. 43/44/62-67). On top of that geometry the
§7.3.8.3 `decode_btt_split` CABAC reader consumes the
`btt_split_flag`/`dir`/`type` bins (Tables 42/43/44 + the §9.3.4.2.5 /
Table 95 ctxInc derivations), applies the §7.3.8.3 presence gating and
§7.4.8.3 inference for any absent element, and resolves the final
`SplitMode`. On top of that the §7.3.8.3 recursion-geometry layer
(`split_unit_children` + `quad_split_children`) enumerates the ordered child
`split_unit()` invocations for every resolved `SplitMode` and for the
`split_cu_flag == 1` quad split — the new child positions / log2 dimensions /
`ctDepth` / `splitUnitOrder` tuples, with the spec's exact picture-boundary
child gating (`x1 < pic_width` / `y1 < pic_height` for quad + BT, no guard for
the bounds-constrained ternary shapes) and the §7.3.8.3 `sps_suco_flag`
mirrored-order reordering. The §7.4.9.3 SUCO-availability layer
(`SucoSizeLimits` eqs. 68/69 + `allow_split_unit_coding_order`) derives the
`MaxSucoLog2Size` / `MinSucoLog2Size` window and the four-condition
`allowSplitUnitCodingOrder` predicate that, with `sps_suco_flag`, gates
whether `split_unit_coding_order_flag` is signalled.

The **EIPD** (extended intra prediction, `sps_eipd_flag == 1`) toolset is
now implemented end-to-end at the prediction-and-mode-derivation layer
(`eipd` + `eipd_mode` modules). The §8.4.4.8/.9/.10 sample kernels cover
the full mode set of Table 15 — `INTRA_DC` (eqs. 286-288 aspect-ratio
average), `INTRA_BI` bilinear (eqs. 297-311 with the `divScaleMult` /
`weightFactor` Tables 17/18), `INTRA_PLN` planar (eqs. 314-325 with the
`mult`/`shift` Table 19), and the 33-direction angular set (§8.4.4.10
Table 20 `dirXYSign`/`divDxy`/`divDyx`, the two-step
`iOffset`/`iX`/`iY`/`refPosition` derivation + 4-tap fractional filter,
eqs. 326-385) — driven by an `EipdRefSamples` neighbourhood that exposes
`p[x][-1]` / `p[-1][y]` / `p[nCbW][y]` with the spec's `-1` origin and the
`AvailLr` (eq. 23) left/right availability codes. The §8.4.2 luma
mode-derivation builds the three ranked lists (`candModeList` /
`extCandModeList` / `remModeList`, eqs. 172-278) across all six
validC×planar/directional branches with the dedup-fill loops, and the
§8.4.3 chroma derivation maps `intra_chroma_pred_mode` through Table 16
with the `modeIdx` skip rule. The CABAC syntax reads that feed these are now wired (`eipd_syntax`
module): `read_luma_mode_selector` consumes the §7.3.8.4 luma group —
`intra_luma_pred_mpm_flag`/`idx` (FL cMax=1, ctxInc 0, Tables 63/64),
`intra_luma_pred_pims_flag` (bypass), `intra_luma_pred_pims_idx`
(FL cMax=7 bypass), `intra_luma_pred_rem_mode` (TB cMax=22 bypass via the
new §9.3.3.6 `decode_tb_bypass` primitive) per Table 95 — and
`read_intra_chroma_pred_mode` decodes the §9.3.3.7 Table 93 bin string
(bin0 ctxInc 0 Table 65, rest bypass). `resolve_eipd_luma_mode` /
`resolve_eipd_chroma_mode` compose those reads with the §8.4.2/.3
derivation and selection, producing the concrete `IntraPredModeY` /
`IntraPredModeC` the §8.4.4 kernels consume; the `EipdCtx` selector
honours `sps_cm_init_flag` (collapse-to-`(0,0)` under Baseline vs the
per-element Main-profile context tables). The §8.4.4.1/.2 reference
construction/substitution remains the next intra follow-up.

The **ATS-intra** (Adaptive Transform Selection, `sps_ats_flag == 1`,
intra path) toolset is implemented end-to-end at the syntax + transform
layer (`ats` module): `read_ats_intra` consumes the §7.3.8.5 group
(`ats_cu_intra_flag` bypass; `ats_hor_mode`/`ats_ver_mode` ctxInc 0,
Table 79) and applies the Table 30 derivation
(`trType{Hor,Ver} = 1 + ats_{hor,ver}_mode`); `AtsIntra::apply_inverse`
bridges that decision to `inverse_transform_ats`, the trType-parameterized
§8.7.4.2 two-stage inverse. The DST-VII (eqs. 1077-1083) and DCT-VIII
(eqs. 1084-1090) kernels now span the **full §8.7.4.3 size set
`nTbS ∈ {4, 8, 16, 32}`** matching the §7.3.8.5 `log2 <= 5` presence
predicate. The two transcribed kernel families per size are cross-checked
against each other by the spec-derivable reflection identity
`DCT8[m][n] = (m&1?-1:1)·DST7[m][N-1-n]`, and an end-to-end test decodes
a synthesised CABAC bin sequence through to the dispatched kernel for
every size.

The **§8.5.2.3 ADMVP merge-mode** candidate-list derivation
(`merge` module, `sps_admvp_flag == 1`) is now implemented at the
derivation layer as pure spec functions over a §6.4.3 neighbour-MV
lookup closure (mirroring the `inter::build_amvp_list_baseline` purity
contract). `spatial_neighbour_positions` resolves the five §8.5.2.3.2
neighbour locations (A1/B1/B0/A0/B2) across all three §6.4.2 `availLR`
branches; `spatial_merge_candidates` appends them with the eqs. 463/464
small-block bi-pred→uni demotion, the `numCurrMergeCand < mLSize−1`
A0/B2 gate and the §8.5.2.3.10 redundancy trim. `combined_bipred_candidates`
runs the §8.5.2.3.7 B-slice Table-21 pairing; `zero_mv_candidates` fills
the §8.5.2.3.8 tail; `HmvpCandList::hmvp_merge_candidates` derives the
§8.5.2.3.6 history-based merge candidates (the `maxNumCheckedHistory`
stride-of-4 tail walk). `build_merge_cand_list` is the §8.5.2.3.1 general
assembly (spatial → temporal → HMVP → combined → zero), and
`select_merge_candidate` is the step-6 bridge that projects a decoded
`merge_idx` into the concrete `mvLX`/`refIdxLX`/`predFlagLX` the §8.5.4
MC path consumes. Still deferred: the §8.5.2.3.3/.4 **temporal
collocated** candidate (needs the DPB-level collocated motion field) and
the **affine** (§8.5.3) / ATS-inter / MMVD-syntax / AMVR / DMVR tools,
plus the picture-level wiring of the EIPD + ATS-intra + merge layers into
a full Main-profile `coding_unit()` reconstruction (needs the §6.4.1
neighbour-mode grid + the per-position MV store).

The remaining Main-profile syntax-decode tools (CABAC-driven BTT tree
walk / SUCO / ADMVP / IBC / ATS-inter / ADCC / ALF / DRA / AMVR / MMVD /
affine / DMVR) still surface `Error::Unsupported`.

The DRA (§8.9) post-filter chain is spec-faithful end-to-end: the
§7.3.6 `dra_data()` parser + §7.4.7 derivation feed the §8.9.3 luma
inverse mapping and the §8.9.6/§8.9.7/§8.9.8 chroma scale derivation
(including the §8.9.8 joined path via `ChromaQpTable` and the
eq. 1398-1409 `tableNum == 0` pivot-boundary sub-case, resolved per
errata #81 / #130 as a guard around eqs. 1400-1402 that falls through
to eqs. 1403-1409 — not a constant-identity short-circuit). The chroma
*apply* now uses the §8.9.4 eqs. 1377-1382 magnitude scale
(`map_one_chroma_sample` / `apply_chroma_inverse_mapping_u8`) driven
by the per-luma-sample `chromaScale`, superseding the round-11
per-segment QP-offset approximation.

### Not yet supported

- 10-bit / high-bit-depth pixel decode.
- Advanced deblocking (`sps_addb_flag == 1`).
- Full Main-profile picture reconstruction (see above).

## Usage

```rust
use oxideav_evc::probe;

let bytes: &[u8] = b""; // an EVC (NAL length-prefixed) bitstream
if let Some(info) = probe(bytes) {
    println!("{}x{}", info.width, info.height);
}
```

The decoder also registers with the framework via
`oxideav_evc::register(&mut codecs)`.

## Clean-room provenance

Every line is derived from ISO/IEC 23094-1:2020 and the in-repo errata
notes; all truth comes from `docs/video/evc/`. No external decoder or
library source was consulted.

## License

MIT — see [LICENSE](LICENSE).
