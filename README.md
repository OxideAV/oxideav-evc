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
`slice_data`, `intra`, `inter`, `affine` / `affine_cand` / `affine_syntax`,
`eipd` / `eipd_mode` / `eipd_ref` / `eipd_syntax`,
`ats`, `transform`, `dequant`, `deblock`, `htdf`, `hmvp`, `rpl`,
`neighbour`, `picture`, and the registered `decoder`
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
(`eipd` + `eipd_mode` modules). The §8.4.4.4/.5/.8/.9/.10 sample kernels
cover the full mode set of Table 15 — `INTRA_DC` (eqs. 286-288
aspect-ratio average), the `availLR`-dependent `INTRA_HOR` (§8.4.4.4: the
eq.-290 `LR_11` left/right-column horizontal blend via
`divScaleMult[Log2(nCbW)]`, the eq.-291 `LR_01` right-column copy, the
eqs.-292 `LR_00`/`LR_10` left-column copy), `INTRA_VER` (§8.4.4.5
eq. 293), `INTRA_BI` bilinear (eqs. 297-311 with the `divScaleMult` /
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
per-element Main-profile context tables).

The §8.4.4.1/.2 **reference-sample construction + substitution**
process is now implemented (`eipd_ref` module). `construct_eipd_refs`
runs the §8.4.4.1 *General* neighbourhood gather — the top row
`p[x][-1]` (x = 0..nCbW+nCbH-1), the left column `p[-1][y]`, the
`p[-1][-1]` corner, and the SUCO `p[nCbW][y]` right column — over two
caller-supplied closures (a §6.4.1 `availableN` predicate already folded
with `constrained_intra_pred_flag`, and a reconstructed-sample lookup),
copying recon samples through where available and marking the rest "not
available for intra prediction". The §8.4.4.2 substitution then fills
every hole: under `sps_eipd_flag == 0` every hole takes the mid-level
constant `1 << (bitDepth-1)` (corner, then top row, then left column);
under `sps_eipd_flag == 1` only an unavailable corner takes the
mid-level and every other hole copies its scan predecessor
(`p[x-1][-1]` along the top, `p[-1][y-1]` down the left, and on the SUCO
path `p[nCbW][y-1]` down the right, the y=0 right predecessor taken from
the top row at x=nCbW). The result is a `ConstructedRefs` carrying the
filled `EipdRefSamples` ready for the §8.4.4.3-§8.4.4.10 kernels. The
process is pure over the closures; `picture::fetch_eipd_refs` is the
data-plane bridge that drives it from the `YuvPicture` reconstructed
buffer (the EIPD analogue of the Baseline `fetch_intra_refs`),
populating the top / left / corner / SUCO-right neighbourhood from the
post-reconstruction plane with a causal availability rule.
`picture::intra_reconstruct_cb_eipd` then ties `fetch_eipd_refs` →
`predict_eipd` → §8.7.5 picture-construction (`clip(pred + res)`,
eq. 1091) → `store_block` into one end-to-end EIPD reconstruct call (the
EIPD analogue of the Baseline `intra_reconstruct_cb`), deriving the
§6.4.2 `availLR` the kernels consume from the causal rule. The remaining
intra wiring is the slice-walker selecting the EIPD vs Baseline path on
`sps_eipd_flag` and threading the real §6.4.1 `IsCoded` raster + tile
predicate + `constrained_intra_pred_flag` through the availability
closure.

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
MC path consumes.

The **§8.5.2.3.3–§8.5.2.3.5 temporal (collocated) merge candidate**
(TMVP) is now derived in the `tmvp` module — the candidate
`build_merge_cand_list` accepts in its `temporal` slot.
`tmvp_collocated_mv` (§8.5.2.3.4) does the POC-ratio scaling
(`distScaleFactor = (currPocDiff << 5) / colPocDiff`, eq. 503; the
eq. 504 round-half-away-from-zero clip) with the invalid-refIdx /
zero-colPocDiff escape and the joint `availableFlagCol` (0/1/2/3) fold;
`constrain_scaled_mv` (§8.5.2.3.5) clips against the padded reference
grid (eqs. 506-511, `picPaddingSize = 144`); `tmvp_merge_candidate`
(§8.5.2.3.3) runs the central → bottom → side collocated-position
fallback (eqs. 485-500, 8×8-grid-quantised, first-available wins) with
the eqs. 487/488 small-block bi-pred→uni demotion. The derivation is
pure over a caller-supplied `ColPic` motion-field closure;
`collocated_mv_from_side_info` bridges the decoder's existing
`SideInfoGrid` per-4×4-cell motion field into that closure, so no new
DPB-level per-picture motion array is needed — only the per-CU wiring of
the §8.5.2.3.3 POC distances from the decoded `RefPicList0/1` remains.

The **§8.5.3 affine** motion toolset (`sps_affine_flag == 1`) is now
implemented at the derivation + syntax layers (`affine` + `affine_syntax`
modules). The §8.5.3.7–§8.5.3.10 **geometric core** turns a 2- or 3-CPMV
(control-point motion-vector) set into a dense per-subblock motion field:
`affine_model_params` (§8.5.3.9) derives `dX`/`dY`/`mvBaseScaled` (eqs.
897-906 with the 4-param `dY` rotation identity and the 6-param vertical
gradient); `affine_subblock_size` (§8.5.3.8) derives `(sizeSbX, sizeSbY)`
from Tables 22/23 plus the EIF bounding-box applicability test
(`(W+2)·(H+2) > 72` → `clipMV`, eqs. 879-896); `affine_subblock_mvs`
(§8.5.3.7) evaluates the model at each subblock centre (eqs. 873-878,
§8.5.3.10 `rightShift = 5` rounding, 18-bit clip) and derives the
§8.5.2.6 chroma field (1/16-pel luma, 1/32-pel chroma out). The §8.5.3.1
`reconstruct_cp_mv` (eqs. 688-691) and `affine_center_mv` (eqs. 696-701,
the §8.5.2.7 HMVP-update centre) plus the §8.5.3.3 `inherited_cp_mvs`
(eqs. 744-763, projecting an affine neighbour's corner MVs onto the
current control points across the `isCTUboundary` + model-idc branches)
complete the derivation surface that feeds the merge/AMVP paths. The
§7.3.8.4 CABAC syntax (`affine_syntax`) reads `affine_flag` (§9.3.4.2.4
neighbour-derived ctxInc, Table 96), `affine_merge_idx` (TR cMax 5, per-bin
ctxInc, Table 56), `affine_mode_flag` (→ `numCpMv`), and the per-list
`affine_mvp_flag`/`affine_mvd_flag` group into an `AffineDecision` the
derivation consumes.

The **§8.5.3.2/.4/.5/.6 affine merge/predictor candidate-list assembly**
is now implemented at the derivation layer (`affine_cand` module), pure
over caller-supplied neighbour/corner sources like the merge/tmvp
contracts. §8.5.3.4 `constructed_merge_candidates` builds the six
corner-combined candidates Const1..Const6 (eqs. 798-835) — the
eqs. 807/813/819 corner-completion arithmetic and the eqs. 828/829
4-param model derivation of Const6's top-right CPMV (shift
`7 + 2·log2W − log2H`, §8.5.3.10 `rightShift = 7` rounding + 16-bit
clip). §8.5.3.2 `build_affine_merge_cand_list` assembles
`affineMergeCandList` (≤ 5): the inherited model-based neighbours
(`availLR`-ordered via `affine_merge_inherited_order` over the eqs.
708-717 positions, projected through §8.5.3.3 `inherited_cp_mvs`), then
Const1..6, then the step-9 zero-CPMV tail with the P/B-slice L1
utilization; `select_affine_merge_candidate` is the eqs. 735-741 index
bridge. §8.5.3.6 `constructed_mvp_candidate` derives the single
constructed predictor (eqs. 868-872, corner-2 completion); §8.5.3.5
`build_affine_mvp_cand_list` assembles `cpMvpListLX` (exactly 2): the
inherited A/B/C refIdx-matched predictors, the constructed predictor,
the per-corner translational fill (cpIdx 2→0 with the cpIdx-2→3
redirect), then the zero tail (eqs. 836-866). `reconstruct_affine_amvp
_cp_mvs` composes the eq.-867 predictor select with the §8.5.3.1 per-CP
MVD reconstruction.

Still deferred: the §8.5.3.4 corner-2/3 collocated-MV resolution wiring
(the `CornerMv` temporal slots are threaded but the caller-side
§8.5.2.3.4 lookup isn't yet bridged), ATS-inter / MMVD-syntax / AMVR /
DMVR, plus the picture-level wiring of the EIPD + ATS-intra + merge +
affine layers into a full Main-profile `coding_unit()` reconstruction
(needs the §6.4.1 neighbour-mode grid + the per-position MV store).

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

The **§8.7.6 Hadamard Transform Domain Filter** (HTDF, `sps_htdf_flag
== 1`) post-reconstruction luma filter is implemented (`htdf` module).
`htdf_applies` encodes the four §8.7.6.1 applicability gates;
`derive_htdf_lut` (§8.7.6.3) selects `(bLUT, aTHR, tblShift)` from the
`setOfLUT[5][16]` / `tblThrLog2[5]` tables (eqs. 1106-1111);
`pad_rec_samples` (§8.7.6.2) builds the `(nCbW+2)×(nCbH+2)`
replicate-padded array with the `dx`/`dy` border clamp gated on a
`BorderAvailability` predicate; `filter_block` (§8.7.6.1) runs the
sliding 2×2 forward Hadamard (eqs. 1093-1097), the bit-depth-branched
soft-threshold of the three AC coefficients (eqs. 1098/1099, DC kept),
the inverse Hadamard (eqs. 1100-1103), the overlap accumulation
(eq. 1104) and the `Clip1Y((accFlt + 2) >> 2)` rounding (eq. 1105). The
filter is pure over the reconstructed-sample accessor;
`picture::apply_htdf_luma` is the data-plane bridge that drives it
in-place on the `YuvPicture` luma plane (short-circuiting on the
applicability gates). Invoking that bridge from the §8.4.1 / §8.6
reconstruction (when `cbf_luma && sps_htdf_flag`) and threading the real
§6.4.1 border predicate is the next wiring step.

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
