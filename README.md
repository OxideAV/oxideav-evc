# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Baseline + Main profiles. Zero C dependencies, zero FFI,
zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round-187 status

Round 187 lands the **§8.9.7 chroma DRA derived-state + §8.9.6 chromaScale
entry point** for the simpler `DraJoinedScaleFlag = 0` branch of §8.9.8
(eq. 1394 — `dra_table_idx == 58` → use `dra_cb_scale_value` /
`dra_cr_scale_value` directly). This closes the long-noted round-148
followup "lacks full §8.9.6 / §8.9.7 / §8.9.8 chromaScale derivation".

New surface:

* `dra::DraChromaDerived` — per-APS chroma DRA derived state for a
  single `ChromaIdx::Cb` / `ChromaIdx::Cr` component, holding
  `out_ranges_c[]` (§8.9.7 eq. 1387 midpoints + the §8.9.5 top sentinel
  at `1 << bit_depth_y`), `chroma_scales[]` (eq. 1394), `inv_chroma_scales[]`
  (eq. 1386), `out_scales_c[]` (eq. 1391 / 1393), and `out_offsets_c[]`
  (eq. 1389 / 1392). Indices follow the spec verbatim:
  `OutScalesC[0] = 0`, `OutOffsetsC[0] = invChromaScales[0]`,
  `OutRangesC[0] = OutRangesL[0]`; per-i recursion for
  `i ∈ [1, num_ranges_minus1]`; explicit top-sentinel at
  `i = num_ranges_l`.

* `dra::derive_dra_chroma_state(syntax, derived, cidx, bit_depth_y)`
  — runs §8.9.7 eq. 1386-1393 over a `(DraSyntax, DraDerived)` pair
  produced by the round-151 `parse_dra_syntax` /
  `derive_dra_state`, for one of the two chroma components.
  **Returns `Err(Error::Unsupported)` for `DraJoinedScaleFlag = 1`** —
  the table-driven §8.9.8 chain (eq. 1395-1419, depending on the SPS's
  `ChromaQpTable` + the §8.9.5 `ScaleQP` / `QpScale` tables) is the
  documented round-187 followup, blocked on threading the SPS chroma-QP
  table through to `DraDerived`. Defensive `Err(Error::Invalid)` on
  zero `dra_cb_scale_value` (Cb path) / `dra_cr_scale_value` (Cr path)
  even though §7.4.7 forbids them; pre-empts a divide-by-zero in
  eq. 1386.

* `dra::chroma_scale_for_luma_sample(luma_sample, chroma_derived) -> i64`
  — §8.9.6 entry point. Given a luma sample, runs §8.9.5 against the
  derived `out_ranges_c[]` (with `numRanges = num_ranges_l + 1`) to
  pick `rangeIdx`, then eq. 1384 (`incValue = lumaSample -
  OutRangesC[rangeIdx]`) and eq. 1385 (`OutOffsetsC[rangeIdx] +
  ((OutScalesC[rangeIdx] * incValue + (1 << 9)) >> 10)`). Returns 0 on
  an empty `DraChromaDerived` (num_ranges_l == 0).

* `dra::ChromaIdx { Cb, Cr }` — explicit Cb/Cr selector with `.as_u32()`
  returning the §8.9.8 `cIdx ∈ {0, 1}`.

* `DRA_MAX_RANGES_C = DRA_MAX_RANGES_V2 + 2 = 34` — storage size for
  `OutRangesC` + its §8.9.5 top sentinel.

### Wiring stance

Independent of `apply_post_filters`. The post-filter chroma path stays
on the round-148 `dra::apply_dra` legacy chain so existing fixtures
don't shift bit-positions in this round; the new entry point is opt-in
behind `derive_dra_chroma_state` + `chroma_scale_for_luma_sample`,
matching the round-181 pattern for the §8.9.3 luma side. A follow-up
round can decide whether to retire the legacy chain or thread §8.9.6
alongside it once a `sps_dra_flag = 1` fixture with a populated
`dra_syntax_aps` lands.

### Tests

16 new unit tests (389 total; was 373):

* `round187_chroma_derive_rejects_joined_path` — `dra_table_idx != 58`
  (joined-flag = 1) errors out; surfaces the §8.9.8 joined-chain gap.
* `round187_chroma_derive_rejects_zero_cb_scale` /
  `round187_chroma_derive_rejects_zero_cr_scale` — divide-by-zero
  defence on eq. 1386.
* `round187_chroma_derive_noop_on_empty_state` — `num_ranges == 0`
  returns an empty `DraChromaDerived` + the §8.9.6 helper returns 0.
* `round187_chroma_derive_eq1386_inv_chroma_scales_identity` /
  `round187_chroma_derive_eq1386_doubled_chroma_scale` — eq. 1386 on
  Q9 = 1.0 / 2.0 scale values.
* `round187_chroma_derive_eq1387_out_ranges_c_midpoints` — eq. 1387
  midpoints between successive `OutRangesL` entries.
* `round187_chroma_derive_eq1389_eq1391_out_scales_offsets_unjoined`
  — under joined = 0, `deltaScale = 0` ⇒ `OutScalesC[i] = 0`;
  `OutOffsetsC[i] = invChromaScales[i−1]`.
* `round187_chroma_derive_eq1392_eq1393_top_sentinel` — `i =
  numRangesL` final-pair derivation.
* `round187_chroma_derive_index_zero_layout_matches_spec` — the
  "respectively" line above eq. 1387 (the `OutOffsetsC[0] =
  invChromaScales[0]` reading).
* `round187_chroma_derive_cb_cr_distinct` — Cb (scale 256) vs Cr
  (scale 1024) derive independently; `OutRangesC` is component-
  agnostic.
* `round187_chroma_scale_for_sample_eq1384_eq1385_unjoined` — §8.9.6
  on the unjoined path collapses to `invChromaScales[0]` regardless
  of luma range (since `OutScalesC = 0` and all `OutOffsetsC =`
  same constant).
* `round187_chroma_scale_constant_property_unjoined` — exhaustive
  10-bit sample-space sweep confirming the constancy property.
* `round187_chroma_scale_for_sample_handles_out_of_range_high`
  — §8.9.5 top sentinel handles luma samples beyond the last
  internal boundary without panic.
* `round187_chroma_derive_single_range_identity` — `num_ranges_minus1
  = 0` edge case (the eq. 1388-1391 loop body is empty).
* `round187_chroma_idx_as_u32` — `ChromaIdx::Cb.as_u32() == 0`,
  `Cr == 1`.

### Documented followup

§8.9.8 `DraJoinedScaleFlag = 1` (joined chroma-scale path):
eq. 1395-1419 + `ScaleQP` / `QpScale` lookup tables + the SPS
`ChromaQpTable` (per §7.4.3). Round 187 surfaces the entry point as
`Err(Error::Unsupported)` so the caller can branch — but the
implementation needs `ChromaQpTable` threaded through from the SPS
into either `DraDerived` or the §8.9.6 entry point's signature, plus
the `ScaleQP[54]` / `QpScale[25]` tables transcribed verbatim from
eq. 1420 / 1421. Suggested split: tables + entry point in a
round 188 EVC slot.

## Round-181 status

Round 181 wires the round-151 spec-faithful `(DraSyntax, DraDerived)`
cache populated by `parse_dra_syntax` / `derive_dra_state` into a
dedicated public §8.9.3 entry point on `EvcDecoder`:

```rust
pub fn apply_luma_inverse_mapping_spec_faithful(
    &self,
    pic: &mut YuvPicture,
    pps_dra_aps_id: Option<u8>,
) -> Result<bool>;
```

The method looks up the cache slot for `pps_dra_aps_id`, runs the
round-174 §7.4.7 off-by-one reconciliation
(`fill_inv_luma_scales_range_zero`) on a local clone so the cached
state stays as the literal spec reading, then applies
`apply_luma_inverse_mapping_u8` over the picture's `y` plane. Returns
`Ok(true)` when the apply ran and `Ok(false)` when the slot was empty
(clean no-op). `dra_scale_value[0] == 0` propagates the
reconciliation's `Err`; the picture stays untouched on the error path.

**Independent of `apply_post_filters`.** The post-filter pipeline
stays on the round-148 `dra::apply_dra` path so existing fixtures
don't shift bit-positions in this round; the new entry point is
opt-in. A follow-up round can decide whether to retire the legacy
path or thread §8.9.3 alongside it once a `sps_dra_flag = 1` fixture
with a populated `dra_syntax_aps` is staged.

6 new unit tests (373 total; was 367) cover empty-slot no-op, strict
`None` `pps_dra_aps_id` no-op, identity scale (Q9 1.0 ⇒ identity LUT
on `[0, 255]`), doubled scale (input 240 ⇒ 120 via
`InvLumaScales[0] = 256`), the `dra_scale_value[0] == 0` error path,
and orthogonality to the legacy `dra_aps` cache.

## Round-174 status

Round 174 lands the **§8.9.3 luma inverse mapping helpers** + an
**empirical pin for the §7.4.7 `InvLumaScales[0]` / `DraOffsets[0]`
docs gap**. The round-151 `derive_dra_state` populates every
[`DraDerived`] field except entry 0 of `inv_luma_scales` /
`dra_offsets` — the spec text on page 86 restricts eq. 117-121 to
`i ∈ [1, dra_number_ranges_minus1]` but §8.9.3 indexes both arrays
with `rangeIdx ∈ [0, numOutRangesL − 1]`. Under the literal reading
every sample falling in the lowest segment collapses to
`(0 + 0 * sample + 256) >> 9 = 0`, clearly degenerate. New surface:
`apply_luma_inverse_mapping(plane, derived, bit_depth_y)` is a pure
§8.9.3 apply over a `u16` plane transcribing eq. 1374
(`incrValue = InvLumaScales[rangeIdx] * lumaSample`), eq. 1375
(`mappedSample = (DraOffsets[rangeIdx] + incrValue + 256) >> 9`),
and eq. 1376 (`invLumaSample = Clip1Y(mappedSample)`) — the
`rangeIdx` selected via the round-148 `find_range_idx` against the
derived `out_ranges_l`. `apply_luma_inverse_mapping_u8` is the
8-bit shortcut that builds a 256-entry LUT once via the new
`build_inv_luma_lut_8bit` helper. `fill_inv_luma_scales_range_zero`
applies the most plausible reconciliation of the docs gap — extending
eq. 118 / 120 / 121 to `i = 0` (treating `dra_scale_value[0]` as the
range-0 scale by symmetry with `OutRangesL`'s `i=0` `= 0` base case)
— in-place on a mutable `DraDerived` so callers that need
non-degenerate §8.9.3 output can opt in. **Round 174 takes no stance**
on the gap: both interpretations are surfaced so a follow-up round
can wire whichever the docs collaborator resolves to. No wiring into
`apply_post_filters` yet — the `dra_syntax_aps` 32-slot cache stays
populated in parallel with the legacy `dra_aps` cache until §8.9.6
chromaScale derivation is also ready. 367 unit tests pass (was 356) —
11 new cover: literal-spec range-0 degeneracy pin (a sample in the
lowest segment maps to 0), off-by-one fill produces a monotonic
non-flat range-0 mapping, defensive rejection of
`dra_scale_value[0] == 0`, num_ranges-zero no-op for the fill helper,
eq. 1376 `Clip1Y` upper-bound clip, eq. 1376 lower-bound clip to 0,
end-to-end `u16` plane apply with identity-Q18 round-trip, `u8`
shortcut bit-for-bit agreement with the LUT path, both `u8`/`u16`
helpers no-op on empty derived state, LUT-builder identity on empty
state, and a 2-range dispatch test (`OutRangesL = [0, 128, 256]`,
distinct per-range `InvLumaScales`/`DraOffsets`) verifying §8.9.5
picks the right segment and eq. 1374-1376 applies per-range with
hand-computed mapped values across the split and at the Clip1Y cap.
Suggested workspace-README row delta: EVC now provides §8.9.3 pure
luma inverse-mapping helpers (eq. 1374-1376) over `DraDerived`,
with the §7.4.7 `InvLumaScales[0]` / `DraOffsets[0]` docs gap
empirically pinned + an off-by-one reconciliation surfaced behind
explicit opt-in (lacks: docs-collaborator resolution of the §7.4.7
range-0 ambiguity + `apply_post_filters` wiring once §8.9.6
chromaScale also lands, Main-profile toolset — BTT/ADMVP/EIPD/ATS/
AMVR/affine).

### Documented spec gap

ISO/IEC 23094-1:2020(E) §7.4.7 page 86 restricts the
`InvLumaScales[apsId][i]` / `DraOffsets[apsId][i]` derivation
(eq. 117-121) to "i in the range of 1 to dra_number_ranges_minus1,
inclusive", leaving index 0 explicitly undefined. §8.9.3 (page 304)
indexes both arrays with `rangeIdx ∈ [0, numOutRangesL − 1]`
— including index 0. Under the literal spec reading, every sample
whose value falls in the lowest segment collapses to
`(0 + 0 * sample + 256) >> 9 = 0`, inconsistent with the spec's
own identity-DRA invariant (a DRA configured with
`dra_scale_value[j] = 1 << dra_descriptor2` for all `j` must
reproduce the input). The most plausible reconciliation is that the
§7.4.7 loop bounds are an off-by-one ("for i in the range of 0 to
dra_number_ranges_minus1, inclusive"), matching the per-range
one-scale-per-range data flow on the parser side and the
`rangeIdx ∈ [0, numOutRangesL − 1]` invocation surface of §8.9.3.
Recommend a §7.4.7 patch clarifying the loop bound — round 174
surfaces both readings (literal default + opt-in reconciliation)
pending the docs collaborator's resolution.

## Round-151 status

Round 151 lands the **§7.3.6-faithful `dra_data()` parser + §7.4.7
derivation** — the long-noted followup that the round-11
[`parse_dra_data`] above does not match the ISO/IEC 23094-1:2020(E)
wire format (it made up its own `dra_descriptor_present_flag` /
`dra_range_l[]` / `dra_chroma_qp_offset[]` shape that no real EVC DRA
APS payload follows). The new spec-faithful pair lives alongside the
legacy types so the round-148 `apply_dra` chroma-offset compatibility
is preserved while a follow-up round wires §8.9.3 luma mapping
(eq. 1374-1376) + §8.9.6 chroma scale (eq. 1384-1385) on top of the
new state and retires the legacy chain. New surface: `DraSyntax`
captures every raw bit `dra_data()` writes per §7.3.6 (page 42 — eight
fields including the two `dra_descriptor*` u(4) widths, the
ue(v)-coded `dra_number_ranges_minus1` + `dra_table_idx`, the u(10)
`dra_global_offset` + `dra_delta_range[]`, and the
`numBitsDraScale`-bit `dra_scale_value[]` + Cb / Cr scale values).
`DraDerived` holds every §7.4.7 per-APS derived variable from those
bits: `num_bits_dra_scale` (eq. 111), `joined_scale_flag` (the
`dra_table_idx == 58` branch), `in_dra_range[0..=num_ranges]`
(eq. 112-114 with the `BitDepthY − 10` pre-shift), `out_ranges_l`
(eq. 115-116 then re-shifted per eq. 122), `inv_luma_scales[]`
(eq. 117-119), and `dra_offsets[]` (eq. 120-121) — sized for up to 32
ranges, matching the spec ceiling on `dra_number_ranges_minus1`.
`parse_dra_syntax(payload, bit_depth_y)` performs the bitstream parse
*and* invokes the §7.4.7 derivation in one call, rejecting every
§7.4.7 bitstream-conformance violation with `Error::Invalid`
(`numBitsDraScale == 0`, `dra_number_ranges_minus1 > 31`,
`dra_global_offset` outside `[1, Min(1023, (1 << BitDepthY) − 1)]`,
`dra_scale_value` outside `[1, (4 << dra_descriptor2) − 1]`,
`dra_table_idx > 58`, `InDraRange[j] > (1 << BitDepthY) − 1`).
`derive_dra_state(syntax, bit_depth_y)` is the standalone §7.4.7 step
surfaced so a re-encoder can compute derived state from a
hand-built `DraSyntax`. `EvcDecoder::dra_syntax_aps` is a 32-slot
parallel cache indexed by `adaptation_parameter_set_id` that the APS
NAL branch populates when an SPS is in hand (so `BitDepthY` is known).
356 unit tests pass (was 339) — 17 new cover the minimal single-range
payload round-trip, the `joined_scale_flag` polarity on
`dra_table_idx ∈ {0, 58}`, equal-ranges delta distribution across 4
ranges, unequal-ranges per-range deltas across 3 ranges, the
`BitDepthY ∈ {8, 10, 12}` shift arithmetic on `InDraRange[]`, the
six §7.4.7 rejection paths (empty payload, zero/overlarge
`dra_scale_value`, zero `dra_global_offset`, overlarge
`dra_table_idx`, overlarge `dra_number_ranges_minus1`, `InDraRange`
overflow), `OutRangesL[]` post-eq.-122 recursion with identity Q4.9
scale (verifies the [0, 100, 200, 300] sequence exactly), and
`InvLumaScales[]` eq.-118 with identity (512 → 512) + halved
(256 → 1024) input scales, plus a decoder test that the parallel
`dra_syntax_aps` cache stores `(DraSyntax, DraDerived)` pairs.
Suggested workspace-README row delta: EVC now parses §7.3.6
`dra_data()` faithfully and derives every §7.4.7 per-APS variable
(`InvLumaScales` / `DraOffsets` / `OutRangesL` / `InDraRange` /
`DraJoinedScaleFlag`) ready for §8.9.3 / §8.9.6 wiring (lacks: §8.9.3
luma inverse mapping + §8.9.6 chromaScale derivation against the new
state, Main-profile toolset — BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-148 status

Round 148 lands the **§8.9.5 piecewise-function range-index identification
helper** (`find_range_idx`, eq. 1383 verbatim) and rewires `apply_dra` so
the chroma offset is applied **per chroma sample** using the segment of
the **co-located pre-DRA luma sample** — rather than the round-11
simplification of "use segment 0's `chroma_qp_offset` uniformly across
every Cb/Cr sample". Per ISO/IEC 23094-1:2020(E) §8.9.2, the §8.9.4 chroma
DRA process takes `decPictureL[ x * SubWidthC, y * SubHeightC ]` (the
decoded, **pre-mapping** luma sample at the co-located position) as one of
its inputs — so `apply_dra` now snapshots the pre-DRA luma plane before
the in-place LUT rewrite, then for each (x, y) chroma sample reads the
matching luma value, runs it through §8.9.5 (range-found-then-break with
the final `Min(rangeIdx, numRanges − 1)` clamp) against a new
`build_ranges_array` helper that appends `1 << bit_depth` as the top
sentinel (since the round-11 `DraData::range_l` table only stores
`num_ranges` lower boundaries, not the §8.9.5-required `num_ranges + 1`),
and adds the parsed `dra_chroma_qp_offset[ rangeIdx ]` to the chroma
sample with the existing `[0, (1 << bit_depth_c) − 1]` clip. Supports
4:2:0, 4:2:2, 4:4:4 via the per-format `SubWidthC` / `SubHeightC`
sub-sampling factors. 339 unit tests pass (was 331) — 8 new cover: eq.
1383 three-range walk including boundary + above-top fall-through, single
range always-zero, zero ranges no-op, ranges-array top-sentinel synthesis
for 8-bit + 10-bit luma, per-sample chroma offset uses co-located luma's
segment (3-segment 8×8 4:4:4 with x-varying luma values giving distinct
chroma offsets per column), the pre-DRA-vs-post-DRA snapshot check (a
post-DRA luma value that would re-classify into a sentinel-offset segment
fails the test if the chroma lookup wrongly reads post-DRA luma), 4:2:0
subsampled-luma alignment (chroma row y reads luma row 2y), and the upper
clip (`+60` over `chroma = 250` → `255` cap). The full §8.9.6 chromaScale
derivation (eq. 1384-1385 with `OutOffsetsC` / `OutScalesC` from §8.9.7 +
§8.9.8) is still parked pending the §7.3.6-faithful APS parser rewrite.
Suggested workspace-README row delta: EVC now applies the chroma DRA
offset per-sample using §8.9.5 to look up the co-located pre-DRA luma's
segment (lacks: full §8.9.6 / §8.9.7 / §8.9.8 chromaScale derivation with
QP-shifted invChromaScales, Main-profile toolset — BTT/ADMVP/EIPD/ATS/
AMVR/affine).

## Round-145 status

Round 145 lands the **§8.8.4.4 per-CTB chroma type filtering process**
— the chroma half of round-113's per-CTB ALF mechanism, and the
documented round-126 follow-up for ALF coverage. Until now
`apply_post_filters` ran the chroma apply over each chroma plane as a
whole regardless of the per-CTU map; §8.8.4.1 (lines 18079-18089)
actually invokes the §8.8.4.4 coding-tree-block chroma type filtering
process per CTB, gated on `alf_ctb_chroma_flag[ rx ][ ry ]` (Cb) and
`alf_ctb_chroma2_flag[ rx ][ ry ]` (Cr) on the
`ChromaArrayType == 3` path. A new `apply_alf_chroma_masked` mirrors
`apply_alf_luma_masked`: it walks the per-CTU map, converts the luma
CTB origin and size to chroma coordinates via
`x_ctb_c = (rx << CtbLog2SizeY) / SubWidthC`,
`blkW_c = (CtbSizeY / SubWidthC) min (PicWidthC − x_ctb_c)` per
§8.8.4.1 lines 18105-18107, applies the §8.8.4.4 eq. 1321 7-coefficient
stencil with picture-edge clamping, and writes per-CTB results
against a whole-plane pre-filter snapshot so a filtered CTB at the
edge of a flagged region still reads neighbours at their unfiltered
values. A new `apply_chroma_alf_masked_or_whole_plane` helper in
`decoder.rs` routes per-CTB when the chroma map has any flag set and
falls back to the round-126 whole-plane `apply_alf_chroma` otherwise
(the `ChromaArrayType ∈ {1, 2}` path where per-CTB chroma flags are
inferred 0). Round 145 also **corrects a round-11 chroma tap-geometry
bug**: `CHROMA_TAPS` / `CHROMA_TAPS_SYM` had positions 3, 4, 5
permuted vs eq. 1321 — `coef[3]` (the spec's diagonal pair at
`rec[x ∓ 1, y ± 1]`) was multiplying the horizontal `rec[x ± 2, y]`
pair instead, and similarly for 4 and 5. No prior test exercised
non-DC chroma coefficients so the bug was silent; the round-145 eq.
1321 reference cross-check now pins the geometry exhaustively. 331
unit tests pass (was 326) — 5 new cover the corrected eq. 1321 stencil
against an independently-coded reference on a synthetic gradient
plane, the per-CTB masked apply reproducing whole-plane apply
bit-for-bit when every CTU is flagged, the per-CTB apply filtering
only flagged CTUs (and leaving the other plane untouched on a 4:4:4
fixture), a partial-CTU edge clamp on a 24×24 4:2:0 picture
(12×12 chroma plane, 4-column-wide right chroma CTU), and the
monochrome no-op. Suggested workspace-README row delta: EVC now
applies the §8.8.4.4 per-CTB chroma type filter under
`ChromaArrayType == 3` (driven by the round-113 chroma map) and
corrects the eq. 1321 chroma tap geometry (lacks: Main-profile
toolset — BTT/ADMVP/EIPD/ATS/AMVR/affine).

## Round-126 status

Round 126 lands the **multi-APS cache indexed by
`adaptation_parameter_set_id`** — the routing layer the spec-mandated
§7.3.4 `slice_alf_luma_aps_id` / `slice_alf_chroma_aps_id` /
`slice_alf_chroma2_aps_id` + PPS `pic_dra_aps_id` reference but every
prior round collapsed to a single `Option<AlfData>` / `Option<DraData>`
slot. Both `EvcDecoder::alf_aps` and `dra_aps` are now 32-slot arrays
indexed by the 5-bit APS id; `send_packet`'s APS branch writes the
parsed payload into the cache slot named by the NAL's
`adaptation_parameter_set_id` (overwriting on update — the spec's
update-by-id semantics), and `apply_post_filters` pulls each plane's
ALF data from the slot the slice references. The slice header parser
now surfaces all three APS ids (formerly `_` discards). `SliceHeader`
gains `slice_alf_luma_aps_id` / `slice_alf_chroma_aps_id` /
`slice_alf_chroma2_aps_id`, each `Option<u8>` exactly when the §7.4.5
gating signals it. ChromaArrayType==3 slices with
`slice_alf_chroma_idc == 3` can now apply different ALF data to the Cb
and Cr planes (Cb / Cr fallback chain: explicit chroma APS id → joint
chroma APS id → luma APS, per §7.4.5's same-APS inference). A back-compat
fallback to the most-recently-stored APS keeps the minimal-header IDR
fixture path (which doesn't decode a slice header and therefore can't
surface an id) working unchanged. A new `PostFilterInputs` struct
bundles the seven §8.9 + §8.10 inputs so the call site stays under the
`clippy::too_many_arguments` lint. 326 unit tests pass (was 321) — 5
new ones cover the three slice APS ids on a 4:4:4 slice, the no-ALF
unset case, two-distinct-ALF-APS resolution by slice id + last-stored
fallback, the DRA mirror of the same, and an APS NAL parse populating
its declared cache slot without touching others. Suggested
workspace-README row delta: EVC now routes ALF + DRA APS lookups
through the slice's / PPS's `aps_id` rather than the most-recently
parsed APS (lacks: per-CTU ALF filter-set selection §8.9.6 picking
between multiple APS sets within one slice, Main-profile toolset —
BTT/ADMVP/EIPD/ATS/AMVR/affine).

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
