# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Baseline + Main profiles. Zero C dependencies, zero FFI,
zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round-278 status

Round 278 lands §6.5.1 eq. (32) — the **`TileIdToIdx[ ]` set and the
`FirstCtbAddrTs[ tileIdx ]` list** — plus the subclause's two trailing
luma-sample derivations, **completing §6.5.1**: every list it derives
(eq. 24-32 + `ColumnWidthInLumaSamples[ ]` /
`RowHeightInLumaSamples[ ]`) now has a pure helper and a `Pps`
dispatch.

eq. (32) is a single walk over round-273's `TileId[ ctbAddrTs ]`
with `tileStartFlag` / `tileEndFlag` run detection: each run start
records `TileIdToIdx[ TileId[ ctbAddrTs ] ] = tileIdx` and
`FirstCtbAddrTs[ tileIdx ] = ctbAddrTs`; each run end advances
`tileIdx`. Because eq. (28)-(30) pack each tile contiguously in tile
scan, `FirstCtbAddrTs[ k ]` is exactly the prefix sum of
`NumCtusInTile[ 0 .. k ]` (pinned by a sweep test).

New surface:

* `pps::TileIndexMaps` — both eq. (32) outputs. The spec calls
  `TileIdToIdx` a *set* over `NumTilesInPic` tileId values (explicit
  tile IDs are sparse, so a dense tileId-indexed list would be
  unbounded); it is carried as `(tileId, tileIdx)` pairs in tile-scan
  first-encounter order with a `tile_idx_for_id` lookup.
* `pps::compute_tile_index_maps` — the eq. (32) walk, verbatim
  semantics including the malformed-input case (a repeated
  non-contiguous ID overwrites the set entry and still appends its
  own `FirstCtbAddrTs` slot, exactly as the loop's assignments do).
* `pps::compute_column_width_in_luma_samples` /
  `pps::compute_row_height_in_luma_samples` —
  `ColWidth[ i ] << CtbLog2SizeY` / `RowHeight[ j ] << CtbLog2SizeY`,
  saturating on overflow.
* `Pps::tile_index_maps` / `Pps::column_width_in_luma_samples` /
  `Pps::row_height_in_luma_samples` — instance dispatch.
  `PicWidthInCtbsY` / `PicHeightInCtbsY` / `CtbLog2SizeY` stay
  explicit caller arguments (§7.4.3.1, SPS side).

### Tests

12 new unit tests (620 total; was 608): eq. (32) single-tile +
`FirstCtbAddrTs` 2×2 hand-trace + `NumCtusInTile[ ]` prefix-sum
sweep + explicit-ID set round-trip + `TileId[ FirstCtbAddrTs ]`
inversion sweep + empty-input + malformed-repeat assignment pin;
luma-sample hand-traces + coverage sweep + saturation pin; `Pps`
dispatch agreement including the explicit-tile-ID path.

### Wiring stance

Same opt-in posture as the round-218 onward helper rollout: pure
functions returning owned values, no behaviour change to existing
decoder paths.

### Next arc

§6.5.1 is done; the natural consumers are the slice-header
derivations — §7.4.5's
`ctbAddrInTs = FirstCtbAddrTs[ SliceTileIdx[ i ] ]` walk and the
§7.4.3.4 `TileIdToIdx[ first_tile_id ]` / `last_tile_id` slice-tile
resolution (eq. around 82).

### Disclaimer

`TileIndexMaps` / `compute_tile_index_maps` /
`compute_column_width_in_luma_samples` /
`compute_row_height_in_luma_samples` (and the matching `Pps`
methods) are derived from ISO/IEC 23094-1:2020 §6.5.1 eq. (32) and
the subclause's trailing luma-sample paragraphs. All truth came from
`docs/video/evc/`.

## Round-273 status

Round 273 completes the §6.5.1 **CTB raster-and-tile-scanning
address chain** by landing the four remaining derivations
eq. (28)-(31), now that the docs collaborator resolved the eq. (30)
vs §7.4.3.2 `tile_id_val[ i ][ j ]` index contradiction (in-repo
errata `evc-errata-and-clarifications.md` #97).

* eq. (28) `CtbAddrRsToTs[ ]` — raster→tile-scan CTB-address map.
  Consumes round-249's `ColWidth[ ]` / `RowHeight[ ]` and round-270's
  `ColBd[ ]` / `RowBd[ ]`. The output is a permutation of
  `0 ..= PicSizeInCtbsY − 1`.
* eq. (29) `CtbAddrTsToRs[ ]` — the inverse permutation.
* eq. (31) `NumCtusInTile[ ]` — `ColWidth[ i ] * RowHeight[ j ]` per
  tile in raster-tile order.
* eq. (30) `TileId[ ]` — the tile-scan-address→tile-ID map.
  The implicit branch assigns the linear `tileIdx`; the explicit
  branch reads `tile_id_val[ i ][ j ]` with **`i` = column, `j` =
  row** per errata #97, indexing the §7.4.3.2 syntax-order table at
  `j * num_tile_columns + i`.

### Errata #97 (the unblock)

§7.4.3.2's first sentence calls `tile_id_val[ i ][ j ]` "the i-th
tile **row** and the j-th tile **column**", but eq. (30)'s loop nest
and §7.4.3.2's own uniqueness constraint both bind `i` to the column
and `j` to the row. The in-repo errata fixes the reading as a
transposed-axis typo; the explicit `TileId[ ]` branch follows it.

New surface:

* `pps::compute_ctb_addr_rs_to_ts` / `pps::compute_ctb_addr_ts_to_rs`
  / `pps::compute_num_ctus_in_tile` / `pps::compute_tile_id` — pure
  module-level functions over the extent/boundary lists.
* `Pps::ctb_addr_rs_to_ts` / `Pps::ctb_addr_ts_to_rs` /
  `Pps::num_ctus_in_tile` / `Pps::tile_id` — instance dispatch that
  derives the four §6.5.1 lists from the parsed PPS.
  `PicWidthInCtbsY` / `PicHeightInCtbsY` stay explicit arguments
  (they derive from §7.4.3.1 against the SPS). `Pps::tile_id` selects
  the explicit branch automatically when `explicit_tile_id_flag` is
  set.

### Tests

13 new unit tests (608 total; was 595): eq. (28) single-tile
identity + 2×2 hand-trace + permutation sweep; eq. (29) two-way
round-trip inversion sweep; eq. (31) 2×2 + non-uniform-remainder +
Σ = PicSizeInCtbsY sweep; eq. (30) implicit-tileIdx + per-tile count
cross-check against `NumCtusInTile[ ]` + the errata-#97 explicit
indexing pin; plus `Pps` dispatch-agreement and a zero-width
defensive case.

### Wiring stance

Same opt-in posture as the round-218 onward helper rollout: pure
functions returning owned vectors, no behaviour change to existing
decoder paths.

### Disclaimer

`compute_ctb_addr_rs_to_ts` / `compute_ctb_addr_ts_to_rs` /
`compute_num_ctus_in_tile` / `compute_tile_id` (and the matching
`Pps` methods) are derived from ISO/IEC 23094-1:2020 §6.5.1
eq. (28)-(31), with the eq. (30) `tile_id_val` indexing resolved by
the in-repo errata `evc-errata-and-clarifications.md` #97. All truth
came from `docs/video/evc/`.

## Round-270 status

Round 270 continues the §6.5.1 CTB-raster-and-tile-scanning chain by
landing the **`ColBd[ i ]` and `RowBd[ j ]` tile-boundary derivations**
(eq. 26 / eq. 27), the immediate prefix-sum primitives that consume
round-249's `ColWidth[ ]` / `RowHeight[ ]` (eq. 24 / eq. 25). They are
the next §6.5.1 list after the per-tile extents and the inputs the
eq. (28) `CtbAddrRsToTs[ ]` walk reads via its `tbX >= ColBd[ i ]` /
`tbY >= RowBd[ j ]` tile-locating tests.

§6.5.1 eq. (26) is the running prefix-sum

```text
for( ColBd[ 0 ] = 0, i = 0; i <= num_tile_columns_minus1; i++ )
    ColBd[ i + 1 ] = ColBd[ i ] + ColWidth[ i ]
```

so `ColBd[ ]` runs `i` from `0` to `num_tile_columns_minus1 + 1`,
inclusive — one entry longer than `ColWidth[ ]`. `ColBd[ 0 ] = 0`,
and `ColBd[ num_tile_columns_minus1 + 1 ]` is the right edge of the
last tile column (= `PicWidthInCtbsY` whenever the eq. (24) widths
cover the picture, which they do by construction). eq. (27) is the
symmetric `RowBd[ ]` walk over `RowHeight[ ]`.

New surface:

* `pps::compute_col_bd(col_widths) -> Vec<u32>` — §6.5.1 eq. (26).
  Pure module-level prefix-sum over the `ColWidth[ ]` list. Output
  length is `col_widths.len() + 1`. The running sum uses
  `u32::saturating_add` so a malformed over-specified explicit-tile
  `ColWidth[ ]` clamps rather than overflows.
* `pps::compute_row_bd(row_heights) -> Vec<u32>` — §6.5.1 eq. (27).
  Symmetric over the `RowHeight[ ]` list.
* `Pps::col_bd(pic_width_in_ctbs_y) -> Vec<u32>` — instance dispatch
  that feeds `Self::col_widths(pic_width_in_ctbs_y)` into
  `compute_col_bd`. `PicWidthInCtbsY` stays an explicit argument (it
  derives from §7.4.3.1 against the SPS).
* `Pps::row_bd(pic_height_in_ctbs_y) -> Vec<u32>` — instance dispatch
  into `compute_row_bd`, symmetric.

### Spec gap (untouched)

The contested eq. (30) `tile_id_val[ i ][ j ]` index ordering (docs
gap #1470) and the §8.9.8 eq. 1398-1409 `tableNum == 0` branch (docs
gap #1278) are both avoided: eq. (26) / (27) are pure prefix-sums over
the extent lists and reach neither.

### Wiring stance

Same opt-in posture as the round-218 / 223 / 229 / 232 / 237 / 242 /
245 / 249 / 258 / 263 helper rollout: pure functions returning owned
vectors, no behaviour change to existing decoder paths.

### Tests

11 new unit tests (595 total; was 584):

* `round270_col_bd_single_tile_is_zero_and_full_width` — pins the
  `n = 1` two-entry boundary list `[ 0, PicWidthInCtbsY ]`.
* `round270_col_bd_prefix_sum_matches_hand_trace` — pins
  `ColWidth = [ 3, 3, 4 ]` → `ColBd = [ 0, 3, 6, 10 ]`.
* `round270_col_bd_length_is_widths_plus_one` — pins the spec's
  inclusive `0 ..= num_tile_columns_minus1 + 1` range as
  `col_widths.len() + 1` entries, `ColBd[ 0 ] = 0`.
* `round270_col_bd_final_entry_equals_total_width` — sweep confirming
  the last boundary equals `PicWidthInCtbsY` over every
  `(cols_minus1, W)` pair.
* `round270_col_bd_is_strictly_monotonic_for_nonempty_tiles` — pins
  the strictly-increasing boundary invariant for a well-formed grid.
* `round270_col_bd_explicit_branch_matches_widths` — eq. (26) over an
  explicit-spacing `ColWidth = [ 3, 1, 6 ]` → `[ 0, 3, 4, 10 ]`.
* `round270_col_bd_empty_widths_is_single_zero` — defensive
  bare-`[ 0 ]` seed.
* `round270_row_bd_prefix_sum_matches_hand_trace` — eq. (27) mirror.
* `round270_row_bd_final_entry_equals_total_height` — eq. (27) sweep
  mirror.
* `round270_col_bd_matches_pps_dispatch` /
  `round270_row_bd_matches_pps_dispatch` — the free function and the
  `Pps` instance method agree on every input.

### Disclaimer

`pps::compute_col_bd` / `pps::compute_row_bd` / `Pps::col_bd` /
`Pps::row_bd` are derived from ISO/IEC 23094-1:2020 §6.5.1 eq. (26)
and eq. (27). All truth came from `docs/video/evc/evc.txt`.

## Round-263 status

Round 263 completes the §6.4 single-block availability quartet by
landing the **base §6.4.1 derivation** (`derive_neighbour_availability`)
as the natural complement to round-242's §6.4.2 `availLR` wrapper
and round-258's §6.4.3 / §6.4.4 derivations. §6.4.1 is the
derivation §6.4.2 invokes inline at the two left / right luma
locations to produce its `availableL` / `availableR` booleans, and
the lowest common shape — six bullets — shared with §6.4.3 and
§6.4.4.

§6.4.1 disqualifies a neighbour at luma location `(xNbY, yNbY)`
when any of six bullets holds: different tile, `xNbY < 0`,
`yNbY < 0`, `xNbY ≥ pic_width_in_luma_samples`,
`yNbY ≥ pic_height_in_luma_samples`, or
`IsCoded[xNbY][yNbY] == FALSE`. Otherwise the neighbour is
available. The function takes the tile-boundary boolean and the
`IsCoded[][]` raster lookup as inputs, matching the pre-resolved-
predicate shape established by §6.4.3 / §6.4.4 in round 258.

The structural relationship across the quartet is:

* §6.4.1 — six bullets: tile + two negative-index + two
  picture-extent + `IsCoded[][]`.
* §6.4.3 = §6.4.1 + bullet 7 (intra / IBC neighbour disqualifies).
* §6.4.4 = §6.4.3 − bullet 1 (ALF reaches across tile boundaries
  when the §7.4.5 cross-tile flag permits it).
* §6.4.2 = packed token `availableL + availableR * 2` over two
  §6.4.1 outputs at `(xCurr − 1, yCurr)` and `(xCurr + nCbW, yCurr)`.

New surface:

* `neighbour::derive_neighbour_availability(x_nb_y, y_nb_y,
  pic_width_in_luma_samples, pic_height_in_luma_samples,
  neighbour_in_different_tile, is_coded) -> bool` — §6.4.1
  derivation. The five geometric bullets (tile-different, two
  negative-index bounds, two picture-extent bounds) are resolved
  from the explicit inputs; the one raster bullet
  (`IsCoded[][]`) is taken as an already-looked-up boolean, in
  the same shape the §6.4.3 / §6.4.4 wrappers established.

### Wiring stance

Same opt-in posture as the round-218 / 223 / 229 / 232 / 237 /
242 / 245 / 249 / 258 helper rollouts: pure function, no
behaviour change to existing decoder paths. Callers that already
inline the §6.4.1 bullets (e.g. the AMVP builder in `inter.rs`,
the ALF classifier in `alf.rs`, anywhere the slice walker checks
"is this neighbour available") keep doing so; a follow-up round
can rebind them once the §6.4 helpers are exhaustively in place.

### Tests

11 new unit tests (584 total; was 573) plus 1 new doc-test:

§6.4.1 (base neighbouring block availability):

* `round263_eq641_all_good_interior_is_available` — pins the
  baseline "all-bullets-pass" return.
* `round263_eq641_different_tile_disqualifies` — bullet 1.
* `round263_eq641_negative_coords_disqualify` — bullets 2-3 (each
  axis independently, plus combined).
* `round263_eq641_oob_picture_extent_disqualifies` — bullets 4-5,
  pinning both the equality (inclusive `>=`) and the
  strict-greater cases.
* `round263_eq641_uncoded_neighbour_disqualifies` — bullet 6.
* `round263_eq641_each_bullet_independently_disqualifies` —
  baseline-good neighbour, flip each bullet one at a time, confirm
  the result becomes `FALSE` in each case.
* `round263_eq641_origin_is_in_bounds` — top-left CTB (x=0, y=0)
  is itself in-bounds when the picture is non-empty.
* `round263_eq641_composes_into_eq642` — drives §6.4.1 at the two
  §6.4.2 locations (`xCurr − 1`, `xCurr + nCbW`) and feeds the
  outputs into `derive_avail_lr`, confirming the §6.4.1 → §6.4.2
  composition resolves to the expected `Lr10` token (left
  in-bounds + right at picture extent, `sps_suco_flag = 0`).

§6.4.1 vs §6.4.3 vs §6.4.4 structural contrast:

* `round263_eq641_and_eq643_agree_when_inter_neighbour` —
  11-tuple sweep across origin / interior / OOB / uncoded /
  inclusive-boundary cases confirms §6.4.1 and §6.4.3 return the
  same boolean whenever the §6.4.3 intra/IBC bullet input is
  `false`. Pins the spec relationship §6.4.3 = §6.4.1 + bullet 7.
* `round263_eq641_and_eq643_diverge_only_on_intra_bullet` — same
  otherwise-good neighbour, §6.4.3 with `neighbour_is_intra_or_ibc
  = true` returns FALSE, §6.4.1 returns TRUE. Pins the one
  structural input difference between the two derivations.
* `round263_eq641_and_eq644_agree_on_geometry_core` — 9-tuple
  sweep against the five-bullet "geometry + IsCoded" core,
  confirming §6.4.1 (tile-bullet = false) and §6.4.4 (intra-bullet
  = false) return the same boolean. Pins the structural
  intersection §6.4.1 ∩ §6.4.4.

### Disclaimer

`neighbour::derive_neighbour_availability` is derived from
ISO/IEC 23094-1:2020 §6.4.1. All truth came from
`docs/video/evc/evc.txt`.

## Round-258 status

Round 258 lands the two companion **single-block availability
derivations** in §6.4 — §6.4.3 (motion-vector candidate availability)
and §6.4.4 (ALF availability) — as the natural rounders for the
round-242 §6.4.2 `availLR` work. Both produce a single `availableN`
boolean from the spec's bullet list, taking caller-on-hand
predicates as inputs in the same shape the existing §6.4.2 wrapper
does.

§6.4.3 disqualifies a neighbour at luma location `(xNbY, yNbY)`
when any of seven bullets holds: different tile, `xNbY < 0`,
`yNbY < 0`, `xNbY ≥ pic_width_in_luma_samples`,
`yNbY ≥ pic_height_in_luma_samples`,
`IsCoded[xNbY][yNbY] == FALSE`, or the neighbour is coded in
intra / IBC mode. The last bullet is the §6.4.3-defining
predicate — an intra-coded neighbour carries no motion vector, so
its slot in the merge / AMVP / affine candidate list is
structurally vacant.

§6.4.4 is §6.4.3 minus the tile-boundary bullet: the ALF filter
deliberately reaches across tile boundaries when
§7.4.5 `alf_loop_filter_across_tiles_enabled_flag` permits it, so
§6.4.4 never disqualifies a neighbour for sitting in a different
tile (the flag itself is consulted by the ALF caller, not inside
§6.4.4).

New surface:

* `neighbour::derive_mv_candidate_availability(x_nb_y, y_nb_y,
  pic_width_in_luma_samples, pic_height_in_luma_samples,
  neighbour_in_different_tile, is_coded, neighbour_is_intra_or_ibc)
  -> bool` — §6.4.3 derivation. The five geometric bullets
  (tile-different, two negative-index bounds, two picture-extent
  bounds) are resolved from the explicit inputs; the two raster
  bullets (`IsCoded[][]`, intra/IBC mode flag) are taken as
  already-looked-up booleans, mirroring the §6.4.2 contract.
* `neighbour::derive_alf_availability(x_nb_y, y_nb_y,
  pic_width_in_luma_samples, pic_height_in_luma_samples, is_coded,
  neighbour_is_intra_or_ibc) -> bool` — §6.4.4 derivation. Same
  shape as §6.4.3 minus the `neighbour_in_different_tile` input.

### Wiring stance

Same opt-in posture as the round-218 / 223 / 229 / 232 / 237 /
242 / 245 / 249 helper rollouts: pure functions, no behaviour
change to existing decoder paths. Callers that already inline
their own §6.4.3 / §6.4.4 bullets (e.g. the AMVP builder in
`inter.rs` and the ALF luma / chroma classifiers in `alf.rs`)
keep doing so; a follow-up round can rebind them once the §6.4
helpers are exhaustively in place.

### Tests

18 new unit tests (573 total; was 555) plus 2 new doc-tests:

§6.4.3 (motion-vector candidate availability):

* `round258_eq643_all_good_interior_is_available` — pins the
  baseline "all-bullets-pass" return.
* `round258_eq643_different_tile_disqualifies` — bullet 1.
* `round258_eq643_negative_coords_disqualify` — bullets 2-3 (each
  axis independently, plus combined).
* `round258_eq643_oob_picture_extent_disqualifies` — bullets 4-5,
  pinning both the equality (exclusive `<` vs inclusive `>=`
  boundary) and the strict-greater cases.
* `round258_eq643_uncoded_neighbour_disqualifies` — bullet 6.
* `round258_eq643_intra_or_ibc_neighbour_disqualifies` —
  bullet 7 (the §6.4.3-defining predicate).
* `round258_eq643_each_bullet_independently_disqualifies` —
  baseline-good neighbour, flip each bullet one at a time, confirm
  the result becomes `FALSE` in each case.
* `round258_eq643_origin_is_in_bounds` — top-left CTB (x=0, y=0)
  is itself in-bounds when the picture is non-empty, pinning the
  zero-coordinate edge of the negative-index bullets.

§6.4.4 (ALF availability):

* `round258_eq644_all_good_interior_is_available` — baseline.
* `round258_eq644_does_not_consult_tile_boundary` — pins the
  defining structural difference: §6.4.4 returns TRUE for the same
  neighbour configuration that §6.4.3 returns FALSE for when the
  only failing §6.4.3 bullet is bullet-1.
* `round258_eq644_negative_coords_disqualify` — bullets 1-2.
* `round258_eq644_oob_picture_extent_disqualifies` — bullets 3-4.
* `round258_eq644_uncoded_neighbour_disqualifies` — bullet 5.
* `round258_eq644_intra_or_ibc_neighbour_disqualifies` —
  bullet 6.
* `round258_eq644_each_bullet_independently_disqualifies` —
  baseline + per-bullet flip sweep.
* `round258_eq644_origin_is_in_bounds` — origin pin matching
  §6.4.3.

§6.4.3 vs §6.4.4 structural contrast:

* `round258_eq643_and_eq644_agree_when_same_tile` — 10-tuple
  sweep across origin / interior / OOB / intra / uncoded /
  inclusive-boundary cases confirms §6.4.3 and §6.4.4 return the
  same boolean whenever the §6.4.3 tile-different input is `false`.
* `round258_eq643_and_eq644_diverge_only_on_tile_bullet` — same
  otherwise-good neighbour, §6.4.3 with `neighbour_in_different_tile
  = true` returns FALSE, §6.4.4 returns TRUE. Pins the one
  structural input difference between the two derivations.

### Disclaimer

`neighbour::derive_mv_candidate_availability` and
`neighbour::derive_alf_availability` are derived from
ISO/IEC 23094-1:2020 §6.4.3 and §6.4.4 respectively. All truth
came from `docs/video/evc/evc.txt`.

## Round-249 status

Round 249 lands the **§6.5.1 `ColWidth[ i ]` and `RowHeight[ j ]`
tile-extent derivations** (eq. 24 / eq. 25), as new module-level
functions and `Pps` instance methods. These are the pre-eq.(30)
primitives the §6.5.1 `TileId[ ]` walk consumes; eq. (30)'s
contested `tile_id_val` index ordering (docs gap #1470) is
deliberately left untouched.

§6.5.1 eq. (24) produces the per-tile-column width list
`ColWidth[ i ]` (in CTBs) of length `num_tile_columns_minus1 + 1`,
selecting between two branches by `uniform_tile_spacing_flag`:

* **Uniform branch.** Closed-form integer-division split of
  `PicWidthInCtbsY` across `n` columns:

  ```text
  ColWidth[ i ] = floor( ( i + 1 ) · PicWidthInCtbsY / n )
                - floor(     i     · PicWidthInCtbsY / n )
  ```

  The differenced-floor form is exact-cover by construction: the
  per-column widths sum to `PicWidthInCtbsY`.

* **Explicit branch.** `ColWidth[ i ] = tile_column_width_minus1[ i ]
  + 1` for `i < n − 1`, and the last column carries the residual
  `PicWidthInCtbsY − Σ ColWidth[ < n − 1 ]`. The running residual is
  computed with `u32::saturating_sub` so a malformed stream that
  over-specifies the explicit widths produces a clamped `0` rather
  than panicking.

§6.5.1 eq. (25) does the symmetric thing for tile rows, against
`PicHeightInCtbsY` and the §7.4.3.2 `tile_row_height_minus1[ ]`
list.

New surface:

* `pps::compute_col_widths(uniform_tile_spacing_flag,
  num_tile_columns_minus1, tile_column_width_minus1,
  pic_width_in_ctbs_y) -> Vec<u32>` — §6.5.1 eq. (24). Pure
  module-level function, primitive inputs.
* `pps::compute_row_heights(uniform_tile_spacing_flag,
  num_tile_rows_minus1, tile_row_height_minus1,
  pic_height_in_ctbs_y) -> Vec<u32>` — §6.5.1 eq. (25). Symmetric.
* `Pps::col_widths(pic_width_in_ctbs_y: u32) -> Vec<u32>` —
  instance dispatch into `compute_col_widths` that pulls the
  three §7.4.3.2 inputs (uniform-spacing flag, columns count,
  explicit-widths slice) off the parsed PPS. `PicWidthInCtbsY`
  stays an explicit argument because it derives from §7.4.3.1
  against the SPS.
* `Pps::row_heights(pic_height_in_ctbs_y: u32) -> Vec<u32>` —
  instance dispatch into `compute_row_heights`, symmetric.

### Wiring stance

Same opt-in posture as the round-218 / 223 / 229 / 232 / 237 /
242 / 245 helper rollout: pure functions returning owned vectors,
no behaviour change to existing decoder paths.

### Tests

13 new unit tests (555 total; was 542):

* `round249_col_widths_single_tile_returns_full_picture` — pins
  the eq. (24) single-tile invariant: `ColWidth = [ W ]` for
  `n = 1`.
* `round249_col_widths_uniform_two_columns_even_split` — pins the
  uniform branch for `n = 2, W = 10`.
* `round249_col_widths_uniform_three_columns_floor_division` —
  pins `n = 3, W = 10` against the hand-computed
  `[ 3, 3, 4 ]` distribution, verifying the floor-difference form
  hands the rounding remainder to the last column.
* `round249_col_widths_uniform_covers_pic_width_exactly` — sweep
  invariant: `Σ ColWidth = PicWidthInCtbsY` for every
  `(cols_minus1, W) ∈ [0, 8] × { 1, 2, 5, 10, 17, 32, 64, 100 }`.
* `round249_col_widths_explicit_branch_pins_eq24_remainder` —
  pins the explicit branch for
  `n = 3, tile_column_width_minus1 = [ 2, 0 ], W = 10` against
  the hand-derived `[ 3, 1, 6 ]`.
* `round249_col_widths_explicit_branch_two_cols_residual` — pins
  `n = 2, tile_column_width_minus1 = [ 3 ], W = 10` → `[ 4, 6 ]`.
* `round249_col_widths_explicit_overflow_saturates_residual` —
  confirms the malformed-input saturation: with explicit width
  `100` against `W = 5`, the residual is clamped at `0` rather
  than underflowing.
* `round249_row_heights_uniform_two_rows_even_split` — eq. (25)
  mirror of the two-columns case.
* `round249_row_heights_uniform_covers_pic_height_exactly` —
  eq. (25) sweep invariant.
* `round249_row_heights_explicit_branch_pins_eq25_remainder` —
  eq. (25) hand-derivation pin.
* `round249_compute_col_widths_uniform_matches_pps_dispatch` —
  the free function and the `Pps::col_widths` instance method
  agree on every uniform input.
* `round249_compute_row_heights_uniform_matches_pps_dispatch` —
  symmetric agreement check for `Pps::row_heights`.
* `round249_col_widths_zero_pic_width_returns_all_zeros` —
  defensive: zero-CTB picture width gives all-zero list of the
  spec-mandated length.

### Disclaimer

`pps::compute_col_widths` / `pps::compute_row_heights` /
`Pps::col_widths` / `Pps::row_heights` are derived from
ISO/IEC 23094-1:2020 §6.5.1 eq. (24) and eq. (25). All truth
came from `docs/video/evc/evc.txt`.

## Round-245 status

Round 245 lands the **§6.5.3 inverse scan order 1D array
initialization process** (eq. 34) plus the matching public §6.5.2
forward zig-zag derivation (eq. 33), as a new `scan` module.

§6.5.2 produces the forward map `ScanOrder[ sPos ] = rPos`, where
`sPos` is the zig-zag scan position (`0 ..= blkWidth · blkHeight − 1`)
and `rPos` is the row-major raster offset within the block. The
walk starts at the top-left corner and visits anti-diagonals (lines
of constant `x + y`); odd anti-diagonals are traversed from the
top-right endpoint toward the bottom-left, even anti-diagonals the
opposite way.

§6.5.3 then inverts the §6.5.2 map (eq. 34):

```text
for( pos = 0; pos < blkWidth * blkHeight; pos++ ) {
    inverseScan[ forwardScan[ pos ] ] = pos                       (34)
}
```

so `InvScanOrder[ rPos ]` returns the zig-zag scan position `sPos`
that visits raster offset `rPos`. By eq. (34), the two maps satisfy
the round-trip identities

```text
inverseScan[ forwardScan[ pos ] ] = pos      for every pos
forwardScan[ inverseScan[ pos ] ] = pos      for every pos
```

New surface:

* `scan::zig_zag_scan(blk_w, blk_h) -> Vec<u32>` — §6.5.2 eq. 33.
  Returns `blkWidth · blkHeight` entries, each the row-major raster
  offset `y * blk_w + x` of the block sample visited at scan position
  `sPos`. Pin-tested against hand-traced 4×4 / 2×2 / 4×2 walks plus
  the §7.4.3.1 invariant that the result is a permutation of
  `0 ..= blk_w · blk_h − 1`.
* `scan::inverse_scan(blk_w, blk_h) -> Vec<u32>` — §6.5.3 eq. 34.
  Returns `blkWidth · blkHeight` entries, each the zig-zag scan
  position `sPos` whose §6.5.2 walk lands at raster offset `rPos`.
  Pin-tested against the 4×4 / 4×2 hand-traces and against the
  eq. (34) round-trip identity sweep across every TB size from 2×2
  up to 16×16.

### Spec usage

§7.4.3.1 (page 64) directs the decoder to build
`ScanOrder[ log2TbWidth ][ log2TbHeight ][ sPos ]` and
`InvScanOrder[ log2TbWidth ][ log2TbHeight ][ rPos ]` for every
`log2TbWidth, log2TbHeight ∈ 1 ..= MaxTbLog2SizeY` by invoking
§6.5.2 / §6.5.3 with `blkWidth = 1 << log2TbWidth`,
`blkHeight = 1 << log2TbHeight`. The round-245 entry points are
the building blocks of that population pass; the per-TB-size table
cache is a follow-up.

### Wiring stance

Same opt-in posture as the round-218 / 223 / 229 / 232 / 237 / 242
helper rollout: pure functions returning owned vectors. The
`slice_data::decode_residual_coding_rle` walker continues to call
its in-module zig-zag builder; rebinding it to `scan::zig_zag_scan`
is a follow-up.

### Tests

12 new unit tests (542 total; was 530):

* `round245_zig_zag_4x4_matches_hand_trace` — pins the 16-entry
  forward map produced by the §6.5.2 anti-diagonal walk for the
  canonical 4×4 transform block.
* `round245_zig_zag_2x2_matches_hand_trace` — pins the §6.5.2 walk
  for the smallest legal §7.4.3.1 transform block (`log2TbW =
  log2TbH = 1`).
* `round245_zig_zag_4x2_non_square_matches_hand_trace` — pins the
  §6.5.2 walk for a non-square block, exercising the `(line −
  (blk_w − 1))` and `(line − (blk_h − 1))` anchor terms with
  unequal dimensions.
* `round245_zig_zag_is_permutation_for_every_tb_size` — sweeps
  every `(log2TbW, log2TbH) ∈ [1, 4]^2` pair and asserts the §6.5.2
  output is a permutation of `0 ..= blk_w · blk_h − 1`.
* `round245_zig_zag_visits_anti_diagonals_in_order` — same sweep,
  asserts the structural invariant that the walk visits points by
  non-decreasing `x + y`.
* `round245_zig_zag_empty_blocks_return_empty_vec` — defensive
  zero-dimension behaviour.
* `round245_inverse_scan_4x4_round_trips_forward` — eq. (34)
  identity for the 4×4 block.
* `round245_inverse_scan_4x4_dc_at_raster_zero_is_scan_zero` —
  pins the DC bridge: scan position 0 lives at raster 0, so
  `InvScanOrder[ 0 ] == 0`.
* `round245_inverse_scan_is_bijection_with_forward` — sweeps every
  `(log2TbW, log2TbH) ∈ [1, 4]^2` pair and asserts the two-way
  round-trip identity.
* `round245_inverse_scan_4x4_pins_eq34_values` — hand-applies
  eq. (34) to the 4×4 forward map and pins all 16 inverse entries.
* `round245_inverse_scan_4x2_pins_eq34_values` — same, for a
  non-square block.
* `round245_inverse_scan_empty_blocks_return_empty_vec` — defensive
  zero-dimension behaviour mirroring §6.5.2.

### Disclaimer

`scan::zig_zag_scan` / `scan::inverse_scan` are derived from
ISO/IEC 23094-1:2020 §6.5.2 eq. (33) and §6.5.3 eq. (34). All
truth came from `docs/video/evc/evc.txt`. The §6.5.2 algorithm
listing in the FDIS includes an even-branch typo whose corrected
form is empirically pinned by the round-5 4×4 hand-trace (still
checked by `slice_data::tests::zigzag_scan_4x4_matches_spec`) and
by the round-245 `round245_zig_zag_4x4_matches_hand_trace`
fixture, which together fix the canonical 4×4 zig-zag table from
the anti-diagonal closed form.

## Round-242 status

Round 242 lands the **§6.4.2 `availLR` derivation** (eq. 23) plus
the four named tokens `LR_00` / `LR_10` / `LR_01` / `LR_11` from the
section's closing paragraph, as a new `neighbour` module.

§6.4.2 turns two per-neighbour availability booleans into a packed
`availLR` value used by every `sps_suco_flag = 1`-aware merge /
AMVP / partitioning derivation (§8.5.2.4.5.2 `DefaultRefIdxLX`,
§8.5.2.3.2 spatial merging, §8.5.3.5 affine CP MVP, etc.). The
formula is the closed-form

```text
availLR = availableL + availableR * 2                            (23)
```

with the four output tokens `LR_LR` named so that the first digit
is the left-neighbour availability and the second the right.

New surface:

* `neighbour::AvailLr` — `repr(u8)` enum carrying the eq. (23)
  integer in its discriminant (`Lr00 = 0`, `Lr10 = 1`, `Lr01 = 2`,
  `Lr11 = 3`). `Copy + Clone + Debug + PartialEq + Eq + Hash`.
* `neighbour::AvailLr::available_l(self) -> bool` /
  `available_r(self) -> bool` — projections back to the eq. (23)
  input booleans.
* `neighbour::AvailLr::as_u8(self) -> u8` /
  `from_u8(u8) -> Option<Self>` — the eq. (23) integer view.
  `from_u8` rejects every `value > 3`; the spec never produces
  such a token.
* `neighbour::AvailLr::is_suco_consistent(self, sps_suco_flag: u8) -> bool`
  — the closing-paragraph invariant. With `sps_suco_flag == 0`,
  `availLR ∈ { LR_00, LR_10 }`; with `sps_suco_flag != 0`, every
  token is reachable.
* `neighbour::derive_avail_lr(available_l: bool, available_r: bool) -> AvailLr`
  — eq. (23). Caller invokes §6.4.1 at the spec-mandated left
  (`xCurr − 1, yCurr`) and right (`xCurr + nCbW, yCurr`) luma
  locations and feeds the two booleans in.

### §6.4.1 deliberately out of scope

§6.4.1 (neighbouring-block availability) is **not** wrapped this
round. Its rule set mixes tile-boundary lookup, the `IsCoded[][]`
raster, and the "intra / IBC mode" predicate — three pieces of
state that already live on the slice walker. The §6.4.2 entry
point mirrors the spec's invocation pattern: callers compute
`availableL` / `availableR` via §6.4.1 themselves, then call
`derive_avail_lr`.

### Wiring stance

Same opt-in posture as the round-218 / 223 / 229 / 232 / 237
helper rollout: pure functions + a typed token, no behaviour
change to existing decoder paths. The AMVP builder in `inter.rs`
already derives its own left/right availability via `Option<MotionVector>`
returns; §8.5.2.4.5.2 / §8.5.3.5 will consume `AvailLr` once they
land.

### Tests

11 new unit tests (530 total; was 519):

* `round242_eq23_both_unavailable_is_lr00` — eq. (23)
  `0 + 0 * 2 == 0`.
* `round242_eq23_left_only_is_lr10` — eq. (23) `1 + 0 * 2 == 1`,
  confirms the `LR_LR` label convention.
* `round242_eq23_right_only_is_lr01` — eq. (23) `0 + 1 * 2 == 2`,
  confirms `availableR` is weighted by `* 2`.
* `round242_eq23_both_available_is_lr11` — eq. (23)
  `1 + 1 * 2 == 3`.
* `round242_projections_invert_derivation` — `available_l` /
  `available_r` reproduce the eq. (23) input booleans for every
  truth-table row.
* `round242_as_u8_matches_eq23_formula` — the packed integer
  matches the eq. (23) formula at every row.
* `round242_from_u8_round_trip` — `from_u8(variant.as_u8()) ==
  Some(variant)` for every legal value.
* `round242_from_u8_rejects_out_of_range` — values 4..=255 surface
  `None`.
* `round242_suco_off_admits_only_lr00_and_lr10` — closing-paragraph
  invariant for `sps_suco_flag == 0`.
* `round242_suco_on_admits_every_token` — closing-paragraph
  invariant for `sps_suco_flag == 1`.
* `round242_discriminants_match_spec_table` — regression pinning
  the `repr(u8)` discriminants to eq. (23).

### Disclaimer

`AvailLr` / `derive_avail_lr` are derived from ISO/IEC
23094-1:2020 §6.4.2 eq. (23) and the four `LR_xx` tokens spelled
out in that section's closing paragraph. No external
implementation was consulted; the formula is a direct
transcription of the spec's two text bullets and the closing
table.

## Round-237 status

Round 237 lands a **typed tile-grid iterator** on `Pps` plus three
typed picture-tile counters, grounded in ISO/IEC 23094-1:2020 §6.5.1
eq. (30) and §7.4.3.2.

The §6.5.1 derivation processes that build `TileId[ ]`, `ColWidth[ ]`
/ `RowHeight[ ]` and `NumCtusInTile[ ]` all share the same outer
loop:

```
for( j = 0, tileIdx = 0; j <= num_tile_rows_minus1; j++ )
   for( i = 0; i <= num_tile_columns_minus1; i++, tileIdx++ )
      ...
```

Round 237 captures that outer loop as a reusable iterator yielding
one `TileGridCoord { tile_idx, tile_row_j, tile_col_i }` per tile,
in eq. (30) raster-tile order. The inner two loops that populate
`TileId[ CtbAddrRsToTs[ … ] ]` per CTB stay caller-controlled — they
need `ColBd[ ]` / `RowBd[ ]` / `CtbAddrRsToTs[ ]` which are derived
once the rest of §6.5.1 lands.

New surface:

* `pps::TileGridCoord` — `(tile_idx, tile_row_j, tile_col_i)` triple
  matching the §6.5.1 eq. (30) local variable names. `Copy + Clone +
  Debug + PartialEq + Eq`.
* `pps::TileGridCoordIter` — `ExactSizeIterator` returned by
  `Pps::tile_grid_coords`. `size_hint` and `len` stay tight against
  the remaining-tile count.
* `Pps::num_tile_columns(&self) -> u32` / `num_tile_rows(&self) -> u32`
  — the `num_tile_*_minus1 + 1` adapter pair. Always at least 1, even
  for the single-tile-in-pic case.
* `Pps::num_tiles_in_pic(&self) -> u32` — `NumTilesInPic` from §6.5.1.
  Product of the two axis counts; the parser's `MAX_TILES_PER_DIM`
  bound (256 per axis) keeps the product inside `u32`.
* `Pps::tile_grid_coords(&self) -> TileGridCoordIter` — the eq. (30)
  outer-loop iterator. Yields `tile_idx` linearly from `0` to
  `NumTilesInPic - 1`, with `tile_row_j * num_tile_columns +
  tile_col_i == tile_idx` as the row-major packing identity.

### Spec gap (deferred)

The explicit-id branch of eq. (30) accesses `tile_id_val[ i ][ j ]`
where the eq's local `i` binds to the column index and `j` to the
row. The §7.4.3.2 prose for `tile_id_val[ i ][ j ]` defines the
first dimension as the **row** and the second as the **column**.
These two binding orders contradict each other.

`Pps::tile_grid_coords` deliberately stops short of resolving
`tile_id` for that reason; it surfaces only the unambiguous
`(tile_idx, tile_row_j, tile_col_i)` triple. The round's followups
note flags this for the docs collaborator.

### Wiring stance

Same posture as the round-187 / 193 / 195 / 201 / 207 / 213 / 218 /
223 / 229 / 232 helper rollout: opt-in helper, no behaviour change
to existing decoder paths. The §6.5.1 derivation continues to live
in the decoder's slice-walker; `tile_grid_coords` exists for it (and
future per-tile iteration callers) to consume once that path is
wired through.

### Tests

10 new unit tests (519 total; was 509):

* `round237_num_tiles_single_tile_picture` — the
  `single_tile_in_pic_flag == 1` shape yields exactly 1.
* `round237_num_tiles_two_by_one_grid` — 2 columns × 1 row.
* `round237_num_tiles_three_by_two_grid` — 3 columns × 2 rows.
* `round237_tile_grid_coords_single_tile` — degenerate iterator
  yields exactly one coord at the origin.
* `round237_tile_grid_coords_two_by_one_order` — confirms `i`
  (columns) is the inner loop and `tile_idx` advances 0, 1.
* `round237_tile_grid_coords_two_by_two_raster_order` — the
  eq. (30) outer-loop order (0,0), (0,1), (1,0), (1,1).
* `round237_tile_grid_coords_three_by_two_full_walk` — exhaustive
  row-major walk of a 3×2 grid.
* `round237_tile_grid_iterator_exhausts_to_none` — fused-style
  `None` after exhaustion.
* `round237_tile_grid_iterator_size_hint` — `size_hint` + `len`
  stay tight as the iterator consumes.
* `round237_tile_idx_matches_row_col_packing` — the row-major
  packing identity `tile_idx == row * num_cols + col`.

### Disclaimer

`TileGridCoord` / `tile_grid_coords` are derived from ISO/IEC
23094-1:2020 §6.5.1 eq. (30) and §7.4.3.2's `num_tile_*_minus1`
semantics. No external implementation was consulted; the iterator
is a direct transcription of the spec's outer two loop levels.

## Round-232 status

Round 232 lands the **§8.5.2.3.10 motion vector prediction redundancy
check** in `inter.rs`. Every §8.5.2.3.x merge-candidate append path
(§8.5.2.3.1 – §8.5.2.3.8) finishes by invoking the §8.5.2.3.10 trim
loop with the just-grown list and the current `numCurrMergeCand`. The
trim compares the tail entry (`mergeCandList[numCurrMergeCand − 1]`)
against every prior entry; on the first match the tail's slot is
reclaimed by decrementing the count, otherwise the count stays put.

The matching predicate composes the four ordered conditions the spec
spells out:

1. Same number of available reference lists.
2. Same available reference list indices (which of L0 / L1 is used).
3. Same `refIdxLX` in each available list.
4. Same `mvLX` in each available list.

The "corresponding to available reference lists" qualifier means
inactive-list slots are dropped from the compare: two candidates with
`predFlagL1 = 0` match even when their residual `refIdxL1` / `mvL1`
values disagree.

New surface:

* `inter::MergeCand` — compact value type for §8.5.2.3.x merge
  entries (`pred_flag_lX`, `ref_idx_lX`, `mv_lX` per list).
* `inter::merge_cand_matches(a, b) -> bool` — §8.5.2.3.10 four-step
  predicate as a pure function. Reflexive + symmetric on the
  active-list projection.
* `inter::merge_cand_redundancy_check(merge_cand_list,
  num_curr_merge_cand) -> Result<usize>` — the trim loop. Returns the
  updated `numCurrMergeCand`. No-op when `num_curr_merge_cand ≤ 1`
  (per the spec's outer guard); first-match short-circuits with
  `count − 1`; oversized counts surface `Error::Unsupported`.

### Wiring stance

Same posture as the rest of the §8.5.2.3 helper rollout (rounds 187 /
193 / 195 / 201 / 207 / 213 / 218 / 223 / 229): opt-in helper, no
behaviour change to existing decoder paths. The §8.5.2.3.x
merge-candidate-list builder remains the open dispatcher pending the
surrounding §8.5.2.3.1 – §8.5.2.3.8 spatial / temporal / HMVP /
zero-MV append paths; this round closes the dedup primitive each of
those append paths needs at their tail.

### Tests

16 new unit tests (509 total; was 493):

* `round232_pred_flag_bitmask_mismatch_blocks_match` — step (1) /
  (2): a bipred candidate cannot match an L0-only candidate.
* `round232_ref_idx_mismatch_blocks_match` — step (3): single-list
  refIdx disagreement defeats the match.
* `round232_mv_component_mismatch_blocks_match` — step (4): a single
  MV component disagreement defeats the match.
* `round232_inactive_list_fields_are_ignored` — the spec's
  "corresponding to available reference lists" qualifier: residual
  junk in inactive-list slots is masked out.
* `round232_pre_test_no_op_when_count_le_1` — outer guard.
* `round232_duplicate_tail_drops_count` — happy path: count − 1.
* `round232_new_tail_preserves_count` — happy path: count unchanged.
* `round232_first_duplicate_short_circuits_scan` — the spec's exit
  predicate fires at the first match, not at every match.
* `round232_penultimate_duplicate_decrements` — boundary at
  `candIndx == numCurrMergeCand − 2`.
* `round232_two_element_duplicate` /
  `round232_two_element_distinct` — smallest non-trivial cases.
* `round232_bipred_full_match_drops_count` /
  `round232_bipred_l1_only_difference_preserves_count` — bipred
  candidates require BOTH lists to agree.
* `round232_oversize_count_errors` — caller-bug surface.
* `round232_predicate_reflexive` / `round232_predicate_symmetric` —
  algebraic identities the spec's "have all the following conditions
  met" wording implies.

### Disclaimer

`merge_cand_redundancy_check` is derived from the §8.5.2.3.10 ordered
steps in ISO/IEC 23094-1:2020. No external implementation was
consulted; the algorithm is straight-line transcription of the four
matching conditions plus the spec's bounded-step exit clause.

## Round-229 status

Round 229 lands the **§8.5.2.3.9 entry-process** signed POC scaling
primitives in `inter.rs`. The §8.5.2.3.9 entry process (eqs. 531-590)
reassigns the selected merge candidate's `(refIdxLX, predFlagLX[0][0])`
and pre-scales one list's motion vector against the other when
`mmvd_group_idx ∈ { 1, 2 }`. Across its many sub-branches it leans on
two arithmetic shapes that round 229 captures as pure helpers:

* `distScaleFactor = ( currPocDiffNum << 5 ) / currPocDiffDen` — eqs.
  542 / 551 / 559 / 571 / 580 / 588. **Signed** in both operands
  (unlike round-223's `mmvd_dist_scale_factor`, which pre-absolutes
  the POC differences before the bipred-tail division).
* `mMv[k] = Clip3(-32768, 32767, Sign(s*v) * ((Abs(s*v) + 16) >> 5))`
  — eqs. 543 / 544 / 552 / 553 / 560 / 561 / 572 / 573 / 581 / 582 /
  589 / 590. **Round-toward-zero** with a half-up magnitude bias.
  Distinct from the round-223 bipred-tail form
  `Clip3(-32768, 32767, (s*v + 16) >> 5)`, which rides arithmetic
  right shift and rounds negatives toward `-infinity`. For most
  operand pairs both forms agree at the i32 boundary, but we mirror
  the spec literally on both sides.

Plus the small `MMVD_P_SAME_TARGET_SHIFT = 3` constant for the P-slice
`targetRefIdxL0 == refIdxL0` branch (eqs. 547 / 576 add or subtract 3
from `mMvL0[0]` depending on `mmvd_group_idx`).

New surface:

* `inter::mmvd_signed_dist_scale_factor(poc_diff_num, poc_diff_den)
  -> Result<i32>` — eq. 542 / 551 / 559 / 571 / 580 / 588 signed
  POC-diff scaling factor `(num << 5) / den`. Zero denominator
  surfaces `Error::Unsupported`.
* `inter::mmvd_signed_scale_component(dist_scale_factor, mv_component)
  -> i32` — eqs. 543-590 round-toward-zero per-component scaler with
  the signed-16-bit clip.
* `inter::mmvd_signed_scale_mv(dist_scale_factor, mv) -> MotionVector`
  — pair-axis wrapper.
* `inter::MMVD_P_SAME_TARGET_SHIFT: i32 = 3` — eqs. 547 / 576 magic
  offset.

### Wiring stance

Same posture as rounds 187 / 193 / 195 / 201 / 207 / 213 / 218 / 223:
opt-in helpers, no behaviour change to existing decoder paths.
Baseline streams set `sps_mmvd_flag = 0` so `mmvd_flag` is inferred 0
per §7.4.7 and §8.5.2.3.9 stays unreached. The full §8.5.2.3.9 entry
dispatcher (`mmvd_group_idx` branches, `slice_type` sub-branches,
`refIdxLX` / `predFlagLX` updates) still depends on the
merge-candidate-list builder + `NumRefIdxActive[]` / `RefPicListX`
threading that the round-218 / round-223 followups documented.
Round 229 closes the arithmetic side: the entry-process signed scaling
is now a tested pure helper ready to drop into the eventual dispatcher.

### Tests

11 new unit tests (493 total; was 482):

* `round229_signed_dist_scale_factor_worked_examples` — eqs. 542 /
  551 / 559 / 571 / 580 / 588 against symmetric, L1-closer, L0-closer,
  sign-flipped, and zero-numerator inputs.
* `round229_signed_dist_scale_factor_rejects_zero_denominator` — self-
  protective check at the helper boundary.
* `round229_signed_dist_scale_factor_zero_numerator` — `0 / den == 0`
  for both signs of `den`.
* `round229_signed_scale_component_symmetric_in_sign` — eqs. 543-590
  round-toward-zero form: `f(s, v) = -f(s, -v)` and `f(-s, v) = -f(s,
  v)`, with worked spot checks at `(sf, v) ∈ { (32, 1), (32, -1),
  (-32, 1), (16, 2), (16, -2) }`.
* `round229_signed_scale_component_half_up_threshold` — `(s, v) = (1,
  15)` ⇒ 0, `(1, 16)` ⇒ 1; the boundary at `|product| == 16`.
* `round229_signed_scale_component_clamps_to_signed16` — both clamp
  rails engage on near-i16 products.
* `round229_signed_scale_mv_applies_to_both_axes` — eq. (543, 544)
  pair structure: same `distScaleFactor` on `x` and `y`.
* `round229_signed_scale_component_unit_factor_identity` — `sf = 32`
  is the identity on non-saturating inputs (the "POC distances match"
  case the spec uses to short-circuit several branches).
* `round229_signed_scale_component_zero_inputs` — `sf == 0` or `v ==
  0` short-circuits to 0.
* `round229_p_same_target_shift_constant_pinned` — pins eqs. 547 /
  576 magnitude at 3.
* `round229_signed_scale_agreement_at_zero_threshold` — establishes
  the agreement of round-229 (`mmvd_signed_scale_component`) and
  round-223 (`mmvd_scale_component`) forms at the `|product| < 16`
  zero-rounding regime.

### Documented followups

* The §8.5.2.3.9 entry-process dispatcher (`mmvd_group_idx == 1` and
  `mmvd_group_idx == 2` branches, `slice_type == B` / P sub-branches,
  eqs. 531-590) — the consumer that picks `targetRefIdxL0`, applies
  the `MMVD_P_SAME_TARGET_SHIFT` offset for the same-target case, and
  threads `mmvd_signed_dist_scale_factor` + `mmvd_signed_scale_mv`
  for the differing-target case — still depends on the merge-
  candidate-list builder + `NumRefIdxActive[]` / `RefPicListX`
  threading. Round 229 ships the arithmetic primitives that the
  dispatcher will reuse across eqs. 531-590.
* The bipred-tail eqs. 591-616 (round 223) and entry-process eqs.
  531-590 (round 229) both feed the same final accumulation `mMvLX
  += mMvdLX` with `wrap16` semantics. Once the dispatcher lands, the
  two halves chain.
* The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

## Round-223 status

Round 223 extends the round-218 MMVD groundwork into the **bipred /
one-list-active offset distribution + final accumulation** of
§8.5.2.3.9 (eqs. 591-616). Given a merge candidate's per-list MVs
`mvL0` / `mvL1`, the round-218 axis-aligned `MmvdOffset`, the two
signed POC differences `currPocDiffL{0,1}`, and the
`predFlagL{0,1}[0][0]` flags, the new helper computes the per-list
deltas `mMvdL{0,1}` and returns the post-accumulation `mMvL{0,1}`.

The bipred branch splits on the relative magnitudes of the two POC
differences:

| `Abs(currPocDiffL0)` vs `Abs(currPocDiffL1)` | `mMvdL0` | `mMvdL1` |
|---|---|---|
| equal (eqs. 593-596) | `MmvdOffset` | `MmvdOffset` |
| greater (eqs. 597-601) | scaled via `(\|L1\| << 5) / \|L0\|` | `MmvdOffset` |
| less (eqs. 602-606) | `MmvdOffset` | scaled via `(\|L0\| << 5) / \|L1\|` |

The scaling form is `Clip3(-32768, 32767, (distScaleFactor * mvd + 16)
>> 5)` (round-half-up, 5-bit fractional reduction). When
`currPocDiffL0 * currPocDiffL1 < 0` (eqs. 607-610) the `mMvdL1`
component is negated (the spec's `mMvdL0 = mMvdL0` identities at
eqs. 607/608 are typographic; only L1 is touched).

The one-list-active "Otherwise" branch (eqs. 611-612) gives the active
list `MmvdOffset` and the inactive list zero. Eqs. 613-616 close the
process with `mMvLX += mMvdLX` under the same `wrap16` modular
semantics shared with eqs. 436/439.

New surface:

* `inter::mmvd_dist_scale_factor(abs_poc_num, abs_poc_den) ->
  Result<i32>` — eq. 599 / 604 POC-difference scaling factor
  `(|num| << 5) / |den|`. Inputs are already absolute; zero / negative
  denominator and negative numerator surface `Error::Unsupported`.
* `inter::mmvd_apply_bipred_offset(mv_l0, mv_l1, mmvd_offset,
  curr_poc_diff_l0, curr_poc_diff_l1, pred_flag_l0, pred_flag_l1) ->
  Result<(MotionVector, MotionVector)>` — eqs. 591-616 as one pure
  helper. Both list flags zero is a caller bug and surfaces
  `Error::Unsupported`.

Spec typo recorded: eq. 601 (page 170) is missing a comma between
`32767` and the inner `(distScaleFactor * mMvdL1[ 1 ] + 16) >> 5`
expression. Context-determined identical to eq. 600 on the y component.

### Wiring stance

Same posture as rounds 187 / 193 / 195 / 201 / 207 / 213 / 218: the
new helpers are opt-in. Baseline streams set `sps_mmvd_flag = 0` so
`mmvd_flag` is inferred to 0 per §7.4.7 and none of the round-218 /
round-223 helpers execute. A future Main-profile decode path threads
`sps_mmvd_flag`, the parsed `mmvd_*_idx` syntax elements, the merge
candidate selection, and `DiffPicOrderCnt` over the populated
`RefPicList0 / RefPicList1` into them. Round 223 closes the
arithmetic side of the eqs. 591-616 tail given those inputs; the
upstream eqs. 531-590 (`mmvd_group_idx ∈ { 1, 2 }` ref-list
reassignment + `slice_type == P` sub-branch) wait on the
merge-candidate-list pass.

### Tests

14 new unit tests (482 total; was 468). Coverage:

* `round223_dist_scale_factor_eq599_eq604_form` — worked examples
  across symmetric, L1-closer, L0-closer, zero-numerator cases.
* `round223_dist_scale_factor_rejects_bad_inputs` — zero / negative
  denominator and negative numerator.
* `round223_bipred_symmetric_same_magnitude_same_sign` — eqs. 593-596
  with same-sign POCs.
* `round223_bipred_symmetric_opposite_sign_flips_l1` — eqs. 593-596
  followed by eqs. 607-610.
* `round223_bipred_l1_closer_scales_l0` — eqs. 597-601 with same-sign
  POCs (`sf = 16`, `mMvdL1 = (64, 0)` ⇒ `mMvdL0 = (32, 0)`).
* `round223_bipred_l0_closer_scales_l1` — eqs. 602-606 worked example
  symmetric to the L1-closer case.
* `round223_bipred_l1_closer_opposite_sign_flips_l1_only` — eqs.
  597-601 then eqs. 607-610; only `mMvdL1` is negated.
* `round223_one_list_active_l0_only` / `round223_one_list_active_l1_only`
  — eqs. 611-612 each direction.
* `round223_rejects_both_lists_inactive` — caller-bug guard surfaces
  `Error::Unsupported`.
* `round223_accumulation_wraps_into_signed_16bit` — eqs. 613-616 share
  `wrap16` semantics: `32000 + 1000` wraps to `-32536`.
* `round223_scaled_component_clips_to_signed16` — `Clip3(-32768, 32767,
  ...)` engagement on the scaled component path.
* `round223_scale_component_eq600_form` — eq. 600's `(sf * mv + 16)
  >> 5` form on positive and negative MV components.
* `round223_bipred_symmetric_property_offset_distribution` — same-sign
  / opposite-sign distribution property across four Table-10-style
  offset directions.

## Round-218 status

Round 218 lands the **MMVD (Merge with Motion Vector Difference)**
distance / sign / offset derivation in `inter.rs`. MMVD is gated by
`sps_mmvd_flag = 1` and lets the encoder modify a regular merge
candidate by a small extra MV offset `MmvdOffset[ x0 ][ y0 ][ k ]`
(k ∈ {0, 1}), derived from two enumerative syntax elements via Tables 9
and 10:

| `mmvd_distance_idx` | `MmvdDistance` |
|---|---|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 5 | 32 |
| 6 | 64 |
| 7 | 128 |

| `mmvd_direction_idx` | `MmvdSign[ 0 ]` | `MmvdSign[ 1 ]` |
|---|---|---|
| 0 | +1 |  0 |
| 1 | −1 |  0 |
| 2 |  0 | +1 |
| 3 |  0 | −1 |

Eqs. 133 / 134 combine the two: `MmvdOffset[ k ] = MmvdDistance *
MmvdSign[ k ]`. Note `MmvdOffset` is always axis-aligned (Table 10 has
no row with two non-zero sign components).

New surface:

* `inter::MMVD_DISTANCE_IDX_MAX = 7` — §9.3.3 TR cMax for
  `mmvd_distance_idx`.
* `inter::MMVD_DIRECTION_IDX_MAX = 3` — §9.3.3 FL cMax for
  `mmvd_direction_idx`.
* `inter::MMVD_GROUP_IDX_MAX = 2` — §9.3.3 TR cMax for
  `mmvd_group_idx`.
* `inter::MMVD_MERGE_IDX_MAX = 3` — §9.3.3 TR cMax for
  `mmvd_merge_idx`.
* `inter::mmvd_distance(mmvd_distance_idx) -> Result<i32>` — Table 9
  lookup; out-of-range surfaces `Error::Unsupported`.
* `inter::mmvd_sign(mmvd_direction_idx) -> Result<(i32, i32)>` — Table 10
  lookup; out-of-range surfaces `Error::Unsupported`.
* `inter::mmvd_offset(mmvd_distance_idx, mmvd_direction_idx) ->
  Result<MotionVector>` — eqs. 133 + 134 combined; axis-aligned by
  construction.
* `inter::mmvd_flag_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the single `mmvd_flag` FL bit (Table 50
  carries one trained state per `initType`). `mmvd_flag` is **not** in
  Table 96.
* `inter::mmvd_group_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `mmvd_group_idx` TR bins (Table 51,
  cMax = 2 ⇒ at most 2 prefix bins). Not in Table 96.
* `inter::mmvd_merge_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `mmvd_merge_idx` TR bins (Table 52,
  cMax = 3 ⇒ at most 3 prefix bins). Not in Table 96.
* `inter::mmvd_distance_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `mmvd_distance_idx` TR bins (Table 53,
  cMax = 7 ⇒ at most 7 prefix bins). Not in Table 96.
* `inter::mmvd_direction_idx_ctx_inc(bin_idx) -> Result<usize>` —
  §9.3.4 positional ctxIdxInc for the `mmvd_direction_idx` FL bins
  (Table 54 carries two trained states per `initType`; FL with
  cMax = 3 ⇒ a 2-bit code). Not in Table 96.

### Wiring stance

Same posture as rounds 187 / 193 / 195 / 201 / 207 / 213: these are
opt-in helpers, not behaviour changes to existing decoder paths.
Baseline streams set `sps_mmvd_flag = 0` so `mmvd_flag` is inferred
to 0 per §7.4.7 and the helpers stay dead code. A future Main-profile
decode path threads `sps_mmvd_flag` from `Sps` and the parsed
`mmvd_*_idx` syntax elements from the CABAC bitstream into the helpers;
round 218 is the table / arithmetic side of that chain.

### Tests

16 new unit tests (468 total; was 452):

* `round218_mmvd_distance_table9_full_range` — Table 9 across the full
  `mmvd_distance_idx ∈ 0..=7`.
* `round218_mmvd_distance_rejects_oob_idx` — `mmvd_distance_idx > 7`
  surfaces `Error::Unsupported`.
* `round218_mmvd_sign_table10_full_range` — Table 10 across the full
  `mmvd_direction_idx ∈ 0..=3`.
* `round218_mmvd_sign_rejects_oob_idx` — `mmvd_direction_idx > 3`
  surfaces `Error::Unsupported`.
* `round218_mmvd_offset_eq133_eq134_spot_checks` — `(5, 1) ⇒ (−32, 0)`,
  `(7, 2) ⇒ (0, 128)`, `(0, 0) ⇒ (1, 0)`, `(7, 3) ⇒ (0, −128)`.
* `round218_mmvd_offset_always_axis_aligned` — Cartesian-product 8 × 4
  property: at least one component is 0 for every legal input.
* `round218_mmvd_offset_magnitude_equals_distance` — Cartesian-product
  property: the non-zero axis's magnitude equals `MmvdDistance` at every
  `(d, dir)`.
* `round218_mmvd_offset_propagates_oob_distance_idx` — defence in depth:
  `mmvd_offset` propagates an out-of-range `mmvd_distance_idx`.
* `round218_mmvd_offset_propagates_oob_direction_idx` — same on the
  direction axis.
* `round218_mmvd_flag_ctx_inc_positional` — bin 0 → ctx 0, bin 1+
  rejects.
* `round218_mmvd_group_idx_ctx_inc_positional` — TR bins 0..1 map 1-to-1,
  bin 2+ rejects.
* `round218_mmvd_merge_idx_ctx_inc_positional` — TR bins 0..2 map
  1-to-1, bin 3+ rejects.
* `round218_mmvd_distance_idx_ctx_inc_positional` — TR bins 0..6 map
  1-to-1, bin 7+ rejects.
* `round218_mmvd_direction_idx_ctx_inc_positional` — FL bins 0, 1 → ctx
  0, 1; bin 2+ rejects (the FL code is 2-bit since
  `Ceil(Log2(cMax + 1)) = 2`).
* `round218_mmvd_worked_chain_dist3_dir2` — `(mmvd_distance_idx,
  mmvd_direction_idx) = (3, 2)` ⇒ `MmvdOffset = (0, 8)`.
* `round218_mmvd_binarization_cmax_constants_match_spec` — pins the
  four cMax constants so a future refactor that drifts them surfaces.

### Documented followups

* The §8.5.2.3.9 "Derivation process for MMVD motion vector" (eqs.
  531–616) — the consumer that adds `MmvdOffset` to a selected merge
  candidate's `mvL0` / `mvL1` while POC-scaling across L0 / L1 (via
  `DiffPicOrderCnt`) — needs the merge-candidate list builder + ref-list
  threading that has not landed yet. Round 218 ships the
  syntax-element → offset half of the chain.
* Main-profile decode path needs to thread `sps_mmvd_flag` from `Sps`
  into the CABAC `mmvd_*_idx` parse + the helpers above. Round 218
  ships the arithmetic / binarization side; the wiring lands when the
  BTT/CU-partition path is in place to host it.
* The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

## Round-213 status

Round 213 lands the **AMVR (Adaptive Motion Vector Resolution)** §8.5
helper trio in `inter.rs`. AMVR is gated by `sps_amvr_flag = 1` and
controls the per-CU resolution of both the AMVP predictor and the
motion-vector difference via the `amvr_idx[ x0 ][ y0 ]` TR-binarised
syntax element (cMax = 4 ⇒ valid range `0..=4`):

| `amvr_idx` | Resolution |
|---|---|
| 0 | 1/4-pel (Baseline; no shift) |
| 1 | 1/2-pel |
| 2 | integer-pel |
| 3 | 2-pel |
| 4 | 4-pel |

New surface:

* `inter::AMVR_IDX_MAX = 4` — TR cMax constant.
* `inter::amvr_apply_to_mvd(mvd_component, amvr_idx) -> Result<i32>` —
  eq. 145: `MvdLX[…] = MvdLX[…] << amvr_idx`. `amvr_idx > 4`
  surfaces `Error::Unsupported`.
* `inter::amvr_apply_to_mvd_vector(mvd, amvr_idx)` — vector form
  (both components).
* `inter::amvr_round_mvp(mvp_component, amvr_idx) -> Result<i32>` —
  eq. 645/646: sign-symmetric magnitude round of the AMVP predictor.
  The spec's branchless ternary:

  ```text
  mvpLX[k] = mvpLX[k] >= 0
    ? (( mvpLX[k] + (1 << (amvr_idx − 1))) >> amvr_idx) << amvr_idx
    : −((((−mvpLX[k]) + (1 << (amvr_idx − 1))) >> amvr_idx) << amvr_idx)
  ```

  i.e. round-half-away-from-zero, sign-symmetric. **Materially
  different** from `round_motion_vector` (§8.5.3.10 eq. 907-909, used
  for affine MV derivation), which is round-toward-negative-infinity:

  | input | `amvr_round_mvp(_, 2)` | `round_motion_vector(_, 2, 2)` |
  |---|---|---|
  |  +2 | +4 | +4 |
  |  −2 | **−4** | **0** |

  The `mv = −2` case is the smoking-gun distinction; round-213 tests
  pin it so a future swap surfaces immediately.
* `inter::amvr_round_mvp_vector(mvp, amvr_idx)` — vector form.
* `inter::amvr_idx_ctx_inc(bin_idx) -> Result<usize>` — §9.3.4
  positional ctxIdxInc for the `amvr_idx` TR bins. `amvr_idx` is
  **not** in Table 96 (which only lists `affine_flag`,
  `cu_skip_flag`, `pred_mode_flag`, `ibc_flag`), so the ctxInc is
  purely positional: bin `k` → ctx `k`, both `initType` halves of
  Table 67 cover the same `0..3` ladder.

### Wiring stance

Same posture as rounds 187 / 193 / 195 / 201 / 207: these are opt-in
helpers, not behaviour changes to existing decoder paths. Baseline
streams set `sps_amvr_flag = 0`, in which case `amvr_idx` is inferred
to 0 and every helper is a no-op (`amvr_apply_to_mvd` shifts by 0,
`amvr_round_mvp` returns its input unchanged). A future Main-profile
decode path threads `sps_amvr_flag` from `Sps` and the parsed
`amvr_idx` from the CABAC bitstream into the helpers; round 213 is
the math/algebra side of that chain.

### Tests

15 new unit tests (452 total; was 437):

* `round213_amvr_apply_to_mvd_zero_idx_identity` — eq. 145 with
  `amvr_idx = 0` is the identity on every component.
* `round213_amvr_apply_to_mvd_shift_examples` — eq. 145 worked at
  `amvr_idx ∈ {1, 2, 4}` for positive and negative components.
* `round213_amvr_apply_to_mvd_vector_both_axes` — vector form mirrors
  component form on both axes.
* `round213_amvr_apply_to_mvd_rejects_oob_idx` — `amvr_idx = 5`
  surfaces `Error::Unsupported`.
* `round213_amvr_round_mvp_zero_idx_identity` — eq. 645/646 with
  `amvr_idx = 0` returns input unchanged (helper is defined for the
  full `0..=4` range so callers can lift the gate without a
  special case).
* `round213_amvr_round_mvp_sign_symmetric_at_idx2` — worked
  ±{1, 2, 3} ↦ {0, ±4, ±4} at `amvr_idx = 2`.
* `round213_amvr_round_mvp_differs_from_affine_round_for_negatives` —
  pin the `mv = −2, amvr_idx = 2` distinguishing case (AMVR ↦ −4,
  affine ↦ 0). Asserts `assert_ne!` between the two.
* `round213_amvr_round_mvp_half_pel` — eq. 645/646 at `amvr_idx = 1`
  for ±{1, 3, 4}.
* `round213_amvr_round_mvp_four_pel` — eq. 645/646 at `amvr_idx = 4`
  for ±{7, 8}.
* `round213_amvr_round_mvp_vector_both_axes` — vector form at
  `amvr_idx = 3` (multiples of 8).
* `round213_amvr_round_mvp_rejects_oob_idx` — `amvr_idx = 5`
  surfaces `Error::Unsupported`.
* `round213_amvr_idx_ctx_inc_is_positional` — bins 0..=3 map 1-to-1.
* `round213_amvr_idx_ctx_inc_rejects_oob_bin` — bin 4+ surfaces
  `Error::Unsupported` (TR with cMax = 4 means at most 4 prefix bins,
  no terminator bin for the all-ones codeword "1111").
* `round213_amvr_baseline_pipeline_identity_at_idx0` — round-trip:
  Baseline pipeline (`amvr_idx = 0`) reconstructs `mv = mvp + mvd`
  unchanged.
* `round213_amvr_worked_chain_at_idx2` — Main-profile reconstruction
  at integer-pel: `mvp = (13, −10) ↦ (12, −12)`, `mvd = (2, −3) ↦
  (8, −12)`, sum `= (20, −24)`.

### Documented followup

* Main-profile decode path needs to thread `sps_amvr_flag` from
  `Sps` into the CABAC `amvr_idx` parse + the helpers above. Round 213
  ships the arithmetic; the wiring lands when the BTT/CU-partition
  path is in place to host it.
* The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

## Round-207 status

Round 207 closes the SPS → §8.9.6 chroma-scale chain at one call site
on **both** branches of `DraJoinedScaleFlag`. Round 201 landed the
table half (`chroma_qp_table_for_sps`); round 207 lands the derivation
half (`derive_dra_chroma_state_for_sps`). Together a consumer can now
take a parsed `Sps` + `DraSyntax` + `DraDerived` and obtain a
`DraChromaDerived` without re-implementing the §8.9.7 vs §8.9.8
dispatch — the adapter reads `derived.joined_scale_flag` (set by
`derive_dra_state` from `syntax.dra_table_idx`) and routes
automatically.

New surface:

* `dra::derive_dra_chroma_state_for_sps(syntax, derived, cidx, sps) ->
  Result<DraChromaDerived>` — two-way dispatch:
  - `derived.joined_scale_flag == false` (i.e. `dra_table_idx == 58`)
    ⇒ `derive_dra_chroma_state(syntax, derived, cidx,
    sps.bit_depth_y())`. ChromaQpTable is not consumed on this path
    (eq. 1394 picks `dra_cb_scale_value` / `dra_cr_scale_value`
    directly, independent of `qPi`).
  - `derived.joined_scale_flag == true` (i.e. `dra_table_idx ∈ [0, 57]`)
    ⇒ `derive_dra_chroma_state_joined(syntax, derived, cidx,
    sps.bit_depth_y(), &chroma_qp_table_for_sps(sps)?)`. The active
    table is the round-201 SPS dispatch: signalled if
    `sps.chroma_qp_table = Some(_)`, else Table 5/6 on 4:2:0, else the
    spec-page-67 "Otherwise" identity.

### Wiring stance

Same posture as rounds 193 / 195 / 201: the adapter is an opt-in
helper, not a behaviour change to existing decoder paths. Consumers
that want SPS → chroma threading call
`derive_dra_chroma_state_for_sps(&syntax, &derived, cidx, &sps)`. The
direct invocations (`derive_dra_chroma_state` /
`derive_dra_chroma_state_joined` with a hand-built table) keep working
unchanged.

### Tests

8 new unit tests (437 total; was 429):

* `round207_for_sps_unjoined_matches_direct_unjoined` — byte-identical
  state on the §8.9.7 path (10-bit, 4:2:0).
* `round207_for_sps_joined_matches_direct_joined` — byte-identical
  state on the §8.9.8 path against the SPS-active `ChromaQpTable`.
* `round207_for_sps_dispatches_on_joined_scale_flag` — same SPS,
  toggling `dra_table_idx`, exercises both branches. The joined output
  has per-range varying `chroma_scales[i]` (driven by `dra_scale_value
  = {256, 512, 1024}`); the unjoined output is constant
  `= dra_cb_scale_value`.
* `round207_for_sps_uses_signalled_chroma_qp_table_on_joined_path` —
  with `sps.chroma_qp_table = Some(signalled_identity)`, the adapter
  feeds the signalled identity to the joined derive (NOT Table 5).
  The distinguishing assertion: signalled-identity-driven
  `chroma_scales` differ materially from Table-5-driven ones.
* `round207_for_sps_propagates_zero_cb_scale_error_unjoined` —
  `dra_cb_scale_value = 0` surfaces from `derive_dra_chroma_state`.
* `round207_for_sps_propagates_zero_cb_scale_error_joined` — same on
  the joined path; Cr-direction derivation still succeeds.
* `round207_for_sps_monochrome_synthesises_identity_on_joined_path` —
  monochrome SPS (no signalled table): adapter pulls the spec-page-67
  identity via `chroma_qp_table_for_sps`; joined chain stays positive-
  definite.
* `round207_for_sps_threads_bit_depth_y_from_sps` — 8-bit vs 10-bit
  luma SPSes produce different §8.9.5 top sentinels (`1 << 8` vs
  `1 << 10`).

### Documented followup

* The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

## Round-201 status

Round 201 closes both round-195 documented follow-ups in one slot: the
**spec-page-67 "Otherwise" identity `ChromaQpTable`** for non-4:2:0
`ChromaArrayType` values (monochrome, 4:2:2, 4:4:4), plus a thin
**SPS → `ChromaQpTable` adapter** that picks the right table out of an
[`Sps`] without re-implementing the three-way dispatch at every call
site.

The spec wording (page 67, lines 3960-3961) for the
`chroma_qp_table_present_flag == 0`, `ChromaArrayType != 1` branch is:

> Otherwise, `ChromaQpTable[m][qPi]` with `m` being equal to 0 and 1,
> and `qPi` being in the range of `−QpBdOffsetC` to 57 are set equal to
> the value of `qPi`.

Round 195 noted this branch but left consumers to synthesise it on
demand because the §7.4.3.1 parser never enters the
`chroma_qp_table_present_flag` body for `chroma_format_idc == 0` —
`Sps::chroma_qp_table` stays `None`. Round 201 materialises the rule
as a first-class helper and wires the SPS-aware adapter that the
round-193 joined chain (`derive_dra_chroma_state_joined`) needs in
order to consume a parsed SPS directly.

New surface:

* `dra::default_chroma_qp_table_identity(bit_depth_chroma_minus8)` —
  identity `ChromaQpTable` per the "Otherwise" branch. Cb and Cr are
  byte-for-byte identical (the spec's "for `m` being equal to 0 and 1"
  language). Indexed by `qPi ∈ [−QpBdOffsetC, 57]` with
  `QpBdOffsetC = 6 * bit_depth_chroma_minus8`; stored value at each
  `qPi` is `qPi` itself. Out-of-range lookups still receive the
  eq. 1403 / 1404 `Clip3(−QpBdOffsetC, 57, qPi)` clamping.
* `dra::chroma_qp_table_for_sps(sps) -> Result<ChromaQpTable>` — the
  three-way dispatch:
  - `Sps::chroma_qp_table = Some(t)` → returns `t.clone()` verbatim
    (the §7.3.2.1 / §7.4.3.1 signalled body).
  - `None` AND `chroma_format_idc == 1` →
    [`default_chroma_qp_table`] for the SPS's `sps_iqt_flag` /
    `bit_depth_chroma_minus8` (Table 5 / Table 6).
  - `None` AND `chroma_format_idc != 1` →
    [`default_chroma_qp_table_identity`] for the SPS's
    `bit_depth_chroma_minus8`.

### Wiring stance

Same posture as round 193 / 195: the adapter is an opt-in helper, not
a behaviour change to existing decoder paths. Consumers that want the
SPS → table threading call `chroma_qp_table_for_sps(&sps)` and pass
the result into `derive_dra_chroma_state_joined`. The legacy paths
that hand-build a table (or call `default_chroma_qp_table` directly)
keep working unchanged.

### Tests

12 new unit tests (429 total; was 417):

* `round201_identity_chroma_qp_table_8bit_returns_qpi` —
  `bit_depth_chroma_minus8 = 0`; `lookup(qpi) == qpi` for every
  `qpi ∈ [0, 57]`, both Cb and Cr; Cb and Cr byte-for-byte equal.
* `round201_identity_chroma_qp_table_10bit_negative_qpi_in_range` —
  `bit_depth_chroma_minus8 = 2 ⇒ QpBdOffsetC = 12`; identity holds
  across the negative `qPi` range.
* `round201_identity_chroma_qp_table_lookup_clamps_out_of_range` —
  eq. 1403 / 1404 clamping: `qPi = ±1000` clamp to the table bounds.
* `round201_identity_chroma_qp_table_rejects_out_of_range_bit_depth`
  — `bit_depth_chroma_minus8 > 8` errors.
* `round201_identity_differs_from_table5_default` — pins the
  observable difference between the identity helper and the Table 5
  default (`qPi = 30 ⇒` 30 vs 29; `qPi = 57 ⇒` 57 vs 41). Prevents a
  refactor from collapsing the two helpers.
* `round201_adapter_some_chroma_qp_table_returns_it_verbatim` — when
  `Sps::chroma_qp_table = Some(t)`, the adapter returns `t.clone()`
  with every field (Cb table, Cr table, `qp_bd_offset_c`) preserved.
* `round201_adapter_420_no_signalled_falls_to_table5` —
  `chroma_format_idc = 1`, `sps_iqt_flag = false`, no signalled table
  ⇒ Table 5 (`qPi = 30 ⇒ 29`).
* `round201_adapter_420_no_signalled_iqt_falls_to_table6` —
  `chroma_format_idc = 1`, `sps_iqt_flag = true` ⇒ Table 6
  (`qPi = 30 ⇒ 29`, `qPi = 31 ⇒ 30`).
* `round201_adapter_monochrome_no_signalled_falls_to_identity` —
  `chroma_format_idc = 0` ⇒ identity (`lookup(qpi) == qpi`).
* `round201_adapter_422_no_signalled_falls_to_identity` —
  `chroma_format_idc = 2` ⇒ identity, materially different from
  Table 5.
* `round201_adapter_444_no_signalled_falls_to_identity` —
  `chroma_format_idc = 3` ⇒ identity; `sps_iqt_flag` is irrelevant in
  the "Otherwise" branch.
* `round201_adapter_chains_into_derive_dra_chroma_state_joined` —
  end-to-end: the adapter's output on a 10-bit monochrome SPS feeds
  the round-193 joined §8.9.7 derivation; every per-range
  `chromaScales[i]` is strictly positive (eq. 1386 reciprocates these
  — zero would collapse the chain).

### Documented followup

* The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

## Round-195 status

Round 195 lands the **§7.4.3.1 SPS-signalled `ChromaQpTable` parse
(eq. 74 + the surrounding pivot-fill loops)** flagged by the round 193
documented followup. With `chroma_qp_table_present_flag = 1`, the SPS
parser now derives a per-stream `ChromaQpTable` from the signalled
`(delta_qp_in_val_minus1[i][j], delta_qp_out_val[i][j])` pivot points
and stores it on `Sps::chroma_qp_table`. Round 193 fell back to
`default_chroma_qp_table` (Tables 5 / 6) because the parser discarded
those values.

New surface:

* `dra::SignalledChromaQpTableParams { same_qp_table_for_chroma,
  global_offset_flag, tables }` — the SPS chroma-QP-table body, with
  one or two `SignalledChromaQpTablePivots` rows depending on
  `same_qp_table_for_chroma`.
* `dra::SignalledChromaQpTablePivots { delta_qp_in_val_minus1[],
  delta_qp_out_val[] }` — one chroma component's pivot points,
  carried verbatim from the SPS body.
* `dra::build_signalled_chroma_qp_table(params, bit_depth_chroma_minus8)`
  — eq. 74 + the down-fill / per-segment linear interpolation /
  up-fill loops (spec page 67–68, lines 4003-4026) transcribed
  verbatim. `same_qp_table_for_chroma == 1` aliases Cr := Cb
  byte-for-byte. Rejects empty / mismatched / out-of-range pivot
  inputs.
* `Sps::chroma_qp_table: Option<ChromaQpTable>` — `Some` when
  `chroma_qp_table_present_flag = 1` and `chroma_format_idc != 0`,
  `None` otherwise. Downstream consumers (the round-193 joined entry,
  the §8.8.4 deblocker) can now pick the per-SPS table instead of the
  Table 5 / 6 default.

The SPS parser also tightens the `num_points_in_qp_table_minus1[i]`
bound from a round-1 placeholder (`> 64`) to the spec's page-67 bound
`57 + QpBdOffsetC − (global_offset_flag == 1 ? 16 : 0)`.

### Spec-text note (eq. 74)

The literal eq. 74 line in the 2020-published PDF lacks the trailing
`)` after `delta_qp_out_val[i][j]`. The bracketing we follow closes
the `(` opened just before `delta_qp_in_val_minus1[i][j] + 1`, so the
recurrence is
`qpOutVal[i][j] = qpOutVal[i][j-1] + (delta_qp_in_val_minus1[i][j] + 1 - delta_qp_out_val[i][j])`.
`qpOutVal[]` is **not** used to fill `ChromaQpTable[]` past the first
anchor (line 4011); the per-segment loop (lines 4015-4019) reads
`delta_qp_out_val[]` directly. So the only observable consequence is
the value of `ChromaQpTable[qpInVal[0]] = qpOutVal[0]` (which is
bracketed unambiguously, line 4005). A docs collaborator should
confirm eq. 74's bracketing before any conformance fixture exercises a
stream whose validity check hinges on `qpOutVal[]`.

### Tests

10 new unit tests (417 total; was 407):

* `round195_signalled_chroma_qp_table_same_qp_table_for_chroma_one_pivot`
  — minimal one-pivot identity table, both Cb and Cr aliased.
* `round195_signalled_chroma_qp_table_two_pivots_interpolates_linearly`
  — two pivots; per-segment linear interpolation; distinct Cr proves
  `same_qp_table_for_chroma == 0` honoured.
* `round195_signalled_chroma_qp_table_down_fill_below_first_pivot_clamps`
  — `bit_depth_chroma_minus8 = 1` ⇒ `QpBdOffsetC = 6`; down-fill from
  the first pivot clamps at the negative lower bound.
* `round195_signalled_chroma_qp_table_global_offset_flag_uses_startqp_16`
  — `global_offset_flag == 1` ⇒ `startQP = 16`.
* `round195_signalled_chroma_qp_table_rejects_mismatched_table_count` —
  `same_qp_table_for_chroma == true` with 2 pivot sets errors.
* `round195_signalled_chroma_qp_table_rejects_qpinval_out_of_range` —
  `qpInVal[0] > 57` errors.
* `round195_signalled_chroma_qp_table_rejects_empty_pivots` — empty
  pivot vector errors.
* `round195_signalled_chroma_qp_table_rejects_mismatched_pivot_lengths`
  — `delta_qp_in_val_minus1.len() != delta_qp_out_val.len()` errors.
* `round195_parses_signalled_chroma_qp_table_two_pivots` — full SPS
  round-trip: build an SPS RBSP with `chroma_qp_table_present_flag = 1`
  + two pivots, parse, verify `Sps::chroma_qp_table` matches the
  expected entries.
* `round195_chroma_qp_table_none_when_not_present` — default minimal
  SPS leaves `chroma_qp_table = None`.

### Documented followups

* `ChromaArrayType == 0` (monochrome) — spec page 67 "Otherwise" says
  `ChromaQpTable[m][qPi] = qPi`. The §7.4.3.1 parser does not enter
  the `chroma_qp_table_present_flag` body when `chroma_format_idc = 0`,
  so the field stays `None`; consumers must synthesise the identity
  table on demand. A `default_chroma_qp_table_monochrome()` helper +
  consumer rewiring is the round 196 slot.
* The joined chroma-scale derivation (`derive_dra_chroma_state_joined`)
  still consumes a caller-built `ChromaQpTable`; it does not yet read
  `Sps::chroma_qp_table` automatically. A thin adapter that picks
  `sps.chroma_qp_table.as_ref().cloned().or_else(|| default_…)` and
  feeds the joined entry would close the SPS → joined path
  end-to-end; round 196 slot.
* The §8.9.8 `tableNum == 0` `draChromaQpShift` ambiguity from round
  193 (docs collaborator task #1278) is still outstanding.

## Round-193 status

Round 193 lands the **§8.9.8 `DraJoinedScaleFlag = 1` (joined chroma
scale via `ChromaQpTable`) path**, plus the spec-page-67 default
`ChromaQpTable` builder for the `chroma_qp_table_present_flag == 0`
case. This closes the round-187 documented followup ("eq. 1395-1419
chain needs `ChromaQpTable` threading"). Round-187 still owned the
unjoined `dra_table_idx == 58` shortcut; round 193 picks up the
table-driven branch.

New surface:

* `dra::SCALE_QP[55]` — eq. 1420 transcribed verbatim. 55 entries (one
  extra trailing entry so eq. 1399's `ScaleQP[IndexScaleQP + 1]` is
  in-bounds at the top).
* `dra::QP_SCALE[25]` — eq. 1421 transcribed verbatim.
* `dra::ChromaQpTableEntry { qp_bd_offset_c, table }` — flat-packed
  `ChromaQpTable[qPi]` for `qPi ∈ [−QpBdOffsetC, 57]`, with
  `QpBdOffsetC = 6 * bit_depth_chroma_minus8`. `lookup(qpi)` applies
  the spec's `Clip3(−QpBdOffsetC, 57, qPi)` clamping from eq. 1403 /
  1404.
* `dra::ChromaQpTable { cb, cr }` — both chroma components' tables in
  one struct. `lookup(cidx, qpi)` dispatches by `ChromaIdx`.
* `dra::default_chroma_qp_table(sps_iqt_flag, bit_depth_chroma_minus8)`
  — Table 5 (`sps_iqt_flag = 0`) / Table 6 (`sps_iqt_flag = 1`)
  builder for the `chroma_qp_table_present_flag == 0` path on
  `ChromaArrayType == 1` (4:2:0). Both `cb` and `cr` are identical
  byte-for-byte (spec page 67: "for m being equal to 0 and 1").
* `dra::chroma_scale_joined(luma_scale, dra_cb_scale_value,
  dra_cr_scale_value, cidx, dra_table_idx, chroma_qp_table) -> i64`
  — eq. 1395 → 1419 verbatim. Pure function: no state besides the
  caller-supplied `ChromaQpTable` + the `SCALE_QP` / `QP_SCALE`
  constants. Used per-luma-range to populate `chromaScales[i]` in
  `derive_dra_chroma_state_joined`.
* `dra::derive_dra_chroma_state_joined(syntax, derived, cidx,
  bit_depth_y, chroma_qp_table)` — full §8.9.7 derivation over the
  joined `DraJoinedScaleFlag = 1` path. `chromaScales[i]` differs
  per luma range (function of `dra_scale_value[i]` via eq. 1395);
  eq. 1386 reciprocates into `inv_chroma_scales[i]`; eq. 1387 /
  1389-1393 / top-sentinel match the round-187 unjoined chain
  byte-for-byte (same downstream `out_ranges_c` / `out_scales_c` /
  `out_offsets_c` shape, so `chroma_scale_for_luma_sample` accepts
  joined state without modification).

The §8.9.8 spec text has one ambiguity at the "If tableNum is equal
to 0" branch on page 308. The "otherwise" branch derives
`qpDraIntAdj` from `qpDraFrac` (eq. 1407) and feeds it into
eq. 1409's `draChromaQpShift`; the `tableNum == 0` branch sets
`qpDraFrac = 0` + `qpDraInt -= 1` but leaves `draChromaQpShift`
undefined in the literal text. Round 193 follows the parallel-
structure reading: `qpDraIntAdj = 0` (since `qpDraFrac == 0` makes
the eq. 1407 product trivially 0), `qpDraFracAdj = 0`, and
eq. 1409 still applies (`draChromaQpShift =
ChromaQpTable[cIdx][dra_table_idx] − qp0 − qp_dra_int`). The
in-tree comment quotes the spec text verbatim; a docs collaborator
should verify the reading before any conformance fixture exercises
the `tableNum == 0` corner.

### Wiring stance

Same as round 187: the new joined entry point is independent of
`apply_post_filters` / the legacy `dra::apply_dra` chain. The
post-filter pipeline stays on the round-148 path so existing
fixtures don't shift bit positions in this round; the joined entry
is opt-in behind `derive_dra_chroma_state_joined` +
`chroma_scale_for_luma_sample`. A subsequent round can decide
whether to retire the legacy chain once a `sps_dra_flag = 1`
fixture with `chroma_qp_table_present_flag = 1` arrives.

### Tests

18 new unit tests (407 total; was 389):

* `round193_scale_qp_table_eq1420_first_and_last_entries` — eq. 1420
  spot checks + monotonicity.
* `round193_qp_scale_table_eq1421_first_and_last_entries` — eq. 1421
  spot checks + strict monotonicity + length = 25.
* `round193_default_chroma_qp_table_iqt_off_table5_spot_checks` —
  Table 5 (`qPi < 30 ⇒ QpC = qPi`, `qPi=43 ⇒ 36`, `qPi=57 ⇒ 41`);
  Cb and Cr identical.
* `round193_default_chroma_qp_table_iqt_on_table6_spot_checks` —
  Table 6 tabulated entries + the `qPi > 43 ⇒ QpC = qPi − 3` tail.
* `round193_default_chroma_qp_table_bit_depth_10_negative_qpi` —
  `bit_depth_chroma_minus8 = 2 ⇒ QpBdOffsetC = 12`; negative `qPi`
  identity (Table 5, `qPi < 30`); out-of-range `Clip3` clamping.
* `round193_default_chroma_qp_table_rejects_out_of_range_bit_depth`
  — `bit_depth_chroma_minus8 > 8` errors.
* `round193_chroma_scale_joined_returns_positive_for_table5` —
  joined chromaScale > 0 for both Cb and Cr.
* `round193_chroma_scale_joined_scales_with_luma_scale` —
  monotonicity in `lumaScale` (doubling input ⇒ output grows).
* `round193_chroma_scale_joined_pure_function_property` — same
  inputs ⇒ same output (no hidden state).
* `round193_derive_dra_chroma_state_joined_basic` — end-to-end
  three-range joined derivation populates positive per-range
  `chroma_scales[]` / `inv_chroma_scales[]`.
* `round193_derive_joined_per_range_chromascales_differ` — the
  signature property of the joined path: with distinct luma scales
  per range, `chroma_scales[]` is NOT constant (distinguishes from
  unjoined collapse).
* `round193_derive_joined_rejects_unjoined_state` — calling the
  joined entry on a `joined_scale_flag == 0` state errors.
* `round193_derive_joined_rejects_zero_cb_scale` — defensive
  zero-divisor guard (§7.4.7 forbids zero scales).
* `round193_derive_joined_rejects_zero_luma_scale` — same guard
  on `dra_scale_value[i] == 0`; verified against a hand-built
  `DraDerived` (since `derive_dra_state` itself divides by the luma
  scales and would panic upstream).
* `round193_derive_joined_noop_on_empty_state` — `num_ranges == 0`
  returns an empty `DraChromaDerived`.
* `round193_chroma_scale_for_sample_works_with_joined_state` —
  `chroma_scale_for_luma_sample` works transparently on joined
  state; samples at distinct `OutRangesC` boundaries return distinct
  scales (joined-path signature, unlike unjoined collapse).
* `round193_chroma_qp_table_lookup_clamps_to_table_range` — direct
  `Clip3(−QpBdOffsetC, 57, qPi)` clamping property.
* `round193_unjoined_derive_error_hint_mentions_joined_entry` —
  round-187 entry's `Err(Unsupported)` now points at
  `derive_dra_chroma_state_joined` instead of saying "round 187
  doesn't thread through".

### Documented followups

* **SPS-signalled `ChromaQpTable` (`chroma_qp_table_present_flag = 1`)
  parsing.** `crates/oxideav-evc/src/sps.rs:340-358` reads
  `delta_qp_in_val_minus1[i][j]` + `delta_qp_out_val[i][j]` but
  discards them. The eq. 74 / spec-page-67 derivation (with
  `qpInVal[]` / `qpOutVal[]` interpolation across pivot points)
  needs to populate a `ChromaQpTable` on the parsed [`Sps`] so the
  joined-path consumer can pick the per-SPS table instead of the
  Table 5/6 default. Pure transcription work; round 194 slot.
* `ChromaArrayType == 0` (monochrome) path — spec page 67 line "Otherwise"
  reads `ChromaQpTable[m][qPi] = qPi`; trivial identity, but the
  `default_chroma_qp_table` helper currently doesn't take a
  `chroma_format_idc` argument and would degenerate to a Table-5
  result for a monochrome SPS. Round 194 follow-up alongside the SPS
  parser change.
* The `tableNum == 0` `draChromaQpShift` ambiguity flagged above —
  needs a docs collaborator's confirmation before the joined path
  ships against a real `chroma_qp_table_present_flag = 1` fixture.

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
