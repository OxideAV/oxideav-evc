# oxideav-evc

Pure-Rust **EVC** — MPEG-5 Essential Video Coding (ISO/IEC 23094-1)
video decoder. Baseline + Main profiles. Zero C dependencies, zero FFI,
zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round-4 status

Working **Baseline-profile** decoder for IDR + P + B slices with the
following constraints (round-5 will lift them):

- 8-bit luma + chroma (4:2:0).
- `slice_deblocking_filter_flag = 0`.
- All CUs have `cbf_luma == cbf_cb == cbf_cr == 0` (no residual coding).
- Single reference picture per list (`num_ref_idx_active_minus1_l? = 0`).
- Sub-pel MV phases restricted to the Baseline 1/4-pel grid for luma
  (Table 25 phases 4, 8, 12) and 1/8-pel grid for chroma (Table 27
  phases 4, 8, 12, 16, 20, 24, 28).
- Transform sizes ∈ {2, 4, 8, 16, 32} (the 64-point IDCT is parked
  pending a clean read of the spec's `m`/`n` indexing).

Anything outside the Baseline toolset (BTT, SUCO, ADMVP, EIPD, IBC, ATS,
ADCC, ALF, DRA, AMVR, MMVD, affine, DMVR, HMVP, …) bubbles up as
`Error::Unsupported`.

109 unit tests cover the CABAC engine, NAL / parameter-set parsing, the
`slice_data()` walker, the round-3 IDR pixel pipeline, and the round-4
inter-prediction primitives (8-tap luma + 4-tap chroma sub-pel
interpolation, AMVP candidate construction, default-weighted bipred).
