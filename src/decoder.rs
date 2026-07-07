//! Registry glue + minimal decode pipeline for the EVC crate.
//!
//! Round-9 status: a working decoder for **Baseline-profile** bitstreams
//! that satisfy:
//!
//! * 8-bit luma + chroma,
//! * `slice_deblocking_filter_flag = 0`,
//! * every CU has `cbf_luma == cbf_cb == cbf_cr == 0` (pure intra/inter
//!   prediction with no residual; round-5 wires real residual decoding),
//! * Inter MVs land on the Baseline 1/4-pel grid (sub-pel phases 4, 8, 12
//!   for luma; 4, 8, 12, 16, 20, 24, 28 for chroma).
//!
//! Round-9 lifts the round-4 single-reference constraint: an in-memory
//! DPB keeps every previously-decoded short-term reference picture
//! indexed by POC, and at each non-IDR slice the decoder builds the
//! L0 / L1 lists by walking the slice's `ref_pic_list_struct()` deltas
//! (relative to `slice_pic_order_cnt_lsb`). Each inter CU's
//! `RefIdxL0` / `RefIdxL1` then resolves to the right DPB entry.
//! Output ordering follows POC, not coding order.
//!
//! Anything else (non-Baseline, 10-bit, deblocked, residuals present,
//! sub-pel outside Baseline grid) bubbles up as `Error::Unsupported`.
//! The decoder consumes length-prefixed NAL units (Annex B raw
//! bitstream framing) per ISO/IEC 23094-1.

use std::collections::VecDeque;

use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::alf::{self, AlfData};
use crate::dra::{self, DraData, DraDerived, DraSyntax};
use crate::inter::RefPictureView;
use crate::nal::{iter_length_prefixed, NalUnitType};
use crate::picture::YuvPicture;
use crate::pps::{self, Pps};
use crate::slice_data::{InterDecodeInputs, SliceDecodeInputs, SliceWalkInputs};
use crate::sps::{self, Sps};
use crate::CODEC_ID_STR;

/// Decoded-picture buffer entry (round-9 / round-10). Each slot keeps
/// the reconstructed picture, its POC and the original presentation
/// timestamp so frames can be re-ordered by POC before emission.
#[derive(Clone, Debug)]
struct DpbEntry {
    pic: YuvPicture,
    poc: i32,
    pts: Option<i64>,
    /// `true` while the picture is still needed as a reference. Round-9
    /// keeps every IDR + short-term ref alive until a fresh IDR flushes
    /// the buffer. Long-term references and explicit RPL-driven
    /// removal are deferred — the field is parked here for round-11's
    /// sliding-window unmark step.
    #[allow(dead_code)]
    used_for_reference: bool,
    /// Round-10: whether this DPB entry has already been pushed to the
    /// `out` output queue. Stops `flush()` from re-emitting frames the
    /// caller has already received.
    output_emitted: bool,
    /// Round 384: the picture's per-4×4 motion field, retained so a later
    /// slice can select this picture as its §8.3.4 collocated picture
    /// (`ColPic`) and read the §8.5.2.3.4 `mvLXCol` arrays from it.
    /// `None` for IDR pictures (all-intra ⇒ no usable collocated motion).
    side_info: Option<crate::deblock::SideInfoGrid>,
    /// Round 384: the POCs of this picture's own reference lists at its
    /// decode time — the eq.-502 `refPicOfColPic[ X ]` resolution tables.
    ref_pocs_l0: Vec<i32>,
    ref_pocs_l1: Vec<i32>,
}

/// Result of decoding one non-IDR (P/B) slice. Round 113 threads the
/// §7.3.8.2 per-CTU ALF applicability map (plus the slice-level chroma ALF
/// enables) out of the decode so the §8.9 post-filter pass can mask the
/// luma ALF apply per coding tree block.
struct NonIdrDecodeResult {
    pic: YuvPicture,
    poc: i32,
    alf_ctb_map: alf::AlfCtbMap,
    chroma_cb_enabled: bool,
    chroma_cr_enabled: bool,
    /// Round 126: the `slice_alf_luma_aps_id` / `slice_alf_chroma_aps_id`
    /// / `slice_alf_chroma2_aps_id` carried by the slice header so the
    /// §8.9 apply can route to the right APS cache slot. `None` when the
    /// slice did not signal an APS id (e.g. ALF disabled at the slice).
    alf_luma_aps_id: Option<u8>,
    alf_chroma_aps_id: Option<u8>,
    alf_chroma2_aps_id: Option<u8>,
    /// Round 384: the slice's per-4×4 motion field + its reference-list
    /// POCs, retained in the DPB so this picture can serve as a later
    /// slice's §8.3.4 collocated picture.
    side_info: crate::deblock::SideInfoGrid,
    ref_pocs_l0: Vec<i32>,
    ref_pocs_l1: Vec<i32>,
}

/// Build the round-3 decoder for the registry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(EvcDecoder::new()))
}

/// Public-but-internal type that callers normally see via the `Decoder`
/// trait. Wraps the per-stream parameter-set cache, the round-9 DPB and
/// the POC-ordered output queue.
pub struct EvcDecoder {
    codec_id: CodecId,
    sps: Option<Sps>,
    pps: Option<Pps>,
    pending_pts: Option<i64>,
    /// Frames waiting to be emitted, sorted by ascending POC.
    out: VecDeque<VideoFrame>,
    /// Parallel POC tracker for `out` so insertion stays sorted even
    /// when the bitstream coding order differs from display order.
    out_pocs: VecDeque<i32>,
    /// Round-9 DPB: every short-term reference picture (IDR + non-IDR)
    /// indexed by POC. Capped at [`MAX_DPB_ENTRIES`]; eviction is by
    /// lowest POC when a fresh picture would overflow.
    dpb: Vec<DpbEntry>,
    /// `PicOrderCntMsb` for the currently-decoded sequence (§8.3.1).
    /// Reset to 0 at every IDR; advanced when a non-IDR's
    /// `slice_pic_order_cnt_lsb` wraps around relative to the previous
    /// picture's POC LSB.
    poc_msb: i32,
    /// `prevPicOrderCntLsb` from §8.3.1: the POC LSB of the most
    /// recently decoded reference picture in coding order.
    prev_poc_lsb: i32,
    /// Round-126: ALF APS cache keyed by `adaptation_parameter_set_id`
    /// (5-bit, 0..=31). A slot is `Some(data)` only when an APS NAL with
    /// `aps_params_type == 0` and that id has been parsed; replacing an
    /// existing id overwrites the slot, matching the spec's update-by-id
    /// semantics. The §8.9 apply consults the slot named by the slice's
    /// `slice_alf_luma_aps_id` / `slice_alf_chroma_aps_id` /
    /// `slice_alf_chroma2_aps_id`. Round-11 / round-120 single-APS
    /// streams continue to work because every APS payload now lands at
    /// its declared id and the slice's `Some(id)` resolves to the same
    /// slot (with a back-compat fallback to the most-recently-cached APS
    /// when the slice doesn't surface an id — i.e. the minimal-header
    /// IDR test fixtures that never decode a slice header).
    alf_aps: [Option<AlfData>; ALF_APS_SLOTS],
    /// Most-recently-stored ALF APS id, for the back-compat fallback
    /// path described on [`Self::alf_aps`].
    last_alf_aps_id: Option<u8>,
    /// Round-126: DRA APS cache, indexed identically to [`Self::alf_aps`].
    /// The PPS `pic_dra_aps_id` selects which slot the §8.10 apply uses.
    dra_aps: [Option<DraData>; ALF_APS_SLOTS],
    /// Most-recently-stored DRA APS id (back-compat fallback for
    /// fixtures that don't drive a PPS through the cache).
    last_dra_aps_id: Option<u8>,
    /// Round-151: §7.3.6-faithful DRA APS cache, populated in parallel
    /// with [`Self::dra_aps`] from the new `parse_dra_syntax` (the
    /// legacy `parse_dra_data` is preserved for round-148 `apply_dra`
    /// chroma-offset compatibility). A follow-up round will route
    /// §8.9.3 luma mapping + §8.9.6 chroma scale derivation through
    /// this cache and retire the legacy one.
    dra_syntax_aps: [Option<(DraSyntax, DraDerived)>; ALF_APS_SLOTS],
}

const MAX_DPB_ENTRIES: usize = 16;

/// Number of `adaptation_parameter_set_id` slots a §7.4.2.3 APS NAL can
/// address. The id is `u(5)`, so the cache has 32 slots indexed 0..=31.
const ALF_APS_SLOTS: usize = 32;

/// Inputs to [`EvcDecoder::apply_post_filters`]. Bundled into a struct so
/// the §8.9 (ALF) + §8.10 (DRA) call site stays inside the
/// `clippy::too_many_arguments` lint threshold while still threading
/// every per-slice / per-PPS APS id the §7.3.4 slice header + PPS
/// describe.
struct PostFilterInputs<'a> {
    alf_ctb_map: &'a alf::AlfCtbMap,
    chroma_cb_enabled: bool,
    chroma_cr_enabled: bool,
    /// Slice-referenced ALF APS ids (round 126). `None` falls back to
    /// the most-recently-stored APS for back-compat with the minimal-
    /// header IDR fixture path.
    alf_luma_aps_id: Option<u8>,
    alf_chroma_aps_id: Option<u8>,
    alf_chroma2_aps_id: Option<u8>,
    /// PPS-resident DRA APS id (round 126). `None` when
    /// `pic_dra_enabled_flag == 0`.
    dra_aps_id: Option<u8>,
}

impl EvcDecoder {
    pub fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            sps: None,
            pps: None,
            pending_pts: None,
            out: VecDeque::new(),
            out_pocs: VecDeque::new(),
            dpb: Vec::new(),
            poc_msb: 0,
            prev_poc_lsb: 0,
            alf_aps: std::array::from_fn(|_| None),
            last_alf_aps_id: None,
            dra_aps: std::array::from_fn(|_| None),
            last_dra_aps_id: None,
            dra_syntax_aps: std::array::from_fn(|_| None),
        }
    }

    /// Resolve the ALF APS cache slot a slice's `aps_id` references. When
    /// the slice didn't surface an explicit id (the minimal-header IDR
    /// fixture path), fall back to whichever id was stored last — matches
    /// pre-round-126 behaviour where there was only one slot.
    fn alf_aps_for_slice(&self, slice_aps_id: Option<u8>) -> Option<&AlfData> {
        if let Some(id) = slice_aps_id {
            return self.alf_aps.get(id as usize).and_then(|s| s.as_ref());
        }
        self.last_alf_aps_id
            .and_then(|id| self.alf_aps.get(id as usize).and_then(|s| s.as_ref()))
    }

    /// Resolve the DRA APS cache slot for the active PPS's
    /// `pic_dra_aps_id`. Same fallback rule as [`Self::alf_aps_for_slice`].
    fn dra_aps_for_pps(&self, pps_dra_aps_id: Option<u8>) -> Option<&DraData> {
        if let Some(id) = pps_dra_aps_id {
            return self.dra_aps.get(id as usize).and_then(|s| s.as_ref());
        }
        self.last_dra_aps_id
            .and_then(|id| self.dra_aps.get(id as usize).and_then(|s| s.as_ref()))
    }

    /// Resolve the round-151 spec-faithful DRA APS cache slot. Returns the
    /// `(DraSyntax, DraDerived)` pair populated by `parse_dra_syntax` +
    /// `derive_dra_state` whenever the §7.3.6 `dra_data()` syntax was
    /// parsed (i.e. modern path); `None` when the slot is empty or the
    /// PPS surfaced no `pic_dra_aps_id`.
    fn dra_syntax_aps_for_pps(
        &self,
        pps_dra_aps_id: Option<u8>,
    ) -> Option<&(DraSyntax, DraDerived)> {
        let id = pps_dra_aps_id?;
        self.dra_syntax_aps
            .get(id as usize)
            .and_then(|s| s.as_ref())
    }

    /// Round-181 — apply §8.9.3 (Inverse mapping process for a luma
    /// sample) to the picture's luma plane using the round-151 spec-faithful
    /// DRA state, returning `true` if the mapping was applied and `false`
    /// if the cache slot for `pps_dra_aps_id` was empty (no-op).
    ///
    /// The pipeline is:
    ///
    /// 1. Look up `(DraSyntax, DraDerived)` for `pps_dra_aps_id` in
    ///    `dra_syntax_aps`. Skip the apply if absent.
    /// 2. Clone the derived state and run the §7.4.7 off-by-one
    ///    reconciliation via [`dra::fill_inv_luma_scales_range_zero`] so
    ///    range 0 carries a non-degenerate `InvLumaScales[0]` /
    ///    `DraOffsets[0]`. This is the interpretation flagged in the
    ///    r174 docs gap; the original cache entry is unchanged.
    /// 3. Apply [`dra::apply_luma_inverse_mapping_u8`] over the picture's
    ///    `y` plane in-place, which iterates §8.9.5 → eq. 1374-1376 per
    ///    sample.
    ///
    /// 8-bit code space — the §8.9.3 LUT is 256-entry, so >8-bit
    /// pictures clamp their index defensively (10-bit DRA is a
    /// documented follow-up).
    ///
    /// **This method is independent of the legacy round-148
    /// [`dra::apply_dra`] path used by [`Self::apply_post_filters`].**
    /// Callers who want the spec-faithful §8.9.3 must invoke this method
    /// explicitly; the post-filter wiring stays on the round-148 path so
    /// existing fixtures don't shift.
    pub fn apply_luma_inverse_mapping_spec_faithful(
        &self,
        pic: &mut YuvPicture,
        pps_dra_aps_id: Option<u8>,
    ) -> Result<bool> {
        let pair = match self.dra_syntax_aps_for_pps(pps_dra_aps_id) {
            Some(p) => p,
            None => return Ok(false),
        };
        let (syntax, derived) = pair;
        let mut local = derived.clone();
        dra::fill_inv_luma_scales_range_zero(&mut local, syntax)?;
        dra::apply_luma_inverse_mapping_u8(&mut pic.y, &local);
        Ok(true)
    }

    /// Find the DPB entry with the given POC, returning a borrow. None
    /// when no entry matches (caller may signal a malformed bitstream).
    fn dpb_find(&self, poc: i32) -> Option<&DpbEntry> {
        self.dpb.iter().find(|e| e.poc == poc)
    }

    /// Insert a freshly-decoded picture into the DPB, evicting the
    /// lowest-POC reference when at capacity.
    fn dpb_insert(&mut self, entry: DpbEntry) {
        if self.dpb.len() >= MAX_DPB_ENTRIES {
            // Evict the entry with the smallest POC (oldest in display
            // order). Round-9 doesn't track sliding-window LTRP yet so
            // any non-LTRP entry is fair game.
            let mut min_idx = 0;
            for (i, e) in self.dpb.iter().enumerate() {
                if e.poc < self.dpb[min_idx].poc {
                    min_idx = i;
                }
            }
            self.dpb.remove(min_idx);
        }
        self.dpb.push(entry);
    }

    /// Flush all reference pictures (IDR boundary). Output queue is
    /// untouched — already-emitted POCs stay where they are.
    fn dpb_flush(&mut self) {
        self.dpb.clear();
        self.poc_msb = 0;
        self.prev_poc_lsb = 0;
    }

    /// Compute a full POC from a slice's `slice_pic_order_cnt_lsb` per
    /// §8.3.1 (simplified for the Baseline single-layer case): if the
    /// LSB has wrapped relative to `prev_poc_lsb`, advance MSB by
    /// `MaxPicOrderCntLsb`. The formal pred is symmetric (also handles
    /// regress) which we mirror here.
    fn derive_poc(&self, slice_lsb: i32, max_poc_lsb: i32) -> i32 {
        let half = max_poc_lsb >> 1;
        let prev_lsb = self.prev_poc_lsb;
        let mut poc_msb = self.poc_msb;
        if slice_lsb < prev_lsb && (prev_lsb - slice_lsb) >= half {
            poc_msb = poc_msb.wrapping_add(max_poc_lsb);
        } else if slice_lsb > prev_lsb && (slice_lsb - prev_lsb) > half {
            poc_msb = poc_msb.wrapping_sub(max_poc_lsb);
        }
        poc_msb + slice_lsb
    }
}

impl Default for EvcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for EvcDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let nals = iter_length_prefixed(&packet.data)
            .map_err(|e| Error::invalid(format!("evc decoder: NAL framing: {e}")))?;
        for nal in nals {
            match nal.header.nal_unit_type {
                NalUnitType::Sps => {
                    let sps = sps::parse(nal.rbsp())
                        .map_err(|e| Error::invalid(format!("evc decoder: SPS parse: {e}")))?;
                    self.sps = Some(sps);
                }
                NalUnitType::Pps => {
                    let pps = pps::parse(nal.rbsp())
                        .map_err(|e| Error::invalid(format!("evc decoder: PPS parse: {e}")))?;
                    self.pps = Some(pps);
                }
                NalUnitType::Idr => {
                    let sps = self
                        .sps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: IDR slice before SPS"))?
                        .clone();
                    let pps = self
                        .pps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: IDR slice before PPS"))?
                        .clone();
                    let (mut pic, stats) = crate::decode_idr_slice(&sps, &pps, nal.rbsp())?;
                    // §8.3.1: IDR resets POC to 0, flushes the DPB.
                    self.dpb_flush();
                    // Round-11: ALF + DRA post-filter pass. Round 113: the
                    // §7.3.8.2 per-CTU ALF map masks the §8.9 luma apply. The
                    // minimal-header IDR path produces an all-off map, so this
                    // falls back to the whole-plane apply (unchanged behaviour).
                    // Round 126: the minimal-header path does not consume a
                    // slice header, so no `slice_alf_*_aps_id` is available
                    // here; `apply_post_filters` falls back to the most-
                    // recently-stored APS id (back-compat). The PPS-resident
                    // `pic_dra_aps_id` IS available, though, so we route DRA
                    // through it.
                    let dra_aps_id = if pps.pic_dra_enabled_flag {
                        Some(pps.pic_dra_aps_id)
                    } else {
                        None
                    };
                    self.apply_post_filters(
                        &mut pic,
                        &sps,
                        PostFilterInputs {
                            alf_ctb_map: &stats.alf_ctb_map,
                            chroma_cb_enabled: false,
                            chroma_cr_enabled: false,
                            alf_luma_aps_id: None,
                            alf_chroma_aps_id: None,
                            alf_chroma2_aps_id: None,
                            dra_aps_id,
                        },
                    );
                    let entry = DpbEntry {
                        pic,
                        poc: 0,
                        pts: packet.pts,
                        used_for_reference: true,
                        output_emitted: false,
                        side_info: None,
                        ref_pocs_l0: Vec::new(),
                        ref_pocs_l1: Vec::new(),
                    };
                    self.prev_poc_lsb = 0;
                    self.poc_msb = 0;
                    self.dpb_insert(entry);
                    self.enqueue_for_output(0);
                }
                NalUnitType::NonIdr => {
                    let sps = self
                        .sps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: NonIDR slice before SPS"))?
                        .clone();
                    let pps = self
                        .pps
                        .as_ref()
                        .ok_or_else(|| Error::invalid("evc decoder: NonIDR slice before PPS"))?
                        .clone();
                    let NonIdrDecodeResult {
                        mut pic,
                        poc,
                        alf_ctb_map,
                        chroma_cb_enabled,
                        chroma_cr_enabled,
                        alf_luma_aps_id,
                        alf_chroma_aps_id,
                        alf_chroma2_aps_id,
                        side_info,
                        ref_pocs_l0,
                        ref_pocs_l1,
                    } = self.decode_non_idr(&sps, &pps, nal.rbsp())?;
                    // Round-11: ALF + DRA post-filter pass. Round 113: the
                    // §7.3.8.2 per-CTU ALF map masks the §8.9 luma apply.
                    // Round 126: route ALF + DRA APS lookups through the
                    // slice / PPS-referenced ids instead of the most-recent
                    // cached APS.
                    let dra_aps_id = if pps.pic_dra_enabled_flag {
                        Some(pps.pic_dra_aps_id)
                    } else {
                        None
                    };
                    self.apply_post_filters(
                        &mut pic,
                        &sps,
                        PostFilterInputs {
                            alf_ctb_map: &alf_ctb_map,
                            chroma_cb_enabled,
                            chroma_cr_enabled,
                            alf_luma_aps_id,
                            alf_chroma_aps_id,
                            alf_chroma2_aps_id,
                            dra_aps_id,
                        },
                    );
                    let entry = DpbEntry {
                        pic,
                        poc,
                        pts: packet.pts,
                        used_for_reference: true,
                        output_emitted: false,
                        side_info: Some(side_info),
                        ref_pocs_l0,
                        ref_pocs_l1,
                    };
                    self.dpb_insert(entry);
                    self.enqueue_for_output(poc);
                }
                NalUnitType::Aps => {
                    // Round-11 parses the APS and caches ALF / DRA data.
                    // Round-126: cache per `adaptation_parameter_set_id`
                    // (5-bit, 0..=31) so a slice's `slice_alf_*_aps_id`
                    // / a PPS's `pic_dra_aps_id` can route to the correct
                    // slot. Errors in APS parsing remain non-fatal — the
                    // filter simply won't fire for that id.
                    let rbsp = nal.rbsp();
                    if let Ok(aps) = crate::aps::parse(rbsp) {
                        let id = aps.adaptation_parameter_set_id as usize;
                        if id < ALF_APS_SLOTS {
                            if aps.is_alf() && !aps.payload_raw.is_empty() {
                                if let Ok(alf_data) = alf::parse_alf_data(&aps.payload_raw) {
                                    self.alf_aps[id] = Some(alf_data);
                                    self.last_alf_aps_id = Some(id as u8);
                                }
                            } else if aps.is_dra() && !aps.payload_raw.is_empty() {
                                if let Ok(dra_data) = dra::parse_dra_data(&aps.payload_raw) {
                                    self.dra_aps[id] = Some(dra_data);
                                    self.last_dra_aps_id = Some(id as u8);
                                }
                                // Round 151: also run the §7.3.6-faithful
                                // parser into the parallel cache. Needs
                                // BitDepthY from the active SPS — skip
                                // gracefully when none has been parsed
                                // yet (the legacy fixture path never
                                // exercises the spec-faithful parser).
                                if let Some(sps) = self.sps.as_ref() {
                                    let bd_y = sps.bit_depth_y();
                                    if let Ok(pair) = dra::parse_dra_syntax(&aps.payload_raw, bd_y)
                                    {
                                        self.dra_syntax_aps[id] = Some(pair);
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    // SEI / FD / etc. — skipped.
                }
            }
        }
        self.pending_pts = packet.pts;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.out.pop_front() {
            Some(v) => {
                self.out_pocs.pop_front();
                Ok(Frame::Video(v))
            }
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Round-10 drain: every DPB entry that hasn't yet been pushed
        // to the output queue is emitted now, in ascending POC order.
        // Pictures already in `out` (the typical case for low-delay GOPs
        // where decode order == display order) stay in place and are
        // not duplicated. After flush(), the next `receive_frame` calls
        // will pull every reconstructed picture in display order.
        self.drain_dpb_to_output();
        Ok(())
    }
}

impl EvcDecoder {
    /// Push the freshly-decoded picture at `poc` to the output queue
    /// in POC order. Round-9 keeps a parallel `out_pocs` buffer so the
    /// queue stays sorted by POC even when bitstream coding-order
    /// differs from display-order (e.g. B-pyramid GOPs). The frame is
    /// inserted at the unique position where every earlier frame has a
    /// smaller POC and every later frame has a greater POC. Round-10
    /// marks the corresponding DPB entry as `output_emitted = true` so
    /// `flush()` doesn't re-push it.
    fn enqueue_for_output(&mut self, poc: i32) {
        let (frame, pts) = match self.dpb_find(poc) {
            Some(e) => (picture_to_video_frame(&e.pic, e.pts), e.pts),
            None => return,
        };
        let _ = pts; // silence unused
        let pos = self.out_pocs.iter().position(|&p| p > poc);
        match pos {
            Some(i) => {
                self.out.insert(i, frame);
                self.out_pocs.insert(i, poc);
            }
            None => {
                self.out.push_back(frame);
                self.out_pocs.push_back(poc);
            }
        }
        if let Some(entry) = self.dpb.iter_mut().find(|e| e.poc == poc) {
            entry.output_emitted = true;
        }
    }

    /// Round-10 `flush()` drain: emit every DPB entry that hasn't yet
    /// been queued for output, in ascending POC order. Pictures already
    /// in `out` (the typical case for low-delay GOPs) are skipped so
    /// the caller never sees the same frame twice.
    ///
    /// Used by [`Decoder::flush`] but also exposed as a helper for
    /// tests that want to verify the drain path independently.
    fn drain_dpb_to_output(&mut self) {
        // Sort indices by POC so we emit in display order.
        let mut idxs: Vec<usize> = (0..self.dpb.len())
            .filter(|&i| !self.dpb[i].output_emitted)
            .collect();
        idxs.sort_by_key(|&i| self.dpb[i].poc);
        for i in idxs {
            let poc = self.dpb[i].poc;
            self.enqueue_for_output(poc);
        }
    }

    /// Decode a NonIDR slice end-to-end. Returns the reconstructed
    /// picture and its POC. The DPB-resolved L0 / L1 reference lists
    /// are built from the slice's `ref_pic_list_struct()` deltas,
    /// applied to the slice's POC.
    fn decode_non_idr(
        &mut self,
        sps: &Sps,
        pps: &Pps,
        slice_nal_rbsp: &[u8],
    ) -> Result<NonIdrDecodeResult> {
        // Round 384: sps_admvp_flag is lifted from the unsupported gate —
        // P/B slices route each coding unit through the §7.3.8.4
        // Main-profile syntax drivers (merge/MMVD/affine/explicit-AMVP)
        // via the InterToolGates threaded below.
        // Round 391: sps_btt_flag / sps_suco_flag are lifted — the P/B
        // walker decodes the §7.3.8.3 BTT split group, the SUCO order
        // flag and the §7.4.9.3 pred_mode_constraint machinery via the
        // threaded CodingTreeGates.
        // Round 397: sps_cm_init_flag is lifted — the P/B walker
        // initialises the §9.3.2.2 Main-profile contexts (initType 1)
        // and routes every regular bin through the §9.3.4.2.1
        // ctxIdxOffset + ctxInc selection.
        // Round 397: sps_dquant_flag lifted (cuQpDeltaCode marks + the
        // eq. 1042 QpY chain in the P/B walker); sps_eipd_flag lifted
        // (the §7.3.8.4 EIPD intra group + §8.4.4 kernels on the P/B
        // intra-CU path).
        if sps.sps_addb_flag || sps.sps_ats_flag || sps.sps_adcc_flag {
            return Err(Error::unsupported(
                "evc decoder: P/B requires Baseline-profile toolset (round-9 adds DPB + POC)",
            ));
        }
        // Round 95: sps_ibc_flag is no longer part of the unsupported gate.
        // The §7.3.8.4 IBC branch is surfaced inside the per-CU inter
        // walker (`decode_inter_coding_unit`) so non-IDR (P/B) slices with
        // IBC-enabled SPS now decode the `ibc_flag` syntax element + the
        // §8.6.1 IBC pipeline symmetrically to the IDR path lifted in
        // round 90.
        // sps_alf_flag and sps_dra_flag are handled by the round-11 post-filter
        // pipeline — they no longer gate the Unsupported path.
        if !pps.single_tile_in_pic_flag {
            return Err(Error::unsupported(
                "evc decoder: P/B requires single_tile_in_pic_flag == 1",
            ));
        }
        let ctx = crate::slice_header::SliceParseContext {
            single_tile_in_pic_flag: pps.single_tile_in_pic_flag,
            arbitrary_slice_present_flag: pps.arbitrary_slice_present_flag,
            tile_id_len_minus1: pps.tile_id_len_minus1,
            num_tile_columns_minus1: pps.num_tile_columns_minus1,
            num_tile_rows_minus1: pps.num_tile_rows_minus1,
            sps_pocs_flag: sps.sps_pocs_flag,
            sps_rpl_flag: sps.sps_rpl_flag,
            sps_alf_flag: sps.sps_alf_flag,
            sps_mmvd_flag: sps.sps_mmvd_flag,
            sps_admvp_flag: sps.sps_admvp_flag,
            sps_addb_flag: sps.sps_addb_flag,
            log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
            chroma_array_type: sps.chroma_array_type(),
            num_ref_pic_lists_in_sps_l0: sps.num_ref_pic_lists_in_sps_l0,
            num_ref_pic_lists_in_sps_l1: sps.num_ref_pic_lists_in_sps_l1,
            rpl1_idx_present_flag: pps.rpl1_idx_present_flag,
            long_term_ref_pics_flag: sps.long_term_ref_pics_flag,
            additional_lt_poc_lsb_len: pps.additional_lt_poc_lsb_len,
        };
        let header = crate::slice_header::parse(slice_nal_rbsp, NalUnitType::NonIdr, &ctx)?;
        let slice_is_b = match header.slice_type {
            crate::slice_header::SliceType::B => true,
            crate::slice_header::SliceType::P => false,
            crate::slice_header::SliceType::I => {
                return Err(Error::invalid(
                    "evc decoder: NonIDR slice cannot be slice_type == I",
                ));
            }
        };
        let num_ref_idx_active_minus1_l0 = header.num_ref_idx_active_minus1[0];
        let num_ref_idx_active_minus1_l1 = header.num_ref_idx_active_minus1[1];
        let slice_deblocking_filter_flag = header.slice_deblocking_filter_flag;
        let slice_qp = header.slice_qp;
        let slice_cb_qp_offset = header.slice_cb_qp_offset;
        let slice_cr_qp_offset = header.slice_cr_qp_offset;

        // Round-9 POC derivation (§8.3.1).
        let max_poc_lsb = if sps.sps_pocs_flag {
            1i32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4)
        } else {
            1
        };
        let slice_poc_lsb = header.slice_pic_order_cnt_lsb as i32;
        let poc = if sps.sps_pocs_flag {
            self.derive_poc(slice_poc_lsb, max_poc_lsb)
        } else {
            // No POC signalling — fall back to coding-order as POC.
            self.dpb.iter().map(|e| e.poc).max().unwrap_or(0) + 1
        };

        // Round-9 reference-list construction. Per §8.3.5: for each list
        // i ∈ {0, 1}, walk the RPL entries (either inline or SPS-resident);
        // each STRP entry contributes a POC delta (positive or negative)
        // applied to the current slice's POC; LTRPs would resolve via
        // poc_lsb_lt (deferred). The resulting POC is matched against
        // the DPB; a missing match makes the slice undecodable.
        let n_active_l0 = (num_ref_idx_active_minus1_l0 + 1) as usize;
        let n_active_l1 = if slice_is_b {
            (num_ref_idx_active_minus1_l1 + 1) as usize
        } else {
            0
        };
        let pocs_l0 = if sps.sps_rpl_flag {
            let rpl_l0 = self.resolve_slice_rpl(&header, sps, 0)?;
            self.build_ref_pocs(&rpl_l0, poc, n_active_l0, max_poc_lsb)?
        } else {
            // §8.3.5 round-9 implicit fallback: with no per-slice or
            // SPS RPL signalling, use the highest-POC decoded picture
            // as the single L0 entry (low-delay coding-order GOP).
            self.implicit_ref_pocs(n_active_l0)?
        };
        let pocs_l1 = if slice_is_b {
            if sps.sps_rpl_flag {
                let rpl_l1 = self.resolve_slice_rpl(&header, sps, 1)?;
                self.build_ref_pocs(&rpl_l1, poc, n_active_l1, max_poc_lsb)?
            } else {
                self.implicit_ref_pocs(n_active_l1)?
            }
        } else {
            Vec::new()
        };

        // Re-run the parse on a counting BitReader to discover where the
        // slice header ends so slice_data can pick up at the next byte.
        let mut br = crate::bitreader::BitReader::new(slice_nal_rbsp);
        crate::slice_header::parse_consume(&mut br, NalUnitType::NonIdr, &ctx)?;
        br.align_to_byte();
        let consumed_bits = br.bit_position();
        if consumed_bits % 8 != 0 {
            return Err(Error::invalid("evc decoder: slice header not byte-aligned"));
        }
        let consumed_bytes = (consumed_bits / 8) as usize;
        if consumed_bytes >= slice_nal_rbsp.len() {
            return Err(Error::invalid(
                "evc decoder: no slice_data bytes after NonIDR header",
            ));
        }
        let slice_data_bytes = &slice_nal_rbsp[consumed_bytes..];
        let ctb_log2_size_y = sps.log2_ctu_size_minus5 + 5;
        let min_cb_log2_size_y = sps.log2_min_cb_size_minus2 + 2;
        let max_tb_log2_size_y = ctb_log2_size_y.min(5);
        let log2_max_ibc_cand_size = sps.log2_max_ibc_cand_size().unwrap_or(0);
        let walk = SliceWalkInputs {
            pic_width: sps.pic_width_in_luma_samples,
            pic_height: sps.pic_height_in_luma_samples,
            ctb_log2_size_y,
            min_cb_log2_size_y,
            max_tb_log2_size_y,
            chroma_format_idc: sps.chroma_format_idc,
            cu_qp_delta_enabled: pps.cu_qp_delta_enabled_flag,
            sps_eipd_flag: sps.sps_eipd_flag,
            sps_dquant_flag: sps.sps_dquant_flag,
            cu_qp_delta_area: pps.log2_cu_qp_delta_area_minus6 + 6,
            sps_ibc_flag: sps.sps_ibc_flag,
            log2_max_ibc_cand_size,
            // §7.3.8.2: thread the slice-header ALF map controls so the
            // CTU walker decodes the per-CTU `alf_ctb_*` applicability
            // map when the slice signals it.
            slice_alf_enabled_flag: header.slice_alf_enabled_flag,
            slice_alf_map_flag: header.slice_alf_map_flag,
            slice_chroma_alf_enabled_flag: header.slice_chroma_alf_enabled_flag,
            slice_alf_chroma_map_flag: header.slice_alf_chroma_map_flag,
            slice_chroma2_alf_enabled_flag: header.slice_chroma2_alf_enabled_flag,
            slice_alf_chroma2_map_flag: header.slice_alf_chroma2_map_flag,
            // Round 391: thread the real §7.3.8.3 BTT/SUCO gates.
            tree_gates: crate::slice_data::CodingTreeGates::from_sps(sps),
        };
        let decode = SliceDecodeInputs {
            slice_qp: slice_qp as i32,
            bit_depth_luma: sps.bit_depth_y(),
            bit_depth_chroma: sps.bit_depth_c(),
            enable_deblock: slice_deblocking_filter_flag,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            sps_ibc_flag: sps.sps_ibc_flag,
            log2_max_ibc_cand_size,
        };
        // Resolve POCs to DPB views. This consumes immutable borrows
        // of `self.dpb`, so we collect into an owned Vec<RefPictureView>
        // first.
        let ref_list_l0 = self.dpb_views_for_pocs(&pocs_l0)?;
        let ref_list_l1 = if slice_is_b {
            self.dpb_views_for_pocs(&pocs_l1)?
        } else {
            Vec::new()
        };
        // §8.3.4 — ColPic = RefPicList[ col_pic_list_idx ][ col_pic_ref_idx ]
        // (col_pic_list_idx inferred 0 for P / 1 for B when unsignalled).
        // The motion field is only retained for previously decoded P/B
        // pictures; an IDR ColPic (all-intra) yields no collocated motion
        // and the temporal merge slot stays empty.
        let col_pocs_list = if slice_is_b && header.col_pic_list_idx == 1 {
            &pocs_l1
        } else {
            &pocs_l0
        };
        let col_pic = col_pocs_list
            .get(header.col_pic_ref_idx as usize)
            .and_then(|&col_poc| {
                self.dpb_find(col_poc).and_then(|e| {
                    e.side_info
                        .as_ref()
                        .map(|grid| crate::slice_data::ColPicInputs {
                            grid,
                            col_poc,
                            ref_pocs_l0: &e.ref_pocs_l0,
                            ref_pocs_l1: &e.ref_pocs_l1,
                        })
                })
            });
        let inputs = InterDecodeInputs {
            walk,
            decode,
            slice_is_b,
            num_ref_idx_active_minus1_l0,
            num_ref_idx_active_minus1_l1,
            ref_list_l0: &ref_list_l0,
            ref_list_l1: &ref_list_l1,
            // §7.3.8.4 tool gates from the SPS + the slice header's
            // mmvd_group_enable_flag. All-false when sps_admvp_flag == 0
            // (the historical Baseline path, byte-identical).
            inter_tool_gates: crate::inter_cu_syntax::InterToolGates {
                sps_admvp_flag: sps.sps_admvp_flag,
                sps_amvr_flag: sps.sps_amvr_flag,
                sps_mmvd_flag: sps.sps_mmvd_flag,
                sps_affine_flag: sps.sps_affine_flag,
                mmvd_group_enable_flag: header.mmvd_group_enable_flag,
                sps_dmvr_flag: sps.sps_dmvr_flag,
            },
            // §8.5.2.3.3 / §8.5.2.3.9 POC context: the derived slice POC
            // plus the resolved reference-list POCs (parallel to the
            // RefPictureView lists above).
            pocs: crate::slice_data::InterPocs {
                curr_poc: poc,
                ref_pocs_l0: &pocs_l0,
                ref_pocs_l1: &pocs_l1,
            },
            col_pic,
        };
        let (pic, stats) =
            crate::slice_data::decode_baseline_inter_slice(slice_data_bytes, inputs)?;

        // Update §8.3.1 prev-LSB tracker for the next non-IDR slice.
        if sps.sps_pocs_flag {
            // Walk poc back into msb / lsb decomposition.
            let lsb = ((poc % max_poc_lsb) + max_poc_lsb) % max_poc_lsb;
            self.poc_msb = poc - lsb;
            self.prev_poc_lsb = lsb;
        }

        Ok(NonIdrDecodeResult {
            pic,
            poc,
            side_info: stats.side_info,
            ref_pocs_l0: pocs_l0,
            ref_pocs_l1: pocs_l1,
            alf_ctb_map: stats.alf_ctb_map,
            // §8.9 chroma path (ChromaArrayType 1..2): the plane is filtered
            // when the slice-level chroma ALF enable is set (the per-CTB
            // chroma map flags are inferred 0 in the Baseline 4:2:0 case).
            chroma_cb_enabled: header.slice_chroma_alf_enabled_flag,
            chroma_cr_enabled: header.slice_chroma2_alf_enabled_flag,
            // Round 126: surface the slice-referenced APS ids so
            // `apply_post_filters` can pull from the right cache slot.
            alf_luma_aps_id: header.slice_alf_luma_aps_id,
            alf_chroma_aps_id: header.slice_alf_chroma_aps_id,
            alf_chroma2_aps_id: header.slice_alf_chroma2_aps_id,
        })
    }

    /// Resolve the slice's RPL for list `list_x` (0 or 1) into a concrete
    /// `RefPicListStruct`. Either inline (`slice_rpl[list_x] = Some`) or
    /// indirected through the SPS via `ref_pic_list_idx[list_x]`.
    fn resolve_slice_rpl(
        &self,
        header: &crate::slice_header::SliceHeader,
        sps: &Sps,
        list_x: usize,
    ) -> Result<crate::rpl::RefPicListStruct> {
        if let Some(ref rpl) = header.slice_rpl[list_x] {
            return Ok(rpl.clone());
        }
        // SPS-resident RPL: `ref_pic_list_idx[list_x]` selects an entry.
        let idx = header.ref_pic_list_idx[list_x] as usize;
        let sps_rpls = if list_x == 0 {
            &sps.ref_pic_list_structs_l0
        } else {
            &sps.ref_pic_list_structs_l1
        };
        sps_rpls.get(idx).cloned().ok_or_else(|| {
            Error::invalid(format!(
                "evc decoder: ref_pic_list_idx[{list_x}] = {idx} but SPS has {} RPLs",
                sps_rpls.len()
            ))
        })
    }

    /// Implicit-RPL fallback for streams with `sps_rpl_flag == 0`:
    /// repeat the highest-POC DPB entry `n_active` times. This matches
    /// the low-delay coding-order GOP (one reference, no reordering)
    /// that round-4 fixtures use.
    fn implicit_ref_pocs(&self, n_active: usize) -> Result<Vec<i32>> {
        let max_poc = self
            .dpb
            .iter()
            .map(|e| e.poc)
            .max()
            .ok_or_else(|| Error::invalid("evc decoder: P/B slice with empty DPB"))?;
        Ok(vec![max_poc; n_active])
    }

    /// Convert a list of resolved POCs into borrowed [`RefPictureView`]s
    /// from the DPB. Bails with `Error::Invalid` if any POC isn't in
    /// the buffer.
    fn dpb_views_for_pocs(&self, pocs: &[i32]) -> Result<Vec<RefPictureView<'_>>> {
        let mut out = Vec::with_capacity(pocs.len());
        for &poc in pocs {
            let entry = self.dpb_find(poc).ok_or_else(|| {
                Error::invalid(format!(
                    "evc decoder: reference picture POC {poc} not in DPB"
                ))
            })?;
            let pic = &entry.pic;
            out.push(RefPictureView {
                y: &pic.y,
                cb: &pic.cb,
                cr: &pic.cr,
                width: pic.width,
                height: pic.height,
                y_stride: pic.y_stride(),
                c_stride: pic.c_stride(),
                chroma_format_idc: pic.chroma_format_idc,
            });
        }
        Ok(out)
    }

    /// Round-11 post-filter pass: apply ALF (§8.9) then DRA (§8.10) to
    /// a decoded picture if the corresponding APS has been cached and the
    /// SPS flags indicate they are active. This is called after deblocking
    /// for every IDR and non-IDR slice.
    ///
    /// Round 113: when the slice decoded a per-CTU `alf_ctb_*` applicability
    /// map (§7.3.8.2) that turns at least one luma CTB on, the §8.9 luma
    /// apply is masked per coding tree block via [`alf::apply_alf_with_map`].
    /// An all-off map (the minimal-header IDR path that doesn't thread the
    /// slice ALF enables) falls back to the whole-plane [`alf::apply_alf`],
    /// preserving the round-11 behaviour.
    ///
    /// Round 126: the slice's `slice_alf_luma_aps_id` /
    /// `slice_alf_chroma_aps_id` / `slice_alf_chroma2_aps_id` (when
    /// present) drive APS selection from the 32-slot cache. When the
    /// slice didn't surface an id (the minimal-header IDR fixture path),
    /// the fallback to the most-recently-stored APS preserves the
    /// pre-round-126 behaviour for existing tests. The DRA path consults
    /// the PPS's `pic_dra_aps_id` via [`Self::dra_aps_for_pps`].
    fn apply_post_filters(&self, pic: &mut YuvPicture, sps: &Sps, in_: PostFilterInputs<'_>) {
        let bd_y = sps.bit_depth_y();
        let bd_c = sps.bit_depth_c();
        let alf_map = in_.alf_ctb_map;
        let chroma_cb_enabled = in_.chroma_cb_enabled;
        let chroma_cr_enabled = in_.chroma_cr_enabled;
        if sps.sps_alf_flag {
            // §7.4.5: when the slice references separate Cb and Cr APS ids
            // (ChromaArrayType == 3 path), the chroma planes may pull from
            // DIFFERENT APS slots than the luma plane. We resolve each
            // referenced id independently and fall back to the luma APS
            // for any chroma slot the slice didn't separately signal —
            // matching the spec's inference of "use the same APS" when
            // only one chroma idc is set.
            let luma_alf = self.alf_aps_for_slice(in_.alf_luma_aps_id);
            let cb_alf = if in_.alf_chroma_aps_id.is_some() {
                self.alf_aps_for_slice(in_.alf_chroma_aps_id)
            } else {
                luma_alf
            };
            let cr_alf = if in_.alf_chroma2_aps_id.is_some() {
                self.alf_aps_for_slice(in_.alf_chroma2_aps_id)
            } else {
                // ChromaArrayType ∈ {1, 2}: a single chroma APS id covers
                // both planes when `slice_alf_chroma_idc == 3`; the Cb
                // resolution applies to Cr too.
                cb_alf
            };
            if let Some(alf_data) = luma_alf {
                if alf_map.any_luma_on() {
                    // Round-120: classified per-sample luma apply (§8.8.4.2
                    // + §8.8.4.3 + §8.9.4) drives the filter selection from
                    // the per-CTB classification rather than always using
                    // filter set 0.
                    alf::apply_alf_luma_classified_masked(pic, alf_data, alf_map, bd_y);
                    apply_chroma_alf_masked_or_whole_plane(
                        pic,
                        alf_map,
                        cb_alf,
                        cr_alf,
                        chroma_cb_enabled,
                        chroma_cr_enabled,
                        bd_c,
                    );
                } else if chroma_cb_enabled || chroma_cr_enabled {
                    // No luma CTUs flagged but chroma is enabled: only chroma
                    // apply (matches §8.9 lines 18099-18116 / round-113 wiring).
                    // Round 126: split the per-plane apply so each pulls from
                    // its own APS slot rather than forcing both through the
                    // luma APS via `apply_alf_with_map`.
                    apply_chroma_alf_masked_or_whole_plane(
                        pic,
                        alf_map,
                        cb_alf,
                        cr_alf,
                        chroma_cb_enabled,
                        chroma_cr_enabled,
                        bd_c,
                    );
                } else {
                    // Minimal-header IDR path (no per-CTU map threaded):
                    // preserve round-11 behaviour and apply the whole-plane
                    // filter (now spec-scaled per §8.8.4.2 eq. 1287).
                    alf::apply_alf(pic, alf_data, bd_y, bd_c);
                }
            }
        }
        if sps.sps_dra_flag && pic.bit_depth == 8 {
            // The DRA apply path is 8-bit-code-space (256-entry LUTs);
            // >8-bit DRA application is a documented follow-up, so
            // high-bit-depth pictures skip the mapping rather than
            // clamp through an 8-bit LUT.
            if let Some(dra_data) = self.dra_aps_for_pps(in_.dra_aps_id) {
                dra::apply_dra(pic, dra_data, bd_y, bd_c);
            }
        }
    }
}

impl EvcDecoder {
    /// Walk an `RefPicListStruct` to produce up to `n_active` reference
    /// POCs per §8.3.5. Each STRP entry contributes a running-sum POC
    /// (delta added to the prior entry's POC, with the slice's current
    /// POC seeding the chain). Round-10 also resolves LTRP entries by
    /// matching the entry's `poc_lsb_lt` against `(POC & (max_poc_lsb − 1))`
    /// for every DPB entry; the matching DPB POC is the LTRP's POC.
    ///
    /// LTRP resolution does NOT advance the running STRP delta sum:
    /// per §8.3.2 / §8.3.5 the spec defines the LTRP slot independently
    /// (the POC carried by the LTRP entry is absolute, not relative to
    /// the previous entry).
    fn build_ref_pocs(
        &self,
        rpl: &crate::rpl::RefPicListStruct,
        cur_poc: i32,
        n_active: usize,
        max_poc_lsb: i32,
    ) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(n_active);
        let mut running = cur_poc;
        for entry in rpl.entries.iter().take(n_active) {
            match entry {
                crate::rpl::RefPicListEntry::Strp { .. } => {
                    // STRP delta-POC is signed per §7.4.8 eq. 124.
                    let delta = entry.signed_delta_poc().expect("STRP carries signed delta");
                    running += delta;
                    out.push(running);
                }
                crate::rpl::RefPicListEntry::Ltrp { poc_lsb_lt } => {
                    // §8.3.2 LTRP marking: match the entry's poc_lsb_lt
                    // against (poc & mask) for each DPB entry. The first
                    // matching long-term-marked entry (or any entry if
                    // none have been explicitly marked LT yet) is the
                    // resolved POC. round-10 keeps every IDR + non-IDR
                    // picture marked as `used_for_reference == true`,
                    // so we accept any matching POC.
                    let mask = max_poc_lsb - 1;
                    let lsb = (*poc_lsb_lt as i32) & mask;
                    let matched =
                        self.dpb
                            .iter()
                            .find(|e| (e.poc & mask) == lsb)
                            .ok_or_else(|| {
                                Error::invalid(format!(
                                    "evc decoder: LTRP poc_lsb_lt {lsb} not in DPB"
                                ))
                            })?;
                    out.push(matched.poc);
                }
            }
        }
        if out.len() < n_active {
            return Err(Error::invalid(format!(
                "evc decoder: RPL has {} entries but n_active = {n_active}",
                out.len()
            )));
        }
        Ok(out)
    }
}

/// Run the §8.8.4.4 chroma type filtering process per coding tree block.
///
/// Round 145 closes round-113's chroma half: when the slice decoded a
/// per-CTU `alf_ctb_chroma_flag` / `alf_ctb_chroma2_flag` (the
/// `ChromaArrayType == 3` path), the chroma apply now walks the per-CTB
/// map via [`alf::apply_alf_chroma_masked`] instead of the
/// round-126 whole-plane fallback. When the chroma map is all-off but
/// the slice-level chroma enable is set (the `ChromaArrayType ∈ {1, 2}`
/// path, where the per-CTB flags are inferred 0 and never override the
/// slice enable), the whole-plane [`alf::apply_alf_chroma`] is invoked —
/// matching §8.9 lines 18099-18116.
#[allow(clippy::too_many_arguments)]
fn apply_chroma_alf_masked_or_whole_plane(
    pic: &mut crate::picture::YuvPicture,
    alf_map: &alf::AlfCtbMap,
    cb_alf: Option<&alf::AlfData>,
    cr_alf: Option<&alf::AlfData>,
    chroma_cb_enabled: bool,
    chroma_cr_enabled: bool,
    bd_c: u32,
) {
    if pic.chroma_format_idc == 0 {
        return;
    }
    let cb_map_on = alf_map.chroma_cb.iter().any(|&b| b);
    let cr_map_on = alf_map.chroma_cr.iter().any(|&b| b);
    if chroma_cb_enabled {
        if let Some(cb_data) = cb_alf {
            if cb_map_on {
                // §8.8.4.4 per-CTB Cb apply, gated by alf_ctb_chroma_flag.
                alf::apply_alf_chroma_masked(pic, &cb_data.chroma_filters[0], alf_map, 1, bd_c);
            } else {
                alf::apply_alf_chroma(pic, &cb_data.chroma_filters[0], 1, bd_c);
            }
        }
    }
    if chroma_cr_enabled {
        if let Some(cr_data) = cr_alf {
            if cr_map_on {
                // §8.8.4.4 per-CTB Cr apply, gated by alf_ctb_chroma2_flag.
                alf::apply_alf_chroma_masked(pic, &cr_data.chroma_filters[0], alf_map, 2, bd_c);
            } else {
                alf::apply_alf_chroma(pic, &cr_data.chroma_filters[0], 2, bd_c);
            }
        }
    }
}

fn picture_to_video_frame(pic: &crate::picture::YuvPicture, pts: Option<i64>) -> VideoFrame {
    let y_stride = pic.y_stride();
    let c_stride = pic.c_stride();
    if pic.bit_depth <= 8 {
        // 8-bit output: one byte per sample (Yuv420P-family layout).
        let pack8 = |src: &[u16]| -> Vec<u8> { src.iter().map(|&v| v as u8).collect() };
        VideoFrame {
            pts,
            planes: vec![
                VideoPlane {
                    stride: y_stride,
                    data: pack8(&pic.y),
                },
                VideoPlane {
                    stride: c_stride,
                    data: pack8(&pic.cb),
                },
                VideoPlane {
                    stride: c_stride,
                    data: pack8(&pic.cr),
                },
            ],
        }
    } else {
        // High-bit-depth output: two little-endian bytes per sample
        // (the Yuv420P10Le-family layout); strides are in bytes.
        let pack16 = |src: &[u16]| -> Vec<u8> {
            let mut out = Vec::with_capacity(src.len() * 2);
            for &v in src {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        };
        VideoFrame {
            pts,
            planes: vec![
                VideoPlane {
                    stride: y_stride * 2,
                    data: pack16(&pic.y),
                },
                VideoPlane {
                    stride: c_stride * 2,
                    data: pack16(&pic.cb),
                },
                VideoPlane {
                    stride: c_stride * 2,
                    data: pack16(&pic.cr),
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `derive_poc` mirrors §8.3.1 wrap detection: when the new LSB is
    /// less than the previous LSB by more than half of `MaxPicOrderCntLsb`,
    /// the MSB advances by `MaxPicOrderCntLsb`.
    #[test]
    fn derive_poc_wraps_on_lsb_rollover() {
        let mut dec = EvcDecoder::new();
        // 8-bit POC LSB → MaxPicOrderCntLsb = 256.
        let max = 256;
        // Start at POC 100.
        dec.poc_msb = 0;
        dec.prev_poc_lsb = 100;
        // Next slice LSB = 110 → no wrap. POC = 110.
        assert_eq!(dec.derive_poc(110, max), 110);
        // Next slice LSB = 5 (after prev=200) → wrap. POC = 256 + 5 = 261.
        dec.prev_poc_lsb = 200;
        assert_eq!(dec.derive_poc(5, max), 261);
        // Next slice LSB = 200 (after prev=5) → reverse wrap. POC = -56.
        dec.prev_poc_lsb = 5;
        assert_eq!(dec.derive_poc(200, max), -56);
    }

    /// DPB inserts up to MAX_DPB_ENTRIES; eviction picks the lowest POC.
    #[test]
    fn dpb_evicts_lowest_poc_at_capacity() {
        use crate::picture::YuvPicture;
        let mut dec = EvcDecoder::new();
        for i in 0..MAX_DPB_ENTRIES + 1 {
            let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
            dec.dpb_insert(DpbEntry {
                pic,
                poc: i as i32,
                pts: None,
                used_for_reference: true,
                output_emitted: false,
                side_info: None,
                ref_pocs_l0: Vec::new(),
                ref_pocs_l1: Vec::new(),
            });
        }
        assert_eq!(dec.dpb.len(), MAX_DPB_ENTRIES);
        // POC 0 was evicted; POC 1..MAX_DPB_ENTRIES still present.
        assert!(dec.dpb_find(0).is_none());
        assert!(dec.dpb_find(1).is_some());
        assert!(dec.dpb_find(MAX_DPB_ENTRIES as i32).is_some());
    }

    /// `dpb_flush` clears every entry + resets POC tracker.
    #[test]
    fn dpb_flush_clears_all() {
        use crate::picture::YuvPicture;
        let mut dec = EvcDecoder::new();
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        dec.dpb_insert(DpbEntry {
            pic,
            poc: 5,
            pts: None,
            used_for_reference: true,
            output_emitted: false,
            side_info: None,
            ref_pocs_l0: Vec::new(),
            ref_pocs_l1: Vec::new(),
        });
        dec.poc_msb = 256;
        dec.prev_poc_lsb = 100;
        dec.dpb_flush();
        assert!(dec.dpb.is_empty());
        assert_eq!(dec.poc_msb, 0);
        assert_eq!(dec.prev_poc_lsb, 0);
    }

    /// **Round-10 flush() drain.** A DPB entry that was never pushed
    /// to the output queue (synthesised here by direct insertion) is
    /// emitted by `flush()` in POC order without duplicating entries
    /// that were already enqueued.
    #[test]
    fn round10_flush_drains_unemitted_dpb_entries_in_poc_order() {
        use crate::picture::YuvPicture;
        let mut dec = EvcDecoder::new();
        // Insert three DPB entries directly. POC 1 is marked emitted
        // (simulating "already in out queue from a prior receive_frame").
        // POCs 0 and 2 are NOT yet in `out`. After flush(), the out
        // queue holds POCs {0, 2} in ascending POC order.
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        dec.dpb_insert(DpbEntry {
            pic: pic.clone(),
            poc: 1,
            pts: None,
            used_for_reference: true,
            output_emitted: true,
            side_info: None,
            ref_pocs_l0: Vec::new(),
            ref_pocs_l1: Vec::new(),
        });
        dec.dpb_insert(DpbEntry {
            pic: pic.clone(),
            poc: 0,
            pts: None,
            used_for_reference: true,
            output_emitted: false,
            side_info: None,
            ref_pocs_l0: Vec::new(),
            ref_pocs_l1: Vec::new(),
        });
        dec.dpb_insert(DpbEntry {
            pic,
            poc: 2,
            pts: None,
            used_for_reference: true,
            output_emitted: false,
            side_info: None,
            ref_pocs_l0: Vec::new(),
            ref_pocs_l1: Vec::new(),
        });
        assert!(dec.out.is_empty());
        dec.drain_dpb_to_output();
        // Expect POCs 0 and 2 in the out queue, sorted ascending.
        let pocs: Vec<i32> = dec.out_pocs.iter().copied().collect();
        assert_eq!(pocs, vec![0, 2]);
        // Calling flush again is idempotent: no duplicate emission.
        dec.drain_dpb_to_output();
        let pocs: Vec<i32> = dec.out_pocs.iter().copied().collect();
        assert_eq!(pocs, vec![0, 2]);
    }

    /// **Round-10 LTRP RPL resolution.** An LTRP entry's `poc_lsb_lt`
    /// resolves to the matching DPB entry's POC (matched against
    /// `poc & (max_poc_lsb − 1)`).
    #[test]
    fn round10_ltrp_rpl_resolves_against_dpb() {
        use crate::picture::YuvPicture;
        use crate::rpl::{RefPicListEntry, RefPicListStruct};
        let mut dec = EvcDecoder::new();
        // 8-bit POC LSB → max_poc_lsb = 256. Insert DPB entries at
        // POCs 0, 1, 5. We'll request an LTRP with poc_lsb_lt = 5
        // (matches POC 5 directly because 5 & 0xFF = 5).
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        for &p in &[0, 1, 5] {
            dec.dpb_insert(DpbEntry {
                pic: pic.clone(),
                poc: p,
                pts: None,
                used_for_reference: true,
                output_emitted: true,
                side_info: None,
                ref_pocs_l0: Vec::new(),
                ref_pocs_l1: Vec::new(),
            });
        }
        let rpl = RefPicListStruct {
            num_strp_entries: 0,
            num_ltrp_entries: 1,
            entries: vec![RefPicListEntry::Ltrp { poc_lsb_lt: 5 }],
        };
        let pocs = dec.build_ref_pocs(&rpl, /* cur_poc */ 6, 1, 256).unwrap();
        assert_eq!(pocs, vec![5]);
    }

    /// LTRP entry with no DPB match is rejected as invalid bitstream.
    #[test]
    fn round10_ltrp_missing_dpb_entry_is_invalid() {
        use crate::rpl::{RefPicListEntry, RefPicListStruct};
        let dec = EvcDecoder::new(); // empty DPB
        let rpl = RefPicListStruct {
            num_strp_entries: 0,
            num_ltrp_entries: 1,
            entries: vec![RefPicListEntry::Ltrp { poc_lsb_lt: 7 }],
        };
        let err = dec.build_ref_pocs(&rpl, 10, 1, 256).unwrap_err();
        assert!(format!("{err}").contains("LTRP"));
    }

    /// Mixed STRP + LTRP RPL: STRP delta chain advances normally
    /// while LTRP slot pulls the absolute POC out of the DPB.
    #[test]
    fn round10_mixed_strp_and_ltrp_resolve() {
        use crate::picture::YuvPicture;
        use crate::rpl::{RefPicListEntry, RefPicListStruct};
        let mut dec = EvcDecoder::new();
        let pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        // DPB: POCs 0, 4, 7. cur_poc = 8. RPL = [STRP delta=-1, LTRP poc_lsb_lt=4].
        for &p in &[0, 4, 7] {
            dec.dpb_insert(DpbEntry {
                pic: pic.clone(),
                poc: p,
                pts: None,
                used_for_reference: true,
                output_emitted: true,
                side_info: None,
                ref_pocs_l0: Vec::new(),
                ref_pocs_l1: Vec::new(),
            });
        }
        let rpl = RefPicListStruct {
            num_strp_entries: 1,
            num_ltrp_entries: 1,
            entries: vec![
                RefPicListEntry::Strp {
                    delta_poc_st: 1,
                    sign: false,
                }, // delta = -1 → 8 + (-1) = 7
                RefPicListEntry::Ltrp { poc_lsb_lt: 4 }, // → POC 4
            ],
        };
        let pocs = dec.build_ref_pocs(&rpl, 8, 2, 256).unwrap();
        assert_eq!(pocs, vec![7, 4]);
    }

    // =================================================================
    // Round 126 — APS cache routed by `adaptation_parameter_set_id`.
    // =================================================================

    /// Distinct ALF APS payloads stored at distinct ids land in distinct
    /// cache slots and don't overwrite each other. The slice's
    /// `slice_alf_luma_aps_id` resolves the correct slot via
    /// [`EvcDecoder::alf_aps_for_slice`]. Round-11 / round-120 single-APS
    /// streams are unaffected because the same lookup path also serves the
    /// "slice references id == cache slot" case.
    #[test]
    fn round126_alf_aps_cache_distinct_slots_resolve_independently() {
        let mut dec = EvcDecoder::new();
        // Build two AlfData instances with different DC offsets so we can
        // tell them apart by the resolved-filter's centre tap.
        let mut luma_a = [alf::AlfLumaFilter::default(); alf::ALF_MAX_LUMA_FILTERS];
        luma_a[0].coef[12] = 100;
        let a = alf::AlfData {
            luma_filter_signal: true,
            luma_filters: luma_a,
            ..alf::AlfData::default()
        };
        let mut luma_b = [alf::AlfLumaFilter::default(); alf::ALF_MAX_LUMA_FILTERS];
        luma_b[0].coef[12] = 7;
        let b = alf::AlfData {
            luma_filter_signal: true,
            luma_filters: luma_b,
            ..alf::AlfData::default()
        };
        dec.alf_aps[3] = Some(a);
        dec.alf_aps[19] = Some(b);
        dec.last_alf_aps_id = Some(19); // simulates round-11 single-APS
                                        // Slice references id 3 → cache slot 3 (the `a` payload).
        let resolved = dec.alf_aps_for_slice(Some(3)).expect("slot 3 populated");
        assert_eq!(resolved.luma_filters[0].coef[12], 100);
        // Slice references id 19 → cache slot 19 (the `b` payload).
        let resolved = dec.alf_aps_for_slice(Some(19)).expect("slot 19 populated");
        assert_eq!(resolved.luma_filters[0].coef[12], 7);
        // Slice references id 5 (empty slot) → None — the spec-correct
        // outcome for a slice that mis-references an APS that wasn't
        // signalled in this CVS.
        assert!(dec.alf_aps_for_slice(Some(5)).is_none());
        // Slice did NOT surface an id (minimal-header IDR fixture path) →
        // back-compat fallback to the most-recently-stored APS (id 19,
        // the `b` payload).
        let resolved = dec
            .alf_aps_for_slice(None)
            .expect("fallback to last-stored APS");
        assert_eq!(resolved.luma_filters[0].coef[12], 7);
    }

    /// DRA APS routing follows the same shape — keyed by the PPS's
    /// `pic_dra_aps_id` (when `pic_dra_enabled_flag == 1`), with the
    /// most-recently-stored id as fallback.
    #[test]
    fn round126_dra_aps_cache_routes_via_pps_id() {
        let mut dec = EvcDecoder::new();
        // Two DraData with different `scale[0]` so we can tell them apart
        // by inspecting the cached struct directly.
        let mut scale_a = [8i16; dra::DRA_MAX_RANGES];
        scale_a[0] = 5;
        let a = dra::DraData {
            descriptor_present: true,
            scale: scale_a,
            ..dra::DraData::default()
        };
        let mut scale_b = [8i16; dra::DRA_MAX_RANGES];
        scale_b[0] = 17;
        let b = dra::DraData {
            descriptor_present: true,
            scale: scale_b,
            ..dra::DraData::default()
        };
        dec.dra_aps[2] = Some(a);
        dec.dra_aps[10] = Some(b);
        dec.last_dra_aps_id = Some(10);
        assert_eq!(
            dec.dra_aps_for_pps(Some(2)).unwrap().scale[0],
            5,
            "PPS id 2 selects scale = 5"
        );
        assert_eq!(
            dec.dra_aps_for_pps(Some(10)).unwrap().scale[0],
            17,
            "PPS id 10 selects scale = 17"
        );
        assert!(dec.dra_aps_for_pps(Some(7)).is_none());
        assert_eq!(
            dec.dra_aps_for_pps(None).unwrap().scale[0],
            17,
            "fallback to most-recently-stored DRA APS"
        );
    }

    /// An ALF NAL with id `N` populates slot `N` of the cache (and only
    /// that slot). Sending a second ALF NAL at id `N` overwrites slot `N`
    /// per the spec's update-by-id semantics. A NAL at a different id
    /// `M` leaves slot `N` intact.
    #[test]
    fn round126_aps_nal_writes_indexed_cache_slot() {
        use crate::aps::APS_PARAMS_TYPE_ALF;
        use crate::sps::tests::BitEmitter;
        let mut dec = EvcDecoder::new();
        // Build a minimal valid ALF APS RBSP at id `id` whose payload is
        // a single `alf_luma_filter_signal_flag = 0` /
        // `alf_chroma_filter_signal_flag = 0` (i.e. NumAlfCoefs neither
        // luma nor chroma) followed by `aps_extension_flag = 0` +
        // rbsp_trailing_bits. `parse_alf_data` produces a Default::default()
        // when both signal flags are 0, so the round-trip just verifies the
        // cache slot is populated by a parse-success.
        fn emit_alf_aps_rbsp(id: u32) -> Vec<u8> {
            let mut e = BitEmitter::new();
            e.u(5, id);
            e.u(3, APS_PARAMS_TYPE_ALF as u32);
            // alf_data() body: luma_signal = 0, chroma_signal = 0.
            e.u(1, 0);
            e.u(1, 0);
            // aps_extension_flag = 0.
            e.u(1, 0);
            e.finish_with_trailing_bits();
            e.into_bytes()
        }
        // Send APS at id = 4 — populates slot 4.
        let aps4 = crate::aps::parse(&emit_alf_aps_rbsp(4)).expect("parse id=4");
        assert_eq!(aps4.adaptation_parameter_set_id, 4);
        let alf4 = alf::parse_alf_data(&aps4.payload_raw).expect("alf body");
        dec.alf_aps[4] = Some(alf4);
        dec.last_alf_aps_id = Some(4);
        assert!(dec.alf_aps[4].is_some());
        assert!(dec.alf_aps[7].is_none());
        // Send APS at id = 7 — populates slot 7, leaves slot 4 intact.
        let aps7 = crate::aps::parse(&emit_alf_aps_rbsp(7)).expect("parse id=7");
        assert_eq!(aps7.adaptation_parameter_set_id, 7);
        let alf7 = alf::parse_alf_data(&aps7.payload_raw).expect("alf body");
        dec.alf_aps[7] = Some(alf7);
        dec.last_alf_aps_id = Some(7);
        assert!(dec.alf_aps[4].is_some(), "id=4 slot must persist");
        assert!(dec.alf_aps[7].is_some());
        // last-stored fallback now points at id 7.
        assert_eq!(dec.last_alf_aps_id, Some(7));
    }

    /// Round 151: the new spec-faithful DRA APS cache slot stores a
    /// `(DraSyntax, DraDerived)` pair indexed by APS id, parallel to
    /// the legacy `dra_aps[id]: Option<DraData>` slot, so a follow-up
    /// round wiring §8.9.3 luma mapping + §8.9.6 chroma scale
    /// derivation can read directly from the spec-faithful state.
    /// Verified here by hand-storing a derived pair and round-tripping
    /// its `dra_descriptor1` / `num_bits_dra_scale` / `joined_scale_flag`
    /// fields out of the cache.
    #[test]
    fn round151_dra_syntax_aps_cache_holds_spec_faithful_pair() {
        use crate::dra::{derive_dra_state, DraSyntax, DRA_MAX_RANGES_V2};
        let mut dec = EvcDecoder::new();
        let syn = DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 0,
            dra_equal_ranges_flag: true,
            dra_global_offset: 1,
            dra_delta_range: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 1;
                a
            },
            dra_scale_value: {
                let mut a = [0u16; DRA_MAX_RANGES_V2];
                a[0] = 512;
                a
            },
            dra_cb_scale_value: 512,
            dra_cr_scale_value: 512,
            dra_table_idx: 58,
        };
        let der = derive_dra_state(&syn, 10).unwrap();
        dec.dra_syntax_aps[5] = Some((syn, der));
        let (got_syn, got_der) = dec.dra_syntax_aps[5].as_ref().expect("slot 5 populated");
        assert_eq!(got_syn.dra_descriptor1, 4);
        assert_eq!(got_syn.dra_descriptor2, 9);
        assert_eq!(got_der.num_bits_dra_scale, 13);
        // dra_table_idx == 58 ⇒ DraJoinedScaleFlag = 0.
        assert!(!got_der.joined_scale_flag);
        // Slot 6 (untouched) must remain unpopulated.
        assert!(dec.dra_syntax_aps[6].is_none());
    }

    // ------------------------------------------------------------------
    // Round 181 — spec-faithful §8.9.3 luma inverse mapping wiring tests.
    //
    // [`EvcDecoder::apply_luma_inverse_mapping_spec_faithful`] is the
    // public entry point that closes the round-151 → round-174 → r181
    // chain: parser cache → derived state with eq. 118/120/121 →
    // off-by-one reconciliation → §8.9.3 per-sample apply.
    //
    // Tests below pin the four observable behaviours:
    //   * empty cache slot ⇒ Ok(false), picture untouched.
    //   * identity scale (dra_scale_value = 512 at all ranges) +
    //     `dra_descriptor2 = 9` (Q9) ⇒ §8.9.3 LUT is exactly the
    //     identity on `[0, 255]`.
    //   * doubled scale (dra_scale_value = 1024) ⇒ mapped values are
    //     halved against the pre-scale (single-range case).
    //   * `dra_scale_value[0] == 0` ⇒ Err propagated from
    //     `fill_inv_luma_scales_range_zero`.
    // ------------------------------------------------------------------

    /// Build a `DraSyntax` with a single range covering the whole 8-bit
    /// luma domain and `dra_scale_value[0] = scale`. The §7.4.7 derivation
    /// runs in Q9 (`dra_descriptor2 = 9`).
    #[cfg(test)]
    fn round181_make_syntax(scale: u16) -> dra::DraSyntax {
        use crate::dra::DRA_MAX_RANGES_V2;
        let mut dra_scale_value = [0u16; DRA_MAX_RANGES_V2];
        dra_scale_value[0] = scale;
        let mut dra_delta_range = [0u16; DRA_MAX_RANGES_V2];
        // Single range covering the whole 8-bit luma codespace [0, 256).
        // §7.4.7 (eq. 117) reads `dra_delta_range[0]` as the range width
        // in input-space units; 256 places the range edge at the top of
        // an 8-bit picture so every sample falls into range 0.
        dra_delta_range[0] = 256;
        dra::DraSyntax {
            dra_descriptor1: 4,
            dra_descriptor2: 9,
            dra_number_ranges_minus1: 0,
            dra_equal_ranges_flag: true,
            dra_global_offset: 0,
            dra_delta_range,
            dra_scale_value,
            dra_cb_scale_value: 512,
            dra_cr_scale_value: 512,
            // dra_table_idx = 58 ⇒ joined_scale_flag = false (single
            // scale per range, no joined-scale path).
            dra_table_idx: 58,
        }
    }

    /// Empty cache slot — the public entry point returns `Ok(false)` and
    /// leaves the picture's luma plane untouched. Exercises the early-out
    /// in `dra_syntax_aps_for_pps`.
    #[test]
    fn round181_apply_luma_inv_map_empty_slot_is_noop() {
        let dec = EvcDecoder::new();
        let mut pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        // Seed luma with a known pattern so we can detect any mutation.
        for (i, v) in pic.y.iter_mut().enumerate() {
            *v = (i as u16).wrapping_mul(17);
        }
        let before = pic.y.clone();
        let applied = dec
            .apply_luma_inverse_mapping_spec_faithful(&mut pic, Some(3))
            .expect("empty slot is a clean no-op, not an error");
        assert!(!applied, "empty cache slot returns Ok(false)");
        assert_eq!(pic.y, before, "luma plane must be untouched");
    }

    /// `pps_dra_aps_id = None` ⇒ no fallback to `last_dra_aps_id`; spec-
    /// faithful mode is strict (the legacy path's last-slot fallback is
    /// deliberately not mirrored here, because §8.9.3 is invoked with an
    /// explicit `pic_dra_aps_id` per §8.9 ordering).
    #[test]
    fn round181_apply_luma_inv_map_none_aps_id_is_noop() {
        let dec = EvcDecoder::new();
        let mut pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        pic.y.fill(123);
        let applied = dec
            .apply_luma_inverse_mapping_spec_faithful(&mut pic, None)
            .expect("None aps id is a clean no-op");
        assert!(!applied);
        assert!(pic.y.iter().all(|&v| v == 123));
    }

    /// `dra_scale_value[0] = 512` at `dra_descriptor2 = 9` is the Q9
    /// representation of 1.0. The eq. 118 inverse `(1 << 18) / 512 = 512`
    /// yields `InvLumaScales[0] = 512`, which §8.9.3 eq. 1374-1375
    /// reduces to `(0 + 512 * s + 256) >> 9 = s` for every input sample
    /// `s ∈ [0, 255]` once `DraOffsets[0]` is the matched zero from the
    /// off-by-one reconciliation. The §8.9.3 LUT is therefore the
    /// identity on the 8-bit codespace, modulo clipping at the upper
    /// edge.
    #[test]
    fn round181_apply_luma_inv_map_identity_scale_is_identity_on_8bit_codespace() {
        use crate::dra::derive_dra_state;
        let mut dec = EvcDecoder::new();
        let syn = round181_make_syntax(512);
        let der = derive_dra_state(&syn, 10).unwrap();
        dec.dra_syntax_aps[7] = Some((syn, der));

        // Walk every 8-bit sample value and check the post-§8.9.3 LUT
        // returns the input unchanged. We populate a 256-px linear ramp
        // so `pic.y[i] = i`.
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        for (i, v) in pic.y.iter_mut().enumerate() {
            *v = i as u16;
        }
        let applied = dec
            .apply_luma_inverse_mapping_spec_faithful(&mut pic, Some(7))
            .expect("identity scale must not error");
        assert!(applied, "populated slot ⇒ Ok(true)");
        for (i, &v) in pic.y.iter().enumerate() {
            assert_eq!(v, i as u16, "identity LUT failed at i = {i}: got {v}");
        }
    }

    /// Doubled scale (`dra_scale_value[0] = 1024`) maps a 0.5× inverse
    /// onto the luma plane. eq. 118 gives `InvLumaScales[0] = (1 << 18) /
    /// 1024 = 256`; eq. 1375 then maps sample `s` to `(off + 256 * s +
    /// 256) >> 9`. With the off-by-one-reconciled `DraOffsets[0]` the
    /// midpoint sample 128 maps to ~64 (halved), and 0 stays 0.
    #[test]
    fn round181_apply_luma_inv_map_doubled_scale_halves_midpoint() {
        use crate::dra::derive_dra_state;
        let mut dec = EvcDecoder::new();
        let syn = round181_make_syntax(1024);
        let der = derive_dra_state(&syn, 10).unwrap();
        dec.dra_syntax_aps[2] = Some((syn, der));

        let mut pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        // 16 samples covering the input domain in 16-wide steps.
        for (i, v) in pic.y.iter_mut().enumerate() {
            *v = (i as u16).saturating_mul(16);
        }
        let applied = dec
            .apply_luma_inverse_mapping_spec_faithful(&mut pic, Some(2))
            .unwrap();
        assert!(applied);
        // 0 ⇒ 0 (incrValue = 0, offset = 0 in the identity-offset
        // single-range case, mappedSample = (0 + 0 + 256) >> 9 = 0).
        assert_eq!(pic.y[0], 0);
        // Strict monotonicity check across the full mapped sequence
        // (eq. 1374-1375 is linear in `s` with positive slope).
        for w in pic.y.windows(2) {
            assert!(
                w[0] <= w[1],
                "§8.9.3 mapping must be monotone non-decreasing in s; saw {} -> {}",
                w[0],
                w[1]
            );
        }
        // Final input s = 240 with InvLumaScales[0] = 256 ⇒
        // (0 + 256*240 + 256) >> 9 = (61440 + 256) >> 9 = 120 (halved
        // against the pre-scale, as expected for a 2× scale on the
        // forward path / 0.5× inverse).
        assert_eq!(pic.y[15], 120);
    }

    /// `dra_scale_value[0] == 0` is explicitly forbidden by §7.4.7 (the
    /// inverse `(1 << 18) / dsv0` would divide by zero).
    /// `fill_inv_luma_scales_range_zero` returns an `Error::invalid` for
    /// this case, and the public entry point must propagate it without
    /// touching the picture.
    #[test]
    fn round181_apply_luma_inv_map_zero_scale_value_propagates_error() {
        use crate::dra::derive_dra_state;
        let mut dec = EvcDecoder::new();
        // Build a derived state on a valid scale, then *poison* the
        // syntax's `dra_scale_value[0]` to zero so the reconciliation
        // step trips the divide-by-zero guard. `derive_dra_state` skips
        // index 0 in its loop so we can't trip eq. 118 there directly;
        // the round-174 reconciliation helper is where the check lives.
        let valid_syn = round181_make_syntax(512);
        let der = derive_dra_state(&valid_syn, 10).unwrap();
        let mut poisoned = valid_syn.clone();
        poisoned.dra_scale_value[0] = 0;
        dec.dra_syntax_aps[1] = Some((poisoned, der));

        let mut pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        pic.y.fill(200);
        let before = pic.y.clone();
        let err = dec
            .apply_luma_inverse_mapping_spec_faithful(&mut pic, Some(1))
            .expect_err("dra_scale_value[0] == 0 must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("dra_scale_value[0] == 0"),
            "error message should carry the §7.4.7 violation phrase; got: {msg}"
        );
        // Picture state must be untouched on the error path (the apply
        // happens after the reconciliation, so a clone+fail sequence
        // means `pic.y` was never written).
        assert_eq!(pic.y, before, "picture must be untouched on error");
    }

    /// The new public API must NOT collide with the legacy round-148
    /// [`dra::apply_dra`] path. We verify by populating BOTH cache slots
    /// for the same `pps_dra_aps_id` and checking that
    /// `apply_luma_inverse_mapping_spec_faithful` only consumes
    /// `dra_syntax_aps`; the legacy `DraData` slot remains a valid input
    /// for the post-filter pipeline.
    #[test]
    fn round181_spec_faithful_path_is_orthogonal_to_legacy_apply_dra() {
        use crate::dra::derive_dra_state;
        let mut dec = EvcDecoder::new();
        let syn = round181_make_syntax(512);
        let der = derive_dra_state(&syn, 10).unwrap();
        dec.dra_syntax_aps[4] = Some((syn, der));
        // Populate the legacy slot with a hand-built `DraData` so we can
        // confirm the spec-faithful entry point doesn't read it. A
        // minimum-viable `DraData` with `descriptor_present = false` is
        // the round-11 no-op case — the legacy slot is "there" but the
        // legacy `apply_dra` would early-return; the spec-faithful entry
        // point ignores this slot entirely.
        let legacy = DraData::default();
        dec.dra_aps[4] = Some(legacy);

        let mut pic = YuvPicture::new(4, 4, 1, 8).unwrap();
        for (i, v) in pic.y.iter_mut().enumerate() {
            *v = (i as u16) * 16;
        }
        let before = pic.y.clone();
        let applied = dec
            .apply_luma_inverse_mapping_spec_faithful(&mut pic, Some(4))
            .unwrap();
        assert!(applied, "spec-faithful path reads dra_syntax_aps[4]");
        // Identity scale ⇒ luma plane unchanged.
        assert_eq!(pic.y, before);
    }

    /// Round 391: `picture_to_video_frame` packs 8-bit pictures one
    /// byte per sample and >8-bit pictures as two little-endian bytes
    /// per sample (the Yuv420P10Le-family layout) with byte strides.
    #[test]
    fn round391_video_frame_packing_8_and_10_bit() {
        let mut pic8 = crate::picture::YuvPicture::new(4, 4, 1, 8).unwrap();
        pic8.y[0] = 200;
        let f8 = picture_to_video_frame(&pic8, Some(7));
        assert_eq!(f8.planes[0].stride, 4);
        assert_eq!(f8.planes[0].data.len(), 16);
        assert_eq!(f8.planes[0].data[0], 200);
        assert_eq!(f8.planes[1].data.len(), 4);

        let mut pic10 = crate::picture::YuvPicture::new(4, 4, 1, 10).unwrap();
        pic10.y[0] = 700; // 0x02BC — exceeds the 8-bit range
        let f10 = picture_to_video_frame(&pic10, Some(7));
        assert_eq!(f10.planes[0].stride, 8, "byte stride = 2 × width");
        assert_eq!(f10.planes[0].data.len(), 32);
        assert_eq!(
            &f10.planes[0].data[0..2],
            &700u16.to_le_bytes(),
            "little-endian 10-bit sample"
        );
        // Mid-level fill 512 = 0x0200 for the untouched samples.
        assert_eq!(&f10.planes[0].data[2..4], &512u16.to_le_bytes());
        assert_eq!(&f10.planes[1].data[0..2], &512u16.to_le_bytes());
    }
}
