//! EVC reconstructed picture buffer + per-CU pipeline glue
//! (ISO/IEC 23094-1 §8.7.5).
//!
//! Round-3 scope: a single 8-bit YUV frame buffer (yuv420p only) plus the
//! intra-prediction reference-sample fetch and the picture-construction
//! step `recSamples = clip(predSamples + resSamples)` (eq. 1091). The
//! picture buffer is initialised with `1 << (bit_depth − 1)` so that a
//! brand-new IDR picture's first CU finds the spec-mandated "not
//! available" substitution value already in place at every neighbour
//! lookup.
//!
//! 10-bit support, deblocking, ALF and DRA are deferred to round 4.

use oxideav_core::{Error, Result};

use crate::intra::{predict, IntraMode, RefSamples};

/// Reconstructed picture buffer for a yuv420p 8-bit frame.
///
/// Coordinates are in luma sample units; chroma planes are accessed via
/// the `cb` / `cr` helpers and use sub-sampled coordinates internally
/// (`SubWidthC = SubHeightC = 2` for 4:2:0).
#[derive(Debug, Clone)]
pub struct YuvPicture {
    pub width: u32,
    pub height: u32,
    pub chroma_format_idc: u32,
    /// Bit depth — round 3 only supports 8.
    pub bit_depth: u32,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

impl YuvPicture {
    /// Allocate a YUV picture of the given dimensions, pre-filled with
    /// the spec's "not available" substitution value
    /// (`1 << (bit_depth - 1) = 128` for 8-bit). This means an IDR
    /// slice's first CU finds neighbour samples that already match what
    /// §8.4.4.2 would compute for the missing-neighbour case.
    pub fn new(width: u32, height: u32, chroma_format_idc: u32, bit_depth: u32) -> Result<Self> {
        if bit_depth != 8 {
            return Err(Error::unsupported(format!(
                "evc picture: round-3 supports 8-bit only (got {bit_depth})"
            )));
        }
        if width == 0 || height == 0 {
            return Err(Error::invalid("evc picture: zero dimensions"));
        }
        let fill: u8 = 1u8 << (bit_depth - 1);
        let n_y = (width as usize) * (height as usize);
        let (cw, ch) = chroma_dims(width, height, chroma_format_idc)?;
        let n_c = cw * ch;
        Ok(Self {
            width,
            height,
            chroma_format_idc,
            bit_depth,
            y: vec![fill; n_y],
            cb: vec![fill; n_c],
            cr: vec![fill; n_c],
        })
    }

    /// Stride of the luma plane in bytes.
    pub fn y_stride(&self) -> usize {
        self.width as usize
    }

    /// Stride of either chroma plane in bytes.
    pub fn c_stride(&self) -> usize {
        chroma_dims(self.width, self.height, self.chroma_format_idc)
            .map(|(w, _)| w)
            .unwrap_or(0)
    }

    /// Build the reference-sample neighbourhood for an intra block at
    /// the given top-left position. cIdx selects the colour component
    /// (0=Y, 1=Cb, 2=Cr). Out-of-picture samples and "not available"
    /// neighbours fall back to `1 << (bit_depth - 1)` per §8.4.4.2
    /// sps_eipd_flag=0 path.
    ///
    /// Round-3 uses the simplified availability rule:
    /// * top row available iff `y > 0`
    /// * left column available iff `x > 0`
    /// * top-left available iff both
    ///
    /// Tile/slice boundary handling is omitted (round-3 fixtures are
    /// single-tile, single-slice).
    pub fn fetch_intra_refs(
        &self,
        x: u32,
        y: u32,
        n_cb_w: usize,
        n_cb_h: usize,
        c_idx: u32,
    ) -> RefSamples {
        let mut refs = RefSamples::unavailable(n_cb_w, n_cb_h, self.bit_depth);
        let span = n_cb_w + n_cb_h;
        let (plane, stride, w, h) = self.plane_view(c_idx);
        if y > 0 {
            for i in 0..span {
                let xi = x as usize + i;
                if xi < w {
                    refs.top[i] = plane[(y as usize - 1) * stride + xi] as i32;
                }
            }
        }
        if x > 0 {
            for j in 0..span {
                let yj = y as usize + j;
                if yj < h {
                    refs.left[j] = plane[yj * stride + (x as usize - 1)] as i32;
                }
            }
        }
        if x > 0 && y > 0 {
            refs.top_left = plane[(y as usize - 1) * stride + (x as usize - 1)] as i32;
        }
        refs
    }

    /// Build the **EIPD** (§8.4.4.1) reference-sample neighbourhood for an
    /// intra block at the given component-domain top-left position, running
    /// the §8.4.4.1 construction + §8.4.4.2 substitution via
    /// [`crate::eipd_ref::construct_eipd_refs`].
    ///
    /// The EIPD neighbourhood spans `nCbW + nCbH` samples on each of the
    /// top row `p[x][-1]` and left column `p[-1][y]`, plus the `p[-1][-1]`
    /// corner and — when `sps_suco_flag == 1` — the right column
    /// `p[nCbW][y]`.
    ///
    /// Availability mirrors [`fetch_intra_refs`](Self::fetch_intra_refs)'s
    /// simplified causal rule (top available iff above the block, left iff
    /// to the block's left, both in-picture), extended to the right column:
    /// `p[nCbW][y]` is available iff `right_available` is set (the caller's
    /// SUCO split-unit-coding-order resolution) **and** the position is
    /// in-picture and causal. `constrained_intra_pred_flag` is taken as 0
    /// here (the round-3 single-slice fixtures are all-intra); a future
    /// wiring threads the real predicate through the `available` closure.
    ///
    /// Returns the constructed [`crate::eipd::EipdRefSamples`] ready for
    /// [`crate::eipd::predict_eipd`].
    #[allow(clippy::too_many_arguments)]
    pub fn fetch_eipd_refs(
        &self,
        x: u32,
        y: u32,
        n_cb_w: usize,
        n_cb_h: usize,
        c_idx: u32,
        sps_suco_flag: bool,
        right_available: bool,
    ) -> crate::eipd::EipdRefSamples {
        let (plane, stride, w, h) = self.plane_view(c_idx);
        let xb = x as i64;
        let yb = y as i64;
        let wi = w as i64;
        let hi = h as i64;

        // Availability: a component-domain offset (ox, oy) relative to the
        // block's top-left. `available` returns true only for causal,
        // in-picture neighbour locations.
        let available = |ox: i64, oy: i64| -> bool {
            let ax = xb + ox;
            let ay = yb + oy;
            if ax < 0 || ay < 0 || ax >= wi || ay >= hi {
                return false;
            }
            if oy < 0 {
                // Top row / corner: must be above the block (y > 0), and the
                // sample column already reconstructed (above-row is always
                // causal for ax to the right; the simplified model treats the
                // whole above row as available when the block is not in the
                // first row).
                yb > 0
            } else if ox < 0 {
                // Left column: must be to the block's left (x > 0).
                xb > 0
            } else {
                // Right column p[nCbW][y]: gated on the caller's SUCO
                // right-available resolution.
                right_available
            }
        };

        let sample = |ox: i64, oy: i64| -> i32 {
            let ax = (xb + ox) as usize;
            let ay = (yb + oy) as usize;
            plane[ay * stride + ax] as i32
        };

        crate::eipd_ref::construct_eipd_refs(
            n_cb_w,
            n_cb_h,
            self.bit_depth,
            true, // sps_eipd_flag (this is the EIPD fetch path)
            sps_suco_flag,
            |ox, oy| available(ox as i64, oy as i64),
            // Only invoked for available offsets, so bounds hold.
            |ox, oy| sample(ox as i64, oy as i64),
        )
        .refs
    }

    /// Apply the §8.7.6 HTDF post-reconstruction filter to the luma block
    /// at `(x, y)` in place, the data-plane bridge for the `htdf` module.
    ///
    /// No-op (returns `false`) when the §8.7.6.1 applicability gates
    /// disqualify the block ([`crate::htdf::htdf_applies`]); otherwise runs
    /// the §8.7.6.2 padding + §8.7.6.3 LUT derivation + §8.7.6.1 filter and
    /// writes the modified samples back into the luma plane, returning
    /// `true`.
    ///
    /// `is_intra` is `LumaPredMode[xCb][yCb] == MODE_INTRA`;
    /// `is_inter_square_ge32` is the eq.-1106 `MODE_INTER && square &&
    /// Min >= 32` predicate selecting the qpIdx branch. Border availability
    /// uses the in-picture-extent rule (single-slice / single-tile,
    /// `constrained_intra_pred_flag == 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_htdf_luma(
        &mut self,
        x: u32,
        y: u32,
        n_cb_w: usize,
        n_cb_h: usize,
        qp_y: i32,
        is_intra: bool,
        is_inter_square_ge32: bool,
    ) -> bool {
        use crate::htdf::{
            derive_htdf_lut, filter_block, htdf_applies, pad_rec_samples, InPictureBorder,
        };

        if !htdf_applies(n_cb_w, n_cb_h, qp_y, is_intra) {
            return false;
        }
        let w = self.width as i32;
        let h = self.height as i32;
        let stride = self.y_stride();
        let lut = derive_htdf_lut(qp_y, is_inter_square_ge32);
        let border = InPictureBorder {
            x_cb: x as i32,
            y_cb: y as i32,
            width: w,
            height: h,
        };
        let pad = pad_rec_samples(
            x as i32,
            y as i32,
            n_cb_w,
            n_cb_h,
            |ax, ay| {
                // The §8.7.6.2 clamp keeps (ax, ay) in-extent; defensively
                // clamp to the plane bounds anyway.
                let cx = ax.clamp(0, w - 1) as usize;
                let cy = ay.clamp(0, h - 1) as usize;
                self.y[cy * stride + cx] as i32
            },
            &border,
        );
        let out = filter_block(&pad, n_cb_w, n_cb_h, &lut, self.bit_depth);
        // Write back, clamped to the picture extent.
        for j in 0..n_cb_h {
            let yy = y as usize + j;
            if yy >= self.height as usize {
                break;
            }
            for i in 0..n_cb_w {
                let xx = x as usize + i;
                if xx >= self.width as usize {
                    break;
                }
                self.y[yy * stride + xx] = out[j * n_cb_w + i] as u8;
            }
        }
        true
    }

    /// Clip to `[0, (1<<bit_depth) - 1]` and store into the right plane.
    pub fn store_block(
        &mut self,
        x: u32,
        y: u32,
        n_cb_w: usize,
        n_cb_h: usize,
        c_idx: u32,
        samples: &[i32],
    ) {
        let max_val = (1i32 << self.bit_depth) - 1;
        let (plane, stride, w, h) = self.plane_view_mut(c_idx);
        for j in 0..n_cb_h {
            let yy = y as usize + j;
            if yy >= h {
                break;
            }
            for i in 0..n_cb_w {
                let xx = x as usize + i;
                if xx >= w {
                    break;
                }
                let v = samples[j * n_cb_w + i].clamp(0, max_val) as u8;
                plane[yy * stride + xx] = v;
            }
        }
    }

    fn plane_view(&self, c_idx: u32) -> (&[u8], usize, usize, usize) {
        match c_idx {
            0 => (
                &self.y[..],
                self.y_stride(),
                self.width as usize,
                self.height as usize,
            ),
            1 => {
                let (cw, ch) =
                    chroma_dims(self.width, self.height, self.chroma_format_idc).unwrap_or((0, 0));
                (&self.cb[..], self.c_stride(), cw, ch)
            }
            2 => {
                let (cw, ch) =
                    chroma_dims(self.width, self.height, self.chroma_format_idc).unwrap_or((0, 0));
                (&self.cr[..], self.c_stride(), cw, ch)
            }
            _ => panic!("evc picture: invalid c_idx {c_idx}"),
        }
    }

    fn plane_view_mut(&mut self, c_idx: u32) -> (&mut [u8], usize, usize, usize) {
        let stride_c = self.c_stride();
        let (cw, ch) =
            chroma_dims(self.width, self.height, self.chroma_format_idc).unwrap_or((0, 0));
        match c_idx {
            0 => {
                let stride = self.width as usize;
                let w = self.width as usize;
                let h = self.height as usize;
                (&mut self.y[..], stride, w, h)
            }
            1 => (&mut self.cb[..], stride_c, cw, ch),
            2 => (&mut self.cr[..], stride_c, cw, ch),
            _ => panic!("evc picture: invalid c_idx {c_idx}"),
        }
    }
}

/// Compute chroma plane dimensions per chroma_format_idc.
fn chroma_dims(width: u32, height: u32, chroma_format_idc: u32) -> Result<(usize, usize)> {
    match chroma_format_idc {
        0 => Ok((0, 0)),
        // 4:2:0
        1 => Ok((width.div_ceil(2) as usize, height.div_ceil(2) as usize)),
        // 4:2:2
        2 => Ok((width.div_ceil(2) as usize, height as usize)),
        // 4:4:4
        3 => Ok((width as usize, height as usize)),
        n => Err(Error::invalid(format!(
            "evc picture: unsupported chroma_format_idc {n}"
        ))),
    }
}

/// Predict + reconstruct a CB at (x, y) given a residual sample array.
/// `pred_mode` selects the §8.4.4 intra mode. `c_idx` chooses the
/// component (0=Y, 1=Cb, 2=Cr). Sub-sampled chroma coordinates are
/// derived from the luma `(x, y)` for cIdx > 0 in 4:2:0 / 4:2:2.
#[allow(clippy::too_many_arguments)]
pub fn intra_reconstruct_cb(
    pic: &mut YuvPicture,
    x_luma: u32,
    y_luma: u32,
    log2_cb_w_luma: u32,
    log2_cb_h_luma: u32,
    pred_mode: IntraMode,
    c_idx: u32,
    residual: &[i32],
) -> Result<()> {
    // Compute component-space coordinates and dimensions.
    let (sub_w, sub_h) = sub_sampling(pic.chroma_format_idc, c_idx)?;
    let x = x_luma / sub_w;
    let y = y_luma / sub_h;
    let n_cb_w = 1usize << (log2_cb_w_luma - sub_w_log2(sub_w));
    let n_cb_h = 1usize << (log2_cb_h_luma - sub_h_log2(sub_h));
    if residual.len() != n_cb_w * n_cb_h {
        return Err(Error::invalid(format!(
            "evc reconstruct: residual len {} != {}*{}={}",
            residual.len(),
            n_cb_w,
            n_cb_h,
            n_cb_w * n_cb_h
        )));
    }
    let refs = pic.fetch_intra_refs(x, y, n_cb_w, n_cb_h, c_idx);
    let mut pred = vec![0i32; n_cb_w * n_cb_h];
    predict(pred_mode, &refs, n_cb_w, n_cb_h, pic.bit_depth, &mut pred);
    // Picture construction: rec = clip(pred + res) — eq. 1091.
    for (p, r) in pred.iter_mut().zip(residual.iter()) {
        *p += *r;
    }
    pic.store_block(x, y, n_cb_w, n_cb_h, c_idx, &pred);
    Ok(())
}

/// EIPD (`sps_eipd_flag == 1`) analogue of [`intra_reconstruct_cb`]: the
/// end-to-end §8.4.4 EIPD predict + §8.7.5 picture-construction path on
/// the picture buffer, the data-plane bridge that ties `fetch_eipd_refs`,
/// `eipd::predict_eipd` and `store_block` together.
///
/// `pred_mode_intra` is the EIPD mode index (0..32, Table 15).
/// `sps_suco_flag` / `right_available` thread the §6.4.2 right-neighbour
/// availability into the §8.4.4.1 construction. The `availLR` consumed by
/// the §8.4.4 kernels is derived from the simplified causal rule (left
/// available iff `x > 0` in-extent; right per `right_available`),
/// matching `fetch_eipd_refs`.
#[allow(clippy::too_many_arguments)]
pub fn intra_reconstruct_cb_eipd(
    pic: &mut YuvPicture,
    x_luma: u32,
    y_luma: u32,
    log2_cb_w_luma: u32,
    log2_cb_h_luma: u32,
    pred_mode_intra: i32,
    c_idx: u32,
    sps_suco_flag: bool,
    right_available: bool,
    residual: &[i32],
) -> Result<()> {
    use crate::eipd::{predict_eipd, AvailLr};

    let (sub_w, sub_h) = sub_sampling(pic.chroma_format_idc, c_idx)?;
    let x = x_luma / sub_w;
    let y = y_luma / sub_h;
    let n_cb_w = 1usize << (log2_cb_w_luma - sub_w_log2(sub_w));
    let n_cb_h = 1usize << (log2_cb_h_luma - sub_h_log2(sub_h));
    if residual.len() != n_cb_w * n_cb_h {
        return Err(Error::invalid(format!(
            "evc reconstruct (eipd): residual len {} != {}*{}={}",
            residual.len(),
            n_cb_w,
            n_cb_h,
            n_cb_w * n_cb_h
        )));
    }

    // §6.4.2 availLR from the simplified causal rule.
    let left_avail = x > 0;
    let avail_lr = match (left_avail, sps_suco_flag && right_available) {
        (false, false) => AvailLr::Lr00,
        (true, false) => AvailLr::Lr10,
        (false, true) => AvailLr::Lr01,
        (true, true) => AvailLr::Lr11,
    };

    let refs = pic.fetch_eipd_refs(x, y, n_cb_w, n_cb_h, c_idx, sps_suco_flag, right_available);
    let mut pred = vec![0i32; n_cb_w * n_cb_h];
    predict_eipd(
        pred_mode_intra,
        &refs,
        n_cb_w,
        n_cb_h,
        pic.bit_depth,
        avail_lr,
        &mut pred,
    );
    // §8.7.5 picture construction: rec = clip(pred + res) — eq. 1091.
    for (p, r) in pred.iter_mut().zip(residual.iter()) {
        *p += *r;
    }
    pic.store_block(x, y, n_cb_w, n_cb_h, c_idx, &pred);
    Ok(())
}

fn sub_sampling(chroma_format_idc: u32, c_idx: u32) -> Result<(u32, u32)> {
    if c_idx == 0 {
        return Ok((1, 1));
    }
    match chroma_format_idc {
        1 => Ok((2, 2)),
        2 => Ok((2, 1)),
        3 => Ok((1, 1)),
        n => Err(Error::invalid(format!(
            "evc reconstruct: unsupported chroma_format_idc {n}"
        ))),
    }
}

fn sub_w_log2(s: u32) -> u32 {
    s.trailing_zeros()
}

fn sub_h_log2(s: u32) -> u32 {
    s.trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Newly-constructed buffer is initialised to 128 (8-bit half-range).
    #[test]
    fn fresh_picture_is_grey() {
        let pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        assert!(pic.y.iter().all(|&b| b == 128));
        assert!(pic.cb.iter().all(|&b| b == 128));
        assert!(pic.cr.iter().all(|&b| b == 128));
    }

    /// Reject 10-bit pictures (round 3 is 8-bit only).
    #[test]
    fn rejects_10bit() {
        let err = YuvPicture::new(16, 16, 1, 10).unwrap_err();
        assert!(format!("{err}").contains("8-bit only"));
    }

    /// Reject zero dimensions.
    #[test]
    fn rejects_zero_dim() {
        let err = YuvPicture::new(0, 16, 1, 8).unwrap_err();
        assert!(format!("{err}").contains("zero"));
    }

    /// Chroma plane sizes for 4:2:0 / 4:2:2 / 4:4:4.
    #[test]
    fn chroma_dims_for_layouts() {
        assert_eq!(chroma_dims(16, 16, 1).unwrap(), (8, 8));
        assert_eq!(chroma_dims(16, 16, 2).unwrap(), (8, 16));
        assert_eq!(chroma_dims(16, 16, 3).unwrap(), (16, 16));
        assert_eq!(chroma_dims(16, 16, 0).unwrap(), (0, 0));
    }

    /// First CU at (0, 0) gets all-128 reference samples (matches the
    /// not-available substitution rule).
    #[test]
    fn first_cu_refs_are_grey() {
        let pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let refs = pic.fetch_intra_refs(0, 0, 8, 8, 0);
        assert_eq!(refs.top_left, 128);
        assert!(refs.top.iter().all(|&v| v == 128));
        assert!(refs.left.iter().all(|&v| v == 128));
    }

    /// Storing a block then reading reference samples around its
    /// neighbouring position recovers the stored values.
    #[test]
    fn store_then_fetch_neighbour() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let block: Vec<i32> = (0..64i32).map(|i| 100 + (i & 0x1F)).collect();
        pic.store_block(0, 0, 8, 8, 0, &block);
        // Now a CB at (8, 0) sees the right-most luma column from the
        // stored block as its left reference.
        let refs = pic.fetch_intra_refs(8, 0, 4, 4, 0);
        // refs.left[j] = block[j*8 + 7] for j in 0..4 (since the stored
        // block is at x=0..7).
        for j in 0..4 {
            let expect = block[j * 8 + 7].clamp(0, 255);
            assert_eq!(refs.left[j], expect);
        }
        // Top is still y=0 → no row above, stays at 128.
        assert!(refs.top.iter().all(|&v| v == 128));
    }

    /// First CU EIPD refs: everything not-available → §8.4.4.2 EIPD
    /// substitution. The corner is mid-level (128) and the copy-predecessor
    /// chain propagates that 128 across the whole top row and left column.
    #[test]
    fn eipd_first_cu_is_grey() {
        let pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let refs = pic.fetch_eipd_refs(0, 0, 4, 4, 0, false, false);
        assert_eq!(refs.top_left(), 128);
        for i in -1..8i32 {
            assert_eq!(refs.top(i), 128, "top[{i}]");
            assert_eq!(refs.left(i), 128, "left[{i}]");
        }
    }

    /// EIPD refs pick up the stored left-neighbour column; the still-grey
    /// top stays 128, and the EIPD copy-predecessor fill leaves available
    /// left samples untouched.
    #[test]
    fn eipd_left_neighbour_carried_through() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let block: Vec<i32> = (0..64i32).map(|i| 100 + (i & 0x1F)).collect();
        pic.store_block(0, 0, 8, 8, 0, &block);
        // Block at (8, 0): left column = right column of the stored block.
        let refs = pic.fetch_eipd_refs(8, 0, 4, 4, 0, false, false);
        // left[j] = block[j*8 + 7] for j in 0..4 (in-picture, x>0 ⇒ avail).
        for j in 0..4i32 {
            let expect = block[(j as usize) * 8 + 7].clamp(0, 255);
            assert_eq!(refs.left(j), expect, "left[{j}]");
        }
        // Top row unavailable (y == 0) → EIPD corner-chain → all 128.
        for i in 0..8i32 {
            assert_eq!(refs.top(i), 128, "top[{i}]");
        }
    }

    /// End-to-end EIPD reconstruct: a first CU (all-grey EIPD refs) with
    /// INTRA_DC + zero residual reconstructs to the mid-level constant.
    #[test]
    fn eipd_reconstruct_first_cu_dc() {
        use crate::eipd::INTRA_DC;
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        let zero = vec![0i32; 64];
        intra_reconstruct_cb_eipd(
            &mut pic, 0, 0, 3, 3, // 8×8
            INTRA_DC, 0, false, false, &zero,
        )
        .unwrap();
        // All refs are 128 → DC average is 128.
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(pic.y[j * 16 + i], 128, "({i},{j})");
            }
        }
    }

    /// End-to-end EIPD reconstruct: INTRA_VER copies the (grey) top row,
    /// then a non-zero residual offsets it; the result clips into range.
    #[test]
    fn eipd_reconstruct_ver_with_residual() {
        use crate::eipd::INTRA_VER;
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        // Seed the row above an 8×8 block at (0, 8) with a ramp.
        for i in 0..8usize {
            pic.y[7 * 16 + i] = (10 + i * 5) as u8;
        }
        let res = vec![3i32; 64];
        intra_reconstruct_cb_eipd(&mut pic, 0, 8, 3, 3, INTRA_VER, 0, false, false, &res).unwrap();
        // INTRA_VER copies p[x][-1] (the ramp) + residual 3.
        for x in 0..8usize {
            let expect = (10 + x * 5) as i32 + 3;
            assert_eq!(pic.y[8 * 16 + x] as i32, expect, "x={x}");
        }
    }

    /// HTDF is a no-op when the §8.7.6.1 applicability gates fail (e.g. a
    /// 4×4 block, < 64 samples) — the plane is untouched and the call
    /// returns false.
    #[test]
    fn htdf_skipped_when_inapplicable() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 123;
        }
        let applied = pic.apply_htdf_luma(0, 0, 4, 4, 30, true, false);
        assert!(!applied);
        assert!(pic.y.iter().all(|&v| v == 123));
    }

    /// HTDF on a flat luma field is the identity (all AC coefficients are
    /// zero), but the call returns true (it ran).
    #[test]
    fn htdf_flat_field_identity() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        let applied = pic.apply_htdf_luma(0, 0, 8, 8, 30, true, false);
        assert!(applied);
        // 8×8 block at (0,0) unchanged on a flat field.
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(pic.y[j * 16 + i], 100, "({i},{j})");
            }
        }
    }

    /// HTDF smooths a single-sample luma impulse without amplifying it.
    #[test]
    fn htdf_smooths_impulse() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        for v in pic.y.iter_mut() {
            *v = 100;
        }
        pic.y[4 * 16 + 4] = 220;
        let applied = pic.apply_htdf_luma(0, 0, 8, 8, 40, true, false);
        assert!(applied);
        let peak = pic.y[4 * 16 + 4];
        assert!(peak <= 220, "peak={peak}");
        // Filtered samples stay within the input value range.
        for j in 0..8 {
            for i in 0..8 {
                let v = pic.y[j * 16 + i];
                assert!((90..=220).contains(&v), "({i},{j})={v}");
            }
        }
    }

    /// SUCO path: when `right_available` is set and the right column is
    /// in-picture and reconstructed, `fetch_eipd_refs` populates the right
    /// column from the picture instead of substituting.
    #[test]
    fn eipd_suco_right_column_populated() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        // Seed a column at x = 4 (the right neighbour of a block at x=0).
        for yy in 0..8usize {
            pic.y[yy * 16 + 4] = 77;
        }
        // Also seed top row so the block isn't first-row (give it a row
        // above at y = 4..).
        let refs = pic.fetch_eipd_refs(0, 4, 4, 4, 0, true, true);
        // p[nCbW=4][y] for y in 0..nCbH+nCbW-1, in-picture rows carry 77.
        for y in 0..4i32 {
            assert_eq!(refs.right(y), 77, "right[{y}]");
        }
    }

    /// Storing a block then intra-reconstructing with INTRA_DC + zero
    /// residual at an adjacent position picks up the average of the
    /// stored block's right column (left refs) and the unavailable top
    /// row (=128). This sanity-checks the reconstruct loop.
    #[test]
    fn intra_reconstruct_dc_with_neighbour() {
        let mut pic = YuvPicture::new(16, 16, 1, 8).unwrap();
        // Fill a 4x4 block with 200s to seed the left neighbour.
        let block = vec![200i32; 16];
        pic.store_block(0, 0, 4, 4, 0, &block);
        // Reconstruct a 4x4 DC at (4, 0) with zero residual. Left refs =
        // 200, top refs = 128. DC = (200*4 + 128*4 + 4) >> 3 = (800 +
        // 512 + 4) >> 3 = 164.
        let zero_res = vec![0i32; 16];
        intra_reconstruct_cb(
            &mut pic,
            4,
            0,
            2, // log2 4
            2,
            IntraMode::Dc,
            0,
            &zero_res,
        )
        .unwrap();
        // Sample at (4, 0) must equal 164.
        assert_eq!(pic.y[4], 164);
    }
}
