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
