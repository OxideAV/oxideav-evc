//! Registry glue for the EVC foundation crate.
//!
//! Round-1 ships parser-only; the registered decoder factory returns a
//! placeholder that rejects every packet with `Error::Unsupported`. The
//! useful surface lives in [`crate::nal`] / [`crate::sps`] / [`crate::pps`]
//! / [`crate::aps`] / [`crate::slice_header`] and the [`crate::probe`]
//! convenience helper on the crate root.

use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::CODEC_ID_STR;

/// Build the placeholder foundation decoder for the registry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(FoundationDecoder {
        codec_id: CodecId::new(CODEC_ID_STR),
    }))
}

pub struct FoundationDecoder {
    codec_id: CodecId,
}

impl Decoder for FoundationDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        Err(Error::unsupported("EVC pixel decode not yet implemented"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::Eof)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
