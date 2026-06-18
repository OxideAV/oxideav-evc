//! EVC Main-profile **EIPD luma intra-mode derivation** (ISO/IEC
//! 23094-1:2020 §8.4.2, `sps_eipd_flag == 1` path, eqs. 172-278).
//!
//! When `sps_eipd_flag == 1` the luma intra prediction mode
//! `IntraPredModeY[ xCb ][ yCb ]` is not read directly: it is selected
//! out of one of three ranked lists built from the neighbouring modes:
//!
//! * `candModeList[ 0..1 ]` — the two most-probable modes (eqs. 172-175),
//! * `extCandModeList[ 0..7 ]` — eight secondary modes (eqs. 176-278),
//! * `remModeList[ 10..32 ]` — the remaining modes filled from the
//!   §8.4.2 `defaultModeList[ 33 ]`.
//!
//! The signalled syntax elements then index:
//!
//! * `intra_luma_pred_mpm_flag == 1` → `candModeList[ mpm_idx ]`,
//! * else `intra_luma_pred_pims_flag == 1` → `extCandModeList[ pims_idx ]`,
//! * else → `remModeList[ rem_mode + 2 + 8 ]`.
//!
//! This module builds all three lists from the neighbour-derived
//! candidate modes and performs the final selection. The CABAC syntax
//! reads (`mpm_flag` / `mpm_idx` / `pims_flag` / `pims_idx` / `rem_mode`)
//! and the §6.4.1 neighbour-availability lookups that feed
//! `candIntraPredModeA/B/C` are the caller's responsibility.
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use crate::eipd::{INTRA_BI, INTRA_DC, INTRA_DIA_L, INTRA_DIA_R, INTRA_DIA_U, INTRA_HOR};
use crate::eipd::{INTRA_PLN, INTRA_VER};

/// Per-neighbour candidate intra mode (§8.4.2 step 2). `valid` records
/// whether the neighbour was available + intra-coded; when not, the spec
/// substitutes `candIntraPredMode = INTRA_DC` and `valid = false`.
#[derive(Clone, Copy, Debug)]
pub struct NeighbourMode {
    pub valid: bool,
    pub mode: i32,
}

impl NeighbourMode {
    /// A valid neighbour carrying `mode`.
    pub fn valid(mode: i32) -> Self {
        Self { valid: true, mode }
    }
    /// An unavailable / non-intra neighbour (§8.4.2: mode = INTRA_DC).
    pub fn invalid() -> Self {
        Self {
            valid: false,
            mode: INTRA_DC,
        }
    }
}

/// Which signalled list the final mode comes from (§8.4.2 step 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeSelector {
    /// `intra_luma_pred_mpm_flag == 1` → `candModeList[ idx ]` (idx 0/1).
    Mpm(usize),
    /// `pims_flag == 1` → `extCandModeList[ idx ]` (idx 0..7).
    Pims(usize),
    /// else → `remModeList[ rem_mode + 10 ]`.
    Rem(usize),
}

/// §8.4.3 — derive the chroma intra prediction mode `IntraPredModeC`
/// from the signalled `intra_chroma_pred_mode` and the co-located luma
/// mode `IntraPredModeY`.
///
/// `intra_chroma_pred_mode == 0` reuses the luma mode (DM); the
/// remaining values index a small set offset around any of
/// `{INTRA_DC, INTRA_HOR, INTRA_VER, INTRA_BI}` that the luma mode might
/// already occupy (Table 16 + the `modeIdx` skip rule). For `MODE_IBC`
/// blocks the spec first forces `IntraPredModeY := INTRA_DC`, so callers
/// pass that in.
pub fn derive_chroma_mode(intra_chroma_pred_mode: i32, intra_pred_mode_y: i32) -> i32 {
    if intra_chroma_pred_mode == 0 {
        return intra_pred_mode_y;
    }
    // The luma mode collides with one of {BI, DC, HOR, VER} → skip-index.
    if intra_pred_mode_y == INTRA_DC
        || intra_pred_mode_y == INTRA_HOR
        || intra_pred_mode_y == INTRA_VER
        || intra_pred_mode_y == INTRA_BI
    {
        let mode_idx = if intra_pred_mode_y == INTRA_BI {
            1
        } else if intra_pred_mode_y == INTRA_DC {
            2
        } else if intra_pred_mode_y == INTRA_HOR {
            3
        } else {
            // INTRA_VER
            4
        };
        if intra_chroma_pred_mode >= mode_idx {
            chroma_table_16(intra_chroma_pred_mode + 1)
        } else {
            chroma_table_16(intra_chroma_pred_mode)
        }
    } else {
        chroma_table_16(intra_chroma_pred_mode)
    }
}

/// Table 16 — map a (possibly +1-skipped) chroma mode index to the
/// concrete `IntraPredModeC`. Index 0 (DM / luma reuse) is handled by
/// the caller; here only 1..4 are meaningful.
fn chroma_table_16(idx: i32) -> i32 {
    match idx {
        1 => INTRA_BI,
        2 => INTRA_DC,
        3 => INTRA_HOR,
        4 => INTRA_VER,
        // Out of the Table 16 range (should not occur after the
        // mode_idx skip); clamp to VER, the last entry.
        _ => INTRA_VER,
    }
}

/// The fully-derived EIPD mode lists for one luma block.
#[derive(Clone, Debug)]
pub struct EipdModeLists {
    pub cand_mode_list: [i32; 2],
    pub ext_cand_mode_list: [i32; 8],
    pub rem_mode_list: [i32; 33],
}

impl EipdModeLists {
    /// §8.4.2 step 6 — select the final `IntraPredModeY` from the three
    /// lists given the signalled selector.
    pub fn select(&self, sel: ModeSelector) -> i32 {
        match sel {
            ModeSelector::Mpm(i) => self.cand_mode_list[i],
            ModeSelector::Pims(i) => self.ext_cand_mode_list[i],
            // remModeList[ rem_mode + 2 + 8 ]
            ModeSelector::Rem(rem) => self.rem_mode_list[rem + 10],
        }
    }
}

/// Resolve the three neighbour candidates A/B/C into the validity-pruned
/// `(candIntraPredModeA, candIntraPredModeB, candIntraPredModeC, validC)`
/// per §8.4.2 step 2 ("When validC is equal to TRUE …").
fn resolve_candidates(
    a: NeighbourMode,
    b: NeighbourMode,
    c: NeighbourMode,
) -> (i32, i32, i32, bool) {
    let mut cand_a = a.mode;
    let mut cand_b = b.mode;
    let cand_c = c.mode;
    let mut valid_c = c.valid;

    if valid_c {
        if a.valid && b.valid {
            if cand_a == cand_b {
                cand_b = cand_c;
                valid_c = false;
            } else if cand_a == cand_c || cand_b == cand_c {
                valid_c = false;
            }
        } else if !a.valid {
            cand_a = cand_c;
            valid_c = false;
        } else if !b.valid {
            cand_b = cand_c;
            valid_c = false;
        }
    }
    (cand_a, cand_b, cand_c, valid_c)
}

/// §8.4.2 steps 3-6 — build `candModeList`, `extCandModeList`,
/// `remModeList` from the (already neighbour-derived) A/B/C candidates.
pub fn derive_mode_lists(a: NeighbourMode, b: NeighbourMode, c: NeighbourMode) -> EipdModeLists {
    let (cand_a, cand_b, cand_intra_pred_mode_c, valid_c) = resolve_candidates(a, b, c);

    // Step 3: candModeList (eqs. 172-175).
    let mut cand0 = cand_a.min(cand_b);
    let mut cand1 = cand_a.max(cand_b);
    if cand1 == cand0 {
        cand0 = INTRA_DC;
        cand1 = if cand1 == INTRA_DC { INTRA_BI } else { cand1 };
    }
    let cand_mode_list = [cand0, cand1];

    // Step 4: extCandModeList.
    let ext_cand_mode_list = if !valid_c {
        derive_ext_no_c(cand0, cand1)
    } else {
        derive_ext_with_c(cand0, cand1, cand_intra_pred_mode_c)
    };

    // Step 5: remModeList (eqs. after 278).
    let rem_mode_list = derive_rem_list(cand_mode_list, ext_cand_mode_list);

    EipdModeLists {
        cand_mode_list,
        ext_cand_mode_list,
        rem_mode_list,
    }
}

/// Formulae 176-182 — the eight "anchor" secondary modes used when both
/// MPMs are planar/DC/BI.
fn anchor_176_182() -> [i32; 8] {
    [
        0, // [0] filled by the caller (the "missing among PLN/DC/BI" mode)
        INTRA_VER,
        INTRA_HOR,
        INTRA_DIA_R,
        INTRA_DIA_L,
        INTRA_DIA_U,
        INTRA_VER + 4,
        INTRA_HOR - 4,
    ]
}

/// The mode among `{INTRA_PLN, INTRA_DC, INTRA_BI}` not present in the
/// candidate list (eq. 176 / 207 "not included in candModeList").
fn missing_planar_dc_bi(cand0: i32, cand1: i32) -> i32 {
    for m in [INTRA_PLN, INTRA_DC, INTRA_BI] {
        if m != cand0 && m != cand1 {
            return m;
        }
    }
    INTRA_BI
}

/// `extCandModeList` derivation for the `validC == FALSE` path
/// (eqs. 176-223).
fn derive_ext_no_c(cand0: i32, cand1: i32) -> [i32; 8] {
    if cand0 < 3 && cand1 < 3 {
        // Both MPMs are planar/DC/BI: anchor list (eqs. 176-182).
        let mut ext = anchor_176_182();
        ext[0] = missing_planar_dc_bi(cand0, cand1);
        ext
    } else if cand0 < 3 && cand1 >= 3 {
        // One planar, one directional (eqs. 183-206).
        let mut ext = [0i32; 8];
        if cand0 == INTRA_PLN {
            ext[0] = INTRA_BI;
            ext[1] = INTRA_DC;
        } else {
            ext[0] = if cand0 == INTRA_BI {
                INTRA_DC
            } else {
                INTRA_BI
            };
            ext[1] = INTRA_PLN;
        }
        fill_ext_2_7_directional(&mut ext, cand1);
        ext
    } else {
        // Both directional (eqs. 207-223).
        let mut ext = [0i32; 8];
        ext[0] = INTRA_BI;
        ext[1] = INTRA_DC;
        let list = list_209_223(cand0, cand1);
        dedup_fill(&mut ext, 2, &list, cand0, cand1);
        ext
    }
}

/// eqs. 187-206 — fill `extCandModeList[ 2..7 ]` around a single
/// directional MPM `cand1` (the `candModeList[ 1 ] >= 3` case).
fn fill_ext_2_7_directional(ext: &mut [i32; 8], cand1: i32) {
    if cand1 > 30 {
        ext[2] = if cand1 == 32 { 31 } else { 32 };
        ext[3] = 30;
        ext[4] = 29;
        ext[5] = 28;
        ext[6] = INTRA_HOR;
        ext[7] = INTRA_DIA_R;
    } else if cand1 < 5 {
        ext[2] = if cand1 == 3 { 4 } else { 3 };
        ext[3] = 5;
        ext[4] = 6;
        ext[5] = 7;
        ext[6] = INTRA_VER;
        ext[7] = INTRA_DIA_R;
    } else {
        ext[2] = cand1 + 2;
        ext[3] = cand1 - 2;
        ext[4] = cand1 + 1;
        ext[5] = cand1 - 1;
        if (13..=23).contains(&cand1) {
            ext[6] = cand1 - 5;
            ext[7] = cand1 + 5;
        } else {
            ext[6] = if cand1 > 23 { cand1 - 5 } else { cand1 + 5 };
            ext[7] = if cand1 > 23 { cand1 - 10 } else { cand1 + 10 };
        }
    }
}

/// `list[ 0..14 ]` of eqs. 209-223 (both-directional, validC == FALSE).
/// list[5]/list[6] (eqs. 214-215) depend on list[4] and are resolved
/// inline.
fn list_209_223(cand0: i32, cand1: i32) -> Vec<i32> {
    let l4 = (cand0 + cand1 + 1) >> 1; // 213
    vec![
        if cand0 == 3 || cand0 == 4 {
            cand0 + 1
        } else {
            cand0 - 2
        }, // 209
        if cand0 == 31 { cand0 - 1 } else { cand0 + 2 }, // 210
        if cand1 == 4 { cand1 + 1 } else { cand1 - 2 },  // 211
        if cand1 == 32 || cand1 == 31 {
            cand1 - 1
        } else {
            cand1 + 2
        }, // 212
        l4,                                              // 213
        (l4 + cand0 + 1) >> 1,                           // 214
        (l4 + cand1 + 1) >> 1,                           // 215
        INTRA_VER,                                       // 216
        INTRA_HOR,                                       // 217
        INTRA_DIA_R,                                     // 218
        INTRA_PLN,                                       // 219
        INTRA_DIA_L,                                     // 220
        INTRA_DIA_U,                                     // 221
        INTRA_VER + 4,                                   // 222
        INTRA_HOR - 4,                                   // 223
    ]
}

/// `extCandModeList` derivation for the `validC == TRUE` path
/// (eqs. 224-278).
fn derive_ext_with_c(cand0: i32, cand1: i32, cand_c: i32) -> [i32; 8] {
    if cand0 < 3 && cand1 < 3 {
        // eqs. (224-236).
        let mut ext = [0i32; 8];
        ext[0] = missing_planar_dc_bi(cand0, cand1);
        if cand_c < 3 {
            // Formulae 176-182.
            let anchor = anchor_176_182();
            ext[1..8].copy_from_slice(&anchor[1..8]);
        } else {
            ext[1] = cand_c;
            ext[2] = if cand_c == 3 || cand_c == 4 {
                cand_c + 1
            } else {
                cand_c - 2
            };
            ext[3] = if cand_c == 32 || cand_c == 31 {
                cand_c - 1
            } else {
                cand_c + 2
            };
            let list = vec![
                INTRA_VER,
                INTRA_HOR,
                INTRA_DIA_R,
                INTRA_PLN,
                INTRA_DIA_L,
                INTRA_DIA_U,
                INTRA_VER + 4,
                INTRA_HOR - 4,
                INTRA_VER - 4,
                INTRA_HOR + 4,
            ];
            dedup_fill(&mut ext, 4, &list, cand0, cand1);
        }
        ext
    } else if cand0 < 3 && cand1 >= 3 {
        // eqs. (237-257).
        if cand_c < 3 {
            // Formulae 183-206.
            let mut ext = [0i32; 8];
            if cand0 == INTRA_PLN {
                ext[0] = INTRA_BI;
                ext[1] = INTRA_DC;
            } else {
                ext[0] = if cand0 == INTRA_BI {
                    INTRA_DC
                } else {
                    INTRA_BI
                };
                ext[1] = INTRA_PLN;
            }
            fill_ext_2_7_directional(&mut ext, cand1);
            ext
        } else {
            let mut ext = [0i32; 8];
            if cand0 == INTRA_PLN {
                ext[0] = INTRA_BI;
                ext[1] = INTRA_DC;
                ext[2] = cand_c;
            } else {
                ext[0] = if cand0 == INTRA_BI {
                    INTRA_DC
                } else {
                    INTRA_BI
                };
                ext[1] = INTRA_PLN;
                ext[2] = cand_c;
            }
            let mut list = vec![
                if cand_c == 3 || cand_c == 4 {
                    cand_c + 1
                } else {
                    cand_c - 2
                }, // 243
                if cand_c == 32 || cand_c == 31 {
                    cand_c - 1
                } else {
                    cand_c + 2
                }, // 244
                if cand1 == 3 || cand1 == 4 {
                    cand1 + 1
                } else {
                    cand1 - 2
                }, // 245
                if cand1 == 32 || cand1 == 31 {
                    cand1 - 1
                } else {
                    cand1 + 2
                }, // 246
                (cand_c + cand1 + 1) >> 1, // 247
                0,                         // 248 placeholder
                0,                         // 249 placeholder
                INTRA_VER,
                INTRA_HOR,
                INTRA_DIA_R,
                INTRA_PLN,
                INTRA_DIA_L,
                INTRA_DIA_U,
                INTRA_VER + 4,
                INTRA_HOR - 4,
            ];
            let l4 = list[4];
            list[5] = (cand_c + l4 + 1) >> 1; // 248
            list[6] = (cand1 + l4 + 1) >> 1; // 249
            dedup_fill(&mut ext, 3, &list, cand0, cand1);
            ext
        }
    } else {
        // Both MPMs directional, validC == TRUE (eqs. 258-278).
        if cand_c < 3 {
            // eqs. (258-259) + list 209-223 from iCount=2.
            let mut ext = [0i32; 8];
            ext[0] = cand_c;
            ext[1] = if cand_c == INTRA_BI {
                INTRA_DC
            } else {
                INTRA_BI
            };
            let list = list_209_223(cand0, cand1);
            dedup_fill(&mut ext, 2, &list, cand0, cand1);
            ext
        } else {
            // eqs. (260-278).
            let mut ext = [0i32; 8];
            ext[0] = INTRA_BI;
            ext[1] = INTRA_DC;
            ext[2] = cand_c;
            let list = list_263_278(cand0, cand1, cand_c);
            dedup_fill(&mut ext, 3, &list, cand0, cand1);
            ext
        }
    }
}

/// `list[ 0..15 ]` of eqs. 263-278 (both-directional, validC == TRUE,
/// candC >= 3). `list[ 6 ]` / `list[ 7 ]` are resolved inline.
///
/// Note: eq. 266 as printed reads `candModeList[ 0 ] - 1 / + 2` inside a
/// branch keyed on `candModeList[ 1 ]`; this is transcribed verbatim as
/// the literal spec wording (a probable axis typo, flagged for the
/// errata channel — see report). eq. 269's `condIntraPredModeC` is read
/// as `candIntraPredModeC`.
fn list_263_278(cand0: i32, cand1: i32, cand_c: i32) -> Vec<i32> {
    let l6 = if cand_c < cand1 {
        (cand0 + cand_c + 1) >> 1
    } else {
        (cand0 + cand1 + 1) >> 1
    };
    let l7 = if cand_c < cand0 {
        (cand0 + cand1 + 1) >> 1
    } else {
        (cand_c + cand1 + 1) >> 1
    };
    vec![
        if cand0 == 3 || cand0 == 4 {
            cand0 + 1
        } else {
            cand0 - 2
        }, // 263
        if cand0 == 31 { cand0 - 1 } else { cand0 + 2 }, // 264
        if cand1 == 4 { cand1 + 1 } else { cand1 - 2 },  // 265
        if cand1 == 32 || cand1 == 31 {
            cand0 - 1
        } else {
            cand0 + 2
        }, // 266 (verbatim cand0 axis)
        if cand_c == 3 || cand_c == 4 {
            cand_c + 1
        } else {
            cand_c - 2
        }, // 267
        if cand_c == 32 || cand_c == 31 {
            cand_c - 1
        } else {
            cand_c + 2
        }, // 268
        l6,                                              // 269
        l7,                                              // 270
        INTRA_VER,                                       // 271
        INTRA_HOR,                                       // 272
        INTRA_DIA_R,
        INTRA_PLN,
        INTRA_DIA_L,
        INTRA_DIA_U,
        INTRA_VER + 4,
        INTRA_HOR - 4,
    ]
}

/// The §8.4.2 dedup-fill loop: starting at `ext[ start ]` (iCount =
/// start), walk `list` and append each entry that is not already present
/// in `ext[ 0..iCount ]`, `cand0`, or `cand1`, until eight modes are
/// gathered. (Equivalent to the nested `i`/`j` loop in the spec.)
fn dedup_fill(ext: &mut [i32; 8], start: usize, list: &[i32], cand0: i32, cand1: i32) {
    // Resolve any list-internal back-references (eqs. 214-215 / 248-249)
    // already baked into `list` by the caller; here we only dedup.
    let mut count = start;
    for &m in list {
        if count > 7 {
            break;
        }
        let present = ext[..count].contains(&m) || m == cand0 || m == cand1;
        if !present {
            ext[count] = m;
            count += 1;
        }
    }
    // If the list was exhausted before filling all 8 (should not happen
    // for spec-conformant inputs), leave the remainder at 0.
}

/// The fixed `defaultModeList[ 33 ]` of §8.4.2 step 5.
fn default_mode_list() -> [i32; 33] {
    [
        INTRA_DC,
        INTRA_BI,
        INTRA_VER,
        INTRA_PLN,
        INTRA_HOR,
        INTRA_VER - 1,
        INTRA_VER + 1,
        INTRA_VER - 2,
        INTRA_VER + 2,
        INTRA_VER - 3,
        INTRA_VER + 3,
        INTRA_HOR - 1,
        INTRA_HOR + 1,
        INTRA_HOR - 2,
        INTRA_HOR + 2,
        INTRA_HOR - 3,
        INTRA_HOR + 3,
        INTRA_VER + 5,
        INTRA_VER + 4,
        INTRA_VER - 5,
        INTRA_VER - 4,
        INTRA_DIA_R,
        INTRA_DIA_L,
        INTRA_DIA_L - 3,
        INTRA_DIA_L - 2,
        INTRA_DIA_L - 1,
        INTRA_DIA_U,
        INTRA_DIA_U + 1,
        INTRA_DIA_U + 2,
        INTRA_HOR - 4,
        INTRA_HOR - 5,
        INTRA_HOR + 5,
        INTRA_HOR + 4,
    ]
}

/// §8.4.2 step 5 — build `remModeList[ 0..32 ]`:
/// `[ candModeList[0..1], extCandModeList[0..7], <defaults not already used> ]`.
fn derive_rem_list(cand_mode_list: [i32; 2], ext: [i32; 8]) -> [i32; 33] {
    let mut rem = [0i32; 33];
    rem[0] = cand_mode_list[0];
    rem[1] = cand_mode_list[1];
    rem[2..10].copy_from_slice(&ext);

    let defaults = default_mode_list();
    let mut count = 10usize;
    for &m in &defaults {
        if count > 32 {
            break;
        }
        let used = m == cand_mode_list[0] || m == cand_mode_list[1] || ext.contains(&m);
        if !used {
            rem[count] = m;
            count += 1;
        }
    }
    rem
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When all three neighbours are unavailable, both MPMs collapse to
    /// {INTRA_DC, INTRA_BI} (eqs. 174-175) and the anchor secondary list
    /// applies.
    #[test]
    fn all_invalid_neighbours() {
        let lists = derive_mode_lists(
            NeighbourMode::invalid(),
            NeighbourMode::invalid(),
            NeighbourMode::invalid(),
        );
        assert_eq!(lists.cand_mode_list, [INTRA_DC, INTRA_BI]);
        // ext[0] is the missing planar/dc/bi → INTRA_PLN.
        assert_eq!(lists.ext_cand_mode_list[0], INTRA_PLN);
        assert_eq!(lists.ext_cand_mode_list[1], INTRA_VER);
    }

    /// Every derived list must contain distinct modes (the dedup loop is
    /// the whole point), and `remModeList[10..]` must avoid everything in
    /// `candModeList` / `extCandModeList`.
    #[test]
    fn lists_are_distinct() {
        // a directional + planar mix to exercise the busiest branch.
        let lists = derive_mode_lists(
            NeighbourMode::valid(18),
            NeighbourMode::valid(INTRA_PLN),
            NeighbourMode::valid(6),
        );
        // extCandModeList entries distinct from candModeList.
        for &e in &lists.ext_cand_mode_list {
            assert!(
                e != lists.cand_mode_list[0] && e != lists.cand_mode_list[1],
                "ext mode {e} duplicates an MPM"
            );
        }
        // remModeList[10..] distinct from the first 10 list entries.
        let first10: Vec<i32> = lists.rem_mode_list[..10].to_vec();
        for &r in &lists.rem_mode_list[10..] {
            assert!(!first10.contains(&r), "rem mode {r} duplicates a primary");
        }
    }

    /// The full 33-entry `remModeList` must be a permutation covering all
    /// modes 0..32 exactly once (defaults are the complete mode set).
    #[test]
    fn rem_list_covers_all_modes() {
        let lists = derive_mode_lists(
            NeighbourMode::valid(12),
            NeighbourMode::valid(24),
            NeighbourMode::invalid(),
        );
        let mut seen = [false; 33];
        for &m in &lists.rem_mode_list {
            assert!((0..=32).contains(&m), "mode {m} out of range");
            assert!(!seen[m as usize], "mode {m} appears twice in remModeList");
            seen[m as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "remModeList missing a mode");
    }

    /// Selector dispatch picks the right list entry.
    #[test]
    fn selector_dispatch() {
        let lists = derive_mode_lists(
            NeighbourMode::valid(12),
            NeighbourMode::valid(24),
            NeighbourMode::invalid(),
        );
        assert_eq!(lists.select(ModeSelector::Mpm(0)), lists.cand_mode_list[0]);
        assert_eq!(lists.select(ModeSelector::Mpm(1)), lists.cand_mode_list[1]);
        assert_eq!(
            lists.select(ModeSelector::Pims(3)),
            lists.ext_cand_mode_list[3]
        );
        assert_eq!(lists.select(ModeSelector::Rem(0)), lists.rem_mode_list[10]);
    }

    /// §8.4.3 chroma DM: intra_chroma_pred_mode == 0 reuses the luma
    /// mode regardless of what it is.
    #[test]
    fn chroma_dm_reuses_luma() {
        for luma in [INTRA_DC, INTRA_PLN, 18, 30, INTRA_HOR] {
            assert_eq!(derive_chroma_mode(0, luma), luma);
        }
    }

    /// §8.4.3 + Table 16: when the luma mode is NOT one of
    /// {DC, HOR, VER, BI}, the chroma index maps straight through Table
    /// 16 with no skip.
    #[test]
    fn chroma_no_skip_for_directional_luma() {
        // luma = 18 (INTRA_DIA_R) — not in the skip set.
        assert_eq!(derive_chroma_mode(1, 18), INTRA_BI);
        assert_eq!(derive_chroma_mode(2, 18), INTRA_DC);
        assert_eq!(derive_chroma_mode(3, 18), INTRA_HOR);
        assert_eq!(derive_chroma_mode(4, 18), INTRA_VER);
    }

    /// §8.4.3 skip rule: when the luma mode collides with a Table 16
    /// entry, indices >= modeIdx are bumped by one so the chroma mode
    /// never duplicates the (already-available-as-DM) luma mode.
    #[test]
    fn chroma_skip_rule() {
        // luma = INTRA_DC → modeIdx = 2. chroma_pred_mode 2 maps to
        // index 3 (INTRA_HOR), skipping the duplicate DC.
        assert_eq!(derive_chroma_mode(1, INTRA_DC), INTRA_BI); // 1 < 2: no skip
        assert_eq!(derive_chroma_mode(2, INTRA_DC), INTRA_HOR); // 2 >= 2: +1 → HOR
        assert_eq!(derive_chroma_mode(3, INTRA_DC), INTRA_VER); // 3 >= 2: +1 → VER
                                                                // luma = INTRA_BI → modeIdx = 1: every index bumps.
        assert_eq!(derive_chroma_mode(1, INTRA_BI), INTRA_DC); // 1 >= 1: +1 → DC
        assert_eq!(derive_chroma_mode(2, INTRA_BI), INTRA_HOR); // 2 >= 1: +1 → HOR
    }

    /// validC pruning: A == B forces candB := candC and validC := false.
    #[test]
    fn validc_pruning_a_eq_b() {
        let (a, b, c, vc) = resolve_candidates(
            NeighbourMode::valid(12),
            NeighbourMode::valid(12),
            NeighbourMode::valid(18),
        );
        assert_eq!((a, b, c, vc), (12, 18, 18, false));
    }
}
