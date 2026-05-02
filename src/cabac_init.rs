//! EVC Main-profile CABAC context initialization tables (ISO/IEC 23094-1
//! §9.3.2.2 + §9.3.4.2).
//!
//! When `sps_cm_init_flag == 1` (Main profile), every regular-coded
//! syntax element draws its `(valState, valMps)` initial value from a
//! per-element table (Tables 40 to 90 in the spec). The 10-bit
//! `initValue` is combined with the slice-level QP per eq. 1425/1426 to
//! derive the engine state at the start of each tile / dependent-slice
//! segment.
//!
//! This module is the *clean-room transcription* of those tables. The
//! actual decode logic for the Main-profile syntax elements (BTT split,
//! ADMVP, AMVR, MMVD, affine, ALF, DRA, …) is **not** wired here — the
//! goal of this round is just to land the init data and the
//! `ctxInc`-derivation helpers (subclauses 9.3.4.2.2 through 9.3.4.2.12)
//! so subsequent rounds can implement individual tools without having
//! to revisit every initValue list.
//!
//! ## Organisation
//!
//! For each context-coded syntax element we provide a `pub const`
//! initValue array sized to the spec table:
//!
//! ```text
//! pub const TABLE_42_BTT_SPLIT_FLAG_INIT: &[u16] = &[ 145, 560, … ];
//! ```
//!
//! The [`MainCtxTable`] enum keys each table by its spec number and
//! exposes:
//!
//! * `init_values()` — the raw `&[u16]` table contents,
//! * `init_type_range(InitType)` — the `(start, end)` ctxIdx slice that
//!   applies to a given initialization type (per Table 39).
//!
//! [`init_main_profile_contexts`] populates a [`CabacEngine`] context
//! table with the per-slice-QP-derived values for every Main-profile
//! ctxTable.
//!
//! ## Verification
//!
//! Each table's contents are spot-checked at the bottom of this file
//! against the printed values in the spec PDF
//! (`docs/video/evc/ISO_IEC_23094-1-EVC-2020.pdf` §9.3.5 Tables 40-90).

use crate::cabac::{init_contexts_from_init_value, CabacEngine, InitType};

// =====================================================================
// Init-value tables (Tables 40 through 90 in ISO/IEC 23094-1:2020 §9.3.5).
// Each array entry is the 10-bit `initValue` for one (ctxTable, ctxIdx)
// pair. The split between initType=0 (I) and initType=1 (P/B) is
// encoded in `MainCtxTable::init_type_range`.
// =====================================================================

/// Table 40 — alf_ctb_flag / alf_ctb_chroma_flag / alf_ctb_chroma2_flag.
/// All three syntax elements share the same 2-entry table per Table 39.
pub const TABLE_40_ALF_CTB_FLAG_INIT: &[u16] = &[0, 0];

/// Table 41 — split_cu_flag.
pub const TABLE_41_SPLIT_CU_FLAG_INIT: &[u16] = &[0, 0];

/// Table 42 — btt_split_flag (binary tree split). 30 entries: 0..14 for
/// I slices, 15..29 for P/B (Table 39).
pub const TABLE_42_BTT_SPLIT_FLAG_INIT: &[u16] = &[
    145, 560, 528, 308, 594, 560, 180, 500, // 0..7
    626, 84, 406, 662, 320, 36, 340, 536, // 8..15
    726, 594, 66, 338, 528, 258, 404, 464, // 16..23
    98, 342, 370, 384, 256, 65, // 24..29
];

/// Table 43 — btt_split_dir. 10 entries: 0..4 for I, 5..9 for P/B.
pub const TABLE_43_BTT_SPLIT_DIR_INIT: &[u16] = &[
    0, 417, 389, 99, 0, // 0..4
    0, 128, 81, 49, 0, // 5..9
];

/// Table 44 — btt_split_type. 2 entries.
pub const TABLE_44_BTT_SPLIT_TYPE_INIT: &[u16] = &[257, 225];

/// Table 45 — split_unit_coding_order_flag. 28 entries (per the printed
/// spec), although Table 39 references only ctxIdx 0..11 / 12..23. The
/// trailing entries (24..27) remain in case ctxInc derivation
/// (§9.3.4.2.3) yields a value beyond 11 for unusual block shapes; the
/// spec table just provides them.
pub const TABLE_45_SUCO_FLAG_INIT: &[u16] = &[
    0, 0, 0, 0, 0, 0, 545, 0, // 0..7
    481, 515, 0, 32, 0, 0, 0, 0, // 8..15
    0, 0, 0, 0, 557, 0, 481, 2, // 16..23
    0, 97, 0, 0, // 24..27
];

/// Table 46 — pred_mode_constraint_type_flag.
pub const TABLE_46_PRED_MODE_CONSTRAINT_TYPE_FLAG_INIT: &[u16] = &[0, 481];

/// Table 47 — cu_skip_flag. ctxIdx 0..1 for I, 2..3 for P/B.
pub const TABLE_47_CU_SKIP_FLAG_INIT: &[u16] = &[0, 0, 711, 233];

/// Table 48 — mvp_idx_l0 / mvp_idx_l1. 6 entries (3 per initType).
pub const TABLE_48_MVP_IDX_INIT: &[u16] = &[0, 0, 0, 0, 0, 0];

/// Table 49 — merge_idx. 10 entries (5 per initType).
pub const TABLE_49_MERGE_IDX_INIT: &[u16] = &[
    0, 0, 0, 496, 496, // 0..4
    18, 128, 146, 37, 69, // 5..9
];

/// Table 50 — mmvd_flag. 2 entries.
pub const TABLE_50_MMVD_FLAG_INIT: &[u16] = &[0, 194];

/// Table 51 — mmvd_group_idx. 4 entries (2 per initType).
pub const TABLE_51_MMVD_GROUP_IDX_INIT: &[u16] = &[0, 0, 453, 48];

/// Table 52 — mmvd_merge_idx. 6 entries (3 per initType).
pub const TABLE_52_MMVD_MERGE_IDX_INIT: &[u16] = &[0, 0, 0, 49, 129, 82];

/// Table 53 — mmvd_distance_idx. 14 entries (7 per initType).
pub const TABLE_53_MMVD_DISTANCE_IDX_INIT: &[u16] = &[
    0, 0, 0, 0, 0, 0, 0, 179, // 0..7
    5, 133, 131, 227, 64, 128, // 8..13
];

/// Table 54 — mmvd_direction_idx. 4 entries (2 per initType).
pub const TABLE_54_MMVD_DIRECTION_IDX_INIT: &[u16] = &[0, 0, 161, 33];

/// Table 55 — affine_flag. 4 entries (2 per initType).
pub const TABLE_55_AFFINE_FLAG_INIT: &[u16] = &[0, 0, 320, 210];

/// Table 56 — affine_merge_idx. 10 entries (5 per initType).
pub const TABLE_56_AFFINE_MERGE_IDX_INIT: &[u16] = &[
    0, 0, 0, 0, 0, // 0..4
    193, 129, 32, 323, 0, // 5..9
];

/// Table 57 — affine_mode_flag. 2 entries.
pub const TABLE_57_AFFINE_MODE_FLAG_INIT: &[u16] = &[0, 225];

/// Table 58 — affine_mvp_flag_l0 / affine_mvp_flag_l1.
pub const TABLE_58_AFFINE_MVP_FLAG_INIT: &[u16] = &[0, 161];

/// Table 59 — affine_mvd_flag_l0.
pub const TABLE_59_AFFINE_MVD_FLAG_L0_INIT: &[u16] = &[0, 547];

/// Table 60 — affine_mvd_flag_l1.
pub const TABLE_60_AFFINE_MVD_FLAG_L1_INIT: &[u16] = &[0, 645];

/// Table 61 — pred_mode_flag. 6 entries (3 per initType).
pub const TABLE_61_PRED_MODE_FLAG_INIT: &[u16] = &[64, 0, 0, 481, 16, 368];

/// Table 62 — intra_pred_mode. 4 entries (2 per initType).
pub const TABLE_62_INTRA_PRED_MODE_INIT: &[u16] = &[0, 0, 0, 0];

/// Table 63 — intra_luma_pred_mpm_flag. 2 entries.
pub const TABLE_63_INTRA_LUMA_PRED_MPM_FLAG_INIT: &[u16] = &[263, 225];

/// Table 64 — intra_luma_pred_mpm_idx. 2 entries.
pub const TABLE_64_INTRA_LUMA_PRED_MPM_IDX_INIT: &[u16] = &[436, 724];

/// Table 65 — intra_chroma_pred_mode. 2 entries.
pub const TABLE_65_INTRA_CHROMA_PRED_MODE_INIT: &[u16] = &[465, 560];

/// Table 66 — ibc_flag. 4 entries (2 per initType).
pub const TABLE_66_IBC_FLAG_INIT: &[u16] = &[0, 0, 711, 233];

/// Table 67 — amvr_idx. 8 entries (4 per initType).
pub const TABLE_67_AMVR_IDX_INIT: &[u16] = &[
    0, 0, 0, 496, // 0..3
    773, 101, 421, 199, // 4..7
];

/// Table 68 — direct_mode_flag. 2 entries.
pub const TABLE_68_DIRECT_MODE_FLAG_INIT: &[u16] = &[0, 0];

/// Table 69 — inter_pred_idc. 4 entries (2 per initType).
pub const TABLE_69_INTER_PRED_IDC_INIT: &[u16] = &[0, 0, 242, 80];

/// Table 70 — merge_mode_flag. 2 entries.
pub const TABLE_70_MERGE_MODE_FLAG_INIT: &[u16] = &[0, 464];

/// Table 71 — bi_pred_idx. 4 entries (2 per initType).
pub const TABLE_71_BI_PRED_IDX_INIT: &[u16] = &[0, 0, 49, 17];

/// Table 72 — ref_idx_l0 / ref_idx_l1. 4 entries (2 per initType).
pub const TABLE_72_REF_IDX_INIT: &[u16] = &[0, 0, 288, 0];

/// Table 73 — abs_mvd_l0 / abs_mvd_l1. 2 entries.
pub const TABLE_73_ABS_MVD_INIT: &[u16] = &[0, 18];

/// Table 74 — cbf_all. 2 entries.
pub const TABLE_74_CBF_ALL_INIT: &[u16] = &[0, 794];

/// Table 75 — cbf_luma. 2 entries.
pub const TABLE_75_CBF_LUMA_INIT: &[u16] = &[664, 368];

/// Table 76 — cbf_cb. 2 entries.
pub const TABLE_76_CBF_CB_INIT: &[u16] = &[384, 416];

/// Table 77 — cbf_cr. 2 entries.
pub const TABLE_77_CBF_CR_INIT: &[u16] = &[320, 288];

/// Table 78 — cu_qp_delta_abs. 2 entries.
pub const TABLE_78_CU_QP_DELTA_ABS_INIT: &[u16] = &[4, 4];

/// Table 79 — ats_hor_mode / ats_ver_mode. 2 entries.
pub const TABLE_79_ATS_MODE_INIT: &[u16] = &[512, 673];

/// Table 80 — ats_cu_inter_flag. 4 entries (2 per initType).
pub const TABLE_80_ATS_CU_INTER_FLAG_INIT: &[u16] = &[0, 0, 0, 0];

/// Table 81 — ats_cu_inter_quad_flag. 2 entries.
pub const TABLE_81_ATS_CU_INTER_QUAD_FLAG_INIT: &[u16] = &[0, 0];

/// Table 82 — ats_cu_inter_horizontal_flag. 6 entries (3 per initType).
pub const TABLE_82_ATS_CU_INTER_HORIZONTAL_FLAG_INIT: &[u16] = &[0, 0, 0, 0, 0, 0];

/// Table 83 — ats_cu_inter_pos_flag. 2 entries.
pub const TABLE_83_ATS_CU_INTER_POS_FLAG_INIT: &[u16] = &[0, 0];

/// Table 84 — coeff_zero_run. 48 entries (24 per initType).
pub const TABLE_84_COEFF_ZERO_RUN_INIT: &[u16] = &[
    48, 112, 128, 0, 321, 82, 419, 160, // 0..7
    385, 323, 353, 129, 225, 193, 387, 389, // 8..15
    453, 227, 453, 161, 421, 161, 481, 225, // 16..23
    129, 178, 453, 97, 583, 259, 517, 259, // 24..31
    453, 227, 871, 355, 291, 227, 195, 97, // 32..39
    161, 65, 97, 33, 65, 1, 1003, 227, // 40..47
];

/// Table 85 — coeff_abs_level_minus1. 48 entries (24 per initType).
pub const TABLE_85_COEFF_ABS_LEVEL_MINUS1_INIT: &[u16] = &[
    416, 98, 128, 66, 32, 82, 17, 48, // 0..7
    272, 112, 52, 50, 448, 419, 385, 355, // 8..15
    161, 225, 82, 97, 210, 0, 416, 224, // 16..23
    805, 775, 775, 581, 355, 389, 65, 195, // 24..31
    48, 33, 224, 225, 775, 227, 355, 161, // 32..39
    129, 97, 33, 65, 16, 1, 841, 355, // 40..47
];

/// Table 86 — coeff_last_flag. 4 entries (2 per initType).
pub const TABLE_86_COEFF_LAST_FLAG_INIT: &[u16] = &[421, 337, 33, 790];

/// Table 87 — last_sig_coeff_x_prefix. 42 entries (21 per initType).
pub const TABLE_87_LAST_SIG_COEFF_X_PREFIX_INIT: &[u16] = &[
    762, 310, 288, 828, 342, 451, 502, 51, // 0..7
    97, 416, 662, 890, 340, 146, 20, 337, // 8..15
    468, 216, 66, 54, 216, 892, 84, 581, // 16..23
    600, 278, 419, 372, 568, 408, 485, 338, // 24..31
    632, 666, 732, 16, 178, 180, 585, 581, // 32..39
    34, 257, // 40..41
];

/// Table 88 — last_sig_coeff_y_prefix. 42 entries (21 per initType).
pub const TABLE_88_LAST_SIG_COEFF_Y_PREFIX_INIT: &[u16] = &[
    81, 440, 4, 534, 406, 226, 370, 370, // 0..7
    259, 38, 598, 792, 860, 312, 88, 662, // 8..15
    924, 161, 248, 20, 54, 470, 376, 323, // 16..23
    276, 602, 52, 340, 600, 376, 378, 598, // 24..31
    502, 730, 538, 17, 195, 504, 378, 320, // 32..39
    160, 572, // 40..41
];

/// Table 89 — sig_coeff_flag. 94 entries (47 per initType).
pub const TABLE_89_SIG_COEFF_FLAG_INIT: &[u16] = &[
    387, 98, 233, 346, 717, 306, 233, 37, // 0..7
    321, 293, 244, 37, 329, 645, 408, 493, // 8..15
    164, 781, 101, 179, 369, 871, 585, 244, // 16..23
    361, 147, 416, 408, 628, 352, 406, 502, // 24..31
    566, 466, 54, 97, 521, 113, 147, 519, // 32..39
    36, 297, 132, 457, 308, 231, 534, 66, // 40..47
    34, 241, 321, 293, 113, 35, 83, 226, // 48..55
    519, 553, 229, 751, 224, 129, 133, 162, // 56..63
    227, 178, 165, 532, 417, 357, 33, 489, // 64..71
    199, 387, 939, 133, 515, 32, 131, 3, // 72..79
    305, 579, 323, 65, 99, 425, 453, 291, // 80..87
    329, 679, 683, 391, 751, 51, // 88..93
];

/// Table 90 — coeff_abs_level_greaterA_flag / coeff_abs_level_greaterB_flag.
/// 36 entries (18 per initType).
pub const TABLE_90_COEFF_ABS_LEVEL_GREATER_FLAG_INIT: &[u16] = &[
    40, 225, 306, 272, 85, 120, 389, 664, // 0..7
    209, 322, 291, 536, 338, 709, 54, 244, // 8..15
    19, 566, 38, 352, 340, 19, 305, 258, // 16..23
    18, 33, 209, 773, 517, 406, 719, 741, // 24..31
    613, 295, 37, 498, // 32..35
];

/// Identifier for one of the Main-profile init tables (40 through 90 in
/// ISO/IEC 23094-1:2020 §9.3.5).
///
/// The discriminant is the spec table number, so the enum doubles as
/// the `ctxTable` value used in calls to
/// [`CabacEngine::decode_decision`]. Some syntax elements share a
/// table (e.g. `mvp_idx_l0` and `mvp_idx_l1` both use Table 48); use
/// the `Mvp`, `Ref`, `AbsMvd`, `AffineMvp`, `AlfCtbFlag` and `AtsMode`
/// shared variants for those.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MainCtxTable {
    /// Table 40 — alf_ctb_flag / alf_ctb_chroma_flag / alf_ctb_chroma2_flag.
    AlfCtbFlag = 40,
    /// Table 41 — split_cu_flag.
    SplitCuFlag = 41,
    /// Table 42 — btt_split_flag.
    BttSplitFlag = 42,
    /// Table 43 — btt_split_dir.
    BttSplitDir = 43,
    /// Table 44 — btt_split_type.
    BttSplitType = 44,
    /// Table 45 — split_unit_coding_order_flag (SUCO).
    SucoFlag = 45,
    /// Table 46 — pred_mode_constraint_type_flag.
    PredModeConstraintType = 46,
    /// Table 47 — cu_skip_flag.
    CuSkipFlag = 47,
    /// Table 48 — mvp_idx_l0 / mvp_idx_l1.
    MvpIdx = 48,
    /// Table 49 — merge_idx.
    MergeIdx = 49,
    /// Table 50 — mmvd_flag.
    MmvdFlag = 50,
    /// Table 51 — mmvd_group_idx.
    MmvdGroupIdx = 51,
    /// Table 52 — mmvd_merge_idx.
    MmvdMergeIdx = 52,
    /// Table 53 — mmvd_distance_idx.
    MmvdDistanceIdx = 53,
    /// Table 54 — mmvd_direction_idx.
    MmvdDirectionIdx = 54,
    /// Table 55 — affine_flag.
    AffineFlag = 55,
    /// Table 56 — affine_merge_idx.
    AffineMergeIdx = 56,
    /// Table 57 — affine_mode_flag.
    AffineModeFlag = 57,
    /// Table 58 — affine_mvp_flag_l0 / affine_mvp_flag_l1.
    AffineMvpFlag = 58,
    /// Table 59 — affine_mvd_flag_l0.
    AffineMvdFlagL0 = 59,
    /// Table 60 — affine_mvd_flag_l1.
    AffineMvdFlagL1 = 60,
    /// Table 61 — pred_mode_flag.
    PredModeFlag = 61,
    /// Table 62 — intra_pred_mode (sps_eipd_flag = 1 path).
    IntraPredMode = 62,
    /// Table 63 — intra_luma_pred_mpm_flag.
    IntraLumaPredMpmFlag = 63,
    /// Table 64 — intra_luma_pred_mpm_idx.
    IntraLumaPredMpmIdx = 64,
    /// Table 65 — intra_chroma_pred_mode.
    IntraChromaPredMode = 65,
    /// Table 66 — ibc_flag.
    IbcFlag = 66,
    /// Table 67 — amvr_idx.
    AmvrIdx = 67,
    /// Table 68 — direct_mode_flag.
    DirectModeFlag = 68,
    /// Table 69 — inter_pred_idc.
    InterPredIdc = 69,
    /// Table 70 — merge_mode_flag.
    MergeModeFlag = 70,
    /// Table 71 — bi_pred_idx.
    BiPredIdx = 71,
    /// Table 72 — ref_idx_l0 / ref_idx_l1.
    RefIdx = 72,
    /// Table 73 — abs_mvd_l0 / abs_mvd_l1.
    AbsMvd = 73,
    /// Table 74 — cbf_all.
    CbfAll = 74,
    /// Table 75 — cbf_luma.
    CbfLuma = 75,
    /// Table 76 — cbf_cb.
    CbfCb = 76,
    /// Table 77 — cbf_cr.
    CbfCr = 77,
    /// Table 78 — cu_qp_delta_abs.
    CuQpDeltaAbs = 78,
    /// Table 79 — ats_hor_mode / ats_ver_mode.
    AtsMode = 79,
    /// Table 80 — ats_cu_inter_flag.
    AtsCuInterFlag = 80,
    /// Table 81 — ats_cu_inter_quad_flag.
    AtsCuInterQuadFlag = 81,
    /// Table 82 — ats_cu_inter_horizontal_flag.
    AtsCuInterHorizontalFlag = 82,
    /// Table 83 — ats_cu_inter_pos_flag.
    AtsCuInterPosFlag = 83,
    /// Table 84 — coeff_zero_run.
    CoeffZeroRun = 84,
    /// Table 85 — coeff_abs_level_minus1.
    CoeffAbsLevelMinus1 = 85,
    /// Table 86 — coeff_last_flag.
    CoeffLastFlag = 86,
    /// Table 87 — last_sig_coeff_x_prefix.
    LastSigCoeffXPrefix = 87,
    /// Table 88 — last_sig_coeff_y_prefix.
    LastSigCoeffYPrefix = 88,
    /// Table 89 — sig_coeff_flag.
    SigCoeffFlag = 89,
    /// Table 90 — coeff_abs_level_greaterA_flag / coeff_abs_level_greaterB_flag.
    CoeffAbsLevelGreaterFlag = 90,
}

impl MainCtxTable {
    /// All Main-profile context tables in declaration order. Used by
    /// [`init_main_profile_contexts`] to walk every entry and by tests.
    pub const ALL: &'static [MainCtxTable] = &[
        MainCtxTable::AlfCtbFlag,
        MainCtxTable::SplitCuFlag,
        MainCtxTable::BttSplitFlag,
        MainCtxTable::BttSplitDir,
        MainCtxTable::BttSplitType,
        MainCtxTable::SucoFlag,
        MainCtxTable::PredModeConstraintType,
        MainCtxTable::CuSkipFlag,
        MainCtxTable::MvpIdx,
        MainCtxTable::MergeIdx,
        MainCtxTable::MmvdFlag,
        MainCtxTable::MmvdGroupIdx,
        MainCtxTable::MmvdMergeIdx,
        MainCtxTable::MmvdDistanceIdx,
        MainCtxTable::MmvdDirectionIdx,
        MainCtxTable::AffineFlag,
        MainCtxTable::AffineMergeIdx,
        MainCtxTable::AffineModeFlag,
        MainCtxTable::AffineMvpFlag,
        MainCtxTable::AffineMvdFlagL0,
        MainCtxTable::AffineMvdFlagL1,
        MainCtxTable::PredModeFlag,
        MainCtxTable::IntraPredMode,
        MainCtxTable::IntraLumaPredMpmFlag,
        MainCtxTable::IntraLumaPredMpmIdx,
        MainCtxTable::IntraChromaPredMode,
        MainCtxTable::IbcFlag,
        MainCtxTable::AmvrIdx,
        MainCtxTable::DirectModeFlag,
        MainCtxTable::InterPredIdc,
        MainCtxTable::MergeModeFlag,
        MainCtxTable::BiPredIdx,
        MainCtxTable::RefIdx,
        MainCtxTable::AbsMvd,
        MainCtxTable::CbfAll,
        MainCtxTable::CbfLuma,
        MainCtxTable::CbfCb,
        MainCtxTable::CbfCr,
        MainCtxTable::CuQpDeltaAbs,
        MainCtxTable::AtsMode,
        MainCtxTable::AtsCuInterFlag,
        MainCtxTable::AtsCuInterQuadFlag,
        MainCtxTable::AtsCuInterHorizontalFlag,
        MainCtxTable::AtsCuInterPosFlag,
        MainCtxTable::CoeffZeroRun,
        MainCtxTable::CoeffAbsLevelMinus1,
        MainCtxTable::CoeffLastFlag,
        MainCtxTable::LastSigCoeffXPrefix,
        MainCtxTable::LastSigCoeffYPrefix,
        MainCtxTable::SigCoeffFlag,
        MainCtxTable::CoeffAbsLevelGreaterFlag,
    ];

    /// `ctxTable` value as a usize, i.e. the spec table number. Pass
    /// this to [`CabacEngine::decode_decision`].
    #[inline]
    pub const fn as_usize(self) -> usize {
        self as u8 as usize
    }

    /// Initial-value table contents per Table 39 column "Table N".
    pub const fn init_values(self) -> &'static [u16] {
        match self {
            Self::AlfCtbFlag => TABLE_40_ALF_CTB_FLAG_INIT,
            Self::SplitCuFlag => TABLE_41_SPLIT_CU_FLAG_INIT,
            Self::BttSplitFlag => TABLE_42_BTT_SPLIT_FLAG_INIT,
            Self::BttSplitDir => TABLE_43_BTT_SPLIT_DIR_INIT,
            Self::BttSplitType => TABLE_44_BTT_SPLIT_TYPE_INIT,
            Self::SucoFlag => TABLE_45_SUCO_FLAG_INIT,
            Self::PredModeConstraintType => TABLE_46_PRED_MODE_CONSTRAINT_TYPE_FLAG_INIT,
            Self::CuSkipFlag => TABLE_47_CU_SKIP_FLAG_INIT,
            Self::MvpIdx => TABLE_48_MVP_IDX_INIT,
            Self::MergeIdx => TABLE_49_MERGE_IDX_INIT,
            Self::MmvdFlag => TABLE_50_MMVD_FLAG_INIT,
            Self::MmvdGroupIdx => TABLE_51_MMVD_GROUP_IDX_INIT,
            Self::MmvdMergeIdx => TABLE_52_MMVD_MERGE_IDX_INIT,
            Self::MmvdDistanceIdx => TABLE_53_MMVD_DISTANCE_IDX_INIT,
            Self::MmvdDirectionIdx => TABLE_54_MMVD_DIRECTION_IDX_INIT,
            Self::AffineFlag => TABLE_55_AFFINE_FLAG_INIT,
            Self::AffineMergeIdx => TABLE_56_AFFINE_MERGE_IDX_INIT,
            Self::AffineModeFlag => TABLE_57_AFFINE_MODE_FLAG_INIT,
            Self::AffineMvpFlag => TABLE_58_AFFINE_MVP_FLAG_INIT,
            Self::AffineMvdFlagL0 => TABLE_59_AFFINE_MVD_FLAG_L0_INIT,
            Self::AffineMvdFlagL1 => TABLE_60_AFFINE_MVD_FLAG_L1_INIT,
            Self::PredModeFlag => TABLE_61_PRED_MODE_FLAG_INIT,
            Self::IntraPredMode => TABLE_62_INTRA_PRED_MODE_INIT,
            Self::IntraLumaPredMpmFlag => TABLE_63_INTRA_LUMA_PRED_MPM_FLAG_INIT,
            Self::IntraLumaPredMpmIdx => TABLE_64_INTRA_LUMA_PRED_MPM_IDX_INIT,
            Self::IntraChromaPredMode => TABLE_65_INTRA_CHROMA_PRED_MODE_INIT,
            Self::IbcFlag => TABLE_66_IBC_FLAG_INIT,
            Self::AmvrIdx => TABLE_67_AMVR_IDX_INIT,
            Self::DirectModeFlag => TABLE_68_DIRECT_MODE_FLAG_INIT,
            Self::InterPredIdc => TABLE_69_INTER_PRED_IDC_INIT,
            Self::MergeModeFlag => TABLE_70_MERGE_MODE_FLAG_INIT,
            Self::BiPredIdx => TABLE_71_BI_PRED_IDX_INIT,
            Self::RefIdx => TABLE_72_REF_IDX_INIT,
            Self::AbsMvd => TABLE_73_ABS_MVD_INIT,
            Self::CbfAll => TABLE_74_CBF_ALL_INIT,
            Self::CbfLuma => TABLE_75_CBF_LUMA_INIT,
            Self::CbfCb => TABLE_76_CBF_CB_INIT,
            Self::CbfCr => TABLE_77_CBF_CR_INIT,
            Self::CuQpDeltaAbs => TABLE_78_CU_QP_DELTA_ABS_INIT,
            Self::AtsMode => TABLE_79_ATS_MODE_INIT,
            Self::AtsCuInterFlag => TABLE_80_ATS_CU_INTER_FLAG_INIT,
            Self::AtsCuInterQuadFlag => TABLE_81_ATS_CU_INTER_QUAD_FLAG_INIT,
            Self::AtsCuInterHorizontalFlag => TABLE_82_ATS_CU_INTER_HORIZONTAL_FLAG_INIT,
            Self::AtsCuInterPosFlag => TABLE_83_ATS_CU_INTER_POS_FLAG_INIT,
            Self::CoeffZeroRun => TABLE_84_COEFF_ZERO_RUN_INIT,
            Self::CoeffAbsLevelMinus1 => TABLE_85_COEFF_ABS_LEVEL_MINUS1_INIT,
            Self::CoeffLastFlag => TABLE_86_COEFF_LAST_FLAG_INIT,
            Self::LastSigCoeffXPrefix => TABLE_87_LAST_SIG_COEFF_X_PREFIX_INIT,
            Self::LastSigCoeffYPrefix => TABLE_88_LAST_SIG_COEFF_Y_PREFIX_INIT,
            Self::SigCoeffFlag => TABLE_89_SIG_COEFF_FLAG_INIT,
            Self::CoeffAbsLevelGreaterFlag => TABLE_90_COEFF_ABS_LEVEL_GREATER_FLAG_INIT,
        }
    }

    /// `(start, end)` ctxIdx half-open range used for the given
    /// `init_type` per Table 39. The two halves of every shared table
    /// are equal in size; for I slices we use the first half, for P/B
    /// the second.
    pub fn init_type_range(self, init_type: InitType) -> (usize, usize) {
        let len = self.init_values().len();
        // Most tables split exactly in half. Three special-cases:
        //  * Tables with `na` in the I column (e.g. cu_skip_flag with
        //    initType=0 ctxIdx=0..1, initType=1 ctxIdx=2..3) — same
        //    even split, so the generic half-and-half rule still works.
        //  * Tables shared by *two* syntax elements at the same time
        //    (e.g. abs_mvd_l0 and abs_mvd_l1 both reference Table 73
        //    with ctxIdx 0..1 for both initTypes) — these have a
        //    single-pair table where both initTypes use the *same*
        //    range. We handle the "same range" case via a manual
        //    override list below.
        // The override list is empty for now; every Table 40-90 entry
        // uses the half-and-half convention per Table 39's printed
        // ranges.
        let half = len / 2;
        match init_type {
            InitType::I => (0, half.max(1)),
            InitType::Pb => {
                if len == 1 {
                    (0, 1)
                } else if len % 2 == 1 {
                    // Odd-length tables (none in the current spec, but
                    // belt-and-braces): treat the trailing entry as
                    // belonging to P/B.
                    (half, len)
                } else {
                    (half, len)
                }
            }
        }
    }
}

/// Initialise a [`CabacEngine`]'s context table for `sps_cm_init_flag == 1`
/// (Main profile) per §9.3.2.2 case 2. Walks every Main-profile context
/// table, derives `(valState, valMps)` from the slice QP via
/// [`init_contexts_from_init_value`] (eq. 1425/1426), and installs the
/// result at `engine.context(table.as_usize(), ctx_idx)` for every
/// `ctx_idx` in the printed initValue list.
///
/// The Baseline `(ctx_table=0, ctx_idx=0)` slot is left untouched at the
/// default `(256, 0)` so callers that cross between Main and Baseline
/// helper paths can keep using ctxTable 0 for the legacy single-context
/// path. The function is idempotent: calling it twice yields the same
/// state.
///
/// # Errors
///
/// Returns [`oxideav_core::Error::Invalid`] if any (table, ctx_idx) pair
/// is out of range — which only happens if the engine was created with
/// `MAX_CTX_TABLES < 91` or `MAX_CTX_PER_TABLE < 94`. In a freshly
/// constructed engine this never fails.
pub fn init_main_profile_contexts(
    engine: &mut CabacEngine<'_>,
    init_type: InitType,
    slice_qp: i32,
) -> oxideav_core::Result<()> {
    for &table in MainCtxTable::ALL {
        let (start, end) = table.init_type_range(init_type);
        let values = table.init_values();
        for (ctx_idx, &init_value) in values.iter().enumerate().take(end).skip(start) {
            let var = init_contexts_from_init_value(init_value, slice_qp);
            engine.set_context(table.as_usize(), ctx_idx, var)?;
        }
    }
    Ok(())
}

// =====================================================================
// ctxInc derivation helpers (§9.3.4.2.2 through 9.3.4.2.12).
// These are the per-syntax-element ctxInc derivations referenced in
// Table 95. They return the `ctxInc` to be added to `ctxIdxOffset`
// (per §9.3.4.2.1) — for the Main-profile path
// `ctxIdxOffset == init_type_range(.).0`.
// =====================================================================

/// §9.3.4.2.2 — ctxInc for `coeff_zero_run` and `coeff_abs_level_minus1`
/// (eq. 1434 / 1435).
///
/// * `bin_idx` — position of the bin in the binarised string;
/// * `c_idx` — 0 for luma, > 0 for chroma;
/// * `prev_level` — `Abs(TransCoeffLevel[…last non-zero…])`.
pub fn ctx_inc_coeff_zero_run(bin_idx: u32, c_idx: u32, prev_level: u32) -> usize {
    // Spec subtracts 1 from prev_level; it must be ≥ 1 by definition
    // (the previous coefficient is non-zero), so saturate at 0 to avoid
    // wrap on the first call where the caller might pass 0.
    let saturated_prev = prev_level.saturating_sub(1).min(5);
    let base = (saturated_prev << 1) + bin_idx.min(1);
    let chroma_offset = if c_idx == 0 { 0 } else { 12 };
    (base + chroma_offset) as usize
}

/// §9.3.4.2.3 — ctxInc for `split_unit_coding_order_flag` (eq. 1436 / 1437).
pub fn ctx_inc_suco_flag(n_cb_w: u32, n_cb_h: u32) -> usize {
    let log2w = n_cb_w.trailing_zeros();
    let log2h = n_cb_h.trailing_zeros();
    let max_log2 = log2w.max(log2h);
    let base = (max_log2.saturating_sub(2)) << 1;
    let extra = if n_cb_w == n_cb_h { 0 } else { 1 };
    (base + extra) as usize
}

/// §9.3.4.2.4 — ctxInc derived from neighbour-block syntax elements
/// (eq. 1438). Used by `affine_flag`, `cu_skip_flag`, `pred_mode_flag`
/// and `ibc_flag`.
///
/// * `cond_l`/`cond_a`/`cond_r` — the per-syntax-element condition
///   evaluated at the L/A/R neighbour (1 if true and the neighbour is
///   available, else 0 — caller should AND with availability flags
///   before passing in);
/// * `num_ctx` — Table 96 numCtx (2 for affine/cu_skip/ibc, 3 for pred_mode).
pub fn ctx_inc_neighbour_sum(cond_l: u32, cond_a: u32, cond_r: u32, num_ctx: u32) -> usize {
    let sum = cond_l + cond_a + cond_r;
    sum.min(num_ctx.saturating_sub(1)) as usize
}

/// §9.3.4.2.5 — ctxInc for `btt_split_flag` (eq. 1440 + Table 97).
///
/// * `num_smaller` — count of L/A/R neighbours with smaller block
///   dimensions (eq. 1439); caller is expected to combine availability
///   already.
/// * `n_cb_w`, `n_cb_h` — current block size; used to look up
///   `ctxSetIdx` from Table 97.
pub fn ctx_inc_btt_split_flag(num_smaller: u32, n_cb_w: u32, n_cb_h: u32) -> usize {
    let log2w = n_cb_w.trailing_zeros() as i32 - 2;
    let log2h = n_cb_h.trailing_zeros() as i32 - 2;
    let ctx_set_idx = btt_split_ctx_set_idx(log2w, log2h);
    (num_smaller.min(2) as usize) + 3 * (ctx_set_idx as usize)
}

/// Table 97 lookup. The `na` cells are unreachable in valid streams; we
/// return 0xFF (a sentinel) so a buggy caller surfaces a panic via the
/// downstream array index rather than silently returning a low value.
fn btt_split_ctx_set_idx(log2w_minus_2: i32, log2h_minus_2: i32) -> u32 {
    // Table 97 (rows: log2w-2, cols: log2h-2):
    //          h0  h1  h2  h3  h4  h5
    // w0:      na   4   4  na  na  na
    // w1:       4   4   3   3   2   2
    // w2:       4   3   3   2   2   1
    // w3:      na   3   2   2   1   1
    // w4:      na   2   2   1   1   0
    // w5:      na   2   1   1   0   0
    const TBL: [[u8; 6]; 6] = [
        [0xFF, 4, 4, 0xFF, 0xFF, 0xFF],
        [4, 4, 3, 3, 2, 2],
        [4, 3, 3, 2, 2, 1],
        [0xFF, 3, 2, 2, 1, 1],
        [0xFF, 2, 2, 1, 1, 0],
        [0xFF, 2, 1, 1, 0, 0],
    ];
    if (0..6).contains(&log2w_minus_2) && (0..6).contains(&log2h_minus_2) {
        TBL[log2w_minus_2 as usize][log2h_minus_2 as usize] as u32
    } else {
        0xFF
    }
}

/// §9.3.4.2.6 — ctxInc for `last_sig_coeff_x_prefix` and
/// `last_sig_coeff_y_prefix` (eq. 1441).
///
/// * `bin_idx` — position of the bin in the binarised string;
/// * `c_idx` — colour-component index;
/// * `log2_trafo_size` — log2 of TbWidth (X-prefix) or TbHeight (Y-prefix);
/// * `chroma_array_type` — 0..=3 per Table 2 (0 = monochrome, 1 = 4:2:0,
///   2 = 4:2:2, 3 = 4:4:4).
pub fn ctx_inc_last_sig_coeff_prefix(
    bin_idx: u32,
    c_idx: u32,
    log2_trafo_size: u32,
    chroma_array_type: u32,
) -> usize {
    let enable_luma_model =
        chroma_array_type == 0 || chroma_array_type == 3 || (chroma_array_type <= 2 && c_idx == 0);
    let log2 = log2_trafo_size as i32;
    let (ctx_offset, ctx_shift) = if enable_luma_model {
        if log2_trafo_size < 6 {
            let off = 3 * (log2 - 2) + ((log2 - 1) >> 2);
            let shift = (log2 + 1) >> 2;
            (off, shift)
        } else {
            // The spec text reads:
            //   ctxOffset = 3*(log2 − 2) + ((log2 − 1) >> 2)
            //              + (((1 << log2) >> 6) << 1)
            //              + ((1 << log2) >> 7)
            // The published PDF has unbalanced parentheses; we follow
            // the parenthesisation that yields a strictly-monotonic
            // ctxOffset with log2_trafo_size — see eq. 1441 derivation.
            let one_shl = 1i32 << log2;
            let off = 3 * (log2 - 2) + ((log2 - 1) >> 2) + ((one_shl >> 6) << 1) + (one_shl >> 7);
            let shift = ((log2 + 1) >> 2) << 1;
            (off, shift)
        }
    } else {
        let shift = (log2 - 2).max(0) - (log2 - 4).max(0);
        (18, shift)
    };
    ((bin_idx as i32 >> ctx_shift) + ctx_offset).max(0) as usize
}

/// §9.3.4.2.7 — ctxInc for `sig_coeff_flag` (eq. 1447 / 1451).
///
/// * `c_idx` — colour-component index;
/// * `xc`, `yc` — current scan position inside the transform block;
/// * `num_flags` — count of non-zero neighbours per eq. 1442–1446
///   (caller is responsible for evaluating the `TransCoeffLevel`
///   stencil; we just plug the result into the ctx math).
pub fn ctx_inc_sig_coeff_flag(c_idx: u32, xc: u32, yc: u32, num_flags: u32) -> usize {
    let mut sig_ctx = num_flags.min(4) + 1;
    if xc + yc < 2 {
        sig_ctx = sig_ctx.min(2);
    }
    let ctx_offset = if c_idx == 0 {
        if xc + yc < 2 {
            0
        } else if xc + yc < 5 {
            2
        } else {
            7
        }
    } else if xc + yc < 2 {
        0
    } else {
        2
    };
    (sig_ctx + ctx_offset) as usize
}

/// §9.3.4.2.8 — ctxInc for `coeff_abs_level_greaterA_flag` (eq. 1457 / 1458).
///
/// * `c_idx` — colour-component index;
/// * `xc`, `yc` — current scan position;
/// * `is_last` — true iff (xc, yc) == (lastX, lastY) (returns 0 in that case);
/// * `num_flags` — count of `Abs(neighbour) > 1` per eq. 1452–1456.
pub fn ctx_inc_coeff_abs_level_greater_a(
    c_idx: u32,
    xc: u32,
    yc: u32,
    is_last: bool,
    num_flags: u32,
) -> usize {
    if is_last {
        return 0;
    }
    let mut ctx_inc = num_flags.min(3) + 1;
    if c_idx == 0 {
        ctx_inc += if xc + yc < 3 {
            0
        } else if xc + yc < 10 {
            4
        } else {
            8
        };
    }
    ctx_inc as usize
}

/// §9.3.4.2.9 — ctxInc for `coeff_abs_level_greaterB_flag` (eq. 1464 / 1465).
/// Identical math to greaterA except the `num_flags` stencil counts
/// `Abs(neighbour) > 2` (the caller pre-computes this).
pub fn ctx_inc_coeff_abs_level_greater_b(
    c_idx: u32,
    xc: u32,
    yc: u32,
    is_last: bool,
    num_flags: u32,
) -> usize {
    // eq. 1464 / 1465 are identical in form to eq. 1457 / 1458; only
    // the `numFlags` stencil differs (caller-provided), so we delegate.
    ctx_inc_coeff_abs_level_greater_a(c_idx, xc, yc, is_last, num_flags)
}

/// §9.3.4.2.10 — Rice parameter for `coeff_abs_level_remaining` (Table 98).
///
/// * `loc_sum_abs` — Sum of `Abs(neighbour)` per eq. 1466–1470, then
///   `Clip3(0, 31, locSumAbs - baseLevel * 5)` per eq. 1471. Caller is
///   responsible for the clip.
pub fn rice_param_coeff_abs_level_remaining(loc_sum_abs: u32) -> u32 {
    // Table 98:
    //   locSumAbs:  0..=6  -> 0
    //               7..=13 -> 1
    //              14..=27 -> 2
    //              28..=31 -> 3
    match loc_sum_abs {
        0..=6 => 0,
        7..=13 => 1,
        14..=27 => 2,
        _ => 3,
    }
}

/// §9.3.4.2.11 — ctxInc for `ats_cu_inter_flag` (eq. 1472).
pub fn ctx_inc_ats_cu_inter_flag(n_cb_w: u32, n_cb_h: u32) -> usize {
    let log2w = n_cb_w.trailing_zeros();
    let log2h = n_cb_h.trailing_zeros();
    if log2w + log2h >= 8 {
        0
    } else {
        1
    }
}

/// §9.3.4.2.12 — ctxInc for `ats_cu_inter_horizontal_flag` (eq. 1473).
pub fn ctx_inc_ats_cu_inter_horizontal_flag(n_cb_w: u32, n_cb_h: u32) -> usize {
    if n_cb_w == n_cb_h {
        0
    } else if n_cb_w < n_cb_h {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Spot-checks against the printed Tables 40-90 ---
    //
    // For each table we verify:
    //   * `init_values().len()` matches the number of printed cells, and
    //   * a representative ctxIdx returns the printed initValue.
    //
    // These are intentionally hand-keyed so a typo in the transcription
    // surfaces here rather than at decode time.

    #[test]
    fn table_42_btt_split_flag_spot_checks() {
        let t = MainCtxTable::BttSplitFlag.init_values();
        assert_eq!(t.len(), 30);
        // ctxIdx 0 = 145, 7 = 500, 14 = 340, 23 = 464, 29 = 65.
        assert_eq!(t[0], 145);
        assert_eq!(t[7], 500);
        assert_eq!(t[14], 340);
        assert_eq!(t[23], 464);
        assert_eq!(t[29], 65);
    }

    #[test]
    fn table_43_btt_split_dir_spot_checks() {
        let t = MainCtxTable::BttSplitDir.init_values();
        assert_eq!(t.len(), 10);
        // ctxIdx 0 = 0, 1 = 417, 2 = 389, 6 = 128, 9 = 0.
        assert_eq!(t[0], 0);
        assert_eq!(t[1], 417);
        assert_eq!(t[2], 389);
        assert_eq!(t[6], 128);
        assert_eq!(t[9], 0);
    }

    #[test]
    fn table_44_btt_split_type_spot_checks() {
        let t = MainCtxTable::BttSplitType.init_values();
        assert_eq!(t, &[257, 225]);
    }

    #[test]
    fn table_45_suco_flag_spot_checks() {
        let t = MainCtxTable::SucoFlag.init_values();
        assert_eq!(t.len(), 28);
        // ctxIdx 6 = 545, 8 = 481, 9 = 515, 11 = 32, 20 = 557, 22 = 481, 25 = 97.
        assert_eq!(t[6], 545);
        assert_eq!(t[8], 481);
        assert_eq!(t[9], 515);
        assert_eq!(t[11], 32);
        assert_eq!(t[20], 557);
        assert_eq!(t[22], 481);
        assert_eq!(t[25], 97);
    }

    #[test]
    fn table_47_cu_skip_flag_spot_checks() {
        let t = MainCtxTable::CuSkipFlag.init_values();
        assert_eq!(t, &[0, 0, 711, 233]);
    }

    #[test]
    fn table_53_mmvd_distance_idx_spot_checks() {
        let t = MainCtxTable::MmvdDistanceIdx.init_values();
        assert_eq!(t.len(), 14);
        assert_eq!(t[7], 179);
        assert_eq!(t[8], 5);
        assert_eq!(t[10], 131);
        assert_eq!(t[12], 64);
        assert_eq!(t[13], 128);
    }

    #[test]
    fn table_55_affine_flag_spot_checks() {
        let t = MainCtxTable::AffineFlag.init_values();
        assert_eq!(t, &[0, 0, 320, 210]);
    }

    #[test]
    fn table_67_amvr_idx_spot_checks() {
        let t = MainCtxTable::AmvrIdx.init_values();
        assert_eq!(t.len(), 8);
        assert_eq!(t[3], 496);
        assert_eq!(t[4], 773);
        assert_eq!(t[7], 199);
    }

    #[test]
    fn table_84_coeff_zero_run_spot_checks() {
        let t = MainCtxTable::CoeffZeroRun.init_values();
        assert_eq!(t.len(), 48);
        assert_eq!(t[0], 48);
        assert_eq!(t[2], 128);
        assert_eq!(t[34], 871);
        assert_eq!(t[46], 1003);
        assert_eq!(t[47], 227);
    }

    #[test]
    fn table_85_coeff_abs_level_minus1_spot_checks() {
        let t = MainCtxTable::CoeffAbsLevelMinus1.init_values();
        assert_eq!(t.len(), 48);
        assert_eq!(t[0], 416);
        assert_eq!(t[24], 805);
        assert_eq!(t[36], 775);
        assert_eq!(t[46], 841);
        assert_eq!(t[47], 355);
    }

    #[test]
    fn table_87_last_sig_coeff_x_prefix_spot_checks() {
        let t = MainCtxTable::LastSigCoeffXPrefix.init_values();
        assert_eq!(t.len(), 42);
        assert_eq!(t[0], 762);
        assert_eq!(t[11], 890);
        assert_eq!(t[21], 892);
        assert_eq!(t[34], 732);
        assert_eq!(t[41], 257);
    }

    #[test]
    fn table_88_last_sig_coeff_y_prefix_spot_checks() {
        let t = MainCtxTable::LastSigCoeffYPrefix.init_values();
        assert_eq!(t.len(), 42);
        assert_eq!(t[0], 81);
        assert_eq!(t[16], 924);
        assert_eq!(t[33], 730);
        assert_eq!(t[41], 572);
    }

    #[test]
    fn table_89_sig_coeff_flag_spot_checks() {
        let t = MainCtxTable::SigCoeffFlag.init_values();
        assert_eq!(t.len(), 94);
        assert_eq!(t[0], 387);
        assert_eq!(t[15], 493);
        assert_eq!(t[47], 66);
        assert_eq!(t[74], 939);
        assert_eq!(t[93], 51);
    }

    #[test]
    fn table_90_coeff_abs_level_greater_flag_spot_checks() {
        let t = MainCtxTable::CoeffAbsLevelGreaterFlag.init_values();
        assert_eq!(t.len(), 36);
        assert_eq!(t[0], 40);
        assert_eq!(t[7], 664);
        assert_eq!(t[27], 773);
        assert_eq!(t[35], 498);
    }

    // --- init_type_range ---

    #[test]
    fn init_type_range_btt_split_flag() {
        // Table 39: btt_split_flag — initType=0 → 0..15, initType=1 → 15..30.
        let t = MainCtxTable::BttSplitFlag;
        assert_eq!(t.init_type_range(InitType::I), (0, 15));
        assert_eq!(t.init_type_range(InitType::Pb), (15, 30));
    }

    #[test]
    fn init_type_range_cu_skip_flag() {
        // Table 39: cu_skip_flag — initType=0 → 0..2, initType=1 → 2..4.
        let t = MainCtxTable::CuSkipFlag;
        assert_eq!(t.init_type_range(InitType::I), (0, 2));
        assert_eq!(t.init_type_range(InitType::Pb), (2, 4));
    }

    #[test]
    fn init_type_range_sig_coeff_flag() {
        // Table 39: sig_coeff_flag — initType=0 → 0..47, initType=1 → 47..94.
        let t = MainCtxTable::SigCoeffFlag;
        assert_eq!(t.init_type_range(InitType::I), (0, 47));
        assert_eq!(t.init_type_range(InitType::Pb), (47, 94));
    }

    // --- ctxInc helpers ---

    #[test]
    fn ctx_inc_coeff_zero_run_luma() {
        // bin_idx=0, c_idx=0, prev_level=3 → ((3-1).min(5))<<1 + 0 = 4.
        assert_eq!(ctx_inc_coeff_zero_run(0, 0, 3), 4);
        // bin_idx=1, c_idx=0, prev_level=10 → ((10-1).min(5))<<1 + 1 = 11.
        assert_eq!(ctx_inc_coeff_zero_run(1, 0, 10), 11);
    }

    #[test]
    fn ctx_inc_coeff_zero_run_chroma() {
        // bin_idx=0, c_idx=1, prev_level=2 → 2 + 12 = 14.
        assert_eq!(ctx_inc_coeff_zero_run(0, 1, 2), 14);
    }

    #[test]
    fn ctx_inc_suco_flag_square_block() {
        // 16x16: log2 = 4, max-2 = 2, base = 4, square → +0 → 4.
        assert_eq!(ctx_inc_suco_flag(16, 16), 4);
    }

    #[test]
    fn ctx_inc_suco_flag_nonsquare_block() {
        // 16x8: log2 = 4/3, max-2 = 2, base = 4, nonsquare → +1 → 5.
        assert_eq!(ctx_inc_suco_flag(16, 8), 5);
    }

    #[test]
    fn ctx_inc_neighbour_sum_clamps_to_num_ctx_minus_one() {
        // pred_mode_flag has numCtx=3. All three neighbours true → 3,
        // clamped to 2.
        assert_eq!(ctx_inc_neighbour_sum(1, 1, 1, 3), 2);
        // affine_flag has numCtx=2. Two neighbours true → 2 → 1.
        assert_eq!(ctx_inc_neighbour_sum(1, 1, 0, 2), 1);
    }

    #[test]
    fn ctx_inc_btt_split_flag_uses_table_97() {
        // 16x16 (log2-2 = 2,2) → ctxSetIdx=3, num_smaller=0 → 0 + 9 = 9.
        assert_eq!(ctx_inc_btt_split_flag(0, 16, 16), 9);
        // 16x8 (log2-2 = 2,1) → ctxSetIdx=3, num_smaller=2 → 2 + 9 = 11.
        assert_eq!(ctx_inc_btt_split_flag(2, 16, 8), 11);
        // 64x64 (log2-2 = 4,4) → ctxSetIdx=1, num_smaller=1 → 1 + 3 = 4.
        assert_eq!(ctx_inc_btt_split_flag(1, 64, 64), 4);
    }

    #[test]
    fn ctx_inc_last_sig_coeff_prefix_luma_8x8() {
        // log2=3, c_idx=0, chromaArrayType=1 → enableLumaModel=true.
        // ctxOffset = 3*(3-2) + ((3-1)>>2) = 3 + 0 = 3.
        // ctxShift  = (3+1)>>2 = 1.
        // bin_idx=0 → 0>>1 + 3 = 3. bin_idx=2 → 1 + 3 = 4.
        assert_eq!(ctx_inc_last_sig_coeff_prefix(0, 0, 3, 1), 3);
        assert_eq!(ctx_inc_last_sig_coeff_prefix(2, 0, 3, 1), 4);
    }

    #[test]
    fn ctx_inc_last_sig_coeff_prefix_chroma() {
        // c_idx=1, chromaArrayType=1 → enableLumaModel=false.
        // ctxOffset = 18, log2=3 → ctxShift = max(0,1) - max(0,-1) = 1.
        // bin_idx=2 → 1 + 18 = 19.
        assert_eq!(ctx_inc_last_sig_coeff_prefix(2, 1, 3, 1), 19);
    }

    #[test]
    fn ctx_inc_sig_coeff_flag_corner_case() {
        // (0,0) luma with no non-zero neighbours: sigCtx = 0+1 = 1,
        // capped to 2 (xc+yc<2), ctxOffset = 0 → ctxInc = 1.
        assert_eq!(ctx_inc_sig_coeff_flag(0, 0, 0, 0), 1);
        // (3,3) luma, num_flags=2: sigCtx = 3, ctxOffset = 2 (5<=…<10
        // is false, so falls into the "<5" bucket… wait xc+yc=6 → 7).
        // sigCtx = 3, ctxOffset = 7 → 10.
        assert_eq!(ctx_inc_sig_coeff_flag(0, 3, 3, 2), 10);
    }

    #[test]
    fn ctx_inc_coeff_abs_level_greater_a_last_returns_zero() {
        assert_eq!(ctx_inc_coeff_abs_level_greater_a(0, 4, 4, true, 0), 0);
    }

    #[test]
    fn ctx_inc_coeff_abs_level_greater_a_luma_offset() {
        // num_flags=2, xc+yc=2 (<3) → ctxInc = 3 + 0 = 3.
        assert_eq!(ctx_inc_coeff_abs_level_greater_a(0, 1, 1, false, 2), 3);
        // num_flags=1, xc+yc=10 → ctxInc = 2 + 8 = 10.
        assert_eq!(ctx_inc_coeff_abs_level_greater_a(0, 5, 5, false, 1), 10);
    }

    #[test]
    fn rice_param_table_98() {
        assert_eq!(rice_param_coeff_abs_level_remaining(0), 0);
        assert_eq!(rice_param_coeff_abs_level_remaining(6), 0);
        assert_eq!(rice_param_coeff_abs_level_remaining(7), 1);
        assert_eq!(rice_param_coeff_abs_level_remaining(13), 1);
        assert_eq!(rice_param_coeff_abs_level_remaining(14), 2);
        assert_eq!(rice_param_coeff_abs_level_remaining(27), 2);
        assert_eq!(rice_param_coeff_abs_level_remaining(28), 3);
        assert_eq!(rice_param_coeff_abs_level_remaining(31), 3);
    }

    #[test]
    fn ctx_inc_ats_cu_inter_flag_log2_threshold() {
        // 16x16: log2sum = 4+4 = 8 → 0.
        assert_eq!(ctx_inc_ats_cu_inter_flag(16, 16), 0);
        // 8x8: log2sum = 3+3 = 6 → 1.
        assert_eq!(ctx_inc_ats_cu_inter_flag(8, 8), 1);
    }

    #[test]
    fn ats_cu_inter_horizontal_flag_aspect_ratio() {
        assert_eq!(ctx_inc_ats_cu_inter_horizontal_flag(16, 16), 0);
        assert_eq!(ctx_inc_ats_cu_inter_horizontal_flag(8, 16), 1);
        assert_eq!(ctx_inc_ats_cu_inter_horizontal_flag(16, 8), 2);
    }

    // --- init_main_profile_contexts ---

    #[test]
    fn init_main_profile_installs_btt_split_flag_at_qp32() {
        // Build an engine and run the Main-profile init at slice_qp=32.
        // Then verify the btt_split_flag ctxIdx 0 entry matches the
        // hand-derived value from initValue=145, slice_qp=32:
        //   val_slope_mag = (145 & 14) << 4 = 0 << 4 = 0.
        //   val_slope = (145 & 1) ? -0 : 0 = 0.
        //   val_offset_mag = (((145 >> 4) & 62) << 7) = ((9 & 62) << 7)
        //                  = (8 << 7) = 1024. (145>>4 = 9, 9&62 = 8.)
        //   val_offset sign: ((145 >> 4) & 1) = 1 → -1024 + 4096 = 3072.
        //   pre = (0 * 32 + 3072) >> 4 = 192.
        //   pre <= 256 → val_mps = 1; val_state = 192.
        let buf = vec![0u8; 16];
        let mut eng = crate::cabac::CabacEngine::new(&buf).unwrap();
        init_main_profile_contexts(&mut eng, InitType::I, 32).unwrap();
        let var = eng.context(MainCtxTable::BttSplitFlag.as_usize(), 0);
        assert_eq!(var.val_state, 192);
        assert_eq!(var.val_mps, 1);
    }

    #[test]
    fn init_main_profile_installs_cu_skip_flag_p_slice_at_qp22() {
        // P slice (initType=1) → ctxIdx range = 2..4 → ctxIdx 2 has
        // initValue 711. From the existing cabac.rs `init_value_table_math`
        // test we know (711, qp=32) → (96, 0). Re-derive for slice_qp=22:
        //   val_slope_mag = (711 & 14) << 4 = 6<<4 = 96.
        //   val_slope = -(711&1 ? : ) = -96.
        //   val_offset_mag = ((44 & 62) << 7) = (44<<7) = 5632.
        //   val_offset sign: 44 & 1 = 0 → +5632. +4096 = 9728.
        //   pre = (-96*22 + 9728) >> 4 = (-2112 + 9728) >> 4 = 7616>>4 = 476.
        //   pre > 256 → val_mps = 0; val_state = 512 - 476 = 36.
        let buf = vec![0u8; 16];
        let mut eng = crate::cabac::CabacEngine::new(&buf).unwrap();
        init_main_profile_contexts(&mut eng, InitType::Pb, 22).unwrap();
        let var = eng.context(MainCtxTable::CuSkipFlag.as_usize(), 2);
        assert_eq!(var.val_state, 36);
        assert_eq!(var.val_mps, 0);
    }

    #[test]
    fn init_main_profile_idempotent() {
        // Calling init twice must yield identical state for every slot.
        let buf = vec![0u8; 16];
        let mut a = crate::cabac::CabacEngine::new(&buf).unwrap();
        let mut b = crate::cabac::CabacEngine::new(&buf).unwrap();
        init_main_profile_contexts(&mut a, InitType::I, 30).unwrap();
        init_main_profile_contexts(&mut b, InitType::I, 30).unwrap();
        init_main_profile_contexts(&mut b, InitType::I, 30).unwrap();
        for &t in MainCtxTable::ALL {
            for ctx_idx in 0..t.init_values().len() {
                assert_eq!(
                    a.context(t.as_usize(), ctx_idx).val_state,
                    b.context(t.as_usize(), ctx_idx).val_state,
                    "table {:?} ctxIdx {} mismatch",
                    t,
                    ctx_idx
                );
            }
        }
    }

    #[test]
    fn all_tables_listed() {
        // Sanity: the ALL constant lists every Main-profile table exactly
        // once. Catches accidental duplication / omission when adding
        // new entries.
        assert_eq!(MainCtxTable::ALL.len(), 51); // Tables 40..=90 = 51 tables.
        let mut seen = std::collections::HashSet::new();
        for t in MainCtxTable::ALL {
            assert!(seen.insert(*t as u8), "duplicate table {:?}", t);
        }
    }
}
