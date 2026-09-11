//! **Coding-tree search + writer** shared by the I ([`crate::slice_enc`])
//! and P/B ([`crate::slice_enc_p`]) slice encoders (round 458): the
//! write-side dual of the decoder's §7.3.8.3 `split_unit()` decision
//! prefix (`resolve_split_unit`) under both SPS shapes —
//!
//! * `sps_btt_flag == 0` — the quad tree: one `split_cu_flag` per
//!   recursable in-picture block, implicit quad splits at the picture
//!   edges (the round-429 shape, unchanged bin for bin);
//! * `sps_btt_flag == 1` — the binary / ternary tree: the
//!   `btt_split_flag` / `btt_split_dir` / `btt_split_type` group
//!   (Tables 42-44, the §9.3.4.2.5 eq. 1439/1440 `numSmaller` ctxInc
//!   and the Table 95 `log2CbWidth − log2CbHeight + 2` dir ctxInc under
//!   `sps_cm_init_flag`), presence-gated by the §7.4.8.3 `allowSplit*`
//!   derivation the decoder's [`crate::split`] module owns, the
//!   §7.4.8.3 inference for absent elements, and the implicit binary
//!   split toward the in-picture side at the picture boundary. The
//!   encoder never emits a quad split under BTT (§7.3.8.3 reads no
//!   `split_cu_flag` there); four squares are two binary levels.
//!
//! ## The BTT search
//!
//! The quad search is exhaustive (every aligned square is decided once,
//! `Σ 4^d` leaf evaluations per CTU). An exhaustive BTT search is
//! exponential — each node has up to four shapes and every shape's
//! children have four more — so the search commits to a *lookahead
//! policy* that keeps the cost near the quad tree's while still
//! reaching every rectangular leaf shape:
//!
//! * a **square** node evaluates its leaf, then `SPLIT_BT_HOR` with
//!   **exact** children (each half-height rectangle recursively
//!   searched, whose orthogonal binary split reaches the four squares
//!   — the quad tree lives inside this branch), then `SPLIT_BT_VER` and
//!   both ternary shapes with **leaf-only** children;
//! * a **rectangular** node evaluates its leaf, the binary split across
//!   its long axis with exact children (back toward squares), and the
//!   same-axis binary split plus the long-axis ternary split with
//!   leaf-only children;
//! * whenever a leaf-only trial wins, its children are **re-searched
//!   exactly** and the exact cost decides against the best exact
//!   alternative — a shallow estimate never selects a tree by itself.
//!
//! Every trial is rate-distortion exact at the point of evaluation: the
//! split bins are committed to the rate model in decode order, each
//! child is decided against the committed reconstruction / side-info /
//! HMVP state, and the winning subtree's state is restored bin for bin.
//! Measured on the crate's corpus the search runs ~3× the quad tree's
//! leaf evaluations (see the CHANGELOG round-458 entry).
//!
//! All clause / equation / table numbers cite ISO/IEC 23094-1:2020(E).

use oxideav_core::Result;

use crate::bin_cost::BitCostModel;
use crate::cabac::BinSink;
use crate::cabac_init::{ctx_inc_btt_split_flag, CtxSel, MainCtxTable};
use crate::deblock::SideInfoGrid;
use crate::split::{self, AllowedSplits, BttSizeLimits, PredModeConstraint, SplitMode};

/// The encoder's `sps_btt_flag == 1` SPS geometry (§7.3.2.1): a 64×64
/// CTU, 4×4 minimum CB, and the three `log2_diff_*` fields at 0 —
/// `MaxCbLog2Size14Ratio = MaxTtLog2Size = 6`, `MinTtLog2Size = 4`
/// (eqs. 65-67), i.e. 1:4 binary children down to 4 samples and ternary
/// splits on every side from 16 to 64.
pub const BTT_LOG2_CTU_SIZE_MINUS5: u32 = 1;
pub const BTT_LOG2_MIN_CB_SIZE_MINUS2: u32 = 0;
pub const BTT_LOG2_DIFF_CTU_MAX_14_CB_SIZE: u32 = 0;
pub const BTT_LOG2_DIFF_CTU_MAX_TT_CB_SIZE: u32 = 0;
pub const BTT_LOG2_DIFF_MIN_CB_MIN_TT_CB_SIZE_MINUS2: u32 = 0;

/// The coding-tree geometry an encoder walks: picture dimensions, the
/// CTB / minimum-CB sizes, and — under `sps_btt_flag == 1` — the
/// §7.3.2.2 size limits.
#[derive(Clone, Copy, Debug)]
pub struct TreeGeometry {
    pub pic_w: u32,
    pub pic_h: u32,
    pub ctb_log2: u32,
    pub min_cb_log2: u32,
    /// `Some` selects the BTT shape.
    pub btt: Option<BttSizeLimits>,
}

impl TreeGeometry {
    /// The encoder's geometry: 64×64 CTUs, 4×4 minimum CBs (the
    /// §7.4.3.1 `sps_btt_flag == 0` defaults, and the same values the
    /// BTT SPS declares explicitly).
    pub fn encoder(pic_w: u32, pic_h: u32, btt: bool) -> Self {
        Self {
            pic_w,
            pic_h,
            ctb_log2: 5 + BTT_LOG2_CTU_SIZE_MINUS5,
            min_cb_log2: 2 + BTT_LOG2_MIN_CB_SIZE_MINUS2,
            btt: btt.then(|| {
                BttSizeLimits::derive(
                    5 + BTT_LOG2_CTU_SIZE_MINUS5,
                    BTT_LOG2_MIN_CB_SIZE_MINUS2,
                    BTT_LOG2_DIFF_CTU_MAX_14_CB_SIZE,
                    BTT_LOG2_DIFF_CTU_MAX_TT_CB_SIZE,
                    BTT_LOG2_DIFF_MIN_CB_MIN_TT_CB_SIZE_MINUS2,
                )
            }),
        }
    }

    fn within(&self, x0: u32, y0: u32, lw: u32, lh: u32) -> bool {
        x0 + (1 << lw) <= self.pic_w && y0 + (1 << lh) <= self.pic_h
    }

    /// The §7.4.8.3 `allowSplit*` set of a block (I slices and the
    /// `sps_admvp_flag == 0` P/B slices both derive
    /// `PRED_MODE_NO_CONSTRAINT`, so only the size limits apply).
    fn allowed(&self, lw: u32, lh: u32) -> AllowedSplits {
        match &self.btt {
            Some(limits) => split::derive_allowed_splits(
                limits,
                lw,
                lh,
                PredModeConstraint::NotInterConstrained,
            ),
            None => AllowedSplits {
                bt_ver: false,
                bt_hor: false,
                tt_ver: false,
                tt_hor: false,
            },
        }
    }
}

/// How a non-leaf `split_unit()` splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitKind {
    /// `split_cu_flag == 1` (or the implicit boundary quad split) —
    /// `sps_btt_flag == 0` only.
    Quad,
    /// A `sps_btt_flag == 1` shape (signalled, or the implicit boundary
    /// binary split).
    Btt(SplitMode),
}

/// One child `split_unit()` of a decided split.
pub struct Child<L> {
    pub x0: u32,
    pub y0: u32,
    pub lw: u32,
    pub lh: u32,
    pub node: TreeNode<L>,
}

/// A decided `split_unit()` subtree.
pub enum TreeNode<L> {
    Leaf(L),
    Split(SplitKind, Vec<Child<L>>),
}

/// Tallies of the tree-level syntax the emit pass wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeStats {
    /// `split_cu_flag` bins.
    pub split_cu_flag_bins: u32,
    /// `btt_split_flag` bins.
    pub btt_flag_bins: u32,
    /// `btt_split_dir` bins.
    pub btt_dir_bins: u32,
    /// `btt_split_type` bins.
    pub btt_type_bins: u32,
    /// Binary splits emitted (signalled or implicit).
    pub bt_splits: u32,
    /// Ternary splits emitted.
    pub tt_splits: u32,
}

/// What the tree search needs from a slice encoder: its geometry and
/// entropy shape, the decode-order side-info grid (the `numSmaller`
/// probes), snapshot / restore of the whole decode-order state over a
/// block, and the exact-RD leaf decision that commits that state.
pub trait TreeCoder {
    /// A decided leaf `coding_unit()`.
    type Leaf;
    /// Everything [`TreeCoder::restore`] needs to rewind a trial.
    type Snap;

    fn geometry(&self) -> &TreeGeometry;
    fn sel(&self) -> CtxSel;
    fn lambda(&self) -> f64;
    fn grid(&self) -> &SideInfoGrid;
    fn snapshot(&self, model: &BitCostModel, x0: u32, y0: u32, lw: u32, lh: u32) -> Self::Snap;
    fn restore(
        &mut self,
        model: &mut BitCostModel,
        snap: &Self::Snap,
        x0: u32,
        y0: u32,
        lw: u32,
        lh: u32,
    );
    /// Decide one leaf under `D + λ · R` at the current state, commit
    /// its reconstruction / side info / rate-model bins, and return it
    /// with its cost.
    fn decide_leaf(
        &mut self,
        model: &mut BitCostModel,
        x0: u32,
        y0: u32,
        lw: u32,
        lh: u32,
    ) -> Result<(Self::Leaf, f64)>;
}

/// The ordered child geometries of a split (the decoder's §7.3.8.3
/// recursion with `splitUnitOrder = 0`; the encoder never signals SUCO).
fn children_of(
    geom: &TreeGeometry,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    kind: SplitKind,
) -> Vec<(u32, u32, u32, u32)> {
    let list = match kind {
        SplitKind::Quad => split::quad_split_children(x0, y0, lw, lh, 0, 0, geom.pic_w, geom.pic_h),
        SplitKind::Btt(mode) => {
            split::split_unit_children(mode, x0, y0, lw, lh, 0, 0, 0, geom.pic_w, geom.pic_h)
        }
    };
    list.iter()
        .map(|c| (c.x0, c.y0, c.log2_cb_width, c.log2_cb_height))
        .collect()
}

/// §7.3.8.3 — the implicit split of a block straddling the picture
/// boundary (no bins): the quad split under `sps_btt_flag == 0`, the
/// §7.4.8.3 boundary binary split toward the in-picture side under
/// `== 1`. `None` when the block lies inside the picture (or is a 4×4
/// leaf that cannot recurse).
fn implicit_split(geom: &TreeGeometry, x0: u32, y0: u32, lw: u32, lh: u32) -> Option<SplitKind> {
    if geom.within(x0, y0, lw, lh) {
        return None;
    }
    match &geom.btt {
        None => {
            let can_recurse = lw > geom.min_cb_log2 && lh > geom.min_cb_log2;
            can_recurse.then_some(SplitKind::Quad)
        }
        Some(_) => {
            let allowed = geom.allowed(lw, lh);
            let mode = split::derive_split_mode(
                false, 0, 0, &allowed, x0, y0, lw, lh, geom.pic_w, geom.pic_h,
            );
            (mode != SplitMode::NoSplit).then_some(SplitKind::Btt(mode))
        }
    }
}

/// Whether the `split_unit()` at this block signals a split decision at
/// all (a `split_cu_flag`, or — when `allowSplit*` permits anything —
/// a `btt_split_flag`).
fn decision_present(geom: &TreeGeometry, x0: u32, y0: u32, lw: u32, lh: u32) -> bool {
    if !geom.within(x0, y0, lw, lh) || (lw <= 2 && lh <= 2) {
        return false;
    }
    match &geom.btt {
        None => lw > geom.min_cb_log2 && lh > geom.min_cb_log2,
        Some(_) => geom.allowed(lw, lh).any(),
    }
}

/// Write the §7.3.8.3 split-decision prefix of one `split_unit()` —
/// exactly the bins the decoder's `resolve_split_unit` reads — for a
/// leaf (`None`) or the given split. `grid` is the decode-order
/// side-info state (the eq. 1439 `numSmaller` neighbour probe).
#[allow(clippy::too_many_arguments)]
pub fn emit_split_syntax<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    geom: &TreeGeometry,
    grid: &SideInfoGrid,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    kind: Option<SplitKind>,
    stats: &mut TreeStats,
) {
    if !geom.within(x0, y0, lw, lh) || (lw <= 2 && lh <= 2) {
        // Implicit boundary split / 4×4 leaf: nothing signalled.
        return;
    }
    match &geom.btt {
        None => {
            if lw > geom.min_cb_log2 && lh > geom.min_cb_log2 {
                // Table 41, ctxInc 0 (Table 95); the Baseline shape
                // lands on the shared ctxTable 0.
                let (t, i) = sel.ctx(MainCtxTable::SplitCuFlag, 0);
                enc.encode_decision(t, i, u8::from(kind.is_some()));
                stats.split_cu_flag_bins += 1;
            }
        }
        Some(_) => {
            let allowed = geom.allowed(lw, lh);
            let mode = match kind {
                None => SplitMode::NoSplit,
                Some(SplitKind::Btt(m)) => m,
                Some(SplitKind::Quad) => unreachable!("no quad split under sps_btt_flag == 1"),
            };
            let num_smaller =
                crate::slice_data::btt_num_smaller_at(grid, geom.pic_w, geom.pic_h, x0, y0, lw, lh);
            emit_btt_split(enc, sel, &allowed, num_smaller, lw, lh, mode, stats);
        }
    }
}

/// The `btt_split_flag` / `btt_split_dir` / `btt_split_type` group —
/// the dual of [`crate::split::decode_btt_split`] for a block whose
/// decision is present (`within` and larger than 4×4). Elements the
/// decoder infers are not written; the chosen `mode` must then agree
/// with the §7.4.8.3 inference, which holds for every `allowed` shape.
#[allow(clippy::too_many_arguments)]
fn emit_btt_split<S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    allowed: &AllowedSplits,
    num_smaller: u32,
    lw: u32,
    lh: u32,
    mode: SplitMode,
    stats: &mut TreeStats,
) {
    if allowed.any() {
        // Table 42; eq. 1440 ctxInc under sps_cm_init_flag == 1.
        let inc = if sel.cm_init {
            ctx_inc_btt_split_flag(num_smaller, 1 << lw, 1 << lh)
        } else {
            0
        };
        let (t, i) = sel.ctx(MainCtxTable::BttSplitFlag, inc);
        enc.encode_decision(t, i, u8::from(mode != SplitMode::NoSplit));
        stats.btt_flag_bins += 1;
    } else {
        debug_assert_eq!(
            mode,
            SplitMode::NoSplit,
            "split chosen where none is allowed"
        );
    }
    let (dir, ty) = match mode {
        SplitMode::NoSplit => return,
        SplitMode::SplitBtHor => (0u32, 0u32),
        SplitMode::SplitTtHor => (0, 1),
        SplitMode::SplitBtVer => (1, 0),
        SplitMode::SplitTtVer => (1, 1),
    };
    if split::btt_split_dir_present(allowed) {
        // Table 43; Table 95: log2CbWidth − log2CbHeight + 2 (0..=4).
        let inc = if sel.cm_init {
            (lw as i32 - lh as i32 + 2).clamp(0, 4) as usize
        } else {
            0
        };
        let (t, i) = sel.ctx(MainCtxTable::BttSplitDir, inc);
        enc.encode_decision(t, i, dir as u8);
        stats.btt_dir_bins += 1;
    } else {
        debug_assert_eq!(split::infer_btt_split_dir(allowed), dir);
    }
    if split::btt_split_type_present(allowed, dir) {
        // Table 44, ctxInc 0.
        let (t, i) = sel.ctx(MainCtxTable::BttSplitType, 0);
        enc.encode_decision(t, i, ty as u8);
        stats.btt_type_bins += 1;
    } else {
        debug_assert_eq!(split::infer_btt_split_type(allowed, dir), ty);
    }
    if ty == 1 {
        stats.tt_splits += 1;
    } else {
        stats.bt_splits += 1;
    }
}

/// Decide one `split_unit()` (the CTU root, or any subtree) under RD
/// cost, committing the winning state. Returns the subtree and its
/// cost (`D + λ · R`, the split syntax included).
pub fn search_split_unit<C: TreeCoder>(
    c: &mut C,
    model: &mut BitCostModel,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
) -> Result<(TreeNode<C::Leaf>, f64)> {
    if c.geometry().btt.is_some() {
        search_btt(c, model, x0, y0, lw, lh, Depth::Full)
    } else {
        search_quad(c, model, x0, y0, lw, lh)
    }
}

/// Commit the split-decision bins of `kind` (or the leaf) to the rate
/// model at the current decode-order state; returns their cost.
fn commit_split_bins<C: TreeCoder>(
    c: &C,
    model: &mut BitCostModel,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    kind: Option<SplitKind>,
) -> f64 {
    let sel = c.sel();
    let geom = c.geometry();
    let grid = c.grid();
    let mut st = TreeStats::default();
    model.commit(|m| emit_split_syntax(m, sel, geom, grid, x0, y0, lw, lh, kind, &mut st))
}

/// The `sps_btt_flag == 0` quad search: leaf vs quad split, bottom-up,
/// exactly the round-429 procedure (the leaf is trialled first with its
/// `split_cu_flag = 0` committed ahead of it, then the split; ties keep
/// the leaf).
fn search_quad<C: TreeCoder>(
    c: &mut C,
    model: &mut BitCostModel,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
) -> Result<(TreeNode<C::Leaf>, f64)> {
    let geom = *c.geometry();
    if let Some(kind) = implicit_split(&geom, x0, y0, lw, lh) {
        let (children, cost) = search_children(c, model, x0, y0, lw, lh, kind, Depth::Full)?;
        return Ok((TreeNode::Split(kind, children), cost));
    }
    if !decision_present(&geom, x0, y0, lw, lh) {
        let (leaf, cost) = c.decide_leaf(model, x0, y0, lw, lh)?;
        return Ok((TreeNode::Leaf(leaf), cost));
    }
    let lambda = c.lambda();
    let before = c.snapshot(model, x0, y0, lw, lh);
    let leaf_flag_bits = commit_split_bins(c, model, x0, y0, lw, lh, None);
    let (leaf, leaf_cost) = c.decide_leaf(model, x0, y0, lw, lh)?;
    let leaf_cost = leaf_cost + lambda * leaf_flag_bits;
    let after_leaf = c.snapshot(model, x0, y0, lw, lh);
    c.restore(model, &before, x0, y0, lw, lh);

    let split_flag_bits = commit_split_bins(c, model, x0, y0, lw, lh, Some(SplitKind::Quad));
    let (children, split_cost) =
        search_children(c, model, x0, y0, lw, lh, SplitKind::Quad, Depth::Full)?;
    let split_cost = split_cost + lambda * split_flag_bits;
    if leaf_cost <= split_cost {
        c.restore(model, &after_leaf, x0, y0, lw, lh);
        Ok((TreeNode::Leaf(leaf), leaf_cost))
    } else {
        Ok((TreeNode::Split(SplitKind::Quad, children), split_cost))
    }
}

/// How deep a BTT trial searches its children.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Depth {
    /// The full lookahead policy (see the module doc).
    Full,
    /// Children are leaves (their own `btt_split_flag = 0` included).
    LeafOnly,
}

/// Decide the children of `kind` in decode order (the split bins are
/// already committed), returning them with their summed cost.
#[allow(clippy::too_many_arguments)]
fn search_children<C: TreeCoder>(
    c: &mut C,
    model: &mut BitCostModel,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    kind: SplitKind,
    depth: Depth,
) -> Result<(Vec<Child<C::Leaf>>, f64)> {
    let geom = *c.geometry();
    let mut out = Vec::with_capacity(4);
    let mut cost = 0f64;
    for (cx, cy, clw, clh) in children_of(&geom, x0, y0, lw, lh, kind) {
        let (node, child_cost) = match (geom.btt.is_some(), depth) {
            (false, _) => search_quad(c, model, cx, cy, clw, clh)?,
            (true, d) => search_btt(c, model, cx, cy, clw, clh, d)?,
        };
        cost += child_cost;
        out.push(Child {
            x0: cx,
            y0: cy,
            lw: clw,
            lh: clh,
            node,
        });
    }
    Ok((out, cost))
}

/// A trialled option of one `split_unit()`.
struct Trial<L, S> {
    node: TreeNode<L>,
    cost: f64,
    /// The state after the trial (`None` while the trial is the current
    /// state).
    snap: Option<S>,
}

/// The `sps_btt_flag == 1` search — the lookahead policy of the module
/// doc.
fn search_btt<C: TreeCoder>(
    c: &mut C,
    model: &mut BitCostModel,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    depth: Depth,
) -> Result<(TreeNode<C::Leaf>, f64)> {
    let geom = *c.geometry();
    if let Some(kind) = implicit_split(&geom, x0, y0, lw, lh) {
        let (children, cost) = search_children(c, model, x0, y0, lw, lh, kind, depth)?;
        return Ok((TreeNode::Split(kind, children), cost));
    }
    if !decision_present(&geom, x0, y0, lw, lh) {
        let (leaf, cost) = c.decide_leaf(model, x0, y0, lw, lh)?;
        return Ok((TreeNode::Leaf(leaf), cost));
    }
    let lambda = c.lambda();
    let allowed = geom.allowed(lw, lh);
    let before = c.snapshot(model, x0, y0, lw, lh);

    // The leaf, its btt_split_flag = 0 committed ahead of it.
    let flag_bits = commit_split_bins(c, model, x0, y0, lw, lh, None);
    let (leaf, leaf_cost) = c.decide_leaf(model, x0, y0, lw, lh)?;
    let mut best_exact = Trial {
        node: TreeNode::Leaf(leaf),
        cost: leaf_cost + lambda * flag_bits,
        snap: None,
    };
    if depth == Depth::LeafOnly {
        return Ok((best_exact.node, best_exact.cost));
    }

    // The split trials, in the emit order of their syntax values;
    // `exact` children are fully searched, the others leaf-only.
    let plan: &[(SplitMode, bool)] = if lw == lh {
        &[
            (SplitMode::SplitBtHor, true),
            (SplitMode::SplitBtVer, false),
            (SplitMode::SplitTtHor, false),
            (SplitMode::SplitTtVer, false),
        ]
    } else if lw > lh {
        &[
            (SplitMode::SplitBtVer, true),
            (SplitMode::SplitBtHor, false),
            (SplitMode::SplitTtVer, false),
        ]
    } else {
        &[
            (SplitMode::SplitBtHor, true),
            (SplitMode::SplitBtVer, false),
            (SplitMode::SplitTtHor, false),
        ]
    };
    let leaf_cost = best_exact.cost;
    let mut best_shallow: Option<Trial<C::Leaf, C::Snap>> = None;
    for &(mode, exact) in plan {
        let permitted = match mode {
            SplitMode::SplitBtHor => allowed.bt_hor,
            SplitMode::SplitBtVer => allowed.bt_ver,
            SplitMode::SplitTtHor => allowed.tt_hor,
            SplitMode::SplitTtVer => allowed.tt_ver,
            SplitMode::NoSplit => false,
        };
        if !permitted {
            continue;
        }
        // Ternary trials only where a binary split already beat the
        // leaf: a block that prefers to stay whole over both binary
        // shapes does not want three pieces either (the corpus picks a
        // ternary shape at ~1 % of the nodes, always such a node), and
        // the three leaf evaluations per trial are the search's
        // largest avoidable cost.
        let is_tt = matches!(mode, SplitMode::SplitTtHor | SplitMode::SplitTtVer);
        if is_tt
            && best_exact.cost >= leaf_cost
            && best_shallow.as_ref().map_or(true, |b| b.cost >= leaf_cost)
        {
            continue;
        }
        // Freeze the incumbent's state before rewinding.
        if best_exact.snap.is_none() {
            best_exact.snap = Some(c.snapshot(model, x0, y0, lw, lh));
        }
        if let Some(t) = best_shallow.as_mut() {
            if t.snap.is_none() {
                t.snap = Some(c.snapshot(model, x0, y0, lw, lh));
            }
        }
        c.restore(model, &before, x0, y0, lw, lh);
        let kind = SplitKind::Btt(mode);
        let bits = commit_split_bins(c, model, x0, y0, lw, lh, Some(kind));
        let child_depth = if exact { Depth::Full } else { Depth::LeafOnly };
        let (children, child_cost) = search_children(c, model, x0, y0, lw, lh, kind, child_depth)?;
        let trial = Trial {
            node: TreeNode::Split(kind, children),
            cost: child_cost + lambda * bits,
            snap: None,
        };
        if exact {
            if trial.cost < best_exact.cost {
                best_exact = trial;
            }
        } else if best_shallow.as_ref().map_or(true, |b| trial.cost < b.cost) {
            best_shallow = Some(trial);
        }
    }

    // A shallow winner earns an exact re-search; the exact cost decides.
    if let Some(shallow) = best_shallow {
        if shallow.cost < best_exact.cost {
            let TreeNode::Split(kind, _) = shallow.node else {
                unreachable!("shallow trials are splits")
            };
            if best_exact.snap.is_none() {
                best_exact.snap = Some(c.snapshot(model, x0, y0, lw, lh));
            }
            c.restore(model, &before, x0, y0, lw, lh);
            let bits = commit_split_bins(c, model, x0, y0, lw, lh, Some(kind));
            let (children, child_cost) =
                search_children(c, model, x0, y0, lw, lh, kind, Depth::Full)?;
            let cost = child_cost + lambda * bits;
            if cost < best_exact.cost {
                best_exact = Trial {
                    node: TreeNode::Split(kind, children),
                    cost,
                    snap: None,
                };
            }
        }
    }
    if let Some(snap) = &best_exact.snap {
        c.restore(model, snap, x0, y0, lw, lh);
    }
    Ok((best_exact.node, best_exact.cost))
}

/// Replay a decided tree into a bin sink in the decoder's exact read
/// order: the split-decision prefix of every `split_unit()`, then its
/// children, or `leaf_fn` for a `coding_unit()` (which also stamps the
/// emit-order `grid` the `numSmaller` probes of later nodes read).
#[allow(clippy::too_many_arguments)]
pub fn emit_tree<L, S: BinSink>(
    enc: &mut S,
    sel: CtxSel,
    geom: &TreeGeometry,
    grid: &mut SideInfoGrid,
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    node: &TreeNode<L>,
    stats: &mut TreeStats,
    leaf_fn: &mut impl FnMut(&mut S, &mut SideInfoGrid, u32, u32, u32, u32, &L),
) {
    match node {
        TreeNode::Split(kind, children) => {
            emit_split_syntax(enc, sel, geom, grid, x0, y0, lw, lh, Some(*kind), stats);
            for ch in children {
                emit_tree(
                    enc, sel, geom, grid, ch.x0, ch.y0, ch.lw, ch.lh, &ch.node, stats, leaf_fn,
                );
            }
        }
        TreeNode::Leaf(leaf) => {
            emit_split_syntax(enc, sel, geom, grid, x0, y0, lw, lh, None, stats);
            leaf_fn(enc, grid, x0, y0, lw, lh, leaf);
        }
    }
}

/// Visit every leaf of a decided tree with its geometry.
pub fn for_each_leaf<L>(
    x0: u32,
    y0: u32,
    lw: u32,
    lh: u32,
    node: &TreeNode<L>,
    f: &mut impl FnMut(u32, u32, u32, u32, &L),
) {
    match node {
        TreeNode::Split(_, children) => {
            for ch in children {
                for_each_leaf(ch.x0, ch.y0, ch.lw, ch.lh, &ch.node, f);
            }
        }
        TreeNode::Leaf(leaf) => f(x0, y0, lw, lh, leaf),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{CabacEncoder, CabacEngine, InitType};
    use crate::cabac_init::init_main_profile_contexts;

    /// Every BTT shape the encoder can choose at every block shape of
    /// the SPS geometry reads back through the decoder's
    /// `decode_btt_split` as the same `SplitMode`, on both entropy
    /// shapes, with every `numSmaller` value.
    #[test]
    fn btt_split_group_reads_back() {
        let geom = TreeGeometry::encoder(256, 256, true);
        let limits = geom.btt.unwrap();
        for &cm in &[false, true] {
            for &init in &[InitType::I, InitType::Pb] {
                let sel = CtxSel::new(cm, init);
                let mut enc = CabacEncoder::new();
                if cm {
                    enc.init_main_profile(init, 30);
                }
                let mut written: Vec<(u32, u32, u32, SplitMode)> = Vec::new();
                let mut st = TreeStats::default();
                for lw in 2..=6u32 {
                    for lh in 2..=6u32 {
                        // 4×4 signals nothing; shapes beyond 1:4 never
                        // arise under the size limits (Table 97 `na`).
                        if (lw == 2 && lh == 2) || lw.abs_diff(lh) > 2 {
                            continue;
                        }
                        let allowed = geom.allowed(lw, lh);
                        for &mode in &[
                            SplitMode::NoSplit,
                            SplitMode::SplitBtHor,
                            SplitMode::SplitBtVer,
                            SplitMode::SplitTtHor,
                            SplitMode::SplitTtVer,
                        ] {
                            let ok = match mode {
                                SplitMode::NoSplit => true,
                                SplitMode::SplitBtHor => allowed.bt_hor,
                                SplitMode::SplitBtVer => allowed.bt_ver,
                                SplitMode::SplitTtHor => allowed.tt_hor,
                                SplitMode::SplitTtVer => allowed.tt_ver,
                            };
                            if !ok {
                                continue;
                            }
                            for ns in 0..=3u32 {
                                emit_btt_split(&mut enc, sel, &allowed, ns, lw, lh, mode, &mut st);
                                written.push((lw, lh, ns, mode));
                            }
                        }
                    }
                }
                enc.encode_terminate(true);
                let bytes = enc.finish();
                let mut eng = CabacEngine::new(&bytes).unwrap();
                if cm {
                    init_main_profile_contexts(&mut eng, init, 30).unwrap();
                }
                let mut dstats = split::BttSplitStats::default();
                for &(lw, lh, ns, mode) in &written {
                    let allowed = split::derive_allowed_splits(
                        &limits,
                        lw,
                        lh,
                        PredModeConstraint::NotInterConstrained,
                    );
                    let got = split::decode_btt_split(
                        &mut eng,
                        &allowed,
                        sel,
                        ns,
                        0,
                        0,
                        lw,
                        lh,
                        256,
                        256,
                        &mut dstats,
                    )
                    .unwrap();
                    assert_eq!(got.mode, mode, "cm{cm} {init:?} {lw}x{lh} ns{ns}");
                }
                assert!(eng.decode_terminate().unwrap());
                assert_eq!(st.btt_flag_bins, dstats.flag_bins);
                assert_eq!(st.btt_dir_bins, dstats.dir_bins);
                assert_eq!(st.btt_type_bins, dstats.type_bins);
            }
        }
    }

    /// The encoder's SPS geometry permits what the module doc claims:
    /// ternary splits from 16 to 64, 1:4 binary children down to 4.
    #[test]
    fn encoder_geometry_limits() {
        let geom = TreeGeometry::encoder(64, 64, true);
        let a64 = geom.allowed(6, 6);
        assert!(a64.bt_hor && a64.bt_ver && a64.tt_hor && a64.tt_ver);
        let a8 = geom.allowed(3, 3);
        assert!(a8.bt_hor && a8.bt_ver && !a8.tt_hor && !a8.tt_ver);
        let a16x4 = geom.allowed(4, 2);
        assert!(!a16x4.bt_hor && a16x4.bt_ver, "16x4: only the width halves");
        assert!(!geom.allowed(2, 2).any());
        // Boundary: a 64×64 CTU with the right edge outside splits
        // vertically toward the picture.
        let geom = TreeGeometry::encoder(96, 64, true);
        assert_eq!(
            implicit_split(&geom, 64, 0, 6, 6),
            Some(SplitKind::Btt(SplitMode::SplitBtVer))
        );
        assert_eq!(
            children_of(&geom, 64, 0, 6, 6, SplitKind::Btt(SplitMode::SplitBtVer)).len(),
            1
        );
        assert!(implicit_split(&geom, 0, 0, 6, 6).is_none());
        // Quad shape: the same CTU splits implicitly into the in-picture quadrants.
        let geom = TreeGeometry::encoder(96, 64, false);
        assert_eq!(implicit_split(&geom, 64, 0, 6, 6), Some(SplitKind::Quad));
        assert_eq!(children_of(&geom, 64, 0, 6, 6, SplitKind::Quad).len(), 2);
    }
}
