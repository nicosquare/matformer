# Granularity-Conditioned Parameter Sign Dynamics

## Scientific question

This diagnostic tests whether optimizer updates selected under different
elastic granularities produce different parameter-sign dynamics, and whether
those differences are durable rather than repeated crossings close to zero.
It is an exact measurement campaign, not a sampled proxy or a controller
signal.

Let the configured global granularities be ordered from narrowest to widest,

$$g_1 < g_2 < \cdots < g_K,$$

and let $A_k$ contain every trainable parameter coordinate used by the
subnetwork at $g_k$. The primary scientific partition is the literal nested
increment:

$$B_1=A_1, \qquad B_j=A_j\setminus A_{j-1}\quad(j\ge2).$$

Every action-to-band result is reported against $B_1,\ldots,B_K$. For audit
purposes, the runtime also retains non-overlapping strata

$$C_0=\text{granularity-independent shared coordinates},$$

$$C_1=\text{controlled FFN coordinates used by }g_1,$$

$$C_j=\text{controlled FFN coordinates added by }g_j\quad(j\ge2),$$

so that $B_1=C_0\cup C_1$ and $B_j=C_j$ for $j>1$. These component strata do
not replace the requested $B$ partition.

## Live support accounting

Parameter counts are never hard-coded. Before training or checkpoint loading,
the runtime resolves the configured prefixes against the live model. It walks
every MatFormer FFN layer and projection, including the complete ×16 FFN
contribution in the TinyStories-Instruct geometry, then assigns every other
trainable coordinate to the shared stratum. Slicing and concat layouts are
both supported.

`sign_dynamics_support.json` maps every named trainable parameter or exact
flattened slice to its primary band, audit stratum, layer, and parameter
family. Its coverage contract requires:

- every trainable coordinate appears exactly once;
- the slices are pairwise disjoint;
- the union equals the full trainable model; and
- every configured FFN increment is nonempty and ordered by the resolved
  prefixes.

Consequently there is no fixed “87% shared” interpretation. The shared/FFN
ratio, layer contribution, and projection contribution are facts of the
resolved model and are recorded in the support manifest.

## Exact committed-step measurement

The diagnostic is opt-in at `evaluation.sign_dynamics`. Its fixed semantics
are all trainable parameters, cadence one, and sufficient-state retention.
After gradient accumulation and clipping but before the optimizer update, the
runtime measures gradient presence, nonzero-gradient support, and gradient RMS
over every band and audit stratum. After the optimizer and any concat LMC
correction have completed, it compares every coordinate with its previous
committed FP32 value.

For $\Delta\theta_i^t=\theta_i^{t+1}-\theta_i^t$, a strict raw crossing is

$$
I_{i,t}^{\mathrm{raw}}
=
\mathbf 1[(\theta_i^t>0\land\theta_i^{t+1}<0)
\lor(\theta_i^t<0\land\theta_i^{t+1}>0)].
$$

Each band and stratum records:

- coordinate and changed-coordinate counts;
- update RMS, prior-parameter RMS, and relative-update RMS;
- gradient-present and nonzero-gradient counts plus gradient RMS;
- strict raw sign-flip count and rate;
- dead-band occupancy, entries, and exits;
- hysteretic establishments and robust transitions for every threshold; and
- the last robust-transition step and responsible global granularity.

Forward support, nonzero-gradient support, and observed update support remain
distinct. In particular, slicing with shared AdamW state may move $B_j$ for
$j>k$ through weight decay or retained moments after an action $g_k$. Concat
inactive blocks normally have `grad is None` and remain unchanged. These are
scientific observations, not support-resolution errors.

## Hysteresis and normalization

For a configured threshold $\tau$, each coordinate has state
$q_i^t\in\{-1,0,+1\}$. The normalized margin uses the RMS of its own atomic
support stratum: one shared tensor, or one layer/projection/FFN-band slice.

$$z_i^t=\theta_i^t/(\operatorname{RMS}(\theta_{r(i)}^t)+\epsilon).$$

The state changes to $+1$ at $z\ge\tau$, to $-1$ at $z\le-\tau$, and otherwise
retains its prior value. A coordinate initialized inside the dead band remains
unknown until its first establishment. Only a transition between established
opposite states is a robust flip. The default thresholds are
$\{0,10^{-3},10^{-2}\}$, and dead-band coverage is always reported alongside
robust results.

## Retention, durability, and resume

The diagnostic writes:

- `sign_dynamics_support.json`, the exhaustive support and contract identity;
- `sign_dynamics.jsonl`, exactly one compact aggregate record per committed
  optimizer step; and
- `sign_dynamics_snapshots/step-*.pt`, atomic, hashed FP32 full-parameter
  snapshots at the configured trajectory milestones and warmup completion.

The resumable training checkpoint holds the previous committed FP32 values,
per-coordinate hysteretic states, establishment and last-transition metadata,
responsible actions, aggregate runtime cost, snapshot identities, and the
committed JSONL byte offset/hash. It does not retain a full per-parameter
trajectory or a per-step bitset.

Resume validates the measurement contract, live support identity, complete
coordinate-state shapes, journal prefix, contiguous step coverage, and every
historical milestone snapshot before loading model weights. A JSONL tail newer
than the durable checkpoint is truncated to the saved byte offset. A missing
step, changed prefix, incompatible state, or missing snapshot aborts resume.

Only a successful optimizer return is a commit. Pre-commit failures produce no
diagnostic event. A diagnostic failure after the optimizer is fatal and cannot
overwrite the preceding durable checkpoint.

## Scope and interpretation

The first campaign accepts one global elastic action per optimizer window:
uniform, balanced, fixed categorical, or global Thompson sampling. It rejects
standalone, nested-all, per-block, adaptive-per-block, and distributed
execution. The campaign uses shared optimizer state and BF16 training by
default, with ordinary validation retained. No standalone reference, portfolio
target, catch-up qualifier, or sealed final-holdout evaluation is part of this
experiment.

Analysis conditions each result on action, band, and training-time bin. It
reports raw and robust heatmaps, normalized action effects, gradient-versus-
update off-support behavior, time trajectories, terminal alignment,
last-transition curves, terminal-sign acquisition, and position within held
$H$ windows. Multi-seed panels use seed-level uncertainty; a one-seed
exploratory report uses held-window blocks and is labeled provisional.

Runtime and storage costs are recorded as metadata in `run_summary.json`.
There is no runtime A/B benchmark, sampled-step fallback, deterministic-panel
fallback, or overhead gate before exact measurement.
