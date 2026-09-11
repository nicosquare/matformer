# Sampling and resume contract

Schema-3 recipe preserves PINNED_COMMON/PINNED_DATA and original five arm
representation/history/clipping values. Each arm sets fixed_global and
`global_sampling_distribution: {g250: .12, g500: .16, g750: .24, g1000: .48}`.
Omit raw global_sampling_schedule/global_sampling_interval_steps because these
are uniform-mode inputs; fixed_global resolves random replacement and cadence 1.
C3 eligibility admits this mode with all original topology/ownership constraints.

The sampling section records `mode`, `probabilities`, `action_seed`, and for IM
only versioned policy identity, ordered widths, membership counts, global scope,
replacement=true, interval_steps=1, inverse_probability_loss_weighting=false.
No correction field is synthesized for IM. Strict IM validation checks these
against resolved controls and required constants; old contracts remain unchanged.

Ownership fixed-global selection uses compact global_sampling_state, preserving
existing uniform keys and adding explicit fixed policy/distribution metadata.
Legacy non-campaign fixed-global checkpoints still have no compact state.
Per-step action records retain sampled_probability. Selection calls the existing
isolated weighted categorical primitive. The normal complete-update commit
advances exposure counts and window/clock once. Every C3 active prefix plus
common steps once, independently clips at 1; inactive owners do not step.

Checkpoint save validates policy state; exact restore stages the whole contract,
model/history/clock/data/RNG bundle. Replay uses seeded weighted categorical draws
for IM and the original randrange for uniform, checks counts, last width and RNG
state against the committed ordinal before live mutation. All four epochs retain
one continuous action/data/optimizer clock. Wider/narrower real-model and malformed
resume/failure tests are mandatory. No partially completed C3 update is resumable.
