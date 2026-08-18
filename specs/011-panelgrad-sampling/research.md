# Research: PanelGrad Sampling

## Decision 1: Add PanelGrad as an explicit global adaptive strategy

- **Decision**: Resolve PanelGrad as `model.granularity_sampling_mode: adaptive_global` plus `model.adaptive_sampler_strategy: panelgrad`. Store its inputs in a distinct `model.panelgrad` mapping and identify artifacts as `panelgrad_gradient_rms` by default or `panelgrad_gradient_l2` when raw L2 importance is selected, with `method_version: 1` and `scope: global`.
- **Rationale**: `adaptive_global` already selects one complete global granularity under `nested-random`. A distinct strategy preserves current mode compatibility while preventing PanelGrad state from being interpreted as a Bayesian posterior or legacy UCB statistics.
- **Alternatives considered**:
  - Add a new sampling mode. Rejected because it duplicates the established global adaptive scope and expands compatibility branching.
  - Reuse `model.adaptive_controller`. Rejected because that mapping and state are Bayesian-specific and would obscure the different PanelGrad lifecycle.
  - Introduce a generic policy registry. Rejected because only one new method is in scope and direct dispatch is easier to inspect.

## Decision 2: Reuse the four heldout data roles

- **Decision**: Extend the existing controller-data predicate to include PanelGrad while leaving Bayesian-controller construction TS-only. Reuse the deterministic 128-example controller manifest, optimizer-training role, ordinary validation, reserved 512-example final holdout, role hashes, and resume checks.
- **Rationale**: The existing split already provides the intended statistical separation and saved provenance. A method-neutral controller-role predicate avoids duplicating partition logic without merging PanelGrad and TS controllers.
- **Alternatives considered**:
  - Reuse ordinary validation. Rejected because sampling decisions would contaminate checkpoint-selection data.
  - Use the final holdout. Rejected because it must remain untouched during training.
  - Create a fifth role. Rejected because the existing controller role has exactly the required purpose.

## Decision 3: Differentiate one aggregate controller objective

- **Decision**: For each granularity, calculate the gradient of

  $$
  L_g=\frac{\sum_b n_b L_{g,b}}{\sum_b n_b},
  $$

  where `n_b` is the valid causal-target count. Accumulate the weighted backward contributions across the complete fixed controller set, then calculate one norm and RMS. Run in evaluation mode with autograd enabled.
- **Rationale**: This produces the gradient of the same target-token-weighted heldout objective regardless of controller microbatch partitioning. Averaging batch or example gradient norms defines a different, partition-sensitive statistic.
- **Alternatives considered**:
  - Average microbatch RMS values. Rejected because norm and averaging do not commute.
  - Use one rotating controller batch. Rejected because scores would no longer be contemporaneous on common data.
  - Reuse the no-gradient validation evaluator unchanged. Rejected because it deliberately detaches losses.

## Decision 4: Define support from resolved FFN granularities

- **Decision**: Add explicit controlled-support/count helpers to both MatFormer FFN variants. For slicing, include selected gate/up rows, gate/up biases, and down-projection columns. For concat, include the selected gate/up/down blocks and selected gate/up bias blocks. Exclude shared down bias, embeddings, attention, and all other granularity-independent parameters. Count unique trainable storage once and keep zero-valued selected gradients in `N_g`.
- **Rationale**: The numerator and denominator must describe the same resolved granularity-controlled parameters. Existing whole-model counts include unrelated parameters, while current FFN `prefix_parameter_count` also includes the shared down bias.
- **Alternatives considered**:
  - Count every active model parameter. Rejected by the clarified scientific definition.
  - Count only nonzero or materialized gradients. Rejected because `N_g` would depend on controller data and numerical sparsity.
  - Infer support only from parameter names. Rejected because slicing and concat store the same conceptual prefixes differently.

## Decision 5: Suspend membership correction during measurement

- **Decision**: Make existing slicing and concat gradient hooks consult a scoped correction-suspension flag. PanelGrad enters that context only for controller backward passes and restores the flag in `finally`.
- **Rationale**: Parameter hooks execute for both `backward` and `autograd.grad`; toggling the construction-time enable field after hooks are registered is insufficient. A narrow runtime flag produces the clarified raw gradient without changing ordinary training correction.
- **Alternatives considered**:
  - Divide corrected gradients by saved scales. Rejected because it couples the measurement to correction configuration and hook ordering and introduces avoidable numerical error.
  - Remove and re-register hooks. Rejected because hook handles are not retained and failure recovery would be fragile.
  - Disable correction for the whole run. Rejected because it would change the training baseline.

## Decision 6: Use replicated controller batches for distributed measurement

- **Decision**: In a PanelGrad FSDP run, every rank iterates the same fixed controller examples in the same batches. Each backward contribution is scaled by `n_b/N`; FSDP averaging of identical rank contributions preserves the desired aggregate gradient. Do not use `no_sync`. Under the required `use_orig_params: true`, summon one wrapped decoder layer at a time with gradients, calculate the exact controlled FFN support norm in float64, and release it before the next layer. Never summon the full model.
- **Rationale**: The current non-padding validation sampler can give ranks unequal controller batch counts, which is unsafe for synchronized backward collectives. Replication is the smallest correct first implementation for a 128-example panel. Per-layer full-parameter access avoids relying on sharded one-dimensional original-parameter views and bounds peak materialization.
- **Alternatives considered**:
  - Reuse the sharded validation sampler directly. Rejected because unequal backward counts can hang and local gradient norms do not form the gradient of the global aggregate loss.
  - Add padded dummy controller rounds. Rejected for the first exploration because masked dummy losses and weighting add substantial failure surface.
  - Summon the full model with gradients. Rejected because it can exceed memory at target model sizes.
  - Average rank-local norms. Rejected because the norm of averaged gradients is not the average of local norms.

## Decision 7: Use stable float64 probability construction and rank-zero sampling

- **Decision**: Compute scores and probabilities in CPU float64. Evaluate the specified power normalization stably through shifted log values, then mix with the uniform floor:

  $$
  q_g\propto\exp\left(\frac{\log(I_g+\eta)}{T}\right),
  \qquad
  p_g=(1-\epsilon)q_g+\epsilon/K.
  $$

  Rank zero draws with `torch.multinomial` and a dedicated CPU `torch.Generator` seeded from `panelgrad_sampling`, then broadcasts the action and validated state.
- **Rationale**: The log form is mathematically equivalent but avoids overflow for small temperatures or disparate scores. A controller-local generator makes exact sampling continuation independent of global training and dataloader randomness.
- **Alternatives considered**:
  - Use softmax of raw scores. Rejected because it is not the specified proportional policy and changes under score rescaling.
  - Use the existing random-global stream. Rejected because unrelated sampling would alter PanelGrad continuation.
  - Sample independently on every rank. Rejected because RNG or numeric drift could produce different training actions.

## Decision 8: Keep refresh and training-step state explicit

- **Decision**: Use phases `initial_refresh_pending`, `active_interval`, `refresh_pending`, `terminal_partial`, `terminal_complete`, and `failed`. Immediately before action selection, refresh if pending and draw one global action. After a successful optimizer/scheduler commit, increment exposure and interval progress; at progress `H`, mark `refresh_pending`. If training ends there, record `terminal_complete` without an unused refresh; an earlier stop records `terminal_partial`. Snapshot PanelGrad state and generator state with the existing optimizer-window rollback.
- **Rationale**: This realizes boundaries at 0, H, 2H while keeping one categorical draw per training step. Post-step refresh callbacks would obscure the policy decision point and perform needless work at terminal boundaries. Commit-only progress prevents failed optimizer attempts from consuming exposure or categorical randomness.
- **Alternatives considered**:
  - Hold one action for the interval. Rejected because that is TS window behavior; PanelGrad freezes `p`, not the action.
  - Refresh after the Hth step. Rejected because training may already be complete and because selection logic becomes split across pre/post-step paths.
  - Infer phase only from step modulo H. Rejected because warmup and exact-boundary resume require explicit state.

## Decision 9: Persist a dedicated PanelGrad schema

- **Decision**: Add `panelgrad_state` beside, never inside, `probabilistic_controller_state` and legacy adaptive state. Persist method/config identity, role hashes, support counts, phase and interval progress, last complete measurement, `q/p`, generator state/sample count, exposures, last action/probability, journal commit state, and resume/failure provenance. Validate all fields against resolved config before restoring.
- **Rationale**: Explicit checkpoint enumeration is already required by the repository. A distinct schema prevents incompatible state coercion and provides enough information to reproduce refreshes and actions at inside-interval and exact-boundary checkpoints.
- **Alternatives considered**:
  - Store only `p` and reconstruct counters/RNG. Rejected because subsequent actions and boundaries would diverge.
  - Reuse Bayesian controller phases. Rejected because posterior observations and fixed action windows do not exist in PanelGrad.
  - Reinitialize missing state. Rejected because it silently changes the experiment.

## Decision 10: Reuse controller artifact filenames with new event schemas

- **Decision**: Reuse `controller_metrics.jsonl` and `controller_summary.json`, because one run selects one adaptive method. PanelGrad events carry explicit family/version and contain full refresh measurements/probabilities/cost. Ordinary metrics add only current sampled granularity/probability, exposure counts, interval progress, phase, and refresh index. The run summary links hashes and cumulative measurement cost.
- **Rationale**: Existing durable JSONL, rank-zero writes, run-summary linkage, and reporting discovery are useful and method-neutral at the file level. Explicit provenance prevents PanelGrad events from being parsed as Bayesian windows while avoiding another parallel artifact stack.
- **Alternatives considered**:
  - Put full vectors on every metrics row. Rejected because it duplicates frozen state and bloats CSV output.
  - Use Bayesian completed-window fields. Rejected because PanelGrad has neither reward nor posterior.
  - Create generic adaptive-event abstractions. Rejected because direct method-keyed validation is clearer for two different lifecycles.

## Decision 11: Preserve warmup without TS resets or acquisition episodes

- **Decision**: Permit the existing balanced-global pre-nested warmup for PanelGrad and default its action interval from PanelGrad `refresh_interval_steps`. Warmup uses the existing forced global schedule and checkpoint state; PanelGrad remains `initial_refresh_pending`, records no exposure, and refreshes only if training continues after warmup. TS reset/acquisition logic is never invoked.
- **Rationale**: Warmup already provides deterministic coverage and exact resume. Reusing its schedule matches the user's integration goal, while keeping PanelGrad state untouched preserves a clear first adaptive boundary.
- **Alternatives considered**:
  - Add PanelGrad-specific forced acquisition episodes. Rejected because every refresh already measures all granularities.
  - Count warmup steps in the first interval. Rejected because those actions were not sampled from PanelGrad.

## Decision 12: Validate as one opt-in research method

- **Decision**: Add focused unit, smoke, resume, distributed, artifact, and compatibility tests plus one opt-in config. Compare against uniform global sampling at matched optimizer steps or target tokens and report PanelGrad measurement work separately. Do not add PanelGrad to default pilot queues.
- **Rationale**: Controlled tests establish method correctness; the matched baseline establishes experimental interpretability without implying that PanelGrad is compute-matched or superior.
- **Alternatives considered**:
  - Add EMA, EXP3, inverse weighting, or per-block variants in the same implementation. Rejected as outside the single-method scope.
  - Automatically expand default comparison queues. Rejected because it changes experiment cost without an explicit request.
