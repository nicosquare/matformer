# Subnet-aware training: S1, S2, standalone results, and CaLR for ElasticNN

Date: 2026-10-01. This is a discussion and design note. CaLR has not been implemented or evaluated here; this note does not launch experiments.

## 1. Paper and relevance

Jeon et al., **Subnet-Aware Dynamic Supernet Training for Neural Architecture Search**, CVPR 2025, pp. 30137–30146.

- [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeon_Subnet-Aware_Dynamic_Supernet_Training_for_Neural_Architecture_Search_CVPR_2025_paper.pdf)
- [Readable author preprint, including supplement](https://arxiv.org/html/2503.10740v1)
- [Authors' implementation](https://github.com/cvlab-yonsei/DYNAS)

The paper identifies two problems in shared-weight supernet training: complexity-dependent convergence and noisy shared momentum. Its **complexity-aware learning-rate scheduler (CaLR)** decays LR faster for smaller subnets and slower for larger ones. Its **momentum separation (MS)** groups subnets by the operation at a chosen layer/edge and maintains a momentum buffer per group, with shared weights. Complexity is measured by parameter count, and the polynomial decay exponent is an affine function of log complexity. Experiments use momentum SGD on CNN search spaces, evaluating subnet ranking and searched architecture accuracy. These results motivate an ElasticNN hypothesis; they do not establish an improvement in language-model perplexity. See the [method and experiments](https://arxiv.org/html/2503.10740v1).

Our adaptation would use width as the optimizer context and directly evaluate the quality of all deployable widths.

## 2. How S1 and S2 are implemented today

The TinyStories model has hidden size 64, four layers, four attention heads, vocabulary 2048, context length 128, and full FFN dimension 256. The linear widths g250/g500/g750/g1000 activate FFN prefixes of 64/128/192/256 channels in every layer. Embeddings, attention, normalization, and the output head remain common across widths.

Both S1 and S2 use the slicing representation (`ModifiedLlamaMLP`): one full-sized parameter tensor per FFN projection, with a prefix used in the forward pass. Neither has separate model weights per width. Both sample one global width uniformly with replacement per update, backpropagate its loss, apply one global L2 gradient clip of 1, and update through AdamW. Membership corrections are disabled in these controls.

| Property | S1 | S2 |
| --- | --- | --- |
| Model weights | One shared model | The same one-model arrangement |
| Optimizer state scope | `shared` | `per_granularity` |
| AdamW optimizers | One | One per width, all referencing the same parameter objects |
| First/second moments and AdamW counters | Shared across widths | Separate per width |
| Optimizer stepped on an update | Shared optimizer | Only the selected width's optimizer |
| Base LR schedule | Global cosine | Global cosine, synchronized across width optimizers |

For example, a g250 update in S1 changes both the shared weights and the optimizer history subsequently used by g1000. In S2 it changes the shared weights and g250's history; g1000's history remains unchanged until g1000 is selected. S2 separates optimization histories, not the objectives or weights.

AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8`, and weight decay `0.1`. The original controls warm up for 64 global updates to LR 0.008 and then decay to zero with cosine scheduling over 348,528 updates. S2's AdamW counters advance on selected-width updates, but its LR clock advances on every global update. These are different clocks with different purposes.

### Important slicing detail

An inactive slice is not necessarily frozen. Backpropagation produces full-shaped FFN gradient tensors with zero entries outside the selected prefix, rather than separate inactive parameters with `grad=None`. AdamW can therefore apply weight decay to the full tensor. In S1, previously populated moments can also move inactive entries. In S2, the selected width has its own moments; for a fixed-width FFN tail that never receives a gradient in that history, weight decay can still move those entries.

Consequently, “update the active prefix” describes gradient support, not a guarantee that only prefix entries mutate. A CaLR adaptation should preserve this existing behavior initially, so a scheduler experiment does not also change inactivity semantics.

### Source locations

- [Slicing FFN implementation](../src/models/ffn.py): `ModifiedLlamaMLP`, its forward path, and physical parameter metadata.
- [Optimizer construction and update ordering](../src/training/steps.py): `build_optimizer_and_scheduler` and `train_for_steps`.
- [Per-width state and global clock](../src/training/optimizer_state.py): `PerGranularityOptimizerCollection`, `GlobalSchedulerClock`, and `build_per_granularity_optimizer_runtime`.
- [Original campaign controls](../configs/controlled_exps/tinystories_instruct_optimizer_ownership.yaml).

## 3. Measured performance against standalone training

These are saved terminal ordinary-validation losses, rounded to six decimals, at seed 42. Lower is better. They are not best-checkpoint or sealed-holdout results. No new model evaluation was run for this note.

Each elastic run trained for 348,528 updates / 2,855,141,376 packed tokens. Each standalone trained for 87,132 updates / 713,785,344 packed tokens. Thus one elastic run matches the aggregate token budget of four standalones, while each selected width receives approximately one quarter of the elastic updates. Matching selection counts does not guarantee identical training examples or shared-weight exposure.

### Linear grid: original matched controls

All rows below use peak LR 0.008 and 64 warmup updates.

| Width | Standalone | S1 | S2 | S1 − standalone | S2 − standalone |
| --- | ---: | ---: | ---: | ---: | ---: |
| g250 | 2.062977 | 2.068157 | 2.067589 | +0.005180 | +0.004612 |
| g500 | 1.991942 | 1.995913 | 1.991578 | +0.003972 | −0.000364 |
| g750 | 1.943363 | 1.966166 | 1.963845 | +0.022803 | +0.020482 |
| g1000 | 1.909057 | 1.953872 | 1.951732 | +0.044816 | +0.042676 |

S2 is slightly better than S1 at every linear width under these matched controls. At g1000 its remaining gap corresponds to approximately 4.36% higher perplexity than standalone, using `exp(loss_gap) − 1`. The small S2 benefit does not remove the larger-width gap, and one seed does not establish statistical reliability.

### Linear grid: completed peak-LR and warmup follow-ups

The first six elastic rows use 64 warmup updates; the final two use 256. Standalone references remain at LR 0.008 / warmup 64.

| Method | Peak LR | Warmup | g250 | g500 | g750 | g1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | 0.008 | 64 | 2.062977 | 1.991942 | 1.943363 | 1.909057 |
| S1 | 0.004 | 64 | 2.062966 | 1.991542 | 1.962110 | 1.951018 |
| S1 | 0.008 | 64 | 2.068157 | 1.995913 | 1.966166 | 1.953872 |
| S1 | 0.012 | 64 | 2.080816 | 2.007175 | 1.977576 | 1.963961 |
| S2 | 0.004 | 64 | 2.069990 | 1.998231 | 1.967343 | 1.955868 |
| S2 | 0.008 | 64 | 2.067589 | 1.991578 | 1.963845 | 1.951732 |
| S2 | 0.012 | 64 | 2.072973 | 2.000423 | 1.972370 | 1.961619 |
| S1 | 0.008 | 256 | 2.079949 | 2.009504 | 1.979595 | 1.966711 |
| S2 | 0.008 | 256 | 2.075300 | 2.000311 | 1.971466 | 1.959512 |

Across these tested settings, **S1 at LR 0.004 / warmup 64 has the lowest linear loss at every width**, including slightly lower losses than S2 at LR 0.008. It essentially matches standalone at g250/g500 but still trails at g750/g1000. Its g1000 gap is +0.041961 loss. Increasing warmup to 256 worsened both methods at every linear width at LR 0.008.

This qualifies the earlier discussion recommendation: S2 is the closer architectural match to MS, but it is not an empirically established default winner. Optimizer history and scheduler settings interact. We should retain the tuned S1 reference when assessing a paper-inspired S2 candidate.

### Historical geometric results, for context

The completed original controls used the same LR 0.008 / warmup 64:

| Width | Standalone | S1 | S2 |
| --- | ---: | ---: | ---: |
| g125 | 2.105963 | 2.128560 | 2.126497 |
| g250 | 2.062977 | 2.074250 | 2.071297 |
| g500 | 1.991942 | 2.031311 | 2.029452 |
| g1000 | 1.909057 | 1.996195 | 1.996280 |

Here S2 slightly trails S1 at g1000. In the completed LR sweep, geometric S2 at LR 0.004 reached g1000 loss 1.985714, still above standalone. These are historical results only. They do not imply restarting the geometric runs stopped at the user's request; the proposed follow-up below is linear only.

## 4. Proposed CaLR adaptation

### Separate the two decisions

S1 versus S2 chooses the optimizer history. CaLR chooses the LR schedule. Either optimizer arrangement can use CaLR:

| Configuration | Interpretation |
| --- | --- |
| S1 + cosine | Existing shared-state baseline |
| S2 + cosine | Existing width-specific AdamW histories |
| S1 + CaLR | Width-conditioned LR with shared history |
| S2 + CaLR | Width-conditioned LR with width-specific history |

S2 is an AdamW analogue of momentum separation: it separates both moments and counters, rather than only an SGD momentum buffer. With four global widths, one optimizer context per width is simple and already implemented.

### Proposed schedule and complexity definition

Preserve warmup, then use the selected width's polynomial schedule:

\[
u(t)=\operatorname{clip}\left(\frac{t-W}{T-W},0,1\right),\qquad
\eta_w(t)=\eta_{\mathrm{peak}}(1-u(t))^{\gamma_w}.
\]

Here `t` is the global scheduler position before the update, `W` is the warmup length, and `T` is the total update budget. This warmup-preserving form is our adaptation. It should retain the existing first-update and scheduler-step indexing convention rather than introducing an off-by-one change.

Use the following log-normalized mapping:

\[
\gamma_w=\gamma_{\max}-(\gamma_{\max}-\gamma_{\min})
\frac{\log(C_w/C_{\min})}{\log(C_{\max}/C_{\min})}.
\]

For the first proposal, define `C_w` as the number of active trainable parameters, including common parameters. It must count the selected network's active FFN entries, not the full allocated slicing tensors, which have identical storage at every width.

| Linear width | Active FFN dimension | Active non-embedding parameters | Active parameters including embeddings/head |
| --- | ---: | ---: | ---: |
| g250 | 64 | 115,264 | 377,408 |
| g500 | 128 | 164,416 | 426,560 |
| g750 | 192 | 213,568 | 475,712 |
| g1000 | 256 | 262,720 | 524,864 |

Using non-embedding parameter counts instead would be a separate declared variant. It changes the intermediate exponents even though the endpoint exponents remain fixed by normalization. Do not substitute width fractions or membership frequencies silently.

As an illustrative mild setting, `gamma_min=0.5` and `gamma_max=2` give g250 LR 25% of peak and g1000 LR about 71% of peak halfway through decay, versus 50% for cosine. These values are a proposed starting point, not validated ElasticNN hyperparameters. Intermediate widths use the mapping above; they need not all have exponents below one.

Apply the selected width's LR to every parameter group of the optimizer being stepped, including common parameters. Preserve the current slicing/AdamW tail behavior. Keep uniform width sampling and one global scheduler tick per successful update; do not replace global time with each width's selection count.

CaLR also changes the LR integrated over training and AdamW's cumulative weight decay. Any observed effect therefore cannot automatically be attributed solely to larger-width convergence or reduced gradient interference.

### How this differs from C4 correction runs

Our C4 LMC-only arm multiplies the selected block's LR by a fixed correction factor, leaves common parameters at the nominal rate, and does not step earlier FFN blocks. CaLR varies the schedule shape with the selected width and global time. The proposed S1/S2 adaptation retains the entire active prefix's gradient update; it does not inherit C4's selected-block-only stepping policy. No GMC/LMC correction should be added to the initial CaLR comparison.

### Implementation implications, without implementing them here

Current S2 explicitly requires all width optimizers to have identical base LRs. `GlobalSchedulerClock` synchronizes those rates, and update/checkpoint checks enforce equality. CaLR therefore needs an explicit schedule contract; adding a config label alone is insufficient.

One compatible approach is to retain synchronized nominal rates outside an update, temporarily apply the selected width's effective LR for its optimizer step, then restore nominal rates before the global clock advances. Another is to extend the clock to compute per-width rates and revise the equality checks. Either approach must validate the actually applied LR, log width/exponent/global position, and checkpoint the schedule definition plus all S2 histories for exact continuation. S1 needs the same selected-width schedule calculation, with one optimizer instead of a collection.

## 5. Recommended comparison and success criteria

For causal interpretation, use a matched linear-grid 2×2 comparison at one fixed peak LR and warmup:

| Schedule | Shared state, S1 | Per-width state, S2 |
| --- | --- | --- |
| Uniform polynomial, every exponent = 1 | Schedule control | Schedule control + separate histories |
| Width-dependent polynomial, CaLR | CaLR alone | Combined adaptation |

The uniform polynomial controls distinguish width dependence from the change from cosine to polynomial decay. Reuse the saved cosine references, including S1 at LR 0.004 and S2 at LR 0.008. Comparing new S1 and S2 arms at different peak LRs would assess tuned configurations, but would not isolate optimizer-state separation.

For a compute-limited first pilot, the strongest measured linear configuration is S1 at LR 0.004 / warmup 64. A matched uniform-polynomial versus CaLR pair on S1 could assess the scheduler hypothesis before investing in the complete factorial. S2 + CaLR remains the candidate for the combined paper-inspired approach. This is a proposal, not a claim that either will close the standalone gap.

Keep initialization, seed/action/data streams, corpus, clipping, precision, token budget, validation set, and correction mode matched. Assess:

- Terminal loss and perplexity at every width, with gaps to matching standalones.
- Full-width gap reduction alongside any deterioration at smaller widths.
- Recorded per-width validation trajectories and actually applied LRs.
- Optimizer-state memory and throughput, especially when comparing S1 with S2.

The primary question is whether CaLR improves the attainable shared-model quality across widths. A correct width ranking alone is insufficient: the current linear widths already rank monotonically by validation loss while larger-width standalone gaps remain. Multiple seeds would be needed before claiming a reliable improvement.

## 6. Evidence used for this note

Implementation was inspected in the current repository. Numeric comparisons were transcribed from existing report CSVs and checked against the earlier analysis; this note does not claim a fresh terminal-checkpoint audit.

- [Original ownership analysis](tinystories_optimizer_ownership_results_analysis_2026-09-10.md).
- [Completed peak-LR endpoint table](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s2-warmup-peak-lr-v1/reports/peak_lr/endpoints.csv).
- [Completed warmup endpoint table](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s2-warmup-peak-lr-v1/reports/warmup/endpoints.csv).
- [Peak-LR report source identities](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s2-warmup-peak-lr-v1/reports/peak_lr/plot_sources.json).
- [Warmup report source identities](/nfs-stor/ivo.navarrete/results/elasticnn/optimizer-ownership-s2-warmup-peak-lr-v1/reports/warmup/plot_sources.json).

These artifact links refer to the local results filesystem; the repository source links are relative to this note.
