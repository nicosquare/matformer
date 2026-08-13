# Notes Index and Status

Reviewed: 2026-08-07

The `specs/` tree and resolved runtime configuration are authoritative for
implemented behavior. Notes explain design history, experimental evidence, or
proposals; they must not silently override a feature specification or saved run
configuration.

## Current probabilistic-adaptation notes

| Note | Status | Still necessary? | Role |
| --- | --- | --- | --- |
| [Probabilistic adaptive granularity discussion](probabilistic_adaptive_granularity_discussion_2026-08-05.md) | Current design summary | Yes | Explains the implemented Bayesian controller, resolved design decisions, durable experimental conclusions, and current research direction. Detailed run transcripts have been removed from it. |
| [Balanced warmup notes](probabilistic_adaptive_granularity_experimental_notes.md) | Current implemented contract | Yes | Concise description of balanced pre-adaptive warmup. The detailed authoritative contracts remain under `specs/010-probabilistic-adaptive-granularity/`. |
| [Fixed-Q calibration experiments](probabilistic_adaptive_granularity_q_calibration_2026-08-07.md) | Historical experimental evidence | Yes, for provenance | Preserves the 100M diagnosis and Batches 1–4, including exact commands and results. Fixed-$Q$ tuning is paused; superseded commands should not be resubmitted blindly. |
| [Episodic-reset proposal](probabilistic_adaptive_granularity_reset_method_proposal_2026-08-07.md) | Active proposal, not implemented | Yes | Defines the candidate $Q=0$ reset lifecycle, persistence contract, overhead, risks, and first ablation. |
| [Spec Kit prompt](probabilistic_adaptive_granularity_speckit_prompt.md) | Superseded implementation input | No for current work | The prompt led to feature 010. `spec.md`, `plan.md`, `tasks.md`, and contracts under the feature directory now supersede it. Keep only as generation provenance or archive it. |

## Other notes

| Note | Status | Still necessary? | Recommended treatment |
| --- | --- | --- | --- |
| [MatFormer reproduction specification](step_1.md) | Foundational brief, partially superseded | Historical reference | Keep for project origin and paper-reproduction intent. Use `specs/001-matformer-lm-reproduction/` for implemented requirements. |
| [Granularity sampling discussion](granularity_sampling_discussion_2026-06-09.md) | Superseded early ideation | No for current decisions | Archive or remove after confirming no external references depend on it. Its useful Bayesian direction is represented by feature 010 and the current discussion. |
| `adaptive_per_block_proposal.md` | Superseded heuristic proposal; currently deleted in the working tree | No | The proposal used training-batch loss and the legacy pseudo-Thompson path. Its pending deletion is consistent with the current method history; do not restore it as active guidance. |
| [dmodel256 command catalog](dmodel256_explicit_granularity_commands.md) | Partially stale operational reference | Not in its current form | It contains useful experiment shapes but old filesystem paths and pre-balanced-warmup Bayesian commands. Refresh it before reuse or archive it in favor of dated experiment records. |
| [Checkpoint-selection discussion](checkpoint_selection_methodology_discussion_2026-08-05.md) | Valid open methodology discussion | Yes while unresolved | Keep. It explicitly records that no selection/reporting change has been approved. |
| [LLaMA hyperparameter recommendations](hyperparams_llama.md) | General background, not project configuration | Optional | Archive or remove if a generic literature summary is not useful. Saved configs and presets are the source of actual hyperparameters. |
| [Cat Llama debugging report](cat_llama_debug_report_2026-05-23.md) | Historical diagnostic evidence | Yes, for provenance | Keep or move to a historical/debug archive. Its experiment conclusions are useful, but it is not current runtime guidance. |

## Recommended minimal active set

Keep directly visible for ongoing adaptive-controller work:

- this index;
- the probabilistic adaptive granularity discussion;
- the balanced warmup note;
- the episodic-reset proposal;
- the checkpoint-selection discussion.

Keep the fixed-$Q$ calibration record and Cat Llama report as historical
evidence. The reproduction brief remains useful project provenance. The early
sampling discussion, Spec Kit prompt, generic hyperparameter summary, and stale
command catalog are candidates for an `archive/` directory or deletion after
any external consumers are checked.
