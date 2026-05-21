# MatFormer Language Reproduction Specification

This document defines the reproduction plan for the language-model experiments from:

> MatFormer: Nested Transformer for Elastic Inference  
> NeurIPS 2024: https://nips.cc/virtual/2024/poster/94199

The goal is not exact numerical replication, since the paper uses proprietary training data, but rather faithful reproduction of:
- architectural behavior
- scaling trends
- nested-model dynamics
- relative performance comparisons

---

# Reproduction Goals

The reproduction should validate the central claims of the paper:

1. Nested Transformer submodels can perform competitively with independently trained models of similar size.
2. A single nested model can replace many separately trained models.
3. Nested models preserve prediction consistency across granularities.
4. Nested models improve speculative decoding alignment.

---

# Core Experimental Principle

A faithful reproduction requires training BOTH:

1. MatFormer nested models
2. Independently trained fixed-size baseline models

The central comparison is:

```math
\text{Nested extracted submodel}
\quad vs \quad
\text{Independently trained model}
```

at approximately equal parameter count.

Without standalone baselines, the evaluation is incomplete.

---

# Model Architecture

## Language Models

All models are:
- decoder-only Transformers
- causal language models
- autoregressive next-token prediction

Shared configuration:
- 16 Transformer layers
- 16 attention heads
- context length = 1024
- SentencePiece tokenizer
- vocabulary size = 256k

Only the FFN width changes between granularities.

---

# MatFormer Granularities

The paper uses 4 nested FFN granularities:

| Granularity | FFN Expansion Ratio |
|---|---|
| S | 0.5 |
| M | 1 |
| L | 2 |
| XL | 4 |

If:
```math
d_{ff} = 4d
```

then:
- S uses the first 0.5d FFN neurons
- M uses the first d FFN neurons
- L uses the first 2d FFN neurons
- XL uses the full 4d FFN neurons

All smaller models are strict prefixes of larger models.

---

# Standalone Baselines

For every MatFormer granularity:
- train an equivalent standalone Transformer

Example:

| Granularity | FFN Ratio | Standalone Baseline |
|---|---|---|
| S | 0.5 | independently trained S model |
| M | 1 | independently trained M model |
| L | 2 | independently trained L model |
| XL | 4 | independently trained XL model |

Standalone models should:
- use identical architecture
- use identical optimizer
- use identical tokenizer
- use identical datasets
- use identical training tokens

The only difference should be FFN width.

---

# Language Model Sizes

| Model Size | d_model | Training Tokens |
|---|---|---|
| 78M | 256 | 10B |
| 180M | 512 | 20B |
| 310M | 768 | 30B |
| 463M | 1024 | 40B |
| 850M | 1536 | 80B |

---

# Training Objective

## Standard LM Loss

Autoregressive next-token prediction:
- cross entropy loss

---

# MatFormer Loss

For every batch:
- all granularities are evaluated
- all granularities contribute to training

Objective:

```math
\mathcal{L}
=
\sum_{i=1}^{g}
\lambda_i \mathcal{L}_i
```

where:
- $g$ = number of granularities
- $\mathcal{L}_i$ = cross entropy loss for granularity $i$

---

# Dataset Strategy

## Important Note

The paper uses proprietary pretraining data.

Exact numerical reproduction is therefore impossible.

The reproduction should instead focus on:
- relative comparisons
- scaling behavior
- architectural trends

---

# Recommended Datasets

## Phase 1 — Small-scale validation

Datasets:
- TinyStories
- Tiny Shakespeare

Goal:
- verify nested FFN training
- validate extraction logic
- debug implementation

---

# Phase 2 — Medium-scale reproduction

Datasets:
- FineWeb subset
- SlimPajama subset

Goal:
- reproduce perplexity trends
- reproduce downstream behavior

---

# Phase 3 — Large-scale reproduction

Datasets:
- FineWeb
- SlimPajama
- C4

Goal:
- reproduce scaling curves
- reproduce Figure 2 behavior
- compare against standalone baselines

---

# Evaluation Framework

Use:
- EleutherAI LM Evaluation Harness

Recommended command interface:

```bash
lm_eval
```

This improves comparability with:
- MatFormer
- DynaBERT
- OFA
- modern decoder-only language models

---

# Validation Evaluation

## Metrics

Primary language-model metrics:
- validation loss
- perplexity (PPL)

Plots:
- loss vs model size
- perplexity vs model size

---

# Downstream Evaluation Tasks

The paper evaluates approximately 25 English tasks.

The reconstructed suite includes the following categories.

---

# Open-Domain QA

| Task |
|---|
| TriviaQA |
| Natural Questions |
| WebQuestions |

---

# Cloze / Completion

| Task |
|---|
| LAMBADA |
| HellaSwag |
| StoryCloze |

---

# Winograd-Style Reasoning

| Task |
|---|
| Winograd |
| WinoGrande |

---

# Reading Comprehension

| Task |
|---|
| RACE |

---

# Commonsense Reasoning

| Task |
|---|
| PIQA |
| ARC-Easy |
| ARC-Challenge |
| OpenBookQA |

---

# SuperGLUE

| Task |
|---|
| BoolQ |
| CB |
| COPA |
| MultiRC |
| ReCoRD |
| RTE |
| WiC |
| WSC |

---

# ANLI

| Task |
|---|
| ANLI-R1 |
| ANLI-R2 |
| ANLI-R3 |

---

# Recommended Minimal Evaluation Suite

For early-stage experiments:

| Task | Category |
|---|---|
| HellaSwag | Completion |
| PIQA | Commonsense |
| ARC-Challenge | Reasoning |
| BoolQ | QA |
| WinoGrande | Coreference |
| OpenBookQA | Commonsense |

This subset is:
- stable
- inexpensive
- widely used
- representative

---

# Figure 2 Reproduction

The reproduction should aim to recreate the equivalent of Figure 2 from the paper.

The figure studies:
- loss
- downstream accuracy
- consistency

as a function of model size.

---

# X-Axis Definition

The paper reports:

```math
N \; (\text{Non-Embedding Parameters})
```

This excludes:
- token embeddings
- output embeddings / LM head

and counts only:
- Transformer block parameters

Reason:
- embeddings dominate parameter count at small scales
- embeddings are mostly unaffected by FFN nesting
- the paper wants to isolate Transformer capacity

Definition:

```math
N =
N_{\text{total}}
-
N_{\text{embeddings}}
```

---

# Required Metrics

## 1. Validation Loss

Metrics:
- validation loss
- perplexity

Plots:
- loss vs non-embedding parameters
- perplexity vs non-embedding parameters

Curves:
- standalone baselines
- MatFormer extracted submodels

---

# 2. Downstream Accuracy

Metric:
- average benchmark accuracy

Plots:
- accuracy vs non-embedding parameters

Curves:
- standalone baselines
- MatFormer extracted submodels

---

# 3. Consistency

Consistency measures alignment between:
- smaller submodels
- larger submodels

This evaluates whether nested models preserve prediction behavior across scales.

Possible implementation:

## Token-Level Agreement

```math
\text{Consistency}
=
\frac{1}{T}
\sum_t
\mathbf{1}
[
\arg\max p_s(x_t)
=
\arg\max p_l(x_t)
]
```

where:
- $p_s$ = smaller model distribution
- $p_l$ = larger model distribution

Alternative metrics:
- KL divergence
- top-k overlap
- speculative decoding acceptance rate

---

# Mix-and-Match Evaluation

The paper evaluates heterogeneous nested models.

Different layers can use different granularities:

| Layer | Granularity |
|---|---|
| 1 | XL |
| 2 | M |
| 3 | S |
| ... | ... |

This enables:
- elastic inference
- compute-performance tradeoffs
- adaptive compute allocation

---

# Speculative Decoding Evaluation

Nested models are evaluated for speculative decoding.

Setup:
- small nested model = draft model
- large nested model = verifier

Metrics:
- acceptance rate
- throughput
- rollback frequency
- latency

Main hypothesis:

```math
p_{\text{small}}(x)
\approx
p_{\text{large}}(x)
```

Nested models should be more distributionally aligned than independently trained models.

Comparisons:
- MatFormer draft/verifier
- standalone draft/verifier

---

# Reporting Requirements

Every experiment should export structured outputs.

At minimum:
- configuration used
- scalar metrics
- CSV summaries that allow to reproduce or generated new plots on demand
- plots
- checkpoints when relevant

Avoid storing metrics only in terminal logs.

---

# Required CSV Outputs

Minimum expected outputs:

```text
metrics.csv
task_results.csv
scaling_results.csv
consistency_results.csv
```

---

# Figure Generation

All plots should be reproducible from exported CSV files.

Preferred workflow:
- Python plotting scripts
- notebook reproducibility
- versioned figure generation

Avoid:
- manual spreadsheet plotting
- hidden postprocessing

---

# Training Efficiency Reporting

Track:
- FLOPs
- tokens/sec
- wall-clock time

Important because MatFormer computes multiple losses simultaneously.

---

# Memory Reporting

Track:
- peak GPU memory
- activation memory
- parameter memory

Useful for analyzing nested training overhead.

---

# Recommended Reproduction Order

## Phase 1
- small-scale proof of concept
- perplexity only
- verify extraction works

---

# Phase 2
- standalone vs nested comparisons
- downstream tasks
- Figure 2 reproduction

---

# Phase 3
- consistency analysis
- speculative decoding
- mix-and-match granularities
- scaling-law analysis

---

# Reproduction Success Criteria

The reproduction should aim to validate:

1. Extracted nested models are competitive with standalone models.
2. Nested models preserve predictive consistency.
3. Scaling behavior resembles the original paper.
4. Speculative decoding alignment improves.
5. Elastic inference tradeoffs emerge naturally.

Exact numerical matching is NOT required.