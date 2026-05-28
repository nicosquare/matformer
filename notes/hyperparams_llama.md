# Recommended AdamW Hyperparameters for LLaMA Pretraining

Based primarily on:
- Benchmarking Optimizers for Large Language Model Pretraining (https://arxiv.org/abs/2509.01440)
- Common LLaMA, GPT-NeoX, and Chinchilla-style training practices

---

# Core Recommended Configuration

| Hyperparameter | Recommended Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| β1 | 0.9 |
| β2 | 0.95 |
| ε | 1e-8 |
| Weight Decay | 0.1 |
| Scheduler | Cosine decay |
| Warmup | 2000 steps or 1–3% of total steps |
| Gradient Clipping | 1.0 |
| Precision | bf16 |

---

# Recommended Learning Rate by Model Size

| Model Size | Suggested Learning Rate |
|---|---|
| ≤1B parameters | 1e-3 |
| 1B–7B parameters | 6e-4 to 3e-4 |
| 7B–13B parameters | 3e-4 |
| 30B+ parameters | 1e-4 to 2e-4 |

As model size increases, the optimal learning rate usually decreases.

---

# Recommended Scheduler Structure

The dominant recipe for modern LLM pretraining is:

Linear Warmup → Cosine Decay

Warmup is especially important for:
- AdamW optimization
- mixed precision training
- large batch sizes
- deep transformer stability

---

# Warmup Recommendations

| Training Scale | Suggested Warmup |
|---|---|
| Small experiments | 500–2000 steps |
| Medium pretraining runs | 1–3% of total steps |
| Very large runs | Often token-ratio based |

Warmup helps:
- stabilize early optimization
- reduce optimizer instability
- improve mixed precision robustness
- avoid early divergence

---

# Adam Betas

Modern LLaMA-family training commonly uses:

| Parameter | Recommended Value |
|---|---|
| β1 | 0.9 |
| β2 | 0.95 |

Older transformer recipes often used β2 = 0.999, but β2 = 0.95 is now frequently preferred because it:
- adapts faster
- improves large-scale transformer optimization
- provides better stability during pretraining

---

# Weight Decay

| Hyperparameter | Recommended Value |
|---|---|
| Weight Decay | 0.1 |

This value is highly standard across:
- LLaMA
- GPT-NeoX
- Chinchilla-style training
- recent optimizer benchmark studies

---

# Gradient Clipping

| Hyperparameter | Recommended Range |
|---|---|
| Gradient Clip Norm | 0.5–1.0 |

Most LLaMA-style training setups use 1.0.

Gradient clipping improves:
- stability
- mixed precision robustness
- large batch training behavior

---

# Typical Large-Scale LLaMA Training Setup

| Component | Typical Value |
|---|---|
| Sequence Length | 2048–4096 |
| Global Batch Size | 1M–4M tokens |
| Precision | bf16 |
| LR Scheduler | Cosine decay |
| Warmup | Linear |
| Final LR Ratio | ~10% of peak LR |

---

# Conservative Safe Default Recipe

| Hyperparameter | Recommended Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| β1 | 0.9 |
| β2 | 0.95 |
| ε | 1e-8 |
| Weight Decay | 0.1 |
| Scheduler | Cosine decay |
| Warmup | 2000 steps |
| Gradient Clipping | 1.0 |
| Precision | bf16 |

This configuration is a strong modern default for pretraining a LLaMA-style transformer using AdamW.