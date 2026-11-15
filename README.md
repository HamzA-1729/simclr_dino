# Self-Supervised Vision from Scratch: SimCLR + DINO

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, first-principles PyTorch implementation and controlled comparison of **SimCLR (Contrastive Learning)** and **DINO (Self-Distillation with No Labels)** on **CIFAR-10 (32×32 RGB)** using a shared **Tiny Vision Transformer (ViT)** backbone—built from scratch without high-level SSL frameworks.

---

## Core Capabilities & Highlights

* **Shared Tiny ViT Backbone (`vit.py`):** `patch_size=4` (64 patch tokens + `[CLS]`), pre-norm GELU blocks, bidirectional multi-head self-attention (with a manual attention-weight path verified against `F.scaled_dot_product_attention`), and dynamic 2D sin-cos positional embeddings computed analytically for both `32×32` global and `16×16` local crops.
* **Batched Tensor Augmentations (`augmentations.py`):** Per-sample `F.affine_grid` + `F.grid_sample` crops, horizontal flip, color jitter (strength `0.5`, applied before normalization), and grayscale—eliminating PIL DataLoader bottlenecks on CPU.
* **SimCLR (`simclr.py`):** 2-layer nonlinear projection head + vectorized symmetric **NT-Xent (InfoNCE)** loss over $2(N-1)$ in-batch negatives.
* **DINO (`dino.py`):** Student–Teacher EMA momentum encoder (covering both backbone and head), multi-crop views (`2` global + `4` local), 3-layer + weight-normalized prototype head (`original0` fixed at `1.0`), `freeze_last_layer_steps`, teacher centering, and linear teacher temperature warmup.
* **Zero-Leakage Evaluation & Collapse Diagnostics (`data.py`, `eval.py`, `diagnostics.py`):** Stratified `45,000 train / 5,000 val / 10,000 test` split, chunked weighted k-NN (`k=20, tau=0.07`), frozen Linear Probe, multi-seed random-init baseline, normalized `[CLS]` feature std, **RankMe** effective rank (`eps=1e-7`), and DINO teacher entropy / KL.

---

## Architecture Comparison

```text
SimCLR (Contrastive)                         DINO (Self-Distillation)
────────────────────────                     ────────────────────────
2 Global Views (32x32)                       2 Global (32x32) + 4 Local (16x16)
        │                                            │
        ▼                                    ┌───────┴───────┐
  Shared TinyViT                             ▼               ▼
        │                               Student ViT     Teacher ViT (EMA, stop-grad)
        ▼                                    │               │
  2-Layer MLP Head                           ▼               ▼
        │                               3-Layer + WN    3-Layer + WN - Center c
        ▼                                    │               │
  NT-Xent Loss (2N-2 Negatives)              └───────► Cross-Entropy ◄───────┘
```

---

## Mathematical Formulations

### 1. SimCLR (NT-Xent Loss)
Given $\ell_2$-normalized projections $z_i = g(h_i) / \|g(h_i)\|_2$ for $i \in \{1, \dots, 2N\}$ and cosine similarity $s_{i,j} = z_i^\top z_j / \tau$:
$$\ell(i, j) = -\log \frac{\exp(s_{i,j})}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(s_{i,k})}, \quad \mathcal{L}_{\text{SimCLR}} = \frac{1}{2N} \sum_{k=1}^{N} \big[\ell(k, k+N) + \ell(k+N, k)\big]$$

### 2. DINO (Centering + Sharpening + EMA)
With teacher center $c \leftarrow m_c c + (1 - m_c)\,\text{mean}(g_{\theta_t}(x_{\text{global}}))$ and stop-gradient on $\theta_t$:
$$P_t(x) = \mathrm{softmax}\!\left(\frac{g_{\theta_t}(x) - c}{\tau_t}\right), \quad P_s(x') = \mathrm{softmax}\!\left(\frac{g_{\theta_s}(x')}{\tau_s}\right)$$
$$\mathcal{L}_{\text{DINO}} = \frac{1}{2(1 + N_{\text{local}})} \sum_{v \in \{1, 2\}} \sum_{v' \neq v} -\sum_{k=1}^{K} P_t(x_v)_k \log P_s(x'_{v'})_k$$

### 3. Collapse Diagnostics on Backbone `[CLS]` Features $h$
* **Per-dim std:** $\frac{1}{D}\sum_{d=1}^D \mathrm{std}(\hat{h}_{:, d})$ where $\hat{h} = h / \|h\|_2$ (healthy $\approx 1/\sqrt{D}$; `0` = constant collapse).
* **RankMe Effective Rank:** $\exp\!\big(-\sum_i p_i \log(p_i + \epsilon)\big)$ where $p_i = \sigma_i / (\sum_j \sigma_j + \epsilon)$, $\epsilon = 10^{-7}$.

---

## Documented Deviations from Original Papers

1. **Shared Augmentation Recipe:** DINO's asymmetric global Gaussian blur and solarization are omitted so both SimCLR and DINO see the identical global augmentation distribution (and Gaussian blur is skipped on 32×32 CIFAR-10 per SimCLR Appendix B.9).
2. **Batched Tensor Jitter:** Implemented directly via `torch` tensor ops on `[0, 1]` images before normalization; numerically close to, but not bit-identical with, `torchvision.transforms.ColorJitter`.
3. **Constant Weight Decay:** Uses constant `weight_decay = 0.05` with AdamW instead of DINO's `0.04 -> 0.4` cosine schedule.
4. **Cached Unaugmented Linear Probe:** Linear probing trains `nn.Linear(D, 10)` on cached, unaugmented frozen `[CLS]` features for fast evaluation; numbers are not directly comparable to augmented ImageNet probes.

---

## Quickstart & Verification Commands

### 1. Run All Unit Tests
```bash
cd "/home/mm-ho/My Projects/simclr_dino"
uv run pytest -v
```

### 2. Random-Init Baseline & CPU Timing Gate (10 Steps)
```bash
cd "/home/mm-ho/My Projects/simclr_dino"
uv run python -m simclr_dino.eval --config configs/simclr.yaml --baseline --split val
uv run python -m simclr_dino.train_simclr --max-steps 10 --eval-interval 0 --checkpoint-dir checkpoints/timing_simclr
uv run python -m simclr_dino.train_dino --max-steps 10 --eval-interval 0 --checkpoint-dir checkpoints/timing_dino
```

### 3. Multi-Seed Training (Val Smoke Gate / Sweeps)
```bash
cd "/home/mm-ho/My Projects/simclr_dino"
uv run python -m simclr_dino.train_simclr --seed 42
uv run python -m simclr_dino.train_simclr --seed 1337
uv run python -m simclr_dino.train_dino --seed 42
uv run python -m simclr_dino.train_dino --seed 1337
```

### 4. Final 10k Test Evaluation (Run Once After Selecting Best Val Config) & Plots
```bash
cd "/home/mm-ho/My Projects/simclr_dino"
uv run python -m simclr_dino.eval --config configs/simclr.yaml --checkpoint checkpoints/simclr/best_seed42.pt --branch backbone --split test
uv run python -m simclr_dino.eval --config configs/dino.yaml --checkpoint checkpoints/dino/best_seed42.pt --branch teacher --split test
uv run python -m simclr_dino.visualize
```

---

## Verification & Baseline Status

* **Unit Tests:** `23 passed in 3.10s` (`uv run pytest -v`).
* **Random-Init TinyViT Baseline (CIFAR-10 5k Val k-NN, `k=20`, `tau=0.07`):** **26.10%** mean (`26.28%` seed 42, `25.92%` seed 1337).
* **CPU 10-Step Timing Gate (`train_subset=5000`, `batch_size=256`):**
  * SimCLR: `33.5s` (`3.35 s/step`)
  * DINO: `60.4s` (`6.04 s/step`)
  * Full multi-seed training is deferred to **GPU (Google Colab)** rather than CPU.

