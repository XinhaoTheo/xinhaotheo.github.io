---
title: Machine Unlearning with the TOFU Dataset
date: 2025-05-01
summary: |
  End-to-end LLM unlearning pipeline on TOFU with a fine-tuned Phi-1.5 —
  curvature-aware gradient ascent (Sophia) and localized, layer-sensitive
  updates. Reached 95.8% Forgetting Quality (+17 pts vs. baseline) with
  quantified Model Utility trade-offs.
tags:
  - LLM
  - Machine Unlearning
  - DeepSpeed
  - Optimization
  - PyTorch
featured: true
links:
  - type: source
    url: https://github.com/XinhaoTheo
---

End-to-end **machine-unlearning** pipeline on the **TOFU** benchmark with a fine-tuned **Phi-1.5**, built between February 2024 and May 2025.

## Highlights

- **Stable forgetting.** GradDiff-style negative gradient ascent (AdamW), augmented with **curvature-aware updates via Sophia** (diagonal Hessian + clipping) for more stable forgetting dynamics.
- **Localized unlearning.** Selective updates on attention projections, **layer-sensitivity ranking** via gradient norms (Top-K) with targeted freezing, plus token-level influence and selective masking to preserve utility.
- **Scaled & reproducible.** DeepSpeed **ZeRO + bf16**, deterministic seeding, and experiment tracking — achieving **95.8% Forgetting Quality (+17 pts vs. baseline)** with quantified Model Utility trade-offs.

## Stack

PyTorch · HuggingFace · DeepSpeed · Sophia · Phi-1.5
