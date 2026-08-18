# Clarq
clarity, engineered
# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## SEMICON India Hackathon 2026 — KLA Problem Statement

This repository contains the final inference pipeline and trained model for
AI-based restoration of degraded semiconductor inspection images.

The submitted model performs:

- Denoising
- 2× spatial upscaling
- Fine-detail preservation

The final architecture is a modified NAFNet-based restoration network with an
integrated 2× super-resolution path and residual learning.

---

# 1. Final Model

The final submitted model is:

**NAFNetSR + High-Frequency Loss**

The architecture combines the restoration capability of NAFNet with a
2× super-resolution output path.

### Architecture

```text
Input NoisyLR
128 × 128 × 1
      │
      ▼
NAFNet Encoder
      │
      ▼
NAFNet Middle Blocks
      │
      ▼
NAFNet Decoder
      │
      ▼
2× Upsampling
      │
      ▼
Learned Residual
      │
      ├───────────────┐
      │               │
      ▼               ▼
Residual        Bicubic ×2 Base
      │               │
      └───────┬───────┘
              ▼
        Restored Image
        256 × 256 × 1
