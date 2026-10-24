# Weights

| File | Model |
|---|---|
| **`final_model.pth`** | **SUBMITTED MODEL** — wide NAFNet-lite (base_ch=48) + FiLM, trained with VGG perceptual loss. This is the checkpoint `run.py` loads by default. |
| `baseline-models/baseline_model.pth` | Ablation — U-Net baseline |
| `baseline-models/adaptive_model.pth` | Ablation — U-Net + FiLM |
| `baseline-models/nafnet_baseline_model.pth` | Ablation — NAFNet-lite baseline |
| `baseline-models/nafnet_adaptive_model.pth` | Ablation — NAFNet-lite + FiLM |

The four ablation checkpoints are kept for reproducibility of the comparison table in
the README; they are not the submitted model. `evaluate.py` reads `final_model.pth`
from this folder and the ablation checkpoints from `baseline-models/` automatically.
