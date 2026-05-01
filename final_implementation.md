# PA-GNN — Final Implementation Blueprint
### Physics-Aware Graph Neural Network for Autonomous Planetary Path Planning
**Last Updated:** May 2, 2026 — reflects all code fixes applied

---

## 1. System Overview

```
Input Image (512×512×3)
    │
    ├──► Stage 2: Physics Features ──► S, R, D, H_physics
    │
    ├──► Stage 3: CNN Risk Heatmap ──► H_learned
    │
    └──► Stage 4: Adaptive Fusion ──► H_final = α·H_learned + (1-α)·H_physics
              │
              ▼
         Stage 5: SLIC Superpixels → 14-dim Node Features → RAG Edges
              │
              ▼
         Stage 6: GATv2 (14→128→32→1) → Traversability Scores p̂_i
              │
              ▼
         Stage 7: Physics-Aware A* → Safe Path
```

**Target:** < 5 sec per tile on CPU (benchmark timing now instrumented in pipeline).

---

## 2. Datasets

### 2.1 AI4Mars (Primary Training)
- **Source:** MSL NavCam EDR images, 1024×1024 grayscale
- **Preprocessing:** Resize→512, per-tile min-max norm, replicate→3-ch
- **Label Remap (continuous risk):**

| Terrain | Pixel Value | Risk Score |
|---|---|---|
| Soil | 0 | 0.1 (Safe) |
| Bedrock | 1 | 0.5 (Uncertain) |
| Sand | 2 | 0.4 (Uncertain) |
| Big Rock | 3 | 0.9 (Hazardous) |
| NULL | 255 | -1 (Ignored) |

- **Splits:** 70/15/15 (train/val/test), stratified by dominant class
- **Augmentation (train only):** H-flip(0.5), V-flip(0.5), rotation(±15°), brightness(±20%), contrast(±20%), Gaussian noise(σ≤0.02)

### 2.2 HiRISE v3 (Cross-Domain Evaluation)
- **Source:** MRO orbital crops, 227×227 grayscale, image-level labels
- **Preprocessing:** Resize→512 (bilinear), per-tile min-max norm, replicate→3-ch
- **Label Remap (9 classes):**

| Class | Index | Risk |
|---|---|---|
| Other | 0 | 0.15 |
| Crater | 1 | 0.90 |
| Dark Dune | 2 | 0.85 |
| Slope Streak | 3 | 0.80 |
| Bright Dune | 4 | 0.50 |
| Impact Ejecta | 5 | 0.55 |
| Swiss Cheese | 6 | 0.85 |
| Spider | 7 | 0.45 |
| Edge Case | 8 | 0.50 |

- **Filtering:** Original crops only (suffix-based, not positional)
- **Evaluation:** Patch-level aggregation via `aggregate_patch_risk()` (mean of H_final)

### 2.3 MurrayLab CTX (Qualitative Demo)
- **Source:** 512×512 orbital tiles, unlabeled
- **Purpose:** Visual demonstration of physics-aware path planning on orbital imagery

---

## 3. Pipeline Stages — Implementation Specs

### Stage 2: Physics Feature Extraction
**Files:** `src/physics/slope.py`, `roughness.py`, `discontinuity.py`, `combine.py`
**Wrapper:** `src/data/transforms/physics_features.py` → `PhysicsFeatureExtractor`

| Feature | Method | Params | Output |
|---|---|---|---|
| Slope (S) | Sobel gradient magnitude | kernel=3 | [0,1] per-tile normalized |
| Roughness (R) | Local std deviation | window=7 | [0,1] per-tile normalized |
| Discontinuity (D) | Abs LoG response | kernel=9, σ=2.0 | [0,1] per-tile normalized |

**Combination:** `H_physics = 0.4·S + 0.3·R + 0.3·D`

All operations: PyTorch, batched, reflect-padding, `@torch.no_grad()`.

### Stage 3: CNN Risk Heatmap
**Files:** `src/models/cnn/risk_model.py`, `mobilenetv3.py`, `deeplabv3plus.py`

| Component | Spec |
|---|---|
| Encoder | MobileNetV3-Large, ImageNet-1k pretrained |
| Decoder | DeepLabV3+ (ASPP rates: 6,12,18, out_ch: 256) |
| Head | 1-channel, sigmoid activation → [0,1] |
| Output | H_learned: (B, 1, 512, 512) |

**Training:**
- Optimizer: AdamW (lr=1e-4, wd=1e-4)
- Scheduler: CosineAnnealing (T_max=60)
- Epochs: 60, early stopping patience=10
- Monitor: `val_hazard_recall` (mode=max)
- **Loss: Compound BCE + Dice + TV** (NOT Focal Loss)
  - BCE weight: 1.0 (hazard_weight=3.0)
  - Dice weight: 0.5 (threshold=0.7)
  - TV weight: 0.1

### Stage 4: Adaptive Hybrid Fusion
**Files:** `src/models/fusion/adaptive_fusion.py`, `fusion_model.py`

**Architecture:** 3-layer lightweight CNN
```
Input: [H_physics | H_learned | grayscale] → 3 channels
Conv(3→16, 3×3) + ReLU
Conv(16→8, 3×3) + ReLU  
Conv(8→1, 1×1) + Sigmoid → α map
```

**Fusion:** `H_final = α · H_learned + (1-α) · H_physics`

**Training Strategy: TWO-PHASE (freeze CNN)**
1. Phase 1: Train CNN alone (`scripts/train_cnn.py`)
2. Phase 2: Freeze CNN, load checkpoint, train fusion only (`scripts/train_fusion.py --cnn_ckpt ...`)
- Config: `joint_with_cnn: false` (prevents lazy-learning)

### Stage 5: Superpixel Graph Construction
**Files:** `src/graph/superpixels.py`, `node_features.py`, `adjacency.py`
**Orchestrator:** `src/models/gnn/graph_builder.py`

**SLIC:** n_segments=300, compactness=10.0, sigma=1.0

**14-Dimensional Node Feature Vector:**

| Index | Feature | Source |
|---|---|---|
| 0 | Centroid Y (normalized) | SLIC |
| 1 | Centroid X (normalized) | SLIC |
| 2 | Mean Slope (S) | Physics |
| 3 | Mean Roughness (R) | Physics |
| 4 | Mean Discontinuity (D) | Physics |
| 5 | Mean H_physics | Physics |
| 6 | Mean H_learned | CNN |
| 7 | Mean H_final | Fusion |
| 8 | Mean α | Fusion |
| 9 | Pixel area (normalized) | SLIC |
| 10 | Mean intensity | Image |
| 11 | Std intensity | Image |
| 12 | Is hazardous (binary) | H_final > 0.7 |
| 13 | Hazardous neighbour count | RAG |

**Edge Weights:** `w(u,v) = 0.6·avg_risk + 0.25·norm_dist + 0.15·|slope_u - slope_v|`

### Stage 6: GATv2 Traversability Refinement
**File:** `src/models/gnn/gatv2.py`

```
Layer 1: GATv2Conv(14→32, 4 heads, concat=True) → 128-dim + ELU + Dropout(0.3)
Layer 2: GATv2Conv(128→32, 4 heads, concat=False) → 32-dim + ELU + Dropout(0.2)
Output:  Linear(32→1) + Sigmoid → p̂_i ∈ [0,1]
```

**Training:**
- Optimizer: Adam (lr=1e-3, wd=5e-4)
- Epochs: 100, early stopping patience=15, monitor=`val_auc_roc`
- Loss: BCE with positive_weight=3.0
- **Weak Labeling:** 2-hop neighbours of hazards → label=0.7

**Node Deactivation:** p̂_i > (1.0 - deactivation_threshold) → obstacle
- Config: `deactivation_threshold: 0.2` → risk > 0.8 → deactivated
- Now reads from config (previously hardcoded)

### Stage 7: Physics-Aware A* Path Planning
**Files:** `src/planning/astar.py`, `heuristics.py`

**Heuristic:** `h(n) = d_euc(n,g) × (1 + γ_r·risk_n + γ_s·S_n)`
- γ_r = 0.4, γ_s = 0.1 (proposed); both 0.0 for baselines
- `risk_n` = GATv2 output (higher = more dangerous)

**Benchmark Timing:** `pipeline.run(..., benchmark=True)` returns per-stage timing dict.

---

## 4. Training Workflow

```bash
# Step 1: Validate datasets
python scripts/validate_datasets.py

# Step 2: Create train/val/test splits
python scripts/create_splits.py

# Step 3: Train CNN (Stage 3)
python scripts/train_cnn.py

# Step 4: Train Fusion (Stage 4) — CNN frozen
python scripts/train_fusion.py --cnn_ckpt checkpoints/cnn/best_model.pth

# Step 5: Train GNN (Stage 6)
python scripts/train_gnn.py

# Step 6: Evaluate on AI4Mars test set
python src/evaluation/evaluate_ai4mars.py

# Step 7: Cross-domain evaluation on HiRISE
python src/evaluation/evaluate_hirise.py

# Step 8: CTX qualitative demo
python src/evaluation/demo_ctx.py
```

---

## 5. Evaluation Plan

### 5.1 Baselines
| ID | Name | Edge Weight | Heuristic | Description |
|---|---|---|---|---|
| B1 | Euclidean | Euclidean distance | Euclidean | No risk awareness |
| B2 | Physics-Only | α_w·H_physics + β_w·dist | Euclidean | Physics only, no CNN |
| B3 | Learned-Only | α_w·H_learned + β_w·dist | Euclidean | CNN only, no physics |
| B4 | Static Fusion | α=0.5 fixed | Euclidean | Fixed blend |
| **Proposed** | **PA-GNN** | **Full edge formula** | **Physics-aware** | **Adaptive fusion + GATv2** |

### 5.2 Metrics
| Metric | What It Measures | Target |
|---|---|---|
| Hazard Recall | % of real hazards detected | > 90% |
| IoU | Overlap of predicted vs true hazard zones | > 0.5 |
| ECE | Calibration error (10 bins) | < 0.10 |
| Success Rate | % of tiles where A* finds a path | > 85% |
| HCR | High-Cost Ratio (path nodes with risk > 0.7) | < 10% |
| Inference Time | End-to-end per tile | < 5 sec CPU |
| AUC-ROC | GATv2 node classification quality | > 0.80 |

### 5.3 Cross-Domain Protocol (HiRISE)
- Train on AI4Mars only → evaluate on HiRISE without fine-tuning
- Use `aggregate_patch_risk()` to convert pixel-level output to patch-level score
- Compare with `compute_hirise_metrics()` for accuracy, hazard recall, precision

---

## 6. Fixes Applied in This Version

| # | Fix | File(s) Changed | Severity |
|---|---|---|---|
| 1 | Added `import numpy as np` | `scripts/train_gnn.py` | 🔴 Bug fix |
| 2 | Added `edge_case` class (index 8) | `label_remap.py`, `hirise.yaml` | 🔴 Crash prevention |
| 3 | Set `joint_with_cnn: false` | `adaptive_fusion.yaml` | 🔴 Training stability |
| 4 | Added `aggregate_patch_risk()` + `compute_hirise_metrics()` | `metrics.py` | 🟡 Missing evaluation |
| 5 | Deactivation threshold reads from config | `pipeline.py` | 🟡 Config consistency |
| 6 | Per-stage benchmark timing | `pipeline.py` | 🟡 Performance tracking |
| 7 | Updated all `pipeline.run()` callers to 4-tuple | `evaluate_ai4mars.py`, `run_inference.py`, `demo_ctx.py` | 🟡 API compatibility |
| 8 | **Fixed `load_label_mask()` — added `.convert('L')`** | `src/utils/io.py` | 🔴 **Training-breaking bug** |
| 9 | Added `warnings.warn()` on empty valid_mask | `src/training/losses.py` | 🟡 Debug aid |

### Fix #8 Detail — Root Cause of Loss = 0.0000

AI4Mars label PNGs are RGB-encoded: `(0,0,0)=soil`, `(1,1,1)=bedrock`, `(2,2,2)=sand`, `(3,3,3)=big_rock`, `(255,255,255)=null`.

`load_label_mask()` called `Image.open(path)` **without** `.convert('L')`. PIL returned a 3D array (H,W,3) or palette-indexed data. When passed through the LUT remapper (which maps only indices 0-3 to valid risk scores), all pixels mapped to `-1.0` (ignore). This made `valid_mask.sum() == 0` every batch → loss function hit the early-return → **loss = 0.0000 forever**.

```diff
# src/utils/io.py
- img = Image.open(path)
+ img = Image.open(path).convert('L')
```

---

## 7. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Per-tile min-max normalization | Risk scores not comparable across tiles | Document as design trade-off |
| No subsurface hazard detection | Invisible terrain hazards (ice, voids) missed | Acknowledged scope limitation |
| Physics weights not tuned via grid search | Weights are reasonable defaults (0.4/0.3/0.3) | Add tuning script before paper |
| Single-seed training | Results may vary across seeds | Run 3 seeds before paper |

---

## 8. Project Structure

```
pa-gnn/
├── configs/
│   ├── base.yaml                    # Paths, seeds, global settings
│   ├── cnn/mobilenetv3.yaml         # CNN architecture + training
│   ├── datasets/ai4mars.yaml        # AI4Mars preprocessing + labels
│   ├── datasets/hirise.yaml         # HiRISE preprocessing + 9 classes
│   ├── fusion/adaptive_fusion.yaml  # Fusion arch + two-phase training
│   ├── gnn/gatv2.yaml               # GATv2 layers + graph params
│   └── physics.yaml                 # S/R/D feature params + weights
├── scripts/
│   ├── train_cnn.py                 # Stage 3 training
│   ├── train_fusion.py              # Stage 4 training (CNN frozen)
│   ├── train_gnn.py                 # Stage 6 training
│   ├── validate_datasets.py         # Data integrity checks
│   └── create_splits.py             # Train/val/test splitting
├── src/
│   ├── data/
│   │   ├── loaders/                 # AI4Mars, HiRISE, CTX dataset classes
│   │   ├── preprocessing/           # normalize, resize, augmentations
│   │   └── transforms/              # label_remap, physics_features
│   ├── models/
│   │   ├── cnn/                     # MobileNetV3 + DeepLabV3+
│   │   ├── fusion/                  # AdaptiveFusion + EndToEndFusionModel
│   │   └── gnn/                     # PAGATv2 + GraphBuilder
│   ├── physics/                     # slope, roughness, discontinuity, combine
│   ├── graph/                       # superpixels, node_features, adjacency
│   ├── planning/                    # astar, heuristics
│   ├── training/                    # losses (BCE+Dice+TV), trainer, weak_labels
│   ├── evaluation/                  # metrics, evaluate_ai4mars/hirise, demo_ctx
│   ├── inference/                   # pipeline (with benchmark timing)
│   └── visualization/               # path plotting utilities
└── implementation.md                # ← This document
```
