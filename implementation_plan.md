# PA-GNN Implementation Plan

> **Pipeline:** Physics-Aware Graph Neural Network for Autonomous Planetary Path Planning  
> **Environment:** Anaconda → `research` interpreter  
> **Workspace:** `d:\Mars\pa-gnn\`  
> **Documentation:** Every action documented in `d:\Mars\pa-gnn\progression.md`

---

## Quick Reference — Dataset Locations on Disk

| Dataset | Path on Disk |
|---|---|
| AI4Mars (MSL NavCam) images | `d:\Mars\ai4mars-dataset-merged-0.6\ai4mars-dataset-merged-0.6\msl\ncam\images\` |
| AI4Mars (MSL NavCam) train labels | `d:\Mars\ai4mars-dataset-merged-0.6\ai4mars-dataset-merged-0.6\msl\ncam\labels\train\` |
| AI4Mars (MSL NavCam) test labels | `d:\Mars\ai4mars-dataset-merged-0.6\ai4mars-dataset-merged-0.6\msl\ncam\labels\test\masked-gold-min3-100agree\` |
| MurrayLab CTX tiles (set 1) | `d:\Mars\archive\original-image-slices-512x512\original-image-slices-512x512\sliced_tiles_1\` |
| MurrayLab CTX tiles (set 2) | `d:\Mars\archive\original-image-slices-512x512\original-image-slices-512x512\sliced_tiles_2\` |
| HiRISE v3 images | `d:\Mars\hirise-map-proj-v3\map-proj-v3\` |
| HiRISE v3 labels | `d:\Mars\hirise-map-proj-v3\labels-map-proj-v3.txt` |
| HiRISE v3 classmap | `d:\Mars\hirise-map-proj-v3\landmarks_map-proj-v3_classmap.csv` |

> [!NOTE]
> The HiRISE classmap has 8 classes (0–7): `other, crater, dark dune, slope streak, bright dune, impact ejecta, swiss cheese, spider`. The blueprint's `edge_case` class does **not** exist in the actual data — remapping will use these 8 real classes only.

---

## Stage 0: Project Scaffolding & Environment Setup

**Goal:** Create the modular project skeleton, verify the `research` conda env, install all dependencies, and validate that all datasets are accessible.

### Tasks

| # | Task | Detail |
|---|---|---|
| 0.1 | Create project directory tree | Follow `folder_structure.md` skeleton under `d:\Mars\pa-gnn\`. Create all dirs: `configs/`, `data/`, `src/`, `scripts/`, `experiments/`, `checkpoints/`, `logs/`, `results/`, `docs/`, `notebooks/` |
| 0.2 | Create `__init__.py` files | In every Python package directory under `src/` |
| 0.3 | Create `configs/base.yaml` | Global config: dataset paths, random seed (42), image size (512), device, etc. All paths reference the real dataset locations above |
| 0.4 | Create `src/utils/config.py` | YAML config loader with dot-notation access and merge support |
| 0.5 | Create `src/utils/seed.py` | Reproducibility: set seed for Python, NumPy, PyTorch, CUDA |
| 0.6 | Create `src/utils/logger.py` | Logging utility (console + file) |
| 0.7 | Create `src/utils/io.py` | File I/O helpers (save/load numpy, images, JSON) |
| 0.8 | Install dependencies | In `research` env: `torch`, `torchvision`, `torch-geometric`, `scikit-image`, `opencv-python`, `networkx`, `matplotlib`, `pyyaml`, `pandas`, `tqdm`, `scipy`, `Pillow` |
| 0.9 | Create `requirements.txt` | Pin all installed versions |
| 0.10 | Create `progression.md` | Initialize with project start timestamp and Stage 0 documentation |
| 0.11 | Validate dataset access | Write and run a quick script that checks all dataset paths exist, counts images/labels, reports basic stats |

### Deliverables for Review

- [ ] Full directory tree created and printable
- [ ] `conda activate research && python -c "import torch; print(torch.__version__)"` works
- [ ] Dataset validation script runs and reports correct counts
- [ ] `progression.md` has Stage 0 entry

### Verification Command
```bash
conda activate research && python -c "from src.utils.config import load_config; cfg = load_config('configs/base.yaml'); print(cfg)"
```

---

## Stage 1: Dataset & Preprocessing

**Goal:** Build all dataset classes, preprocessing pipelines, and data splits. Produce processed data ready for Stages 2 and 3.

### Tasks

| # | Task | Detail |
|---|---|---|
| 1.1 | `src/data/transforms/label_remap.py` | AI4Mars NAV→3-class risk mapping (Soil→0.1, Bedrock→0.5, Sand→0.4, Big Rock→0.9, NULL=255→ignore). HiRISE landmark→risk mapping (8 real classes). |
| 1.2 | `src/data/preprocessing/normalize.py` | Per-tile min-max normalization to [0,1] with ε=1e-8 |
| 1.3 | `src/data/preprocessing/resize.py` | Image resize (bilinear) + label resize (nearest-neighbour). Handle 227→512 for HiRISE |
| 1.4 | `src/data/preprocessing/augmentations.py` | Training augmentations: H/V flip, ±15° rotation, ±20% brightness/contrast, Gaussian noise σ~U(0,0.02). Same spatial transform applied to image AND label |
| 1.5 | `src/data/loaders/ai4mars_loader.py` | PyTorch Dataset: load MSL NavCam EDR images + labels, apply masks (rover+range), normalize, replicate to 3-ch, convert labels to continuous risk. Handles train/val/test splits |
| 1.6 | `src/data/loaders/hirise_loader.py` | PyTorch Dataset: load HiRISE v3 originals only (aug index 0), resize to 512, normalize, load image-level label from txt+classmap, remap to risk score |
| 1.7 | `src/data/loaders/ctx_loader.py` | Simple loader for MurrayLab tiles: load, normalize, quality check (reject >30% saturated), replicate to 3-ch |
| 1.8 | Create train/val/test splits | AI4Mars: 70% train / 15% val from crowdsourced, test = `masked-gold-min3-100agree`. Stratified by dominant terrain class. Save split lists to `data/splits/` |
| 1.9 | Dataset config YAMLs | `configs/datasets/ai4mars.yaml`, `hirise.yaml`, `ctx.yaml` with all paths and params |
| 1.10 | Verification notebook | `notebooks/data_analysis.ipynb`: visualize sample images+labels from each dataset, show class distribution histograms, verify label remapping |

### Deliverables for Review

- [ ] Load a batch of 8 AI4Mars images+labels → display as grid with risk colormap
- [ ] Class distribution histogram for AI4Mars train split
- [ ] Sample HiRISE crops displayed with their remapped risk class
- [ ] 3–5 selected CTX demo tiles displayed with quality check stats
- [ ] Split file counts printed: train / val / test
- [ ] `progression.md` updated with Stage 1 details

---

## Stage 2: Physics Feature Extraction

**Goal:** Implement domain-invariant physics features (slope, roughness, discontinuity) and produce `H_physics` maps.

### Tasks

| # | Task | Detail |
|---|---|---|
| 2.1 | `src/physics/slope.py` | Sobel gradient magnitude (G_x, G_y → G_mag), per-tile min-max normalize to [0,1] |
| 2.2 | `src/physics/roughness.py` | Local std dev in 7×7 sliding window, normalized to [0,1] |
| 2.3 | `src/physics/discontinuity.py` | Absolute LoG response (σ=2.0), normalized to [0,1] |
| 2.4 | `src/physics/combine.py` | H_physics = w1·S + w2·R + w3·D. Default weights: 0.4, 0.3, 0.3. Configurable via YAML |
| 2.5 | `src/data/transforms/physics_features.py` | Wrapper: given an image tensor, compute all 4 physics maps (S, R, D, H_physics) |
| 2.6 | Physics config | `configs/physics.yaml`: window sizes, σ, weights |
| 2.7 | Verification script | Run physics extraction on 5 AI4Mars images + 3 CTX tiles. Save side-by-side visualizations: Original | S | R | D | H_physics |

### Deliverables for Review

- [ ] Visual grid: 5 images × 5 columns (Original, Slope, Roughness, Discontinuity, H_physics)
- [ ] Same for 3 CTX tiles — demonstrate domain invariance visually
- [ ] Physics computation time per tile (should be <100ms on CPU)
- [ ] `progression.md` updated

---

## Stage 3: Deep Learning Risk Heatmap (CNN)

**Goal:** Train MobileNetV3-Large + DeepLabV3+ with sigmoid regression head to produce `H_learned` continuous risk maps.

### Tasks

| # | Task | Detail |
|---|---|---|
| 3.1 | `src/models/cnn/mobilenetv3.py` | MobileNetV3-Large encoder wrapper (pretrained ImageNet, configurable freeze layers) |
| 3.2 | `src/models/cnn/deeplabv3plus.py` | DeepLabV3+ decoder with ASPP (rates: 6,12,18) and skip connections |
| 3.3 | `src/models/cnn/risk_head.py` | Output head: Conv(decoder_out→1) → Sigmoid. Single-channel continuous risk output |
| 3.4 | `src/models/cnn/risk_model.py` | Full model assembly: encoder + decoder + head. Forward pass: (B,3,512,512) → (B,1,512,512) |
| 3.5 | Loss functions | Weighted BCE + Dice on hazardous region + TV smoothness loss. Implement in `src/training/losses.py` |
| 3.6 | `src/training/train_cnn.py` | Training loop: AdamW (wd=1e-4), LR=1e-4 cosine annealing, batch=8, 60 epochs, early stopping (patience=10 on val hazard recall) |
| 3.7 | `src/training/trainer.py` | Generic trainer class: handles device, logging, checkpointing, metric tracking |
| 3.8 | `src/evaluation/metrics.py` | Hazard recall, mean IoU, ECE (Expected Calibration Error), AUC-ROC |
| 3.9 | CNN config | `configs/cnn/mobilenetv3.yaml`, `configs/cnn/deeplabv3.yaml` |
| 3.10 | Training script | `scripts/train_cnn.py` — CLI entry point reading from config |
| 3.11 | Evaluate on AI4Mars test set | Compute hazard recall, mIoU, ECE on `masked-gold-min3` |
| 3.12 | Cross-domain eval on HiRISE v3 | Run trained CNN on HiRISE crops → patch-level risk → compare to ground truth. Report accuracy, precision, recall per class, AUC-ROC |

### Deliverables for Review

- [ ] Training loss + val metric curves (plotted)
- [ ] AI4Mars test results table: hazard recall, mIoU, ECE
- [ ] H_learned heatmap visualizations on 5 AI4Mars test images (side by side with input + ground truth)
- [ ] H_learned on 3 CTX demo tiles (qualitative)
- [ ] Cross-domain HiRISE results table: accuracy, per-class recall
- [ ] Domain gap comparison: AI4Mars test recall vs HiRISE recall
- [ ] Best model checkpoint saved to `checkpoints/cnn/`
- [ ] `progression.md` updated

---

## Stage 4: Adaptive Hybrid Risk Fusion

**Goal:** Build the learned spatial attention fusion that produces `H_final` and the interpretable α map.

### Tasks

| # | Task | Detail |
|---|---|---|
| 4.1 | `src/models/fusion/adaptive_fusion.py` | 3-layer lightweight CNN: Conv(3→16,3×3)→ReLU → Conv(16→8,3×3)→ReLU → Conv(8→1,1×1)→Sigmoid. Input: [H_physics \| H_learned \| original_image] stacked as 3 channels. Output: α map |
| 4.2 | Fusion forward pass | H_final = α · H_learned + (1-α) · H_physics |
| 4.3 | Joint training | Train fusion network jointly with CNN (Stage 3) — loss applied to H_final, not H_learned alone. Or train fusion separately with frozen CNN |
| 4.4 | `src/training/train_fusion.py` | Training script for fusion network (can also be joint with CNN) |
| 4.5 | Fusion config | `configs/fusion/adaptive_fusion.yaml` |
| 4.6 | Baseline: static fusion | Implement α=0.5 fixed globally as comparison (B4 baseline) |
| 4.7 | Evaluation | Compare fused hazard recall vs physics-only, learning-only, static fusion |
| 4.8 | Visualizations | α map overlays; H_physics vs H_learned vs H_final side-by-side |

### Deliverables for Review

- [ ] α map visualizations on 5 images — showing where system trusts physics vs CNN
- [ ] Side-by-side: H_physics | H_learned | H_final for 5 images
- [ ] Comparison table: physics-only recall, learning-only recall, static fusion recall, adaptive fusion recall
- [ ] Fusion model checkpoint saved
- [ ] `progression.md` updated

---

## Stage 5: Superpixel Graph Construction

**Goal:** Convert fused risk maps into sparse superpixel graphs with 14-dim node features and risk-weighted edges.

### Tasks

| # | Task | Detail |
|---|---|---|
| 5.1 | `src/graph/superpixels.py` | SLIC segmentation: K=300, compactness=10, on original grayscale tile. Return label map + segment properties |
| 5.2 | `src/graph/node_features.py` | Compute 14-dim feature vector per superpixel node: [mean_intensity, intensity_std, mean_S, mean_R, mean_D, mean_H_physics, mean_H_learned, mean_H_final, mean_α, segmentation_entropy, centroid_x, centroid_y, area, hazardous_neighbour_count] |
| 5.3 | `src/graph/adjacency.py` | Region Adjacency Graph (RAG) from SLIC label map. Edge weights: w(i,j) = αw·(avg_H_final) + βw·dist + γw·|S_i-S_j|. Default weights: 0.6, 0.25, 0.15 |
| 5.4 | `src/models/gnn/graph_builder.py` | Orchestrator: image → SLIC → node features → adjacency → PyG Data object. Node deactivation: mean(H_final) > 0.7 → virtual obstacle |
| 5.5 | Graph config | SLIC params, edge weight params, deactivation threshold — all in config |
| 5.6 | Verification | Build graphs for 5 AI4Mars images + 3 CTX tiles. Visualize: superpixel boundaries overlaid on image, graph structure with nodes colored by H_final, edges colored by weight |

### Deliverables for Review

- [ ] Superpixel boundary visualization overlaid on images
- [ ] Graph visualization: nodes as colored dots, edges as lines
- [ ] Stats: avg nodes per graph (~280-320), avg edges (~800-1200)
- [ ] Node feature distribution histograms
- [ ] `progression.md` updated

---

## Stage 6: GATv2 Traversability Refinement

**Goal:** Train GATv2 on superpixel graphs to refine traversability scores via neighbourhood context.

### Tasks

| # | Task | Detail |
|---|---|---|
| 6.1 | `src/models/gnn/gatv2.py` | GATv2 model: Layer1 GATv2Conv(14→32, 4 heads, concat→128), Dropout(0.3), ELU → Layer2 GATv2Conv(128→32, 4 heads, concat=False→32), Dropout(0.2), ELU → Linear(32→1), Sigmoid |
| 6.2 | Node labeling | Derive binary labels from AI4Mars ground truth: majority vote of pixel labels within each superpixel. Hazardous if mean risk > 0.5. Weak labeling: 2-hop neighbours of confirmed hazards |
| 6.3 | `src/training/train_gnn.py` | Training loop: BCE loss with positive weight=3.0 on hazardous class. Save attention weights for visualization |
| 6.4 | GNN config | `configs/gnn/gatv2.yaml`: hidden dims, heads, dropout, learning rate |
| 6.5 | Edge weight update | After GATv2: w'(i,j) = αw·(2 - p̂_i - p̂_j) + βw·d(i,j) + γw·|S_i-S_j|. Nodes with p̂_i < 0.2 deactivated |
| 6.6 | Evaluation | Node AUC-ROC. Compare graph before/after GATv2 refinement |
| 6.7 | Ablation | B5 (No-GNN) vs Proposed: run A* on both graphs, compare HCR |

### Deliverables for Review

- [ ] Training curves (loss, node AUC-ROC)
- [ ] Before/after GATv2 risk map comparison (node coloring)
- [ ] Attention weight visualization on sample graphs
- [ ] Node AUC-ROC on test set
- [ ] GATv2 model checkpoint saved
- [ ] `progression.md` updated

---

## Stage 7: Safe Path Planning (A*)

**Goal:** Implement A* with physics-aware heuristic on the refined graph. Produce final paths with per-waypoint risk reports.

### Tasks

| # | Task | Detail |
|---|---|---|
| 7.1 | `src/planning/heuristics.py` | Physics-aware heuristic: h(n) = d_Euclidean(n,g) · (1 + γr·(1-p̂_n) + γs·S_n). γr=0.4, γs=0.1 |
| 7.2 | `src/planning/astar.py` | A* on NetworkX graph with updated edge weights. Returns: node sequence, waypoint coords, per-waypoint risk, per-waypoint dominant risk source (physics vs learned from α) |
| 7.3 | Path visualization | `src/visualization/paths.py`: overlay path on original image with risk coloring |
| 7.4 | Full pipeline inference | `src/inference/pipeline.py`: image → Stage 1-7 → path output. Single-call interface |
| 7.5 | `src/inference/run_inference.py` | CLI entry point: run full pipeline on a given image |
| 7.6 | Baselines | Implement all 7 baselines (B1–B5 + Proposed + Oracle) — see blueprint §9.1 |
| 7.7 | `src/evaluation/evaluate_ai4mars.py` | Quantitative eval on AI4Mars test: HCR, PLR, compute time, success rate |
| 7.8 | `src/evaluation/evaluate_hirise.py` | Cross-domain eval on HiRISE v3 |
| 7.9 | Demo on CTX | `src/evaluation/demo_ctx.py`: run full pipeline on 3-5 CTX tiles, produce all 5 mandatory visualizations from blueprint §10.5 |
| 7.10 | Results tables | Generate final comparison table (all 7 systems × 4 metrics) |

### Deliverables for Review

- [ ] Path visualizations: B1 vs Proposed vs Oracle side-by-side (the "money shot")
- [ ] Full results table: all 7 systems × HCR, PLR, compute time, success rate
- [ ] 5 mandatory visualizations from §10.5: (1) H_physics vs H_learned vs H_final, (2) α map overlay, (3) before/after GATv2, (4) path comparison B1/Proposed/Oracle, (5) baseline table
- [ ] CTX demo figures with paths overlaid
- [ ] Per-waypoint risk breakdown for sample paths
- [ ] HiRISE cross-domain gap table: CNN-only vs physics-only vs hybrid recall
- [ ] `progression.md` updated

---

## Stage 8: Integration, Ablation & Final Polish

**Goal:** Run complete ablation study, finalize all results, and prepare reproducible experiment bundles.

### Tasks

| # | Task | Detail |
|---|---|---|
| 8.1 | Ablation study | `src/evaluation/ablation.py`: systematically remove one component at a time. Generate the delta table from §11 |
| 8.2 | Physics weight grid search | Sweep w1,w2,w3 on validation set. Report optimal weights |
| 8.3 | Experiment configs | Create `configs/experiments/` YAMLs for each baseline and ablation |
| 8.4 | Experiment bundles | Each experiment saves: config.yaml + metrics.json + logs/ + checkpoints/ + visualizations/ |
| 8.5 | Final figures | Publication-quality figures for all 5 mandatory visualizations |
| 8.6 | `README.md` | Project documentation with setup, usage, and reproduction instructions |
| 8.7 | Final `progression.md` update | Complete summary of everything done |

### Deliverables for Review

- [ ] Complete ablation delta table (B2 vs B3 vs B4 vs B5 vs Proposed)
- [ ] Optimal physics weights reported
- [ ] All experiment bundles in `experiments/` with reproducible configs
- [ ] Publication-ready figures
- [ ] Full `progression.md`

---

## Execution Notes

### Environment

```
Interpreter: Anaconda → research
Working directory: d:\Mars\pa-gnn\
All Python commands: conda activate research && python ...
```

### Config-Driven Design

Every hyperparameter, path, and setting goes through YAML configs. **No hardcoded values in Python files.** This allows:
- Switching datasets by changing one config
- Running ablations by swapping config files
- Full reproducibility by saving experiment configs

### Progression Documentation

After completing each stage, `progression.md` will be updated with:
1. **What was done** — files created/modified, with descriptions
2. **How it was done** — key implementation decisions and rationale
3. **Verification results** — actual outputs, metrics, screenshots
4. **Issues encountered** — any problems and how they were resolved
5. **Timestamp** — when the stage was completed

### Modularity Principles

- Each stage is independently testable
- Swapping components (e.g., MobileNetV3 → Swin-T) requires changing only one file + one config
- No circular dependencies between stages
- All data flows through well-defined interfaces (tensors, PyG Data objects, NetworkX graphs)
