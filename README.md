# PA-GNN: Physics-Aware Graph Neural Network Pipeline

**Physics-Aware Graph Neural Network Pipeline for Autonomous Planetary Path Planning**  
From Orbital Imagery to Safe Rover Routes via Hybrid Terrain Risk Estimation

## Overview

This project implements a 7-stage pipeline that transforms Mars orbital imagery into safe rover traversal paths:

1. **Preprocessing** — Normalize images, remap labels to continuous risk scores
2. **Physics Feature Extraction** — Slope, roughness, depth discontinuity from image gradients
3. **CNN Risk Heatmap** — MobileNetV3 + DeepLabV3+ for learned terrain risk
4. **Adaptive Hybrid Fusion** — Learned spatial attention to blend physics + CNN signals
5. **Superpixel Graph Construction** — SLIC + RAG with 14-dim node features
6. **GATv2 Traversability Refinement** — Graph attention for contextual risk propagation
7. **A* Path Planning** — Physics-aware heuristic for safe waypoint generation

## Setup

```bash
conda activate research
pip install -r requirements.txt
```

## Usage

```bash
# Train CNN (Stage 3)
python scripts/train_cnn.py --config configs/experiments/exp_full_pipeline.yaml

# Run full pipeline inference
python scripts/run_pipeline.py --config configs/base.yaml --image <path_to_tile>

# Evaluate
python scripts/evaluate.py --config configs/experiments/exp_full_pipeline.yaml
```

## Project Structure

See `folder_structure.md` for the complete directory layout.

## Authors

- Syed Abir Hossain (20220104013)
- Ashik Mahmud (20220104021)
- Mahadir Rahaman (20220104046)

**Supervised By:** Tamanna Tabassum, Assistant Professor, Dept. of CSE, AUST
