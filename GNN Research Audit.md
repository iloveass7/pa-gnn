# PA-GNN Research Audit — Military-Precision Review (v2 CORRECTED)

> **Project:** Physics-Aware Graph Neural Network Pipeline for Autonomous Planetary Path Planning
> **Audit Date:** 2026-05-08 (Corrected 2026-05-09)
> **Scope:** Full codebase, results, blueprint alignment, experimental rigor
> **Classification:** 🔴 6 Critical | 🟠 9 Moderate | 🟡 5 Minor

> [!NOTE]
> **v2 Correction:** The original audit incorrectly stated CNN/Fusion models were never trained. Training histories confirm all three models were trained. Finding C1 has been **retracted and replaced** with a new finding about CNN underfitting. All downstream assessments have been recalibrated.

---

## Executive Summary

The pipeline architecture is **structurally complete** — all 7 stages exist, are trained, run end-to-end, and produce results. All three models (CNN, Fusion, GNN) have confirmed training histories with real convergence. However, the research has **critical gaps** that would be exposed under thesis examination:

1. **The CNN is severely underfitting** (dice_loss ≈ 0.96, hazard_recall = 0.65) — the learned risk channel is weak
2. **60% success rate** on the primary dataset is a defense liability
3. **Two blueprint-mandated baselines (B5, Oracle) are missing** — the GNN's contribution is unproven in isolation
4. **The GNN trains on its own input feature** — circularity concern needs explicit defense
5. **The A* heuristic is inadmissible** — optimality claims are technically false
6. **No statistical significance testing** — results are from a single 50-sample run

### Training Verification Summary

| Model | Epochs | Best val_hazard_recall | Checkpoint |
|---|---|---|---|
| CNN | 21 | **0.652** (epoch 11) | `checkpoints/cnn/best_model.pth` (134 MB) |
| Fusion | 15 | **0.651** (epoch 5) | `checkpoints/fusion/best_model.pth` (45 MB) |
| GNN | 31 | **0.801** (epoch 17), AUC=0.880 | `checkpoints/gnn_fast/best_gat_model.pth` (470 KB) |

---

## 🔴 CRITICAL FINDINGS (Thesis-Threatening)

### C1: CNN Severely Underfitting — Foundation is Weak

**Evidence:**
- CNN `val_dice_loss` barely moved: 0.962 → 0.954 over 21 epochs (near-random for binary segmentation)
- CNN `val_hazard_recall` peaked at **0.652** — the model misses **35% of hazards**
- `val_bce` plateaued at ~0.570 — minimal learning after epoch 3
- `train_loss` dropped only from 1.073 → 1.042 — extremely slow convergence
- Early stopping triggered after 21 epochs (patience=10 from epoch 11)

**Impact:** The CNN's `H_learned` is barely better than a constant prediction. This means:
- The fusion model's `alpha` map has a weak signal to work with — the fusion cannot distinguish good CNN regions from bad ones when the CNN is uniformly mediocre
- Node features 6 (`H_learned`) and 7 (`H_final`) carry limited discriminative power
- The GNN's H_final-derived labels (feature 7 > 0.5) are based on a weak fusion of weak CNN + physics

**Root Cause Candidates:**
1. Learning rate 1e-4 may be too conservative for fine-tuning MobileNetV3
2. Batch size 8 with ~13K training images = 1,654 steps/epoch — potentially insufficient gradient accumulation
3. The composite loss (BCE + Dice + TV) may have conflicting gradients
4. Per-tile min-max normalization removes global intensity context that helps terrain recognition

**Solution:**
1. **Increase LR to 3e-4 or 5e-4** with warmup (first 2 epochs at 1e-5, then ramp up)
2. **Unfreeze encoder gradually:** First train decoder-only for 5 epochs, then unfreeze encoder with 10× lower LR
3. **Remove TV loss initially** — let the model learn the signal first, add smoothness later
4. **Increase batch size to 16** (RTX 3060 Ti has 8GB — should fit with mixed precision)
5. **Target: val_hazard_recall > 0.85** before proceeding to fusion training

---

### C2: 60% Success Rate on AI4Mars — Defense Liability

**Evidence:**
- [ai4mars_results.csv](file:///d:/Mars/pa-gnn/results/stage7_eval/ai4mars_results.csv): `proposed` success rate = 0.60
- Blueprint §10.4 target: Success Rate > 95%
- All baselines achieve 94–100% success rate

**Impact:** A reviewer's first question: "Your system fails to find a path 40% of the time — how is this usable?" The 12× HCR improvement is meaningless if the system can't find paths.

**Root Cause:** The GNN deactivation threshold (`risk > 0.70`) creates impenetrable barriers of blocked nodes across the image. The start/goal force-activation ([pipeline.py](file:///d:/Mars/pa-gnn/src/inference/pipeline.py) L142-143) doesn't help when the entire mid-graph is blocked.

**Solution:**
1. **Adaptive threshold relaxation:** If A* fails with threshold 0.70, retry with 0.80, then 0.90. Report the relaxed threshold used
2. **Connectivity-aware deactivation:** Before deactivating nodes, check if deactivation would disconnect the graph. Only deactivate if connectivity is preserved
3. **Report conditional metrics:** "Among the 60% successful paths, HCR = 0.014" — make this explicit
4. **Thesis framing:** Present failures as "the system correctly identifies that no safe path exists" — but validate that blocked regions genuinely are hazardous

---

### C3: Missing Baselines B5 (No-GNN) and Oracle

**Evidence:**
- Blueprint §9.1 specifies **7 baselines**: B1–B5, Proposed, Oracle
- Code only implements 5: `b1_euclidean`, `b2_physics`, `b3_learned`, `b4_static`, `proposed`
- **B5 (No-GNN):** A* on the fused graph *without* GATv2 refinement — isolates the GNN's contribution
- **Oracle:** A* on perfect ground-truth labels — the upper bound

**Impact:** Without B5, you cannot prove the GNN adds value. A reviewer can argue: "Your improvement comes from the fusion, not the GNN." Without Oracle, you cannot measure how close you are to theoretical optimum.

**Solution:**
1. Add `b5_no_gnn` mode to `pipeline.py`: skip the GATv2 forward pass, use `H_final` directly as node risk scores
2. Add `oracle` mode: use ground-truth risk map from AI4Mars labels
3. The key results: **B5 vs Proposed** (GNN contribution) and **Proposed vs Oracle** (remaining gap)

---

### C4: GNN Self-Supervision Circularity

**Evidence:**
- [train_gnn_fast.py](file:///d:/Mars/pa-gnn/scripts/train_gnn_fast.py) L63: `targets = (data.x[:, 7] > 0.5).float()`
- Feature index 7 = `mean_H_final` — the GNN is trained to predict a binarized version of its own input feature
- The GNN has direct access to the answer in `x[7]`

**Impact:** A reviewer will ask: "What is the GNN actually learning beyond a linear readout of x[7]?"

**Mitigating Evidence:**
- Diagnosis showed mean |GNN - H_final| = 0.154 and 78.4% of nodes changed by > 0.05
- The GNN IS using neighborhood context, but this needs explicit defense

**Solution:**
1. **Ablation: Remove H_final from input.** Train with 13-dim features (drop index 7). If performance holds, the GNN truly learns from neighborhood context
2. **Alternative: Use ground-truth labels.** Address the 1.9% positive rate with focal loss
3. **Defense line:** "H_final provides a noisy per-node estimate; the GNN refines it using 2-hop neighborhood attention, correcting boundary misclassifications"

---

### C5: Heuristic Admissibility Violation

**Evidence:**
- [heuristics.py](file:///d:/Mars/pa-gnn/src/planning/heuristics.py) L22: `h(n) = d_euc * (1.0 + gamma_r * risk_n + gamma_s * s_n)`
- A* requires `h(n) ≤ true cost to goal` for optimality
- Multiplying Euclidean distance by `(1 + risk + slope)` can **overestimate** true cost → inadmissible

**Impact:** The thesis claims "optimal path on the constructed graph" (progression.md L329). Technically false.

**Solution:**
1. **Option A:** Use pure Euclidean `h(n) = d_euc` (admissible). Risk is already in edge weights `g(n)`. A* finds truly optimal path
2. **Option B:** Keep it but call it "weighted A*" in the thesis. Report the suboptimality trade-off
3. **Option C:** Bound the inflation: `h(n) = d_euc * (1 + ε * ...)` with ε small enough to guarantee bounded suboptimality

---

### C6: No Statistical Significance Testing

**Evidence:**
- All results are single-run means over 50 samples
- No confidence intervals, standard deviations, or p-values
- No multiple-seed experiments

**Impact:** A reviewer can dismiss the 12× HCR improvement as noise.

**Solution:**
1. Run evaluation 5× with different seeds (randomized start/goal pairs)
2. Report mean ± std for all metrics
3. Paired Wilcoxon signed-rank test for Proposed vs. each baseline
4. Error bars on all comparison charts

---

## 🟠 MODERATE FINDINGS (Weaknesses to Address)

### M1: Weak Labeling Disabled — Blueprint Feature Unused

- [gatv2.yaml](file:///d:/Mars/pa-gnn/configs/gnn/gatv2.yaml) L61: `weak_labeling.enabled: false`
- [weak_labels.py](file:///d:/Mars/pa-gnn/src/training/weak_labels.py) fully implemented but never called
- Blueprint §7.7.3 specifies "weak labelling of 2-hop neighbours"
- **Fix:** Enable and test, or explain in thesis why it was empirically unnecessary

### M2: HiRISE Evaluation Is Self-Referential

- [evaluate_hirise.py](file:///d:/Mars/pa-gnn/src/evaluation/evaluate_hirise.py): HCR computed from model's OWN predicted risk scores, not ground-truth
- The "2.6× HCR improvement" on HiRISE is measuring self-consistency, not accuracy
- **Fix:** Acknowledge limitation; add qualitative visual validation; correlate model risk with HiRISE class labels

### M3: B2/B3 Produce Higher HCR Than B1 (Counter-intuitive)

- AI4Mars: B1=0.179, B2=0.212, B3=0.214 — physics and learning alone are WORSE than Euclidean
- B2/B3 use risk-weighted edges but don't deactivate hazardous nodes → paths cost more but don't actually avoid hazards
- **Fix:** Add node deactivation to B2/B3, or explain as "single-signal insufficiency" motivating fusion

### M4: B4 Static Fusion Not Properly Implemented

- [pipeline.py](file:///d:/Mars/pa-gnn/src/inference/pipeline.py) L128-129: B4 falls through to `else: w = edge_attr[i]`, using precomputed adaptive fusion weights
- B4 should use `get_static_fusion(h_physics, h_learned, alpha=0.5)` for edge weights
- **Fix:** Implement proper static α=0.5 fusion in the B4 branch

### M5: Node Feature Normalization Missing

- [node_features.py](file:///d:/Mars/pa-gnn/src/graph/node_features.py): 14-dim features have wildly different scales
  - `area`: ~100–2000 | `centroid_x/y`: 0–512 | `mean_intensity`: 0–1 | `haz_neighbor_count`: 0–6+
- GATv2 attention is scale-sensitive — large features dominate
- **Fix:** Per-feature z-score or min-max normalization

### M6: Checkpoint 2 (Degradation Robustness) Not Started

- Blueprint §12 specifies degradation robustness as Checkpoint 2
- No code, configs, or results exist
- **Fix:** Implement or explicitly scope as "Future Work"

### M7: Stale Argparse Description in `train_gnn_fast.py`

- L135: Still says "SmoothL1 regression" — was changed to BCE classification (v3)
- **Fix:** Update description string

### M8: GNN Hazard Recall Below Target

- `val_hazard_recall = 0.8014` vs thesis target > 0.90
- `val_hazard_precision = 0.3224` — 3:1 false positive ratio
- **Fix:** Improve CNN first (C1); try focal loss; experiment with 3-layer GATv2

### M9: No GNN Attention Weight Visualization

- GATv2's key advantage is dynamic attention — never visualized
- Missing interpretability opportunity for "contextual refinement" narrative
- **Fix:** Use `return_attention_weights=True` in GATv2Conv, visualize as edge thickness

---

## 🟡 MINOR FINDINGS (Polish Items)

### m1: Hardcoded 50-Sample Evaluation Cap
- `evaluate_ai4mars.py` L24: `if i >= 50: break` — only 15% of 322 test samples evaluated
- **Fix:** CLI arg, run full test set for final numbers

### m2: Fixed Start/Goal Coordinates
- All evals use `(10%, 10%)` → `(90%, 90%)` diagonal — biases toward single path topology
- **Fix:** Random start/goal sampling, multiple per image

### m3: No TensorBoard Integration
- Blueprint mentions `logs/tensorboard/` but none exists
- **Fix:** Add `SummaryWriter` for training curves

### m4: Missing Thesis Figures
- No `stage6/` results directory; §10.5 "key visualization outputs" not all generated
- **Fix:** Generate publication-quality figures for all 5 required visualizations

### m5: Minimal README
- Only 1.6 KB — no reproduction instructions
- **Fix:** Document full pipeline reproduction steps

---

## Alignment Check: Blueprint vs. Implementation

| Blueprint Requirement | Status | Gap |
|---|---|---|
| 7-stage pipeline | ✅ Complete | — |
| MobileNetV3 + DeepLabV3+ | ✅ Trained (21 epochs) | 🔴 Underfitting (recall=0.65) |
| Adaptive fusion (3-layer CNN) | ✅ Trained (15 epochs) | 🟡 Limited by weak CNN |
| SLIC K=300, compactness=10 | ✅ Correct | — |
| 14-dim node features | ✅ All 14 present | 🟠 Not normalized |
| GATv2 (14→128→32→1) | ✅ Trained (31 epochs) | — |
| Weak labeling (2-hop) | ✅ Implemented | 🟠 Disabled |
| A* with physics heuristic | ✅ Implemented | 🔴 Inadmissible |
| 7 baselines (B1–B5, Proposed, Oracle) | ❌ Only 5 | 🔴 B5 + Oracle missing |
| HCR < 5% | ✅ 1.4% (AI4Mars) | — |
| PLR < 1.30 | ❌ Not measured | 🔴 Missing metric |
| Success Rate > 95% | ❌ 60% | 🔴 Critical failure |
| Compute < 5s CPU | ✅ ~1s GPU | 🟡 Not tested on CPU |
| Hazard Recall > 0.90 (GNN) | ❌ 0.80 | 🟠 Below target |
| Checkpoint 2 (degradation) | ❌ Not started | 🟠 Scope unclear |
| Cross-validate physics/MOLA (r>0.65) | ❌ Not done | 🟡 Blueprint mention |
| Path Length Ratio (PLR) | ❌ Not computed | 🔴 Missing metric |

---

## Recommended Priority Order

| # | Action | Time Est. | Impact |
|---|---|---|---|
| **1** | Retrain CNN with higher LR + gradual unfreeze (C1) | 4–8 hrs GPU | Fixes weak foundation |
| **2** | Retrain Fusion on improved CNN | 2–4 hrs GPU | Improves H_final quality |
| **3** | Recompute graphs + retrain GNN | 4–6 hrs GPU | Cascading improvement |
| **4** | Add B5 + Oracle baselines (C3) | 2 hrs code | Complete ablation table |
| **5** | Fix success rate — adaptive threshold (C2) | 1 hr code | Removes defense liability |
| **6** | Add PLR metric (C6→now missing metric) | 30 min code | Blueprint compliance |
| **7** | Fix B4 static fusion (M4) | 30 min code | Correct baseline |
| **8** | Add node feature normalization (M5) | 1 hr code | Potentially improves GNN |
| **9** | Multi-seed evaluation (C6) | 3 hrs compute | Statistical credibility |
| **10** | Fix heuristic admissibility (C5) | 30 min code | Correctness claim |
| **11** | GNN attention visualization (M9) | 2 hrs code | Interpretability |
| **12** | Full 322-sample evaluation (m1) | 2 hrs compute | Complete results |

> [!IMPORTANT]
> **The single most impactful action is improving the CNN (C1).** With hazard_recall at 0.65, the learned risk channel is barely contributing. A properly trained CNN will cascade improvements through fusion → graph features → GNN labels → path quality. Target val_hazard_recall > 0.85 before proceeding.
