# PA-GNN Final Review — Current Flaws & Improvements

**After auditing both documents AND the full codebase**  
**Date:** May 1, 2026

---

## Current Status

Your code is in significantly better shape than the documents suggest — most of the critical architecture decisions are correctly implemented. This review only covers **issues that actually exist right now**, not things already fixed.

---

## Part 1: Flaws That Exist Right Now

### 🔴 Flaw 1 — `train_gnn.py` Will Crash at Validation

**Where:** [train_gnn.py L84-L85](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/scripts/train_gnn.py#L84-L85)

**What:** The `evaluate_gnn()` function uses `np.array()` but `import numpy as np` is missing from the imports.

**Impact:** GNN training will run fine for forward passes, but the **first validation epoch will crash** with `NameError: name 'np' is not defined`.

**Fix:**
```diff
 import torch
 import torch.nn.functional as F
 import torch.optim as optim
+import numpy as np
 from torch.utils.data import DataLoader
 from sklearn.metrics import roc_auc_score
```

---

### 🔴 Flaw 2 — Missing `edge_case` HiRISE Class May Cause Runtime Crash

**Where:** [label_remap.py L131-L141](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/src/data/transforms/label_remap.py#L131-L141) and [hirise.yaml L17-L26](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/configs/datasets/hirise.yaml#L17-L26)

**What:** Datasets.md lists **9** HiRISE landmark classes (including `edge_case`), but the code and config only handle **8** classes (indices 0-7). If the actual `landmarks_map-proj-v3_classmap.csv` contains an `edge_case` class mapped to index 8, any image with that label will trigger:

```
ValueError: Unknown HiRISE class index: 8. Expected 0-7.
```

**Impact:** Cross-domain evaluation on HiRISE will crash partway through if `edge_case` images exist in the dataset.

**Fix:** Check the actual CSV file. If it has 9 classes, add to `label_remap.py`:
```python
DEFAULT_RISK_MAP = {
    ...
    7: 0.45,   # spider
    8: 0.50,   # edge_case → uncertain (default)
}
```
And add `edge_case: 0.50` to `hirise.yaml`. If the CSV only has 8 classes, update Datasets.md to remove `edge_case`.

---

### 🔴 Flaw 3 — Fusion Default Config Enables Joint Training (Lazy-Learning Risk)

**Where:** [adaptive_fusion.yaml L29](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/configs/fusion/adaptive_fusion.yaml#L29)

**What:** `joint_with_cnn: true` means the CNN is **not frozen** during fusion training. Gradients from the fusion loss flow back through H_learned into the CNN encoder. This can cause the CNN to "lazy-learn" — producing intentionally noisy H_learned because it learns the fusion network will compensate.

**Why it matters:** You've already built the two-phase infrastructure (CNN checkpoint loading in `train_fusion.py`, `freeze_cnn` flag in `fusion_model.py`). The intended workflow is clearly: train CNN → freeze → train fusion on top. But the config contradicts this by defaulting to joint training.

**Fix:**
```yaml
# In configs/fusion/adaptive_fusion.yaml
training:
  joint_with_cnn: false    # ← Change from true to false
```

Then run fusion training as:
```bash
python scripts/train_fusion.py --cnn_ckpt checkpoints/cnn/best_model.pth
```

---

### 🟡 Flaw 4 — HiRISE Patch-Level Aggregation Is Undefined

**Where:** [evaluate_hirise.py](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/src/evaluation/evaluate_hirise.py) and the pipeline's `run()` method

**What:** The pipeline produces a pixel-level risk map (512×512), but HiRISE ground-truth labels are image-level (one scalar per 227×227 crop). There is no explicit function that converts the pixel-level output to a single patch-level risk score for comparison.

The evaluation calls `evaluate_dataset()` which was designed for AI4Mars pixel-level evaluation — it likely doesn't handle the image-level comparison correctly for HiRISE.

**Impact:** Cross-domain evaluation metrics may be computed incorrectly or not at all.

**Fix:** Add a dedicated aggregation function:
```python
# In src/evaluation/metrics.py
def aggregate_patch_risk(h_final, original_size=227, target_size=512):
    """
    Aggregate pixel-level H_final to a single patch risk score.
    Uses the mean of the central region corresponding to original image content.
    """
    # Compute padding offset (content is centered after resize)
    pad = (target_size - original_size) // 2  # ~142 pixels
    
    # Extract central region (original content, no padding artifacts)
    central = h_final[:, :, pad:pad+original_size, pad:pad+original_size]
    
    return central.mean().item()
```

---

### 🟡 Flaw 5 — Deactivation Threshold Hardcoded, Config Ignored

**Where:** [pipeline.py L88](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/src/inference/pipeline.py#L88) vs [gatv2.yaml L61](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/configs/gnn/gatv2.yaml#L61)

**What:** The config defines `deactivation_threshold: 0.2`, but the pipeline hardcodes `risk_scores[u] > 0.8`. They're semantically equivalent (risk > 0.8 ≈ traversability < 0.2), but the config value is never read.

**Impact:** If you change the config threshold for experiments, nothing will change. Subtle but confusing during ablations.

**Fix:**
```python
# In pipeline.py, read from config
deact_thresh = self.gat_cfg.graph.deactivation_threshold
# ...
if risk_scores[u] > (1.0 - deact_thresh):
    G.nodes[u]['active'] = False
```

---

### 🟡 Flaw 6 — Documents Are Stale in Multiple Places

These won't break anything, but they **will confuse you** if you re-read the docs during writing:

| Document | What It Says | What Code Does | Fix |
|---|---|---|---|
| **Datasets.md** §8 Stage 3 | "Focal loss with γ=2" | BCE + Dice + TV | Update to match code |
| **Datasets.md** §8 Stage 5 | "8-dim node features" | 14-dim features | Update to match code |
| **Datasets.md** §8 Stage 6 | "hidden dim 64" | 32 per head × 4 heads = 128 → 32 | Update to match code |
| **Datasets.md** §8 Stage 4 | "texture entropy in 15×15 window" | 3-channel CNN [H_physics, H_learned, image] | Update to match code |
| **Blueprint** §7.4.1 | "Safe→0.05, Hazardous→0.95" | Safe→0.1, Hazardous→0.9 | Update Blueprint to match code |
| **Datasets.md** §6.2 | 9 HiRISE classes (incl. `edge_case`) | 8 classes in code | Verify against actual CSV |

---

### 🟡 Flaw 7 — No Physics Weight Tuning Infrastructure

**Where:** [physics.yaml](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/configs/physics.yaml) — weights are hardcoded `w_s: 0.4, w_r: 0.3, w_d: 0.3`

**What:** Datasets.md Section 13 says *"The physics feature weights must be tuned on the validation set"* via grid search. No such script exists. You're running with the initial defaults.

**Impact:** The defaults might work fine, but a reviewer could ask *"how were these weights selected?"* and you'd have no answer beyond "initial guess."

**Fix:** Create `scripts/tune_physics_weights.py` that:
1. Sweeps w_s ∈ {0.3, 0.4, 0.5}, w_r ∈ {0.2, 0.3, 0.4}, w_d ∈ {0.2, 0.3, 0.4} (with w_s+w_r+w_d=1)
2. Computes H_physics hazard recall on AI4Mars validation
3. Reports the best combination
4. Shows HiRISE metrics are stable across ±0.1 (sensitivity analysis)

---

## Part 2: Improvements to Make the System Stronger

### Improvement 1 — Add Confidence Calibration to CNN Output

**Current state:** [metrics.py](file:///Users/ashikmahmud/Documents/Thesis/pa-gnn/src/evaluation/metrics.py#L30-L55) computes ECE with 10 bins.

**Problem:** 10 bins is coarse for continuous risk scores. The literature standard is 15 equal-width bins. More importantly, the ECE computation uses raw risk targets (which are discrete: 0.1, 0.4, 0.5, 0.9) as "accuracy" — this isn't standard calibration measurement.

**Improvement:** 
- Change to 15 bins 
- Add a reliability diagram visualization (predicted confidence vs actual accuracy per bin)
- Add post-hoc temperature scaling calibration as an optional step after CNN training
- Report both pre-calibration and post-calibration ECE

---

### Improvement 2 — Add Inference Time Budget per Stage

**Current state:** Blueprint targets <5 sec total on CPU but doesn't break it down. No timing code exists.

**Improvement:** Add a `--benchmark` flag to the pipeline that reports time per stage:

```python
# In pipeline.py run()
import time

timings = {}
t0 = time.time()
fusion_dict = self.fusion_model(img_b)
timings['stages_1-4'] = time.time() - t0

t0 = time.time()
data = self.graph_builder.build(image, fusion_dict)
timings['stage_5_graph'] = time.time() - t0

t0 = time.time()
preds = self.gat_model(data.x, data.edge_index, data.edge_attr)
timings['stage_6_gnn'] = time.time() - t0

t0 = time.time()
path_details = run_astar(G, start_node, goal_node, ...)
timings['stage_7_astar'] = time.time() - t0
```

Expected budget: Stage 2 (~0.1s), Stage 3 (~2.5s), Stage 4 (~0.1s), Stage 5 (~0.5s), Stage 6 (~0.3s), Stage 7 (~0.3s) ≈ **~3.8s total**. If any stage exceeds budget, you know where to optimize.

---

### Improvement 3 — Add Grad-CAM Visualization for Complementarity Argument

**Current state:** The α map shows where the system trusts physics vs CNN. But it doesn't show **what the CNN sees**.

**Improvement:** Add Grad-CAM on the MobileNetV3 encoder to visualize which image regions the CNN focuses on. Side-by-side with the Sobel slope map, this creates a compelling figure:

```
[Original tile] | [CNN Grad-CAM] | [Physics slope] | [α map]
```

This directly visualizes the complementarity: CNN focuses on crater rims (circular patterns), physics focuses on slopes (gradient edges). The α map shows the system correctly delegating to each channel.

---

### Improvement 4 — Add Failure Mode Analysis

**Current state:** No discussion of what terrain the system fundamentally cannot handle.

**Improvement:** Add a section documenting known blind spots:

| Failure Mode | Why It Fails | Mitigation |
|---|---|---|
| Subsurface ice | Flat, smooth, low gradient — invisible to both physics and CNN | Cannot detect from orbital imagery alone — requires spectrometer data |
| Dust-covered rocks | Albedo matches surrounding regolith, gradient is suppressed by dust layer | CNN might learn dust-cover patterns if training data includes them |
| Deep shadow regions | Zero information in dark pixels — all features collapse to zero | Quality filter (saturation check) should reject these tiles |
| Very fine sand vs compacted regolith | Visually identical, physics features identical — only distinguishable via contact mechanics | Limitation of orbital-only approach; active rover data needed |

---

### Improvement 5 — Multi-Run Variance Reporting

**Current state:** Results will be from a single training run.

**Improvement:** Train 3 runs with different seeds (42, 123, 456). Report mean ± std for all metrics. This proves your results are stable and not a lucky seed. The total cost is 3× training time but it makes results significantly more credible.

Add to config:
```yaml
evaluation:
  seeds: [42, 123, 456]
  report_mean_std: true
```

---

### Improvement 6 — MurrayLab Tile Selection Algorithm

**Current state:** Manual tile selection for demo figures.

**Improvement:** Automate selection based on H_physics diversity:

```python
def select_diverse_tiles(tile_paths, n_select=5):
    """Select tiles that maximize H_physics histogram diversity."""
    histograms = []
    for path in tile_paths:
        img = load_and_preprocess(path)
        h_phys = compute_physics(img)
        hist, _ = np.histogram(h_phys.flatten(), bins=20, range=(0, 1))
        histograms.append(hist / hist.sum())  # normalize
    
    # Greedy diversity selection
    selected = [0]  # start with first
    for _ in range(n_select - 1):
        max_div, best_idx = -1, -1
        for i in range(len(histograms)):
            if i in selected: continue
            # Min distance to any already-selected tile
            min_dist = min(
                np.sum((histograms[i] - histograms[j])**2) 
                for j in selected
            )
            if min_dist > max_div:
                max_div = min_dist
                best_idx = i
        selected.append(best_idx)
    
    return [tile_paths[i] for i in selected]
```

---

## Priority Order

```
MUST FIX (before next training run):
  1. Add numpy import to train_gnn.py           [2 minutes]
  2. Verify edge_case class in HiRISE CSV        [5 minutes]
  3. Set joint_with_cnn: false in fusion config   [1 minute]

SHOULD FIX (before evaluation):
  4. Add HiRISE patch-level aggregation function  [30 minutes]
  5. Read deactivation_threshold from config      [5 minutes]
  6. Update stale documents                       [1 hour]

SHOULD ADD (before paper):
  7. Physics weight tuning script                 [2 hours]
  8. Inference timing benchmark                   [1 hour]
  9. Multi-run variance (3 seeds)                 [3× training time]

NICE TO HAVE (strengthens paper):
  10. Grad-CAM visualization                      [2 hours]
  11. Failure mode analysis section               [1 hour]
  12. Automated tile selection                    [1 hour]
  13. Calibration improvements (15-bin ECE)       [1 hour]
```

---

*This review covers all currently existing flaws after accounting for everything already fixed in the codebase.*
