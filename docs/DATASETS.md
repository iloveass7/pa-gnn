# PA-GNN Dataset Reference Document
### Physics-Aware Graph Neural Network Pipeline for Autonomous Planetary Path Planning

**Project:** Thesis / CSE-4733 — Ahsanullah University of Science and Technology  
**Pipeline name:** PA-GNN (Physics-Aware Graph Neural Network)  
**Document purpose:** Complete, self-contained dataset reference. Any person or system reading this document should be able to understand what data exists, what it contains, how it is structured, what transformations are applied to it, and exactly where each dataset feeds into the 7-stage pipeline.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Dataset Summary Table](#2-dataset-summary-table)
3. [Dataset 1 — AI4Mars Merged Dataset](#3-dataset-1--ai4mars-merged-dataset)
4. [Dataset 2 — MurrayLab CTX Orbital Tiles](#4-dataset-2--murraylab-ctx-orbital-tiles)
5. [Dataset 3 — HiRISE Map-Proj-v3](#5-dataset-3--hirise-map-proj-v3)
6. [Label Remapping Specifications](#6-label-remapping-specifications)
7. [Preprocessing Pipeline Per Dataset](#7-preprocessing-pipeline-per-dataset)
8. [Dataset-to-Stage Mapping](#8-dataset-to-stage-mapping)
9. [Train / Validation / Test Split Strategy](#9-train--validation--test-split-strategy)
10. [The Domain Gap — Design Intent](#10-the-domain-gap--design-intent)
11. [Expected Data Statistics](#11-expected-data-statistics)
12. [File Organization](#12-file-organization)
13. [Critical Implementation Notes](#13-critical-implementation-notes)

---

## 1. Pipeline Overview

The PA-GNN pipeline takes a terrain image as input and produces a safe traversal path as output. It has 7 stages:

| Stage | Name | Input | Output |
|---|---|---|---|
| 1 | Preprocessing | Raw image tile | Normalized tensor |
| 2 | Physics feature extraction | Normalized tile | H_physics ∈ [0,1]^{H×W} |
| 3 | CNN risk heatmap | Normalized tile | H_learned ∈ [0,1]^{H×W} |
| 4 | Adaptive hybrid fusion | H_physics + H_learned | H_final ∈ [0,1]^{H×W} |
| 5 | Superpixel graph construction | H_final | G = (V, E, X, W) |
| 6 | GATv2 traversability refinement | G | p̂_i ∈ [0,1] per node |
| 7 | A* path planning | Refined graph | Waypoint sequence |

Three datasets serve this pipeline in completely non-overlapping roles:

- **AI4Mars** → Supervised training and quantitative evaluation of Stage 3 CNN. Contains pixel-level terrain labels.
- **MurrayLab CTX** → Full 7-stage qualitative demonstration on real orbital imagery. No labels required.
- **HiRISE v3** → Cross-domain quantitative evaluation. Orbital images with image-level landmark labels remapped to risk scores.

No dataset substitutes for another. Each covers a gap the others cannot fill.

---

## 2. Dataset Summary Table

| Property | AI4Mars | MurrayLab CTX | HiRISE v3 |
|---|---|---|---|
| **Source camera** | Mars rover NavCam / MastCam | Mars Reconnaissance Orbiter CTX | Mars Reconnaissance Orbiter HiRISE |
| **Imaging altitude** | Ground-level (~1–2 m) | ~300 km orbital | ~300 km orbital |
| **Resolution** | Variable (~0.5–3 cm/px) | ~6 m/px | ~25 cm/px |
| **Image dimensions** | Variable (512×512 common) | 512×512 (pre-sliced) | 227×227 (pre-cropped) |
| **Channels** | Grayscale (NavCam) / RGB (MastCam) | Grayscale | Grayscale |
| **Label type** | Per-pixel segmentation masks (.png) | None | Image-level class label (.txt / .csv) |
| **Label granularity** | Pixel-level | N/A | Crop-level (one class per image) |
| **Number of labeled images** | ~60,000+ merged labels | 0 | 10,433 original + 62,598 augmented |
| **Missions covered** | MSL (Curiosity), MER (Spirit/Opportunity), M2020 (Perseverance) | N/A — surface-agnostic orbital mosaic | Mars surface (MRO HiRISE orbital) |
| **Pipeline role** | CNN training + quantitative test set | Full pipeline demo, qualitative figures | Cross-domain quantitative evaluation |
| **Label schema used** | NAV (4-class) → remapped to 3-class risk | N/A | Landmark class → remapped to 3-class risk |

---

## 3. Dataset 1 — AI4Mars Merged Dataset

### 3.1 What It Is

AI4Mars is a crowdsourced terrain labeling dataset created by the NASA JPL AI4Mars project in collaboration with Zooniverse. Volunteer labelers annotated Mars rover images with terrain class labels. Each pixel in each image was labeled by multiple independent labelers, and the labels were merged in post-processing using majority agreement thresholds. The result is a large, pixel-level semantic segmentation dataset covering three Mars rover missions.

**Original source:** [Zooniverse AI4Mars Project](https://www.zooniverse.org/projects/hiro-ono/ai4mars)  
**Citation:** Wagstaff et al., "Mars Image Content Classification: Three Years of NASA Deployment and Recent Advances," IAAI 2021.

### 3.2 Missions and Subsets

The dataset covers three missions. **For this thesis, use MSL NavCam data only.** The reasons are given in Section 3.4.

#### MSL — Mars Science Laboratory (Curiosity Rover)
- Camera: NavCam (grayscale) and MastCam (color)
- Image type codes in filenames: `EDR` (NavCam), `ML` (MastCam)
- Label quality: Highest — masked+merged+cleaned labels available, 30-meter range masks applied, rover masks applied
- Expert test sets: Available for NavCam
- **Use: NavCam (EDR) only. Do not use MastCam (ML) for this thesis unless extending to color.**

#### MER — Mars Exploration Rovers (Spirit and Opportunity)
- Camera: NavCam (grayscale)
- Image type codes: `EFF`
- Label quality: Lower — some bad images included, range masks not yet applied in all versions, merged training labels available
- Expert test sets: Available
- **Use: Avoid unless needed for augmentation. Labels are noisier than MSL.**

#### M2020 — Mars 2020 (Perseverance Rover)
- Camera: NavCam (grayscale, `Vgnc`, `NLF`) and MastCam-Z (color, `ZL0`)
- Label schema: Two label types exist — NAV (navigation) and M2020_GEO (geology, beta)
- Expert test sets: Not available
- **Use: Do not use for training or testing. No expert test sets, geology labels are beta and untested.**

### 3.3 Label Schema — NAV (Navigation)

All training and evaluation in Stage 3 uses the NAV schema. This schema divides terrain by traversability for rover navigation. Labels are stored as grayscale `.png` files where each pixel value encodes a terrain class.

| Pixel Value | Original Class | Traversability | Remapped Risk Class | Risk Score Range |
|---|---|---|---|---|
| `0` | Soil | High traversability | Safe | 0.0 – 0.2 |
| `1` | Bedrock | Medium — depends on texture | Uncertain | 0.4 – 0.6 |
| `2` | Sand | Medium — slip risk | Uncertain | 0.3 – 0.5 |
| `3` | Big Rock | Not traversable | Hazardous | 0.8 – 1.0 |
| `255` | NULL | No label (masked or insufficient agreement) | Ignore (exclude from loss) | N/A |

> **Implementation note:** Label pixel value `255` must be excluded from all loss computations. In PyTorch, pass `ignore_index=255` to your loss function. Pixels marked `255` exist because fewer than 3 labelers agreed on that pixel, or the pixel is outside the 30-meter range mask, or it overlaps the rover body mask.

### 3.4 Why MSL NavCam Only

Use only MSL NavCam (`EDR`) images and their corresponding NAV labels. The reasons:

1. **Label quality is highest.** MSL NavCam data has masked+merged+cleaned labels — range masks (30m) and rover masks are applied, bad labels have been removed. MER labels have known issues (some bad images, unmasked merges in some versions).
2. **Consistent image type.** Grayscale NavCam images form a homogeneous visual domain. Mixing in MastCam (color) would require handling a 3-channel vs 1-channel input inconsistency unless you explicitly design for it.
3. **Expert test sets exist.** MSL NavCam has expert-labeled test sets with 100% inter-annotator agreement. This gives you a defensible, clean evaluation benchmark.
4. **Volume is sufficient.** MSL NavCam alone provides thousands of images — enough to train and validate a segmentation model.

### 3.5 Accessing the Data

The merged AI4Mars dataset is available on Zenodo and NASA's PDS (Planetary Data System). The specific subset to download is the MSL NavCam merged training and test labels.

```
Expected directory structure after download:
ai4mars/
├── msl/
│   ├── images/
│   │   └── edr/          ← NavCam images (.JPG)
│   ├── labels/
│   │   ├── train/        ← merged crowdsourced labels (.png)
│   │   └── test/
│   │       ├── masked-gold-min1/
│   │       ├── masked-gold-min2/
│   │       └── masked-gold-min3/   ← use min3 (strictest, all 3 experts agree)
│   └── masks/
│       ├── range/        ← 30m range masks (.png, masked=1)
│       └── rover/        ← rover body masks (.png, masked=1)
```

### 3.6 Train / Test Subsets

The dataset provides pre-defined train and test splits. Use them as-is — do not remix train and test images.

- **Train set:** Crowdsourced merged labels, minimum 3 labelers, 2/3 agreement per pixel.
- **Test set:** Use `masked-gold-min3` — expert labels with all 3 expert labelers agreeing. This is the strictest and most reliable.

From the train set, carve out a validation split (see Section 9).

### 3.7 Image-Label Correspondence

Image filenames match label filenames except for the extension and a `_merged` suffix on labels. Example:

```
Image:  NLB_486894538EDR_F0481570NCAM00354M1.JPG
Label:  NLB_486894538EDR_F0481570NCAM00354M1_merged.png
```

Always verify correspondence by stripping suffixes and matching base names. Mismatched pairs will corrupt training.

### 3.8 Preprocessing Applied to AI4Mars

See Section 7.1 for full preprocessing specification.

---

## 4. Dataset 2 — MurrayLab CTX Orbital Tiles

### 4.1 What It Is

MurrayLab CTX tiles are grayscale orbital images captured by the Context Camera (CTX) aboard the Mars Reconnaissance Orbiter (MRO). The original CTX images are large mosaics of the Martian surface at approximately 6 meters per pixel. The MurrayLab dataset pre-processes these mosaics by slicing them into non-overlapping 512×512 pixel tiles, making them immediately compatible with standard deep learning pipelines.

**This dataset has no terrain labels.** It exists solely to provide real orbital imagery for pipeline demonstration.

### 4.2 Structure

```
original-image-slices-512x512/
├── sliced_tiles_1/
│   ├── tile_x0000_y0000_pos(0,0).png
│   ├── tile_x0000_y0001_pos(0,512).png
│   ├── tile_x0001_y0000_pos(512,0).png
│   ├── ...
│   ├── tile_metadata         ← JSON or CSV with tile positions and source image info
│   └── MurrayLab_CTX_V01_E-004_N-04_ReadMe   ← per-mosaic readme with coordinates
│
├── sliced_tiles_2/
│   ├── tile_x0000_y0000_pos(0,0).png
│   ├── ...
│   └── MurrayLab_CTX_V01_E-004_N-08_ReadMe
│
└── sliced_tiles_N/
    └── ...
```

Each `sliced_tiles_N/` directory corresponds to one CTX mosaic strip. The `pos(X,Y)` in tile filenames encodes the pixel offset of the tile's top-left corner within the original full mosaic image, making spatial reconstruction possible if needed.

### 4.3 Image Properties

- **Format:** `.png`, grayscale (single channel)
- **Dimensions:** 512×512 pixels exactly — already correct for the pipeline
- **Pixel values:** 8-bit unsigned integer [0, 255]
- **Spatial resolution:** ~6 meters per pixel (much coarser than HiRISE's 25 cm/px)
- **Coverage:** Multiple strips of Martian surface, geographically referenced by the ReadMe coordinates

### 4.4 What This Dataset Is Used For

MurrayLab tiles are the **input for the full end-to-end pipeline demonstration**. Since there are no labels:

- Do not attempt to compute quantitative metrics (HCR, PLR, IoU) on MurrayLab tiles.
- Use them only for qualitative figures: the 5 mandatory visualization outputs in Section 10.5 of the thesis blueprint.
- Select 3–5 tiles that show visually diverse terrain (flat soil, rocky outcrops, sloped terrain if available). Avoid tiles that are entirely uniform — they will not show the pipeline's differentiation capability.

### 4.5 Tile Selection Criteria

When picking tiles for the demo, prefer tiles that contain:
- At least one region of visually smooth terrain (low physics risk expected)
- At least one region of rough or high-contrast terrain (high physics risk expected)
- Clear enough contrast that features are visible (avoid overexposed or near-black tiles)

Reject tiles where more than 30% of pixels are saturated (value 0 or 255) — these are likely at the edge of a mosaic strip and contain padding artifacts.

### 4.6 Preprocessing Applied to MurrayLab

See Section 7.2 for full preprocessing specification.

---

## 5. Dataset 3 — HiRISE Map-Proj-v3

### 5.1 What It Is

HiRISE (High Resolution Imaging Science Experiment) is the highest-resolution camera aboard MRO, producing images at approximately 25 centimeters per pixel. The map-proj-v3 dataset was compiled by Wagstaff et al. (NASA JPL) and contains 73,031 landmark image crops extracted and augmented from 180 HiRISE browse images.

**Original source:** Wagstaff, K.L., Lu, S., Doran, G., Mandrake, L. — DOI: [10.5281/zenodo.2538136](https://doi.org/10.5281/zenodo.2538136)

This dataset provides the only available labeled HiRISE image crops suitable for cross-domain evaluation in this thesis. Labels are **image-level** (one class per 227×227 crop), not per-pixel.

### 5.2 Structure

```
hirise-map-proj-v3/
├── map-proj-v3/                          ← directory of all cropped landmark images
│   ├── B01_009838_1408_XN_39S198W_0.jpg
│   ├── B01_009838_1408_XN_39S198W_1.jpg
│   └── ...                               ← 73,031 images total
├── labels-map-proj-v3.txt                ← one integer class ID per line, same order as images
└── landmarks_map-proj-v3_classmap.csv    ← maps class ID to semantic name
```

### 5.3 Image Properties

- **Format:** `.jpg`, grayscale
- **Dimensions:** 227×227 pixels
- **Pixel values:** 8-bit unsigned integer [0, 255]
- **Total images:** 73,031 (10,433 original + 62,598 augmented via rotation, flip, brightness)
- **Spatial resolution:** ~25 cm/px — the highest resolution Mars orbital imagery available

### 5.4 Augmentation Lineage

Each of the 10,433 original landmarks was augmented to produce 6 additional variants:

| Augmentation index | Method |
|---|---|
| 0 | Original (no augmentation) |
| 1 | 90° clockwise rotation |
| 2 | 180° clockwise rotation |
| 3 | 270° clockwise rotation |
| 4 | Horizontal flip |
| 5 | Vertical flip |
| 6 | Random brightness adjustment |

When creating the test split from HiRISE v3, **do not allow different augmentations of the same original landmark to appear in both train-equivalent and test splits.** Since this dataset is used only for evaluation (no training here), use the original crops only (index `_0` suffix or equivalent) to avoid evaluating on augmented duplicates of the same scene.

### 5.5 Landmark Class System and Risk Remapping

The dataset classifies each image crop into a landmark type. These landmark classes must be remapped to the 3-class traversability risk schema used throughout this pipeline. The remapping is geologically motivated — each class represents a surface feature with known traversability implications.

See Section 6 for the full remapping table and rationale.

### 5.6 What This Dataset Is Used For

HiRISE v3 is the **cross-domain quantitative evaluation set**. It answers the question: *how well does a system trained on rover imagery (AI4Mars) generalize to real orbital imagery (HiRISE)?*

The evaluation procedure:
1. Run the full pipeline on each HiRISE v3 crop.
2. The pipeline produces a risk score for the image (patch-level, not pixel-level, since labels are image-level).
3. Compare the predicted risk level against the remapped ground-truth risk class.
4. Compute patch-level classification accuracy, precision, recall per risk class, and AUC-ROC.
5. Compare these numbers against the same metrics computed on AI4Mars test images to quantify the domain gap.

> **Key result to demonstrate:** The CNN-only system shows a measurable accuracy drop from rover images (AI4Mars test) to orbital images (HiRISE v3). The physics features show minimal or no drop (they are domain-invariant by design). The hybrid system partially recovers the CNN's accuracy loss. This three-row comparison is the central experimental finding of the thesis.

### 5.7 Input Size Adjustment

HiRISE crops are 227×227. The pipeline is designed around 512×512. Two options:

**Option A (recommended):** Pad each 227×227 image to 256×256 with zero-padding (symmetric), then center-crop or resize to 512×512. This preserves all original content.

**Option B:** Resize directly from 227×227 to 512×512 using bilinear interpolation. Simple, slight loss of sharpness, acceptable for evaluation.

Whichever option is chosen, apply it consistently to all HiRISE v3 crops. Document the choice in the paper.

---

## 6. Label Remapping Specifications

### 6.1 AI4Mars NAV → 3-Class Risk Schema

The AI4Mars NAV schema has 4 meaningful classes plus a NULL class. These are remapped to a 3-class traversability risk schema. The CNN in Stage 3 outputs a **continuous scalar risk score** per pixel, not a discrete class. The remapping below defines the target regression value for each original class.

| Original Class | Pixel Value | Risk Class | Target Risk Score | Rationale |
|---|---|---|---|---|
| Soil | 0 | Safe | **0.1** | Compacted regolith, high traversability, no slip risk |
| Bedrock | 1 | Uncertain | **0.5** | Traversable but potentially uneven; depends on local texture |
| Sand | 2 | Uncertain | **0.4** | Slip and entrapment risk; appears safe but is not |
| Big Rock | 3 | Hazardous | **0.9** | Not traversable; direct obstacle |
| NULL | 255 | Excluded | N/A | Exclude from all loss computation (`ignore_index=255`) |

> **Why continuous scores instead of discrete classes?** A continuous risk score carries more information into the downstream fusion stage (Stage 4). Discrete 3-class labels force a hard boundary that the adaptive fusion cannot reason across. With continuous scores, the CNN outputs a probability-like risk surface, which fuses naturally with the physics risk map (also continuous).

> **Why not 0.0 and 1.0 for safe and hazardous?** Label smoothing. Using 0.1 and 0.9 instead of 0.0 and 1.0 prevents the model from becoming overconfident on extreme predictions and improves calibration (lower ECE). This is standard practice in regression-based segmentation.

### 6.2 HiRISE v3 Landmark Class → 3-Class Risk Schema

The HiRISE v3 classmap maps integer IDs to semantic landmark names. The following remapping applies each landmark class to a traversability risk level. This remapping is used only for evaluation — it defines the ground-truth risk label for each HiRISE crop.

| HiRISE Class Name | Risk Class | Assigned Risk Score | Rationale |
|---|---|---|---|
| `crater` | Hazardous | 0.9 | Rim and interior pose entrapment and slope hazard |
| `dark_dune` | Hazardous | 0.85 | Dark dunes are fine-grained, high slip risk |
| `slope_streak` | Hazardous | 0.8 | Indicates mass movement; unstable slope terrain |
| `bright_dune` | Uncertain | 0.5 | Dunes with variable composition; moderate slip risk |
| `impact_ejecta` | Uncertain | 0.55 | Scattered rock field; passable but rough |
| `edge_case` | Uncertain | 0.5 | Ambiguous feature; moderate default |
| `swiss_cheese` | Hazardous | 0.85 | CO₂ sublimation pits; structurally unsafe terrain |
| `spider` | Uncertain | 0.45 | Dendritic erosion patterns; rough but often passable |
| `other` | Safe | 0.15 | Generic flat terrain with no detected landmark |

> **Note:** The exact class names in `landmarks_map-proj-v3_classmap.csv` should be verified against the file before using this table. Class names may differ slightly from the above. Always load the classmap file programmatically and verify your mappings.

```python
import pandas as pd

classmap = pd.read_csv('landmarks_map-proj-v3_classmap.csv')

HIRISE_RISK_REMAP = {
    'crater':        0.90,
    'dark_dune':     0.85,
    'slope_streak':  0.80,
    'swiss_cheese':  0.85,
    'bright_dune':   0.50,
    'impact_ejecta': 0.55,
    'spider':        0.45,
    'edge_case':     0.50,
    'other':         0.15,
}

def get_risk_score(class_name: str) -> float:
    return HIRISE_RISK_REMAP.get(class_name.lower().strip(), 0.5)
```

---

## 7. Preprocessing Pipeline Per Dataset

### 7.1 AI4Mars Preprocessing

Applied before any image enters Stage 1 of the pipeline during training and evaluation.

```
Step 1 — Load image
  - Load .JPG image using PIL or OpenCV
  - Convert to grayscale if loaded as RGB (MSL NavCam is inherently grayscale, 
    but some loaders will produce 3 identical channels)
  - Shape after: (H, W) or (H, W, 1)

Step 2 — Resize
  - Resize to 512×512 using bilinear interpolation (images may vary in original size)
  - Do NOT use nearest-neighbour interpolation for images (only for labels)

Step 3 — Label resize (if resizing labels)
  - Resize label mask to 512×512 using NEAREST-NEIGHBOUR interpolation only
  - Bilinear interpolation on integer class masks creates invalid intermediate values
  - NULL pixels (value 255) must remain 255 after resize

Step 4 — Apply masks (training only)
  - Load rover mask (.png, masked region = 1)
  - Load range mask (.png, masked region = 1)
  - Set label pixels to 255 (NULL) wherever rover mask = 1 OR range mask = 1
  - This removes the rover body and far-distance unreliable labels

Step 5 — Normalize image
  - Per-tile min-max normalization: pixel = (pixel - min) / (max - min + ε)
  - ε = 1e-8 to avoid division by zero on uniform tiles
  - Output range: [0.0, 1.0] float32

Step 6 — Replicate to 3 channels (CNN input)
  - Replicate single grayscale channel 3 times: shape (H, W, 1) → (H, W, 3)
  - This is required for MobileNetV3 which expects 3-channel input
  - The channels are identical; this is not color information

Step 7 — Data augmentation (training split only, NOT validation or test)
  - Random horizontal flip (p=0.5)
  - Random vertical flip (p=0.5)
  - Random rotation ±15 degrees (fill mode: reflect)
  - Random brightness/contrast jitter ±20%
  - Gaussian noise (σ ~ Uniform(0, 0.02)) applied to normalized image only
  - Apply SAME spatial transform to image AND label mask

Step 8 — Convert label to continuous risk score map
  - Apply remapping from Section 6.1
  - Output: float32 tensor of shape (H, W) with values in [0.1, 0.9]
  - Pixels with original value 255 → mask tensor (boolean), excluded from loss
```

### 7.2 MurrayLab Preprocessing

Applied when using CTX tiles as pipeline input for qualitative demonstration.

```
Step 1 — Load tile
  - Load .png file as grayscale
  - Already 512×512 — no resize needed
  - Shape: (512, 512)

Step 2 — Normalize
  - Per-tile min-max normalization (same as Step 5 above)
  - Output range: [0.0, 1.0] float32

Step 3 — Replicate to 3 channels
  - Same as AI4Mars Step 6
  - Shape: (512, 512, 3)

Step 4 — Quality check
  - Compute fraction of pixels within 5% of min or max (near-saturated)
  - If saturated fraction > 0.30: skip this tile (edge/padding artifact)
  - Log rejected tiles

No label processing. No augmentation. No masking.
```

### 7.3 HiRISE v3 Preprocessing

Applied when using HiRISE crops for cross-domain quantitative evaluation.

```
Step 1 — Load image
  - Load .jpg as grayscale
  - Shape: (227, 227)

Step 2 — Resize to 512×512
  - Use bilinear interpolation
  - Alternative: zero-pad to 256×256 then resize to 512×512 (see Section 5.7)
  - Document choice in paper

Step 3 — Normalize
  - Per-tile min-max normalization
  - Output range: [0.0, 1.0] float32

Step 4 — Replicate to 3 channels
  - Same as other datasets

Step 5 — Load image-level label
  - Read integer class ID from labels-map-proj-v3.txt (line index = image index)
  - Look up class name from classmap CSV
  - Map class name to risk score using Section 6.2 table
  - Store as a scalar ground-truth risk value for this crop

Step 6 — Separate original from augmented
  - For evaluation: use only original crops (augmentation index 0)
  - Identify originals by filename convention or by taking every 7th entry
    (dataset is ordered: original, aug1, aug2, aug3, aug4, aug5, aug6, repeat)
  - Verify this ordering against the actual dataset before assuming it

No augmentation applied to evaluation data.
```

---

## 8. Dataset-to-Stage Mapping

This section specifies precisely which dataset feeds which stage and in what capacity.

### Stage 1 — Preprocessing

| Dataset | Role | Notes |
|---|---|---|
| AI4Mars (MSL NavCam) | Training/eval input | Apply full preprocessing from Section 7.1 |
| MurrayLab CTX | Demo input | Apply Section 7.2 preprocessing |
| HiRISE v3 | Eval input | Apply Section 7.3 preprocessing |

### Stage 2 — Physics Feature Extraction

| Dataset | Role | Notes |
|---|---|---|
| All three | Input | Physics features are computed on whatever image enters Stage 1. No dataset-specific logic. Stage 2 is domain-invariant by design. |

Physics features computed:
- `S` = Sobel gradient magnitude (slope proxy), normalized to [0,1]
- `R` = Local standard deviation in 7×7 sliding window (roughness), normalized to [0,1]
- `D` = Absolute Laplacian of Gaussian response, σ=2.0 (depth discontinuity proxy), normalized to [0,1]
- `H_physics` = 0.4·S + 0.3·R + 0.3·D (weights tuned on AI4Mars validation set)

### Stage 3 — CNN Risk Heatmap

| Dataset | Role | Notes |
|---|---|---|
| AI4Mars (MSL NavCam train split) | CNN training | Pixel-level supervision. Use focal loss with γ=2. |
| AI4Mars (MSL NavCam val split) | CNN validation | Monitor mIoU and hazard recall. Early stopping based on hazard recall. |
| AI4Mars (MSL NavCam test/min3) | Quantitative evaluation | Final test numbers reported in results table. |
| MurrayLab CTX | Demo input | No labels — CNN outputs heatmap for qualitative display only. |
| HiRISE v3 | Cross-domain evaluation | CNN outputs patch-level risk score; compared to remapped ground-truth label. |

**Architecture:** MobileNetV3-Large encoder + DeepLabV3+ decoder with a single-channel regression head (sigmoid output). Input: (B, 3, 512, 512). Output: (B, 1, 512, 512), values in [0, 1].

**Loss function:** Focal loss applied only to non-NULL pixels (where mask = False). Use `γ=2, α=0.25` as starting values; tune α on validation set. Do NOT use cross-entropy — class imbalance (soil dominates) will cause the model to learn "predict everything as safe."

### Stage 4 — Adaptive Hybrid Fusion

| Dataset | Role | Notes |
|---|---|---|
| AI4Mars | Training the α predictor | The small network that predicts α (physics trust weight) is trained on AI4Mars images using H_physics and H_learned as inputs. Ground truth α is derived from which signal was more accurate per pixel on the validation set. |
| All three | Inference input | At inference, fusion is applied to whatever H_physics and H_learned are produced for the input tile. |

**Fusion formula:** `H_final = α · H_physics + (1 - α) · H_learned`  
**α predictor input feature:** Local texture entropy computed from the original normalized image in a 15×15 sliding window. High entropy → high α (trust physics more — CNN uncertain). Low entropy → low α (trust CNN — texture is informative).

### Stage 5 — Superpixel Graph Construction

| Dataset | Role | Notes |
|---|---|---|
| All three | Indirect input via H_final | SLIC superpixels are computed on the original normalized image. Node features are derived from H_final. No dataset-specific logic. |

**Target superpixel count:** ~300 per 512×512 tile. Each superpixel = one graph node.  
**Node features:** [mean H_final, std H_final, mean H_physics, mean H_learned, mean α, superpixel area, centroid_x, centroid_y]  
**Edge features:** [shared boundary length between adjacent superpixels, Euclidean distance between centroids]

### Stage 6 — GATv2 Traversability Refinement

| Dataset | Role | Notes |
|---|---|---|
| AI4Mars | GATv2 training | Node labels derived by majority vote of ground-truth pixel labels within each superpixel. Binary: hazardous (>0.5 mean risk) or safe. |
| All three | GATv2 inference | GATv2 refines the graph for whatever input tile is being processed. |

**Architecture:** 2-layer GATv2, 4 attention heads per layer, hidden dim 64. Input node features: 8-dim vector (see Stage 5). Input edge features: 2-dim vector. Output: scalar traversability probability per node ∈ [0,1].

### Stage 7 — A* Path Planning

| Dataset | Role | Notes |
|---|---|---|
| AI4Mars | Evaluation | HCR (Hazard Crossing Rate) and PLR (Path Length Ratio) computed against ground-truth hazard labels. |
| MurrayLab | Demo | Path visualized qualitatively on orbital tile. No ground truth for quantitative metrics. |
| HiRISE v3 | Evaluation | Patch-level path risk assessment using remapped labels. |

---

## 9. Train / Validation / Test Split Strategy

### 9.1 AI4Mars Split

The AI4Mars dataset provides predefined train and test sets. Do not merge and re-split them. The test set contains expert-labeled images and must remain isolated.

```
ai4mars_test   = all images in labels/test/masked-gold-min3/   (expert labels, DO NOT touch during training)
ai4mars_trainval = all images in labels/train/                  (crowdsourced merged labels)

From ai4mars_trainval:
  ai4mars_train = 70% of trainval (random stratified split by terrain type distribution)
  ai4mars_val   = 15% of trainval (held out during training, used for hyperparameter tuning)

Final ratio: ~70% train / 15% val / 15% test
```

**Stratification:** When splitting trainval into train and val, stratify by terrain type. Compute the dominant class (mode of pixel-level labels) per image and use that as the stratification key. This prevents the val set from being dominated by soil images.

### 9.2 HiRISE v3 Split

Since HiRISE v3 is used only for evaluation (no training or tuning), no split is needed. Use all original crops (augmentation index 0) as the evaluation set. Keep augmented variants out of evaluation to avoid evaluating near-duplicate scenes.

### 9.3 MurrayLab Split

No split needed. MurrayLab is qualitative demo only. Manually select 3–5 tiles for demonstration figures.

---

## 10. The Domain Gap — Design Intent

The domain gap between AI4Mars (rover-level) and HiRISE/CTX (orbital) imagery is not a flaw to be hidden — it is a core experimental variable that validates the thesis's central hypothesis.

**The expected experimental finding:**

| System | AI4Mars test (rover) | HiRISE v3 (orbital) | Domain gap |
|---|---|---|---|
| CNN alone (Stage 3) | High hazard recall | Significantly lower | Large gap |
| Physics alone (Stage 2) | Moderate hazard recall | Similar recall | Minimal gap |
| Hybrid PA-GNN (proposed) | Highest recall | Partially recovered | Smaller gap than CNN-only |

**Why physics features are domain-invariant:** Sobel gradients, local standard deviation, and LoG responses are computed directly from pixel intensity patterns. These operations have no knowledge of whether the image was taken from 1 meter or 300 kilometers. The same crater appears as a circular high-gradient region regardless of imaging altitude — the scale changes, but the feature response remains. This domain invariance is what makes physics features a reliable fallback when the CNN's training distribution does not match the test distribution.

**How to report this in the paper:** Report a 2×3 table: rows = {CNN only, physics only, hybrid PA-GNN}, columns = {AI4Mars test (hazard recall), HiRISE v3 (hazard recall), delta}. This table is the primary novelty claim — no prior paper has shown this specific comparison experimentally.

---

## 11. Expected Data Statistics

These are approximate figures. Verify against actual downloaded data.

| Statistic | Value |
|---|---|
| AI4Mars MSL NavCam training images | ~4,000–6,000 images |
| AI4Mars MSL NavCam validation images | ~800–1,200 images |
| AI4Mars MSL NavCam test images (min3) | ~500–1,000 images |
| MurrayLab tiles selected for demo | 3–5 tiles |
| HiRISE v3 original crops (eval) | 10,433 crops |
| HiRISE v3 total (with augmentation) | 73,031 crops |
| Expected class distribution in AI4Mars MSL train | Soil ~60%, Bedrock ~20%, Sand ~10%, Big Rock ~5%, NULL ~5% |
| Expected superpixels per 512×512 tile | ~250–350 (SLIC, n_segments=300) |
| Expected graph nodes per tile | ~280–320 |
| Expected graph edges per tile | ~800–1,200 (6-connectivity superpixel graph) |

> **Class imbalance warning:** Soil pixels dominate the AI4Mars training set at roughly 60%. This is why focal loss is mandatory (see Stage 3). A model trained with cross-entropy on this distribution will achieve high overall accuracy by predicting "safe" everywhere, while completely failing on the hazardous class — the most important class for rover safety.

---

## 12. File Organization

Recommended directory structure for the project codebase:

```
pagnn/
│
├── data/
│   ├── ai4mars/
│   │   ├── raw/                      ← downloaded files, do not modify
│   │   │   ├── images/edr/
│   │   │   ├── labels/train/
│   │   │   ├── labels/test/masked-gold-min3/
│   │   │   └── masks/
│   │   └── processed/                ← output of preprocessing scripts
│   │       ├── train/
│   │       │   ├── images/           ← normalized .npy tensors (512×512×3)
│   │       │   └── labels/           ← risk score maps .npy (512×512) + mask
│   │       ├── val/
│   │       └── test/
│   │
│   ├── murraylab/
│   │   ├── raw/                      ← original sliced tile directories
│   │   └── selected/                 ← 3-5 tiles chosen for demo, normalized
│   │
│   └── hirise_v3/
│       ├── raw/
│       │   ├── map-proj-v3/          ← all 73,031 images
│       │   ├── labels-map-proj-v3.txt
│       │   └── landmarks_map-proj-v3_classmap.csv
│       └── processed/
│           ├── originals/            ← augmentation index 0 only, normalized
│           └── labels_remapped.csv   ← image filename → risk score
│
├── src/
│   ├── data/
│   │   ├── ai4mars_dataset.py        ← PyTorch Dataset class for AI4Mars
│   │   ├── hirise_dataset.py         ← PyTorch Dataset class for HiRISE v3
│   │   ├── murraylab_loader.py       ← loader for demo tiles
│   │   └── label_remap.py            ← all remapping functions (Section 6)
│   ├── stages/
│   │   ├── stage2_physics.py         ← Sobel, roughness, LoG
│   │   ├── stage3_cnn.py             ← MobileNetV3 + DeepLabV3+
│   │   ├── stage4_fusion.py          ← adaptive alpha fusion
│   │   ├── stage5_graph.py           ← SLIC + graph construction
│   │   ├── stage6_gatv2.py           ← GATv2 implementation
│   │   └── stage7_astar.py           ← A* path planning
│   ├── train.py                      ← CNN training script
│   ├── evaluate.py                   ← evaluation script (AI4Mars + HiRISE)
│   └── demo.py                       ← full pipeline demo on MurrayLab tile
│
├── outputs/
│   ├── figures/                      ← all generated figures for paper
│   ├── checkpoints/                  ← model weights
│   └── results/                      ← CSV files with metric outputs
│
├── DATASETS.md                       ← this file
└── README.md
```

---

## 13. Critical Implementation Notes

These are the most common mistakes that will silently corrupt results. Read carefully.

**1. Never interpolate label masks with bilinear/cubic methods.**  
Always use nearest-neighbour interpolation when resizing label masks. Bilinear interpolation creates pixel values that don't correspond to any valid class, producing nonsense supervision signal.

**2. Never let NULL pixels (value 255) contribute to the loss.**  
Pass `ignore_index=255` to your loss function at every training step. If you forget this, the model will be penalized for predicting risk on pixels that have no meaningful label — it will learn to predict something around the NULL boundaries that has no physical meaning.

**3. Never use the same image in both train and test.**  
The AI4Mars test set contains expert-labeled images. Some of these expert images may overlap with images in the training set (different labelers, same image). Always verify there are no filename overlaps between your train set and the expert test set before training.

**4. Never mix augmented HiRISE crops with originals during evaluation.**  
Augmented crops are near-duplicates of originals. If both appear in your evaluation set, your metrics will be inflated because the model effectively sees the same scene multiple times. Use only the original crops (augmentation index 0) for evaluation.

**5. Verify image-label correspondence by filename before training.**  
Always write a verification script that checks every image file has a corresponding label file and vice versa, with matching base names. Missing pairs will cause silent errors in batch loaders.

**6. Normalize per-tile, not per-dataset.**  
Do not compute dataset-wide mean and standard deviation for normalization. Mars imagery has high inter-image variance in illumination and albedo. Per-tile min-max normalization (Section 7) is the correct approach. ImageNet mean/std normalization is inappropriate for single-channel Mars imagery.

**7. The physics feature weights (w1, w2, w3) must be tuned on the validation set.**  
The initial values (0.4, 0.3, 0.3) are starting points from the blueprint. Run a grid search over w1 ∈ {0.3, 0.4, 0.5}, w2 ∈ {0.2, 0.3, 0.4}, w3 ∈ {0.2, 0.3, 0.4} subject to w1+w2+w3=1, evaluated by H_physics hazard recall on the AI4Mars validation set. Report final weights in the paper.

**8. Report metrics separately for each dataset.**  
Never average AI4Mars and HiRISE v3 metrics together. They measure different things (in-domain vs cross-domain). Report two separate results blocks in your paper: one for the AI4Mars rover test set, one for the HiRISE v3 orbital evaluation.

---

*Document version: 1.0*  
*Thesis: PA-GNN — Physics-Aware Graph Neural Network Pipeline for Autonomous Planetary Path Planning*  
*Authors: Syed Abir Hossain, Ashik Mahmud, Mahadir Rahaman — AUST CSE-4733*
