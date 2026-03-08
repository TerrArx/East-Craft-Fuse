# 📦 EAST and CRAFT Weighted Box Fusion Ensemble

A weighted box fusion ensemble that combines two scene-text detectors — **EAST** and **CRAFT** — to improve recall and robustness. The pipeline uses ensemble_boxes WBF to merge overlapping detections, align bounding boxes, and produce more stable outputs with fewer misses.

<p align="center">
  <img src="WBF/visualizations/17c98e4d-bcda-48da-971c-e0478cfc22c7.png" width="1200">
</p>
<p align="center"><i>Comparison of EAST (Blue), CRAFT (Red), and Weighted Box Fusion (Green)</i></p>

<div align="center">
 
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg) ![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

</div>

---

## 🎨 Visual Results

A dedicated visualization collection showcases EAST, CRAFT, and WBF-fused outputs side-by-side.

<p align="center"><i>EAST Detections (Green)</i></p>
<p align="center">
  <img src="EAST/output.visualizations/2.png" width="900">
</p>

---

<p align="center"><i>CRAFT Detections (Green)</i></p>
<p align="center">
  <img src="CRAFT/visualizations/img_100_viz.png" width="900">
</p>

---

<p align="center"><i>Comparison of EAST (Blue), CRAFT (Red), and Fused WBF (Green)</i></p>
<p align="center">
  <img src="WBF/visualizations/ae5458bc-5eee-45dd-b1e7-2ff44f8cc341.png" width="900">
</p>

---

## ✨ Key Features

- 🎯 **Weighted Box Fusion** — Uses ensemble_boxes WBF algorithm for intelligent bounding box merging  
- 🔄 **Agreement Filtering** — Optional requirement for both models to agree on detections  
- 📐 **Polygon Preservation** — Maps fused boxes back to original polygon coordinates  
- ⚖️ **Configurable Weights** — Adjustable per-model confidence weighting (default: EAST=0.6, CRAFT=0.4)  
- 🎚️ **Tunable IoU Threshold** — Control fusion sensitivity (default: 0.55)  
- 📊 **Automated Evaluation** — ICDAR-style metrics and visualization exports

---

## 🚀 Quickstart

### Prerequisites

```bash
# Create environment
conda create -n text_detection python=3.8
conda activate text_detection

# Install dependencies
pip install opencv-python numpy pandas matplotlib shapely ensemble-boxes tqdm
pip install craft-text-detector
```

### Configuration

The notebook `wbf.ipynb` contains all configuration parameters:

```python
# Model weights for fusion (sum should be 1.0)
MODEL_WEIGHTS = [0.6, 0.4]  # [EAST, CRAFT]

# WBF IoU threshold for merging boxes
IOU_THR = 0.55

# Skip boxes with confidence below this threshold
SKIP_BOX_THR = 0.001

# Agreement filtering
REQUIRE_AGREEMENT = True
AGREEMENT_IOU_THRESH = 0.15
```

### Path Setup

```python
BASE_DIR = os.path.dirname(os.getcwd())  # Parent directory

# Input paths
ICDAR_IMAGE_DIR = os.path.join(BASE_DIR, "test-images", "ch4_test_images")
GT_DIR = os.path.join(BASE_DIR, "test-images", "Challenge4_Test_Task1_GT")

# Output paths
EAST_DIR  = os.path.join(BASE_DIR, "outputs", "east")
CRAFT_DIR = os.path.join(BASE_DIR, "outputs", "craft")
FUSED_DIR = os.path.join(BASE_DIR, "outputs", "fused")
```

### Run Fusion

Open and run `wbf.ipynb` in Jupyter Notebook or VS Code. The notebook will:
1. Load EAST and CRAFT predictions from `east/` and `craft/` folders
2. Apply weighted box fusion with configurable parameters
3. Save fused predictions to `fused/` folder
4. Generate comparison visualizations in `viz_compare/`

---

## 📥 Required Data Downloads

**⚠️ Note:** The following datasets are not included in this repository due to size constraints but are required for full reproduction:

1. **ICDAR 2015 Dataset** 
   - Download test images from [Kaggle - ICDAR 2015](https://www.kaggle.com/datasets/bestofbests9/icdar2015)
   - Place in: `../test-images/ch4_test_images/` (500 images)

2. **Pre-trained Models**
   - **EAST**: Download from [Kaggle - Frozen EAST Text Detection](https://www.kaggle.com/datasets/yelmurat/frozen-east-text-detection)
     - File: `frozen_east_text_detection.pb` → `../model/`
   - **CRAFT**: Download from [CRAFT-pytorch Repository](https://github.com/clovaai/CRAFT-pytorch)
     - File: `craft_ic15_20k.pth` → `../model/`

3. **ICDAR 2015 Ground Truth** 
   - Download ground truth annotations from [ICDAR 2015 Competition](https://rrc.cvc.uab.es/?ch=4&com=downloads)
   - Place in: `../test-images/Challenge4_Test_Task1_GT/` 
   - Required for evaluation metrics (5,230 text instances)

4. **Pre-generated Predictions**
   - EAST predictions → `outputs/east/`
   - CRAFT predictions → `outputs/craft/`
   - Each as `img_N.txt` files (N=1 to 500) in ICDAR format

---

## 📊 Evaluation & Results

### Dataset
- **ICDAR 2015 Text Localization**: 500 test images
- **Ground Truth**: 5,230 text instances (across 500 images)
- **Evaluation Metric**: IoU threshold = 0.5
- The ICDAR 2015 dataset is used under academic fair-use for research evaluation purposes.

### ICDAR 2015 Official Protocol

Evaluated using modified ICDAR 2015 evaluation framework:
- **Intersection over Union (IoU)** based matching
- **Precision**: Correctly detected / Total detected
- **Recall**: Correctly detected / Total ground truth  
- **F1-Score**: Harmonic mean of precision and recall

**📝 Evaluation Note:** Ground truth data (`../test-images/Challenge4_Test_Task1_GT/`) is excluded from git due to licensing restrictions.
Download from [ICDAR 2015 official source](https://rrc.cvc.uab.es/?ch=4&com=downloads) for reproduction.

### Performance Metrics

Below are the measured evaluation metrics using WBF fusion. Values are in ICDAR-format:

Precision, Recall, F1, plus True Positives (TP), Detections (Det) and Ground Truth (GT).

| Model | Precision | Recall | F1 | TP | Det | GT |
|---|---:|---:|---:|---:|---:|---:|
| EAST | 0.4665 | 0.6259 | 0.5345 | 1300 | 2787 | 2077 |
| CRAFT | 0.4814 | 0.4497 | 0.4650 | 934 | 1940 | 2077 |
| FUSED (WBF) | 0.4832 | 0.6892 | 0.5686 | 1432 | 2963 | 2077 |

### Performance Change (FUSED vs EAST) 🔄

- Precision: ➕ +0.0167 (+1.67 percentage points) — relative change: +3.58%  
- Recall: ✅ +0.0633 (+6.33 percentage points) — relative change: +10.12%  
- F1: ➕ +0.0341 (+3.41 percentage points) — relative change: +6.38%  
- TP (True Positives): ➕ +132 — relative change: +10.15%  
- Det (Detections): 🔼 +176 — relative change: +6.31%

**Summary vs EAST:**
- 🎯 WBF fusion increases Recall and F1 significantly (+6.33 pp recall, +3.41 pp F1) while maintaining similar precision, with +132 additional true positives detected.

### Performance Change (FUSED vs CRAFT) 🔁

- Precision: ➕ +0.0018 (+0.18 percentage points) — relative change: +0.37%  
- Recall: ✅ +0.2395 (+23.95 percentage points) — relative change: +53.26%  
- F1: ➕ +0.1036 (+10.36 percentage points) — relative change: +22.28%  
- TP (True Positives): ➕ +498 — relative change: +53.32%  
- Det (Detections): 🔼 +1023 — relative change: +52.73%

**Summary vs CRAFT:**
- 🚀 WBF fusion delivers massive improvements in Recall and F1 (+23.95 pp recall, +10.36 pp F1) with nearly unchanged precision, detecting +498 additional true positives.

---

## 📈 ICDAR-15 Protocol Evaluation Results

The following metrics were obtained using the **ICDAR-15 evaluation protocol** for comprehensive performance assessment:

### Performance Comparison Table

| Model | TP | FP | FN | Precision | Recall | F1-Score |
|-------|---:|---:|---:|----------:|-------:|---------:|
| **EAST** | 1506 | 710 | 571 | 67.96% | 72.51% | 70.16% |
| **CRAFT** | 1677 | 812 | 400 | 67.38% | 80.74% | 73.46% |
| **WBF (Ensemble)** | 1481 | 332 | 596 | 81.69% | 71.30% | 76.14% |

### Performance Improvement Analysis

#### 🎯 WBF vs EAST
- **Precision**: +13.73 pp (67.96% → 81.69%) — **+20.21% relative improvement**
- **Recall**: -1.21 pp (72.51% → 71.30%) — **-1.67% relative change**
- **F1-Score**: +5.98 pp (70.16% → 76.14%) — **+8.52% relative improvement**
- **False Positives**: -378 (710 → 332) — **-53.24% reduction**
- **True Positives**: -25 (1506 → 1481) — **-1.66% change**

**Key Insight:** WBF significantly improves precision through aggressive FP reduction (-53.24%) while maintaining comparable recall, resulting in a **5.98 pp F1-Score gain**.

#### 🚀 WBF vs CRAFT
- **Precision**: +14.31 pp (67.38% → 81.69%) — **+21.24% relative improvement**
- **Recall**: -9.44 pp (80.74% → 71.30%) — **-11.69% relative change**
- **F1-Score**: +2.68 pp (73.46% → 76.14%) — **+3.65% relative improvement**
- **False Positives**: -480 (812 → 332) — **-59.11% reduction**
- **True Positives**: -196 (1677 → 1481) — **-11.69% change**

**Key Insight:** WBF trades some recall for substantial precision gains (+14.31 pp), eliminating **59.11% of false positives** while still achieving a **higher F1-Score**.

### 🏆 Ensemble Benefits Summary

The Weighted Box Fusion ensemble achieves:
- ✅ **Highest Precision (81.69%)** — 13.73-14.31 pp better than individual models
- ✅ **Highest F1-Score (76.14%)** — 2.68-5.98 pp improvement over baselines
- ✅ **Lowest False Positive Rate** — 53-59% reduction in FPs
- ✅ **Balanced Performance** — Optimal precision-recall tradeoff

**Conclusion:** The WBF ensemble successfully fuses EAST and CRAFT predictions to deliver superior precision and F1-score while maintaining competitive recall, making it the most robust model for scene text detection.

---

## 📂 Repository Structure

```
outputs/
├── wbf.ipynb                    # Main fusion notebook
├── east/                        # EAST predictions (img_1.txt - img_500.txt)
│   ├── img_1.txt
│   ├── img_2.txt
│   └── ...
├── craft/                       # CRAFT predictions (img_1.txt - img_500.txt)
│   ├── img_1.txt
│   ├── img_2.txt
│   └── ...
├── fused/                       # WBF fused predictions (img_1.txt - img_500.txt)
│   ├── img_1.txt
│   ├── img_2.txt
│   └── ...
├── WBF/
│   └── visualizations/          # Comparison visualizations (PNG images)
│       ├── 17c98e4d-bcda-48da-971c-e0478cfc22c7.png
│       ├── a0c4b467-b3cf-40fb-b61a-24f971912004.png
│       └── ...
├── Project-Synopis-final.pdf    # Project documentation
└── README.md                    # This file
```

---

## 🔧 Algorithm Details

### Weighted Box Fusion Process

1. **Load Predictions**: Read EAST and CRAFT polygon predictions from text files
2. **Convert to Boxes**: Transform polygons to normalized bounding boxes [x1, y1, x2, y2]
3. **Apply WBF**: Use `ensemble_boxes.weighted_boxes_fusion()` with:
   - Model weights: [0.6, 0.4] (EAST prioritized)
   - IoU threshold: 0.55
   - Skip threshold: 0.001
4. **Agreement Filtering** (optional): Keep only boxes with IoU > 0.15 overlap in both models
5. **Polygon Mapping**: Map fused boxes back to original polygons using:
   - Center distance minimization
   - IoU overlap (threshold: 0.3)
   - Fallback to axis-aligned rectangle if no match

### Parameters Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_WEIGHTS` | [0.6, 0.4] | Confidence weights for EAST and CRAFT |
| `IOU_THR` | 0.55 | WBF fusion IoU threshold |
| `SKIP_BOX_THR` | 0.001 | Minimum confidence to keep a box |
| `REQUIRE_AGREEMENT` | True | Enable agreement filtering |
| `AGREEMENT_IOU_THRESH` | 0.15 | IoU threshold for agreement |

---

## 📄 Output Format

Fused predictions are saved as text files in ICDAR format:

```
x1,y1,x2,y2,x3,y3,x4,y4
```

Each line represents one detected text region with 4 polygon vertices (8 coordinates total).

---

## 🎨 Visualization Outputs

The `WBF/visualizations/` folder contains comparison images showing:
- **Blue boxes**: EAST detections
- **Red boxes**: CRAFT detections  
- **Green boxes**: WBF fused detections

---

## 📜 License & Citation

- **License**: MIT (academic). Please cite if used in research.
- **Attribution**: Mention the use of Weighted Box Fusion and any parameter modifications.

### Citation

If you use this work, please cite:

```bibtex
@misc{east-craft-wbf,
  author = {Your Name},
  title = {EAST and CRAFT Weighted Box Fusion Ensemble for Scene Text Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/your-repo}
}
```

---

## ✍️ Author

### Nabil Ahmed 
**Contact Information:**
- 📧 Email: [nabil13147@gmail.com](mailto:nabil13147@gmail.com)
- 🐙 GitHub: [@TerrArx](https://github.com/TerrArx)
- 🔗 Repository: [EAST-Craft-WBF](https://github.com/TerrArx/EAST-Craft-WBF)

---

## 🤝 Acknowledgments

- **EAST Model**: Pre-trained weights from [Kaggle - Frozen EAST Text Detection](https://www.kaggle.com/datasets/yelmurat/frozen-east-text-detection)
- **CRAFT Model**: Original implementation and weights from [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) by Clova AI
- **ICDAR 2015**: Text localization evaluation dataset and ground truth annotations
- **Weighted Box Fusion**: Implementation from [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) by ZFTurbo

---

## 📌 Notes

- **Agreement Filtering**: Setting `REQUIRE_AGREEMENT=True` keeps only detections that both models agree on (IoU > 0.15), reducing false positives but potentially lowering recall.
- **Model Weights**: EAST is weighted at 0.6 (vs CRAFT at 0.4) based on empirical performance. Adjust these based on your specific dataset characteristics.
- **IoU Threshold**: The fusion IoU threshold of 0.55 balances between merging similar detections and keeping distinct ones. Lower values merge more aggressively.

---

## 🔍 Troubleshooting

**Issue**: No detections in fused output
- Check that EAST and CRAFT prediction files exist and are non-empty
- Verify image paths are correct
- Try lowering `AGREEMENT_IOU_THRESH` or setting `REQUIRE_AGREEMENT=False`

**Issue**: Too many false positives
- Increase `SKIP_BOX_THR` to filter low-confidence detections
- Enable `REQUIRE_AGREEMENT=True` 
- Increase `AGREEMENT_IOU_THRESH`

**Issue**: Missing ground truth detections (low recall)
- Lower `IOU_THR` to merge more aggressively
- Disable `REQUIRE_AGREEMENT` to keep all detections
- Adjust `MODEL_WEIGHTS` to favor the model with higher recall

---


