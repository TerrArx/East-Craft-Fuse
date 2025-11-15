# 🔗 East and Craft Polygon Fusion Ensemble

A polygon-fusion ensemble that combines two scene-text detectors — **EAST** and **CRAFT** — to improve recall and robustness. The pipeline clusters overlapping polygons, aligns vertices, and fuses detections to reduce misses and stabilize outputs.

<p align="center">
  <img src="Polygon-Fusion/img_100_compare.png" width="1200">
</p>
<p align="center"><i>Comparison of EAST (Blue), CRAFT (Red), and Polygon Fusion (Green)</i></p>

<div align="center">
 
- ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) - ![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg) - ![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg) - ![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)
</div>

---
## 🎨 Visual Results

A dedicated visual index makes it easy to inspect each image's EAST / CRAFT / FUSED outputs.

<p align="center"><i>EAST (Green)</i></p>
<p align="center">
  <img src="EAST\output.visualizations\2.png" width="900">
</p>

---

<p align="center"><i>CRAFT (Green)</i></p>
<p align="center">
  <img src="CRAFT\visualizations\img_103_viz.png" width="900">
</p>

---

<p align="center"><i>Comparison of EAST (Blue), CRAFT (Red), and Polygon Fusion (Green)</i></p>
<p align="center">
  <img src="Polygon-Fusion\img_107_compare.png" width="900">
</p>



---

## ✨ Key Features
- 🧭 Greedy IoU clustering of polygon detections from EAST & CRAFT  
- 🔁 Resampling & alignment to unify vertices (4-point fallback)  
- ⚖️ Weighted vertex averaging using per-model confidence and bias  
- 🛡️ Validation + safe fallbacks (convex hull / highest-confidence polygon)  
- 📊 Automated ICDAR-style evaluation and visualization exports

---

## Models

## 🚀 Quickstart (minimal)

### Prerequisites
```bash
# Create environment
conda create -n text_detection python=3.8
conda activate text_detection

# Install dependencies
pip install -r requirements.txt
pip install craft-text-detector
```

2. Configure paths in the script/notebook:
- EAST_MODEL_PATH = 'models/frozen_east_text_detection.pb'  
- CRAFT_MODEL_PATH = 'models/craft_ic15_20k.pth'  
- IMG_DIR, GT_DIR, OUTPUT_ROOT (Polygon-Fusion/outputs)
3. Run:
```bash
python run_fusion.py --img-dir test-images/ch4_test_images --output Polygon-Fusion/outputs
```

### 📥 Required Data Downloads

**⚠️ Note:** The following datasets are not included in this repository due to size constraints but are required for full reproduction:

1. **ICDAR 2015 Dataset** 
   - Download test images from [Kaggle - ICDAR 2015](https://www.kaggle.com/datasets/bestofbests9/icdar2015) (search "ICDAR 2015")
   - Place in: `data/icdar2015/test_images/` (500 images)

2. **Pre-trained Models**
   - **EAST**: Download from [Kaggle - Frozen EAST Text Detection](https://www.kaggle.com/datasets/yelmurat/frozen-east-text-detection)
     - File: `frozen_east_text_detection.pb` → `models/`
   - **CRAFT**: Download from [CRAFT-pytorch Repository](https://github.com/clovaai/CRAFT-pytorch)
     - File: `craft_ict15_20k.pth` → `models/`

3. **ICDAR 2015 Ground Truth** 
   - Download ground truth annotations from [ICDAR 2015 Competition](https://rrc.cvc.uab.es/?ch=4&com=downloads)
   - Place in: `icdar_eval/gt/` 
   - Required for evaluation metrics (5,230 text instances)
---

## Evaluation & Results

### Dataset
- **ICDAR 2015 Text Localization**: 500 test images
- **Ground Truth**: 5,230 text instances
- **Evaluation Metric**: IoU threshold = 0.5
- The ICDAR 2015 dataset is used under academic fair-use for research evaluation purposes.

### ICDAR 2015 Official Protocol
Evaluated using modified ICDAR 2015 evaluation framework:
- **Intersection over Union (IoU)** based matching
- **Precision**: Correctly detected / Total detected
- **Recall**: Correctly detected / Total ground truth  
- **F1-Score**: Harmonic mean of precision and recall

**📝 Evaluation Note:** Ground truth data (`icdar_eval/`) is excluded from git due to licensing restrictions.
Download from [ICDAR 2015 official source](https://rrc.cvc.uab.es/?ch=4&com=downloads) for reproduction.


Below are the measured evaluation metrics for the run provided. Values are in ICDAR-format:

Precision, Recall, F1, plus True Positives (TP), Detections (Det) and Ground Truth (GT).

| Model | Precision | Recall | F1 | TP | Det | GT |
|---|---:|---:|---:|---:|---:|---:|
| EAST | 0.4665 | 0.6259 | 0.5345 | 1300 | 2787 | 2077 |
| CRAFT | 0.4814 | 0.4497 | 0.4650 | 934 | 1940 | 2077 |
| FUSED | 0.4257 | 0.7236 | 0.5360 | 1503 | 3531 | 2077 |

### Performance change (FUSED compared to EAST) 🔄

- Precision: 🔽 −0.0408 (−4.08 percentage points) — relative change: −8.75%  
- Recall: ✅ +0.0977 (+9.77 percentage points) — relative change: +15.62%  
- F1: ➕ +0.0015 (+0.15 percentage points) — relative change: +0.28%  
- TP (True Positives): ➕ +203 — relative change: +15.62%  
- Det (Detections): 🔼 +744 — relative change: +26.70%

Summary vs EAST: 
- 🔎 Fusion substantially increases Recall and True Positives (+9.77 pp, +203 TP), with a small net F1 improvement (+0.15 pp).

### Performance change (FUSED compared to CRAFT) 🔁

- Precision: 🔽 −0.0557 (−5.57 percentage points) — relative change: −11.58%  
- Recall: ✅ +0.2739 (+27.39 percentage points) — relative change: +60.90%  
- F1: ➕ +0.0710 (+7.10 percentage points) — relative change: +15.27%  
- TP (True Positives): ➕ +569 — relative change: +60.96%  
- Det (Detections): 🔼 +1591 — relative change: +82.06%

Summary vs CRAFT:
- 🚀 Fusion delivers very large increases in Recall and F1 (+27.39 pp recall, +7.10 pp F1) and far more true positives (+569).

---


## 📂 Outputs (what you'll find in OUTPUT_ROOT)
- /EAST/ — raw EAST predictions (.txt, .json)  
- /CRAFT/ — raw CRAFT predictions (.txt, .json)  
- /Fused/ — final ensembled predictions (.txt, .json)  
- /Visualization/ — side-by-side comparison images (EAST | CRAFT | FUSED) and index.html

---

## 🗂 Repository structure
```
|East-Craft-Fuse/
├── EAST/
│   └── east.py
├── CRAFT/
│   └── craft.py
├── Polygon-Fusion/
│   └── outputs/
│       └── Visualization/ (img_XXXX_east/craft/fused/compare + index.html)
├── poly-fusion.py
├── README.md
└── requirements.txt
```

---

## 📜 License & Citation
- License: MIT (academic). Please cite if used in research and mention any weightings or fusion modifications.

---

## ✍️ Author
### Nabil Ahmed 
**Contact Information:**
- 📧 Email: [nabil13147@gmail.com](mailto:nabil13147@gmail.com)
- 🐙 GitHub: [@TerrArx](https://github.com/TerrArx)
- 🔗 Repository: [EAST-Craft-Fuse](https://github.com/TerrArx/East-Craft-Fuse)

## 🤝 Acknowledgments

- **EAST Model**: Pre-trained weights from [Kaggle - Frozen EAST Text Detection](https://www.kaggle.com/datasets/yelmurat/frozen-east-text-detection)
- **CRAFT Model**: Original implementation and weights from [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) by Clova AI
- **ICDAR 2015**: Text localization evaluation dataset and ground truth
