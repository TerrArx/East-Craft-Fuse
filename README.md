# East and Craft Polygon Fusion Ensemble

 A polygon-fusion ensemble that combines two scene-text detectors — **EAST** and **CRAFT** — to improve recall and robustness. The pipeline clusters overlapping polygons, aligns vertices, and computes a weighted fusion to produce consolidated text detections (TXT/JSON) and visual comparisons.

<p align="center">
  <img src="Polygon-Fusion\img_100_compare.png" width="1200">
</p>
<p align="center"><i>Comparison of EAST (Red), CRAFT (Blue), and Polygon Fusion (Green)</i></p>

<div align="center">
  
- ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  - ![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg)  - ![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)  - ![License](https://img.shields.io/badge/License-Academic-yellow.svg)
  
</div>
---

Why this project?
- ✅ Improve Recall: combine complementary strengths of EAST (fast quadrilaterals) and CRAFT (flexible polygons)  
- 🔄 Robustness: ensemble reduces model-specific misses and produces more stable detections  
- 🎯 Easy evaluation: produces ICDAR-style TXT/JSON outputs and side-by-side visualizations

---

## Key Features
- 🧭 Greedy IoU-based clustering of polygon detections from both models  
- 🔁 Standardization: resamples complex polygons into 4-point quadrilaterals (with safe fallbacks)  
- ⚖️ Weighted vertex averaging using model confidence and configurable model bias  
- 🛡️ Validation + fallbacks (convex hull or best original polygon) to avoid degenerate outputs  
- 📊 Automatic ICDAR-style evaluation (Precision / Recall / F1) and visualization exports

---

## Core Methodology
<p align="center">

 ```
                           ┌──────────────────────────┐
                           │      START PIPELINE      │
                           └─────────────┬────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │ Load Image from test-images/     │
                         └─────────────────┬─────────────────┘
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 ▼                         ▼                         ▼
  ┌──────────────────────────────┐   ┌──────────────────────────┐   │
  │ Run EAST Detector (east.py)  │   │ Run CRAFT Detector       │   │
  │ - Rotated boxes → polygons   │   │ (craft.py)               │   │
  │ - Conf-threshold + NMS       │   │ - Irregular polygons     │   │
  └──────────────────────────────┘   └──────────────────────────┘   │
                 │                         │                         │
                 └─────────────── Collect Predictions ───────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │ Combine as (poly, score, model) │
                         └─────────────────┬────────────────┘
                                           │
                                           ▼
                           ┌────────────────────────────────┐
                           │ Greedy IoU Clustering          │
                           │ - Sort by score                │
                           │ - Group polys with IoU ≥ T     │
                           └─────────────────┬──────────────┘
                                             │
                           For each cluster  │
                          ───────────────────┘
                                             ▼
                        ┌──────────────────────────────────┐
                        │ Resample polygons → 4 vertices   │
                        │ (approxPolyDP or min-area-rect)  │
                        └───────────────────┬──────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────┐
                     │ Align vertices to anchor polygon         │
                     │ - Anchor = largest-area polygon          │
                     │ - Try 4 rotations → minimal L2 distance  │
                     └───────────────────┬──────────────────────┘
                                         │
                                         ▼
                       ┌───────────────────────────────────────┐
                       │ Weighted Vertex Averaging             │
                       │ - weights = confidence × model bias   │
                       │ - vertex[j] = Σ(wᵢ * vᵢ[j])           │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────┐
                   │ Validate fused polygon                      │
                   │ - If invalid → hull → min-area rectangle    │
                   │ - Else keep fused polygon                   │
                   └───────────────────┬──────────────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────────┐
                     │ Save Visualization                    │
                     │ (EAST / CRAFT / FUSED side-by-side)   │
                     │ → Polygon-Fusion/viz/*.jpg            │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Next Image?         │
                              └───────┬────────────┘
                                      │Yes
                                      ▼
                          (Repeat Entire Pipeline Loop)
                                      │
                                      ▼
                              ┌─────────────────────┐
                              │      END PIPELINE    │
                              └─────────────────────┘


 ```

</p>

Summary (step-by-step)
1. 🔍 Individual Detection  
   - Run EAST → rotated 4-point boxes.  
   - Run CRAFT → variable-point polygons.

2. 🧩 Unification  
   - Collect all polygons into a single list, excluding high-scoring ground-truth "don't care" polygons.

3. 🧠 Greedy IoU Clustering  
   - Sort polygons by confidence (desc).  
   - Seed a cluster with the highest-confidence polygon and add all polygons with polygon-IoU >= CLUSTER_IOU_THRESH (default 0.40).  
   - Repeat until every polygon is clustered.

4. 🛠️ Standardization & Alignment  
   - Resample each polygon in a cluster to 4-point quadrilaterals using cv2.approxPolyDP; fallback to cv2.minAreaRect if needed.  
   - Choose the largest-area polygon as the cluster anchor and rotate other polygons' vertex order to minimize per-vertex L2 distance to the anchor.

5. ⚖️ Weighted Vertex Averaging (Fusion)  
   - Each polygon contributes with weight = confidence_score * MODEL_WEIGHTS[model_name].  
   - Fused vertex coordinates are the weighted averages of aligned vertices.

6. ✅ Validation & Fallback  
   - Ensure fused polygon has non-zero area and conforms to basic geometry checks.  
   - If invalid, use convex hull of cluster or the highest-confidence original polygon.

---

## Quickstart

1) Install dependencies
```bash
pip install shapely craft-text-detector==0.4.3 tqdm opencv-python-headless
pip install opencv-python
pip install numpy
pip install tqdm
pip install ensemble-boxes
pip install shapely
pip install matplotlib

```

2) Configure paths (edit constants in the script / notebook)
- EAST_MODEL_PATH = 'models/frozen_east_text_detection.pb'
- CRAFT_MODEL_PATH = 'models/craft_ic15_20k.pth'
- IMG_DIR = Path('test-images/ch4_test_images')
- GT_DIR = Path('test-images/Challenge4_Test_Task1_GT')
- OUTPUT_ROOT = Path('Polygon-Fusion/outputs')

3) Run
- Python script:
```bash
python run_fusion.py --img-dir test-images/ch4_test_images --output Polygon-Fusion/outputs
```
- Or execute the Jupyter notebook cells in order.

---

## Configuration & Tuning

Core parameters (examples)
- CLUSTER_IOU_THRESH = 0.40  # IoU threshold for clustering
- MODEL_WEIGHTS = {'east': 1.0, 'craft': 1.0}  # relative importance of models
- APPROX_EPS = 0.01  # epsilon for cv2.approxPolyDP resampling

Tuning tips
- Lower CLUSTER_IOU_THRESH to merge more detections (increase recall, risk more false merges).  
- Increase MODEL_WEIGHTS['craft'] if you want CRAFT's shapes to dominate fused vertices.  
- Use APPROX_EPS to control polygon simplification: higher = simpler (fewer vertices).

---

## Outputs (what you'll find in OUTPUT_ROOT)
- /EAST/ — raw EAST predictions (.txt, .json)  
- /CRAFT/ — raw CRAFT predictions (.txt, .json)  
- /Fused/ — final ensembled predictions (.txt, .json)  
- /Visualization/ — side-by-side comparison images (EAST | CRAFT | FUSED) for quick manual inspection

### Repository Structure:
<p align="center">
  
```
|East-Craft-Fuse/
│
│
├── EAST/
│   ├── east.py                              # EAST detector script
│   └── outputs/                             # EAST outputs (optional: images/logs only)
│       └── ...                              
│
├── CRAFT/
│   ├── craft.py                             # CRAFT detector script
│   └── outputs/                             # CRAFT outputs (optional: images/logs only)
│       └── ...
│
├── Polygon-Fusion/
│   └── outputs/
│       └── img_XXXX_compare.jpg
│       ├── img_XXXX_compare.jpg
│       └── ...                              # ONLY visualizations of fusion
│              
│        
├── poly-fusion.py                           # Fusion logic (post-processing)
├── README.md
└── requirements.txt

```
  
</p>

---

## Evaluation & Examples

- The pipeline runs an ICDAR-style evaluation comparing EAST, CRAFT, and Fused outputs against GT_DIR (Precision, Recall, F1).  
- Example visualization: assets/compare-example.png

---

## Fallback & Robustness Notes
- If 4-point resampling fails for a polygon, we use min-area-rect to avoid discarding detections.  
- If averaged polygon is degenerate (zero area / self-intersecting), we first try the convex hull of the cluster; then fall back to the cluster's highest-confidence polygon.  
- The alignment step reduces twisted averaging by rotating polygon vertex sequences to best match an anchor polygon.

---

## Troubleshooting
- If visualizations are missing: confirm OUTPUT_ROOT/Visualization was written and images exist for the first 10 images.  
- If CRAFT fails to load: check PyTorch & path to CRAFT model .pth.  
- If EAST outputs are empty: verify the frozen graph path and OpenCV compatibility.

---

## Contributing
- Bug fixes, clearer diagrams, and improved fallbacks are welcome.  
- Prefer small, focused PRs. Add tests where reasonable (e.g., unit tests for alignment, clustering logic).

---

##  Author: 
### Nabil Ahmed  
#### Netaji Subhash Engineering College  
#### B.Tech | Artificial Intelligence & Machine Learning  

</div>


---

## License & Citation
- License: Academic (MIT License)  
- If you use this method in research, please cite or credit the repository and include a brief description of any weightings or fusion changes you used.
