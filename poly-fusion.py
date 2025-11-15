#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install shapely craft-text-detector==0.4.3 tqdm')


# In[2]:


# In[2]:
import os
import cv2
import glob
import json
import time
import math
import shutil
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union


# In[3]:


# In[3]:
# Cell 2: Paths & parameters (update if needed)
EAST_MODEL_PATH = 'models/frozen_east_text_detection.pb'
CRAFT_MODEL_PATH = "models/craft_ic15_20k.pth"
IMG_DIR = Path('test-images/ch4_test_images')
GT_DIR  = Path('test-images/Challenge4_Test_Task1_GT')

OUTPUT_ROOT = Path('Polygon-Fusion/outputs')
EAST_OUT = OUTPUT_ROOT / 'east'
CRAFT_OUT = OUTPUT_ROOT / 'craft'
FUSED_OUT = OUTPUT_ROOT / 'fused'
VIZ_OUT = OUTPUT_ROOT / 'viz'
for d in [EAST_OUT, CRAFT_OUT, FUSED_OUT, VIZ_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# Image preprocess and EAST params
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
EAST_CONF_THRESH = 0.5
EAST_NMS_THRESH = 0.4

# Fusion params
CLUSTER_IOU_THRESH = 0.40   # polygon IoU threshold for clustering (recommended 0.35-0.50)
MIN_VERTEX_COUNT = 4
VERTEX_APPROX_EPS = 0.02    # fraction of perimeter for approxPolyDP (used by resampling)
MODEL_WEIGHTS = {'east': 1.0, 'craft': 1.0}  # relative bias for models (tweakable)
VIS_LIMIT = 10
IOU_EVAL_THRESH = 0.50


# In[4]:


# In[4]:
# Cell 3: EAST loader + decode + detection function
if not os.path.exists(EAST_MODEL_PATH):
    raise FileNotFoundError(f"EAST model not found at {EAST_MODEL_PATH}")
east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
print("EAST loaded from:", EAST_MODEL_PATH)

def decode_rotated_predictions(scores, geometry, confThresh):
    (numRows, numCols) = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            score = float(scoresData[x])
            if score < confThresh:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = float(anglesData[x])
            cos = math.cos(angle)
            sin = math.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            offset = ([offsetX + cos * xData1[x] + sin * xData2[x],
                       offsetY - sin * xData1[x] + cos * xData2[x]])
            p1 = (-sin * h + offset[0], -cos * h + offset[1])
            p3 = (-cos * w + offset[0], sin * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))

            boxes.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    return boxes, confidences

def east_detect_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h_orig, w_orig = img.shape[:2]
    rW = w_orig / float(INPUT_WIDTH)
    rH = h_orig / float(INPUT_HEIGHT)

    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = cv2.dnn.blobFromImage(img_resized, 1.0, (INPUT_WIDTH, INPUT_HEIGHT),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    boxes, confidences = decode_rotated_predictions(scores, geometry, EAST_CONF_THRESH)

    # NMSBoxesRotated returns indices
    preds = []
    if len(boxes) > 0:
        try:
            indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, EAST_CONF_THRESH, EAST_NMS_THRESH)
            if hasattr(indices, 'flatten'):
                inds = indices.flatten()
            else:
                inds = [int(i[0]) for i in indices]
        except Exception:
            inds = list(range(len(boxes)))
    else:
        inds = []

    for i in inds:
        rect = boxes[i]
        pts = cv2.boxPoints(rect)
        pts[:, 0] *= rW
        pts[:, 1] *= rH
        poly = pts.tolist()
        preds.append((poly, float(confidences[i])))
    return preds


# In[5]:


# In[5]:
# Cell 4: CRAFT loader & detect (patched numpy/adjustment handling)
import torch
import warnings
warnings.filterwarnings('ignore')

# Apply same craft utils patch and numpy patch
import craft_text_detector.craft_utils as craft_utils
original_adjustResultCoordinates = craft_utils.adjustResultCoordinates
def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates

_original_np_array = np.array
def _patched_np_array(arr_input, dtype=None, **kwargs):
    try:
        return _original_np_array(arr_input, dtype=dtype, **kwargs)
    except ValueError as e:
        error_msg = str(e)
        if "inhomogeneous shape" in error_msg or "setting an array element with a sequence" in error_msg:
            return _original_np_array(arr_input, dtype=object)
        raise
np.array = _patched_np_array

from craft_text_detector import Craft
from pathlib import Path

def load_craft_model(model_path, use_cuda=False):
    craft = Craft(output_dir=None,
                  crop_type="poly",
                  cuda=use_cuda,
                  text_threshold=0.7,
                  link_threshold=0.4,
                  low_text=0.4,
                  export_extra=False)
    model_path = Path(model_path)
    if model_path.exists():
        state_dict = torch.load(str(model_path), map_location='cpu')
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        craft.craft_net.load_state_dict(new_state_dict)
        craft.craft_net.eval()
    else:
        print("CRAFT weights not found at", model_path)
    return craft

craft_detector = load_craft_model(CRAFT_MODEL_PATH, use_cuda=torch.cuda.is_available())
print("CRAFT loaded.")

def craft_detect_image(img_path):
    try:
        res = craft_detector.detect_text(str(img_path))
    except Exception as e:
        print("CRAFT detect_text failed for", img_path, ":", e)
        return []

    dets = []
    if 'boxes' in res and res['boxes'] is not None:
        scores = res.get('scores', None)
        if scores is None:
            scores = [0.9] * len(res['boxes'])
        for box, score in zip(res['boxes'], scores):
            try:
                # Return polygon as numpy array (like craft-3)
                poly_arr = np.array(box).astype(int)
                dets.append((poly_arr, float(score)))
            except Exception:
                continue
    return dets


# In[6]:


# In[6]:
# Cell 5: Utilities for polygon handling, clustering, and fusion
def poly_to_shapely(poly):
    return Polygon(poly)

def polygon_iou(poly_a, poly_b):
    """IoU of two polygons using shapely"""
    try:
        a = Polygon(poly_a)
        b = Polygon(poly_b)
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def reorder_polygon_ccw(poly):
    """Return polygon vertices ordered CCW starting from the leftmost-top if possible."""
    arr = np.array(poly, dtype=float)
    cx = arr[:,0].mean()
    cy = arr[:,1].mean()
    angles = np.arctan2(arr[:,1]-cy, arr[:,0]-cx)
    order = np.argsort(angles)
    arr = arr[order]
    # ensure consistent orientation (ccw); shapely's area sign check
    if Polygon(arr).area < 0:
        arr = arr[::-1]
    return arr.tolist()

def resample_polygon_to_4(poly):
    """Approximate/simplify polygon to 4 points using approxPolyDP; fallback to min-area rect"""
    pts = np.array(poly, dtype=np.float32)
    # compute perimeter to pick eps
    peri = cv2.arcLength(pts.reshape((-1,1,2)), True)
    eps = max(1.0, VERTEX_APPROX_EPS * peri)
    approx = cv2.approxPolyDP(pts.reshape((-1,1,2)), eps, True)
    if approx.shape[0] == 4:
        out = approx.reshape((4,2)).tolist()
        return reorder_polygon_ccw(out)
    # if more than 4, try polygon simplification iteratively
    if approx.shape[0] > 4:
        # reduce more aggressively
        eps2 = eps * 2.5
        approx2 = cv2.approxPolyDP(pts.reshape((-1,1,2)), eps2, True)
        if approx2.shape[0] == 4:
            return reorder_polygon_ccw(approx2.reshape((4,2)).tolist())
    # fallback: min-area rectangle from contour
    box = cv2.minAreaRect(pts)
    box_pts = cv2.boxPoints(box)
    return reorder_polygon_ccw(box_pts.tolist())

def align_vertices_to_anchor(polys):
    """
    Polys: list of Nx2 arrays/lists. Convert them to same vertex count (4) and align
    by rotating vertex order to minimize L2 distance to anchor polygon.
    Returns list of arrays shape (4,2)
    """
    # ensure each poly is 4-point
    processed = []
    for p in polys:
        p4 = resample_polygon_to_4(p)
        processed.append(np.array(p4, dtype=float))

    # choose anchor as polygon with largest area (more stable)
    areas = [Polygon(p).area for p in processed]
    anchor_idx = int(np.argmax(areas)) if len(areas) > 0 else 0
    anchor = processed[anchor_idx]

    aligned = []
    for p in processed:
        # try all rotations (4) and pick rotation with smallest sum distance to anchor
        best = None
        best_dist = float('inf')
        for r in range(4):
            candidate = np.roll(p, -r, axis=0)
            dist = np.linalg.norm(candidate - anchor)
            if dist < best_dist:
                best_dist = dist
                best = candidate.copy()
        aligned.append(best)
    return aligned

def weighted_vertex_average(polys, scores, model_biases=None):
    """
    polys: list of (4x2 arrays)
    scores: list of floats (same length)
    model_biases: list of multiplicative biases matching polys order OR None
    returns fused_poly (4x2 list) and fused_score
    """
    weights = np.array(scores, dtype=float)
    if model_biases is not None:
        weights = weights * np.array(model_biases, dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    polys_arr = np.stack(polys, axis=0)  # shape (k,4,2)
    fused = np.tensordot(weights, polys_arr, axes=(0,0))  # (4,2)
    fused_poly = fused.tolist()

    # fused score: weighted mean of scores (without biases)
    fused_score = float(np.dot(weights, scores))
    return fused_poly, fused_score


# In[7]:


# In[7]:
# Cell 6: Clustering + fuse cluster function
def greedy_polygon_clustering(preds, iou_thresh=CLUSTER_IOU_THRESH):
    """
    preds: list of tuples (poly (list Nx2 or np.array), score, model_name)
    returns: list of clusters, each cluster = list of indices into preds
    Greedy: pick highest-score remaining polygon, gather all polygons with IoU >= thresh to it, remove them, repeat.
    """
    indices = list(range(len(preds)))
    used = set()
    clusters = []

    # create shapely polygons (use list conversion to be safe)
    shapely_polys = [Polygon(np.array(p[0]).tolist()) for p in preds]

    # sort by score descending
    order = sorted(indices, key=lambda i: preds[i][1], reverse=True)

    for i in order:
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        base_poly = preds[i][0]
        for j in order:
            if j in used:
                continue
            iou = polygon_iou(base_poly, preds[j][0])
            if iou >= iou_thresh:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    # handle any leftover (shouldn't be any)
    remaining = [i for i in indices if i not in used]
    for i in remaining:
        clusters.append([i])
    return clusters

def fuse_cluster(preds, cluster_indices):
    """
    preds: list of (poly, score, model_name)
    cluster_indices: indices in preds
    returns fused_poly (4x2 list), fused_score
    """
    # collect polygons and scores and model biases
    polys = [preds[i][0] for i in cluster_indices]
    scores = [preds[i][1] for i in cluster_indices]
    models = [preds[i][2] for i in cluster_indices]
    # get model biases
    model_biases = [MODEL_WEIGHTS.get(m, 1.0) for m in models]

    # align / resample to 4 vertices and align orientation
    aligned = align_vertices_to_anchor(polys)  # list of 4x2 arrays

    # weighted average
    fused_poly, fused_score = weighted_vertex_average(aligned, scores, model_biases)

    # validate fused polygon
    try:
        poly_shp = Polygon(fused_poly)
        if not poly_shp.is_valid or poly_shp.area <= 1.0:
            # fallback: convex hull of cluster union
            union = unary_union([Polygon(polys[i]) for i in cluster_indices])
            hull = union.convex_hull
            # sample 4 points from hull as min-area rect fallback
            if hull.area > 1.0:
                # use minAreaRect on hull exterior coords
                coords = np.array(hull.exterior.coords)[:-1].astype(np.float32)
                if len(coords) >= 3:
                    rect = cv2.minAreaRect(coords)
                    box = cv2.boxPoints(rect)
                    fused_poly = reorder_polygon_ccw(box.tolist())
                    fused_score = fused_score
                else:
                    # very small, fallback to highest score polygon
                    best_idx = cluster_indices[np.argmax(scores)]
                    fused_poly = resample_polygon_to_4(preds[best_idx][0])
                    fused_score = preds[best_idx][1]
            else:
                best_idx = cluster_indices[np.argmax(scores)]
                fused_poly = resample_polygon_to_4(preds[best_idx][0])
                fused_score = preds[best_idx][1]
    except Exception:
        best_idx = cluster_indices[np.argmax(scores)]
        fused_poly = resample_polygon_to_4(preds[best_idx][0])
        fused_score = preds[best_idx][1]

    # final cleanup: integer coords, clip small negatives
    fused_poly = [[max(0.0, float(round(x))), max(0.0, float(round(y)))] for (x,y) in fused_poly]
    return fused_poly, fused_score


# In[8]:


# In[8]:
# Cell 7: Run detectors on all images, fuse polygons, save outputs, and visualize examples
img_paths = sorted([p for p in IMG_DIR.glob('*') if p.suffix.lower() in ['.jpg','.png','.jpeg','.bmp']])
print("Images found:", len(img_paths))

east_results = {}   # stem -> [(poly, score)]
craft_results = {}
fused_results = {}

viz_count = 0
start_time = time.time()

for idx, img_path in enumerate(tqdm(img_paths, desc="Processing images")):
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        east_results[stem] = []
        craft_results[stem] = []
        fused_results[stem] = []
        continue
    h, w = img.shape[:2]

    # EAST
    east_preds = east_detect_image(img_path)  # list of (poly, score)
    east_results[stem] = east_preds
    save_lines = [(p, s) for (p,s) in east_preds]
    # Save EAST outputs
    with open(EAST_OUT / f"{stem}.txt", 'w', encoding='utf-8') as f:
        for poly, sc in save_lines:
            coords = ','.join([str(int(round(c))) for p in poly for c in p])
            f.write(f"{coords},{sc:.4f}\n")
    with open(EAST_OUT / f"{stem}.json", 'w', encoding='utf-8') as fj:
        json.dump([{'poly': [[int(round(c)) for c in p] for p in poly], 'score': float(sc)} for poly,sc in save_lines], fj, indent=2)

    # CRAFT
    craft_preds = craft_detect_image(img_path)  # list of (np.array poly, score)
    # convert to the same saving format but keep numpy internally
    craft_results[stem] = craft_preds
    with open(CRAFT_OUT / f"{stem}.txt", 'w', encoding='utf-8') as f:
        for poly, sc in craft_preds:
            coords = ','.join([str(int(round(c))) for p in poly for c in p])
            f.write(f"{coords},{sc:.4f}\n")
    with open(CRAFT_OUT / f"{stem}.json", 'w', encoding='utf-8') as fj:
        json.dump([{'poly': [[int(round(c)) for c in p] for p in np.array(poly).tolist()], 'score': float(sc)} for poly,sc in craft_preds], fj, indent=2)

    # Prepare unified preds list with model tag for clustering
    unified = []
    for poly, sc in east_preds:
        unified.append((poly, float(sc), 'east'))
    for poly, sc in craft_preds:
        unified.append((poly, float(sc), 'craft'))

    # Cluster
    clusters = greedy_polygon_clustering(unified, iou_thresh=CLUSTER_IOU_THRESH)

    # Fuse each cluster
    fused_list = []
    for cluster in clusters:
        fused_poly, fused_sc = fuse_cluster(unified, cluster)
        fused_list.append((fused_poly, fused_sc))

    fused_results[stem] = fused_list

    # Save fused outputs
    with open(FUSED_OUT / f"{stem}.txt", 'w', encoding='utf-8') as f:
        for poly, sc in fused_list:
            coords = ','.join([str(int(round(c))) for p in poly for c in p])
            f.write(f"{coords},{sc:.4f}\n")
    with open(FUSED_OUT / f"{stem}.json", 'w', encoding='utf-8') as fj:
        json.dump([{'poly': [[int(round(c)) for c in p] for p in poly], 'score': float(sc)} for poly,sc in fused_list], fj, indent=2)

    # Visualize first VIS_LIMIT images (side-by-side)
    if viz_count < VIS_LIMIT:
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        # EAST (red)
        img_e = img.copy()
        for poly, sc in east_preds:
            pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(img_e, [pts], isClosed=True, color=(255,0,0), thickness=2)
        axes[0].imshow(cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB))
        axes[0].axis('off')
        axes[0].set_title('EAST (BLUE)', pad=10)

        # CRAFT (blue)
        img_c = img.copy()
        for poly, sc in craft_preds:
            pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(img_c, [pts], isClosed=True, color=(0,0,255), thickness=2)
        axes[1].imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
        axes[1].axis('off')
        axes[1].set_title('CRAFT (RED)', pad=10)

        # FUSED (green)
        img_f = img.copy()
        for poly, sc in fused_list:
            pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(img_f, [pts], isClosed=True, color=(0,255,0), thickness=2)
        axes[2].imshow(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
        axes[2].axis('off')
        axes[2].set_title('FUSED (GREEN)', pad=10)

        plt.suptitle(stem)
        viz_path = VIZ_OUT / f"{stem}_compare.png"
        plt.tight_layout()
        plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        viz_count += 1

end_time = time.time()
print(f"Done. Time elapsed: {end_time - start_time:.1f}s")
print("Outputs ->", OUTPUT_ROOT)
print("Visuals ->", VIZ_OUT)


# In[9]:


# In[9]:
# Cell 8: Evaluate EAST, CRAFT, and FUSED outputs vs ICDAR GT (polygon IoU)
def load_icdar_gt(gt_path):
    polys = []
    if not gt_path.exists():
        return polys
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                # skip "don't care" lines that have transcription '###' at index 8
                if len(parts) > 8 and parts[8].strip() == '###':
                    continue
                try:
                    coords = [int(float(x)) for x in parts[:8]]
                    poly = [[coords[0],coords[1]],[coords[2],coords[3]],[coords[4],coords[5]],[coords[6],coords[7]]]
                    polys.append(poly)
                except Exception:
                    continue
    return polys

def evaluate_dict(pred_dict, gt_dir, iou_thresh=IOU_EVAL_THRESH):
    total_gt = total_det = total_tp = 0
    for stem, preds in pred_dict.items():
        gt_path = gt_dir / f"gt_{stem}.txt"
        gt_polys = load_icdar_gt(gt_path)
        total_gt += len(gt_polys)
        total_det += len(preds)
        matched = set()
        for poly, score in preds:
            for i, gt in enumerate(gt_polys):
                if i in matched:
                    continue
                iou = polygon_iou(poly, gt)
                if iou >= iou_thresh:
                    total_tp += 1
                    matched.add(i)
                    break
    precision = total_tp / total_det if total_det > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return {'precision':precision, 'recall':recall, 'f1':f1, 'tp':total_tp, 'det':total_det, 'gt':total_gt}

print("Evaluating...")
east_metrics = evaluate_dict(east_results, GT_DIR)
craft_metrics = evaluate_dict(craft_results, GT_DIR)
fused_metrics = evaluate_dict(fused_results, GT_DIR)

def print_metrics(name, m):
    print(f"{name:10s} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | TP: {m['tp']} | Det: {m['det']} | GT: {m['gt']}")

print("=== EVALUATION (IoU >= 0.5) ===")
print_metrics('EAST', east_metrics)
print_metrics('CRAFT', craft_metrics)
print_metrics('FUSED', fused_metrics)

