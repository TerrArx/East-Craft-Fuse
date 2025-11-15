#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from craft_text_detector import Craft
from shapely.geometry import Polygon
from tqdm.auto import tqdm
import time
import os
import glob
import warnings
import sys
import torchvision.models.vgg as vgg_module

warnings.filterwarnings('ignore')

import craft_text_detector.craft_utils as craft_utils

original_adjustResultCoordinates = craft_utils.adjustResultCoordinates

def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """Patched version that handles numpy array creation properly"""
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates

_original_np_array = np.array

def _patched_np_array(arr_input, dtype=None, **kwargs):
    """Patched numpy array that uses object dtype for heterogeneous sequences"""
    try:
        return _original_np_array(arr_input, dtype=dtype, **kwargs)
    except ValueError as e:
        error_msg = str(e)
        if "inhomogeneous shape" in error_msg or "setting an array element with a sequence" in error_msg:
            return _original_np_array(arr_input, dtype=object)
        raise

np.array = _patched_np_array
print(" Comprehensive CRAFT numpy compatibility patches applied")

if not hasattr(vgg_module, 'model_urls'):
    print("Applying torchvision.models.vgg.model_urls patch...")
    vgg_module.model_urls = {
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    }
print("✅ Torchvision patch applied")


# In[12]:


CRAFT_MODEL_PATH = 'models/craft_ic15_20k.pth' 
IMAGES_DIR = Path('test-images/ch4_test_images')
GT_DIR = Path('test-images/Challenge4_Test_Task1_GT')
OUTPUT_DIR = Path('CRAFT/outputs')
VIZ_DIR = Path('East-Craft/visualizations')
VIS_LIMIT = 10  # Max number of images to visualize

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"CRAFT Model: {CRAFT_MODEL_PATH}")
print(f"Images Dir:  {IMAGES_DIR}")
print(f"GT Dir:      {GT_DIR}")
print(f"Output Dir:  {OUTPUT_DIR}")
print(f"Viz Dir:     {VIZ_DIR}")


# In[13]:


def load_craft_model(model_path, use_cuda=False):
    """Load CRAFT model using craft-text-detector"""
    print("Loading CRAFT model...")
    
    craft = Craft(
        output_dir=None,
        crop_type="poly", 
        cuda=use_cuda,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        export_extra=False
    )
    
    if Path(model_path).exists():
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value  
            else:
                new_state_dict[key] = value
        
        craft.craft_net.load_state_dict(new_state_dict)
        print(f"✅ CRAFT model loaded successfully from {model_path}")
    else:
        print(f"⚠️ Warning: Model file not found at {model_path}. Using default weights.")
    
    craft.craft_net.eval()
    return craft

craft_detector = load_craft_model(CRAFT_MODEL_PATH, use_cuda=torch.cuda.is_available())


# In[14]:


def save_detections_for_eval(detections, output_path):
    """Save detections to file in format: x1,y1,x2,y2,x3,y3,x4,y4,confidence"""
    with open(output_path, 'w') as f:
        for polygon, conf in detections:
            coords = ','.join([str(int(c)) for point in polygon for c in point])
            f.write(f"{coords},{conf:.4f}\n")

def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """Draw detections on image"""
    img_copy = image.copy()
    for polygon, conf in detections:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
    return img_copy

print("✅ Helper functions ready")


# In[15]:


print("Running CRAFT inference on all images...")
all_results = {}
image_files = sorted(IMAGES_DIR.glob('*.jpg')) + sorted(IMAGES_DIR.glob('*.png'))

start_time = time.time()
for idx, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
    try:
        prediction_result = craft_detector.detect_text(str(img_path))
        
        detections = []
        if 'boxes' in prediction_result:
            scores = prediction_result.get('scores', [0.9] * len(prediction_result['boxes']))
            for box, score in zip(prediction_result['boxes'], scores):
                polygon = np.array(box).astype(int)
                detections.append((polygon, score))
        
        all_results[img_path.stem] = detections
        
        output_path = OUTPUT_DIR / f"{img_path.stem}.txt"
        save_detections_for_eval(detections, output_path)

        if idx < VIS_LIMIT:
            image = cv2.imread(str(img_path))
            img_viz = draw_detections(image, detections)
            viz_path = VIZ_DIR / f"{img_path.stem}_viz.png"
            cv2.imwrite(str(viz_path), img_viz)

    except Exception as e:
        tqdm.write(f"❌ Error processing {img_path.name}: {e}")
        all_results[img_path.stem] = []

end_time = time.time()
print(f"\n✅ Inference complete: {len(all_results)} images processed in {end_time - start_time:.2f}s")
print(f"Text outputs saved to: {OUTPUT_DIR}")
print(f"Visualizations saved to: {VIZ_DIR}")


# In[16]:


def polygon_iou(poly1, poly2):
    """Calculate IoU between two polygons using Shapely"""
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        
        if not p1.is_valid: p1 = p1.buffer(0)
        if not p2.is_valid: p2 = p2.buffer(0)
        
        intersection = p1.intersection(p2).area
        union = p1.union(p2).area
        
        return intersection / union if union != 0 else 0.0
    except:
        return 0.0

def load_ground_truth(gt_path):
    """Load ground truth annotations"""
    polygons = []
    if not gt_path.exists():
        return polygons
    
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                try:
                    if len(parts) > 8 and parts[8].strip() == '###':
                        continue
                    
                    coords = [int(float(x)) for x in parts[:8]]
                    polygon = np.array([
                        [coords[0], coords[1]], [coords[2], coords[3]],
                        [coords[4], coords[5]], [coords[6], coords[7]]
                    ])
                    polygons.append(polygon)
                except ValueError:
                    continue
    return polygons

def evaluate_detections(detections_dict, gt_dir, iou_threshold=0.5):
    """Evaluate detections against ground truth"""
    total_gt, total_det, total_tp = 0, 0, 0
    
    for img_stem, detections in detections_dict.items():
        gt_path = gt_dir / f"gt_{img_stem}.txt"
        gt_polygons = load_ground_truth(gt_path)
        
        total_gt += len(gt_polygons)
        total_det += len(detections)
        
        matched_gt = set()
        for det_poly, det_conf in detections:
            for gt_idx, gt_poly in enumerate(gt_polygons):
                if gt_idx in matched_gt:
                    continue
                
                iou = polygon_iou(det_poly, gt_poly)
                if iou >= iou_threshold:
                    matched_gt.add(gt_idx)
                    total_tp += 1
                    break
    
    precision = total_tp / total_det if total_det > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': total_tp,
        'total_detections': total_det,
        'total_ground_truth': total_gt
    }

print("✅ Evaluation functions ready")


# In[17]:


print("\n" + "="*80)
print("EVALUATING IMPROVED CRAFT IMPLEMENTATION")
print("="*80)

# Evaluate the new CRAFT results
craft_metrics = evaluate_detections(all_results, GT_DIR)

# Print results table
print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Detections':<12}")
print("-"*80)
print(f"{'Improved CRAFT':<20} {craft_metrics['precision']:<12.4f} {craft_metrics['recall']:<12.4f} "
      f"{craft_metrics['f1_score']:<12.4f} {craft_metrics['total_detections']:<12}")
print("-"*80)
print(f"Total Ground Truth Instances: {craft_metrics['total_ground_truth']}")
print("="*80)



