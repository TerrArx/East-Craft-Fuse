#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import math
import shapely
from shapely.geometry import Polygon

MODEL_PATH = 'models/frozen_east_text_detection.pb'
TEST_IMAGE_DIR = 'test-images/ch4_test_images'

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 640
INPUT_HEIGHT = 640


# In[9]:


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
else:
    try:
        net = cv2.dnn.readNet(MODEL_PATH)
        print(f"EAST model loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")


# In[10]:


def decode_rotated_predictions(scores, geometry, confThresh):
    (numRows, numCols) = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y] # d_top
        xData1 = geometry[0, 1, y] # d_right
        xData2 = geometry[0, 2, y] # d_bottom
        xData3 = geometry[0, 3, y] # d_left
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            score = scoresData[x]
            if score < confThresh:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Calculate center of the rotated box
            offset = ([offsetX + cos * xData1[x] + sin * xData2[x],
                       offsetY - sin * xData1[x] + cos * xData2[x]])
            p1 = (-sin * h + offset[0], -cos * h + offset[1])
            p3 = (-cos * w + offset[0], sin * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))

             # Note: EAST outputs angle in radians, NMSBoxesRotated usually expects degrees
            boxes.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    return (boxes, confidences)


# In[11]:


image_paths = glob.glob(os.path.join(TEST_IMAGE_DIR, '*'))
# Filter generic image types
image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

if not image_paths:
    print(f"WARNING: No images found in {TEST_IMAGE_DIR}. Please update TEST_IMAGE_DIR.")
else:
    print(f"Found {len(image_paths)} images. Starting polygon inference...\n")

    for i, img_path in enumerate(image_paths[:10]):
        image = cv2.imread(img_path)
        orig = image.copy()
        (H, W) = image.shape[:2]
        rW = W / float(INPUT_WIDTH)
        rH = H / float(INPUT_HEIGHT)

        image_resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        blob = cv2.dnn.blobFromImage(image_resized, 1.0, (INPUT_WIDTH, INPUT_HEIGHT),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        (boxes, confidences) = decode_rotated_predictions(scores, geometry, CONF_THRESHOLD)
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        if len(indices) > 0:
            for i_box in indices.flatten():
               
                rot_rect = boxes[i_box]
                box_points = cv2.boxPoints(rot_rect)

                for j in range(4):
                    box_points[j][0] *= rW
                    box_points[j][1] *= rH

                box_points = np.int0(box_points)
                cv2.drawContours(orig, [box_points], 0, (0, 255, 0), 2)

        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title(f"Polygon Result: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()


# In[12]:


ICDAR_IMG_DIR = '/kaggle/input/test-images-icdar/ch4_test_images'
ICDAR_GT_DIR = '/kaggle/input/test-images-icdar/Challenge4_Test_Task1_GT'

GROUND_TRUTH_DB = {}

print("Cell 6: Loading ICDAR 2015 GT...")

gt_files = glob.glob(os.path.join(ICDAR_GT_DIR, '*.txt'))

for gt_file_path in gt_files:
    filename = os.path.basename(gt_file_path)
    img_name = filename[3:].replace('.txt', '.jpg')

    img_polys = []
    try:
        with open(gt_file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    coords = [int(x) for x in parts[:8]]
                    poly = np.array(coords).reshape((4, 2)).tolist()
                    img_polys.append(poly)
        GROUND_TRUTH_DB[img_name] = img_polys
    except Exception as e:
        print(f"Warning: Could not parse {filename}: {e}")

print(f"Cell 6: Loaded GT for {len(GROUND_TRUTH_DB)} images.")


# In[13]:


def calculate_iou_polygon(poly_a, poly_b):
    """ Calculate Intersection over Union for two 4-point polygons """
    try:
        a = Polygon(poly_a)
        b = Polygon(poly_b)
        if not a.is_valid or not b.is_valid: return 0.0
        
        intersection_area = a.intersection(b).area
        union_area = a.union(b).area
        
        if union_area == 0: return 0.0
        return intersection_area / union_area
    except Exception:
        return 0.0

def evaluate_single_image(gt_polys, pred_polys, iou_thresh=0.5):
    tp, fp = 0, 0
    matched_gt_indices = set()

    for pred in pred_polys:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_polys):
            if i in matched_gt_indices:
                continue # Already matched this GT
            
            iou = calculate_iou_polygon(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            fp += 1 # No sufficient match found
            
    fn = len(gt_polys) - len(matched_gt_indices)
    return tp, fp, fn

print("Cell 7: Metric functions ready.")


# In[14]:


IOU_THRESH = 0.5
agg_tp, agg_fp, agg_fn = 0, 0, 0
images_processed = 0

print(f"Starting evaluation on {ICDAR_IMG_DIR}...")

img_paths = glob.glob(os.path.join(ICDAR_IMG_DIR, '*'))

for img_path in img_paths:
    img_fn = os.path.basename(img_path)
    
    if img_fn not in GROUND_TRUTH_DB:
        continue

    images_processed += 1
    gt_polys = GROUND_TRUTH_DB[img_fn]
    
    img = cv2.imread(img_path)
    if img is None: continue
    h_orig, w_orig = img.shape[:2]
    rW = w_orig / float(INPUT_WIDTH)
    rH = h_orig / float(INPUT_HEIGHT)
    
    img_resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = cv2.dnn.blobFromImage(img_resized, 1.0, (INPUT_WIDTH, INPUT_HEIGHT), 
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    (boxes, confidences) = decode_rotated_predictions(scores, geometry, CONF_THRESHOLD)
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # --- PREPARE PREDICTIONS ---
    pred_polys = []
    if len(indices) > 0:
        for i in indices.flatten():
            pts = cv2.boxPoints(boxes[i])
            pts[:, 0] *= rW
            pts[:, 1] *= rH
            pred_polys.append(pts.tolist())

    tp, fp, fn = evaluate_single_image(gt_polys, pred_polys, IOU_THRESH)
    agg_tp += tp
    agg_fp += fp
    agg_fn += fn

    if images_processed % 50 == 0:
        print(f"Processed {images_processed} images...")

precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0
recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== FINAL PERFORMANCE METRICS ===")
print(f"Images Evaluated: {images_processed}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=================================")

