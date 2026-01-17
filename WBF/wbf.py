"""
Weighted Box Fusion (WBF) Ensemble for Scene Text Detection
Combines EAST and CRAFT detections using ensemble_boxes WBF algorithm
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon
from ensemble_boxes import weighted_boxes_fusion

# ================= PATH CONFIGURATION =================
BASE_DIR = os.path.dirname(os.getcwd())  # Go up one level from your notebook folder

ICDAR_IMAGE_DIR = os.path.join(BASE_DIR, "test-images", "ch4_test_images")
GT_DIR = os.path.join(BASE_DIR, "test-images", "Challenge4_Test_Task1_GT")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

EAST_DIR  = os.path.join(OUTPUT_DIR, "east")
CRAFT_DIR = os.path.join(OUTPUT_DIR, "craft")
FUSED_DIR = os.path.join(OUTPUT_DIR, "fused")

os.makedirs(FUSED_DIR, exist_ok=True)

print(f"Images: {ICDAR_IMAGE_DIR}")
print(f"Ground Truth: {GT_DIR}")
print(f"EAST predictions: {EAST_DIR}")
print(f"CRAFT predictions: {CRAFT_DIR}")
print(f"Fused output: {FUSED_DIR}")


# ================= UTILITY FUNCTIONS =================
def load_icdar_file(file_path, img_width, img_height):
    """
    Reads an ICDAR text file (x1,y1...x4,y4, score).
    Returns: normalized boxes [[x1, y1, x2, y2]], scores [s1], polygons (original coords) for WBF.
    """
    boxes = []
    scores = []
    polygons = []
    
    if not os.path.exists(file_path):
        return [], [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(',')
        try:
            coords = [float(x) for x in parts[:8]]
            
            if len(parts) > 8:
                score = float(parts[8])
            else:
                score = 1.0
            
            # Convert 8-point polygon to bounding box with normalized coordinates
            xs = coords[0::2]
            ys = coords[1::2]
            
            xmin = max(0, min(xs) / img_width)
            ymin = max(0, min(ys) / img_height)
            xmax = min(1, max(xs) / img_width)
            ymax = min(1, max(ys) / img_height)
            
            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(score)
            polygons.append(coords)
        except ValueError:
            continue
            
    return boxes, scores, polygons


def save_fused_file_with_polygons(file_path, polygons, scores):
    """
    Saves WBF output using original polygon shapes.
    """
    with open(file_path, 'w') as f:
        for poly, score in zip(polygons, scores):
            coords = [str(int(c)) for c in poly]
            line = ','.join(coords) + f',{score:.4f}\n'
            f.write(line)


# ================= WBF FUSION PROCESS =================
def run_fusion():
    """Run Weighted Box Fusion on all images in the dataset."""
    # Fusion parameters
    MODEL_WEIGHTS = [1, 1]
    IOU_THR = 0.45
    SKIP_BOX_THR = 0.30
    REQUIRE_AGREEMENT = True
    AGREEMENT_IOU_THRESH = 0.35

    image_files = glob.glob(os.path.join(ICDAR_IMAGE_DIR, '*'))
    print(f"Starting fusion on {len(image_files)} files...")

    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        txt_name = f"{name_no_ext}.txt"
        
        east_path = os.path.join(EAST_DIR, txt_name)
        craft_path = os.path.join(CRAFT_DIR, txt_name)
        save_path = os.path.join(FUSED_DIR, txt_name)
        
        img = cv2.imread(img_path)
        if img is None: 
            continue
        h, w, _ = img.shape
        
        # Load predictions with original polygons
        boxes_list, scores_list, labels_list = [], [], []
        all_polygons = []
        
        # Load EAST predictions
        b1, s1, p1 = load_icdar_file(east_path, w, h)
        if b1:
            boxes_list.append(b1)
            scores_list.append(s1)
            labels_list.append([0] * len(b1))
            all_polygons.extend([(poly, 'east', i) for i, poly in enumerate(p1)])
            
        # Load CRAFT predictions
        b2, s2, p2 = load_icdar_file(craft_path, w, h)
        if b2:
            boxes_list.append(b2)
            scores_list.append(s2)
            labels_list.append([0] * len(b2))
            all_polygons.extend([(poly, 'craft', i) for i, poly in enumerate(p2)])
        
        if not boxes_list:
            with open(save_path, 'w') as f: 
                pass
            continue

        # Apply weighted box fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=MODEL_WEIGHTS[:len(boxes_list)],
            iou_thr=IOU_THR,
            skip_box_thr=SKIP_BOX_THR
        )
        
        # Agreement-based filtering
        if REQUIRE_AGREEMENT and len(boxes_list) == 2:
            filtered_boxes = []
            filtered_scores = []
            filtered_indices = []
            
            for idx, (fused_box, fused_score) in enumerate(zip(fused_boxes, fused_scores)):
                fx1, fy1, fx2, fy2 = fused_box
                
                east_overlap = False
                for b in b1:
                    iou = max(0, min(fx2, b[2]) - max(fx1, b[0])) * max(0, min(fy2, b[3]) - max(fy1, b[1]))
                    union = (fx2 - fx1) * (fy2 - fy1) + (b[2] - b[0]) * (b[3] - b[1]) - iou
                    if iou / (union + 1e-6) > AGREEMENT_IOU_THRESH:
                        east_overlap = True
                        break
                
                craft_overlap = False
                for b in b2:
                    iou = max(0, min(fx2, b[2]) - max(fx1, b[0])) * max(0, min(fy2, b[3]) - max(fy1, b[1]))
                    union = (fx2 - fx1) * (fy2 - fy1) + (b[2] - b[0]) * (b[3] - b[1]) - iou
                    if iou / (union + 1e-6) > AGREEMENT_IOU_THRESH:
                        craft_overlap = True
                        break
                
                if east_overlap and craft_overlap:
                    filtered_boxes.append(fused_box)
                    filtered_scores.append(fused_score)
                    filtered_indices.append(idx)
            
            fused_boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])
            fused_scores = np.array(filtered_scores) if filtered_scores else np.array([])
        
        # Map fused boxes to original polygons
        fused_polygons = []
        
        for fused_box, fused_score in zip(fused_boxes, fused_scores):
            fx1 = fused_box[0] * w
            fy1 = fused_box[1] * h
            fx2 = fused_box[2] * w
            fy2 = fused_box[3] * h
            fused_center_x = (fx1 + fx2) / 2
            fused_center_y = (fy1 + fy2) / 2
            
            best_poly = None
            best_dist = float('inf')
            
            for poly, source, idx in all_polygons:
                xs = [poly[i] for i in range(0, 8, 2)]
                ys = [poly[i] for i in range(1, 8, 2)]
                poly_center_x = sum(xs) / 4
                poly_center_y = sum(ys) / 4
                
                dist = ((fused_center_x - poly_center_x)**2 + (fused_center_y - poly_center_y)**2)**0.5
                
                poly_xmin, poly_xmax = min(xs), max(xs)
                poly_ymin, poly_ymax = min(ys), max(ys)
                
                xi1 = max(fx1, poly_xmin)
                yi1 = max(fy1, poly_ymin)
                xi2 = min(fx2, poly_xmax)
                yi2 = min(fy2, poly_ymax)
                
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (fx2 - fx1) * (fy2 - fy1)
                box2_area = (poly_xmax - poly_xmin) * (poly_ymax - poly_ymin)
                iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
                
                if iou > 0.3 and dist < best_dist:
                    best_dist = dist
                    best_poly = poly
            
            if best_poly is not None:
                fused_polygons.append(best_poly)
            else:
                fused_polygons.append([
                    int(fx1), int(fy1),
                    int(fx2), int(fy1),
                    int(fx2), int(fy2),
                    int(fx1), int(fy2)
                ])
        
        save_fused_file_with_polygons(save_path, fused_polygons, fused_scores)

    print("Fusion complete.")


# ================= VISUALIZATION =================
def draw_boxes(image, txt_path, color=(0, 255, 0), thickness=2):
    """Parses ICDAR txt file and draws boxes on the image."""
    canvas = image.copy()
    if not os.path.exists(txt_path): 
        return canvas
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                coords = [int(float(x)) for x in parts[:8]]
                pts = np.array(coords).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
            except: 
                continue
    return canvas


def visualize_comparisons(num_samples=10):
    """Generate comparison visualizations for sample images."""
    # Select first N images
    all_images = sorted(glob.glob(os.path.join(ICDAR_IMAGE_DIR, '*')))
    sample_images = all_images[:num_samples]

    # Process each image comparison separately
    for img_idx, img_path in enumerate(sample_images, 1):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        txt_name = f"{name_no_ext}.txt"

        img = cv2.imread(img_path)
        if img is None: 
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        east_txt = os.path.join(EAST_DIR, txt_name)
        craft_txt = os.path.join(CRAFT_DIR, txt_name)
        fused_txt = os.path.join(FUSED_DIR, txt_name)

        # Draw Boxes
        img_east = draw_boxes(img, east_txt, color=(255, 0, 0))   # Red
        img_craft = draw_boxes(img, craft_txt, color=(0, 0, 255)) # Blue
        img_wbf = draw_boxes(img, fused_txt, color=(0, 255, 0))   # Green

        # Create individual comparison figure for each image
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Image {img_idx}: {filename}", fontsize=14, fontweight='bold')

        axes[0].imshow(img_east)
        axes[0].set_title("EAST (Red)", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(img_craft)
        axes[1].set_title("CRAFT (Blue)", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(img_wbf)
        axes[2].set_title("WBF Ensemble (Green)", fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    print(f"\nDisplayed {len(sample_images)} image comparisons")


# ================= EVALUATION =================
def get_polygon_iou(poly1_coords, poly2_coords):
    """Calculates IoU between two 8-point polygons."""
    try:
        p1 = Polygon(np.array(poly1_coords).reshape(4, 2))
        p2 = Polygon(np.array(poly2_coords).reshape(4, 2))
        if not p1.is_valid or not p2.is_valid: 
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / union if union > 0 else 0.0
    except: 
        return 0.0


def load_coords_with_labels(txt_path):
    """
    Load coordinates from ICDAR GT file, separating care and don't-care regions.
    Returns: care_boxes (list), dontcare_boxes (list)
    """
    care_boxes = []
    dontcare_boxes = []
    
    if not os.path.exists(txt_path):
        return care_boxes, dontcare_boxes
    
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                coords = [int(float(x)) for x in parts[:8]]
                
                if len(parts) > 8:
                    transcription = ','.join(parts[8:]).strip().strip('"')
                    if transcription == "###":
                        dontcare_boxes.append(coords)
                    else:
                        care_boxes.append(coords)
                else:
                    care_boxes.append(coords)
            except:
                continue
    
    return care_boxes, dontcare_boxes


def load_pred_coords(txt_path):
    """Load prediction coordinates."""
    coords_list = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                try:
                    c = [int(float(x)) for x in parts[:8]]
                    coords_list.append(c)
                except: 
                    continue
    return coords_list


def evaluate_folder(pred_dir, gt_dir, iou_thresh=0.5):
    """
    Evaluate predictions following ICDAR-15 protocol:
    - Only count 'care' GT boxes (exclude "###" don't-care regions)
    - Predictions overlapping don't-care regions are ignored (not FP)
    - IoU threshold: 0.5
    """
    tp, fp, fn = 0, 0, 0
    gt_files = glob.glob(os.path.join(gt_dir, 'gt_*.txt'))
    
    for gt_path in tqdm(gt_files, desc=f"Eval {os.path.basename(pred_dir)}", leave=False):
        filename = os.path.basename(gt_path)
        pred_filename = filename[3:]
        pred_path = os.path.join(pred_dir, pred_filename)
        
        care_boxes, dontcare_boxes = load_coords_with_labels(gt_path)
        pred_boxes = load_pred_coords(pred_path)
        
        matched_gt = [False] * len(care_boxes)
        
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for g_idx, g_box in enumerate(care_boxes):
                if matched_gt[g_idx]: 
                    continue
                iou = get_polygon_iou(p_box, g_box)
                if iou > iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_gt_idx >= 0:
                matched_gt[best_gt_idx] = True
                tp += 1
                continue
            
            is_dontcare = False
            for dc_box in dontcare_boxes:
                iou_dc = get_polygon_iou(p_box, dc_box)
                if iou_dc > iou_thresh:
                    is_dontcare = True
                    break
            
            if is_dontcare:
                continue
            
            fp += 1
        
        fn += matched_gt.count(False)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": tp, 
        "FP": fp, 
        "FN": fn, 
        "Precision": precision, 
        "Recall": recall, 
        "F1-Score": f1
    }


def run_evaluation():
    """Run ICDAR-15 evaluation on EAST, CRAFT, and WBF predictions."""
    print("\nEvaluating with ICDAR-15 Protocol...")
    results = []
    results.append({"Model": "EAST", **evaluate_folder(EAST_DIR, GT_DIR)})
    results.append({"Model": "CRAFT", **evaluate_folder(CRAFT_DIR, GT_DIR)})
    results.append({"Model": "WBF (Ensemble)", **evaluate_folder(FUSED_DIR, GT_DIR)})

    df = pd.DataFrame(results)
    print("\nPerformance Comparison Table")
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df


# ================= MAIN EXECUTION =================
def main():
    """Main execution function."""
    print("=" * 60)
    print("WBF Ensemble for Scene Text Detection")
    print("=" * 60)
    
    # Step 1: Run fusion
    print("\n[Step 1/3] Running Weighted Box Fusion...")
    run_fusion()
    
    # Step 2: Visualize results
    print("\n[Step 2/3] Generating visualizations...")
    visualize_comparisons(num_samples=10)
    
    # Step 3: Run evaluation
    print("\n[Step 3/3] Running ICDAR-15 evaluation...")
    results_df = run_evaluation()
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    main()
