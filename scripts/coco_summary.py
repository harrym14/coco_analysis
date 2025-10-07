#!/usr/bin/env python3
"""
coco_summary.py

Command-line tool to analyze a COCO-format JSON file.

Features:
- Print total images, total annotations, top categories.
- Detect duplicate annotations (exact + IoU-based near-duplicates).
- Class imbalance statistics.
- Greedy approximate train/val/test split maintaining category balance.
- Optional writing of split COCO JSON files.

This script helps:
1. Understand your dataset quickly.
2. Detect common annotation issues.
3. Split data for ML pipelines in a reproducible way.
"""

# -------------------------
# Import required libraries
# -------------------------
import json  # to read and write JSON files
import os  # to handle file paths and directories
import argparse  # to parse command-line arguments
import random  # for shuffling and random operations
from collections import Counter, defaultdict  # for counting and grouping
import statistics  # for basic stats (mean, std, etc.)
import copy  # for deep copying objects

# Fix the random seed for reproducibility of splits
random.seed(42)

# -------------------------
# Utility functions
# -------------------------

def load_coco(json_path):
    """
    Load a COCO JSON file from the given path.
    
    Ensures that 'images', 'annotations', and 'categories' exist
    in the loaded JSON even if they were missing to prevent errors later.
    """
    with open(json_path, "r") as f:
        coco = json.load(f)  # read JSON file into Python dictionary
    
    # Ensure essential keys exist to avoid KeyErrors
    coco.setdefault("images", [])
    coco.setdefault("annotations", [])
    coco.setdefault("categories", [])
    
    return coco  # return the loaded COCO dictionary

def build_cat_map(coco):
    """
    Build a mapping from category ID to category name.
    This allows us to print category names instead of just IDs.
    
    If categories are missing in the JSON, assign placeholder names like 'category_1'.
    """
    cat_map = {}  # empty dictionary to store mapping
    
    if coco.get("categories"):  # if 'categories' key exists
        for c in coco["categories"]:
            cat_map[c["id"]] = c.get("name", f"category_{c['id']}")
    else:  # fallback: infer category IDs from annotations
        for ann in coco["annotations"]:
            cid = ann["category_id"]
            if cid not in cat_map:
                cat_map[cid] = f"category_{cid}"
    
    return cat_map  # return dictionary mapping category_id -> name

# -------------------------
# Summary and class imbalance
# -------------------------

def print_summary(coco, top_k=3):
    """
    Print summary of the dataset:
    - Total images
    - Total annotations
    - Top K categories by annotation count
    
    Returns:
    - cat_counts: Counter of annotations per category
    - cat_map: dictionary mapping category IDs to names
    """
    total_images = len(coco["images"])  # count total images
    total_annotations = len(coco["annotations"])  # count total annotations
    
    # Count annotations per category
    cat_counts = Counter([ann["category_id"] for ann in coco["annotations"]])
    
    # Build category ID -> name mapping
    cat_map = build_cat_map(coco)
    
    # Get top K categories
    topk = cat_counts.most_common(top_k)
    
    # Print summary
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Top {top_k} categories:")
    for cid, cnt in topk:
        print(f"  - {cat_map.get(cid, cid)} (id={cid}): {cnt} annotations")
    
    return cat_counts, cat_map  # return counts and mapping for further use

def class_imbalance_report(cat_counts):
    """
    Check for class imbalance in dataset.
    
    Calculates:
    - Number of categories
    - Total annotations
    - Mean annotations per category
    - Standard deviation
    - Coefficient of variation
    - Max/min ratio
    - Flags dataset as imbalanced if variation is high
    """
    counts = list(cat_counts.values())
    total = sum(counts)
    ncat = len(counts)
    mean = statistics.mean(counts) if counts else 0
    std = statistics.pstdev(counts) if counts else 0
    coef_var = std / mean if mean else float("inf")  # high variation indicates imbalance
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    
    # Heuristic for imbalance
    imbalance_flag = coef_var > 1 or (min_count > 0 and (max_count / min_count) > 2)
    
    # Print detailed report
    print("\nClass imbalance summary:")
    print(f"  - number of categories: {ncat}")
    print(f"  - total annotation count: {total}")
    print(f"  - mean annotations per category: {mean:.2f}")
    print(f"  - std (population): {std:.2f}")
    print(f"  - coefficient of variation: {coef_var:.2f}")
    print(f"  - max/min ratio: {max_count}/{min_count} = {max_count/min_count if min_count else 'inf'}")
    print(f"  - flagged as imbalanced: {imbalance_flag}")
    
    # Return stats in a dictionary for potential programmatic use
    return {
        "ncat": ncat,
        "total": total,
        "mean": mean,
        "std": std,
        "coef_var": coef_var,
        "max": max_count,
        "min": min_count,
        "imbalanced": imbalance_flag,
    }

# -------------------------
# Duplicate detection
# -------------------------

def detect_exact_duplicates(coco):
    """
    Detect exact duplicate annotations:
    Same image, same category, and same bounding box.
    """
    key_counts = Counter()  # to count identical annotation tuples
    
    for ann in coco["annotations"]:
        key = (ann["image_id"], ann["category_id"], tuple(ann["bbox"]))
        key_counts[key] += 1  # increment count for this annotation tuple
    
    # Keep only duplicates (count > 1)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    
    # Print summary
    print(f"\nExact duplicate annotation groups found: {len(duplicates)}")
    if duplicates:
        print("Examples (image_id, category_id, bbox) -> count:")
        for (imgid, cid, bbox), cnt in list(duplicates.items())[:5]:
            print(f"  - ({imgid}, {cid}, {bbox}) -> {cnt}")
    
    return duplicates  # return dictionary of duplicates

def iou_bbox(b1, b2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    
    Bounding boxes are in COCO format: [x, y, width, height]
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    
    # Convert to (x1, y1, x2, y2) for easy intersection calculation
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    
    # Compute intersection rectangle
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Compute union area
    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter_area
    
    if union == 0:
        return 0.0  # avoid division by zero
    
    return inter_area / union

def detect_iou_duplicates(coco, iou_threshold=0.95):
    """
    Detect near-duplicate annotations using IoU.
    Only considers annotations in the same image and category.
    """
    anns_by_image = defaultdict(list)  # group annotations by image
    
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    
    near_dupes = []  # store pairs of near-duplicate annotations
    
    for img_id, anns in anns_by_image.items():
        n = len(anns)
        for i in range(n):
            for j in range(i + 1, n):
                a = anns[i]
                b = anns[j]
                if a["category_id"] != b["category_id"]:
                    continue  # skip different categories
                iou = iou_bbox(a["bbox"], b["bbox"])
                if iou >= iou_threshold:
                    near_dupes.append((img_id, a["id"], b["id"], a["category_id"], iou))
    
    print(f"\nIoU-based near-duplicate pairs (IoU>={iou_threshold}): {len(near_dupes)}")
    if near_dupes:
        print("Examples (image_id, ann_id1, ann_id2, category_id, iou):")
        for p in near_dupes[:5]:
            print(f"  - {p}")
    
    return near_dupes
# -------------------------
# Train/Validation/Test split
# -------------------------

def greedy_image_split(coco, train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=42):
    """
    Greedy split of images into train/val/test sets while trying to preserve
    category balance.

    Steps:
    1. Shuffle images randomly using seed for reproducibility.
    2. Count annotations per category per image.
    3. Assign images one by one to the split that benefits category balance the most.
    """
    random.seed(seed)  # ensure reproducible shuffling

    # Build dictionary: image_id -> list of annotations
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    # List of all images
    images = coco["images"]
    img_ids = [img["id"] for img in images]
    random.shuffle(img_ids)  # randomize order

    # Count categories per image
    img_cat = {iid: Counter([a["category_id"] for a in ann_by_image.get(iid, [])]) for iid in img_ids}

    # Total annotations per category
    total_by_cat = Counter([a["category_id"] for a in coco["annotations"]])

    # Desired count per split per category
    desired = {
        "train": {cid: int(cnt * train_frac) for cid, cnt in total_by_cat.items()},
        "val": {cid: int(cnt * val_frac) for cid, cnt in total_by_cat.items()},
        "test": {cid: int(cnt * test_frac) for cid, cnt in total_by_cat.items()},
    }

    # Initialize assigned image sets
    assigned = {"train": set(), "val": set(), "test": set()}
    # Track current category counts per split
    current = {"train": Counter(), "val": Counter(), "test": Counter()}

    # Assign images one by one
    for iid in img_ids:
        counts = img_cat.get(iid, Counter())  # categories in this image
        best_split = None
        best_score = -1

        # Evaluate which split benefits most from this image
        for split in ["train", "val", "test"]:
            score = 0
            for cid, num in counts.items():
                needed = max(0, desired[split].get(cid, 0) - current[split].get(cid, 0))
                score += min(num, needed)
            if score == 0:
                score = -len(assigned[split]) * 0.001  # small penalty to avoid empty split
            if score > best_score:
                best_score = score
                best_split = split

        # Assign image to best split
        assigned[best_split].add(iid)
        current[best_split].update(counts)

    return assigned, current  # return image sets and category counts

def write_coco_subset(coco, image_id_set, out_path):
    """
    Write a COCO JSON file containing only images in 'image_id_set'.
    Remaps image and annotation IDs to avoid conflicts.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # create folder if missing

    # Filter images to include only selected IDs
    images = [img for img in coco["images"] if img["id"] in image_id_set]

    # Remap image IDs
    old_to_new = {}
    for new_id, img in enumerate(images, start=1):
        old_to_new[img["id"]] = new_id
        img["id"] = new_id

    # Filter and remap annotations
    annotations = [copy.deepcopy(ann) for ann in coco["annotations"] if ann["image_id"] in image_id_set]
    for new_ann_id, ann in enumerate(annotations, start=1):
        ann["id"] = new_ann_id
        ann["image_id"] = old_to_new[ann["image_id"]]

    # Use original categories if present, otherwise infer from annotations
    categories = coco["categories"] if coco.get("categories") else []
    if not categories:
        cat_ids = sorted({ann["category_id"] for ann in annotations})
        categories = [{"id": cid, "name": f"category_{cid}"} for cid in cat_ids]

    # Prepare final COCO dict
    out = {
        "info": coco.get("info", {}),
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Write to JSON file
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)  # indent for readability

    print(f"Wrote subset COCO to {out_path} (images={len(images)}, annotations={len(annotations)})")
    return out_path

# -------------------------
# Command-line interface
# -------------------------

def main():
    """
    Main entry point for command-line usage.
    Parses arguments and runs requested operations.
    """
    parser = argparse.ArgumentParser(description="COCO JSON summary & utilities")
    parser.add_argument("--json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--summary", action="store_true", help="Print summary (images, annotations, top categories)")
    parser.add_argument("--duplicates", action="store_true", help="Detect duplicates (exact + IoU-based)")
    parser.add_argument("--iou-threshold", type=float, default=0.95, help="IoU threshold for near-duplicate detection")
    parser.add_argument("--imbalance", action="store_true", help="Print class imbalance stats")
    parser.add_argument("--split", action="store_true", help="Create train/val/test splits and write JSONs")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of images for training set")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of images for validation set")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction of images for test set")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory for split JSONs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()

    # Load COCO JSON file
    coco = load_coco(args.json)

    # Summary
    if args.summary:
        cat_counts, cat_map = print_summary(coco)
    else:
        cat_counts = Counter([ann["category_id"] for ann in coco["annotations"]])
        cat_map = build_cat_map(coco)

    # Duplicates
    if args.duplicates:
        detect_exact_duplicates(coco)
        detect_iou_duplicates(coco, iou_threshold=args.iou_threshold)

    # Imbalance
    if args.imbalance:
        class_imbalance_report(cat_counts)

    # Split and write files
    if args.split:
        assigned, current = greedy_image_split(
            coco,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed
        )

        # Ensure output folder exists
        os.makedirs(args.out_dir, exist_ok=True)

        # Write each subset to JSON
        write_coco_subset(coco, assigned["train"], os.path.join(args.out_dir, "train.json"))
        write_coco_subset(coco, assigned["val"], os.path.join(args.out_dir, "val.json"))
        write_coco_subset(coco, assigned["test"], os.path.join(args.out_dir, "test.json"))

# Entry point
if __name__ == "__main__":
    main()