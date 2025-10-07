# COCO Analysis Project

This project provides a simple Python tool to analyze COCO-format JSON datasets. It can help you quickly understand your dataset, detect annotation issues, and prepare train/validation/test splits for machine learning tasks.

---

## Features

1. Dataset Summary  
   - Total number of images and annotations  
   - Top categories with annotation counts  

2. Duplicate Detection 
   - Exact duplicate detection (same image, category, bbox)  
   - IoU-based near-duplicates detection (annotations that heavily overlap)  

3. Class Imbalance Report
   - Basic statistics (mean, std, coefficient of variation, max/min ratio)  
   - Simple heuristic to flag imbalanced classes  

4. Train/Validation/Test Split  
   - Greedy split of images into train/val/test sets  
   - Tries to maintain category balance  
   - Outputs split JSON files (`train.json`, `val.json`, `test.json`)  

---

## Folder Structure

**coco_analysis/**
1.scripts/
   coco_summary.py
2.data/
   mock_coco.json
3.output/ (created after running splits)
4.README.md
5.requirements.txt
6.venv/

---

## Installation

1. Create a virtual environment:
python -m venv venv

2. Activate the environment:

Windows (PowerShell):

.\venv\Scripts\Activate.ps1

3. Install required packages:

pip install -r requirements.txt

---

## Usage
Run the script with different options:

1. Summary only:

python scripts/coco_summary.py --json data/mock_coco.json --summary

2. Detect duplicates:

python scripts/coco_summary.py --json data/mock_coco.json --duplicates

3. Class imbalance report:

python scripts/coco_summary.py --json data/mock_coco.json --imbalance

4. Train/Val/Test split:

python scripts/coco_summary.py --json data/mock_coco.json --split --out_dir output

5. Run everything at once:

python scripts/coco_summary.py --json data/mock_coco.json --summary --duplicates --imbalance --split --out_dir output

---

## Expected Output
Terminal prints summary, duplicates, and imbalance statistics
output/ folder containing:

train.json
val.json
test.json

---

## Notes
1. Bounding boxes are in COCO format: [x, y, width, height]
2. IDs are remapped in split JSONs to avoid conflicts
3. The split is greedy, works well for small/medium datasets

---

## Dependencies
1. Python 3.x
2. pandas >= 1.0
3. matplotlib >= 3.0 (optional, only if plotting is added later)

---

Created By - 
Hari Maheshwari

Github Link - https://github.com/harrym14/coco_analysis
