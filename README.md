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