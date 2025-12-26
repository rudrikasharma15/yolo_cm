"""
🔥 STEP 4: MERGE DATASETS + TRAIN FINAL MODEL - WINDOWS GPU 🔥
================================================================

Merge auto-labeled + manual data, then train final production model.
Includes proper train/val/test split.

Prerequisites:
- Completed Step 2 (initial training)
- Completed Step 3 (auto-labeling)
"""

import os
import json
import shutil
import hashlib
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO
import torch
import cv2

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Inputs
AUTO_LABELED_DIR = r"C:\Users\YourName\Desktop\modi_final\step3_auto_labeled"
MANUAL_LABELS_DIR = r"C:\Users\YourName\Desktop\modi_final\matra_extraction_PERFECT\unlabeled_matras"

# Output
MERGED_OUTPUT = r"C:\Users\YourName\Desktop\modi_final\step4_merged_dataset"
FINAL_MODEL_OUTPUT = r"C:\Users\YourName\Desktop\modi_final\step4_final_model"

# Training parameters
EPOCHS = 150
IMG_SIZE = 640
BATCH_SIZE = 16  # Reduce to 8 if GPU memory issues
MODEL_SIZE = "yolov8m"  # Medium model for final training

# Class mapping
CLASS_MAP = {
    "aa_matra": 0, "aa": 0,
    "i_matra": 1, "i": 1,
    "u_matra": 2, "u": 2,
    "e_matra": 3, "e": 3,
    "ai_matra": 4, "ai": 4,
    "o_matra": 5, "o": 5,
    "au_matra": 6, "au": 6,
    "anusvara": 7
}

CLASS_NAMES = [
    'aa_matra', 'i_matra', 'u_matra', 'e_matra',
    'ai_matra', 'o_matra', 'au_matra', 'anusvara'
]

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       🔥 STEP 4: MERGE + TRAIN FINAL MODEL 🔥                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Check GPU
print(f"🖥️  GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# PART 1: MERGE DATASETS
# ============================================================================

print("\n" + "="*70)
print("📦 PART 1: MERGING AUTO-LABELED + MANUAL DATASETS")
print("="*70)

def get_image_hash(img_path):
    """Calculate MD5 hash to detect duplicates"""
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def convert_json_to_yolo(json_path, img_width, img_height):
    """Convert JSON label to YOLO format"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get("shapes", [])
        if not shapes:
            return None
        
        yolo_lines = []
        for s in shapes:
            label = s["label"].lower().strip().replace("-", "_").replace("_matra", "")
            
            class_id = None
            for key, cid in CLASS_MAP.items():
                if label == key or label in key or key.replace("_matra", "") == label:
                    class_id = cid
                    break
            
            if class_id is None:
                continue
            
            (x1, y1), (x2, y2) = s["points"]
            xc = ((x1 + x2) / 2) / img_width
            yc = ((y1 + y2) / 2) / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height
            
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        return "\n".join(yolo_lines) if yolo_lines else None
    except:
        return None

# Create output directories
merged_dir = Path(MERGED_OUTPUT)
merged_dir.mkdir(parents=True, exist_ok=True)
(merged_dir / "images").mkdir(exist_ok=True)
(merged_dir / "labels").mkdir(exist_ok=True)

# Track statistics
image_hashes = {}
stats = {
    'auto_labeled': 0,
    'manual_labeled': 0,
    'duplicates_removed': 0,
    'total_unique': 0
}

print("\n🤖 Processing auto-labeled data...\n")

# Process auto-labeled data
auto_path = Path(AUTO_LABELED_DIR)
if auto_path.exists():
    all_folders = [f for f in auto_path.iterdir() if f.is_dir()]
    
    print(f"   Found {len(all_folders)} matra folders")
    
    for matra_folder in tqdm(all_folders, desc="   Auto-labeled"):
        img_folder = matra_folder / "images"
        label_folder = matra_folder / "labels"
        
        if not img_folder.exists() or not label_folder.exists():
            continue
        
        images = list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg"))
        
        for img_path in images:
            label_path = label_folder / f"{img_path.stem}.txt"
            
            if not label_path.exists() or label_path.stat().st_size == 0:
                continue
            
            # Check for duplicate
            img_hash = get_image_hash(img_path)
            if img_hash in image_hashes:
                stats['duplicates_removed'] += 1
                continue
            
            # Copy with sequential naming
            img_name = f"{stats['auto_labeled']:06d}.png"
            label_name = f"{stats['auto_labeled']:06d}.txt"
            
            shutil.copy2(img_path, merged_dir / "images" / img_name)
            shutil.copy2(label_path, merged_dir / "labels" / label_name)
            
            image_hashes[img_hash] = img_name
            stats['auto_labeled'] += 1
    
    print(f"\n   ✅ Processed: {stats['auto_labeled']} auto-labeled images")
else:
    print(f"   ⚠️  Auto-labeled directory not found: {AUTO_LABELED_DIR}")

print("\n📝 Processing manual JSON labels...\n")

# Process manual labels
manual_path = Path(MANUAL_LABELS_DIR)
if manual_path.exists():
    all_folders = [f for f in manual_path.iterdir() if f.is_dir()]
    
    print(f"   Found {len(all_folders)} matra folders")
    
    for matra_folder in tqdm(all_folders, desc="   Manual"):
        images = list(matra_folder.glob("*.png")) + list(matra_folder.glob("*.jpg"))
        
        for img_path in images:
            json_path = img_path.with_suffix(".json")
            
            if not json_path.exists():
                continue
            
            # Check for duplicate
            img_hash = get_image_hash(img_path)
            if img_hash in image_hashes:
                stats['duplicates_removed'] += 1
                continue
            
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Convert JSON to YOLO
            yolo_content = convert_json_to_yolo(json_path, w, h)
            if yolo_content is None:
                continue
            
            # Copy with sequential naming
            img_name = f"{(stats['auto_labeled'] + stats['manual_labeled']):06d}.png"
            label_name = f"{(stats['auto_labeled'] + stats['manual_labeled']):06d}.txt"
            
            shutil.copy2(img_path, merged_dir / "images" / img_name)
            
            with open(merged_dir / "labels" / label_name, 'w', encoding='utf-8') as f:
                f.write(yolo_content)
            
            image_hashes[img_hash] = img_name
            stats['manual_labeled'] += 1
    
    print(f"\n   ✅ Processed: {stats['manual_labeled']} manual labels")
else:
    print(f"   ⚠️  Manual labels directory not found: {MANUAL_LABELS_DIR}")

stats['total_unique'] = stats['auto_labeled'] + stats['manual_labeled']

print("\n" + "="*70)
print("📊 MERGE STATISTICS")
print("="*70)
print(f"  Auto-labeled:       {stats['auto_labeled']}")
print(f"  Manual labeled:     {stats['manual_labeled']}")
print(f"  Duplicates removed: {stats['duplicates_removed']}")
print(f"  Total unique:       {stats['total_unique']}")

# Verify files
actual_images = len(list((merged_dir / "images").glob("*.png")))
actual_labels = len(list((merged_dir / "labels").glob("*.txt")))
print(f"\n📁 Verification:")
print(f"  Images:  {actual_images}")
print(f"  Labels:  {actual_labels}")

if actual_images != actual_labels:
    print("\n⚠️  WARNING: Image and label counts don't match!")

# Save statistics
with open(merged_dir / "merge_stats.json", 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Merged dataset: {merged_dir}")

# ============================================================================
# PART 2: PREPARE TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("📦 PART 2: PREPARING TRAIN/VAL/TEST SPLIT")
print("="*70)

# Collect all labeled images
all_data = []
images_dir = merged_dir / "images"
labels_dir = merged_dir / "labels"

for img_path in images_dir.glob("*.png"):
    label_path = labels_dir / f"{img_path.stem}.txt"
    if label_path.exists() and label_path.stat().st_size > 0:
        all_data.append((img_path, label_path))

print(f"\n✅ Found {len(all_data)} labeled image-label pairs")

if len(all_data) == 0:
    print("❌ ERROR: No labeled data found!")
    exit(1)

# Shuffle and split (80% train, 15% val, 5% test)
random.shuffle(all_data)
train_idx = int(len(all_data) * 0.80)
val_idx = train_idx + int(len(all_data) * 0.15)

train_data = all_data[:train_idx]
val_data = all_data[train_idx:val_idx]
test_data = all_data[val_idx:]

print(f"\n📊 Dataset split:")
print(f"  Train: {len(train_data)} images ({len(train_data)/len(all_data)*100:.1f}%)")
print(f"  Val:   {len(val_data)} images ({len(val_data)/len(all_data)*100:.1f}%)")
print(f"  Test:  {len(test_data)} images ({len(test_data)/len(all_data)*100:.1f}%)")

# Create YOLO directory structure
output_dir = Path(FINAL_MODEL_OUTPUT)
dataset_dir = output_dir / "yolo_dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

train_dir = dataset_dir / "train"
val_dir = dataset_dir / "valid"
test_dir = dataset_dir / "test"

for split_dir in [train_dir, val_dir, test_dir]:
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)

# Copy files to respective splits
print("\n📁 Copying files to splits...\n")

for split_name, data, split_dir in [
    ("Train", train_data, train_dir),
    ("Val", val_data, val_dir),
    ("Test", test_data, test_dir)
]:
    print(f"   Copying {split_name} set...")
    for img_path, label_path in tqdm(data, desc=f"      {split_name}"):
        shutil.copy2(img_path, split_dir / "images" / img_path.name)
        shutil.copy2(label_path, split_dir / "labels" / label_path.name)

# Create data.yaml
yaml_content = {
    'path': str(dataset_dir.absolute()),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 8,
    'names': CLASS_NAMES
}

yaml_path = dataset_dir / "data.yaml"
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"\n✅ YOLO dataset ready: {yaml_path}")

# ============================================================================
# PART 3: TRAIN FINAL MODEL
# ============================================================================

print("\n" + "="*70)
print("🚀 PART 3: TRAINING FINAL PRODUCTION MODEL")
print("="*70)
print(f"   Model: {MODEL_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch: {BATCH_SIZE}")
print(f"   Train: {len(train_data)} images")
print(f"   Val:   {len(val_data)} images")
print(f"   Test:  {len(test_data)} images")
print("="*70)

# Initialize model
model = YOLO(f'{MODEL_SIZE}.pt')

print("\n🏋️  Training started... (this will take 1-2 hours)\n")

try:
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=25,
        save=True,
        project=str(output_dir),
        name='final_model',
        exist_ok=True,
        verbose=True,
        plots=True,
        val=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0
    )
    
    model_path = output_dir / "final_model" / "weights" / "best.pt"
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📦 Model saved: {model_path}")
    
    # ========================================================================
    # PART 4: EVALUATE ON TEST SET
    # ========================================================================
    
    print("\n" + "="*70)
    print("🔬 PART 4: EVALUATING ON TEST SET")
    print("="*70)
    
    model = YOLO(str(model_path))
    
    test_results = model.val(
        data=str(yaml_path),
        split='test',
        plots=True,
        conf=0.25,
        iou=0.6
    )
    
    print(f"\n📈 OVERALL TEST METRICS:")
    print(f"   mAP@0.5:      {test_results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {test_results.box.map:.4f}")
    print(f"   Precision:    {test_results.box.mp:.4f}")
    print(f"   Recall:       {test_results.box.mr:.4f}")
    
    print(f"\n📋 PER-CLASS METRICS (Test Set):")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10}")
    print("-" * 55)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(test_results.box.p):
            print(f"{class_name:<15} {test_results.box.p[i]:>10.4f} "
                  f"{test_results.box.r[i]:>10.4f} {test_results.box.ap50[i]:>10.4f}")
    
    print("-" * 55)
    print(f"{'AVERAGE':<15} {test_results.box.mp:>10.4f} "
          f"{test_results.box.mr:>10.4f} {test_results.box.map50:>10.4f}")
    
    # Save metrics
    metrics = {
        'overall': {
            'mAP@0.5': float(test_results.box.map50),
            'mAP@0.5:0.95': float(test_results.box.map),
            'precision': float(test_results.box.mp),
            'recall': float(test_results.box.mr)
        },
        'per_class': {},
        'dataset_info': {
            'train_images': len(train_data),
            'val_images': len(val_data),
            'test_images': len(test_data),
            'total_images': len(all_data)
        }
    }
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(test_results.box.p):
            metrics['per_class'][class_name] = {
                'precision': float(test_results.box.p[i]),
                'recall': float(test_results.box.r[i]),
                'mAP@0.5': float(test_results.box.ap50[i])
            }
    
    metrics_file = output_dir / "final_test_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved: {metrics_file}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("🎉 STEP 4 COMPLETE! FINAL MODEL IS READY!")
    print("="*70)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Dataset:")
    print(f"      Train:  {len(train_data)} images")
    print(f"      Val:    {len(val_data)} images")
    print(f"      Test:   {len(test_data)} images")
    print(f"      Total:  {len(all_data)} images")
    
    print(f"\n   Test Performance:")
    print(f"      mAP@0.5:      {test_results.box.map50:.4f}")
    print(f"      mAP@0.5:0.95: {test_results.box.map:.4f}")
    print(f"      Precision:    {test_results.box.mp:.4f}")
    print(f"      Recall:       {test_results.box.mr:.4f}")
    
    print(f"\n📂 Output Locations:")
    print(f"   Final model:     {model_path}")
    print(f"   Training logs:   {output_dir / 'final_model'}")
    print(f"   Test metrics:    {metrics_file}")
    print(f"   Dataset:         {dataset_dir}")
    
    print(f"\n🎓 Your Modi script matra detector is production-ready!")
    print(f"   Use this model for inference: {model_path}")

except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. If GPU memory error: Reduce BATCH_SIZE to 8 or 4")
    print("   2. If CUDA error: Check CUDA and PyTorch installation")
    print("   3. Check disk space (model checkpoints need space)")
    raise