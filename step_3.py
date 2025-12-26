"""
🔥 STEP 3: SEMI-SUPERVISED AUTO-LABELING - WINDOWS GPU 🔥
===========================================================

Use trained model from Step 2 to auto-label full 7K dataset.
Run this in PowerShell/CMD with GPU support.

Prerequisites:
- Completed Step 2 (trained initial model)
- Full Dataset_Modi available (7K images)
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
from ultralytics import YOLO
import torch

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Input: Trained model from Step 2
MODEL_PATH = r"C:\Users\YourName\Desktop\modi_final\step2_training\initial_model\weights\best.pt"

# Input: Full 7K barakhadi dataset
DATASET_MODI = r"C:\Users\YourName\Desktop\modi_final\Dataset_Modi"

# Output directory
OUTPUT_DIR = r"C:\Users\YourName\Desktop\modi_final\step3_auto_labeled"

# Auto-labeling settings
MIN_CONFIDENCE = 0.15  # Very low to catch everything

# 8 Matra mappings (exactly as in your original code)
MATRA_MAPPINGS = {
    "aa_matra": {
        "class_id": 0,
        "folder_endings": ["AA", "AA-kaku", "AA-KHAR", "GHAA", "CHAA", "CHHAA", "JAA", "ZAA",
                          "TRAA", "TTAA", "BAA", "BHAA", "DAA", "DHAA", "HAA", "LAA", "MAA",
                          "NAA", "PAA", "PHAA", "RAA", "SAA", "SHAA", "TAA-talwar", "THAA",
                          "THHAA", "VAA", "YAA", "ALA-kamal", "aa-kshatriy", "DNYAa"]
    },
    "i_matra": {
        "class_id": 1,
        "folder_endings": ["KI-kiran", "KHI", "GI", "GHI", "CHI", "CHHI", "JI", "ZI", "TRI",
                          "TTI", "BI", "BHI", "bhai", "DI", "DHI", "DHHI-dhag", "HI", "LI",
                          "MI", "NI", "PI", "PHI", "RI", "SHI", "SI", "TI-talwar", "THI",
                          "THHI", "VI", "YI", "ALI-kamal", "i-kshatriy", "DNYi"]
    },
    "u_matra": {
        "class_id": 2,
        "folder_endings": ["KU-kunfu", "KHU", "GU", "GHU", "CHU", "CHHU", "JU", "ZU", "TRU",
                          "TTU", "BU", "BHU", "DU", "DHU", "DHHU-dhag", "HU", "LU", "MU",
                          "NU", "PU", "PHU", "RU", "SHU", "SU", "TU-talwar", "THU", "THHU",
                          "VU", "YU", "ALU-kamal", "u-kshatriy", "DNYu"]
    },
    "e_matra": {
        "class_id": 3,
        "folder_endings": ["KE-kedar", "KHE", "GE", "GHE", "CHE", "CHHE", "JE", "ZE", "TRE",
                          "TTE", "BE", "BHE", "DE", "DHE", "DHHE-dhag", "HE", "LE", "ME",
                          "NE", "PE", "PHE", "RE", "SE", "SHE", "she", "TE--talwar", "THE",
                          "THHE", "VE", "YE", "ALE-kamal", "e-kshatriy", "DNYAe"]
    },
    "ai_matra": {
        "class_id": 4,
        "folder_endings": ["KAI-kailas", "KHAI", "GAI", "GHAI", "CHAI", "CHHAI", "JAI", "ZAI",
                          "TRAI", "TTAI", "BAI", "DAI", "DHAI", "DHHAI-dhag", "HAI", "LAI",
                          "MAI", "NAI", "PAI", "PHAI", "RAI", "SAI", "SHAI", "shai",
                          "TAI-talwar", "THAI", "THHAI", "VAI", "YAI", "ALAI-kamal",
                          "ai-kshatriy", "DNYAai"]
    },
    "o_matra": {
        "class_id": 5,
        "folder_endings": ["KO-komal", "KHO", "GO", "GHO", "CHO", "CHHO", "JO", "ZO", "TRO",
                          "TTO", "BO", "BHO", "DO", "DHO", "DHHO-dhag", "HO", "LO", "MO",
                          "NO", "PO", "PHO", "RO", "SHO", "sho", "SO", "TO-talwar", "THO",
                          "THHO", "VO", "YO", "ALO-kamal", "o-kshatriy", "DNYo"]
    },
    "au_matra": {
        "class_id": 6,
        "folder_endings": ["KAU-kaul", "KHAU", "GAU", "GHAU", "CHAO", "CHHAO", "JAO", "ZAO",
                          "TRAO", "TTAO", "BAO", "BHAO", "DAU", "DAO-daut", "DHAO",
                          "DHHAO-dhag", "HAO", "LAO", "MAO", "NAO", "PAO", "PHAO", "RAO",
                          "SAO", "SHAO", "shau", "TAO-talwar", "THAO", "THHAO", "VAO", "YAO",
                          "ALAO-kamal", "au-kshatriy", "DNYau"]
    },
    "anusvara": {
        "class_id": 7,
        "folder_endings": ["KAN-kangan", "KHAN", "GAM", "GHAM", "CHAM", "CHHAm", "JAM", "ZAM",
                          "TRAM", "TTAM", "BAM", "BHAM", "DAM", "DHAM", "DHHAM-dhag", "HAM",
                          "LAM", "MAM", "NAM", "PAM", "PHAM", "RAM", "SAM", "SHAM", "sham",
                          "TAM-talwar", "THAM", "THHAM", "VAM", "YAM", "ALAM-kamal",
                          "nm-kshatriy", "DNYAm"]
    }
}

MATRA_NAMES = list(MATRA_MAPPINGS.keys())

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       🔥 STEP 3: SEMI-SUPERVISED AUTO-LABELING 🔥                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Check inputs
print("🔍 Checking inputs...")

model_path = Path(MODEL_PATH)
if not model_path.exists():
    print(f"❌ ERROR: Model not found: {MODEL_PATH}")
    print("\n💡 Make sure you've completed Step 2 and updated MODEL_PATH")
    exit(1)
print(f"   ✅ Model found: {model_path}")

dataset_path = Path(DATASET_MODI)
if not dataset_path.exists():
    print(f"❌ ERROR: Dataset not found: {DATASET_MODI}")
    exit(1)
print(f"   ✅ Dataset found: {dataset_path}")

# Check GPU
print(f"\n🖥️  GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# LOAD MODEL AND PREPARE OUTPUT
# ============================================================================

print("\n" + "="*70)
print("🤖 LOADING MODEL AND PREPARING OUTPUT")
print("="*70)

# Load model
print(f"\n📦 Loading model: {model_path.name}")
model = YOLO(str(model_path))
print("   ✅ Model loaded successfully")

# Create output structure
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

for i, name in enumerate(MATRA_NAMES):
    (output_dir / f"{i:02d}_{name}" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / f"{i:02d}_{name}" / "labels").mkdir(parents=True, exist_ok=True)

print(f"✅ Output directory ready: {output_dir}")

# ============================================================================
# COLLECT IMAGES TO PROCESS
# ============================================================================

print("\n" + "="*70)
print("📂 COLLECTING IMAGES FROM DATASET")
print("="*70)

# Skip standalone vowel folders
skip_folders = ['1 a-ananas', '2 aa-aai', '3 i-imarat', '4 u-ukhal',
                '5 e-edka', '6 ai-airan', '7 o-odha', '8 au-aushadh',
                '9 nm-angthi', '10 ahaa']

# Get all folders
all_folders = [f for f in dataset_path.iterdir()
               if f.is_dir() and f.name not in skip_folders]

print(f"\n📁 Found {len(all_folders)} folders to process")

# Collect ALL images
all_images = []
for folder in all_folders:
    imgs = (list(folder.glob("*.png")) +
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")))
    all_images.extend(imgs)

print(f"📸 Total images to process: {len(all_images)}")

if len(all_images) == 0:
    print("❌ ERROR: No images found in dataset!")
    exit(1)

# ============================================================================
# AUTO-LABEL ALL IMAGES
# ============================================================================

print("\n" + "="*70)
print("🔥 AUTO-LABELING IMAGES")
print("="*70)
print(f"   Confidence threshold: {MIN_CONFIDENCE}")
print(f"   Processing {len(all_images)} images...")
print("="*70)

stats = {name: 0 for name in MATRA_NAMES}
stats['no_detection'] = 0
stats['processed'] = 0

print("\n⏳ This will take some time... Progress bar below:\n")

for img_path in tqdm(all_images, desc="Auto-labeling", unit="img"):
    try:
        stats['processed'] += 1
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Run detection
        results = model.predict(
            source=str(img_path),
            conf=MIN_CONFIDENCE,
            verbose=False,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            stats['no_detection'] += 1
            continue
        
        boxes = results[0].boxes
        
        # Group by class
        class_detections = defaultdict(list)
        for box in boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            class_detections[class_id].append((conf, box))
        
        # Save best detection for each class
        for class_id, detections in class_detections.items():
            if class_id >= 8:
                continue
            
            # Get best detection for this class
            best_conf, best_box = max(detections, key=lambda x: x[0])
            
            matra_name = MATRA_NAMES[class_id]
            matra_dir = output_dir / f"{class_id:02d}_{matra_name}"
            
            # Count existing images
            img_count = len(list((matra_dir / "images").glob("*.png")))
            
            # Save image
            img_filename = f"{matra_name}_{img_count:05d}.png"
            dest_img = matra_dir / "images" / img_filename
            shutil.copy2(img_path, dest_img)
            
            # Create YOLO label
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Save label
            label_filename = f"{matra_name}_{img_count:05d}.txt"
            label_path = matra_dir / "labels" / label_filename
            
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            stats[matra_name] += 1
    
    except Exception as e:
        continue

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*70)
print("📊 AUTO-LABELING RESULTS")
print("="*70)

print(f"\n{'Matra':<15} {'Images Labeled':>15}")
print("-" * 40)

total_labeled = 0
for name in MATRA_NAMES:
    count = stats[name]
    total_labeled += count
    print(f"{name:<15} {count:>15}")

print("-" * 40)
print(f"{'TOTAL LABELED':<15} {total_labeled:>15}")
print(f"{'No detection':<15} {stats['no_detection']:>15}")
print(f"{'Processed':<15} {stats['processed']:>15}")

# Calculate coverage
coverage = (total_labeled / stats['processed']) * 100 if stats['processed'] > 0 else 0
print(f"\n📈 Coverage: {coverage:.1f}% of images have at least one matra detected")

# Save statistics
stats_file = output_dir / "auto_labeling_stats.json"
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Statistics saved: {stats_file}")

print("\n" + "="*70)
print("🎉 STEP 3 COMPLETE!")
print("="*70)
print(f"\n✅ Auto-labeled {total_labeled} images across 8 matra classes")
print(f"📂 Output location: {output_dir}")
print(f"\n📝 NEXT STEP: Run Step 4 (merge + final training)")
print(f"   Auto-labeled data location: {output_dir}")
print(f"   Manual labels location: {Path(MODEL_PATH).parent.parent.parent / 'matra_extraction_PERFECT' / 'unlabeled_matras'}")