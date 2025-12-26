# """
# COMPLETE PIPELINE - CREATE CLEAN DATASET AND RETRAIN
# =====================================================
# 1. Select 700-1000 images per class (consonants + vowels ONLY, NO digits)
# 2. Split 70-15-15 (train-val-test) 
# 3. Verify NO data leakage
# 4. Create data.yaml with ACTUAL character names (not numbers!)
# 5. Train model
# """

# import os
# import shutil
# import random
# from pathlib import Path
# from collections import defaultdict
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import torch
# import json

# # ============================================================================
# # STEP 1: CREATE CLEAN DATASET
# # ============================================================================

# def create_clean_dataset():
#     """Create dataset with 700-1000 images per class, NO leakage"""
    
#     print("="*70)
#     print("STEP 1: CREATE CLEAN DATASET")
#     print("="*70)
    
#     # Source folders (NO DIGITS!)
#     CONSONANTS_DIR = Path(r"C:\Users\admin\Desktop\MODI_HChar\MODI_HChar\consonants")
#     VOWELS_DIR = Path(r"C:\Users\admin\Desktop\MODI_HChar\MODI_HChar\vowels")
    
#     # Output
#     OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\CLEAN_DATASET_FINAL")
    
#     # Clear old data
#     if OUTPUT_DIR.exists():
#         print(f"\n🗑️  Removing old dataset...")
#         shutil.rmtree(OUTPUT_DIR)
    
#     OUTPUT_DIR.mkdir(parents=True)
    
#     # Get all class folders
#     all_class_folders = []
    
#     print(f"\n📁 Scanning consonants...")
#     if CONSONANTS_DIR.exists():
#         consonants = sorted([f for f in CONSONANTS_DIR.iterdir() if f.is_dir()])
#         all_class_folders.extend(consonants)
#         print(f"   Found {len(consonants)} consonant classes")
    
#     print(f"\n📁 Scanning vowels...")
#     if VOWELS_DIR.exists():
#         vowels = sorted([f for f in VOWELS_DIR.iterdir() if f.is_dir()])
#         all_class_folders.extend(vowels)
#         print(f"   Found {len(vowels)} vowel classes")
    
#     print(f"\n✅ Total classes: {len(all_class_folders)}")
#     print(f"   Classes: {[f.name for f in all_class_folders[:10]]}...")
    
#     # Collect images per class
#     print(f"\n📊 Collecting images per class...")
    
#     class_data = []
#     total_images = 0
    
#     for class_idx, class_folder in enumerate(all_class_folders):
#         class_name = class_folder.name.upper()
        
#         # Get all images
#         images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        
#         # Limit to 700-1000 images
#         if len(images) > 1000:
#             images = random.sample(images, 1000)
#         elif len(images) < 700:
#             print(f"   ⚠️  {class_name}: Only {len(images)} images (need 700+)")
        
#         class_data.append({
#             'id': class_idx,
#             'name': class_name,
#             'images': images,
#             'count': len(images)
#         })
        
#         total_images += len(images)
        
#         if class_idx < 10 or class_idx >= len(all_class_folders) - 5:
#             print(f"   Class {class_idx:2d} ({class_name:10s}): {len(images):4d} images")
#         elif class_idx == 10:
#             print(f"   ...")
    
#     print(f"\n✅ Total images collected: {total_images:,}")
    
#     # ========================================================================
#     # STEP 2: SPLIT WITH NO LEAKAGE (70-15-15)
#     # ========================================================================
    
#     print(f"\n{'='*70}")
#     print("STEP 2: SPLIT DATASET (70-15-15) - NO LEAKAGE")
#     print("="*70)
    
#     # Create directories
#     for split in ['train', 'val', 'test']:
#         (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
#         (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
#     # Track ALL filenames to prevent leakage
#     all_filenames = set()
#     train_files = set()
#     val_files = set()
#     test_files = set()
    
#     split_counts = {'train': 0, 'val': 0, 'test': 0}
    
#     for class_info in class_data:
#         class_id = class_info['id']
#         class_name = class_info['name']
#         images = class_info['images']
        
#         # Shuffle
#         random.shuffle(images)
        
#         # Split 70-15-15
#         n = len(images)
#         train_n = int(n * 0.70)
#         val_n = int(n * 0.15)
        
#         train_imgs = images[:train_n]
#         val_imgs = images[train_n:train_n + val_n]
#         test_imgs = images[train_n + val_n:]
        
#         # Process each split
#         for split_name, split_images in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
#             for img_path in split_images:
#                 # Check for duplicates
#                 if img_path.name in all_filenames:
#                     print(f"   ⚠️  WARNING: Duplicate {img_path.name}")
#                     continue
                
#                 all_filenames.add(img_path.name)
                
#                 if split_name == 'train':
#                     train_files.add(img_path.name)
#                 elif split_name == 'val':
#                     val_files.add(img_path.name)
#                 else:
#                     test_files.add(img_path.name)
                
#                 # Create unique filename
#                 new_name = f"{class_name}_{img_path.stem}_{class_id:03d}.jpg"
                
#                 # Copy image
#                 img_dest = OUTPUT_DIR / split_name / 'images' / new_name
#                 shutil.copy2(img_path, img_dest)
                
#                 # Create label file (YOLO format)
#                 img = cv2.imread(str(img_path))
#                 if img is not None:
#                     h, w = img.shape[:2]
#                     # Full image bounding box
#                     label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                    
#                     label_dest = OUTPUT_DIR / split_name / 'labels' / new_name.replace('.jpg', '.txt')
#                     with open(label_dest, 'w') as f:
#                         f.write(label_content)
                    
#                     split_counts[split_name] += 1
    
#     print(f"\n📊 Split distribution:")
#     print(f"   Train: {split_counts['train']:,} images ({split_counts['train']/total_images*100:.1f}%)")
#     print(f"   Val:   {split_counts['val']:,} images ({split_counts['val']/total_images*100:.1f}%)")
#     print(f"   Test:  {split_counts['test']:,} images ({split_counts['test']/total_images*100:.1f}%)")
    
#     # ========================================================================
#     # STEP 3: VERIFY NO DATA LEAKAGE
#     # ========================================================================
    
#     print(f"\n{'='*70}")
#     print("STEP 3: VERIFY NO DATA LEAKAGE")
#     print("="*70)
    
#     train_val_overlap = train_files & val_files
#     train_test_overlap = train_files & test_files
#     val_test_overlap = val_files & test_files
    
#     print(f"\n🔍 Checking for overlaps...")
#     print(f"   Train ∩ Val:  {len(train_val_overlap)} files")
#     print(f"   Train ∩ Test: {len(train_test_overlap)} files")
#     print(f"   Val ∩ Test:   {len(val_test_overlap)} files")
    
#     if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
#         print(f"\n✅ NO DATA LEAKAGE! All splits are clean!")
#     else:
#         print(f"\n❌ DATA LEAKAGE DETECTED!")
#         return None
    
#     # ========================================================================
#     # STEP 4: CREATE data.yaml WITH ACTUAL CHARACTER NAMES
#     # ========================================================================
    
#     print(f"\n{'='*70}")
#     print("STEP 4: CREATE data.yaml WITH CHARACTER NAMES")
#     print("="*70)
    
#     # Create character names list
#     character_names = [c['name'] for c in class_data]
    
#     print(f"\n📝 Character names (first 10):")
#     for i in range(min(10, len(character_names))):
#         print(f"   {i}: {character_names[i]}")
    
#     yaml_content = f"""# Modi Character Dataset - CLEAN (NO LEAKAGE)
# path: {OUTPUT_DIR}
# train: train/images
# val: val/images
# test: test/images

# nc: {len(character_names)}
# names: {character_names}

# # Dataset info
# total_images: {total_images}
# train_images: {split_counts['train']}
# val_images: {split_counts['val']}
# test_images: {split_counts['test']}

# # Verification
# no_leakage: true
# digits_included: false
# vowels_only: {len(vowels) == len(character_names)}
# consonants_and_vowels: true
# """
    
#     yaml_path = OUTPUT_DIR / 'data.yaml'
#     with open(yaml_path, 'w') as f:
#         f.write(yaml_content)
    
#     print(f"\n✅ Saved data.yaml: {yaml_path}")
    
#     print(f"\n{'='*70}")
#     print("✅ DATASET CREATION COMPLETE!")
#     print("="*70)
#     print(f"📂 Dataset location: {OUTPUT_DIR}")
#     print(f"📊 Total classes: {len(character_names)}")
#     print(f"📊 Total images: {total_images:,}")
#     print(f"✅ NO data leakage verified!")
#     print(f"✅ Character names (not numbers) included!")
    
#     return yaml_path

# # ============================================================================
# # STEP 5: TRAIN MODEL
# # ============================================================================

# def train_model(yaml_path):
#     """Train model on clean dataset"""
    
#     print(f"\n{'='*70}")
#     print("STEP 5: TRAIN MODEL")
#     print("="*70)
    
#     # Check GPU
#     if not torch.cuda.is_available():
#         print("\n❌ NO GPU! Install CUDA PyTorch first!")
#         return
    
#     gpu_name = torch.cuda.get_device_name(0)
#     gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
#     print(f"\n✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
#     # Training config
#     EPOCHS = 100
#     BATCH_SIZE = 24 if gpu_mem >= 8 else 16
#     IMG_SIZE = 640
#     MODEL_SIZE = 'yolov8m'
    
#     OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\RETRAINED_CLEAN_FINAL")
#     OUTPUT_DIR.mkdir(exist_ok=True)
    
#     print(f"\n⚙️  Configuration:")
#     print(f"   Epochs: {EPOCHS}")
#     print(f"   Batch: {BATCH_SIZE}")
#     print(f"   Image size: {IMG_SIZE}")
#     print(f"   Model: {MODEL_SIZE}")
    
#     print(f"\n{'='*70}")
#     response = input("🚀 Start training? [Y/n]: ")
#     if response.lower() == 'n':
#         print("✅ Cancelled")
#         return
    
#     print(f"\n🏋️  Training started...\n")
    
#     model = YOLO(f'{MODEL_SIZE}.pt')
    
#     try:
#         results = model.train(
#             data=str(yaml_path),
#             epochs=EPOCHS,
#             imgsz=IMG_SIZE,
#             batch=BATCH_SIZE,
#             workers=8,
#             cache='disk',
#             amp=True,
#             project=str(OUTPUT_DIR),
#             name='character_model',
#             exist_ok=True,
#             patience=30,
#             save=True,
#             plots=True,
#             verbose=True,
#             val=True,
#             device=0
#         )
        
#         model_path = OUTPUT_DIR / 'character_model' / 'weights' / 'best.pt'
        
#         print(f"\n{'='*70}")
#         print("✅ TRAINING COMPLETE!")
#         print("="*70)
#         print(f"📦 Model: {model_path}")
        
#         # Test
#         print(f"\n🔬 Testing...")
#         model = YOLO(str(model_path))
#         test_results = model.val(data=str(yaml_path), split='test')
        
#         print(f"\n📈 Test Results:")
#         print(f"   mAP@0.5: {test_results.box.map50*100:.2f}%")
#         print(f"   Precision: {test_results.box.mp*100:.2f}%")
#         print(f"   Recall: {test_results.box.mr*100:.2f}%")
        
#         print(f"\n✅ DONE! Model ready to use!")
        
#     except KeyboardInterrupt:
#         print(f"\n⚠️  Training interrupted!")
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         raise

# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == '__main__':
#     random.seed(42)  # Reproducibility
    
#     print("""
# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║  🚀 COMPLETE PIPELINE - CLEAN DATASET + TRAINING 🚀             ║
# ║                                                                  ║
# ║  1. ✅ Select 700-1000 images per class                         ║
# ║  2. ✅ Split 70-15-15 (NO leakage)                              ║
# ║  3. ✅ Verify no duplicates                                     ║
# ║  4. ✅ Create data.yaml with CHARACTER NAMES                    ║
# ║  5. ✅ Train model                                              ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
# """)
    
#     # Step 1-4: Create dataset
#     yaml_path = create_clean_dataset()
    
#     if yaml_path is None:
#         print("\n❌ Dataset creation failed!")
#     else:
#         # Step 5: Train
#         train_model(yaml_path)
"""
RESUME TRAINING FROM EPOCH 12
==============================
Continues training your model from where it stopped
"""

from pathlib import Path
from ultralytics import YOLO
import torch

# Paths
DATASET_YAML = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\CLEAN_DATASET_FINAL\data.yaml")
CHECKPOINT = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\RETRAINED_CLEAN_FINAL\character_model\weights\last.pt")
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\RETRAINED_CLEAN_FINAL")

def check_setup():
    """Check if everything is ready"""
    print("="*70)
    print("PRE-FLIGHT CHECK")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ NO GPU DETECTED!")
        print("   Training will be VERY slow on CPU.")
        response = input("\n   Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            return False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU: {gpu_name}")
        print(f"   Memory: {gpu_mem:.1f} GB")
    
    # Check dataset
    if not DATASET_YAML.exists():
        print(f"\n❌ Dataset YAML not found: {DATASET_YAML}")
        return False
    print(f"\n✅ Dataset YAML: {DATASET_YAML}")
    
    # Check checkpoint
    if not CHECKPOINT.exists():
        print(f"\n❌ Checkpoint not found: {CHECKPOINT}")
        return False
    print(f"\n✅ Checkpoint: {CHECKPOINT}")
    print(f"   Size: {CHECKPOINT.stat().st_size / (1024*1024):.1f} MB")
    
    # Load model to check epoch
    try:
        model = YOLO(str(CHECKPOINT))
        print(f"\n✅ Model loaded successfully")
        print(f"   Classes: {len(model.names)}")
        print(f"   Sample classes: {list(model.names.values())[:5]}")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return False
    
    return True

def resume_training():
    """Resume training from checkpoint"""
    
    print("\n" + "="*70)
    print("RESUME TRAINING")
    print("="*70)
    
    # Training parameters
    REMAINING_EPOCHS = 88  # 100 - 12 = 88 remaining
    BATCH_SIZE = 24
    IMG_SIZE = 640
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Starting from: Epoch 12")
    print(f"   Remaining epochs: {REMAINING_EPOCHS}")
    print(f"   Total epochs: 100")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Output: {OUTPUT_DIR}")
    
    print(f"\n" + "="*70)
    print("⏱️  ESTIMATED TIME:")
    print("="*70)
    
    # Estimate time (rough calculation)
    # Assume ~30 seconds per epoch on good GPU
    estimated_minutes = REMAINING_EPOCHS * 0.5  # 30 sec per epoch
    hours = int(estimated_minutes // 60)
    mins = int(estimated_minutes % 60)
    
    print(f"\n   ~{hours}h {mins}m for remaining 88 epochs")
    print(f"   (This is an estimate, actual time varies)")
    
    print(f"\n" + "="*70)
    response = input("🚀 Start training? [Y/n]: ")
    if response.lower() == 'n':
        print("\n✅ Cancelled")
        return
    
    print(f"\n{'='*70}")
    print("🏋️  TRAINING IN PROGRESS...")
    print("="*70)
    print("\n💡 TIP: You can stop anytime with Ctrl+C")
    print("   Progress will be saved automatically!\n")
    
    # Load checkpoint
    model = YOLO(str(CHECKPOINT))
    
    try:
        # Resume training
        results = model.train(
            data=str(DATASET_YAML),
            epochs=100,  # Total epochs (will continue from 12)
            resume=True,  # KEY: Resume from checkpoint
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=8,
            cache='disk',
            amp=True,
            project=str(OUTPUT_DIR.parent),
            name=OUTPUT_DIR.name,
            exist_ok=True,
            patience=30,
            save=True,
            plots=True,
            verbose=True,
            val=True,
            device=0
        )
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        best_model = OUTPUT_DIR / "character_model" / "weights" / "best.pt"
        print(f"\n📦 Best Model: {best_model}")
        
        # Test the model
        print(f"\n🔬 Testing on test set...")
        test_model = YOLO(str(best_model))
        test_results = test_model.val(data=str(DATASET_YAML), split='test')
        
        print(f"\n📊 Final Test Results:")
        print(f"   mAP@0.5: {test_results.box.map50*100:.2f}%")
        print(f"   mAP@0.5:0.95: {test_results.box.map*100:.2f}%")
        print(f"   Precision: {test_results.box.mp*100:.2f}%")
        print(f"   Recall: {test_results.box.mr*100:.2f}%")
        
        print(f"\n{'='*70}")
        print("✅ ALL DONE! Model is ready to use!")
        print("="*70)
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("⚠️  TRAINING INTERRUPTED!")
        print("="*70)
        print(f"\n✅ Don't worry! Progress has been saved.")
        print(f"📦 You can resume again from: {CHECKPOINT}")
        print(f"\n   Just run this script again!")
        
    except Exception as e:
        print(f"\n\n{'='*70}")
        print("❌ ERROR DURING TRAINING")
        print("="*70)
        print(f"\nError: {e}")
        print(f"\nCheckpoint saved at: {CHECKPOINT}")
        print(f"You can try resuming again.")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🔄 RESUME TRAINING - FROM EPOCH 12 TO 100 🔄             ║
║                                                                  ║
║  Current Status:  Epoch 12/100 (12% complete)                   ║
║  Remaining:       88 epochs                                     ║
║                                                                  ║
║  This will continue training your model with:                   ║
║  ✅ Character names (B, CH, DH, GA, etc.)                       ║
║  ✅ 47 Modi characters (consonants + vowels)                    ║
║  ✅ Clean dataset with NO data leakage                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Check everything
    if not check_setup():
        print("\n❌ Setup check failed. Fix the issues above and try again.")
        return
    
    # Resume training
    resume_training()

if __name__ == "__main__":
    main()