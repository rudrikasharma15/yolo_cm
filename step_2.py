"""
🔥 STEP 2: TRAIN INITIAL MATRA DETECTION MODEL 🔥
===================================================

This script trains the initial matra model on your manually labeled data.
This model will then be used for semi-supervised auto-labeling.

Prerequisites:
  - Step 1 completed (matras extracted)
  - All images labeled using LabelImg
  - .txt label files exist for each image

Expected Input:
  matra_labeling/
  └── 00_unlabeled_matras/
      ├── 00_aa_matra/
      │   ├── aa_matra_0001.png
      │   ├── aa_matra_0001.txt  ← YOLO label
      │   └── ...
      ├── 01_i_matra/
      └── ... (11 folders)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import random
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to labeled matras (from Step 1)
LABELED_MATRAS_DIR = "/Users/applemaair/Desktop/modi/modi_script/matra_labeling/00_unlabeled_matras"

# Output directory for training
OUTPUT_DIR = "/Users/applemaair/Desktop/modi/modi_script/matra_training"

# Training parameters
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_SIZE = "yolov8n"  # nano - fast training

# ============================================================================
# DATASET PREPARATION
# ============================================================================

class MatraDatasetPreparator:
    """Prepare matra dataset for YOLO training"""
    
    def __init__(self, labeled_dir, output_dir):
        self.labeled_dir = Path(labeled_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Matra class names (must match labeling)
        self.class_names = [
            'aa_matra',    # 0
            'i_matra',     # 1
            'ii_matra',    # 2
            'u_matra',     # 3
            'uu_matra',    # 4
            'e_matra',     # 5
            'ai_matra',    # 6
            'o_matra',     # 7
            'au_matra',    # 8
            'anusvara',    # 9
            'ri_matra'     # 10
        ]
    
    def validate_labels(self):
        """Check if all images have corresponding labels"""
        
        print("\n" + "="*70)
        print("🔍 VALIDATING LABELED DATA")
        print("="*70)
        
        validation_stats = {}
        total_images = 0
        total_labeled = 0
        
        for matra_folder in self.labeled_dir.glob("*_matra"):
            matra_name = matra_folder.name.split("_", 1)[1]  # Get name without number prefix
            
            # Count images
            images = list(matra_folder.glob("*.png")) + list(matra_folder.glob("*.jpg"))
            
            # Count labels
            labels = list(matra_folder.glob("*.txt"))
            
            # Filter out non-label txt files (like stats.json converted to txt)
            actual_labels = []
            for label_path in labels:
                # Check if corresponding image exists
                img_name = label_path.stem
                if any((matra_folder / f"{img_name}{ext}").exists() for ext in ['.png', '.jpg', '.jpeg']):
                    actual_labels.append(label_path)
            
            total_images += len(images)
            total_labeled += len(actual_labels)
            
            validation_stats[matra_name] = {
                "total_images": len(images),
                "labeled_images": len(actual_labels),
                "percentage": (len(actual_labels) / len(images) * 100) if images else 0
            }
            
            status = "✅" if len(actual_labels) == len(images) else "⚠️ "
            print(f"{status} {matra_name:15s} → {len(actual_labels):3d}/{len(images):3d} labeled ({validation_stats[matra_name]['percentage']:.1f}%)")
        
        print("\n" + "="*70)
        print(f"TOTAL: {total_labeled}/{total_images} images labeled ({total_labeled/total_images*100:.1f}%)")
        
        if total_labeled < total_images * 0.9:  # Less than 90% labeled
            print("\n⚠️  WARNING: Not all images are labeled!")
            print("   Please complete labeling before training.")
            print(f"   Missing: {total_images - total_labeled} labels")
            return False
        else:
            print("\n✅ Validation passed! Ready for training.")
            return True
    
    def prepare_yolo_dataset(self, train_split=0.8):
        """Organize labeled data into YOLO format"""
        
        print("\n" + "="*70)
        print("📦 PREPARING YOLO DATASET")
        print("="*70)
        
        dataset_dir = self.output_dir / "initial_matra_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Create train/val directories
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "valid"
        
        for split_dir in [train_dir, val_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Collect all labeled images
        all_data = []
        
        for matra_folder in self.labeled_dir.glob("*_matra"):
            # Get all images with labels
            images = list(matra_folder.glob("*.png")) + list(matra_folder.glob("*.jpg"))
            
            for img_path in images:
                label_path = img_path.parent / f"{img_path.stem}.txt"
                
                if label_path.exists() and label_path.stat().st_size > 0:
                    all_data.append((img_path, label_path))
        
        print(f"\n✅ Found {len(all_data)} labeled images")
        
        if len(all_data) == 0:
            print("\n❌ ERROR: No labeled data found!")
            print("   Please label images first using LabelImg.")
            return None
        
        # Shuffle and split
        random.shuffle(all_data)
        split_idx = int(len(all_data) * train_split)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"   Train: {len(train_data)} images")
        print(f"   Valid: {len(val_data)} images")
        
        # Copy files to train/val directories
        for split_name, data, split_dir in [
            ("train", train_data, train_dir),
            ("valid", val_data, val_dir)
        ]:
            print(f"\n📂 Preparing {split_name} set...")
            
            for img_path, label_path in tqdm(data, desc=f"  Copying {split_name} files"):
                # Copy image
                dest_img = split_dir / "images" / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Copy label
                dest_label = split_dir / "labels" / label_path.name
                shutil.copy2(label_path, dest_label)
        
        # Create data.yaml
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': 11,  # 11 matra classes
            'names': self.class_names
        }
        
        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"\n✅ YOLO dataset prepared: {dataset_dir}")
        print(f"   data.yaml: {yaml_path}")
        
        return yaml_path

# ============================================================================
# MODEL TRAINING
# ============================================================================

class MatraModelTrainer:
    """Train initial matra detection model"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_model(self, data_yaml, epochs=100, img_size=640, batch_size=16):
        """Train YOLO model on matra dataset"""
        
        print("\n" + "="*70)
        print("🚀 TRAINING INITIAL MATRA MODEL")
        print("="*70)
        print(f"   Model: YOLOv8-nano")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {img_size}")
        print(f"   Batch size: {batch_size}")
        print("="*70)
        
        # Initialize model
        model = YOLO(f'{MODEL_SIZE}.pt')
        
        # Train
        print("\n🏋️  Training started... This will take 30-60 minutes.")
        print("   You can monitor progress in the terminal.")
        print("   Training logs will be saved automatically.\n")
        
        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=15,  # Early stopping
                save=True,
                project=str(self.output_dir),
                name='initial_matra_model',
                exist_ok=True,
                verbose=True,
                plots=True,  # Generate training plots
                val=True     # Validate during training
            )
            
            model_path = self.output_dir / "initial_matra_model" / "weights" / "best.pt"
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE!")
            print("="*70)
            print(f"\n📦 Model saved: {model_path}")
            
            # Print final metrics
            print("\n📊 FINAL METRICS:")
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50(B)' in metrics:
                    print(f"   mAP@0.5:  {metrics['metrics/mAP50(B)']:.4f}")
                if 'metrics/precision(B)' in metrics:
                    print(f"   Precision: {metrics['metrics/precision(B)']:.4f}")
                if 'metrics/recall(B)' in metrics:
                    print(f"   Recall:    {metrics['metrics/recall(B)']:.4f}")
            
            return model_path
            
        except Exception as e:
            print(f"\n❌ ERROR during training: {e}")
            print("\nTroubleshooting:")
            print("  1. Check if GPU is available")
            print("  2. Try reducing batch size (set BATCH_SIZE=8 or 4)")
            print("  3. Check if data.yaml paths are correct")
            return None
    
    def validate_model(self, model_path, data_yaml):
        """Run validation on trained model"""
        
        print("\n" + "="*70)
        print("🔬 VALIDATING MODEL ON TEST SET")
        print("="*70)
        
        model = YOLO(str(model_path))
        
        results = model.val(
            data=str(data_yaml),
            split='val',
            save_json=True,
            plots=True
        )
        
        print("\n✅ Validation complete!")
        print("\n📊 VALIDATION METRICS:")
        print(f"   mAP@0.5:   {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall:    {results.box.mr:.4f}")
        
        # Per-class metrics
        print("\n📋 PER-CLASS METRICS:")
        print("   " + "-"*60)
        print(f"   {'Class':<15} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10}")
        print("   " + "-"*60)
        
        class_names = [
            'aa_matra', 'i_matra', 'ii_matra', 'u_matra', 'uu_matra',
            'e_matra', 'ai_matra', 'o_matra', 'au_matra',
            'anusvara', 'ri_matra'
        ]
        
        for i, class_name in enumerate(class_names):
            if i < len(results.box.p):
                print(f"   {class_name:<15} {results.box.p[i]:>10.4f} {results.box.r[i]:>10.4f} {results.box.ap50[i]:>10.4f}")
        
        print("   " + "-"*60)
        
        return results

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run initial matra model training"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        🔥 INITIAL MATRA MODEL TRAINING - STEP 2 🔥                   ║
║                                                                      ║
║  Training detection model on manually labeled matra data            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Prepare dataset
    preparator = MatraDatasetPreparator(
        labeled_dir=LABELED_MATRAS_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Validate labels
    if not preparator.validate_labels():
        print("\n⚠️  Please complete labeling before continuing!")
        return
    
    # Prepare YOLO dataset
    data_yaml = preparator.prepare_yolo_dataset(train_split=0.8)
    
    if data_yaml is None:
        return
    
    # Step 2: Train model
    trainer = MatraModelTrainer(output_dir=OUTPUT_DIR)
    
    model_path = trainer.train_model(
        data_yaml=data_yaml,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    if model_path is None:
        return
    
    # Step 3: Validate model
    trainer.validate_model(model_path, data_yaml)
    
    print("\n" + "="*70)
    print("🎉 STEP 2 COMPLETE!")
    print("="*70)
    print(f"\n✅ Initial matra model trained: {model_path}")
    print("\n📝 NEXT STEP:")
    print("   Run Step 3 script for semi-supervised auto-labeling")
    print("   This will use your trained model to label remaining images!")
    print("\n💡 TIP: If mAP@0.5 is below 0.6, consider:")
    print("   - Labeling more images (50+ per matra)")
    print("   - Checking label quality (are boxes accurate?)")
    print("   - Training for more epochs (increase EPOCHS=150)")

if __name__ == "__main__":
    main()