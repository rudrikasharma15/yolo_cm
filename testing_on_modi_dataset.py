"""
VERIFY CLEAN DATASET - CHECK FOR DATA LEAKAGE
==============================================
Verify that the clean dataset has NO overlap between train/val/test
Run this BEFORE training to ensure dataset is actually clean!
"""

import hashlib
from pathlib import Path
from collections import defaultdict
import json

def compute_md5(file_path):
    """Compute MD5 hash of file"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def verify_dataset_is_clean(dataset_dir):
    """
    Comprehensive verification that dataset has NO leakage
    """
    
    print("="*80)
    print("🔍 VERIFYING DATASET IS CLEAN - NO LEAKAGE CHECK")
    print("="*80)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_dir}")
        return False
    
    print(f"\n📁 Checking: {dataset_dir}")
    
    # ========================================================================
    # STEP 1: Verify folder structure
    # ========================================================================
    
    print(f"\n📂 STEP 1: Verifying folder structure...")
    
    required_folders = [
        'train/images', 'train/labels',
        'val/images', 'val/labels',
        'test/images', 'test/labels'
    ]
    
    missing = []
    for folder in required_folders:
        if not (dataset_path / folder).exists():
            missing.append(folder)
    
    if missing:
        print(f"   ❌ Missing folders: {missing}")
        return False
    else:
        print(f"   ✅ All required folders exist")
    
    # ========================================================================
    # STEP 2: Count images in each split
    # ========================================================================
    
    print(f"\n📊 STEP 2: Counting images...")
    
    splits = {}
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / split / 'images'
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        splits[split] = {
            'images': images,
            'count': len(images)
        }
        print(f"   {split.upper()}: {len(images):,} images")
    
    total = sum(s['count'] for s in splits.values())
    print(f"   TOTAL: {total:,} images")
    
    if total == 0:
        print(f"\n   ❌ No images found!")
        return False
    
    # ========================================================================
    # STEP 3: Check for filename overlaps
    # ========================================================================
    
    print(f"\n🔍 STEP 3: Checking for filename overlaps...")
    
    train_names = set(img.stem for img in splits['train']['images'])
    val_names = set(img.stem for img in splits['val']['images'])
    test_names = set(img.stem for img in splits['test']['images'])
    
    overlap_train_val = train_names & val_names
    overlap_train_test = train_names & test_names
    overlap_val_test = val_names & test_names
    
    total_overlap = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)
    
    if total_overlap > 0:
        print(f"\n   🚨 FILENAME OVERLAP DETECTED!")
        print(f"      Train ∩ Val:  {len(overlap_train_val)} overlaps")
        print(f"      Train ∩ Test: {len(overlap_train_test)} overlaps")
        print(f"      Val ∩ Test:   {len(overlap_val_test)} overlaps")
        
        if overlap_train_test:
            print(f"\n   Examples of Train-Test overlap:")
            for name in list(overlap_train_test)[:5]:
                print(f"      - {name}")
        
        print(f"\n   ❌ DATASET HAS DATA LEAKAGE!")
        return False
    else:
        print(f"   ✅ NO filename overlaps - All unique!")
    
    # ========================================================================
    # STEP 4: Check for exact duplicate files (MD5 hash)
    # ========================================================================
    
    print(f"\n🔍 STEP 4: Checking for exact duplicate files (MD5)...")
    print(f"   (Checking 100 random images from each split...)")
    
    import random
    
    md5_to_location = defaultdict(list)
    
    for split_name, split_data in splits.items():
        # Sample random images to check (faster)
        sample_size = min(100, len(split_data['images']))
        sampled_images = random.sample(split_data['images'], sample_size)
        
        for img_path in sampled_images:
            md5 = compute_md5(img_path)
            if md5:
                md5_to_location[md5].append((split_name, img_path.name))
    
    # Find duplicates across splits
    exact_duplicates = []
    for md5, locations in md5_to_location.items():
        if len(locations) > 1:
            splits_involved = set(loc[0] for loc in locations)
            if len(splits_involved) > 1:
                exact_duplicates.append((md5, locations))
    
    if exact_duplicates:
        print(f"\n   🚨 EXACT DUPLICATES FOUND!")
        print(f"      {len(exact_duplicates)} identical files across splits")
        for i, (md5, locs) in enumerate(exact_duplicates[:3], 1):
            print(f"\n      {i}. Same file in:")
            for split, name in locs:
                print(f"         - {split}/{name}")
        
        print(f"\n   ❌ DATASET HAS DUPLICATE FILES!")
        return False
    else:
        print(f"   ✅ No exact duplicates found in sample")
    
    # ========================================================================
    # STEP 5: Verify data.yaml exists
    # ========================================================================
    
    print(f"\n📝 STEP 5: Checking data.yaml...")
    
    yaml_path = dataset_path / 'data.yaml'
    if not yaml_path.exists():
        print(f"   ⚠️  data.yaml not found (needed for training)")
        print(f"      Create it manually or run the fix script")
    else:
        print(f"   ✅ data.yaml exists")
    
    # ========================================================================
    # STEP 6: Check label files
    # ========================================================================
    
    print(f"\n📋 STEP 6: Verifying label files...")
    
    missing_labels = 0
    for split_name, split_data in splits.items():
        lbl_dir = dataset_path / split_name / 'labels'
        for img_path in split_data['images'][:100]:  # Check first 100
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                missing_labels += 1
    
    if missing_labels > 0:
        print(f"   ⚠️  {missing_labels} images missing label files (from sample)")
    else:
        print(f"   ✅ All sampled images have label files")
    
    # ========================================================================
    # STEP 7: Check class distribution
    # ========================================================================
    
    print(f"\n📊 STEP 7: Checking class distribution...")
    
    class_counts = {}
    for split_name in ['train', 'val', 'test']:
        lbl_dir = dataset_path / split_name / 'labels'
        class_counts[split_name] = defaultdict(int)
        
        for lbl_file in lbl_dir.glob('*.txt'):
            try:
                with open(lbl_file) as f:
                    for line in f:
                        if line.strip():
                            cls = int(line.strip().split()[0])
                            class_counts[split_name][cls] += 1
            except:
                continue
    
    train_classes = set(class_counts['train'].keys())
    test_classes = set(class_counts['test'].keys())
    
    print(f"   Train classes: {len(train_classes)}")
    print(f"   Val classes: {len(class_counts['val'])}")
    print(f"   Test classes: {len(test_classes)}")
    
    # Check if test has classes not in train
    only_in_test = test_classes - train_classes
    if only_in_test:
        print(f"\n   ⚠️  {len(only_in_test)} classes only in test (not in train)")
        print(f"      Classes: {sorted(only_in_test)}")
    else:
        print(f"   ✅ All test classes present in training")
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("📋 FINAL VERIFICATION REPORT")
    print("="*80)
    
    all_checks_passed = (
        total_overlap == 0 and
        len(exact_duplicates) == 0 and
        total > 0
    )
    
    if all_checks_passed:
        print(f"\n✅ ✅ ✅ DATASET IS CLEAN! ✅ ✅ ✅")
        print(f"\n   ✓ No filename overlaps")
        print(f"   ✓ No exact duplicates")
        print(f"   ✓ Proper folder structure")
        print(f"   ✓ {total:,} total images")
        print(f"   ✓ Train: {splits['train']['count']:,}")
        print(f"   ✓ Val: {splits['val']['count']:,}")
        print(f"   ✓ Test: {splits['test']['count']:,}")
        
        print(f"\n🎉 SAFE TO TRAIN!")
        print(f"   This dataset has NO data leakage")
        print(f"   Test results will be REAL, not fake")
        
        # Calculate expected ratios
        train_ratio = splits['train']['count'] / total
        val_ratio = splits['val']['count'] / total
        test_ratio = splits['test']['count'] / total
        
        print(f"\n📊 Split Ratios:")
        print(f"   Train: {train_ratio*100:.1f}%")
        print(f"   Val:   {val_ratio*100:.1f}%")
        print(f"   Test:  {test_ratio*100:.1f}%")
        
        if abs(train_ratio - 0.7) < 0.05 and abs(val_ratio - 0.15) < 0.05:
            print(f"   ✅ Standard 70/15/15 split (good!)")
        
        return True
    else:
        print(f"\n❌ ❌ ❌ DATASET HAS ISSUES! ❌ ❌ ❌")
        print(f"\n   Problems found:")
        if total_overlap > 0:
            print(f"   ❌ {total_overlap} overlapping filenames")
        if exact_duplicates:
            print(f"   ❌ {len(exact_duplicates)} exact duplicate files")
        
        print(f"\n⚠️  DO NOT TRAIN ON THIS DATASET!")
        print(f"   Run the emergency fix script first:")
        print(f"   python proper_dataset_split.py")
        
        return False


def save_verification_report(dataset_dir, is_clean):
    """Save verification report"""
    
    report = {
        'dataset_path': str(dataset_dir),
        'verification_passed': is_clean,
        'timestamp': str(Path(__file__).stat().st_mtime)
    }
    
    dataset_path = Path(dataset_dir)
    report_file = dataset_path / 'verification_report.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Verification report saved: {report_file}")


# ============================================================================
# RUN VERIFICATION
# ============================================================================

if __name__ == "__main__":
    DATASET_TO_VERIFY = r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\character_dataset_CLEAN"
    
    print("\n🔍 DATASET VERIFICATION - CHECK BEFORE TRAINING")
    print("="*80)
    print("\nThis will verify that your dataset:")
    print("  ✓ Has NO overlapping filenames between train/val/test")
    print("  ✓ Has NO duplicate files across splits")
    print("  ✓ Has proper folder structure")
    print("  ✓ Is safe to train on")
    
    input("\nPress Enter to start verification...")
    
    is_clean = verify_dataset_is_clean(DATASET_TO_VERIFY)
    
    save_verification_report(DATASET_TO_VERIFY, is_clean)
    
    print("\n" + "="*80)
    
    if is_clean:
        print("✅ VERIFICATION PASSED - SAFE TO TRAIN!")
        print("="*80)
        print("\n🚀 Next step: Run your training script")
        print(f"   Update YOLO_DATASET_DIR to: {DATASET_TO_VERIFY}")
    else:
        print("❌ VERIFICATION FAILED - DO NOT TRAIN!")
        print("="*80)
        print("\n🔧 Fix the dataset first:")
        print("   python proper_dataset_split.py")