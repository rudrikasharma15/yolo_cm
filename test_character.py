"""
DATA LEAKAGE DETECTOR
=====================
Check if test images are too similar to training images
"""

import cv2
import numpy as np
from pathlib import Path
import hashlib
from collections import defaultdict
import imagehash
from PIL import Image

def compute_image_hash(img_path):
    """Compute perceptual hash of image"""
    try:
        img = Image.open(img_path)
        return str(imagehash.phash(img))
    except:
        return None

def compute_md5(img_path):
    """Compute MD5 hash (exact duplicates)"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def check_data_leakage(base_path):
    """
    Check for data leakage between train/val/test sets
    """
    
    print("="*80)
    print("🔍 DATA LEAKAGE DETECTION")
    print("="*80)
    
    base = Path(base_path)
    
    # Collect images
    sets = {}
    for split in ['train', 'val', 'test']:
        img_dir = base / split / 'images'
        if img_dir.exists():
            sets[split] = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            print(f"✅ {split.upper()}: {len(sets[split])} images")
    
    if 'test' not in sets or 'train' not in sets:
        print("\n❌ Missing train or test folders!")
        return
    
    print("\n" + "="*80)
    print("CHECKING FOR DUPLICATES...")
    print("="*80)
    
    # Check 1: Exact duplicates (MD5)
    print("\n1️⃣  Checking for EXACT duplicates (MD5 hash)...")
    
    md5_hashes = defaultdict(list)
    
    for split, images in sets.items():
        print(f"   Computing MD5 for {split}...")
        for img_path in images:
            md5 = compute_md5(img_path)
            if md5:
                md5_hashes[md5].append((split, img_path.name))
    
    # Find duplicates across splits
    exact_duplicates = []
    for md5, locations in md5_hashes.items():
        if len(locations) > 1:
            splits_involved = set(loc[0] for loc in locations)
            if len(splits_involved) > 1:
                exact_duplicates.append((md5, locations))
    
    if exact_duplicates:
        print(f"\n   🚨 FOUND {len(exact_duplicates)} EXACT DUPLICATES across splits!")
        for i, (md5, locs) in enumerate(exact_duplicates[:5], 1):
            print(f"      {i}. Same image in:")
            for split, name in locs:
                print(f"         - {split}/{name}")
    else:
        print("   ✅ No exact duplicates found")
    
    # Check 2: Perceptual duplicates (similar images)
    print("\n2️⃣  Checking for SIMILAR images (perceptual hash)...")
    
    try:
        import imagehash
        from PIL import Image
        
        phashes = defaultdict(list)
        
        for split, images in sets.items():
            print(f"   Computing perceptual hash for {split}...")
            for img_path in images[:500]:  # Sample for speed
                phash = compute_image_hash(img_path)
                if phash:
                    phashes[phash].append((split, img_path.name))
        
        similar_images = []
        for phash, locations in phashes.items():
            if len(locations) > 1:
                splits_involved = set(loc[0] for loc in locations)
                if len(splits_involved) > 1:
                    similar_images.append((phash, locations))
        
        if similar_images:
            print(f"\n   🚨 FOUND {len(similar_images)} SIMILAR image groups across splits!")
            for i, (phash, locs) in enumerate(similar_images[:5], 1):
                print(f"      {i}. Similar images in:")
                for split, name in locs:
                    print(f"         - {split}/{name}")
        else:
            print("   ✅ No perceptually similar images found")
            
    except ImportError:
        print("   ⚠️  imagehash not installed, skipping perceptual check")
        print("   Install with: pip install imagehash pillow")
    
    # Check 3: Label distribution comparison
    print("\n3️⃣  Checking LABEL DISTRIBUTION...")
    
    label_counts = {}
    for split in ['train', 'val', 'test']:
        label_dir = base / split / 'labels'
        if label_dir.exists():
            counts = defaultdict(int)
            for label_file in label_dir.glob('*.txt'):
                try:
                    with open(label_file) as f:
                        for line in f:
                            cls = int(line.strip().split()[0])
                            counts[cls] += 1
                except:
                    continue
            label_counts[split] = dict(counts)
    
    if 'train' in label_counts and 'test' in label_counts:
        train_classes = set(label_counts['train'].keys())
        test_classes = set(label_counts['test'].keys())
        
        print(f"\n   Train classes: {len(train_classes)}")
        print(f"   Test classes: {len(test_classes)}")
        
        only_in_test = test_classes - train_classes
        if only_in_test:
            print(f"\n   🚨 {len(only_in_test)} classes ONLY in test (never seen in training)!")
            print(f"      Classes: {only_in_test}")
        else:
            print("   ✅ All test classes were in training")
    
    # Check 4: Image name patterns
    print("\n4️⃣  Checking IMAGE NAME PATTERNS...")
    
    train_stems = set(p.stem for p in sets['train'])
    test_stems = set(p.stem for p in sets['test'])
    
    # Check for obvious naming overlaps
    overlap_names = train_stems & test_stems
    if overlap_names:
        print(f"\n   🚨 {len(overlap_names)} images with SAME NAME in train and test!")
        print(f"      Examples: {list(overlap_names)[:5]}")
    else:
        print("   ✅ No overlapping filenames")
    
    # Check for sequential naming (might indicate splits from same source)
    def extract_number(stem):
        import re
        numbers = re.findall(r'\d+', stem)
        return int(numbers[-1]) if numbers else None
    
    train_nums = [extract_number(s) for s in list(train_stems)[:100]]
    test_nums = [extract_number(s) for s in list(test_stems)[:100]]
    train_nums = [n for n in train_nums if n is not None]
    test_nums = [n for n in test_nums if n is not None]
    
    if train_nums and test_nums:
        train_range = (min(train_nums), max(train_nums))
        test_range = (min(test_nums), max(test_nums))
        
        print(f"\n   Train image numbers: {train_range[0]} to {train_range[1]}")
        print(f"   Test image numbers: {test_range[0]} to {test_range[1]}")
        
        # Check for overlap
        if not (test_range[1] < train_range[0] or test_range[0] > train_range[1]):
            print("   ⚠️  Number ranges overlap - might indicate same source images")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    issues_found = 0
    
    if exact_duplicates:
        issues_found += 1
        print(f"\n❌ ISSUE 1: {len(exact_duplicates)} exact duplicates across splits")
        print("   → Model has SEEN these test images during training!")
    
    if similar_images:
        issues_found += 1
        print(f"\n❌ ISSUE 2: {len(similar_images)} similar images across splits")
        print("   → Test images are too similar to training images")
    
    if overlap_names:
        issues_found += 1
        print(f"\n❌ ISSUE 3: {len(overlap_names)} overlapping filenames")
        print("   → Same images in both train and test")
    
    if only_in_test:
        issues_found += 1
        print(f"\n❌ ISSUE 4: {len(only_in_test)} classes only in test set")
        print("   → Model never learned these classes")
    
    if issues_found == 0:
        print("\n✅ No obvious data leakage detected!")
        print("\n💡 But 95% accuracy with wrong predictions suggests:")
        print("   1. Model memorizing image backgrounds/artifacts")
        print("   2. Labels might be swapped/incorrect")
        print("   3. Test on COMPLETELY NEW images to verify")
    else:
        print(f"\n🚨 FOUND {issues_found} POTENTIAL ISSUES!")
        print("\n🔧 SOLUTIONS:")
        print("   1. Re-split your dataset properly")
        print("   2. Ensure NO overlap between train/val/test")
        print("   3. Use random splitting with seed")
        print("   4. Test on COMPLETELY NEW images from different source")
    
    return {
        'exact_duplicates': len(exact_duplicates) if exact_duplicates else 0,
        'similar_images': len(similar_images) if similar_images else 0,
        'overlap_names': len(overlap_names) if overlap_names else 0,
        'classes_only_in_test': len(only_in_test) if only_in_test else 0
    }


if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\character_output_1000\yolo_dataset"
    
    results = check_data_leakage(DATASET_PATH)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Test on BRAND NEW images (never seen before)")
    print("2. Manually check 20 random test images")
    print("3. Compare predictions with actual labels")
    print("4. If still wrong → dataset labels are incorrect")