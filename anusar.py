"""
🔥 QUICK ANUSVARA EXTRACTOR 🔥
================================

Extracts 150 anusvara images for manual labeling.
This will fix the 0% detection rate!
"""

import cv2
import shutil
from pathlib import Path
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_DATASET = "/Users/applemaair/Desktop/modi_final/Dataset_Modi"
OUTPUT_DIR = "/Users/applemaair/Desktop/modi_final/anusvara_manual_labeling"
NUM_SAMPLES = 150

# Anusvara folder endings (same as Step 1)
ANUSVARA_FOLDERS = [
    "KAN-kangan", "KHAN", "GAM", "GHAM", "CHAM", "CHHAm", "JAM", "ZAM", 
    "TRAM", "TTAM", "BAM", "BHAM", "DAM", "DHAM", "DHHAM-dhag", "HAM", 
    "LAM", "MAM", "NAM", "PAM", "PHAM", "RAM", "SAM", "SHAM", "sham", 
    "TAM-talwar", "THAM", "THHAM", "VAM", "YAM", "ALAM-kamal", 
    "nm-kshatriy", "DNYAm"
]

# ============================================================================
# EXTRACTOR
# ============================================================================

def extract_anusvara():
    """Extract anusvara images for manual labeling"""
    
    source = Path(SOURCE_DATASET)
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           🔥 ANUSVARA MANUAL LABELING EXTRACTOR 🔥                   ║
║                                                                      ║
║  Extracting 150 anusvara images to fix 0% detection rate!           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📂 Source: {source}")
    print(f"📂 Output: {output}")
    print(f"🎯 Target: {NUM_SAMPLES} images\n")
    
    # Get all folders
    all_folders = [f for f in source.iterdir() if f.is_dir()]
    
    # Find anusvara folders
    matched_folders = []
    for folder in all_folders:
        folder_name = folder.name
        
        # Skip standalone vowels
        if folder_name in ['1 a-ananas', '2 aa-aai', '3 i-imarat', '4 u-ukhal', 
                           '5 e-edka', '6 ai-airan', '7 o-odha', '8 au-aushadh', 
                           '9 nm-angthi', '10 ahaa']:
            continue
        
        # Check endings
        for ending in ANUSVARA_FOLDERS:
            if folder_name.endswith(ending) or folder_name == ending:
                matched_folders.append(folder)
                break
    
    print(f"✅ Found {len(matched_folders)} anusvara folders:")
    for f in matched_folders[:10]:
        print(f"   - {f.name}")
    if len(matched_folders) > 10:
        print(f"   ... and {len(matched_folders)-10} more\n")
    
    # Collect all images
    all_images = []
    for folder in matched_folders:
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        all_images.extend(images)
    
    print(f"\n📊 Total anusvara images found: {len(all_images)}")
    
    if len(all_images) < NUM_SAMPLES:
        print(f"⚠️  Only {len(all_images)} available (less than {NUM_SAMPLES})")
        n_samples = len(all_images)
    else:
        n_samples = NUM_SAMPLES
    
    # Sample randomly
    sampled = random.sample(all_images, n_samples)
    
    # Copy to output
    print(f"\n💾 Copying {n_samples} images...\n")
    for idx, img_path in enumerate(sampled):
        dest = output / f"anusvara_{idx:04d}.png"
        shutil.copy2(img_path, dest)
    
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"\n📂 Images saved to: {output}")
    print(f"📊 Total extracted: {n_samples} images")
    
    # Create labeling instructions
    instructions = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              ANUSVARA LABELING INSTRUCTIONS                          ║
╚══════════════════════════════════════════════════════════════════════╝

LOCATION: {output}
IMAGES: {n_samples} files
TOOL: LabelImg

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSTRUCTIONS FOR ANUSVARA (ं)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Anusvara is a TINY DOT on top of characters. This is why your model can't
detect it - it needs very precise labels!

STEPS:
1. Install LabelImg (if not installed):
   pip install labelImg

2. Open LabelImg:
   labelImg {output}

3. Set format to YOLO:
   Press Ctrl+Y (or Cmd+Y on Mac)

4. For EACH image:
   a. ZOOM IN! (Scroll wheel or +/- keys)
   b. Look for the tiny dot (ं) on top of the character
   c. Draw a SMALL, TIGHT box around ONLY the dot
   d. Label it as: anusvara
   e. Save (Ctrl+S or Cmd+S)
   f. Next image (D key)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABELING TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ DO:
- Always zoom in first
- Draw tiny, tight boxes
- Include ALL pixels of the dot
- Label ONLY the dot, not the character below

❌ DON'T:
- Don't include the base character
- Don't make boxes too large
- Don't skip images where the dot is very small

EXAMPLE:
   कं  →  Box ONLY the "ं" dot on top

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIME ESTIMATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{n_samples} images × 30 seconds each ≈ {n_samples * 30 / 60:.0f} minutes

Take breaks every 50 images!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AFTER LABELING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Once done, you'll have:
- anusvara_0000.png + anusvara_0000.txt
- anusvara_0001.png + anusvara_0001.txt
- ... and so on

These will be combined with your auto-labeled data in Step 4!
    """
    
    instructions_file = output / "LABELING_INSTRUCTIONS.txt"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\n📖 Instructions saved: {instructions_file}")
    print(f"\n👉 NEXT STEPS:")
    print(f"   1. Install LabelImg: pip install labelImg")
    print(f"   2. Start labeling: labelImg {output}")
    print(f"   3. Label all {n_samples} images (zoom in for the tiny dots!)")
    print(f"   4. Time estimate: ~{n_samples * 30 / 60:.0f} minutes")
    print(f"\n💡 After labeling, proceed to Step 4 to train final model!")

if __name__ == "__main__":
    extract_anusvara()