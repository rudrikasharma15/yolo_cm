"""
🔥 ENHANCED STEP 1: SMART MATRA EXTRACTION 🔥
===============================================

IMPROVEMENTS:
1. Extracts MORE samples for minority classes (aa_matra, anusvara)
2. Balanced dataset with stratified sampling
3. Quality-based selection (clearer images)
4. Better distribution across folders

This will give you a better initial model that can detect ALL matras!
"""

import cv2
import shutil
from pathlib import Path
import random
from collections import defaultdict
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_DATASET = r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\Dataset_Modi"
OUTPUT_DIR = r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\matra_extraction_ENHANCED"


# SMART SAMPLING: More samples for classes the model struggles with
SAMPLES_PER_MATRA = {
    "aa_matra": 100,     # 3x more (model struggles - 11.5% coverage)
    "i_matra": 50,       # Less (model is great - 91% coverage)
    "u_matra": 60,       # Medium
    "e_matra": 60,       # Medium
    "ai_matra": 50,      # Less (model is good - 86% coverage)
    "o_matra": 80,       # More (model struggles - 45%)
    "au_matra": 60,      # Medium
    "anusvara": 150,     # 5x more! (model can't detect - 0% coverage)
}

# Quality filtering
USE_QUALITY_FILTERING = True
MIN_IMAGE_SIZE = 20  # Minimum width/height in pixels

# ============================================================================
# 8 MODI MATRAS - EXACT MAPPING
# ============================================================================

MATRA_MAPPINGS = {
    "aa_matra": {
        "class_id": 0,
        "symbol": "ा",
        "description": "aa-matra (vertical line on right)",
        "folder_endings": ["AA", "AA-kaku", "AA-KHAR", "GHAA", "CHAA", "CHHAA", "JAA", "ZAA", 
                          "TRAA", "TTAA", "BAA", "BHAA", "DAA", "DHAA", "HAA", "LAA", "MAA", 
                          "NAA", "PAA", "PHAA", "RAA", "SAA", "SHAA", "TAA-talwar", "THAA", 
                          "THHAA", "VAA", "YAA", "ALA-kamal", "aa-kshatriy", "DNYAa"]
    },
    "i_matra": {
        "class_id": 1,
        "symbol": "ि",
        "description": "i-matra (curve on left)",
        "folder_endings": ["KI-kiran", "KHI", "GI", "GHI", "CHI", "CHHI", "JI", "ZI", "TRI", 
                          "TTI", "BI", "BHI", "bhai", "DI", "DHI", "DHHI-dhag", "HI", "LI", 
                          "MI", "NI", "PI", "PHI", "RI", "SHI", "SI", "TI-talwar", "THI", 
                          "THHI", "VI", "YI", "ALI-kamal", "i-kshatriy", "DNYi"]
    },
    "u_matra": {
        "class_id": 2,
        "symbol": "ु",
        "description": "u-matra (hook below)",
        "folder_endings": ["KU-kunfu", "KHU", "GU", "GHU", "CHU", "CHHU", "JU", "ZU", "TRU", 
                          "TTU", "BU", "BHU", "DU", "DHU", "DHHU-dhag", "HU", "LU", "MU", 
                          "NU", "PU", "PHU", "RU", "SHU", "SU", "TU-talwar", "THU", "THHU", 
                          "VU", "YU", "ALU-kamal", "u-kshatriy", "DNYu"]
    },
    "e_matra": {
        "class_id": 3,
        "symbol": "े",
        "description": "e-matra (slant on top)",
        "folder_endings": ["KE-kedar", "KHE", "GE", "GHE", "CHE", "CHHE", "JE", "ZE", "TRE", 
                          "TTE", "BE", "BHE", "DE", "DHE", "DHHE-dhag", "HE", "LE", "ME", 
                          "NE", "PE", "PHE", "RE", "SE", "SHE", "she", "TE--talwar", "THE", 
                          "THHE", "VE", "YE", "ALE-kamal", "e-kshatriy", "DNYAe"]
    },
    "ai_matra": {
        "class_id": 4,
        "symbol": "ै",
        "description": "ai-matra (double slant on top)",
        "folder_endings": ["KAI-kailas", "KHAI", "GAI", "GHAI", "CHAI", "CHHAI", "JAI", "ZAI", 
                          "TRAI", "TTAI", "BAI", "DAI", "DHAI", "DHHAI-dhag", "HAI", "LAI", 
                          "MAI", "NAI", "PAI", "PHAI", "RAI", "SAI", "SHAI", "shai", 
                          "TAI-talwar", "THAI", "THHAI", "VAI", "YAI", "ALAI-kamal", 
                          "ai-kshatriy", "DNYAai"]
    },
    "o_matra": {
        "class_id": 5,
        "symbol": "ो",
        "description": "o-matra (right side curve)",
        "folder_endings": ["KO-komal", "KHO", "GO", "GHO", "CHO", "CHHO", "JO", "ZO", "TRO", 
                          "TTO", "BO", "BHO", "DO", "DHO", "DHHO-dhag", "HO", "LO", "MO", 
                          "NO", "PO", "PHO", "RO", "SHO", "sho", "SO", "TO-talwar", "THO", 
                          "THHO", "VO", "YO", "ALO-kamal", "o-kshatriy", "DNYo"]
    },
    "au_matra": {
        "class_id": 6,
        "symbol": "ौ",
        "description": "au-matra (double curve on right)",
        "folder_endings": ["KAU-kaul", "KHAU", "GAU", "GHAU", "CHAO", "CHHAO", "JAO", "ZAO", 
                          "TRAO", "TTAO", "BAO", "BHAO", "DAU", "DAO-daut", "DHAO", 
                          "DHHAO-dhag", "HAO", "LAO", "MAO", "NAO", "PAO", "PHAO", "RAO", 
                          "SAO", "SHAO", "shau", "TAO-talwar", "THAO", "THHAO", "VAO", "YAO", 
                          "ALAO-kamal", "au-kshatriy", "DNYau"]
    },
    "anusvara": {
        "class_id": 7,
        "symbol": "ं",
        "description": "anusvara (dot on top)",
        "folder_endings": ["KAN-kangan", "KHAN", "GAM", "GHAM", "CHAM", "CHHAm", "JAM", "ZAM", 
                          "TRAM", "TTAM", "BAM", "BHAM", "DAM", "DHAM", "DHHAM-dhag", "HAM", 
                          "LAM", "MAM", "NAM", "PAM", "PHAM", "RAM", "SAM", "SHAM", "sham", 
                          "TAM-talwar", "THAM", "THHAM", "VAM", "YAM", "ALAM-kamal", 
                          "nm-kshatriy", "DNYAm"]
    }
}

# ============================================================================
# ENHANCED EXTRACTOR WITH QUALITY FILTERING
# ============================================================================

class EnhancedMatraExtractor:
    """Extract matras with smart sampling and quality filtering"""
    
    def __init__(self, source, output):
        self.source = Path(source)
        self.output = Path(output)
        self.unlabeled = self.output / "unlabeled_matras"
        self.unlabeled.mkdir(parents=True, exist_ok=True)
        self.stats = {}
        
        # Get all folders in dataset
        self.all_folders = [f for f in self.source.iterdir() if f.is_dir()]
        print(f"\n📂 Found {len(self.all_folders)} folders in dataset")
    
    def calculate_image_quality(self, img_path):
        """Calculate image quality score based on clarity and size"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return 0
            
            # Check size
            h, w = img.shape[:2]
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                return 0
            
            # Calculate Laplacian variance (measure of blur/clarity)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize score (higher = better quality)
            quality_score = min(laplacian_var / 100, 10)  # Cap at 10
            
            return quality_score
            
        except Exception as e:
            return 0
    
    def extract_matra(self, matra_name, matra_config, n_samples):
        """Extract one matra using exact folder matching with quality filtering"""
        
        class_id = matra_config['class_id']
        symbol = matra_config['symbol']
        desc = matra_config['description']
        folder_endings = matra_config['folder_endings']
        
        print(f"\n{'='*70}")
        print(f"📌 EXTRACTING: {matra_name} (Class {class_id})")
        print(f"   Symbol: {symbol}")
        print(f"   Description: {desc}")
        print(f"   Target samples: {n_samples}")
        print(f"   Looking for {len(folder_endings)} specific folders")
        print(f"{'='*70}")
        
        # Create output folder
        matra_folder = self.unlabeled / f"{class_id:02d}_{matra_name}"
        matra_folder.mkdir(exist_ok=True)
        
        # Find matching folders
        matched_folders = []
        
        for folder in self.all_folders:
            folder_name = folder.name
            
            # Skip standalone vowel folders (1-10)
            if folder_name in ['1 a-ananas', '2 aa-aai', '3 i-imarat', '4 u-ukhal', 
                               '5 e-edka', '6 ai-airan', '7 o-odha', '8 au-aushadh', 
                               '9 nm-angthi', '10 ahaa']:
                continue
            
            # Check if folder name ends with any of the specified endings
            for ending in folder_endings:
                if folder_name.endswith(ending) or folder_name == ending:
                    matched_folders.append(folder)
                    break
        
        if not matched_folders:
            print(f"   ⚠️  NO MATCHING FOLDERS FOUND!")
            self.stats[matra_name] = 0
            return 0
        
        print(f"\n   ✅ Matched {len(matched_folders)} folders:")
        for f in matched_folders[:5]:
            print(f"      - {f.name}")
        if len(matched_folders) > 5:
            print(f"      ... and {len(matched_folders)-5} more")
        
        # Collect ALL images with quality scores
        all_images = []
        
        print(f"\n   🔍 Collecting images and calculating quality scores...")
        for folder in matched_folders:
            images = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
            
            for img_path in images:
                if USE_QUALITY_FILTERING:
                    quality = self.calculate_image_quality(img_path)
                    if quality > 0:
                        all_images.append((img_path, quality))
                else:
                    all_images.append((img_path, 1.0))
        
        print(f"\n   📊 Total images found: {len(all_images)}")
        
        if not all_images:
            print(f"   ⚠️  NO IMAGES IN MATCHED FOLDERS!")
            self.stats[matra_name] = 0
            return 0
        
        # Sort by quality (best first)
        if USE_QUALITY_FILTERING:
            all_images.sort(key=lambda x: x[1], reverse=True)
            print(f"   🎯 Images sorted by quality (selecting best)")
        
        # STRATIFIED SAMPLING: Sample evenly from all folders
        samples_per_folder = max(1, n_samples // len(matched_folders))
        sampled = []
        
        # First pass: get samples_per_folder from each folder
        for folder in matched_folders:
            folder_images = [(img, q) for img, q in all_images if img.parent == folder]
            n_from_folder = min(len(folder_images), samples_per_folder)
            sampled.extend(folder_images[:n_from_folder])
        
        # Second pass: fill remaining quota with best quality images
        remaining = n_samples - len(sampled)
        if remaining > 0:
            available = [img for img in all_images if img not in sampled]
            sampled.extend(available[:remaining])
        
        # Shuffle to mix folders
        random.shuffle(sampled)
        
        # Take final n_samples
        final_samples = sampled[:n_samples]
        
        # Copy to output
        print(f"\n   💾 Saving images...")
        for idx, (img_path, quality) in enumerate(final_samples):
            dest = matra_folder / f"{matra_name}_{idx:04d}.png"
            shutil.copy2(img_path, dest)
        
        print(f"\n   ✅ EXTRACTED: {len(final_samples)} images")
        
        if USE_QUALITY_FILTERING and len(final_samples) > 0:
            avg_quality = sum(q for _, q in final_samples) / len(final_samples)
            print(f"   📈 Average quality score: {avg_quality:.2f}")
        
        self.stats[matra_name] = len(final_samples)
        return len(final_samples)
    
    def extract_all(self):
        """Extract all 8 matras with smart sampling"""
        
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🔥 ENHANCED MODI MATRA EXTRACTION - SMART SAMPLING 🔥            ║
║                                                                      ║
║  ✅ More samples for minority classes (aa_matra, anusvara)          ║
║  ✅ Quality-based selection (clearer images)                        ║
║  ✅ Stratified sampling across folders                              ║
║  ✅ Balanced dataset for better training                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        
        print(f"\n📂 Source: {self.source}")
        print(f"📂 Output: {self.output}")
        print(f"🎯 Quality filtering: {'ENABLED' if USE_QUALITY_FILTERING else 'DISABLED'}")
        
        print(f"\n📊 Target samples per matra:")
        for matra_name, count in SAMPLES_PER_MATRA.items():
            print(f"   {matra_name:<15} → {count:>3} images")
        
        total = 0
        for matra_name, config in MATRA_MAPPINGS.items():
            n_samples = SAMPLES_PER_MATRA[matra_name]
            count = self.extract_matra(matra_name, config, n_samples)
            total += count
        
        # Summary
        print(f"\n\n{'='*70}")
        print("✅ EXTRACTION COMPLETE!")
        print(f"{'='*70}")
        print(f"\n{'Class':<8} {'Matra':<15} {'Symbol':<8} {'Target':<10} {'Extracted':<10}")
        print("-" * 70)
        
        for matra_name, config in MATRA_MAPPINGS.items():
            count = self.stats.get(matra_name, 0)
            target = SAMPLES_PER_MATRA[matra_name]
            print(f"{config['class_id']:<8} {matra_name:<15} {config['symbol']:<8} {target:<10} {count:<10}")
        
        print("-" * 70)
        print(f"{'TOTAL':<8} {'8 matras':<15} {'':<8} {sum(SAMPLES_PER_MATRA.values()):<10} {total:<10}")
        
        print(f"\n\n📂 Output: {self.unlabeled}")
        
        if total == 0:
            print("\n❌ ERROR: No images extracted!")
        else:
            print(f"\n📝 NEXT STEP:")
            print(f"   Install LabelImg: pip install labelImg")
            print(f"   Start labeling: labelImg {self.unlabeled}")
            print(f"\n⏱️  Estimated time: ~4-5 hours ({total} images)")
            print(f"\n💡 PRIORITY LABELING ORDER:")
            print(f"   1. anusvara (150 images) - CRITICAL!")
            print(f"   2. aa_matra (100 images) - Important")
            print(f"   3. o_matra (80 images) - Important")
            print(f"   4. Rest of the matras")
        
        # Create enhanced labeling guide
        self.create_enhanced_guide()
        
        return total
    
    def create_enhanced_guide(self):
        """Create comprehensive labeling guide"""
        
        guide = f"""
╔══════════════════════════════════════════════════════════════════════╗
║      ENHANCED MODI MATRA LABELING GUIDE - 8 CLASSES                 ║
╚══════════════════════════════════════════════════════════════════════╝

TOTAL IMAGES: {sum(self.stats.values())}
TOOL: LabelImg (YOLO format)
COMMAND: labelImg {self.unlabeled}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8 MATRA CLASSES (ENHANCED SAMPLING)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for matra_name, config in MATRA_MAPPINGS.items():
            count = self.stats.get(matra_name, 0)
            guide += f"Class {config['class_id']}: {matra_name:<15} ({config['symbol']})  - {count:>3} images - {config['description']}\n"
        
        guide += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY LABELING (LABEL THESE FIRST!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔥 HIGH PRIORITY (Model struggles with these):
   1. anusvara ({self.stats.get('anusvara', 0)} images) - Current model has 0% detection!
   2. aa_matra ({self.stats.get('aa_matra', 0)} images) - Current model has 11.5% detection
   3. o_matra ({self.stats.get('o_matra', 0)} images) - Current model has 45% detection

✅ LOWER PRIORITY (Model already performs well):
   4. Rest of the matras (i_matra, ai_matra, etc.)

WHY THIS MATTERS:
- Better labels for weak classes → Better model → Better auto-labeling in Step 3
- You'll get more auto-labeled images in the next iteration!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL LABELING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ Box ONLY the matra (NOT the base character)
   - Image shows "का" (ka + aa)  → Box only the "ा" part
   - Image shows "कं" (kan)        → Box only the "ं" dot (ZOOM IN!)

2. ✅ Draw TIGHT boxes
   - Include all pixels of the matra stroke
   - No extra white space
   - For anusvara (ं), ZOOM IN and draw precise box around the dot

3. ✅ Use EXACT label names
   - aa_matra, i_matra, u_matra, e_matra, ai_matra, o_matra, au_matra, anusvara
   - Use lowercase with underscore

4. ✅ LabelImg Settings:
   - Format: YOLO (press Ctrl+Y or Cmd+Y)
   - Save: Ctrl+S (Cmd+S on Mac) after EACH image
   - Next: D key
   - Previous: A key
   - Zoom: Scroll wheel (IMPORTANT for anusvara!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL ATTENTION: ANUSVARA (ं)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Anusvara is the HARDEST to detect because it's TINY!

TIPS:
1. ALWAYS zoom in (scroll wheel or +/- keys)
2. Look for the small dot/circle on top of the character
3. Draw a SMALL, TIGHT box around JUST the dot
4. Don't include any part of the base character
5. Even if the dot is tiny (3-5 pixels), still label it!

EXAMPLE:
   कं  →  Box ONLY the "ं" dot on top (not the "क")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED ORDER:

1. Start with PRIORITY classes first:
   labelImg {self.unlabeled}/07_anusvara/
   labelImg {self.unlabeled}/00_aa_matra/
   labelImg {self.unlabeled}/05_o_matra/

2. Then do the rest:
   labelImg {self.unlabeled}/

3. For each image:
   - Draw box around ONLY the matra
   - Select correct class
   - Save (Ctrl+S)
   - Next image (D key)

4. Take breaks every 50 images!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ GOOD LABELS:
- Tight boxes with minimal white space
- All matra pixels included
- Consistent across similar images

❌ BAD LABELS:
- Box includes base character
- Box too large with lots of white space
- Matra pixels cut off

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        guide_file = self.output / "ENHANCED_LABELING_GUIDE.txt"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        print(f"\n📖 Enhanced labeling guide saved: {guide_file}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run enhanced matra extraction"""
    
    extractor = EnhancedMatraExtractor(
        source=SOURCE_DATASET,
        output=OUTPUT_DIR
    )
    
    total = extractor.extract_all()
    
    if total > 0:
        print(f"\n\n🎉 SUCCESS! {total} images extracted with smart sampling")
        print(f"\n👉 START LABELING (PRIORITY ORDER):")
        print(f"   1. Label anusvara first: labelImg {extractor.unlabeled}/07_anusvara/")
        print(f"   2. Label aa_matra: labelImg {extractor.unlabeled}/00_aa_matra/")
        print(f"   3. Label o_matra: labelImg {extractor.unlabeled}/05_o_matra/")
        print(f"   4. Label rest: labelImg {extractor.unlabeled}/")
        print(f"\n⏱️  Time estimate: ~4-5 hours")
        print(f"\n📖 Read the guide: {extractor.output}/ENHANCED_LABELING_GUIDE.txt")
        print(f"\n🎯 GOAL: After retraining with these labels,")
        print(f"   your model will detect anusvara and aa_matra much better!")

if __name__ == "__main__":
    main()