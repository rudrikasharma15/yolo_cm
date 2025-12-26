"""
MODI WORD RECOGNITION - CLEAN VERSION
======================================
Uses model's built-in class names directly (no manual mapping)
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ===== CONFIGURATION =====
WORD_IMAGES_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\Word")
CHAR_MODEL_PATH = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\RETRAINED_CLEAN_FINAL\character_model\weights\best.pt")
MATRA_MODEL_PATH = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\matra\best.pt")
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\word_recognition_results")

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "segments").mkdir(exist_ok=True)

def preprocess_image(img):
    """Clean preprocessing"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    img = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = 255 - binary
    return binary

def find_shirorekha(img):
    """Find Shirorekha line"""
    h, w = img.shape
    img_inv = 255 - img if np.mean(img) > 127 else img.copy()
    top_region = img_inv[:int(h * 0.35), :]
    h_proj = np.sum(top_region, axis=1)
    h_proj_smooth = gaussian_filter1d(h_proj.astype(float), sigma=3)
    
    if len(h_proj_smooth) == 0 or np.max(h_proj_smooth) == 0:
        return 0, int(h * 0.2), False
    
    shiro_center = np.argmax(h_proj_smooth)
    max_val = h_proj_smooth[shiro_center]
    threshold = max_val * 0.4
    
    shiro_start = shiro_center
    shiro_end = shiro_center
    while shiro_start > 0 and h_proj_smooth[shiro_start] > threshold:
        shiro_start -= 1
    while shiro_end < len(h_proj_smooth) - 1 and h_proj_smooth[shiro_end] > threshold:
        shiro_end += 1
    
    row_span = np.sum(top_region[shiro_center, :] > 0)
    found = (row_span / w) > 0.3
    
    return shiro_start, shiro_end, found

def segment_word(img, shiro_start, shiro_end, shiro_found):
    """Segment word into characters using aggressive valley detection"""
    h, w = img.shape
    img_inv = 255 - img if np.mean(img) > 127 else img.copy()
    
    if shiro_found:
        y_start = shiro_end + 5
        seg_img = img_inv.copy()
        seg_img[max(0, shiro_start-5):min(h, shiro_end+5), :] = 0
    else:
        y_start = int(h * 0.15)
        seg_img = img_inv.copy()
        seg_img[:y_start, :] = 0
    
    v_proj = np.sum(seg_img[y_start:, :], axis=0)
    v_proj_smooth = gaussian_filter1d(v_proj.astype(float), sigma=3)
    
    if np.max(v_proj_smooth) > 0:
        v_proj_norm = v_proj_smooth / np.max(v_proj_smooth)
    else:
        return [], v_proj_smooth
    
    valleys, _ = find_peaks(-v_proj_norm, height=-0.6, distance=15)
    peaks, _ = find_peaks(v_proj_norm, height=0.3, distance=20)
    
    segments = []
    
    if len(peaks) == 0:
        first = np.argmax(v_proj_smooth > np.max(v_proj_smooth) * 0.05)
        last = len(v_proj_smooth) - np.argmax(v_proj_smooth[::-1] > np.max(v_proj_smooth) * 0.05) - 1
        if last > first:
            segments.append({'x_start': max(0, first-10), 'x_end': min(w, last+10), 'y_start': y_start, 'y_end': h})
    elif len(peaks) == 1:
        peak = peaks[0]
        left = peak
        while left > 0 and v_proj_norm[left] > 0.1:
            left -= 1
        right = peak
        while right < len(v_proj_norm) - 1 and v_proj_norm[right] > 0.1:
            right += 1
        
        width = right - left
        if width > 80 and len(valleys) > 0:
            relevant_valleys = [v for v in valleys if left < v < right]
            if len(relevant_valleys) > 0:
                split_points = [left] + sorted(relevant_valleys) + [right]
                for i in range(len(split_points) - 1):
                    segments.append({'x_start': max(0, split_points[i]-10), 'x_end': min(w, split_points[i+1]+10), 'y_start': y_start, 'y_end': h})
            else:
                mid = (left + right) // 2
                segments.append({'x_start': max(0, left-10), 'x_end': min(w, mid+5), 'y_start': y_start, 'y_end': h})
                segments.append({'x_start': max(0, mid-5), 'x_end': min(w, right+10), 'y_start': y_start, 'y_end': h})
        else:
            segments.append({'x_start': max(0, left-10), 'x_end': min(w, right+10), 'y_start': y_start, 'y_end': h})
    else:
        for i, peak in enumerate(peaks):
            if i == 0:
                left = 0
            else:
                prev_peak = peaks[i-1]
                valleys_between = [v for v in valleys if prev_peak < v < peak]
                left = valleys_between[-1] if valleys_between else (prev_peak + peak) // 2
            
            if i == len(peaks) - 1:
                right = w
            else:
                next_peak = peaks[i+1]
                valleys_between = [v for v in valleys if peak < v < next_peak]
                right = valleys_between[0] if valleys_between else (peak + next_peak) // 2
            
            segments.append({'x_start': max(0, left-10), 'x_end': min(w, right+10), 'y_start': y_start, 'y_end': h})
    
    segments = [s for s in segments if (s['x_end'] - s['x_start']) > 20]
    return segments, v_proj_smooth

def prepare_segment(img, region):
    """Prepare segment for recognition"""
    seg = img[region['y_start']:region['y_end'], region['x_start']:region['x_end']]
    if seg.size == 0 or seg.shape[0] < 5 or seg.shape[1] < 5:
        return None
    
    if len(seg.shape) == 3:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    if np.mean(seg) < 127:
        seg = 255 - seg
    
    _, seg = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    border = 60
    seg_bordered = cv2.copyMakeBorder(seg, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    
    h, w = seg_bordered.shape
    max_dim = max(h, w)
    square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = seg_bordered
    
    seg_rgb = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)
    seg_final = cv2.resize(seg_rgb, (640, 640), interpolation=cv2.INTER_AREA)
    return seg_final

def recognize_character(char_model, matra_model, seg_prepared):
    """Recognize character and matra - use model's built-in names"""
    if seg_prepared is None:
        return None, None
    
    # Character recognition
    char_result = None
    for conf in [0.25, 0.20, 0.15, 0.10, 0.05]:
        result = char_model(seg_prepared, verbose=False, conf=conf)[0]
        if len(result.boxes) > 0:
            class_id = int(result.boxes[0].cls[0])
            # Use model's built-in name directly
            char_name = char_model.names[class_id]
            char_result = {'class': char_name, 'confidence': float(result.boxes[0].conf[0])}
            break
    
    # Matra recognition
    matra_result = None
    if matra_model is not None:
        for conf in [0.25, 0.20, 0.15, 0.10]:
            result = matra_model(seg_prepared, verbose=False, conf=conf)[0]
            if len(result.boxes) > 0:
                matra_class = int(result.boxes[0].cls[0])
                matra_name = matra_model.names[matra_class]
                matra_result = {'class': matra_name, 'confidence': float(result.boxes[0].conf[0])}
                break
    
    return char_result, matra_result

def process_word(word_img, word_name, char_model, matra_model):
    """Process one word"""
    print(f"\n{'='*70}")
    print(f"Processing: {word_name}")
    print(f"{'='*70}")
    
    processed = preprocess_image(word_img)
    shiro_start, shiro_end, shiro_found = find_shirorekha(processed)
    shiro_y = (shiro_start + shiro_end) // 2
    segments, v_proj = segment_word(processed, shiro_start, shiro_end, shiro_found)
    
    print(f"\n✓ Found {len(segments)} character segments")
    
    if len(segments) == 0:
        print("✗ No segments found!")
        return []
    
    results = []
    for idx, seg_region in enumerate(segments, 1):
        seg_prepared = prepare_segment(processed, seg_region)
        
        if seg_prepared is not None:
            seg_path = OUTPUT_DIR / "segments" / f"{word_name}_seg{idx}.png"
            cv2.imwrite(str(seg_path), seg_prepared)
            
            char_det, matra_det = recognize_character(char_model, matra_model, seg_prepared)
            
            if char_det:
                display = f"{char_det['class']}"
                if matra_det:
                    display += f" + {matra_det['class']}"
                print(f"  {idx}. {display:20s} (char={char_det['confidence']:.2f})")
            else:
                print(f"  {idx}. UNKNOWN")
        else:
            char_det = None
            matra_det = None
        
        results.append({'position': idx, 'character': char_det, 'matra': matra_det, 'region': seg_region})
    
    create_visualization(word_img, processed, shiro_y, shiro_found, segments, results, word_name, v_proj)
    return results

def create_visualization(original, processed, shiro_y, shiro_found, segments, results, word_name, v_proj):
    """Create result visualization"""
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Word', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    vis_seg = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    if shiro_found:
        cv2.line(vis_seg, (0, shiro_y), (vis_seg.shape[1], shiro_y), (255, 0, 0), 2)
    for idx, seg in enumerate(segments, 1):
        cv2.rectangle(vis_seg, (seg['x_start'], seg['y_start']), (seg['x_end'], seg['y_end']), (0, 255, 0), 3)
        cv2.putText(vis_seg, str(idx), (seg['x_start']+10, seg['y_start']+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    ax2.imshow(vis_seg)
    ax2.set_title(f'Segmentation: {len(segments)} characters', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    if v_proj is not None and len(v_proj) > 0:
        ax3.plot(v_proj, linewidth=2, color='blue')
        ax3.set_title('Vertical Projection', fontsize=12)
        ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    result_text = f"WORD: {word_name}\n{'='*50}\n\nRECOGNIZED:\n\n"
    pred_chars = []
    for res in results:
        if res['character']:
            char = res['character']['class']
            conf = res['character']['confidence']
            if res['matra']:
                display = f"{char}+{res['matra']['class']}"
            else:
                display = char
            pred_chars.append(display)
            result_text += f"  {res['position']}. {display:15s} ({conf:.2f})\n"
        else:
            pred_chars.append("?")
            result_text += f"  {res['position']}. UNKNOWN\n"
    
    result_text += f"\n{'='*50}\nWORD: {' '.join(pred_chars)}\n"
    
    ax4.text(0.1, 0.5, result_text, ha='left', va='center', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(f'Modi Word Recognition: {word_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    result_path = OUTPUT_DIR / f"{word_name}_result.png"
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: {result_path.name}")

def main():
    print("\n"+"="*70)
    print("MODI WORD RECOGNITION")
    print("="*70)
    
    word_files = list(WORD_IMAGES_DIR.glob("*.png")) + list(WORD_IMAGES_DIR.glob("*.jpg"))
    if not word_files:
        print(f"\n❌ No images in: {WORD_IMAGES_DIR}")
        return
    
    print(f"\n✅ Found {len(word_files)} word images")
    
    print("\n🔧 Loading models...")
    char_model = YOLO(str(CHAR_MODEL_PATH))
    print(f"✅ Character model loaded ({len(char_model.names)} classes)")
    
    matra_model = None
    if MATRA_MODEL_PATH.exists():
        matra_model = YOLO(str(MATRA_MODEL_PATH))
        print(f"✅ Matra model loaded ({len(matra_model.names)} classes)")
    else:
        print("⚠️  No matra model found")
    
    all_results = []
    for word_file in sorted(word_files)[:10]:
        word_img = cv2.imread(str(word_file))
        if word_img is None:
            continue
        if len(word_img.shape) == 3:
            word_img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        
        word_name = word_file.stem
        results = process_word(word_img, word_name, char_model, matra_model)
        if results:
            all_results.append({'word': word_name, 'results': results})
    
    print("\n"+"="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Processed: {len(all_results)} words")
    total_chars = sum(len(r['results']) for r in all_results)
    print(f"Total characters: {total_chars}")
    print(f"\n📁 Results: {OUTPUT_DIR}")
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()