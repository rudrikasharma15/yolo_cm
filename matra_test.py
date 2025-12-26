"""
TEST MATRA MODEL - ACCURACY & METRICS
======================================
Test your matra detection model and show detailed results
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict
import json

print("="*70)
print("TESTING MATRA MODEL")
print("="*70)

# Configuration
MATRA_MODEL_PATH = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\matra\best.pt")
MATRA_DATA_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\matra\step4_final_model\yolo_dataset")
OUTPUT_DIR = Path(r"C:\Users\admin\Desktop\MODI_FINAL\modi_final\matra_test_results")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load model
print("\n📥 Loading matra model...")
model = YOLO(str(MATRA_MODEL_PATH))
print(f"✅ Model loaded with {len(model.names)} classes")

# Show matra classes
print(f"\n📋 Matra classes:")
for i, name in model.names.items():
    print(f"   {i}: {name}")

# Find test images
TEST_IMAGES = MATRA_DATA_DIR / "test" / "images"
TEST_LABELS = MATRA_DATA_DIR / "test" / "labels"

if not TEST_IMAGES.exists():
    print(f"\n❌ Test images not found at: {TEST_IMAGES}")
    print("Please check the path!")
    exit()

test_images = list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
print(f"\n📊 Found {len(test_images)} test images")

if len(test_images) == 0:
    print("❌ No test images found!")
    exit()

# Test the model
print(f"\n{'='*70}")
print("TESTING...")
print(f"{'='*70}")

results_data = []
correct = 0
total = 0

class_correct = defaultdict(int)
class_total = defaultdict(int)
confusion_matrix = defaultdict(lambda: defaultdict(int))

for idx, img_path in enumerate(test_images, 1):
    # Get ground truth
    label_file = TEST_LABELS / f"{img_path.stem}.txt"
    true_class = None
    
    if label_file.exists():
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                true_class = int(first_line.split()[0])
    
    # Predict
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    result = model(img, verbose=False, conf=0.25)[0]
    
    if len(result.boxes) > 0 and true_class is not None:
        pred_class = int(result.boxes[0].cls[0])
        confidence = float(result.boxes[0].conf[0])
        
        true_name = model.names[true_class]
        pred_name = model.names[pred_class]
        
        is_correct = (pred_class == true_class)
        
        if is_correct:
            correct += 1
            class_correct[true_name] += 1
        
        total += 1
        class_total[true_name] += 1
        
        # Confusion matrix
        confusion_matrix[true_name][pred_name] += 1
        
        results_data.append({
            'image': img_path.name,
            'true_class': true_class,
            'true_name': true_name,
            'pred_class': pred_class,
            'pred_name': pred_name,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Save first 20 visualizations
        if idx <= 20:
            img_vis = img.copy()
            h, w = img_vis.shape[:2]
            
            # Resize if too small
            if h < 200:
                scale = 300 / h
                img_vis = cv2.resize(img_vis, None, fx=scale, fy=scale)
            
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            
            cv2.putText(img_vis, f"{status}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(img_vis, f"True: {true_name}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_vis, f"Pred: {pred_name}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img_vis, f"Conf: {confidence:.2f}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            output_path = OUTPUT_DIR / f"test_{idx:03d}_{true_name}_pred_{pred_name}.jpg"
            cv2.imwrite(str(output_path), img_vis)
    
    if idx % 100 == 0:
        print(f"   Processed {idx}/{len(test_images)}...")

# Calculate metrics
accuracy = (correct / total * 100) if total > 0 else 0

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\n📊 Overall Performance:")
print(f"   Total tested: {total}")
print(f"   Correct: {correct}/{total}")
print(f"   ✅ Accuracy: {accuracy:.2f}%")

# Per-class accuracy
print(f"\n📈 Per-Class Accuracy:")
class_accuracies = []
for class_name in sorted(class_total.keys()):
    class_acc = (class_correct[class_name] / class_total[class_name] * 100) if class_total[class_name] > 0 else 0
    class_accuracies.append((class_name, class_acc, class_total[class_name]))
    print(f"   {class_name:15s}: {class_acc:6.2f}% ({class_correct[class_name]:3d}/{class_total[class_name]:3d})")

# Find best and worst classes
if class_accuracies:
    best_class = max(class_accuracies, key=lambda x: x[1])
    worst_class = min(class_accuracies, key=lambda x: x[1])
    
    print(f"\n🏆 Best: {best_class[0]} ({best_class[1]:.1f}%)")
    print(f"⚠️  Worst: {worst_class[0]} ({worst_class[1]:.1f}%)")

# Confusion matrix visualization
print(f"\n📊 Creating confusion matrix...")

class_names = sorted(model.names.values())
n_classes = len(class_names)
cm = np.zeros((n_classes, n_classes))

for i, true_name in enumerate(class_names):
    for j, pred_name in enumerate(class_names):
        cm[i, j] = confusion_matrix[true_name][pred_name]

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)

plt.colorbar(im, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Matra Confusion Matrix\nAccuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')

plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {cm_path.name}")

# Accuracy chart
fig, ax = plt.subplots(figsize=(14, 6))
class_names_sorted = [x[0] for x in sorted(class_accuracies, key=lambda x: x[1], reverse=True)]
accuracies_sorted = [x[1] for x in sorted(class_accuracies, key=lambda x: x[1], reverse=True)]

colors = ['green' if acc >= 90 else 'orange' if acc >= 70 else 'red' for acc in accuracies_sorted]
ax.bar(range(len(class_names_sorted)), accuracies_sorted, color=colors)

ax.set_xticks(range(len(class_names_sorted)))
ax.set_xticklabels(class_names_sorted, rotation=45, ha='right')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(f'Per-Class Matra Accuracy\nOverall: {accuracy:.1f}%', fontsize=14, fontweight='bold')
ax.axhline(y=accuracy, color='blue', linestyle='--', label=f'Overall: {accuracy:.1f}%')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
acc_path = OUTPUT_DIR / "accuracy_chart.png"
plt.savefig(acc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {acc_path.name}")

# Save detailed report
report_path = OUTPUT_DIR / "test_report.txt"
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("MATRA MODEL TEST REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Model: {MATRA_MODEL_PATH}\n")
    f.write(f"Test images: {len(test_images)}\n\n")
    
    f.write(f"OVERALL RESULTS:\n")
    f.write(f"  Accuracy: {accuracy:.2f}%\n")
    f.write(f"  Correct: {correct}/{total}\n\n")
    
    f.write(f"PER-CLASS ACCURACY:\n")
    for class_name, acc, count in sorted(class_accuracies, key=lambda x: x[1], reverse=True):
        f.write(f"  {class_name:15s}: {acc:6.2f}% ({class_correct[class_name]:3d}/{count:3d})\n")
    
    f.write(f"\n\nSAMPLE PREDICTIONS (first 20):\n")
    for i, res in enumerate(results_data[:20], 1):
        status = "✓" if res['correct'] else "✗"
        f.write(f"{i:2d}. {status} {res['image']:30s} True: {res['true_name']:10s} | Pred: {res['pred_name']:10s} ({res['confidence']:.2f})\n")

print(f"   ✓ Saved: {report_path.name}")

# Save JSON metrics
metrics = {
    'overall': {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    },
    'per_class': {
        name: {
            'accuracy': (class_correct[name] / class_total[name] * 100) if class_total[name] > 0 else 0,
            'correct': class_correct[name],
            'total': class_total[name]
        }
        for name in class_total.keys()
    }
}

metrics_path = OUTPUT_DIR / "metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"   ✓ Saved: {metrics_path.name}")

print(f"\n{'='*70}")
print("✅ TESTING COMPLETE!")
print(f"{'='*70}")
print(f"\n📁 Results saved to: {OUTPUT_DIR}")
print(f"   - Sample images (first 20)")
print(f"   - Confusion matrix")
print(f"   - Accuracy chart")
print(f"   - Detailed report")
print(f"   - JSON metrics")
print(f"\n✅ Matra model accuracy: {accuracy:.2f}%")
print(f"{'='*70}")