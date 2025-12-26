from pathlib import Path
from json_to_yolo import json_to_yolo

CLASS_NAMES = [
    "aa_matra",
    "i_matra",
    "u_matra",
    "e_matra",
    "ai_matra",
    "o_matra",
    "au_matra",
    "anusvara"
]

BASE_DIR = Path(
    "/Users/applemaair/Desktop/modi_final/matra_extraction_PERFECT/unlabeled_matras"
)

json_files = list(BASE_DIR.rglob("*.json"))

converted = 0
skipped = 0

for json_file in json_files:
    img_file = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = json_file.with_suffix(ext)
        if candidate.exists():
            img_file = candidate
            break

    if img_file is None:
        skipped += 1
        continue

    yolo_txt = json_file.with_suffix(".txt")

    yolo_content = json_to_yolo(json_file, img_file, CLASS_NAMES)
    if yolo_content:
        yolo_txt.write_text(yolo_content)
        converted += 1
    else:
        skipped += 1

print(f"✅ Converted: {converted}")
print(f"⚠️ Skipped: {skipped}")
