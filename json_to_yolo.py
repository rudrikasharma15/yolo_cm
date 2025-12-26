def json_to_yolo(json_path, img_path, CLASS_NAMES):
    import json
    import cv2

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data.get('imageWidth')
    img_h = data.get('imageHeight')

    if not img_w or not img_h:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img_h, img_w = img.shape[:2]

    yolo_lines = []

    for shape in data.get('shapes', []):
        label = shape['label'].lower().strip()

        if label.endswith('_matra'):
            label = label.replace('_matra', '')

        if label != 'anusvara':
            label = f"{label}_matra"

        if label not in CLASS_NAMES:
            continue

        class_id = CLASS_NAMES.index(label)

        (x1, y1), (x2, y2) = shape['points']
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return "\n".join(yolo_lines) if yolo_lines else None
