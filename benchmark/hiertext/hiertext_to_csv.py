import os
import json
import csv
import numpy as np
from benchmark.sorting_simple import get_sorted_order_simple

def extract_hiertext_with_order(json_path, image_base_path, legible=True):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for annotation in data["annotations"]:
        image_id = annotation["image_id"] + ".jpg"  # ? точно ли тут так
        full_image_path = os.path.join(image_base_path, image_id)

        boxes = []
        texts = []

        for para in annotation.get("paragraphs", []):
            if legible and not para.get("legible", True):
                continue

            for line in para.get("lines", []):
                if legible and not line.get("legible", True):
                    continue

                text = line.get("text", "").strip()
                if not text:
                    continue

                vertices = line["vertices"]
                if len(vertices) < 4:
                    continue

                box = np.array(vertices[:4])  # [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
                boxes.append(box)
                texts.append(text)

        if not boxes:
            continue

        box_text_pairs = [(box, text) for box, text in zip(boxes, texts)]


        sorted_idx = get_sorted_order_simple(np.array(boxes), x_ths=1, y_ths=0.5, mode="ltr")

        sorted_texts = [texts[i] for i in sorted_idx]
        full_text = " ".join(sorted_texts)
        results.append((full_image_path, full_text))

    return results