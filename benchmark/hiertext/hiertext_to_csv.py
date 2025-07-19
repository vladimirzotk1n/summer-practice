import os
import json
import csv
import numpy as np
from benchmark.sorting_simple import get_sorted_order_simple


"""
Без сортировки боксов
"""
def transform_hiertext_to_csv(json_path, image_folder, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        full_image_path = os.path.join(image_folder, image_id)

        full_text = []

        for para in annotation.get("paragraphs", []):
            if not para.get("legible", True):
                continue
            for line in para.get("lines", []):
                if not line.get("legible", True):
                    continue
                line_words = []
                for word in line.get("words", []):
                    if word.get("legible", True) and word.get("text", "").strip():
                        line_words.append(word["text"])
                if line_words:
                    full_text.append(" ".join(line_words))

        if full_text:
            results.append((full_image_path, " ".join(full_text)))

    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "text"])
        for filename, text in results:
            filepath = filename + ".jpg"
            writer.writerow([filepath, text])










def transform_hiertext_to_csv_sorted(json_path, image_base_path, csv_path, legible=True):
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


        sorted_idx = get_sorted_order_simple(np.array(boxes))

        sorted_texts = [texts[i] for i in sorted_idx]
        full_text = " ".join(sorted_texts)
        results.append((full_image_path, full_text))

    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "text"])
        for filename, text in results:
            writer.writerow([filename, text])


