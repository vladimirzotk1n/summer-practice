import json
import cv2
import numpy as np
from collections import defaultdict
from benchmark.sorting_simple import get_sorted_order_simple
import csv


def polygon_to_bbox(polygon):
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x, y, w, h = cv2.boundingRect(pts)
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


# С сортировкой боксов
def transform_sbernotes_to_csv(json_path, csv_path):

    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in annotation['images']}

    annotations_by_image = defaultdict(list)
    for ann in annotation['annotations']:
        annotations_by_image[ann['image_id']].append(ann)



    image_text_mapping = {}

    for image_id, anns in annotations_by_image.items():
        filename = image_id_to_filename[image_id]

        boxes = []
        valid_anns = []

        for ann in anns:
            try:
                polygon = ann["segmentation"][0]
                text = ann["attributes"]["translation"]
                if text is None:
                    continue
                bbox_coords = polygon_to_bbox(polygon)
                boxes.append(bbox_coords)

                valid_anns.append(ann)
            except (KeyError, IndexError):
                continue

        if not boxes:
            continue

        sorted_indices = get_sorted_order_simple(np.array(boxes))
        sorted_anns = [valid_anns[i] for i in sorted_indices]

        texts = []
        for ann in sorted_anns:
            text = ann["attributes"]["translation"]
            if text:
                texts.append(text.strip())

        full_text = " ".join(texts)
        image_text_mapping[filename] = full_text


    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "text"])  # Заголовки столбцов

        for filename, text in image_text_mapping.items():
            writer.writerow([filename, text])