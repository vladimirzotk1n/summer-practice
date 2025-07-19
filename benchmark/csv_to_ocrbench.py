import json
import os
import re
from pathlib import Path

def to_OCRBench(df, output_json, dataset_name):
    data = []

    for _, row in df.iterrows():
        path = row['image']
        label = row['text']

        filename = os.path.basename(path)
        file_id = os.path.splitext(filename)[0]
        image_folder = Path(__file__).parent.parent / "data" / "sbernotes" / "images"

        item = {
            "dataset_name": dataset_name,
            "id": file_id,
            "image_path": os.path.join(image_folder, path),
            "question": "What is written in the image?",
            "answers": label,
            "type": "Regular Text Recognition"
        }
        data.append(item)

    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4, ensure_ascii=False)


# data = pd.read_csv('sorted_train.csv')
# to_OCRBench(data, 'hiertext_train.json', 'hiertext')
