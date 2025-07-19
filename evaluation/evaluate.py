from metrics import f1_score, normalized_edit_distance, cer
from ollama import predict
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).parent.parent

SBER_BENCH_PATH = root / "benchmark" / "sbernotes" / "sber-bench.json"
HIERTEXT_BENCH_PATH = root / "benchmark" / "hiertext" / "hiertext-bench.json"
METRICS = [f1_score, normalized_edit_distance, cer]

"""
Тетрадки: без батчей, изначальная версия
# """
def evaluate(bench_json_path):
    scores = []

    with open(bench_json_path, "r", encoding="utf-8") as f:
        bench_json = json.load(f)[:300]

    for element in tqdm(bench_json):
        path = element['image_path']
        answer = element['answers']
        prediction = predict(path)

        element_scores = []

        for metric in METRICS:
            score = metric(prediction, answer)
            element_scores.append(score)

        scores.append(element_scores)

    # return np.array(scores)

    return {"scores" : scores}

scores = evaluate(HIERTEXT_BENCH_PATH)

with open("scores.json", "w", encoding="utf-8") as f:
    json.dump(scores, f, ensure_ascii=False, indent=4)

# print(np.mean(scores, axis=0))


def metrics_to_df(scores):
    mean_scores = np.mean(scores, axis=0).reshape(1, -1)
    metrics_df = pd.DataFrame(mean_scores, columns=METRICS)
    return metrics_df

# На hiertext 100 штуках: [0.81519207 0.20597173 0.61661168], legible=False
# На hiertext 100 штуках: [0.76170418 0.13367599 0.69961541], legible=True
# На 100 примерах - [0.69734261 0.02130837 0.78774087] - на тетрадках

