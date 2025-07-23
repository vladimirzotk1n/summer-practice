from metrics import f1_score, normalized_edit_distance, cer
from ollama import predict
import json
from tqdm import tqdm
from pathlib import Path


root = Path(__file__).parent.parent

SBER_BENCH_PATH = root / "benchmark" / "sbernotes" / "sber-bench.json"
HIERTEXT_BENCH_PATH = root / "benchmark" / "hiertext" / "hiertext-bench.json"
METRICS = [f1_score, normalized_edit_distance, cer]


def evaluate(bench_json_path):
    scores = []

    with open(bench_json_path, "r", encoding="utf-8") as f:
        bench_json = json.load(f)

    for i, element in enumerate(tqdm(bench_json)):
        path = element['image_path']
        answer = element['answers']
        prediction = predict(path)

        element_scores = []

        for metric in METRICS:
            score = metric(prediction, answer)
            element_scores.append(score)

        scores.append(element_scores)

        if i % 100 == 0:
            with open("scores-sbernotes.json", "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)


    return {"scores" : scores}

scores_hiertext = evaluate(HIERTEXT_BENCH_PATH)
with open("scores-hiertext.json", "w", encoding="utf-8") as f:
    json.dump(scores_hiertext, f, ensure_ascii=False, indent=4)

scores_sbernotes = evaluate(SBER_BENCH_PATH)
with open("scores-sbernotes.json", "w", encoding="utf-8") as f:
    json.dump(scores_sbernotes, f, ensure_ascii=False, indent=4)