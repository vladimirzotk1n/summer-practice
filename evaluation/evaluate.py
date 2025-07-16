from metrics import eval_trans, bleu, normalized_edit_distance
from ollama import predict
import numpy as np
import json
from tqdm import tqdm
import pandas as pd


METRICS = [eval_trans, bleu, normalized_edit_distance]

"""
Тетрадки:
"""
# def evaluate(bench_json_path):
#     scores = []
#
#     with open(bench_json_path, "r", encoding="utf-8") as f:
#         bench_json = json.load(f)
#
#     for element in tqdm(bench_json[:100]):
#         path = "C:\\images\\" + element['image_path']
#         answer = element['answers']
#         prediction = predict(path)
#
#         element_scores = []
#
#         for metric in METRICS:
#             score = metric(prediction, answer)
#             element_scores.append(score)
#
#         scores.append(element_scores)
#         print(f"BLEU: {element_scores[1]}")
#
#     return np.mean(np.array(scores), axis=0)
#
# print(evaluate('bench_val.json'))
# На 100 примерах - [0.69734261 0.02130837 0.78774087] - на тетрадках




def evaluate():
    scores = {}

    df = pd.read_csv('hiertext.csv')

    for i in tqdm(range(100)):
        path = df.loc[i, 'image']
        answer = df.loc[i, 'text']
        prediction = predict(path)

        element_scores = []

        for metric in METRICS:
            score = metric(prediction, answer)
            element_scores.append(score)

        scores[i] = element_scores

    # return np.mean(np.array(scores), axis=0)
    return scores

print(evaluate())

# На hiertext 100 штуках: [0.81519207 0.20597173 0.61661168], legible=False
# На hiertext 100 штуках: [0.76170418 0.13367599 0.69961541], legible=True

