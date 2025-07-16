# Метрика из CC-OCR
from collections import defaultdict

from ollama import predict


def eval_trans(pred, gt):
    pred = pred.strip()
    gt = gt.strip()

    if not pred and not gt:
        return 1.0

    if not pred or not gt:
        return 0.0

    units = set(pred.split()) & set(gt.split())

    pred_counts = defaultdict(int)
    gt_counts = defaultdict(int)

    for word in pred.split():
        if word in units:
            pred_counts[word] += 1

    for word in gt.split():
        if word in units:
            gt_counts[word] += 1

    pred_sum = sum(pred_counts[word] for word in units)
    gt_sum = sum(gt_counts[word] for word in units)

    if pred_sum == 0 or gt_sum == 0:
        return 0.0


    recall = sum(min(pred_counts[word], gt_counts[word]) for word in units) / gt_sum
    precision = sum(min(pred_counts[word], gt_counts[word]) for word in units) / pred_sum

    return 2 * precision * recall / (precision + recall)




# Метрики из OmniDocBench:
import evaluate
bleu_scorer = evaluate.load("bleu")

def bleu(pred, gt):
    pred = pred.strip()
    gt = gt.strip()

    if not pred and not gt:
        return 1.0

    if not pred or not gt:
        return 0.0

    reference = [[gt]]
    prediction = [pred]
    min_len = min(len(pred.split()), len(gt.split()))
    max_order = min(min_len, 4)  # 4 не слишком ли много?

    try:
        score = bleu_scorer.compute(predictions=prediction, references=reference, max_order=max_order)['bleu']
    except ZeroDivisionError:
        score = 0.0

    return score



import editdistance

def normalized_edit_distance(pred, gt):
    pred = pred.strip()
    gt = gt.strip()

    dist = editdistance.eval(pred, gt)

    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0

    return dist / max_len


def cer(pred, gt):
    pred = pred.strip()
    gt = gt.strip()

    dist = editdistance.eval(pred, gt)
    gt_len = len(gt)
    if gt_len == 0:
        return 0.0

    return dist / gt_len

# CER


def test():
    tests = [
        {"pred": "cat", "gt": "cat"},
        {"pred": "cot", "gt": "cat"},
        {"pred": "act", "gt": "cat"},
        {"pred": "the quick brown fox jumps over the lazy dog", "gt": "the quick brown fox jump over the lazy dog"},
        {"pred": "apple banana orange", "gt": "car truck train plane"},
        {"pred": "internationalization", "gt": "internationalisation"},
        {"pred": "I love natural language processing", "gt": "natural language processing I love"},
        {"pred": "this is a test sentence with extra words", "gt": "this is a test sentence"},
        {"pred": "hello world", "gt": "hello there"},
        {"pred": "machine learning is great", "gt": "machine learning is awesome"},
        {"pred": "abcde", "gt": "abxde"},
        {"pred": "one two three four five", "gt": "five four three two one"},
        {"pred": "", "gt": ""},
        {"pred": "long string with some differences", "gt": "long string with several differences"},
        {"pred" : " ", "gt" : "hello my name is apple"},
        {"pred" : "hello my name is apple", "gt" : " "}
    ]


    for i, t in enumerate(tests, start=1):
        pred = t["pred"]
        gt = t["gt"]

        print(f"\n{'='*60}")
        print(f"Test {i}:")
        print(f"Pred: '{pred}'")
        print(f"GT  : '{gt}'")
        print()
        print(f"Eval Trans : {eval_trans(pred, gt)}")
        # print(f"BLEU : {bleu(pred, gt)}")
        print(f"NED : {normalized_edit_distance(pred, gt)}")
        print(f"CER : {cer(pred, gt)}")

        print('-'*60)


if __name__ == "__main__":
    test()

# pred = predict('picture.png')
# gt = "Miss Warren Keene GYM STATE COLLEGE I'm a Working Out! My hands on My favorite Hometown ! food! learner! Reading! Singing! Volunteering! Teacher Vacation My House! Majors! I love to DANCE! Sebago Lake!"
#
# print(pred)
# print()
# print()
# print(f"Eval Trans : {eval_trans(pred, gt)}")
# print(f"NED : {normalized_edit_distance(pred, gt)}")
# print(f"CER : {cer(pred, gt)}")