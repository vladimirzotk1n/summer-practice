from collections import defaultdict

import editdistance


# Метрика из CC-OCR
def f1_score(pred, gt):
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






# Метрика из OmniDocBench:
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
