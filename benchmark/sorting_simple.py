import json
import cv2
import numpy as np
from collections import defaultdict


def sort_heights(results):
    centers = np.zeros([len(results), 2])
    for i, (bbox, _) in enumerate(results):
        centers[i] = np.array(bbox).mean(0)
    srt_idx = np.argsort(centers[:, 1])
    return [results[i] for i in srt_idx]

def get_paragraph(boxes, x_ths=1, y_ths=0.5, mode="ltr"):
    box_group = []
    for i, box in enumerate(boxes):
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        height = max_y - min_y
        box_group.append([i, min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0])

    current_group = 1
    while any(box[7] == 0 for box in box_group):
        box_group0 = [box for box in box_group if box[7] == 0]
        if not any(box[7] == current_group for box in box_group):
            box_group0[0][7] = current_group
        else:
            current_box_group = [box for box in box_group if box[7] == current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height

            add_box = False
            for box in box_group0:
                same_horizontal = (min_gx <= box[1] <= max_gx) or (min_gx <= box[2] <= max_gx)
                same_vertical = (min_gy <= box[3] <= max_gy) or (min_gy <= box[4] <= max_gy)
                if same_horizontal and same_vertical:
                    box[7] = current_group
                    add_box = True
                    break
            if not add_box:
                current_group += 1

    result = []
    for i in set([box[7] for box in box_group]):
        current_box_group = [box for box in box_group if box[7] == i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        order = []
        while current_box_group:
            highest = min([box[6] for box in current_box_group])
            candidates = [box for box in current_box_group if box[6] < highest + 0.4 * mean_height]
            if mode == "ltr":
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left:
                        best_box = box
            else:
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right:
                        best_box = box
            order.append(best_box[0])
            current_box_group.remove(best_box)

        result.append([
            [[min_gx, min_gy], [max_gx, min_gy], [max_gx, max_gy], [min_gx, max_gy]],
            order
        ])
    return result

def get_sorted_order_simple(boxes, x_ths=1, y_ths=0.5, mode="ltr"):
    pr = get_paragraph(boxes, x_ths, y_ths, mode)
    order = np.array(sum([s[-1] for s in sort_heights(pr)], []))
    return order