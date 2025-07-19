import json
import numpy as np
import matplotlib.pyplot as plt


def plot(scores, title, ax):
    n, bins, patches = ax.hist(scores, bins=100, color='skyblue', alpha=0.7, edgecolor='navy')

    std = np.std(scores)
    mean = np.mean(scores)
    median = np.median(scores)

    ax.vlines(mean - std, 0, max(n) * 1.05, color='red', linestyle='--', linewidth=1.5, label='±1 std')
    ax.vlines(mean + std, 0, max(n) * 1.05, color='red', linestyle='--', linewidth=1.5)
    ax.vlines(mean, 0, max(n) * 1.05, color='blue', linewidth=2, label='Mean')

    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)

    stats_text = f'Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))


with open("scores.json", "r", encoding="utf-8") as file:
    data = json.load(file)

scores = np.array(data['scores'])

METRICS = ['F1', 'NED', 'CER']

plt.figure(figsize=(18, 5))
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

for i, metric_name in enumerate(METRICS):
    values = scores[:, i]
    plot(values, metric_name, ax[i])

plt.tight_layout(pad=3.0)
plt.show()