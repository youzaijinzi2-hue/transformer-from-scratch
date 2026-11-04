
import csv
from pathlib import Path

import matplotlib.pyplot as plt


with open("results/embedding_ablation.csv") as f:
    rows = list(csv.DictReader(f))

conditions = ["A", "B", "C", "D"]

for metric, ylabel in [
    ("content_accuracy", "Content Accuracy"),
    ("exact_match", "Sequence Exact Match"),
]:
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition in conditions:
        selected = [
            row for row in rows
            if row["condition"] == condition
        ]

        steps = [int(row["step"]) for row in selected]
        values = [float(row[metric]) for row in selected]

        ax.plot(
            steps,
            values,
            marker="o",
            label=condition,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        Path("results") / f"ablation_{metric}.png",
        dpi=200,
    )

    plt.close(fig)
