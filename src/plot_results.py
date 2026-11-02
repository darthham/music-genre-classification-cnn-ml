
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import REPORTS_DIR


def main():
    results_file = REPORTS_DIR / "classical_results.json"

    with open(results_file, "r") as file:
        results = json.load(file)

    features = [
        "raw_mfcc",
        "pca_50",
        "cnn_layer1",
        "cnn_layer2"
    ]

    classifiers = [
        "Logistic Regression",
        "SVM_RBF",
        "Random Forest",
        "GDA_LDA"
    ]

    labels = [
        "Raw MFCC",
        "PCA (50)",
        "CNN Layer 1",
        "CNN Layer 2"
    ]

    x = np.arange(len(features))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, classifier in enumerate(classifiers):
        values = []

        for feature in features:
            accuracy = next(
                item["accuracy"]
                for item in results
                if item["features"] == feature
                and item["classifier"] == classifier
            )

            values.append(accuracy * 100)

        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=classifier
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3,
            fontsize=8
        )

    ax.set_title(
        "Music Genre Classification - Accuracy Comparison",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Feature Representation", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 115)
    ax.legend(title="Classifier")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_file = REPORTS_DIR / "accuracy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"Accuracy comparison graph saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
