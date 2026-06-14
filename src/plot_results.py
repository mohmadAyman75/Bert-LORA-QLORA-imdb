from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_utils import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "Output"
RESULTS_PATH = OUTPUT_DIR / "results" / "summary_results.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"


PLOTS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "peak_gpu_memory_mb": "Peak GPU memory (MB)",
    "total_parameters": "Total model parameters",
    "training_time_seconds": "Training time (seconds)",
}


def plot_metric(frame: pd.DataFrame, metric: str, label: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean = frame.dropna(subset=[metric]).copy()
    if clean.empty:
        print(f"Skipping {metric}: no valid values found.")
        return None

    summary = clean.groupby("method", as_index=False).tail(1)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=summary, x="method", y=metric)
    plt.title(f"{label} by method")
    plt.xlabel("Method")
    plt.ylabel(label)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"{metric}_comparison.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "No summary results found. Run at least one experiment first: "
            "python -m src.experiments --method feature --epochs 1"
        )

    frame = pd.read_csv(RESULTS_PATH)
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("ok").eq("ok")]

    created = []
    for metric, label in PLOTS.items():
        if metric in frame.columns:
            path = plot_metric(frame, metric, label)
            if path is not None:
                created.append(path)

    print("Created figures:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
