"""
EVALUATION MODULE
- Aggregate, compare, and visualize experiment metrics
- Input: pd.DataFrame from run_experiments()
- No training, no inference
"""

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# METRIC AGGREGATION
# ======================================================

def select_metrics(
    results_df: pd.DataFrame,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Select subset of metrics for comparison
    """
    metrics = metrics or ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cols = ["model", "model_type"] + [m for m in metrics if m in results_df.columns]
    return results_df[cols].set_index("model")


# ======================================================
# TABULAR COMPARISON
# ======================================================

def rank_models(
    results_df: pd.DataFrame,
    by: str = "f1",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank models by a given metric
    """
    if by not in results_df.columns:
        raise ValueError(f"Metric '{by}' not found")

    ranked = results_df.sort_values(by=by, ascending=ascending)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked[["model", "model_type", by, "rank"]]


# ======================================================
# VISUALIZATION
# ======================================================

def plot_metric_bar(
    results_df: pd.DataFrame,
    metric: str,
    figsize=(8, 4),
    save_path: str = None
):
    """
    Bar chart comparison for a single metric
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    plt.figure(figsize=figsize)
    sns.barplot(
        data=results_df,
        x="model",
        y=metric,
        hue="model_type"
    )
    plt.title(f"Model Comparison – {metric.upper()}")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    plt.show()


def plot_metric_group(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    save_path: str = None

):
    """
    Compare multiple metrics across models
    """
    metrics = metrics or ["accuracy", "precision", "recall", "f1"]

    plot_df = results_df.melt(
        id_vars=["model", "model_type"],
        value_vars=[m for m in metrics if m in results_df.columns],
        var_name="metric",
        value_name="score"
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="model",
        y="score",
        hue="metric"
    )
    plt.title("Multi-metric Model Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    plt.show()


# ======================================================
# CONFUSION MATRIX
# ======================================================

def plot_confusion_matrix(
    confusion_matrix: List[List[int]],
    model_name: str,
    class_names=("Legit", "Phishing"),
    save_path: str = None
):
    """
    Plot confusion matrix for one model
    """
    cm = np.array(confusion_matrix)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    plt.show()


# ======================================================
# SUMMARY REPORT
# ======================================================

def summarize(results_df: pd.DataFrame) -> None:
    """
    Print concise summary
    """
    print("\n===== BEST MODELS =====")
    for metric in ["accuracy", "f1", "roc_auc"]:
        if metric in results_df.columns:
            best = results_df.loc[results_df[metric].idxmax()]
            print(f"- Best {metric}: {best['model']} ({best[metric]:.4f})")