"""
RUN EXPERIMENTS MODULE
- Orchestrates ML & DL experiments
- Assumes data is already preprocessed and split
"""

from typing import List, Dict, Any
import pandas as pd

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.base_experiment import BaseExperiment

def run_experiments(
    experiments: List[BaseExperiment],
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
) -> pd.DataFrame:
    """
    Run multiple experiments on the same dataset splits
    """

    results: List[Dict[str, Any]] = []

    for exp in experiments:
        print(f"\n========== Running: {exp.name} ==========")

        # -------- TRAIN --------
        exp.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # -------- EVALUATE --------
        metrics = exp.evaluate(X_test, y_test)

        # meta info (rất quan trọng khi visualize)
        metrics.update({
            "model": exp.name,
            "model_class": exp.__class__.__name__,
        })

        results.append(metrics)

    return pd.DataFrame(results)