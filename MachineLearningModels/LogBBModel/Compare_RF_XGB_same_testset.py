"""
Compare Random Forest and XGBoost predictions on the same PETBD test split.

The script loads the saved prediction CSVs, checks that the true values come
from the same 85-sample test set (seed=42), computes evaluation metrics, and
stores/prints a unified summary so we can reference a reproducible experiment
when responding to reviewer questions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

RF_FILE = RESULT_DIR / "rf_18F_test_predictions_with_compound_index.csv"
XGB_FILE = RESULT_DIR / "xgb_18F_test_predictions.csv"
OUTPUT_CSV = RESULT_DIR / "RF_vs_XGB_same_dataset_metrics.csv"
OUTPUT_JSON = RESULT_DIR / "RF_vs_XGB_same_dataset_metrics.json"


def load_predictions(path: Path, true_col: str, pred_col: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read a prediction csv and return the true/pred arrays along with the raw dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)
    for col in (true_col, pred_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}")
    return df[true_col].to_numpy(), df[pred_col].to_numpy(), df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return R², RMSE, and MAE for the given arrays."""
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def main() -> None:
    rf_true, rf_pred, rf_df = load_predictions(RF_FILE, "True_logBB", "Predicted_logBB")
    xgb_true, xgb_pred, xgb_df = load_predictions(XGB_FILE, "True_Values", "Predicted_Values")

    if len(rf_true) != len(xgb_true):
        raise ValueError("RF and XGB prediction files use different sample counts.")

    same_true = np.allclose(rf_true, xgb_true)
    if not same_true:
        raise ValueError("True labels differ between RF and XGB predictions; not the same test set.")

    metrics_data = []
    metrics_data.append(
        {
            "model": "Random Forest (RF_FP_18F_Resample)",
            "n_samples": len(rf_true),
            **compute_metrics(rf_true, rf_pred),
        }
    )
    metrics_data.append(
        {
            "model": "XGBoost (XGBoost_FP_18F_Resample)",
            "n_samples": len(xgb_true),
            **compute_metrics(xgb_true, xgb_pred),
        }
    )

    df = pd.DataFrame(metrics_data)
    print("RF vs XGB on identical PETBD test split (seed=42, n=85)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTrue label sequence identical between models:", same_true)

    df.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"same_true_labels": same_true, "metrics": metrics_data}, f, indent=2)
    print(f"\nSaved metrics to:\n- {OUTPUT_CSV}\n- {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
