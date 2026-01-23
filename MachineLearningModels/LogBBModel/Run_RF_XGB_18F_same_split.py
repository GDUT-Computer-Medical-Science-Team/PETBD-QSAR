"""Train RF and XGBoost logBB models on identical 18F-balanced splits.

This helper script reproduces the combined oversampling strategy used in
`RF_FP_18F_Resample.py` and `XGBoost_FP_18F_Resample.py`, but it keeps the
train/validation/test partitions fixed so both algorithms see the exact same
data.  The resulting metrics, predictions, and split metadata are stored under
`MachineLearningModels/logBBModel/result/` and can be cited in reviews or
appendices when comparing RF vs XGBoost on PETBD/PTBD data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "PTBD_v20240912.csv"
RESULT_DIR = Path(__file__).resolve().parent / "result"
OUTPUT_DIR = RESULT_DIR / "same_split_run"

TARGET_COLUMN = "logBB at60min"
MORGAN_BITS = 1024
MAX_SELECTED_FEATURES = 50
RANDOM_SEED = 42

RF_PARAMS = {
    "n_estimators": 106,
    "max_depth": 20,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

XGB_PARAMS = {
    "n_estimators": 954,
    "max_depth": 9,
    "learning_rate": 0.1670406775452887,
    "subsample": 0.8914816699880982,
    "colsample_bytree": 0.8906616429952791,
    "reg_alpha": 2.5322443715452554,
    "reg_lambda": 2.9399228775445607,
    "random_state": RANDOM_SEED,
}


def contains_18f(compound_index: str) -> bool:
    """Return True if the compound index contains the 18F isotope string."""
    if not isinstance(compound_index, str):
        return False
    return "18F" in compound_index.strip()


def balance_18f_dataset(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Apply the 'combined' strategy (oversample minority class, keep originals)."""
    df_copy = df.copy()
    df_copy["has_18F"] = df_copy["compound index"].apply(contains_18f)
    df_copy["label_18F"] = df_copy["has_18F"].astype(int)

    df_18f = df_copy[df_copy["has_18F"] == True]  # noqa: E712
    df_non_18f = df_copy[df_copy["has_18F"] == False]  # noqa: E712

    n_18f, n_non = len(df_18f), len(df_non_18f)
    if n_18f < n_non:
        extra = df_18f.sample(n=n_non - n_18f, replace=True, random_state=seed)
        combined = pd.concat([df_copy, extra], ignore_index=True)
    elif n_non < n_18f:
        extra = df_non_18f.sample(n=n_18f - n_non, replace=True, random_state=seed)
        combined = pd.concat([df_copy, extra], ignore_index=True)
    else:
        combined = df_copy

    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def calculate_morgan_fingerprints(
    smiles_list: List[str], radius: int = 2, n_bits: int = MORGAN_BITS
) -> np.ndarray:
    """Compute Morgan fingerprints for every SMILES string."""
    fps = []
    zero_vec = np.zeros(n_bits, dtype=float)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(zero_vec.copy())
            continue
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=float)
        DataStructs.ConvertToNumpyArray(bit_vect, arr)
        fps.append(arr)
    return np.asarray(fps, dtype=float)


def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Compute adjusted R² while guarding against division by zero."""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if n - n_features - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """Return regression metrics for a given split."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "adj_r2": float(adjusted_r2(y_true, y_pred, n_features)),
        "mape": float(mape),
    }


def prepare_balanced_features() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load PTBD data, drop missing rows, and return original/balanced tables."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing PTBD dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["SMILES", TARGET_COLUMN]).reset_index(drop=True)
    df["compound index"] = df["compound index"].astype(str)

    balanced_df = balance_18f_dataset(df, seed=RANDOM_SEED)
    features = calculate_morgan_fingerprints(balanced_df["SMILES"].tolist())
    targets = balanced_df[TARGET_COLUMN].astype(float).to_numpy()
    return df.reset_index(drop=True), balanced_df.reset_index(drop=True), features, targets


def split_transform(
    features: np.ndarray, targets: np.ndarray, row_ids: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]], List[int]]:
    """Split data into train/val/test and apply imputation, selection, and scaling."""
    X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
        features,
        targets,
        row_ids,
        test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    imputer = SimpleImputer(strategy="mean")
    X_tv = imputer.fit_transform(X_tv)
    X_test = imputer.transform(X_test)

    selector_model = ExtraTreesRegressor(n_estimators=256, random_state=RANDOM_SEED, n_jobs=-1)
    selector_model.fit(X_tv, y_tv)
    selector = SelectFromModel(selector_model, prefit=True, max_features=MAX_SELECTED_FEATURES)
    X_tv = selector.transform(X_tv)
    X_test = selector.transform(X_test)
    selected_indices = selector.get_support(indices=True).tolist()

    scaler = MinMaxScaler()
    X_tv = scaler.fit_transform(X_tv)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_tv, y_tv, idx_tv, test_size=0.1, random_state=RANDOM_SEED
    )

    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    indices = {
        "train": idx_train.tolist(),
        "val": idx_val.tolist(),
        "test": idx_test.tolist(),
    }
    return arrays, indices, selected_indices


def train_random_forest(
    arrays: Dict[str, np.ndarray]
) -> Tuple[RandomForestRegressor, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Fit RF on the training split and return metrics/predictions."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(arrays["X_train"], arrays["y_train"])

    preds = {
        "train": model.predict(arrays["X_train"]),
        "val": model.predict(arrays["X_val"]),
        "test": model.predict(arrays["X_test"]),
    }
    metrics = {
        split: compute_metrics(arrays[f"y_{split}"], preds[split], arrays[f"X_{split}"].shape[1])
        for split in ("train", "val", "test")
    }
    return model, metrics, preds


def train_xgboost(
    arrays: Dict[str, np.ndarray]
) -> Tuple[XGBRegressor, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Fit XGBoost on the training split and return metrics/predictions."""
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(arrays["X_train"], arrays["y_train"])

    preds = {
        "train": model.predict(arrays["X_train"]),
        "val": model.predict(arrays["X_val"]),
        "test": model.predict(arrays["X_test"]),
    }
    metrics = {
        split: compute_metrics(arrays[f"y_{split}"], preds[split], arrays[f"X_{split}"].shape[1])
        for split in ("train", "val", "test")
    }
    return model, metrics, preds


def build_prediction_rows(
    df_balanced: pd.DataFrame,
    indices: Dict[str, List[int]],
    preds: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
    model_name: str,
) -> List[Dict[str, object]]:
    """Attach metadata (compound index, has_18F) to prediction outputs."""
    rows: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        split_indices = indices[split]
        y_true = arrays[f"y_{split}"]
        y_pred = preds[split]

        for i, row_idx in enumerate(split_indices):
            row = df_balanced.iloc[row_idx]
            rows.append(
                {
                    "model": model_name,
                    "dataset": split.capitalize(),
                    "compound_index": row["compound index"],
                    "SMILES": row["SMILES"],
                    "has_18F": bool(row["has_18F"]),
                    "true_logBB": float(y_true[i]),
                    "pred_logBB": float(y_pred[i]),
                }
            )
    return rows


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_original, df_balanced, features, targets = prepare_balanced_features()
    print("=============== RF/XGB 18F same-split experiment ===============")
    print(f"Dataset file: {DATA_FILE.name}")
    print(f"Original samples (after dropping NaN logBB): {len(df_original)}")
    isotope_counts = df_balanced["has_18F"].value_counts().to_dict()
    print(
        f"Balanced samples: {len(df_balanced)} "
        f"(18F: {isotope_counts.get(True, 0)}, non-18F: {isotope_counts.get(False, 0)})"
    )
    print(f"Fingerprint matrix shape before selection: {features.shape}")

    row_ids = np.arange(len(df_balanced))
    arrays, index_map, selected_indices = split_transform(features, targets, row_ids)
    print(f"Selected feature count: {len(selected_indices)}")
    print(
        "Split sizes -> "
        f"Train: {len(index_map['train'])}, "
        f"Validation: {len(index_map['val'])}, "
        f"Test: {len(index_map['test'])}"
    )

    rf_model, rf_metrics, rf_preds = train_random_forest(arrays)
    xgb_model, xgb_metrics, xgb_preds = train_xgboost(arrays)

    prediction_rows = []
    prediction_rows.extend(build_prediction_rows(df_balanced, index_map, rf_preds, arrays, "RandomForest"))
    prediction_rows.extend(build_prediction_rows(df_balanced, index_map, xgb_preds, arrays, "XGBoost"))

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_path = OUTPUT_DIR / "rf_xgb_same_split_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    split_dir = OUTPUT_DIR / "per_split_predictions"
    split_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in ("Train", "Validation", "Test"):
        subset = predictions_df[predictions_df["dataset"] == dataset_name]
        subset.to_csv(split_dir / f"{dataset_name.lower()}_predictions.csv", index=False)

    metrics_summary = {"random_forest": rf_metrics, "xgboost": xgb_metrics}
    metrics_path = OUTPUT_DIR / "rf_xgb_same_split_metrics.json"
    save_json(metrics_path, metrics_summary)

    split_metadata = {
        "dataset": DATA_FILE.name,
        "target_column": TARGET_COLUMN,
        "balancing": "Combined 18F oversampling",
        "seed": RANDOM_SEED,
        "n_original_samples": int(len(df_original)),
        "n_balanced_samples": int(len(df_balanced)),
        "selected_feature_indices": selected_indices,
        "split_sizes": {k: len(v) for k, v in index_map.items()},
        "train_compounds": df_balanced.loc[index_map["train"], "compound index"].tolist(),
        "val_compounds": df_balanced.loc[index_map["val"], "compound index"].tolist(),
        "test_compounds": df_balanced.loc[index_map["test"], "compound index"].tolist(),
    }
    metadata_path = OUTPUT_DIR / "rf_xgb_same_split_metadata.json"
    save_json(metadata_path, split_metadata)

    print("\nRandom Forest metrics (Train/Val/Test):")
    for split in ("train", "val", "test"):
        metric = rf_metrics[split]
        print(
            f"  {split.capitalize()}: R^2={metric['r2']:.3f}, "
            f"RMSE={metric['rmse']:.3f}, MAE={metric['mae']:.3f}"
        )
    print("\nXGBoost metrics (Train/Val/Test):")
    for split in ("train", "val", "test"):
        metric = xgb_metrics[split]
        print(
            f"  {split.capitalize()}: R^2={metric['r2']:.3f}, "
            f"RMSE={metric['rmse']:.3f}, MAE={metric['mae']:.3f}"
        )

    print("\nArtifacts:")
    print(f"  Combined predictions: {predictions_path}")
    print(f"  Per-split CSV folder: {split_dir}")
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  Split metadata JSON: {metadata_path}")


if __name__ == "__main__":
    main()
