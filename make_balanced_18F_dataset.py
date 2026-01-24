import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import re


# Strictly accepted radionuclides for classification
ALLOWED_ISOTOPES = {"18F", "11C", "125I", "131I", "123I", "77Br", "76Br"}


def extract_isotope_strict(compound_index: str) -> Optional[str]:
    """
    Extract isotope strictly from the compound index, only if it belongs to ALLOWED_ISOTOPES.

    This avoids mis-parsing IDs like "1096-218F" or "8F" that are not radionuclides.
    """
    if not isinstance(compound_index, str):
        return None

    text = compound_index.strip()

    # Try exact tokens first
    for iso in ALLOWED_ISOTOPES:
        if re.search(rf"(^|[^0-9A-Za-z]){re.escape(iso)}([^0-9A-Za-z]|$)", text):
            return iso

    return None


def label_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "compound index" not in df.columns:
        raise ValueError("Expected column 'compound index' not found in CSV")

    isotopes: List[Optional[str]] = [extract_isotope_strict(x) for x in df["compound index"].fillna("")]
    df = df.copy()
    df["isotope"] = isotopes

    # Keep only rows where we confidently detected an isotope
    df = df[df["isotope"].notna()].reset_index(drop=True)

    # Binary label: 1 for 18F, 0 for non-18F
    df["label_18F"] = (df["isotope"] == "18F").astype(int)
    return df


def balance_dataset(df: pd.DataFrame, method: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Balance the dataset between 18F (label_18F=1) and non-18F (label_18F=0).

    method:
      - undersample: downsample the majority class to the minority count
      - oversample: upsample the minority class to the majority count
    """
    rng = np.random.default_rng(seed)

    pos = df[df["label_18F"] == 1]
    neg = df[df["label_18F"] == 0]

    n_pos, n_neg = len(pos), len(neg)
    report: Dict[str, Any] = {
        "initial_counts": {"18F": n_pos, "non18F": n_neg},
        "method": method,
        "seed": seed,
    }

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot balance: one class has zero rows after filtering by allowed isotopes.")

    if method == "undersample":
        target = min(n_pos, n_neg)
        pos_bal = pos.sample(n=target, replace=False, random_state=seed)
        neg_bal = neg.sample(n=target, replace=False, random_state=seed)
    elif method == "oversample":
        target = max(n_pos, n_neg)
        pos_bal = pos.sample(n=target, replace=True, random_state=seed)
        neg_bal = neg.sample(n=target, replace=True, random_state=seed)
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")

    balanced = pd.concat([pos_bal, neg_bal], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    report["final_counts"] = {
        "18F": int((balanced["label_18F"] == 1).sum()),
        "non18F": int((balanced["label_18F"] == 0).sum()),
    }
    return balanced, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced 18F vs non-18F dataset from PETBD CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset_PETBD") / "PTBD_v20240912.csv",
        help="Input CSV path (default: dataset_PETBD/PTBD_v20240912.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result") / "PTBD_v20240912_balanced_18F.csv",
        help="Output balanced CSV path (default: result/PTBD_v20240912_balanced_18F.csv)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("result") / "PTBD_v20240912_balanced_18F_report.json",
        help="Output JSON report path (default: result/PTBD_v20240912_balanced_18F_report.json)",
    )
    parser.add_argument(
        "--method",
        choices=["undersample", "oversample"],
        default="undersample",
        help="Balancing strategy (default: undersample)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    labeled = label_and_filter(df)
    balanced, report = balance_dataset(labeled, method=args.method, seed=args.seed)

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Save outputs
    balanced.to_csv(args.out, index=False)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved balanced dataset to: {args.out}")
    print(f"Report: {args.report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

