"""
5-Fold CV Training (Classical ML) for HGG vs LGG

- Uses GitHub-friendly, OS-independent paths
- No hard-coded absolute paths
- Safe to run on Windows / Linux / macOS / GitHub Actions

Run from thesis root:
    python -m ml_pipeline.training.train_classifier_cv
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import joblib


# 1) Project paths 

# train_classifier_cv.py
# ml_pipeline/training/train_classifier_cv.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "ml_features"
MODELS_DIR = OUTPUTS_DIR / "ml_models"

FEATURES_CSV = FEATURES_DIR / "features_from_gtmask.csv"


# 2) Configuration 

@dataclass
class MLConfig:
    k_folds: int = 5
    seed: int = 42

    non_feature_cols: Tuple[str, ...] = (
        "patient_id",
        "label_hgg",
        "who_cns_grade",
        "fold",
    )


# 3) Metrics computation

def compute_metrics_binary(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": prec,
        "recall_sensitivity": rec,
        "f1": f1,
        "specificity": spec,
    }


# 4) Model building

def build_models(seed: int):
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


# 5) Main runner

def main():
    cfg = MLConfig()

    print(">>> ML classification started")
    print("OS:", platform.system())
    print("Project root:", PROJECT_ROOT)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing features CSV:\n{FEATURES_CSV}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = pd.read_csv(FEATURES_CSV)

    df["label_hgg"] = df["label_hgg"].astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in cfg.non_feature_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    X = df[feature_cols]
    y = df["label_hgg"].values

    print(f"Patients: {len(df)}")
    print(f"Features used: {len(feature_cols)}")

    skf = StratifiedKFold(
        n_splits=cfg.k_folds,
        shuffle=True,
        random_state=cfg.seed,
    )

    models = build_models(cfg.seed)

    all_rows = []
    best_model_name = None
    best_auc = -1.0

    for name, model in models.items():
        print(f"\n--- {name} ---")
        aucs = []

        for fold, (tr, va) in enumerate(skf.split(X, y)):
            model.fit(X.iloc[tr], y[tr])
            y_prob = model.predict_proba(X.iloc[va])[:, 1]

            m = compute_metrics_binary(y[va], y_prob)
            aucs.append(m["roc_auc"])

            all_rows.append({
                "model": name,
                "fold": fold,
                **m,
            })

            print(f"Fold {fold}: AUC={m['roc_auc']:.3f}")

        mean_auc = np.nanmean(aucs)
        print(f"Mean AUC: {mean_auc:.3f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_name = name

    # Save metrics
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(MODELS_DIR / "per_fold_metrics.csv", index=False)

    # Refit best model
    best_model = build_models(cfg.seed)[best_model_name]
    best_model.fit(X, y)

    joblib.dump({
        "model": best_model,
        "features": feature_cols,
    }, MODELS_DIR / "best_model.joblib")

    summary = {
        "best_model": best_model_name,
        "best_mean_auc": best_auc,
    }

    with open(MODELS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 6) Print summary
    
    print("\n=== Training done ===")
    print("Best model:", best_model_name)
    print("Saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()
