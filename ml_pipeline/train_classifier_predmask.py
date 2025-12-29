"""
ML Classification (5-Fold CV) Using Features from Predicted Masks

Goal:
  Train an ML classifier (HGG vs LGG) using features extracted from
  the DL-predicted segmentation masks (not GT masks).

Why this matters:
  - GT-mask features show an "upper bound" (ideal segmentation).
  - Pred-mask features show the realistic end-to-end pipeline:
      MRI -> DL segmentation -> ML grade classification

Expected input:
  outputs/ml_predmask_features/features_from_predmask.csv

Outputs:
  outputs/ml_models_predmask/
    - per_fold_metrics.csv
    - metrics_summary.json
    - best_model.joblib

Run:
  python -m ml_pipeline.train_classifier_predmask
"""

import csv
import json
import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import joblib


# 1) Config

@dataclass
class Config:
    k_folds: int = 5
    seed: int = 42

    # Input (predicted-mask features)
    features_csv_rel: str = os.path.join(
        "outputs", "ml_predmask_features", "features_from_predmask.csv"
    )

    # Output folder for ML models trained on predicted masks
    out_dir_rel: str = os.path.join("outputs", "ml_models_predmask")

    # Which label column to use
    label_col: str = "label_hgg"

    # Optional: columns to always ignore if present
    ignore_cols: Tuple[str, ...] = (
        "patient_id",
        "fold",
        "split",
    )


# 2) Paths (GitHub-friendly)

def get_project_root() -> str:
    """
    This file is expected in: <root>/ml_pipeline/train_classifier_predmask.py
    So project root is one level above ml_pipeline/.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def resolve_paths(cfg: Config) -> Tuple[str, str, str]:
    root = get_project_root()
    features_csv = os.path.join(root, cfg.features_csv_rel)
    out_dir = os.path.join(root, cfg.out_dir_rel)
    os.makedirs(out_dir, exist_ok=True)
    return root, features_csv, out_dir


# 3) Data loading

def read_features_csv(
    csv_path: str,
    label_col: str,
    ignore_cols: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read a features CSV and return:
      X: (N, D) float matrix
      y: (N,) int labels
      feature_names: list of D feature names
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Features CSV not found:\n{csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("Features CSV is empty (0 rows).")

    fieldnames = list(rows[0].keys())
    if label_col not in fieldnames:
        raise ValueError(
            f"Label column '{label_col}' not found. Available columns:\n{fieldnames}"
        )

    # Features = all columns except label + ignore list
    feature_names = []
    for c in fieldnames:
        if c == label_col:
            continue
        if c in ignore_cols:
            continue
        feature_names.append(c)

    X_list: List[List[float]] = []
    y_list: List[int] = []

    for r in rows:
        lab_str = str(r.get(label_col, "")).strip()
        if lab_str == "":
            raise ValueError("Found empty label_hgg in the features CSV.")
        y_list.append(int(float(lab_str)))

        feats: List[float] = []
        for c in feature_names:
            s = str(r.get(c, "")).strip()
            if s == "" or s.lower() in {"na", "nan", "none"}:
                feats.append(np.nan)
            else:
                feats.append(float(s))
        X_list.append(feats)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    return X, y, feature_names


# 4) Metrics computation

def specificity_from_cm(cm: np.ndarray) -> float:
    """
    Confusion matrix for labels [0,1]:
      [[TN, FP],
       [FN, TP]]
    Specificity = TN / (TN + FP)
    """
    tn, fp, fn, tp = cm.ravel()
    denom = (tn + fp)
    return float(tn / denom) if denom > 0 else 0.0


def eval_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spec = specificity_from_cm(cm)

    return {
        "roc_auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "f1": f1,
        "specificity": spec,
    }


# 5) Model construction

def make_models(cfg: Config) -> Dict[str, object]:
    """
    Two baselines:
      - Logistic Regression (scaled)
      - Random Forest (no scaling needed)
    """
    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=cfg.seed)),
        ]
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                random_state=cfg.seed,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ]
    )

    return {"logreg": logreg, "rf": rf}


# 6) Cross-validation

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, object],
    cfg: Config,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)

    per_fold_rows: List[Dict[str, object]] = []
    metrics_by_model: Dict[str, List[Dict[str, float]]] = {m: [] for m in models.keys()}

    for model_name, est in models.items():
        print(f"\n--- {model_name} (pred-mask features) ---")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            est.fit(X_tr, y_tr)

            y_prob = est.predict_proba(X_va)[:, 1]
            y_pred = est.predict(X_va)

            m = eval_binary_metrics(y_va, y_prob, y_pred)
            metrics_by_model[model_name].append(m)

            print(f"Fold {fold_idx}: AUC={m['roc_auc']:.3f}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}")

            per_fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "accuracy": m["accuracy"],
                    "roc_auc": m["roc_auc"],
                    "precision": m["precision"],
                    "recall_sensitivity": m["recall_sensitivity"],
                    "f1": m["f1"],
                    "specificity": m["specificity"],
                }
            )

    # summary means/stdevs per model
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, fold_metrics in metrics_by_model.items():
        summary[model_name] = {}
        for key in fold_metrics[0].keys():
            summary[model_name][f"mean_{key}"] = float(np.mean([fm[key] for fm in fold_metrics]))
            summary[model_name][f"std_{key}"] = float(np.std([fm[key] for fm in fold_metrics], ddof=0))

    return per_fold_rows, summary


def pick_best_model_by_mean_auc(summary: Dict[str, Dict[str, float]]) -> str:
    best_name = None
    best_auc = -1.0
    for name, stats in summary.items():
        auc = stats.get("mean_roc_auc", -1.0)
        if auc > best_auc:
            best_auc = auc
            best_name = name
    if best_name is None:
        raise RuntimeError("Could not select best model (summary was empty).")
    return best_name


# 7) Main function

def main():
    cfg = Config()
    root, features_csv, out_dir = resolve_paths(cfg)

    print("OS name (Python):", os.name)
    print("Platform:", platform.system())
    print("Project root:", root)
    print("Features CSV:", features_csv)
    print("Output dir:", out_dir)

    X, y, feature_names = read_features_csv(
        csv_path=features_csv,
        label_col=cfg.label_col,
        ignore_cols=cfg.ignore_cols,
    )

    print("Patients:", X.shape[0])
    print("Features used:", X.shape[1])

    models = make_models(cfg)

    # 1) CV evaluation
    per_fold_rows, summary = run_cv(X, y, models, cfg)

    # 2) Save per-fold CSV
    per_fold_csv = os.path.join(out_dir, "per_fold_metrics.csv")
    with open(per_fold_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "fold", "accuracy", "roc_auc", "precision", "recall_sensitivity", "f1", "specificity"],
        )
        writer.writeheader()
        writer.writerows(per_fold_rows)

    # 3) Pick best model by mean AUC
    best_model_name = pick_best_model_by_mean_auc(summary)
    best_mean_auc = float(summary[best_model_name]["mean_roc_auc"])

    # 4) Fit best model on ALL data
    best_model = models[best_model_name]
    best_model.fit(X, y)

    # 5) Save best model
    best_model_path = os.path.join(out_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)

    # 6) Save summary JSON
    summary_json_path = os.path.join(out_dir, "metrics_summary.json")
    full_summary = {
        "k_folds": cfg.k_folds,
        "seed": cfg.seed,
        "n_patients": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features_csv": os.path.relpath(features_csv, root),
        "out_dir": os.path.relpath(out_dir, root),
        "label_col": cfg.label_col,
        "summary_by_model": summary,
        "best_model": best_model_name,
        "best_mean_auc": best_mean_auc,
        "best_model_path": os.path.relpath(best_model_path, root),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    # 7) Print quick summary
    print("\n=== ML Classification (Pred-mask Features) Evaluation is complete ===")
    for mname, stats in summary.items():
        print(f"{mname}: mean AUC={stats['mean_roc_auc']:.3f}  std AUC={stats['std_roc_auc']:.3f}")

    print("\nBest model:", best_model_name)
    print("Best mean AUC:", best_mean_auc)
    print("Saved per-fold metrics:", per_fold_csv)
    print("Saved summary JSON:", summary_json_path)
    print("Saved best model:", best_model_path)


if __name__ == "__main__":
    main()
