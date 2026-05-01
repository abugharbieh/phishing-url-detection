# src/evaluate.py
# Deterministic 10-fold CV evaluation (no random_state, no shuffle)
# URL-only feature subset

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from preprocessing import CorrelationFilter


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_phishing_augmented.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "metrics.json"
CM_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"


# URL-only feature subset
URL_ONLY_KEEP = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_or",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "path_extension",
    "nb_redirection",
    "nb_external_redirection",
    "length_words_raw",
    "char_repeat",
    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",
    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",
    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",
    "phish_hints",
]


def save_confusion_matrix(cm, labels, path):
    fig, ax = plt.subplots()
    img = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion Matrix (10-fold CV)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Check required columns
    required = set(URL_ONLY_KEEP + ["status"])
    missing = list(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    print("URL-only features:", len(URL_ONLY_KEEP))

    X = df[URL_ONLY_KEEP].copy()
    y = df["status"].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Non-numeric values detected in feature columns.")

    # Pipeline
    pipeline = Pipeline([
        ("corr_filter", CorrelationFilter(threshold=0.90)),
        ("rf", RandomForestClassifier(
            n_estimators=600,
            criterion="entropy",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="log2",
            bootstrap=False,
            n_jobs=-1
        ))
    ])

    # Deterministic CV
    cv = StratifiedKFold(n_splits=10, shuffle=False)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    # Cross-validation
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    metrics = {
        "rows": int(df.shape[0]),
        "features_used": len(URL_ONLY_KEEP),
        "target": "status",
        "class_distribution": y.value_counts().to_dict(),
        "cv_splits": 10,
        "shuffle": False,
        "scores": {}
    }

    for key in scoring:
        fold_scores = np.array(results[f"test_{key}"])
        metrics["scores"][key] = {
            "mean": float(fold_scores.mean()),
            "std": float(fold_scores.std()),
            "per_fold": fold_scores.tolist()
        }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", METRICS_PATH)

    # Confusion matrix
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y, y_pred, labels=labels)

    save_confusion_matrix(cm, labels, CM_PNG_PATH)
    print("Saved confusion matrix:", CM_PNG_PATH)

    # Console summary
    print("\n10-Fold CV Results (macro):")
    for key in scoring:
        print(f"{key.capitalize():<10}: {metrics['scores'][key]['mean']:.4f} ± {metrics['scores'][key]['std']:.4f}")

if __name__ == "__main__":
    main()