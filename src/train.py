# src/train.py
# Train final deployable pipeline on FULL dataset (no CV here)
# Saves: artifacts/phishing_rf_pipeline.joblib

from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import CorrelationFilter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_phishing_augmented.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "phishing_rf_pipeline.joblib"

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

def main():
    df = pd.read_csv(DATA_PATH)

    required = set(URL_ONLY_KEEP + ["status"])
    missing = list(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    X = df[URL_ONLY_KEEP].copy()
    y = df["status"].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Non-numeric values detected in feature columns.")

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

    # Fit using DataFrame to preserve column names
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    print("Trained on full dataset.")
    print("Saved model ->", MODEL_PATH)

if __name__ == "__main__":
    main()