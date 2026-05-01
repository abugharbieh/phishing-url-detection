# src/predict.py
# Single URL prediction (CLI)
# Usage:
# python src/predict.py "https://example.com/login"

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
from features import extract_features

BASE_DIR = SRC_DIR.parent
MODEL_PATH = BASE_DIR / "artifacts" / "phishing_rf_pipeline.joblib"

def main():
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "https://example.com/login"')
        sys.exit(1)

    url = sys.argv[1].strip()
    if not url:
        print("Error: URL is empty.")
        sys.exit(1)

    # Normalize: if user enters without scheme, add https://
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at: {MODEL_PATH}")
        print("Run: python src/train.py")
        sys.exit(1)

    # Extract URL-only features (51 columns)
    features_df = extract_features(url)

    # Load trained pipeline
    model = joblib.load(MODEL_PATH)

    # Predict class
    prediction = model.predict(features_df)[0]

    # Predict probabilities
    probabilities = model.predict_proba(features_df)[0]
    class_to_prob = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}

    confidence = class_to_prob[prediction]

    print("\nInput URL:")
    print(url)

    print("\nPrediction:")
    print(prediction)

    print("\nConfidence (probability of predicted class):")
    print(f"{confidence:.2f}")

    print("\nClass probabilities:")
    for cls, prob in class_to_prob.items():
        print(f"  {cls}: {prob:.2f}")

if __name__ == "__main__":
    main()