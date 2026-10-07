"""
ml_classifier.py  -  HR question classifier (TF-IDF + Logistic Regression)
==========================================================================

This is the MACHINE LEARNING part of the project.

It does ONE job: read an employee's question and predict which HR policy
document (category) it belongs to. rag_bot.py then uses that category to
guide retrieval. The classifier never answers questions - RAG + the LLM do.

    question  ->  TF-IDF vectoriser  ->  Logistic Regression  ->  category

Files used / created:
    ml_dataset.csv                      training data (question, category, ...)
    ml_artifacts/hr_classifier.joblib   saved model (vectoriser + classifier)
    ml_artifacts/ml_metrics.json        evaluation results (shown in the UI)
    ml_artifacts/confusion_matrix.png   confusion matrix image

Train / evaluate with:   python train_ml_model.py
Quick test with:         python ml_classifier.py "How many casual leaves do I get?"
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# Paths (relative to this file, so it works from any launch folder)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "ml_dataset.csv"
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"
MODEL_PATH = ARTIFACT_DIR / "hr_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "ml_metrics.json"
CONFUSION_PNG = ARTIFACT_DIR / "confusion_matrix.png"

# --------------------------------------------------------------------------
# Category -> source PDF. One category per policy document.
# rag_bot.py uses this to restrict retrieval to the predicted document.
# --------------------------------------------------------------------------
CATEGORY_TO_DOCUMENT = {
    "Company Profile": "00_Company_Profile.pdf",
    "Employee Handbook": "01_Employee_Handbook.pdf",
    "Leave": "02_Leave_Policy.pdf",
    "Work From Home": "03_Work_From_Home_Policy.pdf",
    "Code of Conduct": "04_Code_of_Conduct.pdf",
    "Performance": "05_Performance_Review_Policy.pdf",
    "Compensation and Benefits": "06_Compensation_and_Benefits_Policy.pdf",
    "IT and Data Security": "07_IT_and_Data_Security_Policy.pdf",
    "POSH": "08_Prevention_of_Sexual_Harassment_Policy.pdf",
    "Onboarding and Separation": "09_Onboarding_and_Separation_Policy.pdf",
    "Travel and Expense": "10_Travel_and_Expense_Policy.pdf",
}

# If the classifier's top probability is below this, we do NOT trust it:
# no category is applied and RAG falls back to its normal retrieval.
# This stops the model from forcing unrelated questions into an HR category.
CONFIDENCE_THRESHOLD = 0.30


def build_pipeline(C: float = 10.0) -> Pipeline:
    """TF-IDF (words + word pairs, common stop words removed) followed by
    Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                                  stop_words="english")),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


_model = None  # loaded once, reused for every question


def load_model() -> Pipeline:
    """Load the saved model. If it is missing or was saved by an incompatible
    scikit-learn version, quickly re-train it from ml_dataset.csv instead."""
    global _model
    if _model is not None:
        return _model
    try:
        _model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[ml_classifier] Could not load {MODEL_PATH.name} ({type(e).__name__}); "
              "training from ml_dataset.csv. Run `python train_ml_model.py` to save it.")
        df = pd.read_csv(DATASET_PATH)
        _model = build_pipeline().fit(df["question"], df["category"])
    return _model


def predict_category(question: str) -> str:
    """Simplest API: question -> predicted category (always returns a label)."""
    return str(load_model().predict([question])[0])


def classify_question(question: str) -> dict:
    """Used by the RAG pipeline: category + confidence + a 'trusted' flag.

    Returns:
        {"category":   best guess (always set),
         "confidence": its probability, 0..1,
         "confident":  True if confidence >= CONFIDENCE_THRESHOLD,
         "top3":       [(category, probability), ...]}
    """
    model = load_model()
    probs = model.predict_proba([question])[0]
    order = probs.argsort()[::-1]
    classes = model.classes_
    top3 = [(str(classes[i]), float(probs[i])) for i in order[:3]]
    best, conf = top3[0]
    return {
        "category": best,
        "confidence": conf,
        "confident": conf >= CONFIDENCE_THRESHOLD,
        "top3": top3,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many casual leaves do I get per year?"
    r = classify_question(q)
    print(f"Question  : {q}")
    print(f"Category  : {r['category']}  (confidence {r['confidence']:.0%}, "
          f"{'trusted' if r['confident'] else 'LOW - would not be used'})")
    print("Top 3     :", ", ".join(f"{c} {p:.0%}" for c, p in r["top3"]))
