"""
train_ml_model.py  -  Train and evaluate the HR question classifier
===================================================================

Run:   python train_ml_model.py

Steps:
  1. Load ml_dataset.csv (questions written from the HR policy PDFs)
  2. Split into train (80%) and test (20%), stratified by category
  3. TF-IDF + Logistic Regression; pick C using 5-fold CV on the TRAIN set only
  4. Evaluate on the untouched TEST set: accuracy, precision, recall, F1,
     confusion matrix (+ 5-fold CV on the whole dataset as a stability check)
  5. Save the model, metrics (JSON) and the confusion-matrix image
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

from ml_classifier import (ARTIFACT_DIR, BASE_DIR, CONFIDENCE_THRESHOLD,
                           CONFUSION_PNG, DATASET_PATH, METRICS_PATH,
                           MODEL_PATH, build_pipeline)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def check_no_leakage():
    """Sanity check: no training question may be identical to a competition
    test.csv question. test.csv is only READ here to verify - never trained on."""
    test_csv = BASE_DIR / "test.csv"
    if not test_csv.exists():
        return "test.csv not found - check skipped"
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum() or ch == " ").strip()
    mine = {norm(q) for q in pd.read_csv(DATASET_PATH)["question"]}
    theirs = {norm(q) for q in pd.read_csv(test_csv)["question"]}
    overlap = mine & theirs
    return f"{len(overlap)} identical questions with test.csv" + (" - REMOVE THEM!" if overlap else " (good)")


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    X, y = df["question"], df["category"]
    labels = sorted(y.unique())
    print(f"Dataset: {len(df)} questions, {len(labels)} categories")
    print("Leakage check:", check_no_leakage())

    # 2. split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # 3. choose C by cross-validation on the training set only
    grid = GridSearchCV(build_pipeline(), {"clf__C": [1, 5, 10, 50, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                        scoring="accuracy")
    grid.fit(X_train, y_train)
    best_C = grid.best_params_["clf__C"]
    model = grid.best_estimator_
    print(f"Best C = {best_C}  (CV accuracy on train: {grid.best_score_:.3f})")

    # 4. evaluate on the held-out test set
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    cv_scores = cross_val_score(build_pipeline(best_C), X, y,
                                cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))

    # confidence analysis on the test set (how the threshold behaves)
    probs = model.predict_proba(X_test)
    top = probs.max(axis=1)
    correct = (model.classes_[probs.argmax(axis=1)] == y_test.values)
    conf_rows = []
    for t in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        m = top >= t
        conf_rows.append({"threshold": t, "kept": float(m.mean()),
                          "accuracy_when_kept": float(correct[m].mean()) if m.any() else None})

    # what the model learned: top TF-IDF terms per category
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coefs = model.named_steps["clf"].coef_
    top_terms = {c: vocab[coefs[i].argsort()[::-1][:6]].tolist()
                 for i, c in enumerate(model.named_steps["clf"].classes_)}

    metrics = {
        "dataset_size": int(len(df)), "n_classes": len(labels), "labels": labels,
        "class_counts": y.value_counts().sort_index().to_dict(),
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "best_C": best_C, "confidence_threshold": CONFIDENCE_THRESHOLD,
        "accuracy": acc,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "cv5_accuracy_mean": float(cv_scores.mean()), "cv5_accuracy_std": float(cv_scores.std()),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in labels},
        "confusion_matrix": cm.tolist(),
        "confidence_analysis": conf_rows,
        "top_terms": top_terms,
        "misclassified": [{"question": q, "true": t, "predicted": p}
                          for q, t, p in zip(X_test, y_test, pred) if t != p],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # confusion-matrix picture
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("True category")
    ax.set_title(f"Confusion matrix (test set, n={len(X_test)}, accuracy={acc:.1%})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=130); plt.close(fig)

    # 5. save the model (TF-IDF vectoriser + Logistic Regression in one object)
    joblib.dump(model, MODEL_PATH)

    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy           : {acc:.3f}")
    print(f"Precision (weighted): {p_w:.3f}   (macro {p_m:.3f})")
    print(f"Recall    (weighted): {r_w:.3f}   (macro {r_m:.3f})")
    print(f"F1-score  (weighted): {f_w:.3f}   (macro {f_m:.3f})")
    print(f"5-fold CV accuracy : {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print("\n" + classification_report(y_test, pred, labels=labels, zero_division=0))
    print("Misclassified:", *[f"\n  '{m['question']}' true={m['true']} pred={m['predicted']}" for m in metrics["misclassified"]] or ["none"])
    print(f"\nSaved: {MODEL_PATH.name}, {METRICS_PATH.name}, {CONFUSION_PNG.name} in {ARTIFACT_DIR.name}/")


if __name__ == "__main__":
    main()
