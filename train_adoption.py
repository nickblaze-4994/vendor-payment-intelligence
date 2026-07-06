"""Train the virtual-card ADOPTION propensity model (Concept B).

Population: vendors currently paid by ACH or check.
Target:     vc_accept (would they accept a virtual card if enrolled?)

Evaluated as a RANKING problem (we act on the top of the list), so the headline
metrics are ROC-AUC, PR-AUC (average precision), precision@k, and calibration
(Brier score) -- not raw accuracy.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CATEGORICAL = ["segment", "mcc", "current_rail", "risk_level"]
BOOL = ["is_mature_ap_program", "is_new_to_platform", "accepts_card_other_channel"]
DROP = ["vendor_id", "vc_accept"]
TARGET = "vc_accept"


def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1][:k]
    return float(np.asarray(y_true)[order].mean())


def main() -> None:
    df = pd.read_csv(os.path.join("data", "adoption.csv"))
    # only the targetable population: currently on ACH or check
    df = df[df["current_rail"].isin(["ach", "check"])].reset_index(drop=True)

    for col in BOOL:
        df[col] = df[col].astype(int)

    encoders = {}
    for col in CATEGORICAL:
        codes, uniques = pd.factorize(df[col].astype(str))
        df[col] = codes
        encoders[col] = {v: i for i, v in enumerate(uniques)}

    y = df[TARGET].astype(int)
    X = df.drop(columns=DROP)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    base_rate = y_test.mean()

    print(f"Positive (accept) base rate:  {base_rate:.3f}")
    print(f"ROC-AUC:                      {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC (avg precision):       {average_precision_score(y_test, proba):.3f}")
    print(f"Brier score (calibration):    {brier_score_loss(y_test, proba):.3f}")
    for k in (50, 100, 250):
        k = min(k, len(y_test))
        print(f"Precision@{k:<4}                 {precision_at_k(y_test, proba, k):.3f}"
              f"   (lift {precision_at_k(y_test, proba, k) / base_rate:.2f}x over base)")
    print("\nClassification report @0.5:")
    print(classification_report(y_test, preds, target_names=["decline", "accept"]))

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("Top feature importances:")
    print(importances.to_string())

    os.makedirs("model", exist_ok=True)
    model.save_model(os.path.join("model", "xgb_adoption.json"))
    with open(os.path.join("model", "adoption_encoders.pkl"), "wb") as f:
        pickle.dump({"features": encoders, "columns": list(X.columns)}, f)
    print("\nSaved model/xgb_adoption.json and model/adoption_encoders.pkl")


if __name__ == "__main__":
    main()
