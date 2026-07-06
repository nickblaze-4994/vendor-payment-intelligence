"""Train the offer-conditioned acceptance model for value-based dynamic pricing.

Target: accepted (did the supplier accept THIS offer?)
Features: full vendor economic profile + offer terms (rate, settlement, fee cap).

This is the predictive core of the pricing stack: a downstream optimizer calls
predict_proba across the offer grid per vendor and picks the offer maximizing
expected long-term contribution. Evaluated as probability estimation
(ROC-AUC, PR-AUC, calibration) since the optimizer consumes the probabilities.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CATEGORICAL = [
    "industry", "supplier_margin_profile", "payment_speed_preference",
    "card_acceptance_history", "reconciliation_complexity", "geography",
    "currency", "servicing_cost", "offer_settlement", "offer_fee_cap",
]
DROP = ["vendor_id", "accepted", "mcc"]
TARGET = "accepted"


def main():
    df = pd.read_csv(os.path.join("data", "pricing.csv"))

    encoders = {}
    for col in CATEGORICAL:
        codes, uniques = pd.factorize(df[col].astype(str))
        df[col] = codes
        encoders[col] = {v: i for i, v in enumerate(uniques)}

    y = df[TARGET].astype(int)
    X = df.drop(columns=DROP)
    groups = df["vendor_id"]  # keep a vendor's offers on one side of the split

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=450, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42,
    )
    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"Rows: {len(df)}  |  vendors: {df['vendor_id'].nunique()}  |  "
          f"accept base rate: {y_test.mean():.3f}")
    print(f"ROC-AUC:                 {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC (avg precision):  {average_precision_score(y_test, proba):.3f}")
    print(f"Brier score:             {brier_score_loss(y_test, proba):.3f}")
    print(f"Accuracy @0.5:           {(preds == y_test).mean():.3f}\n")
    print(classification_report(y_test, preds, target_names=["decline", "accept"]))

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False).head(12)
    )
    print("Top feature importances:")
    print(importances.to_string())

    # elasticity check: does predicted acceptance fall as offered rate rises?
    print("\nPredicted accept rate by offered rate (test set):")
    chk = pd.DataFrame({"rate": df.iloc[te]["offer_rate_bps"].values, "p": proba})
    print(chk.groupby("rate")["p"].mean().round(3).to_string())

    os.makedirs("model", exist_ok=True)
    model.save_model(os.path.join("model", "xgb_pricing.json"))
    with open(os.path.join("model", "pricing_encoders.pkl"), "wb") as f:
        pickle.dump({"features": encoders, "columns": list(X.columns)}, f)
    print("\nSaved model/xgb_pricing.json and model/pricing_encoders.pkl")


if __name__ == "__main__":
    main()
