"""Train an XGBoost multiclass classifier: vendor profile -> metadata tier.

Saves the model to model/xgb_metadata_tier.json and the fitted label
encoders to model/encoders.pkl.
"""

import os
import pickle

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CATEGORICAL = ["mcc", "current_payment_method", "card_acceptance_history", "risk_level"]
TARGET = "metadata_strategy"


def main() -> None:
    df = pd.read_csv(os.path.join("data", "vendors.csv"))
    df["strategic_vendor"] = df["strategic_vendor"].astype(int)
    df["is_new_to_platform"] = df["is_new_to_platform"].astype(int)

    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc

    target_enc = LabelEncoder()
    y = target_enc.fit_transform(df[TARGET])
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(target_enc.classes_),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    weights = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, preds)
    majority_baseline = pd.Series(y_test).value_counts(normalize=True).max()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Macro-F1 rewards catching the minority "deviate from default" strategies,
    # which is where the model earns its keep vs. an always-extract baseline.
    model_macro_f1 = f1_score(y_test, preds, average="macro")
    majority_class = pd.Series(y_train).value_counts().idxmax()
    baseline_macro_f1 = f1_score(
        y_test, [majority_class] * len(y_test), average="macro"
    )

    print(f"Train accuracy:     {train_acc:.3f}")
    print(f"Test accuracy:      {test_acc:.3f}   (gap {train_acc - test_acc:+.3f})")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Majority baseline:  {majority_baseline:.3f}")
    print(f"Macro-F1 model:     {model_macro_f1:.3f}   "
          f"(always-extract baseline: {baseline_macro_f1:.3f})\n")
    print(classification_report(y_test, preds, target_names=target_enc.classes_))
    print("Confusion matrix (rows=true, cols=pred):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, preds),
            index=target_enc.classes_,
            columns=target_enc.classes_,
        ).to_string()
    )

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop feature importances:")
    print(importances.to_string())

    os.makedirs("model", exist_ok=True)
    model.save_model(os.path.join("model", "xgb_metadata_tier.json"))
    with open(os.path.join("model", "encoders.pkl"), "wb") as f:
        pickle.dump(
            {"features": encoders, "target": target_enc, "columns": list(X.columns)}, f
        )
    print("\nSaved model/xgb_metadata_tier.json and model/encoders.pkl")


if __name__ == "__main__":
    main()
