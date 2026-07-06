"""Recommend a metadata-release strategy (the dynamic-pricing knob) for a vendor.

Usage:
    python predict.py --json '{"mcc": 5085, "avg_ticket": 12000, ...}'
    python predict.py --file vendor.json
"""

import argparse
import json
import os
import pickle

import pandas as pd
from xgboost import XGBClassifier

# Illustrative effective interchange by strategy — actual rates are set by the
# card networks and vary by program, region, and qualification.
RATE = {
    "send_minimal": 0.025,   # withhold metadata -> highest interchange (extract)
    "send_partial": 0.019,
    "send_full": 0.012,      # release all metadata -> lowest interchange (acquire)
}
STANCE = {
    "send_minimal": "EXTRACT — withhold metadata, keep high interchange (vendor is locked in)",
    "send_partial": "BALANCE — release partial metadata",
    "send_full": "ACQUIRE — release full metadata, low interchange to win/retain the vendor",
}

CATEGORICAL = ["mcc", "current_payment_method", "card_acceptance_history", "risk_level"]


def load_artifacts():
    model = XGBClassifier()
    model.load_model(os.path.join("model", "xgb_metadata_tier.json"))
    with open(os.path.join("model", "encoders.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    return model, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Vendor profile as a JSON string")
    group.add_argument("--file", help="Path to a JSON file with the vendor profile")
    args = parser.parse_args()

    if args.json:
        profile = json.loads(args.json)
    else:
        with open(args.file) as f:
            profile = json.load(f)

    model, artifacts = load_artifacts()
    encoders, target_enc, columns = (
        artifacts["features"],
        artifacts["target"],
        artifacts["columns"],
    )

    row = pd.DataFrame([profile])
    for flag in ("strategic_vendor", "is_new_to_platform"):
        row[flag] = row.get(flag, False)
        row[flag] = row[flag].astype(int)
    for col in CATEGORICAL:
        row[col] = encoders[col].transform(row[col].astype(str))
    row = row[columns]

    proba = model.predict_proba(row)[0]
    strategy = target_enc.classes_[proba.argmax()]

    print(f"Recommended strategy: {strategy}")
    print(f"  {STANCE[strategy]}\n")
    print("Class probabilities:")
    for cls, p in sorted(zip(target_enc.classes_, proba), key=lambda x: -x[1]):
        print(f"  {cls:<14} {p:.3f}")

    rate = RATE[strategy]
    print(f"\nIllustrative interchange: {rate:.1%} of transaction value")
    if "annual_spend" in profile:
        rev = profile["annual_spend"] * rate
        print(f"  Projected annual issuer revenue at this rate: "
              f"${rev:,.0f}  (on ${profile['annual_spend']:,.0f} spend)")
        # what withholding would earn vs full release, to show the tradeoff
        hi = profile["annual_spend"] * RATE["send_minimal"]
        lo = profile["annual_spend"] * RATE["send_full"]
        print(f"  Revenue band across strategies: ${lo:,.0f} (full) .. "
              f"${hi:,.0f} (minimal)")


if __name__ == "__main__":
    main()
