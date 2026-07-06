"""Business layer: rank enterprise ACH/check vendors (>$100k spend) as
virtual-card conversion targets, by expected annual Corpay revenue.

    EV = P(accept) x annual_spend x CORPAY_NET_TAKE - ENROLL_COST

Uses the trained adoption model (train_adoption.py) to score P(accept).
"""

import os
import pickle

import pandas as pd
from xgboost import XGBClassifier

from features import CORPAY_NET_TAKE, ENROLL_COST, ENTERPRISE_SPEND_GATE

CATEGORICAL = ["segment", "mcc", "current_rail", "risk_level"]
BOOL = ["is_mature_ap_program", "is_new_to_platform", "accepts_card_other_channel"]
TOP_N = 25


def load_model():
    model = XGBClassifier()
    model.load_model(os.path.join("model", "xgb_adoption.json"))
    with open(os.path.join("model", "adoption_encoders.pkl"), "rb") as f:
        art = pickle.load(f)
    return model, art["features"], art["columns"]


def main() -> None:
    df = pd.read_csv(os.path.join("data", "adoption.csv"))

    # actionable population: enterprise, currently ACH/check, above the spend gate
    mask = (
        (df["segment"] == "enterprise")
        & (df["current_rail"].isin(["ach", "check"]))
        & (df["annual_spend"] > ENTERPRISE_SPEND_GATE)
    )
    targets = df[mask].copy().reset_index(drop=True)
    if targets.empty:
        print("No vendors match the enterprise / ACH-check / >$100k filter.")
        return

    model, encoders, columns = load_model()

    X = targets.copy()
    for col in BOOL:
        X[col] = X[col].astype(int)
    for col in CATEGORICAL:
        X[col] = X[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
    X = X[columns]

    targets["p_accept"] = model.predict_proba(X)[:, 1]
    targets["expected_annual_revenue"] = (
        targets["p_accept"] * targets["annual_spend"] * CORPAY_NET_TAKE - ENROLL_COST
    )
    targets = targets.sort_values("expected_annual_revenue", ascending=False)

    cols = [
        "vendor_id", "segment", "mcc", "annual_spend", "current_rail",
        "accepts_card_other_channel", "p_accept", "expected_annual_revenue",
    ]
    out = os.path.join("data", "target_list.csv")
    targets[cols].to_csv(out, index=False)

    total_ev = targets["expected_annual_revenue"].clip(lower=0).sum()
    print(f"Enterprise ACH/check vendors >${ENTERPRISE_SPEND_GATE:,}: {len(targets)}")
    print(f"Total expected annual Corpay revenue if top targets convert: "
          f"${total_ev:,.0f}")
    print(f"(P(accept) x spend x {CORPAY_NET_TAKE:.1%} net take, minus "
          f"${ENROLL_COST:.0f} enrollment cost each)\n")
    print(f"Top {TOP_N} conversion targets:")
    show = targets[cols].head(TOP_N).copy()
    show["annual_spend"] = show["annual_spend"].map("${:,.0f}".format)
    show["p_accept"] = show["p_accept"].map("{:.3f}".format)
    show["expected_annual_revenue"] = show["expected_annual_revenue"].map("${:,.0f}".format)
    print(show.to_string(index=False))
    print(f"\nFull ranked list written to {out}")


if __name__ == "__main__":
    main()
