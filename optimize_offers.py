"""Value-based offer optimizer.

For each vendor, sweep the core-lever offer grid (rate x settlement x fee-cap),
score P(accept) with the trained acceptance model, and pick the offer with the
highest expected LONG-TERM contribution -- subject to a policy floor/ceiling and
a sustainability constraint. An ACH flat-fee fallback competes with the card
offers, so irrational-percentage cases (huge invoices) fall back cleanly.

Objective (illustrative, per vendor):
    annual_card = card_gross(offer) - servicing_cost
    retention   = 0.60 + 0.35 * P(accept)          # comfortable acceptance stays
    E[LTV_card] = P(accept) * annual_card * geom(retention, H)
                  + (1 - P(accept)) * ach_annual * H     # decline -> stays on ACH
    E[LTV_ach]  = ach_annual * H                          # certain, low margin

Policy: only offers with P(accept) >= MIN_SUSTAINABLE_ACCEPT are eligible; if no
card offer qualifies, recommend the ACH fallback. This encodes "keep acceptance
sustainable" instead of maximizing conditional fee on reluctant suppliers.
"""

import itertools
import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import (
    ACH_FLAT_FEE, FEE_CAP, FEE_CAP_PER_TXN, GEO_COST_BPS, RATE_BPS,
    SAME_DAY_COST_BPS, SERVICING_COST_ANNUAL, SETTLEMENT,
)

HORIZON = 3
MIN_SUSTAINABLE_ACCEPT = 0.40
CATEGORICAL = [
    "industry", "supplier_margin_profile", "payment_speed_preference",
    "card_acceptance_history", "reconciliation_complexity", "geography",
    "currency", "servicing_cost", "offer_settlement", "offer_fee_cap",
]


def load_model():
    model = XGBClassifier()
    model.load_model(os.path.join("model", "xgb_pricing.json"))
    with open(os.path.join("model", "pricing_encoders.pkl"), "rb") as f:
        art = pickle.load(f)
    return model, art["features"], art["columns"]


def geom(r, n):
    """1 + r + r^2 + ... + r^(n-1)."""
    return sum(r ** t for t in range(n))


def candidate_offers():
    return [{"offer_rate_bps": r, "offer_settlement": s, "offer_fee_cap": c}
            for r, s, c in itertools.product(RATE_BPS, SETTLEMENT, FEE_CAP)]


def score(model, encoders, columns, frame):
    X = frame.copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
    return model.predict_proba(X[columns])[:, 1]


def _annual_card(v, o):
    annual_txns = max(1.0, v["annual_payment_volume"] / v["avg_invoice_size"])
    net_bps = o["offer_rate_bps"]
    if o["offer_settlement"] == "same_day":
        net_bps -= SAME_DAY_COST_BPS
    net_bps -= GEO_COST_BPS[v["geography"]]
    per_txn = v["avg_invoice_size"] * net_bps / 10000.0
    if o["offer_fee_cap"] == "capped":
        per_txn = min(per_txn, FEE_CAP_PER_TXN)
    card_gross = per_txn * annual_txns
    return card_gross - SERVICING_COST_ANNUAL[v["servicing_cost"]]


def _ach_annual(v):
    annual_txns = max(1.0, v["annual_payment_volume"] / v["avg_invoice_size"])
    return ACH_FLAT_FEE * annual_txns - SERVICING_COST_ANNUAL["low"]


def evaluate(v, model, encoders, columns):
    """Return a candidate table (all offers + ACH) with P(accept) and E[LTV]."""
    offers = candidate_offers()
    frame = pd.DataFrame([{**{k: v[k] for k in v if not k.startswith("_")}, **o}
                          for o in offers])
    p = score(model, encoders, columns, frame)

    ach = _ach_annual(v)
    rows = []
    for o, pa in zip(offers, p):
        annual_card = _annual_card(v, o)
        retention = 0.60 + 0.35 * pa
        e_ltv = pa * annual_card * geom(retention, HORIZON) + (1 - pa) * ach * HORIZON
        rows.append({
            "rate_bps": o["offer_rate_bps"], "settlement": o["offer_settlement"],
            "fee_cap": o["offer_fee_cap"], "p_accept": pa,
            "annual_if_accept": annual_card, "e_ltv": e_ltv,
            "sustainable": pa >= MIN_SUSTAINABLE_ACCEPT, "rail": "virtual_card",
        })
    # ACH fallback candidate
    rows.append({
        "rate_bps": 0, "settlement": "next_day", "fee_cap": "n/a", "p_accept": 0.97,
        "annual_if_accept": ach, "e_ltv": ach * HORIZON, "sustainable": True, "rail": "ach_flat",
    })
    return pd.DataFrame(rows)


def recommend(v, model, encoders, columns):
    cand = evaluate(v, model, encoders, columns)
    eligible = cand[cand["sustainable"]]
    pool = eligible if not eligible.empty else cand
    best = pool.loc[pool["e_ltv"].idxmax()]
    return best, cand


def main():
    model, encoders, columns = load_model()

    # ---- portfolio pass over all vendors ----
    df = pd.read_csv(os.path.join("data", "pricing.csv"))
    profile_cols = [c for c in df.columns if not c.startswith("offer_")
                    and c not in ("accepted",)]
    vendors = df.drop_duplicates("vendor_id")[profile_cols].reset_index(drop=True)

    recs = []
    for _, row in vendors.iterrows():
        best, _ = recommend(row.to_dict(), model, encoders, columns)
        recs.append({"vendor_id": row["vendor_id"], "rail": best["rail"],
                     "rate_bps": int(best["rate_bps"]), "settlement": best["settlement"],
                     "fee_cap": best["fee_cap"], "p_accept": round(float(best["p_accept"]), 3),
                     "e_ltv": round(float(best["e_ltv"]), 2)})
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join("data", "offers.csv"), index=False)

    print(f"Optimized offers for {len(out)} vendors -> data/offers.csv\n")
    print("Recommended rail mix:")
    print(out["rail"].value_counts().to_string())
    print("\nRecommended card rate (bps) distribution:")
    print(out[out.rail == "virtual_card"]["rate_bps"].value_counts().sort_index().to_string())
    print(f"\nMean recommended card rate: "
          f"{out[out.rail=='virtual_card']['rate_bps'].mean():.0f} bps")

    # ---- worked examples (Supplier A / B / C) ----
    examples = {
        "A  $20M, thin margin, can demand ACH": {
            "industry": "freight_logistics", "avg_invoice_size": 50000.0,
            "monthly_txn_count": 33, "annual_payment_volume": 20_000_000.0,
            "supplier_margin_profile": "thin", "payment_speed_preference": "standard",
            "card_acceptance_history": "rejected", "reconciliation_complexity": "medium",
            "geography": "domestic", "currency": "USD", "servicing_cost": "medium",
        },
        "B  $200k, wants immediate liquidity, accepts cards": {
            "industry": "professional_services", "avg_invoice_size": 3000.0,
            "monthly_txn_count": 6, "annual_payment_volume": 200_000.0,
            "supplier_margin_profile": "moderate", "payment_speed_preference": "immediate",
            "card_acceptance_history": "accepted", "reconciliation_complexity": "low",
            "geography": "domestic", "currency": "USD", "servicing_cost": "low",
        },
        "C  $1M invoices, rejects percentage pricing": {
            "industry": "construction", "avg_invoice_size": 1_000_000.0,
            "monthly_txn_count": 1, "annual_payment_volume": 2_000_000.0,
            "supplier_margin_profile": "moderate", "payment_speed_preference": "standard",
            "card_acceptance_history": "rejected", "reconciliation_complexity": "high",
            "geography": "domestic", "currency": "USD", "servicing_cost": "high",
        },
    }
    for name, v in examples.items():
        best, cand = recommend(v, model, encoders, columns)
        print("\n" + "=" * 70)
        print(f"Supplier {name}")
        rail = best["rail"]
        if rail == "ach_flat":
            print(f"  -> RECOMMEND: ACH flat fee (${ACH_FLAT_FEE:.0f}/payment)  "
                  f"E[LTV]=${best['e_ltv']:,.0f}")
        else:
            print(f"  -> RECOMMEND: virtual card @ {int(best['rate_bps'])} bps "
                  f"({int(best['rate_bps'])/100:.2f}%), {best['settlement']}, "
                  f"fee_cap={best['fee_cap']}  |  P(accept)={best['p_accept']:.2f}  "
                  f"E[LTV]=${best['e_ltv']:,.0f}")
        top = cand.sort_values("e_ltv", ascending=False).head(5).copy()
        top["p_accept"] = top["p_accept"].map("{:.2f}".format)
        top["annual_if_accept"] = top["annual_if_accept"].map("${:,.0f}".format)
        top["e_ltv"] = top["e_ltv"].map("${:,.0f}".format)
        print(top[["rail", "rate_bps", "settlement", "fee_cap", "p_accept",
                   "annual_if_accept", "e_ltv", "sustainable"]].to_string(index=False))


if __name__ == "__main__":
    main()
