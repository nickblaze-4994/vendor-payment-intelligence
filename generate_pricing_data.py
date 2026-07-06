"""Generate offer-conditioned acceptance data for VALUE-BASED dynamic pricing.

Each row is a (vendor profile, offer terms) pair with a simulated accept/decline
outcome. Training on this lets the model predict P(accept | vendor, offer) so a
downstream optimizer can trade rate against acceptance and pick the offer with
the highest expected long-term contribution.

Uses the full expanded parameter list:
  annual payment volume, average invoice size, supplier margin profile,
  payment-speed preference, card-acceptance history, reconciliation complexity,
  industry, geography & currency, servicing cost.

Acceptance is driven by the supplier's economics (margin, speed value, capability)
and the offer terms (rate, settlement speed, fee cap) -- NOT by how locked-in the
supplier is. Latent elasticity/capability noise keeps genuine Bayes error.
"""

import os

import numpy as np
import pandas as pd

from features import (
    CURRENCIES, FEE_CAP, GEOGRAPHIES, INDUSTRY_BY_MCC, LARGE_INVOICE_THRESHOLD,
    MARGIN_PROFILES, MARGIN_SENSITIVITY, MCC_CARD_FRIENDLINESS, MCC_TICKET,
    RATE_BPS, RECON_COMPLEXITY, RECON_VALUE, SERVICING_COST, SETTLEMENT,
    SPEED_PREFS, SPEED_VALUE, sigmoid,
)

RNG = np.random.default_rng(11)
N_VENDORS = 15000
OFFERS_PER_VENDOR = 2  # sample a couple of offers per vendor -> offer-conditioned data


def make_vendor(rng):
    mcc = int(rng.choice(list(MCC_TICKET)))
    mu, sig = MCC_TICKET[mcc]
    avg_invoice = round(float(rng.lognormal(mu, sig)), 2)
    monthly_txn = int(max(1, rng.gamma(2.0, 6.0) + 1))
    scale = rng.uniform(0.5, 25.0)  # spread across SMB..enterprise volumes
    annual_volume = round(avg_invoice * monthly_txn * 12 * scale, 2)

    margin = rng.choice(MARGIN_PROFILES, p=[0.35, 0.40, 0.25])
    speed_pref = rng.choice(SPEED_PREFS, p=[0.5, 0.3, 0.2])
    recon = rng.choice(RECON_COMPLEXITY, p=[0.4, 0.4, 0.2])
    geography = rng.choice(GEOGRAPHIES, p=[0.8, 0.2])
    currency = "USD" if geography == "domestic" else rng.choice(CURRENCIES, p=[0.1, 0.4, 0.3, 0.1, 0.1])
    servicing = rng.choice(SERVICING_COST, p=[0.5, 0.35, 0.15])

    # card-acceptance history correlates with industry card-friendliness
    cap_base = MCC_CARD_FRIENDLINESS[mcc]
    if rng.random() < 0.35:
        history = "never_offered"
    else:
        p_acc = np.clip(cap_base + rng.normal(0, 0.15), 0.05, 0.95)
        history = "accepted" if rng.random() < p_acc else "rejected"

    return {
        "mcc": mcc,
        "industry": INDUSTRY_BY_MCC[mcc],
        "avg_invoice_size": avg_invoice,
        "monthly_txn_count": monthly_txn,
        "annual_payment_volume": annual_volume,
        "supplier_margin_profile": margin,
        "payment_speed_preference": speed_pref,
        "card_acceptance_history": history,
        "reconciliation_complexity": recon,
        "geography": geography,
        "currency": currency,
        "servicing_cost": servicing,
        # latent, never exported: unobserved elasticity + capability noise
        "_cap": float(np.clip(cap_base + rng.normal(0, 0.12), 0, 1)),
        "_elast_noise": float(rng.normal(0, 0.35)),
    }


def sample_offer(rng):
    return {
        "offer_rate_bps": int(rng.choice(RATE_BPS)),
        "offer_settlement": rng.choice(SETTLEMENT),
        "offer_fee_cap": rng.choice(FEE_CAP),
    }


def accept_prob(v, o):
    """P(accept) as a function of supplier economics and offer terms."""
    margin_sens = MARGIN_SENSITIVITY[v["supplier_margin_profile"]]
    # rate pain: scales with margin sensitivity and the rate offered
    rate_pain = margin_sens * (o["offer_rate_bps"] / 100.0) * 1.15

    # large invoices make an UNCAPPED percentage fee painful in absolute dollars
    # (a % of a $1M invoice is a huge check); a fee cap removes this pain
    if o["offer_fee_cap"] == "none" and v["avg_invoice_size"] >= LARGE_INVOICE_THRESHOLD:
        size_factor = np.log10(v["avg_invoice_size"] / LARGE_INVOICE_THRESHOLD)
        rate_pain += 0.85 * size_factor * (o["offer_rate_bps"] / 100.0)

    # settlement: fast/immediate suppliers value same-day; standard suppliers
    # don't, and immediate suppliers dislike next-day
    speed_val = SPEED_VALUE[v["payment_speed_preference"]]
    if o["offer_settlement"] == "same_day":
        speed_term = 0.9 * speed_val
    else:
        speed_term = -0.6 * speed_val

    # fee cap materially helps large-invoice suppliers (the painful % case)
    cap_term = 0.0
    if o["offer_fee_cap"] == "capped" and v["avg_invoice_size"] >= LARGE_INVOICE_THRESHOLD:
        cap_term = 1.1

    hist_term = {"accepted": 0.8, "never_offered": 0.0, "rejected": -0.7}[v["card_acceptance_history"]]
    recon_term = 0.5 * RECON_VALUE[v["reconciliation_complexity"]]  # value of full metadata
    geo_term = -0.3 if v["geography"] == "cross_border" else 0.0

    logit = (
        0.4
        + 2.4 * v["_cap"]
        + hist_term
        + recon_term
        + speed_term
        + cap_term
        + geo_term
        - rate_pain
        + v["_elast_noise"]
    )
    return float(sigmoid(logit))


def main():
    rows = []
    for i in range(N_VENDORS):
        v = make_vendor(RNG)
        for _ in range(OFFERS_PER_VENDOR):
            o = sample_offer(RNG)
            p = accept_prob(v, o)
            row = {k: val for k, val in v.items() if not k.startswith("_")}
            row["vendor_id"] = f"V{i:06d}"
            row.update(o)
            row["accepted"] = int(RNG.random() < p)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "pricing.csv")
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows ({N_VENDORS} vendors x {OFFERS_PER_VENDOR} offers) to {out}")
    print(f"Overall accept rate: {df['accepted'].mean():.3f}\n")
    print("Accept rate by offered rate (bps):")
    print(df.groupby("offer_rate_bps")["accepted"].mean().round(3).to_string())
    print("\nAccept rate by margin profile:")
    print(df.groupby("supplier_margin_profile")["accepted"].mean().round(3).to_string())
    print("\nAccept rate by settlement x speed preference:")
    print(pd.crosstab(df["payment_speed_preference"], df["offer_settlement"],
                      values=df["accepted"], aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
