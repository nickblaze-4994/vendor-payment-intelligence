"""Generate synthetic vendor data for a DYNAMIC-PRICING metadata strategy.

Framing
-------
Corpay (the issuer) has *all* the metadata Visa needs. The lever is how much
of it we *release* on each virtual-card authorization. More metadata -> the
transaction qualifies for a lower interchange program -> the vendor pays less.
Less metadata -> higher interchange -> more issuer revenue per dollar of spend.

So this is yield management, not data availability. Per vendor we choose a
release strategy:

    send_full     -> richest metadata, LOWEST interchange  (acquisition lever)
    send_partial  -> middle
    send_minimal  -> leanest metadata, HIGHEST interchange  (extraction lever)

The optimal knob depends on the vendor's position in the network:

  * New / fee-elastic vendors have near-zero switching cost. Push a high fee
    and they refuse the card (or churn), so you earn nothing. Release full
    metadata, give them the low rate, WIN the relationship and its future
    volume.

  * Entrenched vendors -- high tenure, huge annual volume, high switching
    cost -- are fee-INELASTIC. They will not leave over a few basis points.
    Withhold metadata, keep the high interchange. Because revenue = volume x
    rate, a small rate increase on an enormous spender is large absolute
    revenue. This is the "trillions in volume" extraction case.

Labels come from an expected-revenue-over-horizon objective with a churn
model + multiplicative noise, NOT from a rule over the observed features, so
classes overlap and there is irreducible Bayes error.
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 20000
HORIZON_YEARS = 3.0

STRATEGIES = ["send_minimal", "send_partial", "send_full"]
# Illustrative effective interchange (= vendor cost = issuer revenue share).
RATE = {"send_minimal": 0.025, "send_partial": 0.019, "send_full": 0.012}

MCC_TICKET = {
    5065: (7.0, 1.0), 5085: (7.4, 1.1), 5111: (5.8, 0.8), 5122: (7.8, 1.0),
    5137: (6.2, 0.9), 5169: (7.6, 1.1), 5172: (8.4, 1.0), 5199: (6.5, 1.0),
    5211: (7.9, 1.1), 5734: (7.2, 1.4), 7372: (8.1, 1.3), 7392: (8.3, 1.2),
    7399: (6.8, 1.2), 4214: (8.6, 1.0), 8911: (8.8, 1.2),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    mccs = RNG.choice(list(MCC_TICKET), size=N)

    # ---- latent traits (never exported) ----
    fee_sens = RNG.beta(2.5, 2.5, size=N)      # price elasticity
    base_loyalty = RNG.beta(2.0, 2.5, size=N)  # inherent stickiness
    ops_soph = RNG.beta(3.0, 2.0, size=N)      # invoice cleanliness (proxy noise)

    # ---- true activity ----
    mu = np.array([MCC_TICKET[m][0] for m in mccs])
    sig = np.array([MCC_TICKET[m][1] for m in mccs])
    avg_ticket = np.round(RNG.lognormal(mu, sig), 2)
    txn_rate = RNG.gamma(shape=2.0, scale=6.0, size=N) + 1
    monthly_txn = np.maximum(1, RNG.poisson(txn_rate).astype(int))
    annual_spend = np.round(avg_ticket * monthly_txn * 12, 2)

    # Heavy tail: ~4% of vendors are "whales" processing enormous volume.
    whale = RNG.random(N) < 0.04
    annual_spend[whale] *= RNG.uniform(30, 300, size=whale.sum())

    # Tenure: bimodal — a wave of new vendors + a base of long-tenured ones.
    is_newcomer = RNG.random(N) < 0.30
    tenure_months = np.where(
        is_newcomer,
        RNG.integers(0, 6, size=N),
        RNG.integers(6, 180, size=N),
    ).astype(int)

    rows = []
    for i in range(N):
        # ---- entrenchment / switching cost ----
        tenure_norm = min(tenure_months[i] / 60.0, 1.0)          # cap at 5 yrs
        vol_norm = np.clip((np.log1p(annual_spend[i]) - 10) / 6.0, 0, 1)
        lock_in = sigmoid(-2.2 + 2.2 * tenure_norm + 2.5 * vol_norm
                          + 1.0 * base_loyalty[i])

        # ---- expected revenue per strategy over the horizon ----
        eff_sens = fee_sens[i] * (1.0 - 0.9 * lock_in)  # entrenched -> inelastic
        values = {}
        for s in STRATEGIES:
            r = RATE[s]
            # retention/acceptance falls as the fee (rate) rises; damped by lock-in
            p_keep = sigmoid(2.6 - eff_sens * (r * 100.0) * 2.9)
            annual_rev = p_keep * annual_spend[i] * r
            values[s] = annual_rev * HORIZON_YEARS * RNG.lognormal(0, 0.10)
        label = max(values, key=values.get)

        # ---- observed features (noisy / partially missing proxies) ----
        dq = np.clip(ops_soph[i] + RNG.normal(0, 0.15), 0, 1)
        if RNG.random() < 0.10:
            dq = np.nan

        # Payment method and acceptance history are strong-but-imperfect
        # proxies for the latent fee sensitivity the model can't see.
        p_check = np.clip(0.10 + 0.60 * fee_sens[i] - 0.2 * base_loyalty[i], 0.03, 0.92)
        p_card = np.clip(0.45 - 0.40 * fee_sens[i], 0.02, 0.9)
        p_ach = max(0.05, 1 - p_check - p_card)
        tot = p_check + p_ach + p_card
        method = RNG.choice(["check", "ach", "card"],
                            p=[p_check / tot, p_ach / tot, p_card / tot])

        if tenure_months[i] < 6 and RNG.random() < 0.7:
            history = "never_offered"
        else:
            p_acc = sigmoid(1.6 - 5.5 * fee_sens[i] + 1.6 * lock_in)
            history = "accepted" if RNG.random() < p_acc else "rejected"

        rows.append({
            "mcc": int(mccs[i]),
            "avg_ticket": avg_ticket[i],
            "max_ticket": round(avg_ticket[i] * RNG.uniform(1.5, 8.0), 2),
            "monthly_txn_count": int(monthly_txn[i]),
            "annual_spend": annual_spend[i],
            "tenure_months": int(tenure_months[i]),
            "is_new_to_platform": bool(tenure_months[i] < 6),
            "current_payment_method": method,
            "card_acceptance_history": history,
            "past_data_quality_score": round(dq, 3) if not np.isnan(dq) else np.nan,
            "days_sales_outstanding": int(np.clip(RNG.normal(35 + 25 * fee_sens[i], 12), 5, 120)),
            "strategic_vendor": bool(lock_in > 0.8 and RNG.random() < 0.6),
            "risk_level": RNG.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
            "metadata_strategy": label,
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "vendors.csv")
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    print(df["metadata_strategy"].value_counts().to_string())
    print(f"\nMissing data_quality_score: {df['past_data_quality_score'].isna().sum()} rows")
    print(f"Whales (heavy-tail volume):  {int(whale.sum())} rows")


if __name__ == "__main__":
    main()
