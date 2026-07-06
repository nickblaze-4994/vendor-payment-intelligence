"""Generate synthetic vendor data for the VIRTUAL-CARD ADOPTION model (Concept B).

Goal: among vendors currently paid by ACH or check, identify who would accept a
virtual card if enrolled -- so we can target enterprise ACH/check vendors with
>$100k annual spend for conversion.

Realism: segment (enterprise / mid_market / smb) drives both spend scale and the
current-rail mix, matching estimated US B2B market ranges (see features.py).

Label discipline (no circularity): each vendor has a *latent*
card_acceptance_capability and fee_sensitivity the model never sees. We simulate
the outcome of an enrollment offer (accept ~ Bernoulli(p_accept)) and use that as
the label. Observable features (segment, MCC, spend, current rail, whether the
supplier takes cards on other channels, DSO, tenure) are strong-but-imperfect
proxies, so classes overlap and there is real Bayes error.
"""

import os

import numpy as np
import pandas as pd

from features import (
    MCC_CARD_FRIENDLINESS,
    MCC_TICKET,
    SEGMENT_PREVALENCE,
    SEGMENT_SPEND_SCALE,
    sample_rail,
    sigmoid,
)

RNG = np.random.default_rng(7)
N = 25000

SEG_ENROLL_BONUS = {"enterprise": 0.4, "mid_market": 0.15, "smb": -0.2}
SEG_MATURE_RATE = {"enterprise": 0.30, "mid_market": 0.10, "smb": 0.0}


def main() -> None:
    segments = RNG.choice(
        list(SEGMENT_PREVALENCE),
        size=N,
        p=list(SEGMENT_PREVALENCE.values()),
    )
    mccs = RNG.choice(list(MCC_TICKET), size=N)

    # ---- latent traits (never exported) ----
    fee_sens = RNG.beta(2.5, 2.5, size=N)
    ops_soph = RNG.beta(3.0, 2.0, size=N)

    rows = []
    for i in range(N):
        seg = segments[i]
        mcc = int(mccs[i])
        is_mature = seg == "enterprise" and RNG.random() < SEG_MATURE_RATE[seg]

        # ---- activity, scaled by segment ----
        mu, sig = MCC_TICKET[mcc]
        avg_ticket = round(float(RNG.lognormal(mu, sig)), 2)
        monthly_txn = int(max(1, RNG.gamma(2.0, 6.0) + 1))
        scale = RNG.uniform(*SEGMENT_SPEND_SCALE[seg])
        annual_spend = round(avg_ticket * monthly_txn * 12 * scale, 2)

        tenure_months = int(RNG.integers(0, 180))

        # ---- latent supplier card-acceptance capability ----
        seg_bonus = {"enterprise": 0.5, "mid_market": 0.35, "smb": 0.2}[seg]
        capability = float(np.clip(
            RNG.normal(0.5 * MCC_CARD_FRIENDLINESS[mcc] + 0.5 * seg_bonus, 0.15),
            0, 1,
        ))

        rail = sample_rail(seg, is_mature, RNG)

        # supplier's card-acceptance cost burden (the 1.5-3.5% objection)
        supplier_fee_pct = RNG.uniform(1.5, 3.5) / 100.0
        burden = fee_sens[i] * (supplier_fee_pct * 100.0) / 3.5

        # ---- adoption label: would they accept a VC offer? ----
        is_check = rail == "check"
        logit = (
            -0.6
            + 3.3 * capability
            + SEG_ENROLL_BONUS[seg]
            + 0.4 * is_mature
            - 2.0 * burden
            - 0.45 * is_check
        )
        p_accept = sigmoid(logit)
        vc_accept = int(RNG.random() < p_accept)

        # ---- observable proxies ----
        # does the supplier take cards on some other channel? strong (noisy) proxy
        accepts_card_other_channel = int(RNG.random() < np.clip(0.10 + 0.82 * capability, 0, 1))
        dq = np.clip(ops_soph[i] + RNG.normal(0, 0.15), 0, 1)
        if RNG.random() < 0.10:
            dq = np.nan

        rows.append({
            "vendor_id": f"V{i:06d}",
            "segment": seg,
            "is_mature_ap_program": bool(is_mature),
            "mcc": mcc,
            "avg_ticket": avg_ticket,
            "max_ticket": round(avg_ticket * RNG.uniform(1.5, 8.0), 2),
            "monthly_txn_count": monthly_txn,
            "annual_spend": annual_spend,
            "tenure_months": tenure_months,
            "is_new_to_platform": bool(tenure_months < 6),
            "current_rail": rail,
            "accepts_card_other_channel": accepts_card_other_channel,
            "days_sales_outstanding": int(np.clip(RNG.normal(35 + 25 * fee_sens[i], 12), 5, 120)),
            "past_data_quality_score": round(dq, 3) if not np.isnan(dq) else np.nan,
            "risk_level": RNG.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
            # label is only meaningful for the targetable (ACH/check) population;
            # kept for all rows so downstream code can filter explicitly.
            "vc_accept": vc_accept,
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "adoption.csv")
    df.to_csv(out, index=False)

    targetable = df["current_rail"].isin(["ach", "check"])
    print(f"Wrote {len(df)} rows to {out}")
    print("\nCurrent-rail mix (whole universe):")
    print((df["current_rail"].value_counts(normalize=True).round(3)).to_string())
    print("\nRail mix by segment:")
    print(pd.crosstab(df["segment"], df["current_rail"], normalize="index").round(3).to_string())
    print(f"\nTargetable (ACH/check) vendors: {targetable.sum()}")
    print(f"  Accept rate among targetable: {df.loc[targetable, 'vc_accept'].mean():.3f}")
    ent_targets = targetable & (df["segment"] == "enterprise") & (df["annual_spend"] > 100_000)
    print(f"  Enterprise ACH/check >$100k:  {ent_targets.sum()} "
          f"(accept rate {df.loc[ent_targets, 'vc_accept'].mean():.3f})")


if __name__ == "__main__":
    main()
