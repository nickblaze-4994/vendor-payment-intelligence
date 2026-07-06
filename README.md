# Vendor Payment Intelligence

Two XGBoost models covering the virtual-card vendor lifecycle:

```
   ACH / check vendor                 vendor already on virtual card
 ┌────────────────────┐  convert    ┌──────────────────────────┐
 │ CONCEPT B           │ ─────────▶ │ CONCEPT A                 │
 │ Adoption targeting  │            │ Metadata dynamic pricing  │
 │ "who to move to VC" │            │ "how much metadata now"   │
 └────────────────────┘            └──────────────────────────┘
```

Both share one feature layer (`features.py`) and the same label discipline:
labels come from *simulated outcomes* driven by latent traits the model never
sees, so classes overlap and there is genuine Bayes error (not a rule the model
can reverse-engineer).

---

## Concept B — Virtual-card adoption targeting

**Goal:** among vendors currently paid by **ACH or check**, find enterprise
vendors with **>$100k annual spend** who would accept a virtual card — the
conversion hit-list.

**Market realism.** `features.py` encodes segment-conditioned payment-rail mixes
matching estimated US outbound B2B ranges (these are estimated market ranges, not
a Corpay disclosure):

| Segment | Virtual card | ACH | Wire | Check | Other |
|---|---|---|---|---|---|
| Enterprise (avg) | 15% | 50% | 12% | 18% | 5% |
| Mature Corpay enterprise | 27% | 45% | 10% | 13% | 5% |
| SMB | 5.5% | 35% | 7.5% | 40% | 12% |

**Model + business layer.** `train_adoption.py` predicts `P(accept)` for the
ACH/check population. `target_list.py` then filters to enterprise / ACH-check /
>$100k and ranks by expected value:

```
EV = P(accept) × annual_spend × CORPAY_NET_TAKE − ENROLL_COST
```

**Results** (25k synthetic vendors): evaluated as a ranking problem — ROC-AUC
~0.65, **precision@50 ~0.80 (≈1.6× lift over base)**, calibrated (Brier ~0.23).
Rail mix reproduces the table above (enterprise VC ~18%, SMB VC ~6%). Output is a
ranked CSV of conversion targets with `p_accept` and projected annual revenue.

```bash
python generate_adoption_data.py   # -> data/adoption.csv
python train_adoption.py           # -> model/xgb_adoption.json + ranking metrics
python target_list.py              # -> data/target_list.csv (ranked hit-list)
```

The **metadata knob is also an acquisition lever**: releasing full metadata lowers
the supplier's acceptance cost, which raises `P(accept)`. So a converted target
graduates into Concept A, where the rate is walked back up as lock-in grows —
*acquire cheap, extract later.*

---

## Concept A — Interchange metadata dynamic pricing

An XGBoost model that treats **virtual-card metadata release as a dynamic-pricing
knob**. Corpay (the issuer) already holds every field Visa needs; the decision is
how much of it to *release* on each authorization, per vendor.

## The financial model

Interchange revenue = `spend × rate`, and the **rate is a function of the metadata
level submitted**:

| Strategy       | Metadata released | Interchange (illustrative) | Stance    |
|----------------|-------------------|----------------------------|-----------|
| `send_minimal` | lean              | ~2.5% (highest)            | **Extract** |
| `send_partial` | medium            | ~1.9%                      | Balance   |
| `send_full`    | full L2/L3        | ~1.2% (lowest)             | **Acquire** |

The optimal knob depends on the vendor's position in the network:

- **New / fee-elastic vendors** have near-zero switching cost. Charge a high fee and
  they refuse the card — you earn nothing. Release full metadata, give the low rate,
  **win the relationship and its future volume**.
- **Entrenched whales** (high tenure, huge volume, high switching cost) are fee-
  *inelastic* — they won't churn over a few basis points. Withhold metadata, keep the
  high rate. Because revenue = volume × rate, a small rate bump on an enormous spender
  is large absolute revenue. This is the **"trillions in volume" extraction case**.

The classifier learns to pick the strategy that maximizes **expected revenue over a
3-year horizon**, net of churn risk.

## Why the labels aren't circular

`generate_data.py` does **not** label from a rule over the visible features. Each
vendor has *latent* traits (fee elasticity, loyalty, ops sophistication) the model
never sees. We Monte-Carlo simulate expected revenue per strategy — using a churn
model driven by those latent traits plus lock-in — and label the argmax, with
multiplicative noise. Observed features (payment method, acceptance history, tenure,
volume) are strong-but-imperfect *proxies*. Result: overlapping classes and genuine
Bayes error, like real transaction data.

## Results (20k synthetic vendors)

- 5-fold CV accuracy **0.786 ± 0.002** vs always-extract baseline **0.773**
- Macro-F1 **0.57** vs baseline **0.29** — nearly 2× on the balanced metric that
  rewards catching the *deviate-from-default* cases (the profitable part)
- Train/test gap ~0.12 (not overfit)
- Top features: `is_new_to_platform`, `tenure_months`, `card_acceptance_history`,
  `annual_spend` — i.e. exactly the acquire-vs-extract signals

The headline single-split accuracy sits near the majority baseline **by design**:
balanced class weights trade a little majority-class accuracy to recover the minority
`send_full` / `send_partial` opportunities, which is where the strategy earns money
over "always extract."

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py   # -> data/vendors.csv (20k simulated vendors)
python train.py           # -> model/ + metrics
python predict.py --json '{"mcc": 4214, "avg_ticket": 85000, "max_ticket": 300000,
  "monthly_txn_count": 40, "annual_spend": 40800000, "tenure_months": 96,
  "is_new_to_platform": false, "current_payment_method": "card",
  "card_acceptance_history": "accepted", "past_data_quality_score": 0.9,
  "days_sales_outstanding": 30, "strategic_vendor": true, "risk_level": "low"}'
# -> send_minimal (extract): entrenched whale, keep the high rate
```

Interchange percentages are illustrative — actual rates are set by the card networks.
