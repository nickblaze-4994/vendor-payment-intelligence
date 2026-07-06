"""Shared domain priors for the vendor-payment models.

Segment definitions and payment-rail mixes are ESTIMATED US outbound B2B
supplier-payment market ranges, NOT a Corpay disclosure. Corpay does not
publicly publish its customers' rail mix. Values are midpoints of commonly
cited enterprise/SMB ranges and are used only to make synthetic data realistic.
"""

import numpy as np

RAILS = ["virtual_card", "ach", "wire", "check", "other"]

# current-rail mix by segment (each row sums to 1.0)
#                    VC     ACH    wire   check  other
RAIL_MIX = {
    "enterprise":        [0.15, 0.50, 0.12, 0.18, 0.05],  # average enterprise
    "enterprise_mature": [0.27, 0.45, 0.10, 0.13, 0.05],  # mature Corpay AP program
    "mid_market":        [0.09, 0.42, 0.09, 0.30, 0.10],
    "smb":               [0.055, 0.35, 0.075, 0.40, 0.12],
}

# rough count-prevalence of segments in the addressable universe
SEGMENT_PREVALENCE = {"enterprise": 0.20, "mid_market": 0.30, "smb": 0.50}

# annual-spend scale multiplier by segment (enterprise runs far more volume)
SEGMENT_SPEND_SCALE = {
    "enterprise": (5.0, 20.0),
    "mid_market": (1.5, 5.0),
    "smb": (0.3, 1.2),
}

# how readily a supplier in this MCC can/will accept a commercial card (0..1)
MCC_CARD_FRIENDLINESS = {
    5065: 0.55, 5085: 0.55, 5111: 0.80, 5122: 0.55, 5137: 0.65,
    5169: 0.35, 5172: 0.25, 5199: 0.55, 5211: 0.45, 5734: 0.85,
    7372: 0.85, 7392: 0.80, 7399: 0.75, 4214: 0.30, 8911: 0.70,
}

# MCC -> (lognormal mean, sigma) for per-transaction ticket size
MCC_TICKET = {
    5065: (7.0, 1.0), 5085: (7.4, 1.1), 5111: (5.8, 0.8), 5122: (7.8, 1.0),
    5137: (6.2, 0.9), 5169: (7.6, 1.1), 5172: (8.4, 1.0), 5199: (6.5, 1.0),
    5211: (7.9, 1.1), 5734: (7.2, 1.4), 7372: (8.1, 1.3), 7392: (8.3, 1.2),
    7399: (6.8, 1.2), 4214: (8.6, 1.0), 8911: (8.8, 1.2),
}

# net interchange share Corpay retains per $ of virtual-card spend (illustrative,
# after rebate share to the buyer and program costs)
CORPAY_NET_TAKE = 0.008
ENROLL_COST = 200.0  # cost to run a supplier-enrollment campaign for one vendor
ENTERPRISE_SPEND_GATE = 100_000  # actionable target floor

# ---------------------------------------------------------------------------
# Expanded economic profile for VALUE-BASED dynamic pricing.
# The objective is long-term contribution while keeping acceptance sustainable
# -- price on the supplier's economics and the value delivered, not on how hard
# it would be for the supplier to leave.
# ---------------------------------------------------------------------------

# explicit industry sector per MCC (richer than the raw code)
INDUSTRY_BY_MCC = {
    5065: "industrial", 5085: "industrial", 5199: "industrial",
    5111: "office_supplies", 5137: "office_supplies",
    5122: "pharma", 5169: "chemicals", 5172: "fuel_energy",
    5211: "construction", 5734: "software", 7372: "it_services",
    7392: "professional_services", 7399: "professional_services",
    4214: "freight_logistics", 8911: "engineering",
}

# supplier margin profile -> rate elasticity (thin margins resist % fees hardest)
MARGIN_PROFILES = ["thin", "moderate", "healthy"]
MARGIN_SENSITIVITY = {"thin": 1.0, "moderate": 0.55, "healthy": 0.28}

# how much a supplier values faster settlement (liquidity preference)
SPEED_PREFS = ["standard", "fast", "immediate"]
SPEED_VALUE = {"standard": 0.0, "fast": 0.5, "immediate": 1.0}

# reconciliation burden -> value of rich remittance metadata
RECON_COMPLEXITY = ["low", "medium", "high"]
RECON_VALUE = {"low": 0.0, "medium": 0.4, "high": 0.9}

GEOGRAPHIES = ["domestic", "cross_border"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "other"]
# extra cost/friction (in bps of volume) for cross-border settlement + FX
GEO_COST_BPS = {"domestic": 0.0, "cross_border": 20.0}

SERVICING_COST = ["low", "medium", "high"]          # support/servicing tier
SERVICING_COST_ANNUAL = {"low": 120.0, "medium": 480.0, "high": 1500.0}

# ---------------------------------------------------------------------------
# Offer decision space -- CORE LEVERS only (per product scope).
# Metadata is always offered at FULL quality as value (reconciliation), never
# withheld. The price lever is the effective acceptance rate; settlement speed
# and a fee cap / ACH flat-fee fallback round out the negotiable package.
# ---------------------------------------------------------------------------
RATE_BPS = [80, 150, 220, 280]           # effective acceptance cost, basis points
SETTLEMENT = ["next_day", "same_day"]     # same_day commands higher acceptable fee
FEE_CAP = ["none", "capped"]              # cap protects large-invoice suppliers
SAME_DAY_COST_BPS = 25.0                  # Corpay's float cost for same-day
RATE_FLOOR_BPS, RATE_CEIL_BPS = 50.0, 300.0
LARGE_INVOICE_THRESHOLD = 25_000          # above this an uncapped % is painful
FEE_CAP_PER_TXN = 500.0                   # $ cap when fee_cap == "capped"
ACH_FLAT_FEE = 5.0                        # per-payment ACH fallback fee


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_rail(segment: str, is_mature: bool, rng: np.random.Generator) -> str:
    """Draw a current payment rail from the segment-conditioned mix, with
    small per-vendor Dirichlet noise so the mix isn't perfectly rigid."""
    key = "enterprise_mature" if (segment == "enterprise" and is_mature) else segment
    base = np.array(RAIL_MIX[key])
    probs = rng.dirichlet(base * 60.0)  # concentrated near the base mix
    return RAILS[rng.choice(len(RAILS), p=probs)]
