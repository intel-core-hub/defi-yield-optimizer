import time

import pandas as pd

from risk import RECENT_HACK_WINDOW_DAYS
from scoring import apply_hack_risk

NOW = time.time()
RECENT_HACK_DATE = NOW - 30 * 86400  # 30日前
OLD_HACK_DATE = NOW - (RECENT_HACK_WINDOW_DAYS + 30) * 86400  # 窓の外


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "project": ["p1", "p1", "p2", "p2", "p3"],
        "chain": ["Ethereum", "Arbitrum", "Ethereum", "Arbitrum", "Ethereum"],
    })


def test_chain_specific_hack_only_affects_same_chain():
    hack_events = pd.DataFrame({
        "project": ["p1"],
        "chain": ["Ethereum"],
        "hack_date": [RECENT_HACK_DATE],
        "hack_name": ["p1 exploit"],
    })
    result = apply_hack_risk(_candidates(), hack_events, now=NOW)

    p1_eth = result[(result["project"] == "p1") & (result["chain"] == "Ethereum")].iloc[0]
    p1_arb = result[(result["project"] == "p1") & (result["chain"] == "Arbitrum")].iloc[0]
    assert p1_eth["hack_recent"]
    assert not p1_arb["was_hacked"]


def test_chain_agnostic_hack_applies_to_all_chains():
    hack_events = pd.DataFrame({
        "project": ["p2"],
        "chain": [None],
        "hack_date": [RECENT_HACK_DATE],
        "hack_name": ["p2 exploit, chain unknown"],
    })
    result = apply_hack_risk(_candidates(), hack_events, now=NOW)

    p2_rows = result[result["project"] == "p2"]
    assert p2_rows["hack_recent"].all()


def test_old_hack_sets_was_hacked_but_not_recent():
    hack_events = pd.DataFrame({
        "project": ["p3"],
        "chain": ["Ethereum"],
        "hack_date": [OLD_HACK_DATE],
        "hack_name": ["p3 old exploit"],
    })
    result = apply_hack_risk(_candidates(), hack_events, now=NOW)

    p3 = result[result["project"] == "p3"].iloc[0]
    assert p3["was_hacked"]
    assert not p3["hack_recent"]


def test_no_matching_hack_leaves_row_clean():
    hack_events = pd.DataFrame(columns=["project", "chain", "hack_date", "hack_name"])
    result = apply_hack_risk(_candidates(), hack_events, now=NOW)
    assert not result["was_hacked"].any()
    assert not result["hack_recent"].any()
