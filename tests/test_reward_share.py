import numpy as np
import pandas as pd

from scoring import _apy_reward_share


def _pool(**overrides) -> pd.Series:
    row = {"apy": 10.0, "apyReward": 2.0}
    row.update(overrides)
    return pd.Series(row)


def test_share_is_reward_divided_by_total():
    assert _apy_reward_share(_pool(apy=10.0, apyReward=8.0)) == 0.8


def test_no_reward_field_is_zero_share():
    # DeFiLlamaはapyRewardがNoneのことがある(報酬トークンによる上乗せなし)。
    assert _apy_reward_share(_pool(apy=10.0, apyReward=None)) == 0.0


def test_nan_reward_is_zero_share():
    assert _apy_reward_share(_pool(apy=10.0, apyReward=np.nan)) == 0.0


def test_zero_or_missing_apy_is_nan_not_zero():
    # APY自体が無い/ゼロだと「報酬依存度ゼロ」ではなく「判定不能」であるべき。
    assert np.isnan(_apy_reward_share(_pool(apy=0.0, apyReward=0.0)))
    assert np.isnan(_apy_reward_share(_pool(apy=np.nan, apyReward=2.0)))
