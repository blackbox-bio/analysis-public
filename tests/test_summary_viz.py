import numpy as np
import pandas as pd
import pytest
from scipy.stats import f_oneway

from summary_viz import column_significance, rank_columns_by_significance


def _table(n_groups, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], 10)
    shift = np.repeat(np.arange(n_groups), 10).astype(float)
    return pd.DataFrame({
        "PV: group": groups,
        "none": rng.normal(size=len(groups)),
        "weak": rng.normal(size=len(groups)) + 0.8 * shift,
        "strong": rng.normal(size=len(groups)) + 5 * shift,
        "constant": 1.0,
        "mostly_missing": [1.0] + [np.nan] * (len(groups) - 1),
        "PV: dose": shift,  # numeric grouping-like column, still testable
    })


@pytest.mark.parametrize("n_groups", [2, 3])
def test_rank_orders_by_effect_size(n_groups):
    df = _table(n_groups)
    ranked = rank_columns_by_significance(df, "PV: group")

    # same columns, nothing dropped
    assert sorted(ranked) == sorted(df.select_dtypes(include="number").columns)
    # the old code compared each group with itself, so every p was ~1 and nothing moved
    assert ranked.index("strong") < ranked.index("weak") < ranked.index("none")
    # untestable columns last, in their original order
    assert ranked[-2:] == ["constant", "mostly_missing"]


def test_p_values_are_anova_with_bonferroni():
    df = _table(3)
    sig = column_significance(df, "PV: group")

    tested = [c for c, (_, p) in sig.items() if not np.isnan(p)]
    expected = f_oneway(*[g["strong"] for _, g in df.groupby("PV: group")])[1]
    p_adjusted, p = sig["strong"]
    assert p == pytest.approx(expected)
    assert p_adjusted == pytest.approx(min(1.0, expected * len(tested)))
    assert all(np.isnan(sig[c][0]) for c in ["constant", "mostly_missing"])


def test_group_variable_itself_is_not_tested():
    df = _table(2)
    sig = column_significance(df, "PV: dose")
    assert np.isnan(sig["PV: dose"][1])
    assert rank_columns_by_significance(df, "PV: dose")[-1] == "PV: dose"


def test_single_group_ranks_nothing():
    df = _table(1)
    ranked = rank_columns_by_significance(df, "PV: group")
    # no comparison possible: original order
    assert ranked == list(df.select_dtypes(include="number").columns)
