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


# ---- graph bugs (A1, A2, A8, A10, A15) ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import summary_viz
from summary_viz import (
    generate_bar_plots,
    generate_cluster_heatmap,
    generate_PairGrid_plot,
    summary_viz_preprocess,
)


def _recordings(n_groups=3, per_group=6, n_features=12, seed=1):
    # features with different group patterns, so group- and individual-level clustering disagree
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    data = {"PV: group": groups}
    for f in range(n_features):
        effect = rng.normal(size=n_groups) * (f % 3)
        data[f"feature_{f}"] = rng.normal(size=len(groups)) + np.repeat(effect, per_group)
    return pd.DataFrame(data, index=[f"mouse_{i}" for i in range(len(groups))])


@pytest.fixture
def grids(monkeypatch):
    """Record every ClusterGrid the heatmap makes."""
    made = []
    real = sns.clustermap

    def recording_clustermap(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(summary_viz.sns, "clustermap", recording_clustermap)
    return made


@pytest.mark.parametrize("mode", ["group", "individual"])
def test_heatmap_row_labels_match_rows(tmp_path, grids, mode):
    generate_cluster_heatmap(_recordings(), "PV: group", str(tmp_path / "h.png"), mode)

    final = grids[-1]
    labels = [t.get_text() for t in final.ax_heatmap.get_yticklabels()]
    # data2d holds the plotted matrix in display order
    assert labels == list(final.data2d.index)
    assert (tmp_path / "h.png").exists()


def test_heatmap_individual_mode_reorders_rows(tmp_path, grids):
    # guard for the test above: the individual clustering really does reorder the features here
    generate_cluster_heatmap(_recordings(), "PV: group", str(tmp_path / "h.png"), "individual")
    group_order, individual_order = list(grids[0].data2d.index), list(grids[-1].data2d.index)
    assert group_order != individual_order


@pytest.mark.parametrize("mode", ["group", "individual"])
def test_heatmap_omits_inf_columns(tmp_path, grids, mode):
    df = _recordings()
    df.loc["mouse_0", "feature_1"] = np.inf
    generate_cluster_heatmap(df, "PV: group", str(tmp_path / "h.png"), mode)

    plotted = grids[-1].data2d
    assert "feature_1" not in plotted.index
    assert np.isfinite(plotted.to_numpy()).all()


def test_heatmap_single_group(tmp_path, grids):
    df = _recordings(n_groups=1)
    generate_cluster_heatmap(df, "PV: group", str(tmp_path / "h.png"), "group")
    assert (tmp_path / "h.png").exists()
    # legend title without the PV: prefix
    legend = grids[-1].ax_col_dendrogram.get_legend()
    assert legend.get_title().get_text() == "group"
    assert [t.get_text() for t in grids[-1].ax_col_colors.get_yticklabels()] == ["group"]


def test_heatmap_drops_time_columns(tmp_path, grids):
    df = _recordings().assign(**{"bin duration (min)": np.linspace(1, 5, 18), "total recording_time (min)": np.linspace(5, 9, 18)})
    generate_cluster_heatmap(df, "PV: group", str(tmp_path / "h.png"), "group")
    assert not {"bin duration (min)", "total recording_time (min)"} & set(grids[-1].data2d.index)


def test_heatmap_single_readout(tmp_path):
    df = _recordings(n_features=1)
    for mode in ["group", "individual"]:
        generate_cluster_heatmap(df, "PV: group", str(tmp_path / f"{mode}.png"), mode)
        assert (tmp_path / f"{mode}.png").exists()


def test_heatmap_nothing_left_to_plot(tmp_path):
    df = _recordings(n_features=2)
    df["feature_0"] = 1.0
    df["feature_1"] = np.nan
    with pytest.raises(ValueError, match="No readouts are left"):
        generate_cluster_heatmap(df, "PV: group", str(tmp_path / "h.png"), "group")


def test_heatmaps_close_their_figures(tmp_path):
    plt.close("all")
    for mode in ["group", "individual"]:
        generate_cluster_heatmap(_recordings(), "PV: group", str(tmp_path / "h.png"), mode)
    assert plt.get_fignums() == []


def test_preprocess_keeps_group_once_and_leaves_input_alone():
    df = _recordings()
    columns = ["feature_0", "PV: group", "feature_1"]
    out = summary_viz_preprocess(df, [True] * len(df), columns, "PV: group")
    assert list(out.columns) == ["feature_0", "feature_1", "PV: group"]
    assert columns == ["feature_0", "PV: group", "feature_1"]


def test_bar_plots_skip_numeric_group_and_close_figures(tmp_path, monkeypatch):
    df = _recordings()[["feature_0", "feature_1"]].assign(**{"PV: dose": [0.0, 1.0, 2.0] * 6})
    plotted = []
    real = sns.pointplot
    monkeypatch.setattr(summary_viz.sns, "pointplot", lambda **kw: plotted.append(kw["y"]) or real(**kw))
    plt.close("all")

    generate_bar_plots(df, "PV: dose", str(tmp_path / "bars.png"))

    assert plotted == ["feature_0", "feature_1"]
    assert plt.get_fignums() == []


def test_bar_plots_without_numeric_readouts(tmp_path):
    df = pd.DataFrame({"PV: group": ["a", "b"], "notes": ["x", "y"]})
    with pytest.raises(ValueError, match="nothing to plot"):
        generate_bar_plots(df, "PV: group", str(tmp_path / "bars.png"))


def test_pairgrid_uses_significance_order(tmp_path, monkeypatch):
    df = _table(3)[["PV: group", "none", "strong", "weak"]]
    seen = {}
    real = sns.PairGrid
    monkeypatch.setattr(summary_viz.sns, "PairGrid", lambda data, **kw: seen.update(kw) or real(data, **kw))
    plt.close("all")

    generate_PairGrid_plot(df, "PV: group", "hist", "kde", "reg", str(tmp_path / "pg.png"), sort_by_significance=True)

    assert seen["vars"] == ["strong", "weak", "none"]
    assert plt.get_fignums() == []
