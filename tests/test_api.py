import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.path.join(ROOT, "api.py")

# what `capabilities` must not load: Palmreader runs it at every startup
HEAVY = ["numpy", "pandas", "h5py", "scipy", "matplotlib", "seaborn", "cv2", "statsmodels", "torch", "deeplabcut"]


def _api(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, API, *args], cwd=cwd, capture_output=True, text=True, timeout=60
    )


def test_capabilities_prints_one_json_object():
    # as Palmreader calls it (python/mod.rs `spawn_api`)
    out = _api("--v2", "--function", "capabilities", "--api-version", "1", "--args", "{}")

    assert out.returncode == 0, out.stderr
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "cluster_heatmap": {"svg": True, "stats_table": True},
        "bar_plots": {"svg": True, "stats_table": True, "sort_by_significance": True},
    }


def test_capabilities_without_v2():
    out = _api("--function", "capabilities", "--args", "{}")
    assert out.returncode == 0, out.stderr
    assert "bar_plots" in json.loads(out.stdout)


def test_capabilities_imports_nothing_heavy():
    script = (
        "import runpy, sys, json\n"
        f"sys.argv = [{API!r}, '--v2', '--function', 'capabilities', '--args', '{{}}']\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        f"runpy.run_path({API!r}, run_name='__main__')\n"
        f"print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)

    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.splitlines()[-1]) == []


def test_unknown_function_exits_2():
    # what Palmreader sees from a package without a function: argparse's usage error
    out = _api("--v2", "--function", "no_such_function", "--args", "{}")
    assert out.returncode == 2
    assert "invalid ApiFunction value" in out.stderr


def _events(stdout):
    events = []
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if isinstance(message, dict) and message.get("tag") == "event":
            events.append(message)
    return events


def _summary(path):
    pytest.importorskip("pandas")
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    groups = np.repeat(["control", "treated"], 6)
    df = pd.DataFrame(
        {
            "PV: group": groups,
            "a": rng.normal(size=12) + np.repeat([0, 4], 6),
            "b": rng.normal(size=12),
            "c": rng.normal(size=12) + np.repeat([0, 1], 6),
        },
        index=[f"M{i}" for i in range(12)],
    )
    df.to_csv(path)
    return df


@pytest.mark.parametrize(
    "function,option",
    [("cluster_heatmap", {"groupingMode": "group"}), ("bar_plots", {"sortBySignificance": True})],
)
@pytest.mark.parametrize("ext", ["png", "svg"])
def test_graph_with_stats_path(tmp_path, function, option, ext):
    _summary(tmp_path / "summary.csv")
    args = {
        "summaryPath": str(tmp_path / "summary.csv"),
        "enabledRows": [True] * 12,
        "vars": ["a", "b", "c"],
        "hue": "PV: group",
        "destPath": str(tmp_path / f"graph.{ext}"),
        "statsPath": str(tmp_path / "stats.csv"),
        **option,
    }

    out = _api("--v2", "--function", function, "--api-version", "1", "--args", json.dumps(args))

    assert out.returncode == 0, out.stderr
    assert [e for e in _events(out.stdout) if e["level"] == "error"] == []
    head = (tmp_path / f"graph.{ext}").read_bytes()[:512]
    if ext == "svg":
        assert b"<svg" in head
    else:
        assert head.startswith(b"\x89PNG")

    import pandas as pd

    stats = pd.read_csv(tmp_path / "stats.csv", keep_default_na=False)
    assert sorted(stats["feature"]) == ["a", "b", "c"]
    assert set(stats["groups"]) == {"control (n=6); treated (n=6)"}


def test_graph_without_stats_path_writes_none(tmp_path):
    _summary(tmp_path / "summary.csv")
    args = {
        "summaryPath": str(tmp_path / "summary.csv"),
        "enabledRows": [True] * 12,
        "vars": ["a", "b"],
        "hue": "PV: group",
        "sortBySignificance": False,
        "destPath": str(tmp_path / "graph.png"),
    }

    out = _api("--v2", "--function", "bar_plots", "--args", json.dumps(args))

    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["graph.png", "summary.csv"]
