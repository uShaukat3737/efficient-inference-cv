import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import pytest


def test_latest_result_returns_newest_by_filename(tmp_path):
    #failing test for plot_utils.latest_result helper
    from benchmarks.plot_utils import latest_result

    (tmp_path / "batch_experiments_20260101_120000.json").write_text("{}")
    (tmp_path / "batch_experiments_20260425_221239.json").write_text("{}")
    (tmp_path / "batch_experiments_20260423_135542.json").write_text("{}")
    (tmp_path / "unrelated_file.json").write_text("{}")

    result = latest_result("batch_experiments", results_dir=tmp_path)

    assert result.name == "batch_experiments_20260425_221239.json"


def test_latest_result_raises_when_no_match(tmp_path):
    from benchmarks.plot_utils import latest_result

    with pytest.raises(FileNotFoundError):
        latest_result("nonexistent_prefix", results_dir=tmp_path)


def test_plot_latency_formats_returns_figure(tmp_path):
    from benchmarks.visualize import plot_latency_formats

    src = tmp_path / "latency_benchmark_20260101_000000.json"
    src.write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00",
        "models": {
            "torchscript": {"latency": {"mean_ms": 12.3, "min_ms": 11.0, "max_ms": 15.0, "std_ms": 1.1}},
            "pytorch":     {"latency": {"mean_ms": 18.7, "min_ms": 17.0, "max_ms": 22.0, "std_ms": 1.5}},
            "onnx":        {"latency": {"mean_ms": 22.4, "min_ms": 21.0, "max_ms": 28.0, "std_ms": 2.0}},
        },
    }))

    fig = plot_latency_formats(src)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    bar_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert set(bar_labels) == {"TorchScript", "PyTorch", "ONNX"}
