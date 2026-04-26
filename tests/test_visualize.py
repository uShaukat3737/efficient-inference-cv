import json

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


def _write_batch_fixture(path):
    path.write_text(json.dumps({
        "torchscript": [
            {"batch_size": 1,  "total_latency_ms": 12.0, "latency_per_image_ms": 12.0},
            {"batch_size": 8,  "total_latency_ms": 40.0, "latency_per_image_ms": 5.0},
            {"batch_size": 32, "total_latency_ms": 96.0, "latency_per_image_ms": 3.0},
        ],
        "pytorch": [
            {"batch_size": 1,  "total_latency_ms": 18.0, "latency_per_image_ms": 18.0},
            {"batch_size": 8,  "total_latency_ms": 56.0, "latency_per_image_ms": 7.0},
            {"batch_size": 32, "total_latency_ms": 128.0, "latency_per_image_ms": 4.0},
        ],
        "onnx": [
            {"batch_size": 1,  "total_latency_ms": 22.0, "latency_per_image_ms": 22.0},
            {"batch_size": 8,  "total_latency_ms": 72.0, "latency_per_image_ms": 9.0},
            {"batch_size": 32, "total_latency_ms": 160.0, "latency_per_image_ms": 5.0},
        ],
    }))


def test_plot_batch_latency_returns_figure(tmp_path):
    from benchmarks.visualize import plot_batch_latency

    src = tmp_path / "batch_experiments_20260101_000000.json"
    _write_batch_fixture(src)

    fig = plot_batch_latency(src)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    line_labels = {line.get_label() for line in ax.get_lines()}
    assert {"TorchScript", "PyTorch", "ONNX"}.issubset(line_labels)


def test_plot_batch_throughput_returns_figure(tmp_path):
    from benchmarks.visualize import plot_batch_throughput

    src = tmp_path / "batch_experiments_20260101_000000.json"
    _write_batch_fixture(src)

    fig = plot_batch_throughput(src)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    #throughput should be increasing with batch size for at least one format
    ts_line = next(line for line in ax.get_lines() if line.get_label() == "TorchScript")
    ydata = ts_line.get_ydata()
    assert ydata[-1] > ydata[0]
