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


def test_plot_worker_scaling_returns_figure(tmp_path):
    from benchmarks.visualize import plot_worker_scaling

    src = tmp_path / "worker_experiments_20260101_000000.json"
    src.write_text(json.dumps({
        "sync_api_baseline": [
            {"concurrency": 1,  "rps": 12.0, "metrics": {"avg_latency": 80.0}, "successful": 50, "failed": 0},
        ],
        "async_queue_scaling": {
            "1_workers": [
                {"concurrency": 1,  "rps":  9.0, "metrics": {"avg_latency": 110.0}, "successful": 50, "failed": 0},
                {"concurrency": 10, "rps": 80.0, "metrics": {"avg_latency": 120.0}, "successful": 50, "failed": 0},
                {"concurrency": 50, "rps": 180.0, "metrics": {"avg_latency": 250.0}, "successful": 250, "failed": 0},
            ],
            "4_workers": [
                {"concurrency": 1,  "rps":  9.0, "metrics": {"avg_latency": 110.0}, "successful": 50, "failed": 0},
                {"concurrency": 10, "rps":120.0, "metrics": {"avg_latency": 80.0},  "successful": 50, "failed": 0},
                {"concurrency": 50, "rps":210.0, "metrics": {"avg_latency": 220.0}, "successful": 250, "failed": 0},
            ],
        },
    }))

    fig = plot_worker_scaling(src)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    line_labels = {line.get_label() for line in ax.get_lines()}
    #expect one line per concurrency level + sync baseline reference
    assert any("workers" in lbl.lower() or "1" in lbl for lbl in line_labels)


def test_plot_protocol_comparison_returns_figure(tmp_path):
    from benchmarks.visualize import plot_protocol_comparison

    src = tmp_path / "grpc_experiments_20260101_000000.json"
    src.write_text(json.dumps({
        "payload_sizes": {"grpc_protobuf_bytes": 1416, "rest_multipart_bytes": 1594, "rest_overhead_pct": 12.57},
        "grpc_results": [
            {"protocol": "grpc", "concurrency": 1,  "rps":  9.1, "avg_latency_ms": 109.0, "p95_latency_ms": 110.0, "p99_latency_ms": 119.0},
            {"protocol": "grpc", "concurrency": 10, "rps": 80.0, "avg_latency_ms": 120.0, "p95_latency_ms": 140.0, "p99_latency_ms": 150.0},
        ],
        "rest_async_results": [
            {"protocol": "rest", "concurrency": 1,  "rps":  9.0, "avg_latency_ms": 110.0, "p95_latency_ms": 112.0, "p99_latency_ms": 138.0},
            {"protocol": "rest", "concurrency": 10, "rps": 78.0, "avg_latency_ms": 128.0, "p95_latency_ms": 145.0, "p99_latency_ms": 160.0},
        ],
    }))

    fig = plot_protocol_comparison(src)

    assert isinstance(fig, Figure)
    #expect 2 subplots: rps and latency
    assert len(fig.axes) == 2


def _write_device_fixture(path):
    path.write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00",
        "platform": "darwin-arm64",
        "devices_available": ["cpu", "mps"],
        "results": {
            "cpu": {
                "torchscript": [
                    {"batch_size": 1,  "total_latency_ms": 13.0, "latency_per_image_ms": 13.0, "throughput_img_per_sec": 76.0},
                    {"batch_size": 8,  "total_latency_ms": 40.0, "latency_per_image_ms": 5.0,  "throughput_img_per_sec": 200.0},
                ],
                "pytorch": [
                    {"batch_size": 1,  "total_latency_ms": 18.0, "latency_per_image_ms": 18.0, "throughput_img_per_sec": 55.0},
                    {"batch_size": 8,  "total_latency_ms": 56.0, "latency_per_image_ms": 7.0,  "throughput_img_per_sec": 142.0},
                ],
                "onnx": [
                    {"batch_size": 1,  "total_latency_ms": 22.0, "latency_per_image_ms": 22.0, "throughput_img_per_sec": 45.0},
                ],
            },
            "mps": {
                "torchscript": [
                    {"batch_size": 1,  "total_latency_ms":  6.6, "latency_per_image_ms":  6.6, "throughput_img_per_sec": 151.0},
                    {"batch_size": 8,  "total_latency_ms": 12.0, "latency_per_image_ms": 1.5,  "throughput_img_per_sec": 666.0},
                ],
                "pytorch": [
                    {"batch_size": 1,  "total_latency_ms":  8.0, "latency_per_image_ms":  8.0, "throughput_img_per_sec": 125.0},
                    {"batch_size": 8,  "total_latency_ms": 14.0, "latency_per_image_ms": 1.75, "throughput_img_per_sec": 571.0},
                ],
            },
        },
    }))


def test_plot_device_latency_returns_figure(tmp_path):
    from benchmarks.visualize import plot_device_latency

    src = tmp_path / "device_benchmark_20260101_000000.json"
    _write_device_fixture(src)

    fig = plot_device_latency(src)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    line_labels = {line.get_label() for line in ax.get_lines()}
    #4 lines: CPU x {TorchScript, PyTorch} + MPS x {TorchScript, PyTorch}
    assert any("CPU" in lbl and "TorchScript" in lbl for lbl in line_labels)
    assert any("MPS" in lbl and "TorchScript" in lbl for lbl in line_labels)
    assert not any("ONNX" in lbl for lbl in line_labels)


def test_plot_device_throughput_returns_figure(tmp_path):
    from benchmarks.visualize import plot_device_throughput

    src = tmp_path / "device_benchmark_20260101_000000.json"
    _write_device_fixture(src)

    fig = plot_device_throughput(src)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    mps_ts = next(line for line in ax.get_lines() if "MPS" in line.get_label() and "TorchScript" in line.get_label())
    cpu_ts = next(line for line in ax.get_lines() if "CPU" in line.get_label() and "TorchScript" in line.get_label())
    #MPS should outperform CPU on TorchScript at batch_size=8
    assert mps_ts.get_ydata()[-1] > cpu_ts.get_ydata()[-1]


def test_generate_all_writes_seven_pngs(tmp_path):
    from benchmarks.visualize import generate_all

    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    results_dir.mkdir()

    (results_dir / "latency_benchmark_20260101_000000.json").write_text(json.dumps({
        "models": {
            "torchscript": {"latency": {"mean_ms": 12.3, "std_ms": 1.1}},
            "pytorch":     {"latency": {"mean_ms": 18.7, "std_ms": 1.5}},
            "onnx":        {"latency": {"mean_ms": 22.4, "std_ms": 2.0}},
        },
    }))
    _write_batch_fixture(results_dir / "batch_experiments_20260101_000000.json")
    (results_dir / "worker_experiments_20260101_000000.json").write_text(json.dumps({
        "sync_api_baseline": [{"concurrency": 1, "rps": 12.0, "metrics": {"avg_latency": 80.0}, "successful": 50, "failed": 0}],
        "async_queue_scaling": {
            "1_workers": [{"concurrency": 1, "rps": 9.0, "metrics": {"avg_latency": 110.0}, "successful": 50, "failed": 0}],
        },
    }))
    (results_dir / "grpc_experiments_20260101_000000.json").write_text(json.dumps({
        "payload_sizes": {"grpc_protobuf_bytes": 1416, "rest_multipart_bytes": 1594, "rest_overhead_pct": 12.57},
        "grpc_results": [{"protocol": "grpc", "concurrency": 1, "rps": 9.1, "avg_latency_ms": 109.0, "p95_latency_ms": 110.0, "p99_latency_ms": 119.0}],
        "rest_async_results": [{"protocol": "rest", "concurrency": 1, "rps": 9.0, "avg_latency_ms": 110.0, "p95_latency_ms": 112.0, "p99_latency_ms": 138.0}],
    }))
    _write_device_fixture(results_dir / "device_benchmark_20260101_000000.json")

    written = generate_all(results_dir=results_dir, plots_dir=plots_dir)

    assert len(written) == 7
    for path in written:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
