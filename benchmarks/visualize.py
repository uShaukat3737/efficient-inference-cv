import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from benchmarks.plot_utils import DEFAULT_PLOTS_DIR, latest_result

FORMAT_LABELS = {"torchscript": "TorchScript", "pytorch": "PyTorch", "onnx": "ONNX"}
FORMAT_COLORS = {"torchscript": "#1f77b4", "pytorch": "#ff7f0e", "onnx": "#2ca02c"}
DEVICE_COLORS = {"cpu": "#1f77b4", "mps": "#d62728"}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_latency_formats(src: Path) -> Figure:
    data = _load(src)
    models = data["models"]
    formats = ["torchscript", "pytorch", "onnx"]
    means = [models[f]["latency"]["mean_ms"] for f in formats]
    stds = [models[f]["latency"]["std_ms"] for f in formats]
    labels = [FORMAT_LABELS[f] for f in formats]
    colors = [FORMAT_COLORS[f] for f in formats]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean inference latency (ms)")
    ax.set_title("Single-image inference latency by model format")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_batch_latency(src: Path) -> Figure:
    data = _load(src)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fmt in ["torchscript", "pytorch", "onnx"]:
        if fmt not in data:
            continue
        rows = data[fmt]
        xs = [r["batch_size"] for r in rows]
        ys = [r["latency_per_image_ms"] for r in rows]
        ax.plot(xs, ys, marker="o", color=FORMAT_COLORS[fmt], label=FORMAT_LABELS[fmt])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency per image (ms)")
    ax.set_title("Per-image latency vs batch size")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


DEVICE_LABELS = {"cpu": "CPU", "mps": "MPS"}
DEVICE_LINESTYLES = {"cpu": "--", "mps": "-"}


def _device_lines(ax, data, y_field):
    #ONNX is excluded: ONNX Runtime has no MPS backend; comparing apples-to-apples requires same-format pairs
    for device in ["cpu", "mps"]:
        if device not in data["results"]:
            continue
        for fmt in ["torchscript", "pytorch"]:
            rows = data["results"][device].get(fmt)
            if not rows:
                continue
            xs = [r["batch_size"] for r in rows]
            ys = [r[y_field] for r in rows]
            ax.plot(xs, ys, marker="o",
                    color=FORMAT_COLORS[fmt],
                    linestyle=DEVICE_LINESTYLES[device],
                    label=f"{DEVICE_LABELS[device]} - {FORMAT_LABELS[fmt]}")


def plot_device_latency(src: Path) -> Figure:
    data = _load(src)
    fig, ax = plt.subplots(figsize=(8, 5))
    _device_lines(ax, data, "latency_per_image_ms")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency per image (ms)")
    ax.set_title("CPU vs MPS: per-image latency by batch size")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_device_throughput(src: Path) -> Figure:
    data = _load(src)
    fig, ax = plt.subplots(figsize=(8, 5))
    _device_lines(ax, data, "throughput_img_per_sec")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (images / sec)")
    ax.set_title("CPU vs MPS: throughput by batch size")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_protocol_comparison(src: Path) -> Figure:
    data = _load(src)
    grpc = sorted(data["grpc_results"], key=lambda r: r["concurrency"])
    rest = sorted(data["rest_async_results"], key=lambda r: r["concurrency"])

    fig, (ax_rps, ax_lat) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_rps.plot([r["concurrency"] for r in grpc], [r["rps"] for r in grpc],
                marker="o", color="#1f77b4", label="gRPC")
    ax_rps.plot([r["concurrency"] for r in rest], [r["rps"] for r in rest],
                marker="s", color="#ff7f0e", label="REST")
    ax_rps.set_xlabel("Concurrency")
    ax_rps.set_ylabel("Throughput (requests / sec)")
    ax_rps.set_title("Throughput: gRPC vs REST")
    ax_rps.grid(True, linestyle="--", alpha=0.4)
    ax_rps.legend()

    ax_lat.plot([r["concurrency"] for r in grpc], [r["p95_latency_ms"] for r in grpc],
                marker="o", color="#1f77b4", label="gRPC p95")
    ax_lat.plot([r["concurrency"] for r in rest], [r["p95_latency_ms"] for r in rest],
                marker="s", color="#ff7f0e", label="REST p95")
    ax_lat.set_xlabel("Concurrency")
    ax_lat.set_ylabel("p95 latency (ms)")
    ax_lat.set_title("p95 latency: gRPC vs REST")
    ax_lat.grid(True, linestyle="--", alpha=0.4)
    ax_lat.legend()

    fig.tight_layout()
    return fig


def plot_worker_scaling(src: Path) -> Figure:
    data = _load(src)
    async_scaling = data["async_queue_scaling"]

    worker_counts = sorted(int(k.split("_")[0]) for k in async_scaling.keys())
    concurrency_levels = sorted({row["concurrency"] for k in async_scaling for row in async_scaling[k]})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for conc in concurrency_levels:
        ys = []
        for w in worker_counts:
            rows = async_scaling[f"{w}_workers"]
            match = next((r for r in rows if r["concurrency"] == conc), None)
            ys.append(match["rps"] if match else None)
        ax.plot(worker_counts, ys, marker="o", label=f"concurrency={conc}")

    sync_baseline = data.get("sync_api_baseline") or []
    if sync_baseline:
        baseline_rps = sync_baseline[0]["rps"]
        ax.axhline(baseline_rps, color="gray", linestyle="--", linewidth=1,
                   label=f"sync baseline (rps={baseline_rps:.1f})")

    ax.set_xlabel("Worker count")
    ax.set_ylabel("Throughput (requests / sec)")
    ax.set_title("Async queue throughput vs worker count")
    ax.set_xticks(worker_counts)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_batch_throughput(src: Path) -> Figure:
    data = _load(src)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fmt in ["torchscript", "pytorch", "onnx"]:
        if fmt not in data:
            continue
        rows = data[fmt]
        xs = [r["batch_size"] for r in rows]
        ys = [r["batch_size"] / r["total_latency_ms"] * 1000.0 for r in rows]
        ax.plot(xs, ys, marker="o", color=FORMAT_COLORS[fmt], label=FORMAT_LABELS[fmt])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (images / sec)")
    ax.set_title("Throughput vs batch size")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig
