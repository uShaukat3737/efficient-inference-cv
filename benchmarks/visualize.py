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
