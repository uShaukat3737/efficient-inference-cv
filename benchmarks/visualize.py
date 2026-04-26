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
