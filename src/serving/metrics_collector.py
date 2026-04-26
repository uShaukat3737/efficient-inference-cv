from collections import deque
import numpy as np


class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self._records = deque(maxlen=window_size)

    def record(self, metrics: dict) -> None:
        #append a single metrics dict to the rolling window
        self._records.append(metrics)

    def get_stats(self) -> dict:
        #compute aggregate statistics over the rolling window
        if not self._records:
            return {"count": 0}

        records = list(self._records)

        def _agg(key: str):
            #aggregate helper: compute mean, percentiles, min, max for a metric
            vals = [r[key] for r in records if key in r]
            if not vals:
                return None
            return {
                "avg_ms": float(np.mean(vals)),
                "p50_ms": float(np.percentile(vals, 50)),
                "p95_ms": float(np.percentile(vals, 95)),
                "p99_ms": float(np.percentile(vals, 99)),
                "min_ms": float(np.min(vals)),
                "max_ms": float(np.max(vals)),
            }

        batch_sizes = [r["batch_size"] for r in records if "batch_size" in r]
        size_triggers = sum(1 for r in records if r.get("batch_trigger") == "size")

        return {
            "count": len(records),
            "total_latency": _agg("total_latency_ms"),
            "queue_wait": _agg("queue_wait_ms"),
            "inference_time": _agg("inference_time_ms"),
            "avg_batch_size": float(np.mean(batch_sizes)) if batch_sizes else None,
            "batch_trigger_size_pct": (size_triggers / len(records) * 100) if records else 0,
        }
