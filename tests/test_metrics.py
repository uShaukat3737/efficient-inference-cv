import pytest
import numpy as np
from src.serving.metrics_collector import MetricsCollector


def test_record_and_get_stats_empty():
    #empty collector should return count: 0
    collector = MetricsCollector()
    stats = collector.get_stats()
    assert stats == {"count": 0}


def test_record_single_entry():
    #single recorded entry should have count == 1 and all required stat keys
    collector = MetricsCollector()
    metrics = {
        "total_latency_ms": 100.0,
        "queue_wait_ms": 20.0,
        "inference_time_ms": 50.0,
        "batch_size": 8,
        "batch_trigger": "size",
    }
    collector.record(metrics)

    stats = collector.get_stats()
    assert stats["count"] == 1
    assert "total_latency" in stats
    assert "queue_wait" in stats
    assert "inference_time" in stats
    assert "avg_batch_size" in stats
    assert stats["avg_batch_size"] == 8.0


def test_window_capped_at_max_size():
    #collector with window_size=10 should keep only 10 items
    collector = MetricsCollector(window_size=10)
    for i in range(20):
        collector.record({
            "total_latency_ms": float(i),
            "queue_wait_ms": 10.0,
            "inference_time_ms": 50.0,
            "batch_size": 5,
            "batch_trigger": "size",
        })

    stats = collector.get_stats()
    assert stats["count"] == 10
    #oldest 10 items should be dropped, so min should be 10, not 0
    assert stats["total_latency"]["min_ms"] == 10.0


def test_percentiles_correct():
    #percentiles should be computed correctly
    collector = MetricsCollector()
    values = list(range(1, 101))
    for val in values:
        collector.record({
            "total_latency_ms": float(val),
            "queue_wait_ms": 10.0,
            "inference_time_ms": 50.0,
            "batch_size": 5,
            "batch_trigger": "size",
        })

    stats = collector.get_stats()
    latency_stats = stats["total_latency"]

    assert stats["count"] == 100
    assert abs(latency_stats["avg_ms"] - 50.5) < 1.0
    assert abs(latency_stats["p50_ms"] - 50.0) < 1.0
    assert abs(latency_stats["p95_ms"] - 95.0) < 1.0
    assert abs(latency_stats["p99_ms"] - 99.0) < 1.0


def test_batch_trigger_percentage():
    #batch_trigger_size_pct should reflect proportion of "size" triggers
    collector = MetricsCollector()
    for i in range(10):
        trigger = "size" if i < 7 else "timeout"
        collector.record({
            "total_latency_ms": 100.0,
            "queue_wait_ms": 20.0,
            "inference_time_ms": 50.0,
            "batch_size": 8,
            "batch_trigger": trigger,
        })

    stats = collector.get_stats()
    #7 out of 10 are "size", so 70%
    assert abs(stats["batch_trigger_size_pct"] - 70.0) < 0.1
