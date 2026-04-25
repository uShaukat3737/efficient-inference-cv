import json
from pathlib import Path

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
