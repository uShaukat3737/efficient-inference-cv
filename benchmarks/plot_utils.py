from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_PLOTS_DIR = Path(__file__).parent / "plots"


def latest_result(prefix: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    #returns newest matching file by lexicographic filename order (ISO-style timestamps sort correctly)
    matches = sorted(Path(results_dir).glob(f"{prefix}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no result files matching '{prefix}_*.json' in {results_dir}")
    return matches[-1]
