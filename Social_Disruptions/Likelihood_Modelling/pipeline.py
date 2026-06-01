import sys
import importlib.util
from pathlib import Path

_SRC      = Path(__file__).resolve().parent / "src"
_ANALYSIS = Path(__file__).resolve().parent / "analysis"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_acled_country_day = _load("build_acled_country_day", _SRC / "build_acled_country_day.py")
build_panel_country_day = _load("build_panel_country_day", _SRC / "build_panel_country_day.py")
train_backtest          = _load("train_backtest",          _SRC / "train_backtest.py")
run_analysis            = _load("run_analysis",            _ANALYSIS / "run_analysis.py")


def run():
    print("Step 1: Building ACLED country-day panel...")
    build_acled_country_day.main()

    print("\nStep 2: Building modelling panel...")
    build_panel_country_day.main()

    print("\nStep 3: Training and backtesting models...")
    train_backtest.main()

    print("\nStep 4: Generating figures and tables...")
    run_analysis.main()

    print("\nDone.")


if __name__ == "__main__":
    run()
