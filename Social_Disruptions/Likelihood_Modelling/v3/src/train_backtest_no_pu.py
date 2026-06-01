# Same as train_backtest.py but PU learning disabled — all zeros treated as genuine negatives.

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import train_backtest as tb

# Disable PU learning — treat all zeros as genuine negatives
tb.USE_PU_LEARNING = False
tb.PROC_DIR = tb._V2 / "data" / "processed_no_pu"

if __name__ == "__main__":
    tb._setup_logging()
    panel = tb.load_panel()
    tb.PROC_DIR.mkdir(parents=True, exist_ok=True)
    tb.run_backtest(panel)
