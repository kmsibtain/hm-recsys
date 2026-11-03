"""Entry point for Exploratory Data Analysis.

Can be run via:
    python data/eda.py
    python data/eda_analysis.py
    python -m data.eda
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.eda_analysis import run_eda

if __name__ == "__main__":
    run_eda()

