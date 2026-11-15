"""
Main CLI Entry Point for the 1090 MHz ADS-B Physical-Layer RF Fingerprinting Pipeline (`main.py`).
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adsb_rf.pipeline import run_online_pipeline
from adsb_rf.synthesizer import synthesize_portfolio_iq_stream

__all__ = ["run_online_pipeline", "synthesize_portfolio_iq_stream"]


if __name__ == "__main__":
    npy_file = ROOT_DIR / "receivedSignals.npy"
    setx_file = ROOT_DIR / "vsa_settings.setx"

    run_online_pipeline(
        measurement_path=npy_file if npy_file.exists() else None,
        vsa_setx_path=setx_file if setx_file.exists() else None,
        output_dir=ROOT_DIR,
    )
