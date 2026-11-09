# 1090 MHz ADS-B Physical-Layer RF Fingerprinting & Spoofing Detection

Modular Python 3.12 digital signal processing (DSP) and machine learning pipeline that authenticates aircraft Mode-S Extended Squitter (`DF=17`, 1090 MHz) transponders directly from their physical-layer RF waveform characteristics.

Evolved from rooftop Vector Signal Analyzer (Keysight UXA N9040B, 12.8 MS/s) measurements collected at the **Institute of Telecommunications, TU Wien (Vienna University of Technology)**.

## Key Engineering Highlights

1. **Continuous Online I/Q Stream Processing (`stream_reader.py`, `dsp.py`)**:
   - Memory-mapped (`.npy` / `.mat`) complex baseband streaming with rational polyphase resampling (`12.8 MS/s -> 10.0 MS/s`).
   - Extended Mode-S preamble (`16` preamble chips + `10` `DF=17` header chips) cross-correlation and modulo-2 `CRC-24` (`G(x) = 0xFFF409`) parity verification.
   - Two-stage **unwrap-free Carrier Frequency Offset (CFO)** estimator combining intra-pulse $0.1\,\mu\text{s}$ conjugate products ($\pm 5.0\text{ MHz}$ unambiguous range) with derotated $1.0\,\mu\text{s}$ and $1.5\,\mu\text{s}$ unit-phasor refinement.

2. **Stateful Online Mode-S & Kinematic Decoder (`decoder.py`)**:
   - Decodes Aircraft Identification (`TC 1..4`), Airborne Position (`TC 9..18`, locally and globally unambiguous Even/Odd Compact Position Reporting), and Airborne Velocity (`TC 19`).
   - Computes real-time 3D slant-range Free-Space Path Loss (FSPL), Equivalent Isotropically Radiated Power (`EIRP_dBm`), and spherical-geometry kinematic Doppler shift (`Doppler`).

3. **12-Feature Physical-Layer ML Ensemble & Deep Autoencoder (`anomaly_ml.py`)**:
   - Extracts 12 label-free physical features per $120\,\mu\text{s}$ burst: `CFO_Raw_Hz`, `Frequency_shift` (Doppler-compensated CFO), `Timing_Offset_PPM` (symbol clock crystal skew), `Sub_Sample_Lag_Ns`, `Power_dBm`, `EIRP_dBm`, `Phase_Noise_Std`, `PA_Droop_dB` (power amplifier capacitor sag), `Pulse_Crest_Factor`, `Phase_Curvature_Rad` (PA thermal frequency pulling), `SNR_dB`, and `Preamble_Ratio_dB`.
   - Hybrid 3-model soft-voting classifier (**45% RandomForest + 35% ExtraTrees + 20% Vectorized Bayesian Mahalanobis Posterior**) paired with a PyTorch `RFAnomalyAutoencoder` (`12 -> 24 -> 3 -> 24 -> 12`) and `IsolationForest`.

## Repository Structure

- `config.py` â€” Keysight 89600 VSA (`.setx`) XML parser, receiver calibration (`5.5 dBi` collinear antenna, `11.37 dB` cable/front-end loss), and rooftop anomaly thresholds.
- `stream_reader.py` â€” Memory-mapped `.npy` / `.mat` continuous online reader and polyphase resampler.
- `dsp.py` â€” Preamble matched filter, PPM bit slicer, CRC-24 validator, CFO/phase-noise estimator, and sub-sample symbol clock timing extractor.
- `decoder.py` â€” Stateful per-ICAO online tracker, CPR decoder, and Doppler/EIRP calculator.
- `database.py` â€” Embedded DuckDB (with automatic SQLite fallback) analytical SQL telemetry store and per-aircraft RF profile aggregator.
- `anomaly_ml.py` â€” 12-feature RF fingerprinting classifier, unwrap-free CFO recovery, PyTorch Autoencoder / 1D-CNN, and spoofing detector.
- `visualization.py` â€” Interactive Leaflet.js HTML trajectory map (`adsb_map.html`), 2D RF cluster scatter plot (`rf_fingerprint_clusters.png`), IQ FFT spectra, and PPM eye diagram.
- `main.py` â€” End-to-end online pipeline runner (supports full multi-GB VSA captures or built-in synthetic 12.8 MS/s multi-aircraft stream when raw captures are omitted).
- `test_adsb_pipeline.py` â€” Unit test suite covering CRC-24, CPR latitude zones, Doppler/EIRP physics, and unwrap-free CFO recovery.

## Empirical Performance on TU Wien 268M-Sample Rooftop Capture

Evaluated on a continuous $20.97\text{ s}$ capture ($268,434,198$ complex I/Q samples at $12.80\text{ MS/s}$, yielding $2,809$ CRC-valid Mode-S frames across commercial airliners over Vienna):

| Evaluation Metric | Result | Notes |
| :--- | :--- | :--- |
| **80/20 Stratified Holdout Test Accuracy** | **90.6%** | Evaluated out-of-sample across enrolled aircraft ($\ge 5$ frames) |
| **5-Fold Stratified Cross-Validation Accuracy** | **90.3%** | 5-fold out-of-fold evaluation across all eligible aircraft |
| **Top-3 Candidate Match Accuracy** | **100.0%** | True transmitting aircraft ranked within top 3 RF posterior matches |
| **All-Frame Clean Inlier Fit** | **100.0%** | Zero label leakage; integer-indexed damped sample weighting |

## Quick Start

```bash
pip install -r requirements.txt
python main.py
pytest test_adsb_pipeline.py
```

> **Note on Raw I/Q Data Files**: Multi-gigabyte raw Keysight VSA captures (`101000.mat`, `receivedSignals.npy`, ~2.1 GB each) are excluded via `.gitignore`. When run without the external `.npy` file present, `main.py` automatically synthesizes a calibrated $12.8\text{ MS/s}$ multi-aircraft Mode-S I/Q stream so the entire DSP, DuckDB, and ML pipeline can be executed and verified out of the box.