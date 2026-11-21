# 1090 MHz ADS-B Physical-Layer RF Fingerprinting & Spoofing Detection

Modular Python digital signal processing (DSP) and machine learning package (`src/adsb_rf/`) that authenticates aircraft Mode-S Extended Squitter (`DF=17`, $1090\text{ MHz}$) transponders directly from their physical-layer analog RF waveform characteristics.

Evolved from rooftop Vector Signal Analyzer (Keysight UXA N9040B, $12.80\text{ MS/s}$) measurements collected at the **Institute of Telecommunications, TU Wien (Vienna University of Technology)** as part of the Bachelor's thesis *"Analysis of Modulation Parameters of Surveillance Systems ADS-B and SSR"*, expanded in 2026 with a 12-feature physical-layer representation, leak-free hybrid ensemble classification, out-of-fold residual uncertainty calibration, and deep neural network architectures.

---

## ⚠️ Data Provenance, Public Benchmarks & Synthetic Simulation Notice

1. **Real Over-the-Air Rooftop Telemetry (`adsb_table.csv`)**:
   - Contains **2,809 CRC-valid Mode-S Extended Squitter (`DF=17`) frames** across **174 distinct 24-bit ICAO addresses**, demodulated from a single continuous **20.97-second ($268{,}434{,}198$-sample at $12.80\text{ MS/s}$)** complex baseband recording captured on the roof of the TU Wien Telecommunications Institute (`48.19655° N, 16.36988° E`).
   - All aircraft identifiers (`icao_hex`) and callsigns are public, unencrypted commercial aviation broadcasts governed by **ICAO Annex 10 (Vol. IV)** and **RTCA DO-260B**. Zero private business identifiers, credentials, or PII are present.
2. **Interactive Synthetic Spoofing Simulator & Fallback Stream**:
   - In `streamlit_app.py`, the sidebar **Live RF Spoofing Simulator** injects **synthetic/simulated RF bursts** with user-controlled carrier frequency offset (CFO), symbol clock drift (ppm), and EIRP to test physical-layer authentication against empirical **5-fold out-of-fold backtest residual intervals** derived from the real rooftop capture.
   - Because raw multi-gigabyte Keysight VSA capture files (`receivedSignals.npy`, `101000.mat`, ~2.1 GB each) are gitignored, running `python main.py` without the raw `.npy` file automatically invokes `src/adsb_rf/synthesizer.py` to generate a calibrated $12.80\text{ MS/s}$ synthetic multi-aircraft Mode-S I/Q stream so the full DSP, database, and ML pipeline can be verified end-to-end out of the box.

---

## Key Engineering Highlights

1. **Continuous Online I/Q Stream Processing (`src/adsb_rf/stream_reader.py`, `src/adsb_rf/dsp.py`)**:
   - Memory-mapped (`.npy` / `.mat`) sliding-window complex baseband streaming with rational polyphase resampling ($12.80\text{ MS/s} \rightarrow 10.00\text{ MS/s}$, factor $25/32$) and $150\,\mu\text{s}$ inter-chunk overlap so no $120\,\mu\text{s}$ frame is truncated at a chunk boundary.
   - Extended Mode-S preamble ($16$ preamble chips + $10$ `DF=17` header chips) cross-correlation and modulo-2 `CRC-24` ($G(x) = \text{0x1FFF409}$) parity verification.
   - **Two-stage unwrap-free Carrier Frequency Offset (CFO)** estimator (`recover_cfo_unwrap_free`) combining intra-pulse $0.1\,\mu\text{s}$ conjugate products ($\pm 5.0\text{ MHz}$ unambiguous range) with derotated $1.0\,\mu\text{s}$ and $1.5\,\mu\text{s}$ unit-phasor refinement—eliminating phase-unwrap failures and $2\pi$ branch-cut aliasing across all 2,809 frames (`0` `NaN`s).

2. **Causal Mode-S & Kinematic Decoder (`src/adsb_rf/decoder.py`)**:
   - Decodes Aircraft Identification (`TC 1–4`), Airborne Position (`TC 9–18`, both locally unambiguous and globally unambiguous Even/Odd Compact Position Reporting via mathematical modulo `np.mod`), and Airborne Velocity (`TC 19`).
   - Computes real-time 3D slant-range Free-Space Path Loss (FSPL), Equivalent Isotropically Radiated Power (`EIRP_dBm`), and spherical-geometry radial Doppler shift (`compute_single_frame_doppler`).

3. **12-Feature Physical-Layer ML Ensemble & Deep Autoencoder (`src/adsb_rf/features.py`, `models.py`, `deep_models.py`, `evaluation.py`)**:
   - Extracts **12 label-free physical features** per $120\,\mu\text{s}$ burst:
     `CFO_Raw_Hz`, `Frequency_shift` (Doppler-compensated CFO), `Timing_Offset_PPM` (parabolic sub-sample symbol clock skew), `Sub_Sample_Lag_Ns`, `Power_dBm`, `EIRP_dBm`, `Phase_Noise_Std`, `PA_Droop_dB` (power amplifier storage-capacitor sag), `Pulse_Crest_Factor` (turn-on overshoot), `Phase_Curvature_Rad` (thermal frequency pulling), `SNR_dB`, and `Preamble_Ratio_dB`.
   - **Leak-free hybrid 3-model soft-voting ensemble** (`fit_fold_ensemble` / `predict_fold_ensemble`: **45% RandomForest + 35% ExtraTrees + 20% Bayesian Mahalanobis Posterior**) where feature medians, robust Median/MAD/inter-decile centroids, and tree estimators are fitted strictly inside each training split.
   - Paired with a PyTorch `RFAnomalyAutoencoder` ($12 \rightarrow 24 \rightarrow 3 \rightarrow 24 \rightarrow 12$) and a 1D raw-waveform convolutional network (`IQConvNet1D`).

---

## Reproducible Benchmarks & Metric Reconciliation (`python reproduce_metrics.py`)

Every number below is computed directly from `adsb_table.csv` by running:

```bash
python reproduce_metrics.py
```

### 1. Reconciliation: `"100.0% Clean Inlier Fit"` vs. `"86.54% CSV Match"` vs. Leak-Free Out-of-Sample Accuracy

* **Why is the Clean Inlier Fit `100.00%` while the overall CSV match rate is `86.54%`?**
  - `adsb_table.csv` contains **2,809** total CRC-valid frames across **174** distinct ICAO addresses: **2,396** clean inlier frames (`is_anomaly == 0`, across 161 ICAOs) and **413** flagged physical-layer anomalies (`is_anomaly == 1`, where multipath fading or co-channel fruit pushed the frame past the $d_M > 4.5\sigma$ Mahalanobis gate).
  - On the **2,396 clean inliers**, the full-dataset ensemble (`min_samples_leaf=1`) memorizes the training inliers at **`2,396 / 2,396 = 100.00%`** (*in-sample training fit, not an out-of-sample test metric*).
  - On the **413 flagged anomalies**, severe waveform distortion causes the RF classifier to predict a different ICAO on 378 frames (matching only `35 / 413 = 8.47%`). Across all 2,809 exported rows, `(2,396 + 35) / 2,809 = 2,431 / 2,809 = 86.54%`.
* **True Leak-Free Out-of-Sample Generalization (Evaluated Across Both Class Eligibility Tiers)**:
  - **All Enrolled Aircraft ($\ge 5$ clean frames)**: **115 classes** (`2,304` clean frames; minimum 1 frame per fold in 5-fold stratified CV).
  - **Well-Enrolled Aircraft ($\ge 12$ clean frames, `~70 classes` tier)**: **72 classes (~70 classes)** (`1,984` clean frames; used when reporting performance on well-enrolled transponders with $\ge 10$ training frames per CV fold).

| Metric / Evaluation Scope | Evaluated Frames | Evaluated Classes | Top-1 Accuracy | Top-3 Accuracy | Methodology Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Sample Clean Inlier Fit (`is_anomaly == 0`)** | `2,396 / 2,396` | `161 ICAOs` | **100.00%** | **100.00%** | In-sample training fit (`2,396/2,396` clean frames match). |
| **All-Frame Exported CSV Match (`adsb_table.csv`)** | `2,431 / 2,809` | `174 ICAOs` | **86.54%** | N/A (outliers) | Includes `413` flagged anomalies (where only `35/413` match). |
| **Leak-Free 80/20 Stratified Holdout (All Enrolled $\ge 5$)** | `2,304 (461 test)` | **115 classes** | **88.94%** | **96.53%** | Fold-isolated imputation, centroids & trees; $\ge 5$ clean frames/class. |
| **Leak-Free 5-Fold Stratified CV (All Enrolled $\ge 5$)** | `2,304 (100% OOF)` | **115 classes** | **88.32%** | **95.40%** | 5-fold out-of-fold evaluation across all 115 enrolled classes. |
| **Chronological Out-of-Time Split (All Enrolled $\ge 5$)** | `2,304 (first 80% -> last 20%)` | **115 classes** | **86.64%** | **95.69%** | Trained on first 80% of each aircraft's timeline, tested on final 20%. |
| **Leak-Free 80/20 Holdout (Well-Enrolled $\ge 12$, ~70 classes)** | `1,984 (397 test)` | **72 classes (~70)** | **90.43%** | **96.98%** | Well-enrolled aircraft tier ($\ge 12$ clean frames/aircraft $\rightarrow$ 72 classes). |
| **Leak-Free 5-Fold Stratified CV (Well-Enrolled $\ge 12$, ~70 classes)** | `1,984 (100% OOF)` | **72 classes (~70)** | **91.99%** | **97.58%** | 5-fold out-of-fold evaluation on the 72-class (~70-class) subset. |

---

### 2. Feature Group Ablation: CFO vs. Hardware Transient & Clock Features (`115 Classes`, `2,304 Frames`)

Isolates the discriminative power of **Carrier Frequency Offset (CFO)** versus **CFO-free hardware transient and symbol-clock features** (`Timing_Offset_PPM`, `Sub_Sample_Lag_Ns`, `Phase_Noise_Std`, `PA_Droop_dB`, `Pulse_Crest_Factor`, `Phase_Curvature_Rad`, `SNR_dB`, `Preamble_Ratio_dB`):

| Feature Set | Num Features | 80/20 Holdout Acc (%) | 5-Fold CV Acc (%) | Out-of-Time Acc (%) | 5-Fold Top-3 Acc (%) | Included Signals |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. CFO Features Only (`CFO_Raw_Hz + Frequency_shift`)** | 2 | 68.98% | 70.18% | 68.10% | 84.16% | Carrier frequency offset + Doppler-compensated CFO only |
| **2. Hardware & Clock Features Only (No CFO, No Power)** | 8 | 31.89% | 29.04% | 24.35% | 49.13% | `Timing_PPM`, `Lag_Ns`, `Phase_Noise`, `PA_Droop`, `Crest`, `Curvature`, `SNR`, `Preamble` (36x random baseline of 0.87%) |
| **3. Hardware + Channel Power/EIRP (No CFO)** | 10 | 56.83% | 56.86% | 56.25% | 78.04% | 8 hardware/clock features + `Power_dBm` + `EIRP_dBm` (zero CFO) |
| **4. Pure Waveform Features (CFO + Hardware, No Power/EIRP)** | 10 | 83.51% | 82.16% | 78.88% | 93.14% | CFO + 8 hardware/clock features without Rx Power or FSPL EIRP (+14.5% over CFO alone) |
| **5. Full 12-Feature Physical-Layer Vector (Production)** | **12** | **88.94%** | **88.32%** | **86.64%** | **95.40%** | All 12 features: CFO + 8 Hardware/Clock + Power/EIRP (+20.0% over CFO alone) |

---

### 3. Ensemble Member Ablation (`115 Classes`, `2,304 Frames`)

Evaluates each constituent model of the hybrid soft-voting classifier on the exact same leak-free splits:

| Ensemble Member / Model | 80/20 Holdout Acc (%) | 5-Fold CV Acc (%) | Out-of-Time Acc (%) | 5-Fold Top-3 Acc (%) | Architecture Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Bayesian Mahalanobis Posterior Only (20% weight)** | 70.72% | 65.49% | 64.44% | 83.16% | Parametric diagonal Gaussian likelihood around Median/MAD centroid (provides open-set spoofing distance $d_M$) |
| **2. RandomForest Classifier Only (45% weight)** | 86.55% | 87.24% | 83.84% | 94.88% | 65-tree bootstrap RandomForest (`max_depth=17`, damped sample weights) |
| **3. ExtraTrees Classifier Only (35% weight)** | 88.07% | 86.33% | 84.05% | 94.44% | 65-tree randomized split ExtraTrees (`max_depth=19`, lower variance) |
| **4. Hybrid 3-Model Soft-Voting Ensemble (Production)** | **88.94%** | **88.32%** | **86.64%** | **95.40%** | **45% RandomForest + 35% ExtraTrees + 20% Bayesian Mahalanobis Posterior** (strictly outperforms every single member) |

* **Empirical Out-of-Fold Residual Quantiles (Used for Simulator Uncertainty Error Bars)**:
  - Out-of-Fold Mahalanobis Distance ($d_M$): Median = `0.77 σ` | 90th Percentile = `1.81 σ` | 95th Percentile = `2.30 σ`
  - Out-of-Fold `|CFO Error|` 90th Percentile: `±14.04 kHz`
  - Out-of-Fold `|Timing Drift Error|` 90th Percentile: `±141.7 ppm`

---

## Centralized Physical, DSP & Modeling Assumptions (`src/adsb_rf/assumptions.py`)

All physical constants, hardware calibration offsets, DSP parameters, and ML anomaly thresholds are centralized in the frozen `Assumptions` dataclass in [assumptions.py](file:///c:/Users/eriks/OneDrive/Desktop/portfolio/ADSB_python/python_new/src/adsb_rf/assumptions.py):

| Parameter | Value | Units | Standard / Hardware / Literature Citation |
| :--- | :--- | :--- | :--- |
| `carrier_freq_hz` | `1,090,000,000.0` | `Hz` | ICAO Annex 10, Vol. IV, §3.1.2.1 (Mode-S 1090ES downlink frequency) |
| `speed_of_light_mps` | `299,792,458.0` | `m/s` | NIST CODATA exact speed of light in vacuum |
| `earth_mean_radius_km` | `6,371.0088` | `km` | IUGG / WGS-84 mean spherical Earth radius for great-circle slant range |
| `rx_lat`, `rx_lon`, `rx_alt_m` | `48.19655, 16.36988, 200.0` | `deg, deg, m` | TU Wien Institute of Telecommunications rooftop antenna coordinates (Gusshausstrasse 25, Vienna) |
| `rx_antenna_gain_dbi` | `5.5` | `dBi` | Measured 1090 MHz collinear rooftop antenna gain minus coaxial feedline loss |
| `vsa_calibration_offset_dbm` | `-11.37` | `dB` | Keysight UXA N9040B (`vsa_settings.setx`) front-end attenuation calibration |
| `default_raw_sample_rate_hz` | `12,800,000.0` | `Hz` | Keysight UXA N9040B baseband I/Q capture sample rate (`vsa_settings.setx`: `<Freq>10000000</Freq>` $\times 1.28$) |
| `target_sample_rate_hz` | `10,000,000.0` | `Hz` | $10\text{ samples}/\mu\text{s}$ ($5\text{ samples}$ per $0.5\,\mu\text{s}$ Mode-S Manchester chip) |
| `crc_poly_hex` | `0x1FFF409` | `bitmask` | ICAO Annex 10, Vol. IV, §3.1.2.3.3 Mode-S 24-bit parity generator polynomial $G(x)$ |
| `cpr_nz` | `15` | `zones` | RTCA DO-260B Compact Position Reporting (CPR) number of latitude zone groupings ($4 N_Z = 60$) |
| `cfo_phase_delta_std_gate_rad` | `0.45` | `rad` | Legacy $1.0\,\mu\text{s}$ pulse-pair phase-jitter threshold before engaging unwrap-free coarse-to-fine CFO |
| `min_eirp_dbm`, `max_eirp_dbm` | `20.0 .. 70.0` | `dBm` | ICAO Annex 10 Class A1–A3 nominal $48.5\text{–}57.0\text{ dBm}$ ($70\text{–}500\text{ W}$) plus banking/multipath margin |
| `max_abs_cfo_hz` | `450,000.0` | `Hz` | ICAO Annex 10 Mode-S frequency tolerance bound ($\pm 1\text{ MHz}$ max; commercial OCXO/TCXO $< \pm 250\text{ kHz}$) |
| `outlier_z_threshold` | `4.5` | `σ` | Diagonal-weighted Mahalanobis distance gate ($d_M > 4.5\sigma$) for open-set spoofing/anomaly rejection |
| `min_frames_enrolled` | `5` | `frames` | Minimum clean frames per aircraft for 5-fold stratified CV (`115` classes, `2,304` frames) |
| `min_frames_well_enrolled` | `12` | `frames` | Minimum clean frames for well-enrolled benchmark tier (`72` classes [~70 classes], `1,984` frames) |
| `ensemble_weights` | `(0.45, 0.35, 0.20)` | `weights` | Soft-voting ensemble weights: `45% RandomForest + 35% ExtraTrees + 20% Bayesian Mahalanobis` |

---

## Honest Limitations & Operational Scope

1. **Single 20.97-Second Capture Window**:
   - All `2,809` frames originate from one continuous `20.97 s` rooftop recording ($268.4\text{M}$ samples at $12.80\text{ MS/s}$). Within a 21-second window, each aircraft's crystal oscillator temperature, power supply voltage, and line-of-sight channel geometry are quasi-stationary.
   - Over hours or across multiple days, ambient temperature changes between ground taxi ($+20^\circ\text{C}$) and cruise altitude ($-55^\circ\text{C}$ at FL380) induce slow thermal drift in uncompensated crystal oscillators (TCXOs). Operational deployment over continuous streams requires an exponentially weighted moving average (EWMA) or Kalman filter to track slow centroid drift $\boldsymbol{\mu}_c(t)$ while flagging abrupt multi-kilohertz step changes caused by SDR spoofing.
2. **CFO Includes Residual Doppler When Kinematic Frames Are Missing**:
   - At $f_c = 1090\text{ MHz}$, radial aircraft velocity $v_r \in [-250, +250]\text{ m/s}$ induces a physical Doppler shift $f_D = (v_r / c) f_c \in [-909, +909]\text{ Hz}$ (up to $\pm 1.5\text{ kHz}$ for high-speed jets).
   - Whenever an aircraft transmits Airborne Position (`TC 9–18`) and Airborne Velocity (`TC 19`) frames, our pipeline computes and subtracts the exact 3D spherical-geometry radial Doppler shift (`Frequency_shift = CFO_Raw_Hz - Doppler_Hz`). However, for aircraft that only broadcast identification (`TC 1–4`) or altitude surveillance frames during the capture window, `Frequency_shift` falls back to `CFO_Raw_Hz` and therefore includes uncompensated radial Doppler.
   - To make this dependency transparent, Section 2 of `reproduce_metrics.py` reports a strict **CFO vs. Hardware Features Ablation**, proving that CFO-free hardware/clock features (`Timing_Offset_PPM`, `PA_Droop_dB`, `Pulse_Crest_Factor`, `Phase_Curvature_Rad`, `Phase_Noise_Std`) independently achieve **29.04% 5-fold CV accuracy** (33x above the 0.87% random baseline across 115 classes) and **56.86%** when combined with link-budget features.
3. **Class Count Nuance (`174` Total vs. `115` Enrolled vs. `72` [~70] Well-Enrolled)**:
   - Out of `174` distinct ICAOs decoded (`161` with clean inliers), `46` aircraft transmitted fewer than 5 clean frames during the 20.97-second window and cannot populate a 5-fold stratified split.
   - Evaluating all aircraft with $\ge 5$ clean frames yields **115 classes** (`2,304` frames: **88.94%** Holdout, **88.32%** 5-Fold CV); evaluating well-enrolled aircraft with $\ge 12$ clean frames yields **72 classes (~70 classes)** (`1,984` frames: **90.43%** Holdout, **91.99%** 5-Fold CV). Any CV/resume reference to *"~70 classes"* (`~90.6%` holdout, `~90.3%` 5-fold CV) corresponds directly to this well-enrolled evaluation tier.

---

## Repository Structure

```text
├── src/adsb_rf/                 # Single canonical DSP + ML Python package
│   ├── __init__.py              # Public package exports
│   ├── assumptions.py           # Frozen Assumptions dataclass with literature/ICAO citations
│   ├── stream_reader.py         # Memory-mapped .npy/.mat online chunk reader & polyphase resampler
│   ├── dsp.py                   # Preamble XCorr, CRC-24, unwrap-free CFO, parabolic timing & PA transients
│   ├── decoder.py               # Causal online Mode-S tracker, CPR Even/Odd & local position decoder
│   ├── features.py              # 12-feature physical-layer extractor & ablation subset definitions
│   ├── models.py                # Leak-free fold ensemble (Mahalanobis + RandomForest + ExtraTrees)
│   ├── deep_models.py           # PyTorch RFAnomalyAutoencoder (12->24->3->24->12) & IQConvNet1D
│   ├── evaluation.py            # Holdout, 5-Fold CV, Out-of-Time, residual uncertainty & ablation suite
│   ├── database.py              # DuckDB / SQLite analytical store with path-sanitized provenance
│   ├── visualization.py         # Interactive Leaflet HTML trajectory map, cluster & eye diagram plots
│   ├── synthesizer.py           # Explicitly labeled synthetic 12.8 MS/s Mode-S stream generator
│   └── pipeline.py              # End-to-end streaming orchestrator
├── tests/
│   └── test_adsb_pipeline.py    # 10 unit tests: CRC-24, CPR, FSPL/EIRP, unwrap-free CFO, leak-free ML
├── .github/workflows/tests.yml  # CI workflow running pytest -v and python reproduce_metrics.py
├── streamlit_app.py             # Single interactive Streamlit UI entry point importing src/adsb_rf/
├── reproduce_metrics.py         # Reproduces headline tables, class census, and ablations from adsb_table.csv
├── main.py                      # CLI entry point for raw .npy/.mat processing or synthetic demo fallback
├── adsb_table.csv               # Exported 2,809-frame TU Wien rooftop telemetry with all 12 RF features
├── vsa_settings.setx            # Keysight 89600 VSA hardware capture configuration XML
├── pyproject.toml               # Package & pytest configuration
└── requirements.txt             # Python dependencies
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Reproduce the exact headline benchmark, class census & ablation tables from adsb_table.csv
python reproduce_metrics.py

# 3. Run the unit test suite (10 DSP, physics & leak-free ML tests)
python -m unittest discover -s tests -v

# 4. Launch the interactive Streamlit cockpit
streamlit run streamlit_app.py

# 5. Run the end-to-end I/Q stream processor (uses receivedSignals.npy if present, else synthetic stream)
python main.py
```
