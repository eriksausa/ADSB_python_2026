"""
Streamlit Interactive Cockpit for the 1090 MHz ADS-B Physical-Layer RF Fingerprinting
& Spoofing Detection Pipeline (`streamlit_app.py`).

Author: Erik Šauša, Dipl.-Ing. (M.Sc.)
Origin: 2026 Machine Learning & Deep Learning expansion of Erik Šauša's B.Sc. Thesis
        ("Analysis of Modulation Parameters of Surveillance Systems ADS-B and SSR",
        Institute of Telecommunications, TU Wien / Vienna University of Technology).

Single UI entry point importing the modular `src/adsb_rf/` package. Loads the TU Wien
Rooftop Capture telemetry (`adsb_table.csv` — 2,809 CRC-valid frames across 174 distinct
ICAO addresses from 268M I/Q samples at 12.80 MS/s) for leak-free evaluation, ablation
analysis, and residual-driven spoofing simulation.

Run locally:
    streamlit run streamlit_app.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from adsb_rf import (
    DEFAULT_ASSUMPTIONS,
    build_aircraft_profiles_from_df,
    compute_reconciliation_and_headline_metrics,
    format_default_feature_ablation_table,
    format_ensemble_ablation_table,
    format_reconciliation_table,
    get_assumptions_table_rows,
    map_planes,
    plot_eye_diagram,
    plot_iq_new,
    resample_data,
    run_ablation_suite,
    score_simulated_burst_with_residuals,
    synthesize_portfolio_iq_stream,
)

st.set_page_config(
    page_title="ADS-B RF Fingerprinting & ML Spoofing Detector | Erik Šauša",
    page_icon="📡",
    layout="wide",
)

CLUSTER_PALETTE_20 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#0077b6", "#2a9d8f", "#8338ec", "#fb8500", "#3a86ff",
    "#e63946", "#457b9d", "#6d597a", "#b56576", "#52b788",
]


# =============================================================================
# CACHED TELEMETRY, BACKTEST & ABLATION LOADERS (DELEGATED TO `src/adsb_rf/`)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_rooftop_csv_telemetry(csv_path_str: str, csv_mtime: float = 0.0) -> pd.DataFrame:
    """Loads the exported `adsb_table.csv` from the 268M-sample TU Wien rooftop capture."""
    df = pd.read_csv(csv_path_str)
    if "freq_offset_hz" in df.columns and "cfo_raw_hz" in df.columns:
        df["freq_offset_hz"] = df["freq_offset_hz"].fillna(df["cfo_raw_hz"])
    return df


@st.cache_data(show_spinner="Computing per-aircraft RF profiles...")
def get_cached_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Delegates aircraft profile computation to `adsb_rf.evaluation.build_aircraft_profiles_from_df`."""
    return build_aircraft_profiles_from_df(df)


@st.cache_data(show_spinner="Running leak-free 80/20 Holdout, 5-Fold CV & Out-of-Time backtests...")
def get_cached_headline_metrics(df: pd.DataFrame) -> dict:
    """Computes live leak-free metrics and out-of-fold residual uncertainty from `adsb_table.csv`."""
    return compute_reconciliation_and_headline_metrics(df)


@st.cache_data(show_spinner="Running CFO vs. Hardware & Ensemble Member Ablations...")
def get_cached_ablations(df: pd.DataFrame, min_frames: int, _precomputed_tier: dict | None = None) -> dict:
    """Runs the feature and ensemble member ablation suites via `adsb_rf.evaluation.run_ablation_suite`."""
    return run_ablation_suite(df, min_frames=min_frames, precomputed_full_res=_precomputed_tier)


# =============================================================================
# MAIN STREAMLIT DASHBOARD UI
# =============================================================================

def main() -> None:
    csv_path = ROOT_DIR / "adsb_table.csv"

    # --- Header & Author Attribution ---
    st.title("📡 1090 MHz ADS-B Physical-Layer RF Fingerprinting & Spoofing Detection")
    st.markdown(
        "#### **By Erik Šauša, Dipl.-Ing. (M.Sc.)**  \n"
        "*A 2026 Machine Learning & Deep Neural Network expansion of my Bachelor's Thesis at the "
        "**Institute of Telecommunications, TU Wien (Vienna University of Technology)**: "
        "\"Analysis of Modulation Parameters of Surveillance Systems ADS-B and SSR.\"*"
    )

    # --- Explicit Data Provenance, Public Benchmarks & Synthetic Demo Notice ---
    st.warning(
        "⚠️ **Data Provenance, Privacy & Synthetic Simulation Notice:**  \n"
        "* **Real Over-the-Air Rooftop Capture (`adsb_table.csv`)**: Contains **2,809 CRC-valid Mode-S frames** "
        "(174 distinct ICAO hex addresses) demodulated from a single **20.97-second ($268{,}434{,}198$-sample at $12.80\\text{ MS/s}$)** "
        "complex baseband capture recorded on the roof of the TU Wien Telecommunications Institute using a Keysight UXA N9040B Vector Signal Analyzer. "
        "All aircraft identifiers are public commercial aviation broadcasts governed by **ICAO Annex 10 (Vol. IV)** and **RTCA DO-260B**; "
        "zero private or PII data are present.  \n"
        "* **Interactive Spoofing Simulator & Fallback Mode**: The sidebar **Live RF Spoofing Simulator** injects **synthetic/simulated RF bursts** "
        "with user-controlled oscillator/timing offsets and evaluates them using empirical **5-fold out-of-fold backtest residuals** from the real dataset. "
        "Similarly, when the raw 2.0 GB `.npy` capture is absent, `main.py` provides an explicitly labeled synthetic I/Q generator (`src/adsb_rf/synthesizer.py`)."
    )

    # --- Abstract & Project Motivation Card ---
    st.info(
        "**Abstract & Physical-Layer Security Motivation:**  \n"
        "Modern Air Traffic Control relies on **ADS-B** (*Automatic Dependent Surveillance–Broadcast* at $1090\\text{ MHz}$), "
        "where aircraft broadcast their GPS coordinates, altitude, velocity, and 24-bit ICAO identity every half-second. "
        "However, **ADS-B has zero cryptographic authentication**: anyone with a \\$300 Software-Defined Radio (SDR) can "
        "broadcast spoofed digital messages claiming to be a commercial airliner or inject ghost aircraft onto radar screens.  \n\n"
        "**The Physical-Layer Solution:**  \n"
        "While digital bits are trivial to forge, an attacker cannot easily replicate the microscopic **analog hardware imperfections** "
        "of a specific aircraft's 250W–500W Mode-S transponder—such as crystal local oscillator carrier frequency offset (CFO), "
        "Manchester $2\\text{ MHz}$ symbol clock drift (ppm), power amplifier (PA) storage-capacitor voltage droop, turn-on overshoot, "
        "and phase noise. This pipeline extracts a **12-dimensional physical-layer RF fingerprint** (`src/adsb_rf/`) from every $120\\,\\mu\\text{s}$ burst "
        "to authenticate the transmitting aircraft hardware and flag spoofing anomalies."
    )

    if not csv_path.exists():
        st.error(f"Telemetry file `{csv_path.name}` not found. Run `python main.py` first to generate `adsb_table.csv`.")
        return

    df = load_rooftop_csv_telemetry(str(csv_path), csv_path.stat().st_mtime)
    prof_df = get_cached_profiles(df)
    metrics_bundle = get_cached_headline_metrics(df)
    res_unc = metrics_bundle["residual_uncertainty"]

    # --- Sidebar Author & Controls ---
    st.sidebar.markdown("### 👨‍💻 **Erik Šauša, Dipl.-Ing.**")
    st.sidebar.caption("M.Sc. & B.Sc. in Telecommunications & Information Engineering — TU Wien")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Evaluation Tier & Security Gate")

    tier_choice = st.sidebar.radio(
        "Benchmark Class Eligibility Tier:",
        [
            f"Well-Enrolled (>=12 clean frames: {metrics_bundle['tier12']['n_classes']} classes, ~70)",
            f"All Enrolled (>=5 clean frames: {metrics_bundle['tier5']['n_classes']} classes)",
        ],
        index=0,
        help="Selects whether top-level KPIs display the 72 well-enrolled aircraft (~70 classes, >=12 clean frames) or all 115 aircraft with >=5 clean frames (minimum for 5-fold CV).",
    )
    active_tier = metrics_bundle["tier5"] if ">=5" in tier_choice else metrics_bundle["tier12"]

    outlier_z = st.sidebar.slider(
        "Mahalanobis Spoofing Gate (σ):",
        min_value=2.5,
        max_value=6.0,
        value=float(DEFAULT_ASSUMPTIONS.outlier_z_threshold),
        step=0.1,
        help="Frames whose physical-layer RF vector sits further than this many standard deviations from their claimed aircraft's hardware centroid are flagged as spoofing.",
    )

    # --- Live Reproducible Top KPI Row (Zero Hardcoded Numbers) ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        "CRC-Valid Frames",
        f"{metrics_bundle['total_frames']:,}",
        delta=f"{metrics_bundle['clean_frames']:,} clean / {metrics_bundle['anom_frames']} anom",
        delta_color="off",
    )
    c2.metric(
        "Evaluated Classes",
        f"{active_tier['n_classes']} / {metrics_bundle['total_icaos']}",
        delta=f"{active_tier['n_frames']:,} eval frames",
        delta_color="off",
    )
    c3.metric(
        "80/20 Holdout Acc",
        f"{active_tier['holdout_top1']*100:.2f}%",
        delta=f"Top-3: {active_tier['holdout_top3']*100:.2f}%",
    )
    c4.metric(
        "5-Fold Stratified CV",
        f"{active_tier['cv_top1']*100:.2f}%",
        delta=f"Top-3: {active_tier['cv_top3']*100:.2f}%",
    )
    c5.metric(
        "Out-of-Time Split Acc",
        f"{active_tier['oot_top1']*100:.2f}%",
        delta=f"Top-3: {active_tier['oot_top3']*100:.2f}%",
    )
    c6.metric(
        "Exported CSV Match",
        f"{metrics_bundle['all_match_rate']*100:.2f}%",
        delta=f"{metrics_bundle['clean_inlier_fit']*100:.1f}% on clean inliers",
        delta_color="off",
    )

    # --- Sidebar Interactive Spoofing Simulator ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Live RF Spoofing Simulator (Synthetic)")
    st.sidebar.caption(
        "Simulate an SDR transmitter broadcasting a fake ADS-B frame claiming to be one of the enrolled aircraft, "
        "and test it against the Physical-Layer Mahalanobis gate + empirical out-of-fold residual intervals."
    )

    enrolled_list = prof_df["icao_hex"].tolist() if not prof_df.empty else ["4D00D7"]
    target_icao = st.sidebar.selectbox(
        "Claimed Aircraft ICAO:",
        options=enrolled_list,
        format_func=lambda x: f"{x} ({prof_df.loc[prof_df['icao_hex'] == x, 'callsign'].values[0]})"
        if not prof_df.empty and x in prof_df["icao_hex"].values
        else x,
    )

    target_row = prof_df[prof_df["icao_hex"] == target_icao].iloc[0] if not prof_df.empty else None
    default_cfo = float(target_row["mean_cfo_khz"]) if target_row is not None else -16.1
    default_ppm = float(target_row["mean_timing_ppm"]) if target_row is not None else -4.3
    default_eirp = float(target_row["mean_eirp_dbm"]) if target_row is not None else 56.4

    preset_attack = st.sidebar.radio(
        "Transmitter Scenario:",
        ["Legitimate Aircraft Transponder", "SDR Ground Spoofer (Mismatched Oscillator)"],
        index=1,
    )

    if preset_attack == "Legitimate Aircraft Transponder":
        sim_cfo_khz = st.sidebar.slider("Transmitted CFO (kHz):", -200.0, 200.0, round(default_cfo, 1), 0.5)
        sim_ppm = st.sidebar.slider("Symbol Clock Drift (ppm):", -300.0, 300.0, round(default_ppm, 1), 1.0)
        sim_eirp = st.sidebar.slider("Estimated EIRP (dBm):", 20.0, 75.0, round(default_eirp, 1), 0.5)
    else:
        spoof_default_cfo = round(default_cfo - 35.0 if default_cfo > 0 else default_cfo + 35.0, 1)
        spoof_default_ppm = round(default_ppm - 110.0 if default_ppm > 0 else default_ppm + 110.0, 1)
        sim_cfo_khz = st.sidebar.slider("Transmitted CFO (kHz):", -200.0, 200.0, spoof_default_cfo, 0.5)
        sim_ppm = st.sidebar.slider("Symbol Clock Drift (ppm):", -300.0, 300.0, spoof_default_ppm, 1.0)
        sim_eirp = st.sidebar.slider("Estimated EIRP (dBm):", 20.0, 75.0, round(default_eirp + 4.0, 1), 0.5)

    sim_verdict = score_simulated_burst_with_residuals(
        prof_df=prof_df,
        target_icao=target_icao,
        sim_cfo_khz=sim_cfo_khz,
        sim_ppm=sim_ppm,
        sim_eirp_dbm=sim_eirp,
        outlier_z=outlier_z,
        residual_uncertainty=res_unc,
    )

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎯 RF Fingerprint Clusters & Spoofing Gate",
            "🧪 Ablation Study, Assumptions & Limitations",
            "🗺️ Vienna Airspace Trajectory Map",
            "📊 Aircraft Hardware RF Profiles",
            "📈 I/Q Waveform, Spectrum & Eye Diagram",
        ]
    )

    with tab1:
        col_left, col_right = st.columns([2.1, 1.05])

        with col_left:
            upper_bound = min(25, max(4, len(prof_df)))
            default_val = min(12, upper_bound)
            max_classes_plot = st.slider(
                "Number of Top Aircraft to Display on Cluster Plot:",
                min_value=3,
                max_value=upper_bound,
                value=default_val,
            )
            top_icaos = set(prof_df.head(max_classes_plot)["icao_hex"].tolist())
            top_icaos.add(target_icao)

            plot_df = df[df["icao_hex"].isin(top_icaos)].dropna(subset=["freq_offset_hz", "timing_offset_ppm"]).copy()
            plot_df["cfo_khz"] = plot_df["freq_offset_hz"] / 1e3

            fig, ax = plt.subplots(figsize=(10, 5.8), dpi=130)

            for idx, (icao_val, grp) in enumerate(plot_df.groupby("icao_hex")):
                cs_lbl = prof_df.loc[prof_df["icao_hex"] == icao_val, "callsign"].values
                cs_str = f" ({cs_lbl[0]})" if len(cs_lbl) > 0 and cs_lbl[0] != "N/A" else ""
                is_target = str(icao_val) == str(target_icao)
                color_hex = CLUSTER_PALETTE_20[idx % len(CLUSTER_PALETTE_20)]
                ax.scatter(
                    grp["cfo_khz"],
                    grp["timing_offset_ppm"],
                    label=f"{icao_val}{cs_str}",
                    s=75 if is_target else 42,
                    alpha=0.92 if is_target else 0.70,
                    edgecolors="black" if is_target else "white",
                    linewidth=1.1 if is_target else 0.5,
                    color=color_hex,
                )

            # Plot empirical 90% backtest residual error bars around the claimed aircraft centroid
            if target_row is not None:
                cfo_err_bar = float(target_row.get("cfo_iqr_90_khz", res_unc["cfo_err_p90_khz"]))
                ppm_err_bar = float(target_row.get("ppm_iqr_90", res_unc["ppm_err_p90"]))
                ax.errorbar(
                    [default_cfo],
                    [default_ppm],
                    xerr=[cfo_err_bar],
                    yerr=[ppm_err_bar],
                    fmt="o",
                    color="#1d3557",
                    ecolor="#1d3557",
                    elinewidth=1.8,
                    capsize=5,
                    zorder=9,
                    label=f"Claimed {target_icao} Centroid ±90% Residual CI",
                )

            is_spoof = sim_verdict["is_spoof"]
            ax.scatter(
                [sim_cfo_khz],
                [sim_ppm],
                color="#d90429" if is_spoof else "#2b9348",
                marker="X" if is_spoof else "*",
                s=260,
                edgecolors="black",
                linewidth=1.5,
                zorder=10,
                label=f"Simulated Burst ({'SPOOF FLAGGED' if is_spoof else 'AUTHENTIC'})",
            )

            ax.set_title("Physical-Layer RF Fingerprint Space: Carrier Frequency Offset vs. Symbol Clock Drift", weight="bold")
            ax.set_xlabel("Doppler-Compensated Carrier Frequency Offset (kHz)")
            ax.set_ylabel("Symbol Clock Timing Offset (ppm)")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best", fontsize=8, ncol=2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.subheader("🔍 Live Burst Authentication Verdict")
            if is_spoof:
                st.error(f"🚨 **SPOOFING / ANOMALY DETECTED** for claimed ICAO `{target_icao}`")
            else:
                st.success(f"✅ **HARDWARE AUTHENTICATED** for claimed ICAO `{target_icao}`")

            st.markdown(
                f"""
                * **Claimed Aircraft**: `{target_icao}` (`{target_row['callsign'] if target_row is not None else 'N/A'}`)
                * **Enrolled Hardware Centroid (±90% Empirical Residual CI)**:
                  * Median CFO: `{default_cfo:+.2f} ± {sim_verdict['cfo_ci90_khz']:.2f} kHz`
                  * Median Timing Skew: `{default_ppm:+.1f} ± {sim_verdict['ppm_ci90']:.1f} ppm`
                  * Median EIRP: `{default_eirp:.1f} dBm`
                * **Simulated Burst Measurements**:
                  * Burst CFO: `{sim_cfo_khz:+.2f} kHz` ($\\Delta = {sim_cfo_khz - default_cfo:+.2f}\\text{{ kHz}}$)
                  * Burst Timing Skew: `{sim_ppm:+.1f} ppm` ($\\Delta = {sim_ppm - default_ppm:+.1f}\\text{{ ppm}}$)
                  * Burst EIRP: `{sim_eirp:.1f} dBm`
                * **Residual-Calibrated Security Metrics**:
                  * **Mahalanobis Distance ($d_M$)**: `{sim_verdict['d_claimed']:.2f} σ`
                  * **90% Residual-Propagated Band**: `[{sim_verdict['d_ci_low']:.2f} σ, {sim_verdict['d_ci_high']:.2f} σ]`
                  * **Configured Gate / 95% OOF Inlier Bound**: `{outlier_z:.1f} σ` / `{sim_verdict['oof_p95_gate']:.2f} σ`
                  * **Hardware Match Probability**: `{sim_verdict['claimed_prob']*100:.1f}%`
                  * **Closest RF Emitter Match**: `{sim_verdict['pred_icao']}` (`{sim_verdict['pred_cs']}`)
                """
            )

    with tab2:
        st.subheader("1️⃣ Metric Reconciliation: 100.0% Clean Inlier Fit vs. 86.54% CSV Match vs. Leak-Free OOS")
        st.markdown(
            "Reconciles the **in-sample training fit on clean inliers (`100.00%`)**, the **overall exported CSV match rate (`86.54%`)** "
            "(which includes `413` flagged physical-layer anomalies where `d_M > 4.5 σ`), and the **leak-free out-of-sample evaluations** "
            "across both the **115-class ($\\ge 5$ clean frames)** and **72-class (~70 classes, $\\ge 12$ clean frames)** tiers."
        )
        st.dataframe(format_reconciliation_table(metrics_bundle), width="stretch", hide_index=True)

        st.subheader(f"2️⃣ Feature Group Ablation: CFO vs. Hardware Transient & Clock Features ({active_tier['n_classes']} Classes)")
        run_live_feat_ablation = st.checkbox(
            "Re-fit all 4 partial feature subsets live in browser (otherwise displays cached benchmark table to conserve Streamlit Cloud RAM)",
            value=False,
        )
        if run_live_feat_ablation:
            abl_dict = get_cached_ablations(df, min_frames=active_tier["min_frames"], _precomputed_tier=active_tier)
            feat_abl_df = abl_dict["feature_ablation_df"]
        else:
            feat_abl_df = format_default_feature_ablation_table(active_tier)
        st.dataframe(feat_abl_df, width="stretch", hide_index=True)

        st.subheader(f"3️⃣ Ensemble Member Ablation: Mahalanobis vs. RandomForest vs. ExtraTrees ({active_tier['n_classes']} Classes)")
        st.dataframe(format_ensemble_ablation_table(active_tier), width="stretch", hide_index=True)

        st.subheader("4️⃣ Explicit Physical, DSP & Modeling Assumptions (`src/adsb_rf/assumptions.py`)")
        st.dataframe(pd.DataFrame(get_assumptions_table_rows()), width="stretch", hide_index=True)

        st.subheader("5️⃣ Honest Limitations & Operational Scope")
        st.markdown(
            """
            1. **Single 20.97-Second Capture Window**: All `2,809` frames originate from one continuous `20.97 s` rooftop recording ($268.4\\text{M}$ samples at $12.8\\text{ MS/s}$). Within $21\\text{ seconds}$, transponder crystal oscillator temperature, supply voltage, and line-of-sight channel geometry are quasi-stationary. Cross-day re-identification requires continuous Kalman/EWMA centroid tracking as aircraft climb from $+20^\\circ\\text{C}$ to $-55^\\circ\\text{C}$ at cruise altitude.
            2. **CFO Includes Residual Doppler When Kinematics Are Missing**: At $1090\\text{ MHz}$, radial motion induces up to $f_D = \\pm (v/c) f_c \\approx \\pm 900\\text{ Hz}$ of Doppler shift for subsonic commercial airliners. While our pipeline explicitly subtracts spherical-geometry Doppler (`compute_single_frame_doppler`) whenever CPR position (`TC 9–18`) and velocity (`TC 19`) frames are decoded, aircraft that only broadcast identification or altitude frames during the capture retain uncompensated radial Doppler in `Frequency_shift`. The **Feature Group Ablation** above isolates `CFO Only` vs. `Hardware & Clock Features Only (No CFO)` to quantify this dependency transparently.
            3. **Evaluated Class Counts (`174` Total vs. `115` Enrolled vs. `72` [~70] Well-Enrolled)**: Of the `174` distinct ICAOs decoded (`161` with clean inliers), `46` clean aircraft transmitted fewer than 5 frames in $20.97\\text{ s}$ and cannot populate a 5-fold stratified split. All headline metrics state the exact evaluated frame and class counts (`115` classes at $\\ge 5$ frames; `72` classes [~70 classes] at $\\ge 12$ frames).
            """
        )

    with tab3:
        st.subheader("🗺️ Interactive Mode-S / ADS-B Trajectory Map (TU Wien Receiver)")
        map_file = ROOT_DIR / "adsb_map.html"
        if not map_file.exists():
            messages_for_map = [
                {
                    "TC": int(r["tc"]),
                    "ICAO": int(str(r["icao_hex"]), 16),
                    "ICAO_Hex": str(r["icao_hex"]),
                    "Predicted_ICAO_Hex": str(r.get("predicted_icao_hex", r["icao_hex"])),
                    "Call_sign": None if pd.isna(r.get("callsign")) else str(r.get("callsign")),
                    "Latitude": float(r["latitude"]) if pd.notna(r.get("latitude")) else float("nan"),
                    "Longitude": float(r["longitude"]) if pd.notna(r.get("longitude")) else float("nan"),
                    "Altitude": float(r["altitude_ft"]) if pd.notna(r.get("altitude_ft")) else 0.0,
                    "Power": float(r["rx_power_dbm"]) if pd.notna(r.get("rx_power_dbm")) else -75.0,
                    "Distance_from_Rx": float(r["distance_m"]) if pd.notna(r.get("distance_m")) else 0.0,
                    "Frequency_shift": float(r["freq_offset_hz"]) if pd.notna(r.get("freq_offset_hz")) else 0.0,
                    "Timing_Offset_PPM": float(r["timing_offset_ppm"]) if pd.notna(r.get("timing_offset_ppm")) else 0.0,
                    "Is_Anomaly": bool(r.get("is_anomaly", 0)),
                    "Anomaly_Reasons": str(r.get("anomaly_reasons", "Normal")),
                }
                for _, r in df.iterrows()
            ]
            map_planes(messages_for_map, output_html=map_file)
        components.html(map_file.read_text(encoding="utf-8"), height=560, scrolling=False)

    with tab4:
        st.subheader("✈️ Enrolled Aircraft RF Fingerprint Profiles")
        st.caption(
            "Aggregated per-aircraft physical-layer hardware profiles across the 20.97-second TU Wien rooftop capture "
            "(sorted by number of received frames, including 90% empirical residual intervals)."
        )
        st.dataframe(prof_df, width="stretch", hide_index=True)

    with tab5:
        st.subheader("📈 Physical-Layer Baseband I/Q Waveform & Mode-S PPM Eye Diagram")
        col_iq, col_eye = st.columns(2)
        iq_png = ROOT_DIR / "iq_spectrum_sample.png"
        eye_png = ROOT_DIR / "eye_diagram_sample.png"

        if not iq_png.exists() or not eye_png.exists():
            synth_raw = synthesize_portfolio_iq_stream(duration_sec=0.05)
            synth_10m = resample_data(synth_raw, fs_raw=12.8e6, fs_target=10.0e6)
            burst_10m = synth_10m[:1200]
            plot_iq_new(
                [{"Timestamp": 0.0, "ICAO_Hex": "3C6541", "IQ": synth_raw[:1536], "IQ_res": burst_10m}],
                save_path=iq_png,
            )
            plot_eye_diagram(burst_10m, msps=10.0, pulse_duration_us=1.0, save_path=eye_png)

        with col_iq:
            st.image(str(iq_png), caption="1090 MHz Mode-S Baseband I/Q, Envelope & Phase Trajectory", width="stretch")

        with col_eye:
            st.image(str(eye_png), caption="10 MS/s Mode-S Manchester PPM Eye Diagram (2-Symbol Window)", width="stretch")


if __name__ == "__main__":
    main()
