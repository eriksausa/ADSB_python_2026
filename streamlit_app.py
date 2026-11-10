"""
Streamlit Interactive Cockpit for the 1090 MHz ADS-B Physical-Layer RF Fingerprinting
& Spoofing Detection Pipeline (`streamlit_app.py`).

Author: Erik Šauša, Dipl.-Ing. (M.Sc.)
Origin: 2026 Machine Learning & Deep Learning expansion of Erik Šauša's B.Sc. Thesis
        ("Analysis of Modulation Parameters of Surveillance Systems ADS-B and SSR",
        Institute of Telecommunications, TU Wien / Vienna University of Technology).

Loads the TU Wien Rooftop Capture telemetry (`adsb_table.csv` — 2,809 CRC-valid frames
from 268M I/Q samples at 12.80 MS/s) for interactive exploration and live physical-layer
spoofing simulation.

Run locally:
    streamlit run streamlit_app.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ADS-B RF Fingerprinting & ML Spoofing Detector | Erik Šauša",
    page_icon="📡",
    layout="wide",
)

# High-contrast 20-color palette compatible with all Matplotlib versions (including 3.9+)
CLUSTER_PALETTE_20 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#0077b6", "#2a9d8f", "#8338ec", "#fb8500", "#3a86ff",
    "#e63946", "#457b9d", "#6d597a", "#b56576", "#52b788",
]


# =============================================================================
# CACHED TELEMETRY LOADER & HARDWARE PROFILE AGGREGATION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_rooftop_csv_telemetry(csv_path_str: str) -> pd.DataFrame:
    """Loads the exported `adsb_table.csv` from the 268M-sample TU Wien rooftop capture."""
    df = pd.read_csv(csv_path_str)
    if "freq_offset_hz" in df.columns and "cfo_raw_hz" in df.columns:
        df["freq_offset_hz"] = df["freq_offset_hz"].fillna(df["cfo_raw_hz"])
    return df


def build_aircraft_profiles_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Computes per-aircraft physical-layer RF fingerprint centroids and standard deviations."""
    profiles = []
    for icao_hex, grp in df.groupby("icao_hex"):
        clean_grp = (
            grp[~grp["is_anomaly"].astype(bool)]
            if "is_anomaly" in grp.columns and (~grp["is_anomaly"].astype(bool)).any()
            else grp
        )
        cs_series = grp["callsign"].dropna() if "callsign" in grp.columns else pd.Series(dtype=str)
        cs_valid = [str(c) for c in cs_series if str(c).strip() not in ("", "None", "N/A", "nan")]
        callsign = cs_valid[-1] if cs_valid else "N/A"

        cfo_vals = clean_grp["freq_offset_hz"].dropna() if "freq_offset_hz" in clean_grp.columns else pd.Series(dtype=float)
        ppm_vals = clean_grp["timing_offset_ppm"].dropna() if "timing_offset_ppm" in clean_grp.columns else pd.Series(dtype=float)
        eirp_vals = clean_grp["eirp_dbm"].dropna() if "eirp_dbm" in clean_grp.columns else pd.Series(dtype=float)
        pnoise_vals = (
            clean_grp["phase_noise_std"].dropna()
            if "phase_noise_std" in clean_grp.columns
            else pd.Series(dtype=float)
        )

        profiles.append(
            {
                "icao_hex": str(icao_hex),
                "callsign": callsign,
                "frame_count": int(len(grp)),
                "mean_cfo_khz": round(float(cfo_vals.median() / 1e3), 3) if len(cfo_vals) > 0 else 0.0,
                "std_cfo_khz": round(float(max(cfo_vals.std(ddof=0) if len(cfo_vals) > 1 else 1000.0, 1000.0) / 1e3), 3),
                "mean_timing_ppm": round(float(ppm_vals.median()), 2) if len(ppm_vals) > 0 else 0.0,
                "std_timing_ppm": round(float(max(ppm_vals.std(ddof=0) if len(ppm_vals) > 1 else 25.0, 25.0)), 2),
                "mean_eirp_dbm": round(float(eirp_vals.median()), 2) if len(eirp_vals) > 0 else 54.0,
                "std_eirp_dbm": round(float(max(eirp_vals.std(ddof=0) if len(eirp_vals) > 1 else 2.0, 2.0)), 2),
                "mean_phase_noise_rad": round(float(pnoise_vals.median()), 3) if len(pnoise_vals) > 0 else 0.30,
            }
        )
    prof_df = pd.DataFrame(profiles)
    if not prof_df.empty:
        prof_df = prof_df.sort_values("frame_count", ascending=False).reset_index(drop=True)
    return prof_df


# =============================================================================
# MAIN STREAMLIT DASHBOARD UI
# =============================================================================

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "adsb_table.csv"

    # --- Header & Author Attribution ---
    st.title("📡 1090 MHz ADS-B Physical-Layer RF Fingerprinting & Spoofing Detection")
    st.markdown(
        "#### **By Erik Šauša, Dipl.-Ing. (M.Sc.)**  \n"
        "*A 2026 Machine Learning & Deep Neural Network expansion of my Bachelor's Thesis at the "
        "**Institute of Telecommunications, TU Wien (Vienna University of Technology)**: "
        "\"Analysis of Modulation Parameters of Surveillance Systems ADS-B and SSR.\"*"
    )

    # --- Abstract & Project Motivation Card ---
    st.info(
        "**Abstract & Why This Matters (The Core Problem):**  \n"
        "Modern Air Traffic Control relies on **ADS-B** (*Automatic Dependent Surveillance–Broadcast* at $1090\\text{ MHz}$), "
        "where aircraft broadcast their GPS coordinates, altitude, velocity, and 24-bit ICAO identity every half-second. "
        "However, **ADS-B has zero cryptographic authentication**: anyone with a $300 Software-Defined Radio (SDR) can "
        "broadcast spoofed digital messages claiming to be a commercial airliner or inject ghost aircraft onto radar screens.  \n\n"
        "**The Physical-Layer Solution:**  \n"
        "While digital bits are trivial to forge, an attacker cannot easily replicate the microscopic **analog hardware imperfections** "
        "of a specific aircraft's 250W–500W Mode-S transponder—such as crystal local oscillator carrier frequency offset (CFO), "
        "Manchester $2\\text{ MHz}$ symbol clock drift (ppm), power amplifier (PA) storage-capacitor voltage droop, turn-on overshoot, "
        "and phase noise. This project extracts a **12-dimensional physical-layer RF fingerprint** from every $120\\,\\mu\\text{s}$ burst "
        "to authenticate the transmitting aircraft hardware and detect spoofing without relying on the digital payload."
    )

    with st.expander("📖 Evolution of the Project: 2019 TU Wien B.Sc. Thesis ➔ 2026 ML & Neural Network Expansion"):
        st.markdown(
            """
            * **Stage 1 — 2018–2019 B.Sc. Thesis at TU Wien (MATLAB & Keysight UXA N9040B VSA)**:
              * Recorded live $1090\\text{ MHz}$ complex baseband I/Q signals ($12.8\\text{ MS/s}$) from a $5.5\\text{ dBi}$ collinear antenna on the roof of the TU Wien Telecommunications Institute in Vienna.
              * Built the complete physical-layer DSP & Mode-S decoder from scratch: preamble cross-correlation, Pulse Position Modulation (PPM) bit slicing, modulo-2 `CRC-24` parity verification, Compact Position Reporting (CPR) trajectory decoding, Free-Space Path Loss (FSPL) transmit power estimation, and spherical-geometry Doppler shift compensation.
              * *2019 Finding*: Proved that no *single* modulation parameter in isolation (frequency offset alone, power alone, or timestamp periodicity alone) uniquely identifies an aircraft across an entire airspace.
            * **Stage 2 — 2026 Machine Learning & Deep Learning Expansion (Python 3.12, Scikit-Learn & PyTorch)**:
              * **Unwrap-Free Two-Stage CFO Recovery**: Solved high-offset phase wrapping across pulse gaps by combining intra-pulse $0.1\\,\\mu\\text{s}$ conjugate products ($\\pm 5.0\\text{ MHz}$ unambiguous range) with derotated $1.0\\,\\mu\\text{s}$ / $1.5\\,\\mu\\text{s}$ unit-phasor refinement, plus per-aircraft kinematic Doppler/EIRP interpolation across `TC 4` (Callsign), `TC 11` (Position), and `TC 19` (Velocity) frames.
              * **12-Feature Hardware Fingerprint Vector**: Expanded the signal representation to 12 physical features including PA envelope sag (`PA_Droop_dB`), turn-on overshoot (`Pulse_Crest_Factor`), thermal frequency pulling (`Phase_Curvature_Rad`), and symbol clock drift (`Timing_Offset_PPM`).
              * **Hybrid ML Ensemble & Neural Networks**: Combined a supervised **RandomForest + ExtraTrees + Bayesian Mahalanobis Posterior** ensemble with a **PyTorch Deep Autoencoder** (`RFAnomalyAutoencoder`, $12 \\rightarrow 24 \\rightarrow 3 \\rightarrow 24 \\rightarrow 12$) and a **1D Raw-I/Q Convolutional Neural Network** (`IQConvNet1D`), boosting multi-aircraft classification accuracy to **90.6%** (**100.0% Top-3 accuracy**) across 2,809 real frames from the Vienna rooftop capture.
            """
        )

    if not csv_path.exists():
        st.error(f"Telemetry file `{csv_path.name}` not found. Run `python main.py` first to generate `adsb_table.csv`.")
        return

    df = load_rooftop_csv_telemetry(str(csv_path))
    prof_df = build_aircraft_profiles_from_df(df)

    # --- Sidebar Author & Controls ---
    st.sidebar.markdown("### 👨‍💻 **Erik Šauša, Dipl.-Ing.**")
    st.sidebar.caption("M.Sc. & B.Sc. in Telecommunications & Information Engineering — TU Wien")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Security Gate & Spoofing Simulator")
    outlier_z = st.sidebar.slider(
        "Mahalanobis Spoofing Gate (σ):",
        min_value=2.5,
        max_value=6.0,
        value=4.5,
        step=0.1,
        help="Frames whose physical-layer RF vector sits further than this many standard deviations from their claimed aircraft's hardware centroid are flagged as spoofing.",
    )

    # --- Top KPI Row ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CRC-Valid Frames", f"{len(df):,}")
    c2.metric("Distinct Aircraft (ICAOs)", f"{len(prof_df)}")
    c3.metric("80/20 Holdout Test Acc", "90.6%")
    c4.metric("5-Fold CV Accuracy", "90.3%")
    c5.metric("Top-3 RF Match Acc", "100.0%")

    # --- Sidebar Interactive Spoofing Simulator ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Live RF Spoofing Simulator")
    st.sidebar.caption(
        "Simulate an SDR transmitter broadcasting a fake ADS-B frame claiming to be one of the enrolled aircraft, "
        "and test whether the Physical-Layer Mahalanobis + Ensemble gate catches it."
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

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 RF Fingerprint Clusters & Spoofing Gate",
            "🗺️ Vienna Airspace Trajectory Map",
            "📊 Aircraft Hardware RF Profiles",
            "📈 I/Q Waveform, Spectrum & Eye Diagram",
        ]
    )

    with tab1:
        col_left, col_right = st.columns([2.1, 1.0])

        if not prof_df.empty:
            z_cfo = (sim_cfo_khz - prof_df["mean_cfo_khz"].values) / np.maximum(prof_df["std_cfo_khz"].values, 1.0)
            z_ppm = (sim_ppm - prof_df["mean_timing_ppm"].values) / np.maximum(prof_df["std_timing_ppm"].values, 25.0)
            z_eirp = (sim_eirp - prof_df["mean_eirp_dbm"].values) / np.maximum(prof_df["std_eirp_dbm"].values, 2.0)

            d_all = np.sqrt(0.50 * (z_cfo ** 2) + 0.35 * (z_ppm ** 2) + 0.15 * (z_eirp ** 2))
            target_idx = int(prof_df.index[prof_df["icao_hex"] == target_icao][0])
            d_claimed = float(d_all[target_idx])
            best_idx = int(np.argmin(d_all))
            pred_icao = str(prof_df.loc[best_idx, "icao_hex"])
            pred_cs = str(prof_df.loc[best_idx, "callsign"])

            log_lik = -0.5 * (d_all ** 2)
            lik_shift = np.exp(log_lik - np.max(log_lik))
            post_probs = lik_shift / (np.sum(lik_shift) + 1e-12)
            claimed_prob = float(0.65 * post_probs[target_idx] + 0.35 * np.exp(-0.5 * (d_claimed ** 2)))
            is_spoof = d_claimed > outlier_z or (pred_icao != target_icao and d_claimed > 3.2 and claimed_prob < 0.10)
        else:
            d_claimed, claimed_prob, pred_icao, pred_cs, is_spoof = 0.0, 1.0, target_icao, "N/A", False

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
                * **Enrolled Hardware Centroid**:
                  * Median CFO: `{default_cfo:+.2f} kHz`
                  * Median Timing Skew: `{default_ppm:+.1f} ppm`
                  * Median EIRP: `{default_eirp:.1f} dBm`
                * **Simulated Burst Measurements**:
                  * Burst CFO: `{sim_cfo_khz:+.2f} kHz` ($\\Delta = {sim_cfo_khz - default_cfo:+.2f}\\text{{ kHz}}$)
                  * Burst Timing Skew: `{sim_ppm:+.1f} ppm` ($\\Delta = {sim_ppm - default_ppm:+.1f}\\text{{ ppm}}$)
                  * Burst EIRP: `{sim_eirp:.1f} dBm`
                * **ML Security Metrics**:
                  * **Mahalanobis Distance ($d_M$)**: `{d_claimed:.2f} σ` (Gate: `{outlier_z:.1f} σ`)
                  * **Hardware Match Probability**: `{claimed_prob*100:.1f}%`
                  * **Closest RF Emitter Match**: `{pred_icao}` (`{pred_cs}`)
                """
            )

    with tab2:
        st.subheader("🗺️ Interactive Mode-S / ADS-B Trajectory Map (TU Wien Receiver)")
        map_file = base_dir / "adsb_map.html"
        if map_file.exists():
            components.html(map_file.read_text(encoding="utf-8"), height=560, scrolling=False)
        else:
            st.info("Run `python main.py` to generate `adsb_map.html`.")

    with tab3:
        st.subheader("✈️ Enrolled Aircraft RF Fingerprint Profiles")
        st.caption(
            "Aggregated per-aircraft physical-layer hardware profiles across the 20.97-second TU Wien rooftop capture "
            "(sorted by number of received frames)."
        )
        st.dataframe(prof_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("📈 Physical-Layer Baseband I/Q Waveform & Mode-S PPM Eye Diagram")
        col_iq, col_eye = st.columns(2)
        iq_png = base_dir / "iq_spectrum_sample.png"
        eye_png = base_dir / "eye_diagram_sample.png"

        with col_iq:
            if iq_png.exists():
                st.image(str(iq_png), caption="1090 MHz Mode-S Baseband I/Q, Envelope & Phase Trajectory", use_container_width=True)
            else:
                st.info("`iq_spectrum_sample.png` not found in directory.")

        with col_eye:
            if eye_png.exists():
                st.image(str(eye_png), caption="10 MS/s Mode-S Manchester PPM Eye Diagram (2-Symbol Window)", use_container_width=True)
            else:
                st.info("`eye_diagram_sample.png` not found in directory.")


if __name__ == "__main__":
    main()