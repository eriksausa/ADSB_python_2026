"""
Visualization Module for ADS-B RF Fingerprinting & Anomaly Detection.
Includes:
1. `map_planes`: Interactive HTML map (via Folium if installed, or standalone Leaflet.js HTML)
   with normal vs. anomalous/spoofed aircraft trajectory markers.
2. `plot_rf_fingerprint_clusters`: 2D scatter plot of Carrier Frequency Offset (kHz) vs.
   Symbol Clock Timing Offset (ppm) by ICAO transponder with anomaly highlights and
   CRC-failed bursts attributed via Blind RF Fingerprinting.
3. `plot_iq_new`: Time-domain I/Q/Amplitude/Phase and single-sided FFT spectra.
4. `plot_eye_diagram`: Mode-S PPM eye diagram visualizer.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


def map_planes(
    messages: List[Dict[str, Any]],
    output_html: str | Path = "adsb_map.html",
    rx_lat: float = 48.196674,
    rx_lon: float = 16.370998,
) -> Path:
    """
    Generates an interactive HTML map of decoded aircraft positions, highlighting
    anomalous/spoofed frames in red and valid frames in blue/teal.
    """
    out_path = Path(output_html)
    pos_msgs = [
        m for m in messages
        if 8 < m.get("TC", 0) < 19
        and not np.isnan(m.get("Latitude", float("nan")))
        and not np.isnan(m.get("Longitude", float("nan")))
        and abs(m.get("Latitude", 0.0)) <= 90.0
        and abs(m.get("Longitude", 0.0)) <= 180.0
    ]

    if HAS_FOLIUM:
        fmap = folium.Map(
    location=[rx_lat, rx_lon], 
    zoom_start=9, 
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
)

        folium.Marker(
            location=[rx_lat, rx_lon],
            popup="<b>Antenna Location</b><br>TU Wien EI Building",
            icon=folium.Icon(color="green", icon="radio"),
        ).add_to(fmap)

        for msg in pos_msgs:
            icao_hex = msg.get("ICAO_Hex", f"{msg['ICAO']:06X}")
            lat = float(msg["Latitude"])
            lon = float(msg["Longitude"])
            alt = float(msg.get("Altitude", 0.0))
            pwr = float(msg.get("Power", 0.0))
            dist = float(msg.get("Distance_from_Rx", 0.0))
            cfo = float(msg.get("Frequency_shift", 0.0))
            t_ppm = float(msg.get("Timing_Offset_PPM", 0.0))
            is_anom = bool(msg.get("Is_Anomaly", False))
            reasons = msg.get("Anomaly_Reasons", "Normal")

            popup_html = (
                f"<b>ICAO:</b> {icao_hex} (Pred: {msg.get('Predicted_ICAO_Hex', icao_hex)})<br>"
                f"<b>Status:</b> {'ANOMALY: ' + reasons if is_anom else 'Valid'}<br>"
                f"<b>Altitude:</b> {alt:.0f} ft | <b>Distance:</b> {dist/1000.0:.1f} km<br>"
                f"<b>Rx Power:</b> {pwr:.1f} dBm | <b>CFO:</b> {cfo/1e3:.2f} kHz<br>"
                f"<b>Timing Offset:</b> {t_ppm:.2f} ppm<br>"
                f"<a href='https://www.planespotters.net/search?q={icao_hex}' target='_blank'>Planespotters</a>"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=7 if is_anom else 5,
                color="#d90429" if is_anom else "#0077b6",
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fmap)

        fmap.save(str(out_path))
        return out_path

    markers_data = []
    for msg in pos_msgs:
        icao_hex = msg.get("ICAO_Hex", f"{msg['ICAO']:06X}")
        markers_data.append(
            {
                "icao": icao_hex,
                "pred_icao": msg.get("Predicted_ICAO_Hex", icao_hex),
                "callsign": msg.get("Call_sign") or "N/A",
                "lat": float(msg["Latitude"]),
                "lon": float(msg["Longitude"]),
                "alt": float(msg.get("Altitude", 0.0)),
                "pwr": float(msg.get("Power", 0.0)),
                "dist_km": float(msg.get("Distance_from_Rx", 0.0)) / 1000.0,
                "cfo_khz": float(msg.get("Frequency_shift", 0.0)) / 1e3,
                "timing_ppm": float(msg.get("Timing_Offset_PPM", 0.0)),
                "is_anomaly": bool(msg.get("Is_Anomaly", False)),
                "reasons": str(msg.get("Anomaly_Reasons", "Normal")),
            }
        )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ADS-B RF Fingerprinting & Anomaly Detection Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; }}
    #map {{ height: 100vh; width: 100vw; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([{rx_lat}, {rx_lon}], 9);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    L.marker([{rx_lat}, {rx_lon}]).addTo(map)
      .bindPopup("<b>Receiver Antenna</b><br>TU Wien EI Building ({rx_lat:.4f}, {rx_lon:.4f})");

    const points = {json.dumps(markers_data)};
    points.forEach(p => {{
      const color = p.is_anomaly ? '#d90429' : '#0077b6';
      const radius = p.is_anomaly ? 8 : 5;
      const popup = `
        <b>ICAO:</b> ${{p.icao}} (${{p.callsign}})<br>
        <b>RF Predicted ICAO:</b> ${{p.pred_icao}}<br>
        <b>Status:</b> <span style="color:${{color}};font-weight:bold;">${{p.is_anomaly ? 'ANOMALY' : 'VALID'}}</span><br>
        <b>Details:</b> ${{p.reasons}}<br>
        <b>Altitude:</b> ${{p.alt.toFixed(0)}} ft | <b>Dist:</b> ${{p.dist_km.toFixed(1)}} km<br>
        <b>Rx Power:</b> ${{p.pwr.toFixed(1)}} dBm<br>
        <b>CFO:</b> ${{p.cfo_khz.toFixed(2)}} kHz | <b>Timing:</b> ${{p.timing_ppm.toFixed(2)}} ppm
      `;
      L.circleMarker([p.lat, p.lon], {{
        radius: radius,
        color: color,
        fillColor: color,
        fillOpacity: 0.85,
        weight: 2
      }}).addTo(map).bindPopup(popup);
    }});
  </script>
</body>
</html>"""
    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def plot_rf_fingerprint_clusters(
    df: pd.DataFrame,
    crc_failed_df: Optional[pd.DataFrame] = None,
    save_path: str | Path = "rf_fingerprint_clusters.png",
) -> Path:
    """
    Plots the 2D physical-layer RF Fingerprint feature space:
    Carrier Frequency Offset (kHz) vs. Symbol Clock Timing Offset (ppm),
    grouped by aircraft ICAO, highlighting both Spoofed/Anomalous frames and
    CRC-Failed bursts attributed via Blind RF Fingerprinting.
    """
    out_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=100)

    valid_df = df.dropna(subset=["freq_offset_hz", "timing_offset_ppm"]).copy()
    valid_df["cfo_khz"] = valid_df["freq_offset_hz"] / 1e3

    normal_df = valid_df[~valid_df["is_anomaly"].astype(bool)]
    anom_df = valid_df[valid_df["is_anomaly"].astype(bool)]

    palette = {"3C6541": "#0077b6", "4401A2": "#2a9d8f", "4B1809": "#8338ec"}
    fallback_colors = ["#0077b6", "#2a9d8f", "#8338ec", "#fb8500", "#3a86ff"]

    for idx, (icao_hex, grp) in enumerate(normal_df.groupby("icao_hex")):
        color = palette.get(str(icao_hex), fallback_colors[idx % len(fallback_colors)])
        cs = grp["callsign"].dropna().iloc[-1] if not grp["callsign"].dropna().empty else ""
        label_str = f"ICAO {icao_hex} ({cs}) - Valid CRC" if cs else f"ICAO {icao_hex} - Valid CRC"
        ax.scatter(
            grp["cfo_khz"],
            grp["timing_offset_ppm"],
            label=label_str,
            color=color,
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.7,
        )

    # Overlay CRC-failed bursts attributed purely via Blind RF Fingerprinting
    if crc_failed_df is not None and not crc_failed_df.empty:
        crc_df = crc_failed_df.dropna(subset=["freq_offset_hz", "timing_offset_ppm"]).copy()
        crc_df["cfo_khz"] = crc_df["freq_offset_hz"] / 1e3
        ax.scatter(
            crc_df["cfo_khz"],
            crc_df["timing_offset_ppm"],
            label="CRC-Failed Burst (Blind RF Attributed)",
            color="#ffb703",
            marker="^",
            s=125,
            edgecolors="#1d3557",
            linewidth=1.2,
            zorder=4,
        )
        for _, row in crc_df.iterrows():
            ax.annotate(
                f"CRC-Fail -> {row['predicted_icao_hex']} ({row['rf_confidence']*100:.0f}%)",
                (row["cfo_khz"], row["timing_offset_ppm"]),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=8,
                color="#023047",
                weight="bold",
            )

    if not anom_df.empty:
        ax.scatter(
            anom_df["cfo_khz"],
            anom_df["timing_offset_ppm"],
            label="Flagged Anomaly / Spoofed",
            color="#d90429",
            marker="X",
            s=130,
            edgecolors="black",
            linewidth=1.0,
            zorder=5,
        )
        for _, row in anom_df.iterrows():
            reason_short = str(row["anomaly_reasons"]).split("(")[0].strip()
            ax.annotate(
                f"{row['icao_hex']}: {reason_short}",
                (row["cfo_khz"], row["timing_offset_ppm"]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=8,
                color="#9b0000",
                weight="bold",
            )

    ax.set_title(
        "ADS-B Transponder RF Fingerprinting: Valid Frames, Spoofing & Blind CRC-Failed Attribution",
        fontsize=12.5,
        weight="bold",
    )
    ax.set_xlabel("Doppler-Compensated Carrier Frequency Offset (kHz)", fontsize=11)
    ax.set_ylabel("Symbol Clock Timing Offset (ppm)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_iq_new(
    messages: List[Dict[str, Any]],
    timestamp: Optional[float] = None,
    fs: float = 12.8e6,
    save_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Plots time-domain Inphase, Quadrature, Amplitude envelope, Phase, and single-sided FFT spectra.
    """
    if not messages:
        return None

    if timestamp is None:
        mes = messages[0]
    else:
        matching = [m for m in messages if abs(m["Timestamp"] - timestamp) < 1e-4]
        mes = matching[0] if matching else messages[0]

    iq_signal = mes.get("IQ") if mes.get("IQ") is not None and len(mes.get("IQ")) > 0 else mes["IQ_res"]
    inphase = np.real(iq_signal)
    quadrature = np.imag(iq_signal)
    amplitude = np.abs(iq_signal)
    phase = np.angle(iq_signal)

    fig, axs = plt.subplots(4, 1, figsize=(10, 7), sharex=True, dpi=140)
    axs[0].plot(inphase, "#0077b6"); axs[0].set_ylabel("V"); axs[0].set_title(f"Inphase (I) - ICAO {mes.get('ICAO_Hex')}")
    axs[1].plot(quadrature, "#2a9d8f"); axs[1].set_ylabel("V"); axs[1].set_title("Quadrature (Q)")
    axs[2].plot(amplitude, "#d90429"); axs[2].set_ylabel("V"); axs[2].set_title("Amplitude |I + jQ|")
    axs[3].plot(phase, "#8338ec"); axs[3].set_ylabel("rad"); axs[3].set_title("Phase"); axs[3].set_xlabel("Sample Index")
    plt.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    plt.close(fig)
    return None


def plot_eye_diagram(
    signal: np.ndarray,
    msps: float = 10.0,
    pulse_duration_us: float = 1.0,
    save_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Plots an eye diagram for a Mode-S PPM signal."""
    samples_per_symbol = int(round(pulse_duration_us * msps))
    window_size = samples_per_symbol * 2
    step_size = samples_per_symbol

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for i in range(0, len(signal) - window_size, step_size):
        chunk = np.abs(signal[i : i + window_size])
        ax.plot(chunk, color="#0077b6", alpha=0.08)

    ax.set_title(f"Mode-S PPM Eye Diagram ({msps} MS/s)")
    ax.set_xlabel("Sample Index in 2-Symbol Window")
    ax.set_ylabel("Envelope Magnitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    plt.close(fig)
    return None