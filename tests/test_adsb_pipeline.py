"""
Comprehensive Unit & Integration Test Suite for the 1090 MHz ADS-B DSP, Mode-S Decoder,
and Leak-Free RF Fingerprinting ML Pipeline (`tests/test_adsb_pipeline.py`).
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adsb_rf.assumptions import (
    DEFAULT_ASSUMPTIONS,
    AnomalyThresholds,
    Assumptions,
    ReceiverConfig,
    get_assumptions_table_rows,
)
from adsb_rf.database import _sanitize_source_label
from adsb_rf.decoder import (
    ADSB_altitude,
    ADSB_callsign,
    ADSB_position,
    ADSB_velocity,
    NL,
    OnlineADSBDecoder,
    two_message_position,
)
from adsb_rf.dsp import (
    build_blind_energy_mask_10mhz,
    build_ideal_10mhz_mask,
    calculate_amplitude,
    calculate_freq_shift_and_phase_noise,
    compute_eirp_dbm,
    compute_single_frame_doppler,
    decode_ppm,
    estimate_timing_offset,
    extract_waveform_transient_features,
    is_valid,
    lldistkm,
    recover_cfo_unwrap_free,
)
from adsb_rf.evaluation import (
    build_aircraft_profiles_from_df,
    evaluate_subset_metrics,
    score_simulated_burst_with_residuals,
)
from adsb_rf.features import (
    RF_FEATURE_NAMES,
    _recover_missing_cfo_from_iq,
    extract_12_rf_features_from_frame,
)
from adsb_rf.models import fit_fold_ensemble, predict_fold_ensemble
from adsb_rf.stream_reader import resample_data
from adsb_rf.synthesizer import (
    _build_df17_callsign_bits,
    _build_df17_position_bits,
    _build_df17_velocity_bits,
)


class TestADSBDSPCore(unittest.TestCase):
    def test_crc24_validation_detects_bit_flips(self):
        poly = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=int)
        bits_88 = np.zeros(88, dtype=int)
        bits_88[0:5] = [1, 0, 0, 0, 1]  # DF=17
        reg = np.concatenate([bits_88, np.zeros(24, dtype=int)])
        for i in range(88):
            if reg[i] == 1:
                reg[i : i + 25] ^= poly
        valid_frame = np.concatenate([bits_88, reg[88:112]])
        self.assertTrue(is_valid(valid_frame))

        corrupted = valid_frame.copy()
        corrupted[21] ^= 1
        self.assertFalse(is_valid(corrupted))

    def test_eirp_and_doppler_physics(self):
        cfg = ReceiverConfig()
        # At d = 100 km (100,000 m) and f = 1090 MHz, FSPL = 20*log10(4*pi*100000 / 0.275039) = 133.196 dB.
        # A standard 250W (54 dBm) Mode-S transponder has Rx power ~ -79.2 dBm; at -79.2 dBm Rx power, EIRP = 54.0 dBm.
        eirp_nominal = compute_eirp_dbm(-79.196, 100000.0, cfg.carrier_freq_hz)
        self.assertAlmostEqual(eirp_nominal, 54.0, places=1)

        # At -65.0 dBm Rx power and 100 km, EIRP = -65.0 + 133.196 = 68.196 dBm (within rooftop 20..70 dBm window)
        eirp_high = compute_eirp_dbm(-65.0, 100000.0, cfg.carrier_freq_hz)
        self.assertAlmostEqual(eirp_high, 68.196, places=2)
        self.assertTrue(DEFAULT_ASSUMPTIONS.min_eirp_dbm < eirp_high < DEFAULT_ASSUMPTIONS.max_eirp_dbm)

        # Invalid distance or NaN returns NaN
        self.assertTrue(np.isnan(compute_eirp_dbm(-70.0, 0.0, cfg.carrier_freq_hz)))

        doppler = compute_single_frame_doppler(48.30, 16.50, 35000.0, 450.0, 225.0, cfg)
        self.assertFalse(np.isnan(doppler))
        self.assertTrue(abs(doppler) < 1500.0)
        # Aircraft flying Southwest (225 deg) from Northeast of Vienna (48.30, 16.50) is approaching -> positive Doppler
        self.assertGreater(doppler, 500.0)

    def test_cpr_latitude_zones_standard(self):
        self.assertEqual(NL(0.0), 59)
        self.assertTrue(35 <= NL(48.196674) <= 45)
        self.assertEqual(NL(87.5), 1)
        self.assertEqual(NL(-88.0), 1)

    def test_ideal_10mhz_mask_and_ppm_roundtrip(self):
        bits = _build_df17_callsign_bits(0x4401A2, "AUA101")
        mask = build_ideal_10mhz_mask(bits)
        self.assertEqual(len(mask), 1200)
        decoded = decode_ppm(mask.astype(float), sample_rate=10e6)
        np.testing.assert_array_equal(decoded[0], bits)
        self.assertTrue(is_valid(decoded[0]))

    def test_unwrap_free_cfo_recovery_and_12_features(self):
        bits = _build_df17_position_bits(0x4401A2, 31000.0, 48.245, 16.480, 0)
        mask = build_ideal_10mhz_mask(bits).astype(float)
        t = np.arange(1200) / 10e6
        # Test both moderate (-106 kHz) and high (+620 kHz, past +/- 500 kHz 1us wrapping limit!) CFO
        for true_cfo_hz in (-106447.0, 620000.0):
            iq_res = mask * np.exp(1j * 2.0 * np.pi * true_cfo_hz * t)
            rec_cfo, _ = _recover_missing_cfo_from_iq({"IQ_res": iq_res, "Bits": bits})
            self.assertLess(abs(rec_cfo - true_cfo_hz), 1500.0)
            cfo_dsp, _ = calculate_freq_shift_and_phase_noise(iq_res, bits)
            self.assertFalse(np.isnan(cfo_dsp))
            self.assertLess(abs(cfo_dsp - true_cfo_hz), 1500.0)

        vec12 = extract_12_rf_features_from_frame(
            {"IQ_res": iq_res, "Bits": bits, "CFO_Raw_Hz": rec_cfo, "Power": -55.0}
        )
        self.assertEqual(vec12.shape, (12,))
        self.assertEqual(len(RF_FEATURE_NAMES), 12)

    def test_waveform_transient_features_pa_droop_and_snr(self):
        bits = _build_df17_callsign_bits(0x3C6541, "DLH402")
        mask = build_ideal_10mhz_mask(bits).astype(float)
        t = np.arange(1200) / 10e6
        # Inject linear amplitude droop across the 120 us burst (from 1.0 at start to 0.8 at end -> ~1.94 dB sag)
        sag_profile = np.linspace(1.0, 0.80, 1200)
        iq_sagged = (mask * sag_profile + 0.01) * np.exp(1j * 2.0 * np.pi * 15000.0 * t)
        pa_droop_db, crest, curv, snr_db, pre_ratio = extract_waveform_transient_features(iq_sagged)
        self.assertGreater(pa_droop_db, 1.0)
        self.assertGreater(snr_db, 25.0)
        self.assertGreaterEqual(crest, 1.0)

    def test_timing_offset_and_blind_energy_mask(self):
        bits = _build_df17_callsign_bits(0x4B1809, "SWR219")
        mask = build_ideal_10mhz_mask(bits).astype(float)
        blind_mask, conf_slots = build_blind_energy_mask_10mhz(mask + 0.01)
        self.assertEqual(len(blind_mask), 1200)
        self.assertEqual(np.mean(conf_slots), 1.0)
        t_ppm, lag_ns = estimate_timing_offset(mask, bits, corr_triplet=(0.8, 1.0, 0.8))
        self.assertAlmostEqual(lag_ns, 0.0, places=5)
        self.assertLess(abs(t_ppm), 5.0)

    def test_polyphase_resampler_preserves_envelope(self):
        # 1280 samples at 12.8 MS/s (100 us) -> 1000 samples at 10.0 MS/s
        t_raw = np.arange(1280) / 12.8e6
        sig_raw = np.exp(1j * 2.0 * np.pi * 25000.0 * t_raw)
        res = resample_data(sig_raw, new_fs=10e6, old_fs=12.8e6)
        self.assertEqual(len(res["Z"]), 1000)

    def test_decoder_callsign_altitude_velocity_and_cpr(self):
        cs_frame = _build_df17_callsign_bits(0x4401A2, "AUA101")
        self.assertEqual(ADSB_callsign(cs_frame[37:88]), "AUA101")

        pos_even = _build_df17_position_bits(0x4401A2, 31000.0, 48.2450, 16.4800, 0)
        pos_odd = _build_df17_position_bits(0x4401A2, 31000.0, 48.2455, 16.4805, 1)
        self.assertAlmostEqual(ADSB_altitude(pos_even[40:52]), 31000.0, places=1)

        loc_pos = ADSB_position(pos_even[52:88], 48.196674, 16.370998)
        self.assertAlmostEqual(loc_pos[0], 48.2450, places=3)
        self.assertAlmostEqual(loc_pos[1], 16.4800, places=3)

        glob_pos = two_message_position(
            {"Bits": pos_even, "Timestamp": 1.0},
            {"Bits": pos_odd, "Timestamp": 1.5},
        )
        self.assertTrue(glob_pos["cpr_zone_valid"])
        self.assertAlmostEqual(glob_pos["global_lat"], 48.2455, places=2)

        vel_frame = _build_df17_velocity_bits(0x4401A2, 300, 400, -640)
        spd, hdg, vr = ADSB_velocity(vel_frame[37:88])
        self.assertAlmostEqual(spd, 500.0, places=1)
        self.assertAlmostEqual(vr, -640.0, places=1)

    def test_leak_free_fold_ensemble_and_privacy_sanitization(self):
        # Verify path sanitization strips Windows & Unix directories
        self.assertEqual(
            _sanitize_source_label(r"C:\Users\eriks\OneDrive\Desktop\receivedSignals.npy"),
            "receivedSignals.npy",
        )
        self.assertEqual(
            _sanitize_source_label("/working_dir/c_2aa6628f/user_uploaded/101000.mat"),
            "101000.mat",
        )

        # Verify leak-free fold ensemble fitting & prediction
        rng = np.random.default_rng(7)
        X_synth = np.vstack([
            rng.normal(loc=-20000.0, scale=400.0, size=(15, 12)),
            rng.normal(loc=+25000.0, scale=400.0, size=(15, 12)),
        ])
        y_synth = np.array([0] * 15 + [1] * 15, dtype=np.int64)
        ts_synth = np.arange(30, dtype=np.float64)
        metrics = evaluate_subset_metrics(X_synth, y_synth, ts_synth, n_classes=2)
        self.assertEqual(metrics["cv_top1"], 1.0)
        self.assertEqual(metrics["oot_top1"], 1.0)

        rows = get_assumptions_table_rows()
        self.assertGreaterEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
