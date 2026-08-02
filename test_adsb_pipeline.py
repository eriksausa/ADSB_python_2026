"""
Unit test suite for the modular 1090 MHz ADS-B DSP, Mode-S Decoder, and RF Fingerprinting pipeline.
"""

import unittest
import numpy as np

from config import AnomalyThresholds, ReceiverConfig
from decoder import ADSB_altitude, ADSB_callsign, ADSB_position, NL, OnlineADSBDecoder
from dsp import build_ideal_10mhz_mask, compute_eirp_dbm, compute_single_frame_doppler, is_valid, lldistkm


class TestADSBDSPCore(unittest.TestCase):
    def test_crc24_validation_detects_bit_flips(self):
        # Construct a valid 112-bit Mode-S frame using generator polynomial G(x) = 0xFFF409
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
        eirp = compute_eirp_dbm(-65.0, 100000.0, cfg.carrier_freq_hz)
        self.assertTrue(45.0 < eirp < 65.0)

        doppler = compute_single_frame_doppler(48.30, 16.50, 35000.0, 450.0, 225.0, cfg)
        self.assertFalse(np.isnan(doppler))
        self.assertTrue(abs(doppler) < 1500.0)

    def test_cpr_latitude_zones_standard(self):
        self.assertEqual(NL(0.0), 59)
        self.assertTrue(35 <= NL(48.196674) <= 45)
        self.assertEqual(NL(87.5), 1)
        self.assertEqual(NL(-88.0), 1)

    def test_ideal_10mhz_mask_length(self):
        dummy_bits = np.zeros(112, dtype=int)
        mask = build_ideal_10mhz_mask(dummy_bits)
        self.assertEqual(len(mask), 1200)


if __name__ == "__main__":
    unittest.main()