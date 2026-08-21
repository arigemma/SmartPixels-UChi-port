import unittest

import numpy as np

from dataloader import LAST_FRAME, N_TIME, N_X, N_Y, _adc_y_profile, get_transform


class ADCProfileTest(unittest.TestCase):
    def test_adc2a_threshold_codes_are_summed_by_row(self):
        values = np.zeros((N_TIME, N_Y, N_X), dtype=np.float32)
        values[LAST_FRAME, 0, :4] = [399.0, 400.0, 1600.0, 2400.0]
        profile = _adc_y_profile(values.reshape(-1), "adc2a")
        self.assertEqual(profile.shape, (N_Y,))
        self.assertEqual(profile[0], 0 + 1 + 2 + 3)
        self.assertTrue(np.all(profile[1:] == 0))

    def test_transform_keeps_y0_and_thirteen_rows(self):
        raw_x = np.zeros(N_TIME * N_Y * N_X, dtype=np.float32)
        raw_y = np.zeros(13, dtype=np.float32)
        features = get_transform("y-profile-adc2a", scale_dict=None)(raw_x, raw_y)
        self.assertEqual(tuple(features.shape), (14,))

    def test_unknown_adc_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown ADC mapping"):
            _adc_y_profile(np.zeros(N_TIME * N_Y * N_X), "adc9z")


if __name__ == "__main__":
    unittest.main()
