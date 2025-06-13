import unittest
import numpy as np
from whale_call_project.preprocessing.normalization import (
    peak_normalization,
    rms_normalize,
    spectrogram_normalization
)

class TestNormalization(unittest.TestCase):
    """
    Test suite for normalization functions.
    """

    def test_peak_normalization(self):
        """
        Test that peak normalization scales the maximum absolute value to 1.0.
        """
        audio = np.random.randn(100)
        norm = peak_normalization(audio)
        self.assertTrue(np.all(np.abs(norm) <= 1.0))
        self.assertAlmostEqual(np.max(np.abs(norm)), 1.0)

    def test_peak_normalization_zero_input(self):
        """
        Test that peak normalization of a zero input returns all zeros.
        """
        audio = np.zeros(100)
        norm = peak_normalization(audio)
        self.assertTrue(np.all(norm == 0))

    def test_rms_normalize(self):
        """
        Test that RMS normalization scales the input to have unit RMS.
        """
        audio = np.random.randn(100)
        norm = rms_normalize(audio)
        rms = np.sqrt(np.mean(norm ** 2))
        self.assertAlmostEqual(rms, 1.0)

    def test_rms_normalize_zero_input(self):
        """
        Test that RMS normalization of a zero input returns all zeros.
        """
        audio = np.zeros(100)
        norm = rms_normalize(audio)
        self.assertTrue(np.all(norm == 0))

    def test_spectrogram_normalization_range(self):
        """
        Test that spectrogram normalization outputs values in the [0, 1] range
        and preserves the input shape.
        """
        spec = np.random.rand(64, 244) * 80 - 40
        norm = spectrogram_normalization(spec)
        self.assertTrue(np.all((norm >= 0.0) & (norm <= 1.0)))
        self.assertEqual(norm.shape, spec.shape)

if __name__ == "__main__":
    unittest.main()