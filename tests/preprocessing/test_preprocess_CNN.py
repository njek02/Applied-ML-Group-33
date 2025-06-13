import unittest
import numpy as np
from unittest.mock import patch
from whale_call_project.preprocessing.preprocess_CNN import preprocess_sample


class TestPreproccessCNN(unittest.TestCase):
    """
    Test suite for the preprocess_sample function used for CNN input preparation.
    """

    @patch('whale_call_project.preprocessing.preprocess_CNN.librosa.load')
    def test_grayscale_input(self, mock_load):
        """
        Test that preprocessing with rgb_output=False returns a (1, 32, 32) grayscale image
        with values in the [0, 1] range.
        """
        audio = np.random.randn(4000)
        mock_load.return_value = (audio, 2000)

        output = preprocess_sample("mock_path.aiff", rgb_output=False)
        self.assertEqual(output.shape, (1, 32, 32))
        self.assertTrue(np.all((output >= 0.0) & (output <= 1.0)))

    @patch('whale_call_project.preprocessing.wave_to_image.librosa.load')
    def test_rgb_input(self, mock_load):
        """
        Test that preprocessing with rgb_output=True returns a (32, 32, 3) RGB image
        with values in the [0, 1] range.
        """
        mock_audio = np.random.randn(4000)
        mock_load.return_value = (mock_audio, 2000)

        output = preprocess_sample("fake_path.wav", rgb_output=True)
        self.assertEqual(output.shape, (64, 64, 3))
        self.assertTrue(np.all((output >= 0.0) & (output <= 1.0)))

if __name__ == "__main__":
    unittest.main()
