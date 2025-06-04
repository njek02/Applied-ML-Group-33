import unittest
import numpy as np
from whale_call_project.preprocessing.wave_to_image import wave_to_spec, spec_to_image

class TestWaveToImage(unittest.TestCase):

    def test_output_shape(self):
        y = np.random.randn(16000).astype(np.float32)
        sr = 16000
        spec = wave_to_spec(y, sr)
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[0], 64)

    def test_rgb_output_shape(self):
        spec = np.random.rand(64, 100) * 80 - 40
        img = spec_to_image(spec, rgb_output=True)
        self.assertEqual(img.shape, (32, 32, 3))
        self.assertTrue(np.all((img >= 0.0) & (img <= 1.0)))

    def test_grayscale_output_shape(self):
        spec = np.random.rand(64, 100) * 80 - 40
        print(spec)
        img = spec_to_image(spec)
        self.assertEqual(img.shape, (1, 32, 32))
        self.assertTrue(np.all((img >= 0.0) & (img <= 1.0)))

if __name__ == "__main__":
    unittest.main()
