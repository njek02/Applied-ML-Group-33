import unittest
import torch
from whale_call_project.models.CNN import CNN

class TestCNN(unittest.TestCase):

    def test_forward_pass_output_shape(self):
        model = CNN()
        dummy_input = torch.randn(4, 1, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 2))

if __name__ == "__main__":
    unittest.main()
