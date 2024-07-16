import unittest

import torch

from llava.model.multimodal_projector import base_projector


class TestDownsampleProjector(unittest.TestCase):
    def setUp(self):
        self.projector = base_projector.DownSampleBlock()

    def test_forward(self):
        x = torch.zeros((1, 9, 5))
        y = self.projector(x)
        assert y.shape == (1, 4, 20)


if __name__ == "__main__":
    unittest.main()
