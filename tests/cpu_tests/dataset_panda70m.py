import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module

class TestStringMethods(unittest.TestCase):
    @requires_lustre()
    def test_dataloader_panda70m(self):
        test_make_supervised_data_module(dataset_name="panda70m", batch_size=32, num_workers=32)


if __name__ == "__main__":
    unittest.main()
