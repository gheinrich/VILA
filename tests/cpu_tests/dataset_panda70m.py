import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module

class TestStringMethods(unittest.TestCase):
    @requires_lustre()
    def test_dataloader_panda70m(self):
        test_make_supervised_data_module(dataset_name="panda70m")


if __name__ == "__main__":
    unittest.main()
