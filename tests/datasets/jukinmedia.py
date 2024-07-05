import os
import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module

class TestStringMethods(unittest.TestCase):
    @requires_lustre()
    def test_dataloader_panda70m(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="jukinmedia", batch_size=workers * 2, num_workers=workers)


if __name__ == "__main__":
    unittest.main()