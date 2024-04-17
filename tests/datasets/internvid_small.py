import os
import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module

class DatasetTestMethods(unittest.TestCase):
    @requires_lustre()
    def test_internvid_test(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="internvid_test", batch_size=2, num_workers=workers, max_samples=50)
        
    @requires_lustre()
    def test_internvid_1300K(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="internvid_1300K", batch_size=2, num_workers=workers, max_samples=50)
        
    @requires_lustre()
    def test_internvid_10M(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="internvid_10M", batch_size=2, num_workers=workers, max_samples=50)


if __name__ == "__main__":
    unittest.main()
