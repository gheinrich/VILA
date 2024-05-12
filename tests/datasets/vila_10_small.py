import os
import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module

class DatasetTestMethods(unittest.TestCase):
    @requires_lustre()
    def test_ccs_recaptioned(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="ccs_recaptioned", batch_size=2, num_workers=workers, max_samples=50)
        
    @requires_lustre()
    def test_ccs_recaptioned_test(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="ccs_recaptioned_test", batch_size=2, num_workers=workers, max_samples=50)
        
    @requires_lustre()
    def test_vflan(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="vflan", batch_size=2, num_workers=workers, max_samples=50)
        
    @requires_lustre()
    def test_coyo_25m(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="coyo_25m", batch_size=2, num_workers=workers, max_samples=50)
        
    @requires_lustre()
    def test_coyo_25m(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        test_make_supervised_data_module(dataset_name="mmc4core", batch_size=2, num_workers=workers, max_samples=50)


if __name__ == "__main__":
    unittest.main()
