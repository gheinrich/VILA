import os
import os.path as osp
import sys
import unittest

from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module


class TestDatasetLoading(unittest.TestCase):

    @requires_lustre()
    def test_coyo_25m_small(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        workers = 1
        test_make_supervised_data_module(dataset_name="internvid_10M_recap", 
            batch_size=1, num_workers=workers, max_samples=1)
        

if __name__ == "__main__":
    unittest.main()
