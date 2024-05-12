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
        workers = 2
        if osp.isdir("/lustre"):
            test_make_supervised_data_module(dataset_name="coyo_25m", batch_size=2, num_workers=workers, max_samples=50)
        elif osp.isdir("/mnt"):
            test_make_supervised_data_module(dataset_name="osmo_coyo_25m", batch_size=2, num_workers=workers, max_samples=50)
        else:
            raise Exception("No lustre or mnt path found")

if __name__ == "__main__":
    unittest.main()
