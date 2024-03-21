import os
import os.path as osp
import sys
import unittest

import torch
from llava.unit_test_utils import requires_gpu, requires_lustre


class TestDatasetLoading(unittest.TestCase):
    def test_print(self):
        print("hello world")

    @requires_lustre()
    def test_sam(self):
        from llava.data.simple_vila_webdataset import VILAWebDataset

        data_path = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/sam-reformat"
        dst = VILAWebDataset(
            data_path=osp.abspath(data_path),
        )
        for idx, data in enumerate(dst):
            print(idx, data.keys())
        print("SAM loading finish")

    @requires_lustre()
    def test_coyo_25m(self):
        from llava.data.simple_vila_webdataset import VILAWebDataset

        data_path = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila"
        dst = VILAWebDataset(
            data_path=osp.abspath(data_path),
        )
        for idx, data in enumerate(dst):
            print(idx, data.keys())
        print("Coyo-25M loading finish")


if __name__ == "__main__":
    unittest.main()
