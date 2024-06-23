import os
import os.path as osp
import sys
import unittest

import torch
from torch.utils.data import DataLoader

from llava.unit_test_utils import requires_gpu, requires_lustre


class TestDatasetLoading(unittest.TestCase):
    @requires_lustre()
    def test_sam(self):
        from llava.data.simple_vila_webdataset import VILAWebDataset

        data_path = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/sam-reformat"
        dst = VILAWebDataset(
            data_path=osp.abspath(data_path),
        )
        dl = DataLoader(dst, batch_size=16, num_workers=16, collate_fn=VILAWebDataset.custom_collate)
        for idx, data in enumerate(dl):
            print(idx, data.keys())
            if idx > 100:
                break
        print("SAM loading finish")

    @requires_lustre()
    def test_coyo_25m(self):
        from llava.data.simple_vila_webdataset import VILAWebDataset

        data_path = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coyo-25m-vila"
        dst = VILAWebDataset(
            data_path=osp.abspath(data_path),
        )
        dl = DataLoader(dst, batch_size=16, num_workers=16, collate_fn=VILAWebDataset.custom_collate)
        for idx, data in enumerate(dl):
            print(idx, data.keys())
            if idx > 100:
                break
        print("Finish loading 100 examples from COYO25M")

if __name__ == "__main__":
    unittest.main()