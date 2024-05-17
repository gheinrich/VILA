import os
import os.path as osp
import unittest
import unittest.mock as unittest_mock
from llava.unit_test_utils import requires_gpu, requires_lustre, test_fps_module, test_sp_dataloader_module

class TestStringMethods(unittest.TestCase):

    @requires_lustre()
    def test_dataloader_panda70m(self):

        class MockSeqParallel:
            def __init__(self, sp_rank=1):
                self.sp_rank= sp_rank
        def mock_get_pg_manager():
            return MockSeqParallel()

        with unittest_mock.patch("llava.train.sequence_parallel.get_pg_manager", new=mock_get_pg_manager):
            print("cpu cores: ", os.cpu_count())
            workers = os.cpu_count() // 4
            
            workers = 2
            test_sp_dataloader_module(
                dataset_name="internvid_1300K" if osp.isdir("/lustre") else "osmo_internvid_1300K",
                batch_size=workers * 2, 
                num_workers=workers, 
                max_samples=20,
                num_video_frames=48,
                fps=2.0,
            )

if __name__ == "__main__":
    
    unittest.main()
