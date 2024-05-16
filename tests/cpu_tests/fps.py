import os
import os.path as osp
import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_fps_module

class TestStringMethods(unittest.TestCase):
    @requires_lustre()
    def test_dataloader_panda70m(self):
        print("cpu cores: ", os.cpu_count())
        workers = os.cpu_count() // 4
        
        workers = 2
        test_fps_module(
            dataset_name="sharegpt_video" if osp.isdir("/lustre") else "osmo_sharegpt_video", 
            batch_size=workers * 2, 
            num_workers=workers, 
            max_samples=20,
            num_video_frames=32,
            fps=2.0,
        )
        test_fps_module(
            dataset_name="shot2story_shotonly" if osp.isdir("/lustre") else "osmo_shot2story_shotonly", 
            batch_size=workers * 2, 
            num_workers=workers, 
            max_samples=20,
            num_video_frames=32,
            fps=2.0,
        )
        test_fps_module(
            dataset_name="internvid_10M" if osp.isdir("/lustre") else "osmo_internvid_10M",
            batch_size=workers * 2, 
            num_workers=workers, 
            max_samples=20,
            num_video_frames=48,
            fps=2.0,
        )
        test_fps_module(
            dataset_name="panda70m" if osp.isdir("/lustre") else "osmo_panda70m",
            batch_size=workers * 2, 
            num_workers=workers, 
            max_samples=20,
            num_video_frames=48,
            fps=2.0,
        )

if __name__ == "__main__":
    unittest.main()
