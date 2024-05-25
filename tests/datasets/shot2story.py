import os
import unittest
from llava.unit_test_utils import requires_gpu, requires_lustre, test_make_supervised_data_module

class TestStringMethods(unittest.TestCase):
    @requires_lustre()
    def test_dataloader_panda70m(self):
        workers = os.cpu_count() // 4
        print("cpu cores: ", os.cpu_count(), " workers: ", workers)
        test_make_supervised_data_module(dataset_name="shot2story_shotonly", batch_size=workers * 2, num_workers=workers, skip_before=180)


if __name__ == "__main__":
    unittest.main()
