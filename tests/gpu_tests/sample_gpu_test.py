import unittest

from llava.unit_test_utils import requires_lustre, requires_gpu


class TestStringMethods(unittest.TestCase):
    @unittest.expectedFailure
    def test_expected_failure(self):
        a = []
        print(a[5])

    @requires_gpu()
    def test_gpu_funcs(self):
        import torch

        a = torch.randn(3).cuda()
        b = torch.randn(3).cuda()
        print(a + b)

    @requires_lustre()
    def test_lustre_access(self):
        from llava.data.simple_vila_webdataset import VILAWebDataset

        train_dataset = VILAWebDataset(
            data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
        )
        print(train_dataset[0])

    @requires_gpu()
    @requires_lustre()
    def test_both(self):
        from llava.data.simple_vila_webdataset import VILAWebDataset

        train_dataset = VILAWebDataset(
            data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
        )
        print(train_dataset[0])
        import torch

        a = torch.randn(3).cuda()
        b = torch.randn(3).cuda()
        print(a + b)


if __name__ == "__main__":
    unittest.main()
