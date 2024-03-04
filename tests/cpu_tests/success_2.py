import torch
import unittest

from llava.unit_test_utils import requires_gpu, requires_lustre


class TestInputPacking(unittest.TestCase):
    def test_print(self):
        print("hello world")
    
    @requires_lustre()
    @requires_gpu()
    def test_gpu(self):
        print("hello world")

    def test_should_fail(self):
        a = {}
        print(a["123"])
        

if __name__ == "__main__":
    unittest.main()