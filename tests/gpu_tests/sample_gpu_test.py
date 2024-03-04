import unittest

from llava.unit_test_utils import requires_lustre, requires_gpu

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
    
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    @requires_gpu()
    def test_gpu_funcs(self):
        import torch
        a = torch.randn(3).cuda()
        b = torch.randn(3).cuda()
        print(a + b)
    
    
if __name__ == '__main__':
    unittest.main()
