import unittest


class TestSimpleCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_should_fail(self):
        d = {}
        print(d["123"])


if __name__ == "__main__":
    unittest.main()
