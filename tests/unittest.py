import unittest
from src.topotoolbox.placeholder import example

class TestPlaceholder(unittest.TestCase):
    def test_example1(self):
        self.assertEqual(example("test"), "test example", "failed unittest")

    def test_example2(self):
        self.assertEqual(example(""), " example", "failed unittest")

if __name__ == '__main__':
    unittest.main()
