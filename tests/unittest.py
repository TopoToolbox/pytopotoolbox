import unittest

class TestPlaceholder(unittest.TestCase):
    def test1(self):
        self.assertEqual("", "", "failed unittest")

if __name__ == '__main__':
    unittest.main()
