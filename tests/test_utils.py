
import unittest
from src.utils import normalize_label

class TestUtils(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label(" Hello "), "hello")
        self.assertEqual(normalize_label("ABC"), "abc")
        self.assertEqual(normalize_label("  Mixed Case  "), "mixed case")
        self.assertEqual(normalize_label(""), "")
        self.assertEqual(normalize_label(None), "")

if __name__ == "__main__":
    unittest.main()
