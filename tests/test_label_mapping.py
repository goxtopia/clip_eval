
import unittest
import json
import tempfile
import os
from src.label_mapping import LabelMapper

class TestLabelMapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary mapping file
        self.mapping_data = {
            "equivalents": [
                ["person", "people", "human"],
                ["bike", "bicycle"]
            ],
            "implications": {
                "man": ["person"],
                "woman": ["person"],
                "boy": ["man"],
                "child": ["person"]
            }
        }
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        json.dump(self.mapping_data, self.temp_file)
        self.temp_file.close()

        self.mapper = LabelMapper(self.temp_file.name)

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_equivalents(self):
        # Test basic expansion within equivalent group
        res = self.mapper.expand_label("person")
        self.assertIn("people", res)
        self.assertIn("human", res)
        self.assertIn("person", res)

        # Test unknown label
        res = self.mapper.expand_label("alien")
        self.assertEqual(res, {"alien"})

    def test_implications(self):
        # Test implication A -> B
        res = self.mapper.expand_label("man")
        self.assertIn("man", res)
        self.assertIn("person", res)
        # Should also include equivalents of person
        self.assertIn("people", res)

    def test_transitive_implication(self):
        # boy -> man -> person -> people
        res = self.mapper.expand_label("boy")
        self.assertIn("boy", res)
        self.assertIn("man", res)
        self.assertIn("person", res)
        self.assertIn("people", res)

    def test_expand_label_set(self):
        # Test expanding a set
        res = self.mapper.expand_label_set({"boy", "bike"})
        expected_subset = {"boy", "man", "person", "people", "bike", "bicycle"}
        self.assertTrue(expected_subset.issubset(res))

if __name__ == "__main__":
    unittest.main()
