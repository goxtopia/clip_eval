
import unittest
import os
import shutil
import tempfile
import json
from src.data import DataLoader, DatasetItem
from src.label_mapping import LabelMapper

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mapping_file = os.path.join(self.test_dir, "mapping.json")
        self.filter_file = os.path.join(self.test_dir, "filters.json")

        # Create mapping
        with open(self.mapping_file, "w") as f:
            json.dump({"equivalents": [], "implications": {}}, f)

        self.mapper = LabelMapper(self.mapping_file)

        # Create dummy data
        # Img 1
        with open(os.path.join(self.test_dir, "img1.jpg"), "wb") as f:
            f.write(b"fakeimage1")
        with open(os.path.join(self.test_dir, "img1.txt"), "w") as f:
            f.write("cat\n")

        # Img 2 (no text)
        with open(os.path.join(self.test_dir, "img2.jpg"), "wb") as f:
            f.write(b"fakeimage2")
        # Missing txt file test -> handled by _find_images_and_labels (will skip)

        # Img 3
        with open(os.path.join(self.test_dir, "img3.jpg"), "wb") as f:
            f.write(b"fakeimage3")
        with open(os.path.join(self.test_dir, "img3.txt"), "w") as f:
            f.write("dog\nanimal\n")

        # Create filter file
        # We need MD5 of img1. "fakeimage1" md5
        import hashlib
        md5_1 = hashlib.md5(b"fakeimage1").hexdigest()

        filters = {
            md5_1: {"Time": "Day"}
        }
        with open(self.filter_file, "w") as f:
            json.dump(filters, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_data(self):
        loader = DataLoader(self.test_dir, self.mapper, filter_json_path=self.filter_file)
        items = loader.load()

        # Should find img1 and img3. img2 has no txt so it might be skipped if logic requires it.
        # Check logic: _find_images_and_labels appends if txt exists.
        # So expected 2 items.
        self.assertEqual(len(items), 2)

        # Check Item 1
        item1 = next(i for i in items if "img1.jpg" in i.image_path)
        self.assertIn("cat", item1.gt_class_set)
        self.assertEqual(item1.attributes.get("Time"), "Day")

        # Check Item 3
        item3 = next(i for i in items if "img3.jpg" in i.image_path)
        self.assertIn("dog", item3.gt_class_set)
        self.assertIn("animal", item3.gt_class_set)
        self.assertEqual(item3.attributes, {})

if __name__ == "__main__":
    unittest.main()
