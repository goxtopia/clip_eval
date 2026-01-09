
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import cv2
import numpy as np
import sys

# Mocking modules that might not be present or require GPU/API keys
sys.modules['ultralytics'] = MagicMock()
sys.modules['openai'] = MagicMock()

from src.autolabel import AutoLabeler

class TestAutoLabeler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_json = os.path.join(self.test_dir, "output.json")

        # Create a dummy image
        self.img_path = os.path.join(self.test_dir, "test.jpg")
        # Create a 2000x2000 black image
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        cv2.imwrite(self.img_path, img)

        self.labeler = AutoLabeler(self.test_dir, self.output_json, "http://fake", "fakekey")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_resolution_check(self):
        # We created a 2000x2000 image, should be High
        tag = self.labeler._get_resolution_tag(self.img_path)
        self.assertEqual(tag, "High")

        # Resize to small
        small_path = os.path.join(self.test_dir, "small.jpg")
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.imwrite(small_path, img)

        tag_small = self.labeler._get_resolution_tag(small_path)
        self.assertEqual(tag_small, "Low")

    @patch("src.autolabel.cv2.imencode")
    def test_encode_image_resizing(self, mock_imencode):
        # We want to verify that _encode_image resizes if > 1024
        # We can't easily check the internal state without mocking cv2.resize

        with patch("src.autolabel.cv2.resize") as mock_resize:
             # Just return original img to avoid errors
            mock_resize.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)
            mock_imencode.return_value = (True, b"fakebuffer")

            self.labeler._encode_image(self.img_path)

            # Should have called resize
            self.assertTrue(mock_resize.called)

            # Check args: 2000 -> 1024. Scale = 1024/2000 = 0.512
            # new_w = 1024, new_h = 1024
            args, _ = mock_resize.call_args
            self.assertEqual(args[1], (1024, 1024))

if __name__ == "__main__":
    unittest.main()
