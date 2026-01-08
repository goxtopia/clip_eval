import os
import json
import hashlib
import cv2
import argparse
from typing import Dict, Any, List
import requests
import base64
from openai import OpenAI
from ultralytics import YOLO

class AutoLabeler:
    def __init__(self, dataset_dir: str, output_json: str, api_url: str, api_key: str, model_name: str = "gpt-4o"):
        self.dataset_dir = dataset_dir
        self.output_json = output_json
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

        # Initialize YOLO model
        self.yolo_model = YOLO("yolo11x.pt")  # Will download if not present

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )

        # Load existing data
        self.data = {}
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except:
                print("Failed to load existing JSON, starting fresh.")

    def _calculate_md5(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _encode_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return ""

        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1024:
            scale = 1024 / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def _get_person_size(self, image_path: str) -> str:
        try:
            results = self.yolo_model(image_path, verbose=False)
            img = cv2.imread(image_path)
            h, w, _ = img.shape
            img_area = h * w

            max_bbox_area = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    # Class 0 is 'person' in COCO
                    if cls == 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_bbox_area:
                            max_bbox_area = area

            if max_bbox_area == 0:
                return "None"

            ratio = max_bbox_area / img_area

            # Define thresholds (adjust as needed)
            if ratio < 0.05:
                return "Small"
            elif ratio < 0.3:
                return "Medium"
            else:
                return "Large"

        except Exception as e:
            print(f"Error in YOLO processing for {image_path}: {e}")
            return "Error"

    def _get_vlm_attributes(self, image_path: str) -> Dict[str, str]:
        base64_image = self._encode_image(image_path)

        prompt = """
        Analyze the image and provide the following attributes in JSON format:
        1. "time_of_day": "Day" or "Night"
        2. "is_blurry": "Yes" or "No"
        3. "resolution": "High" or "Low" (Consider High if it looks sharp and detailed, Low otherwise)

        Output ONLY the JSON object.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            content = response.choices[0].message.content
            # Clean up potential markdown blocks
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error in VLM processing for {image_path}: {e}")
            return {"time_of_day": "Unknown", "is_blurry": "Unknown", "resolution": "Unknown"}

    def run(self):
        images = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(images)
        print(f"Found {total} images.")

        for i, img_name in enumerate(images):
            img_path = os.path.join(self.dataset_dir, img_name)
            md5 = self._calculate_md5(img_path)

            if md5 in self.data:
                # print(f"Skipping {img_name} (already processed)")
                continue

            print(f"Processing {i+1}/{total}: {img_name}")

            person_size = self._get_person_size(img_path)
            vlm_attrs = self._get_vlm_attributes(img_path)

            self.data[md5] = {
                "filename": img_name,
                "person_size": person_size,
                **vlm_attrs
            }

            # Save periodically
            if (i + 1) % 10 == 0:
                self._save()

        self._save()
        print("Done!")

    def _save(self):
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default="gpt-4o", help="VLM model name")
    args = parser.parse_args()

    labeler = AutoLabeler(args.dataset, args.output, args.api_url, args.api_key, args.model)
    labeler.run()
