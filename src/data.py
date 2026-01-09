import os
import json
import hashlib
import random
from typing import List, Tuple, Dict, Set, Optional
from .utils import normalize_label
from .label_mapping import LabelMapper

class DatasetItem:
    def __init__(self, image_path: str, txt_path: str, 
                 raw_labels: List[str], 
                 gt_class_set: Set[str], 
                 representative_text: str,
                 md5: str = "",
                 attributes: Optional[Dict] = None):
        self.image_path = image_path
        self.txt_path = txt_path
        self.raw_labels = raw_labels
        self.gt_class_set = gt_class_set  # Expanded GT labels (normalized)
        self.representative_text = representative_text
        self.md5 = md5
        self.attributes = attributes or {}

class DataLoader:
    def __init__(self, data_dir: str, mapper: LabelMapper, seed: int = 42, filter_json_path: str = None, text_json_path: str = None):
        self.data_dir = data_dir
        self.mapper = mapper
        self.rng = random.Random(seed)
        self.filter_json_path = filter_json_path
        self.text_json_path = text_json_path
        self.items: List[DatasetItem] = []
        self.filter_data = {}
        self.text_attributes = {}

        if self.filter_json_path and os.path.exists(self.filter_json_path):
            try:
                with open(self.filter_json_path, 'r', encoding='utf-8') as f:
                    self.filter_data = json.load(f)
            except Exception as e:
                print(f"[Error] Failed to load filter JSON: {e}")

        if self.text_json_path and os.path.exists(self.text_json_path):
            try:
                with open(self.text_json_path, 'r', encoding='utf-8') as f:
                    self.text_attributes = json.load(f)
            except Exception as e:
                print(f"[Error] Failed to load text attributes JSON: {e}")

    def _calculate_md5(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load(self) -> List[DatasetItem]:
        pairs = self._find_images_and_labels(self.data_dir)
        if not pairs:
            print(f"[Error] No jpg/txt pairs found in {self.data_dir}")
            return []

        print(f"[Info] Found {len(pairs)} image/text pairs.")
        
        items = []
        for img_path, txt_path in pairs:
            raw_lines, valid_lines = self._read_txt(txt_path)
            
            if not valid_lines:
                # Handle empty case
                items.append(DatasetItem(img_path, txt_path, raw_lines, set(), ""))
                continue
            
            # Construct GT set (normalized + expanded)
            base_gt = {normalize_label(l) for l in valid_lines}
            expanded_gt = self.mapper.expand_label_set(base_gt)
            
            # Sample representative text
            rep_text = self.rng.choice(valid_lines)
            
            # Calculate MD5
            md5 = self._calculate_md5(img_path)

            # Get attributes from filter data (Image attributes)
            # Key could be image name or md5, we check both. Prioritize MD5.
            img_name = os.path.basename(img_path)
            attributes = self.filter_data.get(md5) or self.filter_data.get(img_name) or {}
            
            # Get text attributes
            text_attrs = self.text_attributes.get(rep_text, {})
            
            # Merge attributes (Text attributes overwrite Image attributes if conflict, but keys should be distinct)
            attributes.update(text_attrs)

            items.append(DatasetItem(img_path, txt_path, valid_lines, expanded_gt, rep_text, md5, attributes))
            
        self.items = items
        return items

    def _find_images_and_labels(self, data_dir: str) -> List[Tuple[str, str]]:
        pairs = []
        if not os.path.exists(data_dir):
            print(f"[Error] Directory not found: {data_dir}")
            return []
            
        for fname in os.listdir(data_dir):
            if fname.lower().endswith(".jpg"):
                img_path = os.path.join(data_dir, fname)
                base = os.path.splitext(fname)[0]
                txt_path = os.path.join(data_dir, base + ".txt")
                if os.path.exists(txt_path):
                    pairs.append((img_path, txt_path))
                else:
                    # Optional: print warning or skip
                    pass
        pairs.sort()
        return pairs

    def _read_txt(self, path: str) -> Tuple[List[str], List[str]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            valid = [ln for ln in lines if ln]
            return lines, valid
        except Exception as e:
            print(f"[Error] Failed to read {path}: {e}")
            return [], []

def get_unique_texts(items: List[DatasetItem]) -> List[str]:
    """Extracts unique representative texts from dataset items."""
    seen = set()
    unique = []
    for item in items:
        t = item.representative_text
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique
