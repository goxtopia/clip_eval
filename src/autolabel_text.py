import os
import json
import argparse
import hashlib
from typing import List, Dict, Any
from openai import OpenAI

class TextAutoLabeler:
    def __init__(self, dataset_dir: str, output_json: str, api_url: str, api_key: str, model_name: str = "gpt-4o"):
        self.dataset_dir = dataset_dir
        self.output_json = output_json
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(base_url=api_url, api_key=api_key)

        self.data = {}
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except:
                pass

    def _calculate_md5(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_length_tag(self, text: str) -> str:
        # Split by whitespace to count words
        words = text.strip().split()
        count = len(words)
        if count < 3: return "Short"
        elif count < 6: return "Mid"
        elif count < 9: return "Long"
        else: return "Very Long"

    def _get_llm_attributes(self, text: str) -> Dict[str, str]:
        prompt = f"""
        Analyze the following text: "{text}"

        Provide the following attributes in JSON format:
        1. "Subject": One of ["Person", "Car", "Animal", "Scene", "Other"]
        2. "Spelling Error": "Yes" or "No" (Check for typos or grammatical errors)

        Output ONLY the JSON object.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error LLM processing text '{text}': {e}")
            return {"Subject": "Unknown", "Spelling Error": "Unknown"}

    def run(self):
        # 1. Gather all unique texts
        unique_texts = set()

        if not os.path.exists(self.dataset_dir):
            print("Dataset directory not found.")
            return

        files = [f for f in os.listdir(self.dataset_dir) if f.endswith(".txt")]
        print(f"Found {len(files)} text files.")

        for fname in files:
            path = os.path.join(self.dataset_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            unique_texts.add(line)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        print(f"Found {len(unique_texts)} unique text lines.")

        # 2. Process
        count = 0
        for text in unique_texts:
            # key by text hash (or text itself? hash is safer for JSON keys)
            key = self._calculate_md5(text)

            if key in self.data:
                continue

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} new texts...")
                self._save()

            length_tag = self._get_length_tag(text)
            llm_attrs = self._get_llm_attributes(text)

            self.data[key] = {
                "text": text,
                "Length": length_tag,
                "Subject": llm_attrs.get("Subject", "Unknown"),
                "Spelling Error": llm_attrs.get("Spelling Error", "Unknown")
            }

        self._save()
        print("Text Auto-labeling Done!")

    def _save(self):
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    args = parser.parse_args()

    labeler = TextAutoLabeler(args.dataset, args.output, args.api_url, args.api_key, args.model)
    labeler.run()
