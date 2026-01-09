import os
import json
import argparse
import hashlib
from typing import Dict, Any, List, Set
from openai import OpenAI

class TextAutoLabeler:
    def __init__(self, dataset_dir: str, output_json: str, api_url: str, api_key: str, model_name: str = "gpt-4o"):
        self.dataset_dir = dataset_dir
        self.output_json = output_json
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

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
            except Exception as e:
                print(f"Failed to load existing JSON, starting fresh. Error: {e}")

    def _get_length_tag(self, text: str) -> str:
        # Simple word count split
        if not text:
            return "Empty"
        
        words = text.split()
        count = len(words)
        
        if count < 3:
            return "Short"
        elif count < 6:
            return "Mid"
        elif count < 9:
            return "Long"
        else:
            return "Very Long"

    def _get_llm_attributes(self, text: str) -> Dict[str, Any]:
        prompt = """
        Analyze the following text and provide attributes in JSON format:
        1. "Text Subject": A list containing one or more of ["Person", "Car", "Animal", "Scene"]. If none match, use ["Other"].
        2. "Text Spelling": "No Error" or "Has Error". (Check for spelling mistakes).

        Text: "{text}"

        Output ONLY the JSON object.
        """
        
        formatted_prompt = prompt.replace("{text}", text.replace('"', '\\"'))

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                max_tokens=200,
                temperature=0.0
            )

            content = response.choices[0].message.content
            # Clean up potential markdown blocks
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error in LLM processing for text '{text[:20]}...': {e}")
            return {"Text Subject": ["Unknown"], "Text Spelling": "Unknown"}

    def run(self):
        # 1. Gather all unique text lines from the dataset
        unique_texts: Set[str] = set()
        
        files = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(".txt")]
        print(f"Found {len(files)} text files.")

        for f_name in files:
            path = os.path.join(self.dataset_dir, f_name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            unique_texts.add(line)
            except Exception as e:
                print(f"Error reading {f_name}: {e}")

        total = len(unique_texts)
        print(f"Found {total} unique text lines.")

        # 2. Process unique texts
        sorted_texts = sorted(list(unique_texts))
        
        changes_count = 0
        for i, text in enumerate(sorted_texts):
            if text in self.data:
                continue
            
            print(f"Processing {i+1}/{total}: {text[:50]}...")
            
            # Length Tag (Deterministic)
            len_tag = self._get_length_tag(text)
            
            # LLM Tags
            llm_attrs = self._get_llm_attributes(text)
            
            self.data[text] = {
                "Text Length": len_tag,
                **llm_attrs
            }
            
            changes_count += 1
            
            # Save periodically
            if changes_count % 10 == 0:
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
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    args = parser.parse_args()

    labeler = TextAutoLabeler(args.dataset, args.output, args.api_url, args.api_key, args.model)
    labeler.run()
