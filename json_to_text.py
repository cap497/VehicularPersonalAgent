import json
import os
import re
from pathlib import Path

def clean_text(text: str) -> str:
    text = text.replace("-\n ", "").replace("-\n", "")  # remove hyphenated breaks
    text = text.replace("\n", " ")                      # remove newlines
    text = re.sub(r"\s+", " ", text)                    # collapse multiple spaces
    return text.strip()

def extract_text_blocks(data):
    """
    Traverse JSON. For each object:
      - If it has 'type': 'text', get 'value'.
      - Else, if it has 'text' (and no 'type'), get 'text'.
    """
    extracted = []

    def walk(item):
        if isinstance(item, dict):
            if "type" in item:
                if item["type"] == "text" and "value" in item and isinstance(item["value"], str):
                    extracted.append(clean_text(item["value"]))
            elif "text" in item and isinstance(item["text"], str):
                extracted.append(clean_text(item["text"]))
            # Recurse into all values
            for v in item.values():
                walk(v)
        elif isinstance(item, list):
            for elem in item:
                walk(elem)
        # Ignore raw strings, ints, etc.

    walk(data)
    return extracted

def json_to_txt(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_texts = extract_text_blocks(data)
    final_text = " ".join(extracted_texts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ Converted: {input_path} → {output_path}")

if __name__ == "__main__":
    input_dir = "json_inputs"
    output_dir = "knowledge_base"
    os.makedirs(output_dir, exist_ok=True)

    for json_file in Path(input_dir).glob("*.json"):
        out_file = os.path.join(output_dir, f"{json_file.stem}.txt")
        json_to_txt(str(json_file), out_file)
