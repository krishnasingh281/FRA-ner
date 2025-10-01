import os
import json

def extract_text_for_labeling(json_dir: str, output_dir: str):
    """
    Extracts the 'text' field from each OCR JSON file for annotation.
    """
    if not os.path.isdir(json_dir):
        print(f"Error: JSON directory not found at {json_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files to process.")

    for filename in json_files:
        json_path = os.path.join(json_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data.get("text", ""))
            
    print(f"Successfully created {len(json_files)} .txt files in '{output_dir}'.")

if __name__ == "__main__":
    extract_text_for_labeling(
        json_dir="data/ocr_jsons", 
        output_dir="data/for_annotation"
    )