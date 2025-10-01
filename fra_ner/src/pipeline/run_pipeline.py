import json
import argparse
import os

from src.ocr.vision_ocr import ocr_image
from src.preprocess.normalize import group_words_into_lines, apply_common_fixes
from src.parser.rule_parser import extract_fields_by_rules
from src.ner.infer_ner import NERPredictor # <-- Import your new predictor

def run_pipeline(image_path: str):
    """
    Executes the full OCR and extraction pipeline using both rules and the NER model.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    ocr_output_path = f"data/ocr_jsons/{filename_base}.json"

    # 1. Run OCR
    ocr_image(image_path, ocr_output_path)

    # 2. Load and Preprocess
    with open(ocr_output_path, "r", encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    full_text = ocr_data.get("text", "")
    words = ocr_data.get("words", [])
    lines = group_words_into_lines(words)

    # 3. First Pass: Extract fields using rules
    rule_based_results = extract_fields_by_rules(lines)
    print("\n--- Rule-Based Results ---")
    print(json.dumps(rule_based_results, indent=2))

    # 4. Second Pass: Use the NER model as a fallback
    print("\nLoading NER model for fallback...")
    ner_predictor = NERPredictor()
    ner_results = ner_predictor.predict(full_text)
    print("\n--- NER Model Results ---")
    print(json.dumps(ner_results, indent=2))

    # 5. Merge the results
    # The rule-based results are usually more reliable, so we keep them first.
    final_results = rule_based_results.copy()
    for key, value in ner_results.items():
        if key not in final_results: # Only add if the rule-based parser missed it
            final_results[key] = value

    # Clean the final values
    for key, value in final_results.items():
        final_results[key] = apply_common_fixes(value)

    print("\n--- FINAL MERGED RESULTS ---")
    print(json.dumps(final_results, indent=2))
    print("--------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full OCR extraction pipeline.")
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the input image file."
    )
    args = parser.parse_args()
    
    run_pipeline(args.image_path)