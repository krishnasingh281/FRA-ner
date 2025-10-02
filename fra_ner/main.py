import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
import json

# Import the functions you've already built
from src.ocr.vision_ocr import ocr_image
from src.preprocess.normalize import group_words_into_lines, apply_common_fixes
from src.parser.rule_parser import extract_fields_by_rules
from src.ner.infer_ner import NERPredictor

# Initialize the FastAPI application
app = FastAPI(title="FRA Data Extraction API")

# Load your trained NER model into memory when the server starts
# ner_predictor = NERPredictor() # Temporarily disabled for memory test
ner_predictor = None
print("INFO: NER model loading is temporarily disabled for this deployment.")

@app.post("/extract/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    This endpoint receives an image, runs it through the full pipeline,
    and returns the extracted data as JSON.
    """
    # Create a temporary folder for uploads
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a temporary, unique path for the uploaded file
    temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")

    try:
        # Save the uploaded file to the temporary path
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- BEGIN PIPELINE ---
        
        # 1. Run OCR
        ocr_output_path = temp_image_path + ".json"
        ocr_image(temp_image_path, ocr_output_path)

        # 2. Load and Preprocess
        with open(ocr_output_path, "r", encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        full_text = ocr_data.get("text", "")
        words = ocr_data.get("words", [])
        lines = group_words_into_lines(words)
        
        # 3. Rule-Based Extraction
        rule_based_results = extract_fields_by_rules(lines)

        # 4. NER Model Fallback
        ner_results = {}
        if ner_predictor:
            ner_results = ner_predictor.predict(full_text)
        
        # 5. Merge Results
        final_results = rule_based_results.copy()
        for key, value in ner_results.items():
            if key not in final_results:
                final_results[key] = value
        
        for key, value in final_results.items():
            final_results[key] = apply_common_fixes(value)

        # --- END PIPELINE ---

        return final_results

    except Exception as e:
        # If anything goes wrong, return an error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    finally:
        # Clean up the temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(ocr_output_path):
            os.remove(ocr_output_path)