import os
import sys

# Add the root directory to the Python path to allow for `src` imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.ocr.vision_ocr import ocr_image

def run_batch_ocr(input_dir: str, output_dir: str):
    """
    Runs OCR on all images in an input directory and saves the results.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    
    print(f"Found {len(image_files)} images to process in '{input_dir}'.")

    for i, filename in enumerate(image_files):
        image_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing image {i+1}/{len(image_files)}: {filename}")
        try:
            ocr_image(image_path, output_path)
        except Exception as e:
            print(f"  -> Failed to process {filename}. Error: {e}")

if __name__ == "__main__":
    run_batch_ocr(
        # This is the line that has been updated to match your folder structure
        input_dir="data/raw_images/image", 
        output_dir="data/ocr_jsons"
    )