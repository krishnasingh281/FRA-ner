import io
import json
from google.cloud import vision

def ocr_image(image_path: str, output_path: str):
    """
    Performs OCR on a local image file and saves the detailed response to a JSON file.
    """
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Use document_text_detection for dense text like forms
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    # Transform the complex response into a simpler, usable format
    words_list = []
    full_text = response.full_text_annotation.text

    # The first annotation is the full text, subsequent ones are words.
    for annotation in response.text_annotations[1:]:
        words_list.append({
            "text": annotation.description,
            "bbox": [(v.x, v.y) for v in annotation.bounding_poly.vertices]
        })

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump({"text": full_text, "words": words_list}, f, indent=2)

    print(f"OCR output saved to {output_path}")
    return output_path