from transformers import pipeline
import os

class NERPredictor:
    def __init__(self, model_path="models/ner"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        self.pipe = pipeline("ner", model=model_path)
        print("NER model loaded successfully.")

    def predict(self, text: str) -> dict:
        predictions = self.pipe(text)
        
        # The model gives predictions for parts of words (tokens). 
        # We need to group them back together.
        grouped_predictions = {}
        for pred in predictions:
            entity = pred['entity']
            word = pred['word']
            
            # Remove the B- or I- prefix from the entity label
            if entity.startswith("B-") or entity.startswith("I-"):
                clean_entity = entity[2:]
            else:
                clean_entity = entity

            if word.startswith("##"):
                # Append sub-word to the last word
                if clean_entity in grouped_predictions:
                    grouped_predictions[clean_entity] += word[2:]
            else:
                # Start a new word or append a full word
                if clean_entity in grouped_predictions:
                    grouped_predictions[clean_entity] += ' ' + word
                else:
                    grouped_predictions[clean_entity] = word

        return grouped_predictions