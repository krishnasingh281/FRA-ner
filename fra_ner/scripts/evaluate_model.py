import os
import sys

# Add the root directory to the Python path to allow for `src` imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.ner.infer_ner import NERPredictor

def load_ground_truth(file_path):
    """Loads entities from a CoNLL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    docs = content.strip().split('\n\n')
    ground_truth_docs = []
    
    for doc in docs:
        full_text = []
        entities = []
        current_entity_text = []
        current_entity_label = None

        for line in doc.split('\n'):
            if not line:
                continue
            token, label = line.split()
            full_text.append(token)

            if label.startswith('B-'):
                if current_entity_text:
                    entities.append({
                        "label": current_entity_label,
                        "text": " ".join(current_entity_text)
                    })
                current_entity_text = [token]
                current_entity_label = label[2:]
            elif label.startswith('I-') and current_entity_label == label[2:]:
                current_entity_text.append(token)
            else:
                if current_entity_text:
                    entities.append({
                        "label": current_entity_label,
                        "text": " ".join(current_entity_text)
                    })
                current_entity_text = []
                current_entity_label = None
        
        if current_entity_text:
            entities.append({
                "label": current_entity_label,
                "text": " ".join(current_entity_text)
            })

        ground_truth_docs.append({
            "text": " ".join(full_text),
            "entities": entities
        })
    return ground_truth_docs

def evaluate_model(test_data_path="data/for_annotation/test.conll"):
    # Load the ground truth data
    ground_truth = load_ground_truth(test_data_path)
    
    # Load the trained NER model
    predictor = NERPredictor()

    exact_matches = 0
    missed_entities = 0
    false_positives = 0
    total_true_entities = 0

    for doc in ground_truth:
        true_entities = { (e['label'], e['text']) for e in doc['entities'] }
        predicted_results = predictor.predict(doc['text'])
        predicted_entities = { (label, text) for label, text in predicted_results.items() }

        total_true_entities += len(true_entities)
        
        exact_matches += len(true_entities.intersection(predicted_entities))
        missed_entities += len(true_entities - predicted_entities)
        false_positives += len(predicted_entities - true_entities)

    # Calculate metrics
    precision = exact_matches / (exact_matches + false_positives) if (exact_matches + false_positives) > 0 else 0
    recall = exact_matches / total_true_entities if total_true_entities > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print the report
    print("\n" + "="*20)
    print("  ACCURACY METRICS")
    print("="*20)
    print(f"Exact Matches:    {exact_matches}/{total_true_entities} ({exact_matches/total_true_entities:.1%})")
    print(f"Missed Entities:  {missed_entities}/{total_true_entities} ({missed_entities/total_true_entities:.1%})")
    print(f"False Positives:  {false_positives}")
    print("-"*20)
    print("  STRICT METRICS")
    print("-"*20)
    print(f"Precision:        {precision:.3f} ({precision:.1%})")
    print(f"Recall:           {recall:.3f} ({recall:.1%})")
    print(f"F1 Score:         {f1_score:.3f} ({f1_score:.1%})")
    print("="*20)
    
    if f1_score > 0.9:
        print("OVERALL ASSESSMENT: EXCELLENT")
    elif f1_score > 0.8:
        print("OVERALL ASSESSMENT: GOOD")
    elif f1_score > 0.7:
        print("OVERALL ASSESSMENT: FAIR")
    else:
        print("OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
    print("="*20 + "\n")


if __name__ == "__main__":
    evaluate_model()