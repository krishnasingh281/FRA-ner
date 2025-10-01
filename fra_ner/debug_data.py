import json

ANNOTATIONS_PATH = "data/for_annotation/labeled_data.json"

def preprocess_label_studio_data(data):
    """This is the exact same function from the training script."""
    processed_data = []
    # Label Studio exports one big JSON object, which is a list of tasks
    for task in data:
        # We only care about tasks that were actually annotated
        if not task.get('annotations') or not task['annotations'][0].get('result'):
            continue
            
        text = task['data']['text']
        labels = []
        
        # Collect all labeled spans
        for annotation in task['annotations'][0]['result']:
            span = annotation['value']
            labels.append({
                'start': span['start'],
                'end': span['end'],
                'label': span['labels'][0]
            })
            
        processed_data.append({"text": text, "labels": labels})
    return processed_data

print("--- Starting Data Debug ---")
try:
    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
        ls_data = json.load(f)
    
    print(f"Successfully loaded '{ANNOTATIONS_PATH}'. Total tasks in file: {len(ls_data)}")
    
    processed_data = preprocess_label_studio_data(ls_data)
    
    print(f"\nFunction found {len(processed_data)} documents with annotations.")
    
    if not processed_data:
        print("\nERROR: The preprocessing function could not find any annotations.")
        print("This means the format of the JSON file is different than expected.")
    else:
        print("\nSUCCESS: The function is working correctly.")
        print("First processed item found:", processed_data[0])

except FileNotFoundError:
    print(f"ERROR: Could not find the file at '{ANNOTATIONS_PATH}'. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\n--- End of Data Debug ---")