import os
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import classification_report
import logging

# Suppress verbose logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# --- 1. Configuration ---
DATA_FILE = "data/for_annotation/train.conll"
MODEL_CHECKPOINT = "distilbert-base-uncased"
MODEL_OUTPUT_PATH = "models/ner"

# --- 2. Load Data ---
print("Loading data from CoNLL file...")
dataset = load_dataset("text", data_files={"train": DATA_FILE})

def parse_conll(example):
    lines = example['text'].strip().split('\n')
    tokens = []
    ner_tags = []
    for line in lines:
        if line:
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                ner_tags.append(parts[-1])
    return {'tokens': tokens, 'ner_tags': ner_tags}

dataset = dataset.map(parse_conll, remove_columns=['text'])

# --- 3. Prepare Labels and Tokenize ---
label_names = sorted(list(set(tag for tags_list in dataset["train"]["ner_tags"] for tag in tags_list)))
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for i, label in enumerate(label_names)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_str = label[word_idx]
                label_ids.append(label2id.get(label_str, -1))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["tokens", "ner_tags"])

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["macro avg"]["precision"],
        "recall": results["macro avg"]["recall"],
        "f1": results["macro avg"]["f1-score"],
    }

# --- 4. Train the Model ---
train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.2)
final_dataset = DatasetDict({
    'train': train_test_split['train'],
    'eval': train_test_split['test']
})

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=len(label_names), id2label=id2label, label2id=label2id
)

# --- THIS IS THE CORRECTED SECTION ---
# Using older, more compatible arguments for evaluation
args = TrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=10,
    do_eval=True, # Use this to enable evaluation
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting model training with evaluation...")
trainer.train()
print("Training complete.")

trainer.save_model(MODEL_OUTPUT_PATH)
print(f"Model saved to {MODEL_OUTPUT_PATH}")