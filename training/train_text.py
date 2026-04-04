import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("[INFO] Loading dataset (GonzaloA/fake_news)...")
    # GonzaloA/fake_news has columns 'text', 'title', 'label'
    # labels: 0 for Fake, 1 for True. 
    # We explicitly map 0 -> FAKE, 1 -> REAL for our backend expectations.
    dataset = load_dataset("GonzaloA/fake_news")
    
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    model_name = "hamzab/roberta-fake-news-classification"
    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # Increased truncation max_length slightly to capture more article context
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    print("[INFO] Tokenizing dataset...")
    # NOTE: Using a subset of the data (2,000 samples) so that training completes in a reasonable time.
    # To train on the FULL dataset for maximum accuracy, remove the '.select(range(...))' constraint!
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    small_eval_dataset = dataset["validation"].shuffle(seed=42).select(range(500))

    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)

    print("[INFO] Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    )

    # Output dir setup matches the fallback architecture
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "text_model"))
    os.makedirs(output_dir, exist_ok=True)

    # Hardware detection and optimization
    if torch.cuda.is_available():
        print("[INFO] Target Hardware: CUDA (GPU) detected. Preparing optimal batch sizes.")
        batch_size = 16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Target Hardware: MPS (Apple Silicon) detected.")
        batch_size = 16
    else:
        print("[WARNING] Target Hardware: CPU detected! Training may be very slow.")
        batch_size = 4

    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training phase...")
    trainer.train()

    print(f"\n[INFO] Training complete. Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] 🚀 Successfully saved! TruthLens API will now instantly switch to using this model.")

if __name__ == "__main__":
    main()
