import os
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    # Replace this dataset with your specific Vision Deepfake dataset
    print("[INFO] Loading image dataset...")
    try:
        # Example dataset placeholder 
        dataset = load_dataset("dima806/deepfake_faces") # Example Binary image dataset
    except Exception as e:
        print(f"[WARNING] Could not load demo dataset. Error: {e}")
        print("[WARNING] Please modify line 25 to use your actual Deepfake Image dataset.")
        return

    # Determine labels depending on your dataset mapping
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    # Use the pre-trained model listed in README.md!
    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    print(f"[INFO] Loading image processor: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Basic image augmentation / transforms
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def preprocess_function(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    print("[INFO] Preprocessing dataset...")
    try:
        # NOTE: Selecting a subset for demonstration/speed. Remove `.select()` for full training.
        small_train = dataset["train"].shuffle(seed=42).select(range(500))
        small_eval = dataset["test"].shuffle(seed=42).select(range(100))
        
        encoded_train = small_train.with_transform(preprocess_function)
        encoded_eval = small_eval.with_transform(preprocess_function)
    except Exception:
        print("[WARNING] Dataset structure didn't match expectations. Adjust preprocessing accordingly.")
        return

    print(f"[INFO] Loading fine-tune base model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(
        model_name, 
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    )

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "image_model"))
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print("[INFO] Target Hardware: CUDA (GPU) detected.")
        batch_size = 16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Target Hardware: MPS (Apple Silicon) detected.")
        batch_size = 16
    else:
        print("[WARNING] Target Hardware: CPU detected! Training may be very slow.")
        batch_size = 4

    training_args = TrainingArguments(
        output_dir="./tmp_image_trainer",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_eval,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting image training phase...")
    trainer.train()

    print(f"\n[INFO] Training complete. Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    print("[INFO] 🚀 Successfully saved! The TruthLens backend will now use your locally trained IMAGE model automatically!")

if __name__ == "__main__":
    main()
