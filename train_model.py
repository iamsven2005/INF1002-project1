# Standard library imports
import os

# Data processing
import pandas as pd
from sklearn.model_selection import train_test_split

# Version handling (not strictly needed here but useful for compatibility checks)
from packaging import version

# HuggingFace / evaluation
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Configuration
# -----------------------------

# Base model to fine-tune
MODEL_NAME = "distilbert-base-uncased"

# Where the trained model + tokenizer will be saved
OUTPUT_DIR = "./food_sentiment_model"


# -----------------------------
# Label conversion
# -----------------------------

def score_to_label(score: int) -> int:
    """
    Convert a numeric review score into sentiment labels:
        0 → negative
        1 → neutral
        2 → positive

    Example mapping:
        1,2 → negative
        3   → neutral
        4,5 → positive
    """
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2


# -----------------------------
# Load & preprocess dataset
# -----------------------------

def load_and_prepare(csv_path: str, text_col: str = "Text", score_col: str = "Score"):
    """
    Loads a CSV file and prepares training / testing datasets.

    Steps:
    1. Load CSV
    2. Remove missing values
    3. Clean text
    4. Convert numeric scores to sentiment labels
    5. Split into train/test sets
    """

    # Validate CSV path exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV into pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    # Validate required columns exist
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {list(df.columns)}")

    initial_len = len(df)

    # Drop rows with missing text or score
    df = df.dropna(subset=[text_col, score_col])

    # Ensure text is string and strip whitespace
    df[text_col] = df[text_col].astype(str).str.strip()

    # Remove empty text rows
    df = df[df[text_col].str.len() > 0]

    # Validate we have data after cleaning
    if len(df) == 0:
        raise ValueError(f"No valid data after cleaning (started with {initial_len} rows)")

    # Convert numeric scores into sentiment labels with error handling
    try:
        df["label"] = df[score_col].astype(int).apply(score_to_label)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert scores to integers: {e}")

    # Validate class distribution for stratified split
    min_class_count = df["label"].value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            f"Insufficient data for stratified split. "
            f"Label distribution: {dict(df['label'].value_counts())}. "
            f"Each class needs at least 2 samples."
        )

    # Optional: reduce dataset size for fast experiments
    # df = df.sample(n=min(len(df), 100000), random_state=42)

    # Split into training and testing sets (80/20 split)
    try:
        train_df, test_df = train_test_split(
            df[[text_col, "label"]],
            test_size=0.2,
            random_state=42,
            stratify=df["label"]  # preserve class distribution
        )
    except ValueError as e:
        raise ValueError(f"Failed to split dataset: {e}")

    print(f"✓ Loaded {len(df)} samples (train: {len(train_df)}, test: {len(test_df)})")
    print(f"  Label distribution: {dict(train_df['label'].value_counts())}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------
# Training pipeline
# -----------------------------

def train(csv_path: str):
    """
    Full training pipeline:
    1. Load and clean data
    2. Tokenize text
    3. Load pretrained DistilBERT
    4. Fine-tune on sentiment classification
    5. Save trained model + tokenizer
    """

    # Prepare dataset
    try:
        train_df, test_df = load_and_prepare(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Data preparation failed: {e}")
        raise

    # Validate datasets are not empty
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Training or test dataset is empty")

    # Convert pandas DataFrames into HuggingFace Datasets
    try:
        train_ds = Dataset.from_pandas(train_df)
        test_ds = Dataset.from_pandas(test_df)
    except Exception as e:
        raise ValueError(f"Failed to convert to HuggingFace datasets: {e}")

    # Load tokenizer for the base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for '{MODEL_NAME}': {e}")

    # Tokenization function
    def tokenize(batch):
        """
        Tokenizes a batch of texts:
        - Truncates long sequences
        - Pads dynamically later via DataCollator
        """
        return tokenizer(batch["Text"], truncation=True, max_length=256)

    # Apply tokenization to datasets
    try:
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["Text"])
        test_ds = test_ds.map(tokenize, batched=True, remove_columns=["Text"])
    except Exception as e:
        raise ValueError(f"Tokenization failed: {e}")

    # Handles dynamic padding in each batch
    try:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    except Exception as e:
        raise ValueError(f"Failed to create data collator: {e}")

    # Load the pretrained model and configure for 3-class classification
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3
        )
    except Exception as e:
        raise ValueError(f"Failed to load model '{MODEL_NAME}': {e}")

    # Load evaluation metrics
    try:
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
    except Exception as e:
        raise ValueError(f"Failed to load evaluation metrics: {e}")

    def compute_metrics(eval_pred):
        """
        Compute metrics during evaluation.
        Returns:
        - Accuracy
        - Macro F1 score (balanced across classes)
        """
        try:
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            return {
                "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
                "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            }
        except Exception as e:
            print(f"Warning: Metric computation failed: {e}")
            return {"accuracy": 0.0, "f1_macro": 0.0}

    # Ensure output directory exists before training
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Failed to create output directory '{OUTPUT_DIR}': {e}")

    # Training configuration
    try:
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",              # Evaluate once per epoch
            save_strategy="no",                 # Do not save checkpoints during training
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_steps=50,
            fp16=True,                          # Enable if you have a GPU that supports FP16
            report_to="none",                   # Disable external logging (WandB, etc.)
        )
    except Exception as e:
        raise ValueError(f"Failed to configure training arguments: {e}")

    # Trainer object handles training loop, evaluation, and logging
    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize trainer: {e}")

    # Start training
    try:
        print("\n🚀 Starting training...")
        trainer.train()
        print("✓ Training completed successfully")
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Save trained model
    try:
        # - safe_serialization=True → saves as .safetensors
        # - max_shard_size keeps files small enough for GitHub
        trainer.model.save_pretrained(
            OUTPUT_DIR,
            safe_serialization=True,
            max_shard_size="100MB",
        )
        print(f"✓ Model saved to {OUTPUT_DIR}")
    except Exception as e:
        raise ValueError(f"Failed to save model: {e}")

    # Save tokenizer files
    try:
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✓ Tokenizer saved to {OUTPUT_DIR}")
    except Exception as e:
        raise ValueError(f"Failed to save tokenizer: {e}")

    return {"model_dir": OUTPUT_DIR}


# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    """
    Run from command line:
        python train.py --csv reviews.csv

    This trains the model and writes:
        ./food_sentiment_model/
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Train sentiment classifier on food reviews")
    p.add_argument("--csv", required=True, help="Path to reviews CSV file")
    args = p.parse_args()

    try:
        result = train(args.csv)
        print(f"\n✓ Training successful: {result}")
        sys.exit(0)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n✗ Training failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)
