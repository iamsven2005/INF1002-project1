import os
import pandas as pd
from sklearn.model_selection import train_test_split
from packaging import version

import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased" 
OUTPUT_DIR = "./food_sentiment_model"

def score_to_label(score: int) -> int:
    # 0=neg, 1=neu, 2=pos
    if score <= 2:
        return 0
    if score == 3:
        return 1
    return 2

def load_and_prepare(csv_path: str, text_col: str = "Text", score_col: str = "Score"):
    df = pd.read_csv(csv_path)

    # Basic cleaning
    df = df.dropna(subset=[text_col, score_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]

    # Map labels
    df["label"] = df[score_col].astype(int).apply(score_to_label)

    # Optional: reduce training size for quick iteration
    # df = df.sample(n=min(len(df), 100000), random_state=42)

    train_df, test_df = train_test_split(
        df[[text_col, "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def train(csv_path: str):
    train_df, test_df = load_and_prepare(csv_path)

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["Text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["Text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["Text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        fp16=True,  # set True if you have a compatible GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save model in sharded safetensors (GitHub-friendly)
    trainer.model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,      # produces .safetensors
        max_shard_size="100MB",       # adjust: "50MB", "100MB", "200MB"
    )

    # Save tokenizer files
    tokenizer.save_pretrained(OUTPUT_DIR)

    return {"model_dir": OUTPUT_DIR}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to reviews CSV")
    args = p.parse_args()
    print(train(args.csv))
