

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
DATA_PATH = "dataset/Tweets.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "airline_sentiment"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "saved_bert_model"
MAX_LEN = 64
EPOCHS = 4
BATCH_SIZE = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42


class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.lower()

    plt.figure(figsize=(6, 4))
    df[LABEL_COLUMN].value_counts().plot(kind="bar", color=["#4caf50", "#f44336", "#9e9e9e"])
    plt.title("Class Distribution of Dataset")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.close()


    labels_sorted = sorted(df[LABEL_COLUMN].unique())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df[LABEL_COLUMN].map(label2id)

    print("Label mapping:", label2id)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[TEXT_COLUMN].tolist(),
        df["label_id"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_id"].tolist(),
    )

    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LEN
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=MAX_LEN
    )

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    # ----------------------------------------------------------------------
    # 5. Model
    # ----------------------------------------------------------------------
    print("Loading model:", MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # ----------------------------------------------------------------------
    # 6. Train
    # ----------------------------------------------------------------------
    print("Starting training...")
    train_result = trainer.train()

    # ----------------------------------------------------------------------
    # 7. Loss / accuracy graphs from log history
    # ----------------------------------------------------------------------
    history = trainer.state.log_history
    train_loss = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
    eval_loss = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]
    eval_acc = [(h["epoch"], h["eval_accuracy"]) for h in history if "eval_accuracy" in h]

    if train_loss and eval_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(*zip(*train_loss), label="Train Loss", marker="o")
        plt.plot(*zip(*eval_loss), label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/loss_curve.png")
        plt.close()

    if eval_acc:
        plt.figure(figsize=(6, 4))
        plt.plot(*zip(*eval_acc), label="Validation Accuracy", marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy per Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/accuracy_curve.png")
        plt.close()

    # ----------------------------------------------------------------------
    # 8. Final evaluation + confusion matrix + classification report
    # ----------------------------------------------------------------------
    print("Evaluating on validation set...")
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)

    acc = accuracy_score(val_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels, preds, average="weighted", zero_division=0
    )
    report = classification_report(
        val_labels, preds, target_names=labels_sorted, zero_division=0
    )

    print("\n=== FINAL EVALUATION ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(report)

    with open("results/evaluation_report.txt", "w") as f:
        f.write("=== FINAL EVALUATION ===\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(val_labels, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels_sorted))
    plt.xticks(tick_marks, labels_sorted, rotation=45)
    plt.yticks(tick_marks, labels_sorted)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("Saving model and tokenizer to:", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print("Done. Model saved. Graphs saved in results/ folder.")


if __name__ == "__main__":
    main()
