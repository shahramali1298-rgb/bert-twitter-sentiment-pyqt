"""
app.py
------
PyQt5 desktop GUI for BERT-based Twitter Sentiment Analysis.
(Styled version - nicer colors, fonts, and layout)

Features:
- Load Dataset button -> loads a CSV file (tweet, sentiment columns) into a table.
- Load Model button -> loads a saved_bert_model/ folder (model + tokenizer).
- Click any row in the table -> runs prediction using the loaded BERT model.
- Manual text box + Predict button -> predicts sentiment for typed text.
- Status bar shows whether model/dataset are loaded.

Run:
    python app.py
"""

import sys
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QTextEdit, QMessageBox, QGroupBox, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from transformers import BertTokenizerFast, BertForSequenceClassification


# ---------------------------------------------------------------------------
# COLORS / THEME - dark, vibrant, high-contrast
# ---------------------------------------------------------------------------
COLOR_BG = "#12101c"          # deep dark navy/purple background
COLOR_CARD = "#1c1a2b"        # card background
COLOR_CARD_ALT = "#221f35"    # alternating row color
COLOR_BORDER = "#312d4a"
COLOR_TEXT = "#f2f0fb"
COLOR_MUTED = "#a29ecb"

COLOR_PRIMARY = "#8b5cf6"     # vivid purple
COLOR_PRIMARY_2 = "#ec4899"   # pink (used in gradients)
COLOR_PRIMARY_DARK = "#7c3aed"

COLOR_POSITIVE = "#10d97a"    # bright green
COLOR_NEGATIVE = "#ff4d6d"    # bright red/pink
COLOR_NEUTRAL = "#7dd3fc"     # bright cyan

STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}

QGroupBox {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    margin-top: 14px;
    font-weight: 600;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {COLOR_TEXT};
}}

QPushButton {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLOR_PRIMARY}, stop:1 {COLOR_PRIMARY_2});
    color: white;
    border: none;
    border-radius: 9px;
    padding: 11px 20px;
    font-weight: 700;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLOR_PRIMARY_DARK}, stop:1 #d6408f);
}}
QPushButton:pressed {{
    background-color: {COLOR_PRIMARY_DARK};
}}

QTableWidget {{
    background-color: {COLOR_CARD};
    alternate-background-color: {COLOR_CARD_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
    gridline-color: {COLOR_BORDER};
    selection-background-color: {COLOR_PRIMARY};
    selection-color: white;
    color: {COLOR_TEXT};
}}
QHeaderView::section {{
    background-color: #2a2640;
    color: {COLOR_TEXT};
    padding: 9px;
    border: none;
    border-bottom: 2px solid {COLOR_PRIMARY};
    font-weight: 700;
}}
QTableWidget::item {{
    padding: 4px;
}}

QTextEdit {{
    background-color: #221f35;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 9px;
    padding: 8px;
}}

QLabel#statusLabel {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 9px;
    padding: 10px 14px;
    color: {COLOR_MUTED};
    font-weight: 600;
}}

QLabel#titleLabel {{
    font-size: 22px;
    font-weight: 800;
    color: {COLOR_TEXT};
}}

QLabel#subtitleLabel {{
    color: {COLOR_MUTED};
    font-size: 12px;
}}

QLabel#resultLabel {{
    font-size: 21px;
    font-weight: 800;
    padding: 18px;
    border-radius: 12px;
    background-color: #2a2640;
    color: {COLOR_MUTED};
}}

QLabel#confidenceLabel {{
    color: {COLOR_MUTED};
    font-size: 13px;
    font-weight: 600;
    padding-left: 4px;
}}

QScrollBar:vertical {{
    background: {COLOR_BG};
    width: 12px;
    border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_PRIMARY};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.dataframe = None

        self.setWindowTitle("Twitter Sentiment Analysis - BERT PyQt GUI")
        self.resize(1050, 600)
        self.setStyleSheet(STYLESHEET)
        self.init_ui()

    # ----------------------------------------------------------------
    # UI SETUP
    # ----------------------------------------------------------------
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # --- Header ---
        header_layout = QVBoxLayout()
        title = QLabel("Twitter Sentiment Analysis")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Load a trained BERT model, load a tweet dataset, click a tweet or type your own sentence.")
        subtitle.setObjectName("subtitleLabel")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)

        # --- Top bar: buttons + status ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        self.btn_load_dataset = QPushButton("📂  Load Dataset CSV")
        self.btn_load_dataset.clicked.connect(self.load_dataset)

        self.btn_load_model = QPushButton("🤖  Load BERT Model")
        self.btn_load_model.clicked.connect(self.load_model)

        self.status_label = QLabel("Model: Not loaded    |    Dataset: Not loaded")
        self.status_label.setObjectName("statusLabel")

        top_bar.addWidget(self.btn_load_dataset)
        top_bar.addWidget(self.btn_load_model)
        top_bar.addWidget(self.status_label, stretch=1)

        main_layout.addLayout(top_bar)

        # --- Middle: table (left) + prediction panel (right) ---
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(16)

        # Table of tweets
        table_box = QGroupBox("Loaded Tweets  (click a row to predict)")
        table_layout = QVBoxLayout()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Tweet Text", "Actual Label"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self.on_row_clicked)
        table_layout.addWidget(self.table)
        table_box.setLayout(table_layout)

        # Prediction panel
        pred_box = QGroupBox("Prediction Panel")
        pred_layout = QVBoxLayout()
        pred_layout.setSpacing(10)

        pred_layout.addWidget(QLabel("Selected / Typed Sentence:"))
        self.selected_text = QTextEdit()
        self.selected_text.setFixedHeight(90)
        pred_layout.addWidget(self.selected_text)

        self.btn_predict = QPushButton("🔮  Predict Sentiment")
        self.btn_predict.clicked.connect(self.predict_manual)
        pred_layout.addWidget(self.btn_predict)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"color: {COLOR_BORDER};")
        pred_layout.addWidget(divider)

        self.result_label = QLabel("Predicted Sentiment: —")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setObjectName("confidenceLabel")
        pred_layout.addWidget(self.confidence_label)

        pred_layout.addStretch()
        pred_box.setLayout(pred_layout)

        middle_layout.addWidget(table_box, stretch=2)
        middle_layout.addWidget(pred_box, stretch=1)

        main_layout.addLayout(middle_layout)
        self.setLayout(main_layout)

    # ----------------------------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------------------------
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Twitter Sentiment CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            # try to find text/label columns flexibly
            text_col = None
            label_col = None

            # Pass 1: exact column name match (best case)
            for c in df.columns:
                lc = c.lower()
                if lc in ("tweet", "text"):
                    text_col = c
                if lc in ("sentiment", "label"):
                    label_col = c

            # Pass 2: fuzzy match, but skip helper columns like
            # "airline_sentiment_confidence" or "negativereason"
            if text_col is None:
                for c in df.columns:
                    lc = c.lower()
                    if ("tweet" in lc or "text" in lc) and "id" not in lc:
                        text_col = c
                        break
            if label_col is None:
                for c in df.columns:
                    lc = c.lower()
                    if ("sentiment" in lc or "label" in lc) and "confidence" not in lc and "reason" not in lc:
                        label_col = c
                        break

            if text_col is None or label_col is None:
                QMessageBox.warning(
                    self, "Column Error",
                    "CSV must contain a tweet/text column and a sentiment/label column."
                )
                return

            self.dataframe = df.rename(columns={text_col: "tweet", label_col: "sentiment"})
            self.populate_table()
            self.update_status()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{e}")

    def populate_table(self):
        self.table.setRowCount(0)
        for _, row in self.dataframe.iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(row["tweet"])))
            self.table.setItem(r, 1, QTableWidgetItem(str(row["sentiment"])))

    def on_row_clicked(self, row, column):
        tweet_item = self.table.item(row, 0)
        if tweet_item is None:
            return
        text = tweet_item.text()
        self.selected_text.setPlainText(text)
        self.run_prediction(text)

    # ----------------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------------
    def load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select saved_bert_model folder")
        if not folder:
            return
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(folder)
            self.model = BertForSequenceClassification.from_pretrained(folder)
            self.model.eval()

            label_map_path = os.path.join(folder, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    mapping = json.load(f)
                    self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            else:
                self.id2label = self.model.config.id2label

            self.update_status()
            QMessageBox.information(self, "Model Loaded", f"Model loaded from:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    # ----------------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------------
    def predict_manual(self):
        text = self.selected_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Needed", "Please type a sentence or click a tweet.")
            return
        self.run_prediction(text)

    def run_prediction(self, text):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load the BERT model first.")
            return
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=64
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]
                pred_id = int(torch.argmax(probs))
                confidence = float(probs[pred_id])

            label = self.id2label.get(pred_id, str(pred_id)) if isinstance(self.id2label, dict) \
                else self.id2label[pred_id]
            label_str = str(label).upper()

            self.result_label.setText(f"Predicted Sentiment: {label_str}")
            self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")

            lc = str(label).lower()
            if "pos" in lc:
                bg_color, fg_color = COLOR_POSITIVE, "#0a2417"
            elif "neg" in lc:
                bg_color, fg_color = COLOR_NEGATIVE, "#2b0a10"
            else:
                bg_color, fg_color = COLOR_NEUTRAL, "#062430"

            self.result_label.setStyleSheet(
                f"font-size: 21px; font-weight: 800; padding: 18px; "
                f"border-radius: 12px; background-color: {bg_color}; color: {fg_color};"
            )
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    # ----------------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------------
    def update_status(self):
        model_state = "Loaded ✅" if (self.model is not None) else "Not loaded"
        dataset_state = "Loaded ✅" if (self.dataframe is not None) else "Not loaded"
        self.status_label.setText(f"Model: {model_state}    |    Dataset: {dataset_state}")


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
