# Twitter Sentiment Analysis - BERT + PyQt GUI
Shifa Tameer-e-Millat University | NLP Lab | Assignment 03

## 1. Project Overview
This project trains a BERT (`bert-base-uncased`) model to classify tweets as
**Positive / Negative / Neutral**, saves the trained model, and connects it
to a PyQt5 desktop GUI where a user can:
- Load a CSV dataset of tweets
- Load the saved BERT model
- Click any tweet to see its predicted sentiment
- Type a new sentence and predict its sentiment

## 2. Dataset Source
`dataset/twitter_sentiment.csv` included here is a small **sample dataset**
(90 rows, 3 balanced classes: positive/negative/neutral) created for demo and
testing purposes so the whole pipeline can be verified end-to-end.

**For the real submission, replace this file with a real, larger Twitter
sentiment dataset**, for example one of these free public datasets:
- Sentiment140 (Kaggle): https://www.kaggle.com/datasets/kazanova/sentiment140
- Twitter US Airline Sentiment (Kaggle): https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- Twitter Sentiment Analysis Dataset (Kaggle, various)

Just make sure the CSV has:
- a text column named `tweet` or `text`
- a label column named `sentiment` or `label` (values like positive/negative/neutral)

Then update `DATA_PATH` at the top of `train_bert.py` if the filename is different.

## 3. Project Structure
```
assignment_03_bert_twitter_sentiment/
|-- train_bert.py          # training + evaluation + saving the model
|-- app.py                 # PyQt5 GUI application
|-- requirements.txt
|-- README.md
|-- dataset/
|   |-- twitter_sentiment.csv
|-- saved_bert_model/      # created automatically after training
|-- results/                # graphs + evaluation report created automatically
|-- screenshots/            # add your GUI screenshots here
```

## 4. Setup Instructions

### Step 1 - Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 2 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 - Train the BERT model
```bash
python train_bert.py
```
This will:
- Load `dataset/twitter_sentiment.csv`
- Fine-tune BERT for sentiment classification
- Print accuracy, precision, recall, F1-score, and a classification report
- Save graphs to `results/`:
  - `class_distribution.png`
  - `loss_curve.png`
  - `accuracy_curve.png`
  - `confusion_matrix.png`
  - `evaluation_report.txt`
- Save the trained model + tokenizer to `saved_bert_model/`

Training time depends on dataset size and whether a GPU is available.
With a small dataset and CPU only, training will still finish but may take
a few minutes.

### Step 4 - Run the PyQt GUI
```bash
python app.py
```
In the GUI:
1. Click **Load BERT Model** and select the `saved_bert_model` folder.
2. Click **Load Dataset CSV** and select `dataset/twitter_sentiment.csv`
   (or any properly formatted CSV).
3. Click any row in the tweet table -> predicted sentiment appears on the
   right panel with confidence score.
4. Or type any sentence in the text box and click **Predict**.

## 5. Model Details (for the report)
- Base model: `bert-base-uncased` (Hugging Face Transformers)
- Task: Sequence classification, 3 classes (positive / negative / neutral)
- Max sequence length: 64 tokens
- Epochs: 4
- Batch size: 8
- Train/test split: 80/20, stratified by label
- Optimizer/scheduler: default `Trainer` (AdamW)

*(Update these numbers in the report to match what you actually get when you
retrain on the real dataset.)*

## 6. Paul's Critical Thinking Standards (short notes for the report)
- **Clarity** - Dataset has two columns: `tweet` (raw text) and `sentiment`
  (positive/negative/neutral). GUI flow: load model -> load dataset -> click
  tweet or type text -> see prediction.
- **Accuracy** - Report the actual accuracy/F1 printed by `train_bert.py`,
  even if it's not perfect - do not report inflated numbers.
- **Precision** - Model: `bert-base-uncased`, 80/20 split, 4 epochs, batch
  size 8, max length 64 tokens.
- **Relevance** - Loss curve and accuracy curve show training behavior;
  confusion matrix shows exactly which classes get confused.
- **Depth** - Discuss in the report why some tweets are misclassified
  (e.g. sarcasm, slang, short/ambiguous neutral tweets).
- **Logic** - Training script saves the model -> GUI loads that exact saved
  model -> GUI never hardcodes predictions, it always calls the model.
- **Fairness** - Mention class imbalance (if any) and slang/informal
  language in tweets that BERT may not have seen during pretraining.

## 7. Notes / Restrictions Followed
- No Streamlit used - PyQt5 only.
- Full training pipeline included (not a ready-made pipeline).
- Predictions are generated live by the loaded model, never hardcoded.
- Model saving (`save_pretrained`) and loading (`from_pretrained`) are both
  implemented.

## 8. Screenshots
Add these to `screenshots/` before submission:
- `gui_home.png` - GUI right after opening
- `dataset_loaded.png` - after loading the CSV
- `model_loaded.png` - after loading the saved model
- `prediction_result.png` - after clicking a tweet / predicting manual text
- - Saved model download link:https://drive.google.com/file/d/1Incxs5aWcViN5MaOsoGRPM1_VAXpw1hh/view?usp=sharing
