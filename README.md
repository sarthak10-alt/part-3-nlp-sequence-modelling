# part-3-nlp-sequence-modelling
# NLP and Sequence Modeling Mini Project

## Problem Statement

The goal of this project is to build an NLP pipeline to classify customer support messages by sentiment.

The target column is `sentiment_label`, which contains three classes:

- positive
- neutral
- negative

The input text column is `customer_message`.

---

## Task 1: Dataset Understanding

The dataset contains customer support messages from different channels such as chat, phone, email, and social media.

The dataset includes:

- ticket_id
- channel
- customer_message
- sentiment_label
- word_count
- urgent_flag

The target variable is `sentiment_label`.

---

## Task 2: Text Preprocessing

Text preprocessing was performed using the following steps:

- Lowercasing text
- Removing special characters and numbers
- Removing extra spaces
- Tokenizing text
- Removing stopwords using TF-IDF
- Padding sequences for the LSTM model

---

## Task 3: Text Vectorization

Text must be converted into numerical form because machine learning models cannot directly understand raw text.

Two approaches were used:

1. TF-IDF Vectorization
2. Tokenizer-based sequences for LSTM

TF-IDF gives importance to useful words while reducing the importance of common words.

Tokenizer-based sequences convert words into integer sequences so they can be used in deep learning models.

---

## Task 4: Baseline Model

A Logistic Regression model with TF-IDF features was used as the baseline model.

This model was evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix

---

## Task 5: Sequence Model

An LSTM model was built using:

- Input sequence
- Embedding layer
- LSTM layer
- Dense output layer
- Softmax activation

The model uses sparse categorical crossentropy because this is a multi-class classification problem.

---

## Task 6: Attention and Transformer Reflection

### Why RNNs Struggle with Long-Term Dependencies

RNNs process text step by step. When the sequence is long, they may forget earlier information because previous information becomes weaker over time.

### How LSTMs Help with Memory

LSTMs use gates to decide what information to remember, forget, and pass forward. This helps them handle longer text better than simple RNNs.

### What Attention Solves

Attention helps the model focus on the most important words in a sentence instead of treating all words equally.

### Why Transformers are Important

Transformers use attention mechanisms to understand relationships between words more effectively. They are important in modern NLP and Generative AI because they can process large text data efficiently and generate high-quality language outputs.

---

## Results

The evaluation results were saved inside the `results` folder.

Files generated:

- model_evaluation.csv
- confusion_matrix.png
- sample_predictions.txt
