# Spam/Ham Classifier

# Spam-Ham Email Classification Using Deep Learning

## Overview

This project builds a machine learning model to classify emails as either **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and a Neural 
Network built with PyTorch.

The model processes email text, converts it into numerical features using **TF-IDF Vectorization**, and uses a deep learning classifier to predict whether an email is spam 
or legitimate.

---

## Project Features

- Text preprocessing and cleaning
- TF-IDF feature extraction
- Neural Network classifier using PyTorch
- Binary classification:
  - Spam → 1
  - Ham → 0
- Model evaluation using accuracy
- Ability to predict new unseen emails

---

## Dataset

The dataset contains email messages labeled as:

| Label | Description |
|------|-------------|
| Spam | Unwanted or malicious emails |
| Ham | Legitimate emails |

Dataset columns:

| Column | Description |
|--------|-------------|
| label | Original text label (spam/ham) |
| label_num | Numerical label (0/1) |
| text | Original email content |
| Cleaned | Preprocessed email text |

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorizer

---

## Project Workflow

### Updated
updated the README.md file.
