"""Train the spam/ham classifier and save the model + vectorizer.

Usage:
    python src/train.py --data data/sample_data.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import pandas as pd

from model import SpamClassifier
from preprocess import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"


class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


def load_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path, encoding="latin-1")
    return data


def train(data_path: Path = DEFAULT_DATA_PATH, no_epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load & preprocess ---
    data = load_data(data_path)
    data["Cleaned"] = data["text"].apply(preprocess_text)

    # --- Feature extraction ---
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data["Cleaned"]).toarray()
    y = data["label_num"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(EmailDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # --- Model, loss, optimizer ---
    num_inputs = X_train.shape[1]
    model = SpamClassifier(num_inputs).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    for epoch in range(no_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{no_epochs}] - Loss: {epoch_loss / len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # --- Save model + vectorizer ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "num_inputs": num_inputs},
        MODEL_PATH,
    )
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train the spam/ham classifier.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(Path(args.data), no_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
