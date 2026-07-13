"""Load the trained model and vectorizer to predict spam/ham for new text.

Usage:
    python src/predict.py "Hi, you have won a free lottery, click here!"
"""

import argparse
import pickle
import sys
from pathlib import Path

# Ensure src/ is importable when predict.py is imported from the project root.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from model import SpamClassifier
from preprocess import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_vectorizer = None


def load_artifacts(model_path: Path = MODEL_PATH, vectorizer_path: Path = VECTORIZER_PATH):
    """Load the trained model and vectorizer into memory (cached after first call)."""
    global _model, _vectorizer

    if _vectorizer is None:
        with open(vectorizer_path, "rb") as f:
            _vectorizer = pickle.load(f)

    if _model is None:
        checkpoint = torch.load(model_path, map_location=_device)
        _model = SpamClassifier(checkpoint["num_inputs"]).to(_device)
        _model.load_state_dict(checkpoint["state_dict"])
        _model.eval()

    return _model, _vectorizer


def predict_email(email_text: str) -> str:
    """Return 'Spam' or 'Ham' for the given email/message text."""
    model, vectorizer = load_artifacts()

    cleaned = preprocess_text(email_text)
    input_vector = vectorizer.transform([cleaned]).toarray()
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(_device)

    with torch.no_grad():
        output = model(input_tensor)

    return "Spam" if output.item() > 0.5 else "Ham"


def main():
    parser = argparse.ArgumentParser(description="Predict spam/ham for a message.")
    parser.add_argument("text", type=str, help="The message text to classify")
    args = parser.parse_args()

    result = predict_email(args.text)
    print(f"The message is: {result}")


if __name__ == "__main__":
    main()
