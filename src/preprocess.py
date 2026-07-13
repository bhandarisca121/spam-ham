"""Text preprocessing utilities for spam/ham classification."""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None

_stop_words = set(ENGLISH_STOP_WORDS)
_stemmer = PorterStemmer() if PorterStemmer is not None else None
_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z']+\b")


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords, and stem."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    tokens = [t for t in tokens if t not in _stop_words]
    if _stemmer is not None:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Congratulations! You've WON a free lottery ticket, click now!!!"
    print(preprocess_text(sample))
