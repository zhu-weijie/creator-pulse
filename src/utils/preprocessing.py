# In src/utils/preprocessing.py
import re
import nltk
from nltk.stem import WordNetLemmatizer


nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    A robust, shared function for cleaning and preparing text for sentiment analysis.
    This version uses the simple, high-performing logic from our initial experiments.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    text = " ".join(text.split())

    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(lemmatized_words)
