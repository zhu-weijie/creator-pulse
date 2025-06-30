# src/utils/preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
negation_words = {"not", "no", "never", "nor", "don't"}
english_stopwords = set(stopwords.words("english")) - negation_words
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    A robust, shared function for cleaning and preparing text for sentiment analysis.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    words = nltk.word_tokenize(text)

    processed_words = [
        lemmatizer.lemmatize(word) for word in words if word not in english_stopwords
    ]

    return " ".join(processed_words)
