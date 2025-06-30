import re
import pickle
from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer

from src.utils.preprocessing import preprocess_text


app = Flask(__name__)


try:
    with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("artifacts/lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    vectorizer = None
    model = None

lemmatizer = WordNetLemmatizer()


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "CreatorPulse API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    """Receives text data and returns sentiment predictions."""
    if not model or not vectorizer:
        return (
            jsonify(
                {"error": "Model or vectorizer not loaded. Run the DVC pipeline first."}
            ),
            500,
        )

    try:
        data = request.get_json()
        comments = data.get("comments")

        if not isinstance(comments, list):
            return jsonify({"error": "Input must be a list of comment strings."}), 400

        processed_comments = [preprocess_text(comment) for comment in comments]
        vectorized_comments = vectorizer.transform(processed_comments)
        predictions = model.predict(vectorized_comments)

        predictions_list = [int(p) for p in predictions]

        return jsonify({"predictions": predictions_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
