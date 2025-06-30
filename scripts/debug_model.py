# scripts/debug_model.py
import pickle
from src.utils.preprocessing import preprocess_text

print("--- Loading final production artifacts ---")
with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("artifacts/lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Artifacts loaded successfully.")

comments = [
    "This video was absolutely amazing, thank you so much!",
    "I did not like this at all.",
    "This is a neutral statement about the video.",
]

print("\nPreprocessing test comments...")
processed_comments = [preprocess_text(comment) for comment in comments]

print("Making predictions...")
vectorized_comments = vectorizer.transform(processed_comments)
predictions = model.predict(vectorized_comments)

print("\n--- FINAL PREDICTION RESULTS ---")
for comment, pred in zip(comments, predictions):
    print(f"Comment: '{comment}' -> Predicted: {pred}")
