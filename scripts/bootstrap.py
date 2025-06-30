import nltk

print("--- Bootstrapping: Downloading NLTK data ---")

required_packages = ["wordnet", "punkt", "punkt_tab"]

for package in required_packages:
    try:
        nltk.data.find(f"tokenizers/{package}")
    except LookupError:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package, quiet=True)

print("--- Bootstrap complete ---")
