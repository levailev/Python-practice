import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# -- Load the dataset --
df = pd.read_csv("IMDB Dataset.csv") 
# print("Number of records:", len(df))
# print(df.head())

# -- Split into train and test sets --
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -- Text vectorization using TF-IDF --
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -- Train the model --
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -- Evaluate the model --
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed classification report:\n", classification_report(y_test, y_pred))


def run_internal_tests(df, vectorizer, model):
    
    print("\nRunning internal tests...")

    # Dataset check
    assert df.shape[0] > 0, "Dataset is empty!"
    assert "review" in df.columns and "sentiment" in df.columns, "Missing required columns"
    print("✅ Dataset check passed")

    # Train/test split check
    X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(df), "Train/test split sizes don't match"
    print("✅ Train/test split check passed")

    # Vectorization check
    X_vec = vectorizer.transform(df["review"][:5])  # first 5 samples
    assert X_vec.shape[0] == 5, "Vectorization failed"
    print("✅ Vectorization check passed")

    # Prediction check
    sample_preds = model.predict(X_vec)
    assert all(p in ["positive", "negative"] for p in sample_preds), "Invalid prediction labels"
    print("✅ Prediction labels check passed")

    # Sample text prediction
    sample_review = ["This movie was absolutely fantastic!"]
    sample_vec = vectorizer.transform(sample_review)
    pred = model.predict(sample_vec)[0]
    assert pred in ["positive", "negative"], "Sample prediction invalid"
    print(f"✅ Sample review prediction: '{pred}'")

    print("All internal tests passed ✅")

# --- Calling internal tests ---
run_internal_tests(df, vectorizer, model)

# -- Test on custom reviews --
print("\n")
sample_reviews = [
    "This was a disaster",
    "Amazing movie!",
    "Not my taste",
    "Absolutely fantastic!"
]

for review in sample_reviews:
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)[0]
    print(f"Prediction for '{review}': {prediction}")