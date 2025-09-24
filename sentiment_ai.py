import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# -- 1. Load data --
df = pd.read_csv("IMDB Dataset.csv")

# print("Number of rows:", len(df))
# print(df.head())

# -- 2. Train / Test split --
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -- 3. Text vectorization (TF-IDF) --
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -- 4. Training the model --
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -- 5. Prediction --
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed report:\n", classification_report(y_test, y_pred))

# -- 6. Testing with a sample --
sample = ["This movie was absolutely bad, I hated it!"]
sample_vec = vectorizer.transform(sample)
print("Prediction:", model.predict(sample_vec)[0])