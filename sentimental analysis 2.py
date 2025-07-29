import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
data = {
    'review': [
        "Excellent product, I loved it!",
        "Worst product ever, very disappointed.",
        "Average quality, okay for the price.",
        "Absolutely fantastic and worth the money!",
        "Do not buy this, it broke in a week.",
        "Good value for money, satisfied.",
        "Terrible quality, complete waste.",
        "I am happy with this purchase.",
        "Not as expected, pretty bad.",
        "Satisfied with the product performance."
    ],
    'sentiment': [
        "positive", "negative", "neutral", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive"
    ]
}

df = pd.DataFrame(data)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation/numbers
    return text

df['cleaned'] = df['review'].apply(clean_text)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sample_review = ["This product is amazing and high quality!"]
sample_clean = [clean_text(sample_review[0])]
sample_vector = vectorizer.transform(sample_clean)

prediction = clf.predict(sample_vector)
print("\nSentiment Prediction:", prediction[0])
