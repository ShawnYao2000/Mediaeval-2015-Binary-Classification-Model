import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataSanitation import load_and_preprocess_data
import time
start_time = time.time()
# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with TF-IDF and Naive Bayes
pipeline_tfidf_bayes = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('naive_bayes', MultinomialNB())
])

# Train the model
pipeline_tfidf_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tfidf_bayes = pipeline_tfidf_bayes.predict(X_test)

# Evaluate the model
accuracy_tfidf_bayes = accuracy_score(y_test, y_pred_tfidf_bayes)
classification_rep_tfidf_bayes = classification_report(y_test, y_pred_tfidf_bayes, zero_division=1)  # Set zero_division parameter
end_time = time.time()

print('time consumed', start_time-end_time)
# Print the results
print('Pipeline with TF-IDF and Naive Bayes:')
print(f'Accuracy: {accuracy_tfidf_bayes}')
print('Classification Report:')
print(classification_rep_tfidf_bayes)

