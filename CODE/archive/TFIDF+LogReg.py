import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataSanitation import load_and_preprocess_data
# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with TF-IDF and Logistic Regression
pipeline_tfidf_logistic = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('logistic_regression', LogisticRegression(max_iter=1000))  # Increased max_iter
])

# Train the model
pipeline_tfidf_logistic.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tfidf_logistic = pipeline_tfidf_logistic.predict(X_test)

# Evaluate the model
accuracy_tfidf_logistic = accuracy_score(y_test, y_pred_tfidf_logistic)
classification_rep_tfidf_logistic = classification_report(y_test, y_pred_tfidf_logistic, zero_division=1)  # Set zero_division parameter

# Print the results
print('Pipeline with TF-IDF and Logistic Regression:')
print(f'Accuracy: {accuracy_tfidf_logistic}')
print('Classification Report:')
print(classification_rep_tfidf_logistic)
