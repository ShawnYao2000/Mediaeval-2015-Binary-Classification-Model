import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataSanitation import load_and_preprocess_data
# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with N-gram (bigram) and Logistic Regression
pipeline_ngram_logreg = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(2, 2))),  # Bigram representation
    ('logistic_regression', LogisticRegression(max_iter=1000))  # Increased max_iter
])

# Train the model
pipeline_ngram_logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ngram_logreg = pipeline_ngram_logreg.predict(X_test)

# Evaluate the model
accuracy_ngram_logreg = accuracy_score(y_test, y_pred_ngram_logreg)
classification_rep_ngram_logreg = classification_report(y_test, y_pred_ngram_logreg, zero_division=1)  # Set zero_division parameter

# Print the results
print('Pipeline with N-gram (Bigram) and Logistic Regression:')
print(f'Accuracy: {accuracy_ngram_logreg}')
print('Classification Report:')
print(classification_rep_ngram_logreg)
