import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataSanitation import load_and_preprocess_data
# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with TF-IDF and Random Forest
pipeline_tfidf_rf = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('random_forest', RandomForestClassifier())
])

# Train the model
pipeline_tfidf_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tfidf_rf = pipeline_tfidf_rf.predict(X_test)

# Evaluate the model
accuracy_tfidf_rf = accuracy_score(y_test, y_pred_tfidf_rf)
classification_rep_tfidf_rf = classification_report(y_test, y_pred_tfidf_rf, zero_division=1)

# Print the results
print('Pipeline with TF-IDF and Random Forest Classifier:')
print(f'Accuracy: {accuracy_tfidf_rf}')
print('Classification Report:')
print(classification_rep_tfidf_rf)
