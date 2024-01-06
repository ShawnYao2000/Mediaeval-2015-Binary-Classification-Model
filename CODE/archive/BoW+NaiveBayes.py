import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from dataSanitation import load_and_preprocess_data
# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with BOW and Naive Bayes
pipeline_bow_bayes = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('naive_bayes', MultinomialNB())
])

# Train the model
pipeline_bow_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred_bow_bayes = pipeline_bow_bayes.predict(X_test)

# Evaluate the model
accuracy_bow_bayes = accuracy_score(y_test, y_pred_bow_bayes)
classification_rep_bow_bayes = classification_report(y_test, y_pred_bow_bayes, zero_division=1)  # Set zero_division parameter

# Print the results
print('Pipeline with BOW and Naive Bayes:')
print(f'Accuracy: {accuracy_bow_bayes}')
print('Classification Report:')
print(classification_rep_bow_bayes)
