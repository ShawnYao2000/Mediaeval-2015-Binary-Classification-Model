import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load training dataset
train_file_path = "mediaeval-2015-trainingset.txt"
train_data = pd.read_csv(train_file_path, sep='\t')

# Split data into features (X) and labels (y)
X_train = train_data['tweetText']
y_train = train_data['label']

# Load testing dataset
test_file_path = "mediaeval-2015-testset.txt"
test_data = pd.read_csv(test_file_path, sep='\t')

# Split data into features (X) and labels (y)
X_test = test_data['tweetText']
y_test = test_data['label']

# Design and implement the pipeline with TF-IDF and SGD
pipeline_tfidf_sgd = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('sgd', SGDClassifier())
])

# Train the model
pipeline_tfidf_sgd.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tfidf_sgd = pipeline_tfidf_sgd.predict(X_test)

# Evaluate the model
accuracy_tfidf_sgd = accuracy_score(y_test, y_pred_tfidf_sgd)
classification_rep_tfidf_sgd = classification_report(y_test, y_pred_tfidf_sgd, zero_division=1)  # Set zero_division parameter

# Print the results
print('Pipeline with TF-IDF and SGD Classifier:')
print(f'Accuracy: {accuracy_tfidf_sgd}')
print('Classification Report:')
print(classification_rep_tfidf_sgd)
