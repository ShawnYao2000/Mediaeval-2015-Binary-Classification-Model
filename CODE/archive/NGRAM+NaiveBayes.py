import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load training dataset
train_file_path = "../mediaeval-2015-trainingset.txt"
train_data = pd.read_csv(train_file_path, sep='\t')

# Split data into features (X) and labels (y)
X_train = train_data['tweetText']
y_train = train_data['label']

# Load testing dataset
test_file_path = "../mediaeval-2015-testset.txt"
test_data = pd.read_csv(test_file_path, sep='\t')

# Split data into features (X) and labels (y)
X_test = test_data['tweetText']
y_test = test_data['label']

# Design and implement the pipeline with N-gram (bigram) and Naive Bayes
pipeline_ngram_bayes = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(2, 2))),  # Bigram representation
    ('naive_bayes', MultinomialNB())
])

# Train the model
pipeline_ngram_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ngram_bayes = pipeline_ngram_bayes.predict(X_test)

# Evaluate the model
accuracy_ngram_bayes = accuracy_score(y_test, y_pred_ngram_bayes)
classification_rep_ngram_bayes = classification_report(y_test, y_pred_ngram_bayes, zero_division=1)  # Set zero_division parameter

# Print the results
print('Pipeline with N-gram (Bigram) and Naive Bayes:')
print(f'Accuracy: {accuracy_ngram_bayes}')
print('Classification Report:')
print(classification_rep_ngram_bayes)
