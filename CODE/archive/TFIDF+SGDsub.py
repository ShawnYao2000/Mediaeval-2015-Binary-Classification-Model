from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from nltk import WordNetLemmatizer, word_tokenize, download
from nltk.corpus import stopwords
from dataSanitation import load_and_preprocess_data

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Define the pipeline
pipeline_tfidf_sgd = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(max_features=3500)),
    ('sgd', SGDClassifier())
])

# Parameters for Grid Search
param_grid = {
    #'sgd__loss': ['hinge', 'log', 'modified_huber'],
    'sgd__penalty': ['l2', 'l1', 'elasticnet'],
    'sgd__alpha': [0.0001, 0.001, 0.01, 0.2],
    #'sgd__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    #'sgd__eta0': [0.1, 0.01, 0.001]
}

# Create grid search object
grid_search = GridSearchCV(pipeline_tfidf_sgd, param_grid, cv=5, scoring='f1_weighted')

# Train the model using Grid Search
grid_search.fit(X_train, y_train)

# Best parameter set
print("Best parameters set found on development set:")
print(grid_search.best_params_)

# Evaluate the model with the best parameters
y_pred_tfidf_sgd = grid_search.predict(X_test)
accuracy_tfidf_sgd = accuracy_score(y_test, y_pred_tfidf_sgd)
classification_rep_tfidf_sgd = classification_report(y_test, y_pred_tfidf_sgd, zero_division=1, digits=3)

# Confusion Matrix for the Testing Set
confusion_matrix_test = confusion_matrix(y_test, y_pred_tfidf_sgd)

# Function to plot confusion matrix
def plot_confusion_matrix(conf_mat, title='Confusion Matrix', labels=['Fake', 'Real']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(title)
    plt.show()

# Plot Confusion Matrix for Testing Set
plot_confusion_matrix(confusion_matrix_test, title='Confusion Matrix on Testing Set')

# Print the results for the testing set
print('Performance on Testing Set:')
print(f'Accuracy: {accuracy_tfidf_sgd}')
print('Classification Report:')
print(classification_rep_tfidf_sgd)