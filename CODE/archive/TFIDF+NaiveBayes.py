import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataSanitation import load_and_preprocess_data


X_train, y_train, X_test, y_test = load_and_preprocess_data()
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Design and implement the pipeline with TF-IDF and Naive Bayes
pipeline_tfidf_bayes = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('naive_bayes', MultinomialNB())
])

# Set up the grid search
parameters = {
    'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf_vectorizer__max_df': [0.4, 0.8],
    'tfidf_vectorizer__min_df': [0.00005],
    'tfidf_vectorizer__max_features': [1975],
    'naive_bayes__alpha': [0.01, 1, 1.5, 2]
}
"""
parameters = {
    'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 5)],
    'tfidf_vectorizer__max_df': [0.25, 0.5, 0.75],
    'tfidf_vectorizer__min_df': [0.01, 0.05, 0.1],
    'tfidf_vectorizer__use_idf': [True, False],
    'tfidf_vectorizer__norm': ['l1', 'l2'],
    'naive_bayes__alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1],
}
"""
grid_search = GridSearchCV(pipeline_tfidf_bayes, parameters, cv=10, n_jobs=-1, scoring='f1_weighted')
# Train the model with grid search
grid_search.fit(X_train, y_train)
# Best model
best_model = grid_search.best_estimator_
# Update the Naive Bayes classifier in the pipeline with class weights
best_model.set_params(naive_bayes__class_prior=class_weight_dict)
# Make predictions on the test set
y_pred_tfidf_bayes = best_model.predict(X_test)
# Binarize the output labels for real and fake
y_test_binarized = label_binarize(y_test, classes=['fake', 'real'])
# Get the probability predictions
y_scores = best_model.predict_proba(X_test)[:, 1]  # probability of being 'real'
# Calculate precision and recall for various thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test_binarized, y_scores)
# Find the threshold that provides the best balance between precision and recall
optimal_idx = np.argmax(precisions + recalls)
optimal_threshold = thresholds[optimal_idx]
# Apply the optimal threshold to make final predictions
y_pred_optimized = (y_scores >= optimal_threshold).astype(int)
# Evaluate the model with the optimized threshold
accuracy_tfidf_bayes = accuracy_score(y_test_binarized, y_pred_optimized)
classification_rep_tfidf_bayes = classification_report(y_test_binarized, y_pred_optimized, zero_division=1, target_names=['fake', 'real'])
# Evaluate the model on the training set
y_train_pred = best_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
classification_rep_train = classification_report(y_train, y_train_pred, zero_division=1)
# Confusion Matrix for the Training Set
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
# Confusion Matrix for the Testing Set (with Optimized Threshold)
confusion_matrix_test = confusion_matrix(y_test_binarized, y_pred_optimized)
def plot_confusion_matrix(conf_mat, title='Confusion Matrix', labels=['Fake', 'Real']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(title)
    plt.show()
# Plot Confusion Matrix for Training Set
plot_confusion_matrix(confusion_matrix_train, title='Confusion Matrix on Training Set')
# Plot Confusion Matrix for Testing Set (Optimized Threshold)
plot_confusion_matrix(confusion_matrix_test, title='Confusion Matrix on Testing Set (Optimized Threshold)')
# Print the results for the training set
print('Performance on Training Set:')
print(f'Accuracy: {accuracy_train}')
print('Classification Report:')
print(classification_rep_train)
# Print the results
print('Performance on Testing Set')
print(f'Accuracy: {accuracy_tfidf_bayes}')
print('Classification Report:')
print(classification_rep_tfidf_bayes)

