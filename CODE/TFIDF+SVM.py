import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
from dataSanitation import load_and_preprocess_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with TF-IDF and SVM
pipeline_tfidf_svm = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('svm', SVC(class_weight='balanced', probability=True))  # Using probability=True to enable predict_proba
])

# Set up the grid search
parameters = {
    'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf_vectorizer__max_df': [0.5, 0.75, 1.0],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline_tfidf_svm, parameters, cv=5, n_jobs=-1, scoring='f1_macro')

# Train the model with grid search
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred_tfidf_svm = best_model.predict(X_test)

# Binarize the output labels for real and fake
y_test_binarized = label_binarize(y_test, classes=['fake', 'real'])

# Get the decision scores or probability predictions
y_scores = best_model.decision_function(X_test)  # If you used probability=True, you can use predict_proba

# Calculate precision and recall for various thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test_binarized, y_scores)

# Find the threshold that provides the best balance between precision and recall
optimal_idx = np.argmax(precisions + recalls)
optimal_threshold = thresholds[optimal_idx]

# Apply the optimal threshold to make final predictions
y_pred_optimized = (y_scores >= optimal_threshold).astype(int)

# Evaluate the model with the optimized threshold
accuracy_tfidf_svm = accuracy_score(y_test_binarized, y_pred_optimized)
classification_rep_tfidf_svm = classification_report(y_test_binarized, y_pred_optimized, zero_division=1, target_names=['fake', 'real'])

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
    plt.ylabel('True Label')
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
print(f'Accuracy: {accuracy_tfidf_svm}')
print('Classification Report:')
print(classification_rep_tfidf_svm)
