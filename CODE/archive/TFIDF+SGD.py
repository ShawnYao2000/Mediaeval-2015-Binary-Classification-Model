from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from nltk import WordNetLemmatizer, word_tokenize, download
from nltk.corpus import stopwords

download('stopwords')
# Create individual sets for each language's stopwords
english_stopwords = set(stopwords.words('english'))
spanish_stopwords = set(stopwords.words('spanish'))
french_stopwords = set(stopwords.words('french'))
italian_stopwords = set(stopwords.words('italian'))
portuguese_stopwords = set(stopwords.words('portuguese'))
finnish_stopwords = set(stopwords.words('finnish'))
swedish_stopwords = set(stopwords.words('swedish'))
catalan_stopwords = set(stopwords.words('catalan'))
hungarian_stopwords = set(stopwords.words('hungarian'))

# Use the union method to combine all the sets
stop_words = english_stopwords.union(spanish_stopwords, french_stopwords, italian_stopwords, portuguese_stopwords, finnish_stopwords, swedish_stopwords, catalan_stopwords, hungarian_stopwords)


# File paths
train_file_path = "mediaeval-2015-trainingset.txt"
test_file_path = "mediaeval-2015-testset.txt"
count = 0

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def preprocess_text(text):
    # Replace URLs with a placeholder
    text = re.sub(r'http\S+|www\S+|https\S+', '[U]', text, flags=re.MULTILINE)
    # Replace emojis with ~
    text = emoji_pattern.sub('~', text)
    # Replace non ascii with
    #text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    # Remove extra whitespaces
    text = text.strip()
    return text

def generate_image_id_mapping(data):
    unique_image_ids = data['imageId(s)'].unique()
    image_id_dict = {id: f"{i}" for i, id in enumerate(unique_image_ids, start=1)}
    return image_id_dict

def load_and_preprocess_data():
    # Load training dataset
    train_data = pd.read_csv(train_file_path, sep='\t')
    # Handle NaN values for tweetText
    train_data['tweetText'] = train_data['tweetText'].fillna('')

    # Load testing dataset
    test_data = pd.read_csv(test_file_path, sep='\t')
    test_data['tweetText'] = test_data['tweetText'].fillna('')

    # Generate image ID mapping from combined dataset
    combined_data = pd.concat([train_data, test_data])
    image_id_dict = generate_image_id_mapping(combined_data)

    # Replace image IDs with generated IDs in training data
    train_data['imageId(s)'] = train_data['imageId(s)'].map(image_id_dict)
    # Preprocess text and replace labels in training dataset
    train_data['tweetText'] = train_data['tweetText'].apply(preprocess_text)
    train_data['label'] = train_data['label'].replace('humor', 'fake')
    # Select relevant columns for features
    X_train = train_data['tweetText'].astype(str) + '<' + \
              train_data['userId'].astype(str) + '<' + \
              train_data['timestamp'].astype(str) + '<' + \
              train_data['imageId(s)'].astype(str)
    y_train = train_data['label']

    # Replace image IDs with generated IDs in testing data
    test_data['imageId(s)'] = test_data['imageId(s)'].map(image_id_dict)
    # Preprocess text and replace labels in testing dataset
    test_data['tweetText'] = test_data['tweetText'].apply(preprocess_text)
    test_data['label'] = test_data['label'].replace('humor', 'fake')
    # Select relevant columns for features
    X_test = test_data['tweetText'].astype(str) + '<' + \
             test_data['userId'].astype(str) + '<' + \
             test_data['timestamp'].astype(str) + '<' + \
             test_data['imageId(s)'].astype(str)
    y_test = test_data['label']

    return X_train, y_train, X_test, y_test

# Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Design and implement the pipeline with TF-IDF and SGD
pipeline_tfidf_sgd = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('sgd', SGDClassifier())
])

# Train the model
pipeline_tfidf_sgd.fit(X_train, y_train)

# Make predictions on the test set
y_pred_tfidf_sgd = pipeline_tfidf_sgd.predict(X_test)
y_train_pred = pipeline_tfidf_sgd.predict(X_train)
# Evaluate the model
accuracy_tfidf_sgd = accuracy_score(y_test, y_pred_tfidf_sgd)
classification_rep_tfidf_sgd = classification_report(y_test, y_pred_tfidf_sgd, zero_division=1)  # Set zero_division parameter
# Calculate accuracy for the training set
accuracy_train = accuracy_score(y_train, y_train_pred)
classification_rep_train = classification_report(y_train, y_train_pred, zero_division=1)

# Confusion Matrix for the Training Set
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)

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

# Plot Confusion Matrix for Training Set
plot_confusion_matrix(confusion_matrix_train, title='Confusion Matrix on Training Set')

# Plot Confusion Matrix for Testing Set
plot_confusion_matrix(confusion_matrix_test, title='Confusion Matrix on Testing Set')

# Print the results for the training set
print('Performance on Training Set:')
print(f'Accuracy: {accuracy_train}')
print('Classification Report:')
print(classification_rep_train)

# Print the results for the testing set
print('Performance on Testing Set:')
print(f'Accuracy: {accuracy_tfidf_sgd}')
print('Classification Report:')
print(classification_rep_tfidf_sgd)