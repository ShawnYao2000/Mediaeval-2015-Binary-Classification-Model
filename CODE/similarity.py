import pandas as pd
import re
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set of stopwords from multiple languages
stop_words = set(stopwords.words('english') +
                 stopwords.words('spanish') +
                 stopwords.words('french') +
                 stopwords.words('italian') +
                 stopwords.words('portuguese') +
                 stopwords.words('finnish') +
                 stopwords.words('swedish') +
                 stopwords.words('catalan') +
                 stopwords.words('hungarian') +
                 stopwords.words('dutch') +
                 stopwords.words('romanian') +
                 stopwords.words('turkish') +
                 stopwords.words('german') +
                 stopwords.words('norwegian') +
                 stopwords.words('slovene') +
                 stopwords.words('danish'))

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

def compute_tfidf_vectors(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

def calculate_similarity(tfidf_matrix, index):
    cosine_similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    # Excluding self similarity
    cosine_similarities[index] = 0
    print(np.mean(cosine_similarities)*10000)
    return np.mean(cosine_similarities*10000)


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

def split_text(text, max_length=150):
    # Split text into chunks of max_length characters
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def load_and_preprocess_data():
    # Load and preprocess training dataset
    train_data = pd.read_csv(train_file_path, sep='\t')
    train_data['tweetText'] = train_data['tweetText'].apply(preprocess_text)

    # Computing TF-IDF vectors for the entire dataset
    tfidf_matrix = compute_tfidf_vectors(train_data['tweetText'])

    # Adding similarity scores
    train_data['similarity'] = [calculate_similarity(tfidf_matrix, i) for i in range(tfidf_matrix.shape[0])]

    # Split long texts and duplicate corresponding rows
    new_rows = []
    for _, row in train_data.iterrows():
        split_texts = split_text(row['tweetText'])
        for text in split_texts:
            new_row = row.copy()
            new_row['tweetText'] = text
            new_rows.append(new_row)
    train_data = pd.DataFrame(new_rows)

    # Replace 'humor' label with 'fake'
    train_data['label'] = train_data['label'].replace('humor', 'fake')

    # Select relevant columns for features
    X_train = train_data['tweetText'].astype(str) + '<' + \
              train_data['userId'].astype(str) + '<' + \
              train_data['timestamp'].astype(str) + '<' + \
              train_data['similarity'].astype(str)

    y_train = train_data['label']

    # Load testing dataset
    test_data = pd.read_csv(test_file_path, sep='\t')
    test_data['tweetText'] = test_data['tweetText'].apply(preprocess_text)

    # Computing TF-IDF vectors for the testing dataset
    tfidf_matrix_test = compute_tfidf_vectors(test_data['tweetText'])

    # Adding similarity scores for testing dataset
    test_data['similarity'] = [calculate_similarity(tfidf_matrix_test, i) for i in range(tfidf_matrix_test.shape[0])]

    # Replace 'humor' label with 'fake'
    test_data['label'] = test_data['label'].replace('humor', 'fake')

    # Select relevant columns for features
    X_test = test_data['tweetText'].astype(str) + '<' + \
             test_data['userId'].astype(str) + '<' + \
             test_data['timestamp'].astype(str) + '<' + \
             test_data['similarity'].astype(str)

    y_test = test_data['label']

    return X_train, y_train, X_test, y_test

def load_and_preprocess_data_with_augmentation():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Data augmentation on training data
    X_train_augmented = X_train.apply(lambda x: synonym_replacement(x))

    return X_train_augmented, y_train, X_test, y_test

def synonym_replacement(sentence, n=5):
    words = sentence.split()
    new_sentence = sentence
    replaced = 0
    for word in words:
        if replaced >= n or word in stop_words:
            continue
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        if synonyms:
            synonym = synonyms.pop()
            new_sentence = new_sentence.replace(word, synonym, 1)
            replaced += 1
    return new_sentence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Assuming the functions compute_tfidf_vectors, preprocess_text, calculate_similarity are defined as before

def load_and_preprocess_data_for_plotting():
    # Load data
    train_data = pd.read_csv(train_file_path, sep='\t')
    train_data['tweetText'] = train_data['tweetText'].apply(preprocess_text)

    # Compute TF-IDF vectors
    tfidf_matrix = compute_tfidf_vectors(train_data['tweetText'])

    # Calculate similarity scores
    similarity_scores = [calculate_similarity(tfidf_matrix, i) for i in range(tfidf_matrix.shape[0])]

    return similarity_scores

def plot_similarity_counts(similarity_scores):
    # Count the occurrences of each similarity score
    counts = pd.Series(similarity_scores).value_counts().sort_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)
    plt.xlabel('Similarity Index')
    plt.ylabel('Counts')
    plt.title('Similarity Index Distribution')
    plt.show()

# Load and preprocess data
similarity_scores = load_and_preprocess_data_for_plotting()

# Plot the chart
plot_similarity_counts(similarity_scores)
