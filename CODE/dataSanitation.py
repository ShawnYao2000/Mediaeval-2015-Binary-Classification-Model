import pandas as pd
import re
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from spellchecker import SpellChecker
# Set of stopwords from multiple languages
stop_words = set(stopwords.words('english') +
                 stopwords.words('spanish') +
                 stopwords.words('french') +
                 stopwords.words('italian') +
                 stopwords.words('portuguese') +
                 stopwords.words('finnish') +
                 stopwords.words('swedish') +
                 stopwords.words('catalan') +
                 stopwords.words('hungarian'))

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

def count_mentions(text):
    count = text.count("@")
    return "{" + str(count) +"}"

def load_and_preprocess_data():
    # Load and preprocess training dataset
    train_data = pd.read_csv(train_file_path, sep='\t')
    train_data['tweetText'] = train_data['tweetText'].apply(preprocess_text)
    # Count mentions in tweets
    train_data['mention_count'] = train_data['tweetText'].apply(count_mentions)
    # Replace 'humor' label with 'fake'
    train_data['label'] = train_data['label'].replace('humor', 'fake')
    # Select relevant columns for features
    X_train = train_data['tweetText'].astype(str) + '<' + \
              train_data['userId'].astype(str) + '<' + \
              train_data['timestamp'].astype(str) + '<'
    y_train = train_data['label']
    #==============================
    # Load testing dataset
    test_data = pd.read_csv(test_file_path, sep='\t')
    test_data['tweetText'] = test_data['tweetText'].apply(preprocess_text)
    # Count mentions in tweets
    test_data['mention_count'] = test_data['tweetText'].apply(count_mentions)
    # Replace 'humor' label with 'fake'
    test_data['label'] = test_data['label'].replace('humor', 'fake')
    # Select relevant columns for features
    X_test = test_data['tweetText'].astype(str) + '<' + \
             test_data['userId'].astype(str) + '<' + \
             test_data['timestamp'].astype(str) + '<'
    y_test = test_data['label']
    return X_train, y_train, X_test, y_test

load_and_preprocess_data()