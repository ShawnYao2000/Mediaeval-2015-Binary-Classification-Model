import pandas as pd
import re
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

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
