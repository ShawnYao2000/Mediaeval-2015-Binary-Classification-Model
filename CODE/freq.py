import re
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords

# Set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Additional words to exclude (URL parts, numbers, dates, etc.)
additional_exclusions = {
    'http', 'https', 'co', 'com', 'org', 'net', '0000',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
    'sun', 'rt', 'hurricane', 'girl', 'boy',
    'hero', 'syrianboy_1', 'new', 'rescue',
    'nyc', 'de', 'tower', 'historic', 'sandya_fake_29',
    'photo', 'n', 'york', 'via', 'السوري', 'nepal_25', 'lo', 'go', 'ct',
    'pic', 'label', 'la', 'el', 'ny', 'nepal_01', 'nh'
}

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize(text):
    # Tokenize the text into words, excluding punctuation
    return re.findall(r'\b\w+\b', text.lower())

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def find_base_word(word, base_words):
    for base in base_words:
        if base in word or word in base:
            return base
    return word

def group_keywords(word_counts):
    grouped_counts = defaultdict(int)
    word_groupings = defaultdict(set)
    base_words = set()

    for word in word_counts:
        base_word = find_base_word(word, base_words)
        base_words.add(base_word)
        grouped_counts[base_word] += word_counts[word]
        word_groupings[base_word].add(word)

    return grouped_counts, word_groupings

def get_most_common_words(text, exclude_words, top_n=10):
    words = tokenize(text)
    word_counts = Counter(words)

    for word in list(word_counts):
        if word in exclude_words or is_number(word):
            del word_counts[word]

    grouped_counts, word_groupings = group_keywords(word_counts)
    return Counter(grouped_counts).most_common(top_n), word_groupings

def main():
    training_set_text = read_file('mediaeval-2015-trainingset.txt')
    test_set_text = read_file('mediaeval-2015-testset.txt')

    exclude_labels = {'real', 'humor', 'fake'}
    exclude_words = stop_words.union(exclude_labels, additional_exclusions)

    combined_text = training_set_text + ' ' + test_set_text
    top_words, word_groupings = get_most_common_words(combined_text, exclude_words)

    print("Top 10 most frequent words (excluding stopwords, labels, and additional exclusions):")
    for word, count in top_words:
        grouped_word_count = len(word_groupings[word])
        print(f"{word} {grouped_word_count}: {count}")

if __name__ == "__main__":
    main()