import pandas as pd


def load_data(file_path):
    # Custom parser for handling diverse timestamp formats
    dateparse = lambda x: pd.to_datetime(x, errors='coerce')

    # Load the dataset
    data = pd.read_csv(file_path, delimiter='\t', parse_dates=['timestamp'], date_parser=dateparse)
    return data

def check_missing_values(data):
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values in each column:\n", missing_values)

def validate_data_format(data):
    # Validate data formats
    if data['tweetId'].dtype == 'int64' and data['userId'].dtype == 'int64':
        print("tweetId and userId formats are correct.")
    else:
        print("Error in tweetId or userId formats.")

    if pd.to_datetime(data['timestamp'], errors='coerce').notnull().all():
        print("Timestamp format is correct.")
    else:
        print("Error in timestamp format.")

def check_duplicates(data):
    # Check for duplicate rows
    duplicates = data.duplicated().sum()
    print("Number of duplicate rows:", duplicates)

def analyze_label_distribution(data):
    # Analyze label distribution
    label_distribution = data['label'].value_counts()
    print("Label distribution:\n", label_distribution)

def examine_unique_values(data):
    # Examine unique values in specific columns
    unique_users = data['userId'].nunique()
    unique_images = data['imageId(s)'].nunique()
    unique_usernames = data['username'].nunique()
    print(f"Unique users: {unique_users}, Unique images: {unique_images}, Unique usernames: {unique_usernames}")

def data_quality_review(file_path):
    try:
        data = load_data(file_path)
        check_missing_values(data)
        validate_data_format(data)
        check_duplicates(data)
        analyze_label_distribution(data)
        examine_unique_values(data)
    except Exception as e:
        print(f"Error occurred: {e}")

# Example usage
print('testing set review:')
file_path = 'mediaeval-2015-testset.txt'
data_quality_review(file_path)
print('training set review')
file_path = 'mediaeval-2015-trainingset.txt'
data_quality_review(file_path)
