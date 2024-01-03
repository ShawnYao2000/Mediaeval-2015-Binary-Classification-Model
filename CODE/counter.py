import csv
import matplotlib.pyplot as plt


def count_tweet_lengths(file_name):
    lengths = []

    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')

        for row in reader:
            tweet_text = row['tweetText'].strip()
            lengths.append(len(tweet_text))

    return lengths



def plot_data(test_lengths, training_lengths):
    # Create subplots
    fig, ax = plt.subplots()

    # Calculate the range for the bins
    max_length = max(max(test_lengths), max(training_lengths))
    bins = range(0, 175 + 10, 10)  # Adjust bin size as needed

    # Plotting histograms and getting the bin counts
    test_hist, bins, _ = plt.hist(test_lengths, bins=bins, alpha=0.5, label='Test Dataset', color='blue', edgecolor='black')
    train_hist, _, _ = plt.hist(training_lengths, bins=bins, alpha=0.5, label='Training Dataset', color='green', edgecolor='black')

    # Annotating the histogram with the count of tweets in each bin
    for count, x in zip(test_hist, bins):
        if count > 0:
            plt.text(x, count, str(int(count)), ha='center', va='bottom')

    for count, x in zip(train_hist, bins):
        if count > 0:
            plt.text(x, count, str(int(count)), ha='center', va='bottom')

    # Setting the x-axis and y-axis labels
    plt.xlabel('Tweet Length (Number of Characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tweet Text Lengths(0-185)')

    # Adding the legend and showing the plot
    plt.legend()
    plt.show()


# Count labels in each file
test_counts = count_tweet_lengths('mediaeval-2015-testset.txt')
training_counts = count_tweet_lengths('mediaeval-2015-trainingset.txt')
print(max(test_counts))
print(max(training_counts))

# Plotting the data
plot_data(test_counts, training_counts)

