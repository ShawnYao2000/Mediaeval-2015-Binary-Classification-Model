import os
import matplotlib.pyplot as plt

# Path to the result folder
result_folder = '../result'

# Lists to store classifier names, accuracy and F1 scores
classifier_names = []
accuracy_scores = []
f1_scores = []

# Read each .txt file in the result folder
for filename in os.listdir(result_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(result_folder, filename), 'r') as f:
            lines = f.readlines()
            # Get the classifier name from the filename
            classifier_names.append(filename[:-4].replace("_result",""))
            # Get the accuracy and F1 scores from the file content
            for line in lines:
                if 'Accuracy:' in line:
                    accuracy_scores.append(float(line.split(': ')[1].strip()))
                if 'weighted avg' in line:
                    f1_scores.append(float(line.split()[-2]))

# Function to add value labels on the bars
def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        label = "{:.2f}".format(y_value)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)


# After processing all files
print("Number of classifiers:", len(classifier_names))
print("Number of accuracy scores:", len(accuracy_scores))
print("Number of F1 scores:", len(f1_scores))

# Print the lists if there is a mismatch
if len(classifier_names) != len(accuracy_scores) or len(classifier_names) != len(f1_scores):
    print("Classifier Names:", classifier_names)
    print("Accuracy Scores:", accuracy_scores)
    print("F1 Scores:", f1_scores)


# Plot accuracy scores
fig, ax = plt.subplots(figsize=(11, 7))
rects1 = ax.bar(classifier_names, accuracy_scores, color='r', alpha=0.7)
plt.xticks(rotation='vertical')
plt.xlabel('Classifier Name')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of Classifiers')
add_value_labels(ax)
plt.tight_layout()
plt.savefig('accuracy_scores.png')

# Plot F1 scores
fig, ax = plt.subplots(figsize=(11, 7))
rects2 = ax.bar(classifier_names, f1_scores, color='r', alpha=0.7)
plt.xticks(rotation='vertical')
plt.xlabel('Classifier Name')
plt.ylabel('F1 Score')
plt.title('F1 Scores of Classifiers')
add_value_labels(ax)
plt.tight_layout()
plt.savefig('f1_scores.png')
