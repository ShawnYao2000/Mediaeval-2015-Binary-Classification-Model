import os
import matplotlib.pyplot as plt

# Path to the result folder
result_folder = '/Users/mac/PycharmProjects/COMP3222/CODE/result'

# Lists to store classifier names, accuracy and F1 scores
classifier_names = []
accuracy_scores = []
f1_scores = []

# ... [previous code] ...

# Read each .txt file in the result folder
for filename in os.listdir(result_folder):
    if filename.endswith('.txt'):
        with open(os.path.join(result_folder, filename), 'r') as f:
            lines = f.readlines()

            classifier_name = filename[:-4].replace("_result", "")
            accuracy = None
            f1_score = None

            for line in lines:
                if 'Accuracy:' in line:
                    if accuracy is not None:
                        print(f"Multiple accuracy entries found in {filename}")
                    accuracy = float(line.split(': ')[1].strip())
                if 'weighted avg' in line:
                    if f1_score is not None:
                        print(f"Multiple F1 score entries found in {filename}")
                    f1_score = float(line.split()[-2])

            if accuracy is not None and f1_score is not None:
                classifier_names.append(classifier_name)
                accuracy_scores.append(accuracy)
                f1_scores.append(f1_score)
            else:
                print(f"Missing data in {filename}")

# Function to add value labels on the bars
def add_value_labels(ax, spacing=5, fontsize=12):
    for rect in ax.patches:
        x_value = rect.get_x() + rect.get_width() / 2
        y_value = rect.get_height()
        space = spacing
        va = 'bottom'
        label = "{:.2f}".format(y_value)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(space, 0),
            textcoords="offset points",
            ha='center',
            va=va,
            rotation='vertical',
            fontsize=fontsize)

# After processing all files
print("Number of classifiers:", len(classifier_names))
print("Number of accuracy scores:", len(accuracy_scores))
print("Number of F1 scores:", len(f1_scores))

# Plot accuracy scores
fig, ax = plt.subplots(figsize=(11, 7))
rects1 = ax.bar(classifier_names, accuracy_scores, color='r', alpha=0.7)
plt.xticks(rotation='vertical', fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Classifier Name', fontsize=16)
plt.ylabel('Accuracy Score', fontsize=16)
plt.title('Accuracy Scores of Classifiers', fontsize=18)
add_value_labels(ax, fontsize=12)
plt.tight_layout()
plt.savefig('accuracy_scores.png')

# Plot F1 scores
fig, ax = plt.subplots(figsize=(11, 7))
rects2 = ax.bar(classifier_names, f1_scores, color='r', alpha=0.7)
plt.xticks(rotation='vertical', fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Classifier Name', fontsize=16)
plt.ylabel('F1 Score', fontsize=16)
plt.title('F1 Scores of Classifiers', fontsize=18)
add_value_labels(ax, fontsize=12)
plt.tight_layout()
plt.savefig('f1_scores.png')