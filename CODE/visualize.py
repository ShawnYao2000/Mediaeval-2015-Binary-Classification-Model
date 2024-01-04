import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the training data
train_file_path = "/Users/mac/PycharmProjects/COMP3222/CODE/mediaeval-2015-trainingset.txt"

# Read the file, specifying that the first line is the header
df = pd.read_csv(train_file_path, sep='\t', header=0)

# Clean up the timestamp format by removing extra spaces around colons
df['timestamp'] = df['timestamp'].str.replace(' :', ':').str.replace(': ', ':')

# Convert 'timestamp' to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %H:%M:%S %z %Y')

# Convert time to seconds since midnight
df['time_in_seconds'] = df['timestamp'].dt.hour * 3600 + df['timestamp'].dt.minute * 60 + df['timestamp'].dt.second

# Plot the relationship between time (in seconds) and label
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='time_in_seconds', hue='label', element='step', bins=50)
plt.title("Time (in seconds since midnight) and Their Relationship with Labels")
plt.xlabel("Time (seconds since midnight)")
plt.ylabel("Count")
# Optional: Format x-axis to show time in hours and minutes
tick_positions = np.arange(0, 86400, 3600)  # Every hour
tick_labels = [f"{h:02d}:00" for h in range(24)]
plt.xticks(tick_positions, tick_labels, rotation=45)


plt.show()
