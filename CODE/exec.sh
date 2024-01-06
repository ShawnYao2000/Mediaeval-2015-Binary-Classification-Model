#!/bin/bash

# Create the result folder in the current directory if it doesn't exist
mkdir -p result

# Change directory to where the scripts are located
cd /Users/mac/PycharmProjects/COMP3222/CODE/archive/

# Get the total number of scripts
total_scripts=$(ls -1 *.py | wc -l)
completed_scripts=0

# Iterate through each .py script in the CODE directory
for script in *.py; do
    # Extract the script name without the extension
    script_name_no_extension="${script%.py}"

    # Execute the Python script and redirect the output to a file in the result folder
    /Users/mac/venv/bin/python "$script" > "/Users/mac/PycharmProjects/COMP3222/CODE/result/${script_name_no_extension}_result.txt"

    # Increment the completed_scripts count
    ((completed_scripts++))

    # Calculate the percentage completion
    percentage=$((completed_scripts * 100 / total_scripts))

    # Print the percentage completion
    echo "Percentage complete: $percentage%"

done

# Execute plot.py
/Users/mac/venv/bin/python "plot/plot.py"
