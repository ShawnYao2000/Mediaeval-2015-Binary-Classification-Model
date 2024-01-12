# README.md for MediaEval-2015 Dataset Training Project

## Overview
This project uses the MediaEval 2015 dataset to categorize tweets into 'real', 'fake', or 'humor' classifications. It addresses the challenge of misinformation in social media, aiming to enhance the reliability of information dissemination.

## Dataset
The MediaEval 2015 dataset includes tweets with labels. The dataset has been cleaned to remove missing values, duplicates, and to address label imbalances.

## Preprocessing
The preprocessing steps included emoji and URL replacement, white space removal, case normalization, stop word removal, lemmatization, and tokenization. These steps are crucial for preparing the dataset for machine learning algorithms.

## Feature Selection
Selected features are 'tweetText', 'userId', 'imageId(s)', and 'timeStamp'. These were chosen based on insights from data analysis and their predictive power.

## Algorithm Selection and Evaluation
We compared 11 classifiers with TFIDF+SGD and TFIDF+NaiveBayes showing the best performance based on the F1 score. Class weight adjustment was used to counteract the label imbalance in the dataset.

## Final Model Performance
The Naïve Bayes classifier achieved an F1 score of 0.901, while the SGD classifier's final F1 score was 0.897, indicating a high level of accuracy in tweet classification.

## Conclusion
The project highlights the effectiveness of TF-IDF classifiers combined with Naive Bayes and SGD. Preprocessing and feature selection were pivotal in improving the model's performance.

## How to Use
1. **Clone the repository:**
   ```
   git clone https://github.com/ShawnYao2000/COMP3222.git
   ```
2. **Navigate to the Jupyter notebook:**
   ```
   cd CODE/Jupyter/
   ```
3. **Open the notebook:**
   ```
   jupyter notebook source_code.ipynb
   ```
   Ensure that you have Jupyter installed and running.

4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   This step ensures that you have all necessary libraries.

5. **Run the notebook:**
   Execute the cells in `source_code.ipynb` to reproduce the analysis and the model training process.

## Future Work
Future improvements could include exploring alternative data augmentation methods and investigating other dimensionality reduction techniques.

## References
See the detailed list of references within the report for the studies and methodologies referenced throughout this project.

## Appendix
Additional charts and detailed data analyses are included in the appendix of the project report.