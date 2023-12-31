import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin

# Load training dataset
train_file_path = "mediaeval-2015-trainingset.txt"
train_data = pd.read_csv(train_file_path, sep='\t')

# Modify labels: Treat 'humor' as 'fake'
train_data['label'] = train_data['label'].replace('humor', 'fake')

# Load testing dataset
test_file_path = "mediaeval-2015-testset.txt"
test_data = pd.read_csv(test_file_path, sep='\t')

# Modify labels in the test set: Treat 'humor' as 'fake'
test_data['label'] = test_data['label'].replace('humor', 'fake')

# Doc2Vec Transformer
class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = Doc2Vec(vector_size=50, min_count=2, epochs=40)

    def fit(self, X, y=None):
        tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(X)]
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, X, y=None):
        return [self.model.infer_vector(doc.split()) for doc in X]

# Pipeline with Doc2Vec and SVM
pipeline_doc2vec_svm = Pipeline([
    ('doc2vec', Doc2VecTransformer()),
    ('svm', SVC())
])

# Train the model
pipeline_doc2vec_svm.fit(train_data['tweetText'], train_data['label'])

# Make predictions on the test set
y_pred_doc2vec_svm = pipeline_doc2vec_svm.predict(test_data['tweetText'])

# Evaluate the model
accuracy_doc2vec_svm = accuracy_score(test_data['label'], y_pred_doc2vec_svm)
classification_rep_doc2vec_svm = classification_report(test_data['label'], y_pred_doc2vec_svm, zero_division=1)

# Print the results
print('Pipeline with Doc2Vec and SVM Classifier:')
print(f'Accuracy: {accuracy_doc2vec_svm}')
print('Classification Report:')
print(classification_rep_doc2vec_svm)
