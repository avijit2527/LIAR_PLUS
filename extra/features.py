
import numpy as np
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize






data = pd.read_csv('dataset/train.tsv',delimiter='\t')
data_val = pd.read_csv('dataset/val.tsv',delimiter='\t')

corpus = data['justification'].to_numpy()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
X = normalize(X)
print(X.toarray())
print(X.toarray().shape)


