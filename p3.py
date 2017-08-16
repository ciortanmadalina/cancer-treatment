import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
# docA= 'the cat is under the table'
# bowA =  docA.split()
# wordSet = set(bowA)
# wordDict = dict.fromkeys(wordSet, 0)
# print(wordDict)
text = 'pip install nltk scikit-learn $ python -m nltk.downloader all ... review the complete code if trying to execute the snippets on this tutorial. Pipelines. The heart of building machine learning tools with Scikit-Learn is the Pipeline . ... the tokens in the document as a feature vector, for example as a TF-IDF vector.'

tags = [
  "le chat est fantastique",
  "il fait beau",
  "le chat biboubable",
]

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(tags)

print('done', train_data_features)

np.asarray(train_data_features)

# ******* Train a random forest using the bag of words
#
print ("Training the random forest (this may take a while)...")
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, [1,2,1])

test = vectorizer.fit_transform(["le chat adorable"])
np.asarray(test)
p=forest.predict(test)
print(p)

vectorizer.
