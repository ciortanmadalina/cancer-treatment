import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('data\\training_variants')
trainingText = pd.read_csv('data\\training_text', sep="\|\|", engine='python', header=None, names=["ID","Text"], skiprows=1)

test = pd.read_csv('data\\test_variants')
testText = pd.read_csv('data\\test_text', sep="\|\|", engine='python', header=None, names=["ID","Text"], skiprows=1)
pid = test['ID'].values

train = train.merge(trainingText, on='ID', how='left')
trainGenes = train.groupby('Gene')['Gene'].count()
# trainGenes.plot(kind='bar')
# plt.xlabel('Class Count', fontsize=5)
# plt.xticks(rotation=60)
# plt.show()


plt.figure(figsize=(12, 8))
plt.hist(trainGenes.values, bins=50, log=True)
plt.xlabel('Number of times Gene appeared', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.show()


trainVariation = train.groupby('Variation')['Variation'].count()
plt.figure(figsize=(12, 8))
plt.hist(trainVariation.values, bins=50, log=True)
plt.xlabel('Number of times Variation appeared', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.show()


train["nbWords"] = train["Text"].apply(lambda x: len(str(x).split()) )

sns.distplot(train["nbWords"].values, bins=50, kde=False, color='red')
plt.xlabel('Number of words in text', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Frequency of number of words", fontsize=15)
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(x='Class', y='nbWords', data=train)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Text - Number of words', fontsize=12)
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(x="Class", data=train)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class Count', fontsize=12)
plt.xticks(rotation=60)
plt.title("Class frequency", fontsize=15)
plt.show()


# plt.figure(figsize=(12,8))
# plt.hist(train_genes, bins=50, log=True)
# plt.ylabel('log of count', fontsize=12)
# plt.xlabel('Genes distribution', fontsize=12)
# plt.xticks(rotation=60)
# plt.title("Gene frequency", fontsize=15)
# plt.show()

print('done')