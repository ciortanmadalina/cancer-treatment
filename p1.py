import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing
from collections import Counter
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm

trainData = pd.read_csv('data\\training_variants')
trainingText = pd.read_csv('data\\training_text', sep="\|\|", engine='python', header=None, names=["ID","Text"], skiprows=1)

testData = pd.read_csv('data\\test_variants')
testText = pd.read_csv('data\\test_text', sep="\|\|", engine='python', header=None, names=["ID","Text"], skiprows=1)
pid = testData['ID'].values

trainData = trainData.merge(trainingText, on='ID', how='left')
testData = testData.merge(testText, on='ID', how='left')

y = trainData['Class']

all = trainData.drop('Class', axis = 1)

all = all.append(testData)

for c in all.columns:
    if all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            all[c+'_lbl_enc'] = lbl.fit_transform(all[c].values)
            all[c+'_len'] = all[c].map(lambda x: len(str(x)))
            all[c+'_words'] = all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            all[c] = lbl.fit_transform(all[c].values)
        if c=='Text':
            all[c+'_len'] = all[c].map(lambda x: len(str(x)))
            all[c+'_words'] = all[c].map(lambda x: len(str(x).split(' ')))

all = all.drop(['Gene', 'Variation','ID','Text'], axis = 1)

train = all[:trainData.shape[0]]
test = all[trainData.shape[0]:]

x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.2)

def pipelineText():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', svm.LinearSVC())
                         ])
    x1, x2, y1, y2 = model_selection.train_test_split(trainData, y, test_size=0.2)
    text_clf = text_clf.fit(x1['Text'], y1)
    y_test_predicted = text_clf.predict(x2['Text'])
    print('accuracy %s ' % np.mean(y_test_predicted == y2))
    return None

def xgboost():
    print('\n##################\nXGBoost\n##################')
    param = {}
    # param['booster'] = 'gbtree'
    #param['objective'] = 'binary:logistic'
    # param["eval_metric"] = "error"
    # param['eta'] = 0.3
    # param['gamma'] = 0
    param['max_depth'] = 4
    param['n_estimators'] = 80
    param['learning_rate'] = 0.1
    # param['min_child_weight'] = 1
    # param['max_delta_step'] = 0
    # param['subsample'] = 1
    # param['colsample_bytree'] = 1
    # param['silent'] = 1
    # param['seed'] = 0
    # param['base_score'] = 0.4

    estimator = XGBClassifier()
    estimator.set_params(**param)

    calibratedCV = CalibratedClassifierCV(estimator, method='sigmoid', cv=5)
    calibratedCV.fit(x1, y1)
    prediction = calibratedCV.predict(x2)


    # estimator.fit(x1, y1)
    # prediction = estimator.predict(x2)

    print('accuracy %s ' % np.mean(prediction == y2))
    result = calibratedCV.predict(test)
    # result = estimator.predict(test)
    return result


def tensorFlow():
    print('\n##################\nTF\n##################')
    tf.set_random_seed(42)
    x_train = x1
    y_train = y1
    # y_train = list([i] for i in y1)

    input = tf.placeholder(tf.float32, [None, x1.shape[1]])
    label = tf.placeholder(tf.int32, [None,])

    neuronsNb = 5
    weights = tf.Variable(tf.random_normal([x_train.shape[1], neuronsNb]))
    bias = tf.Variable(tf.random_normal([neuronsNb]))

    hiddenLayer1 = tf.matmul(input, weights) + bias
    hiddenLayer1 = tf.nn.relu(hiddenLayer1)

    weights = tf.Variable(tf.random_normal([neuronsNb, 9]))
    outputLayer = tf.matmul(hiddenLayer1, weights)
    pred = tf.argmax(outputLayer, 1)

    onehotLabels = tf.one_hot(label, 9)
    #onehotLabels = tf.argmax(onehotLabels, axis=1)

    lossFunction = tf.nn.softmax_cross_entropy_with_logits(logits=outputLayer, labels=onehotLabels)
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(lossFunction)

    isCorrect = tf.equal(tf.cast(tf.argmax(outputLayer,1), tf.int32), label)
    accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

    # train
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        batchSize = 13
        print("len xtrain %s , %s " % (len(x_train), (int(len(x_train) / batchSize)) ))
        for i in range(1, (int(len(x_train) / batchSize)) + 2):
            batch_x = x_train[(i - 1) * batchSize:i * batchSize]
            batch_y = y_train[(i - 1) * batchSize:i * batchSize]

            train_step.run(feed_dict={input: batch_x, label: batch_y})
            xxx = sess.run([accuracy], feed_dict={input: batch_x, label: batch_y})

            print('accuracy %s ' % xxx)

        prediction = pred.eval(feed_dict={input: x2})
        print('Score %s ' % np.mean(prediction == y2))

        print('done tf %s ' % prediction)


def tensorFlow1():
    print('\n##################\nTF1\n##################')
    tf.set_random_seed(42)
    x_train = x1
    y_train = pd.get_dummies(y1)
    dropoutRate = 0.9

    input = tf.placeholder(tf.float32, [None, x1.shape[1]])
    label = tf.placeholder(tf.int32, [None,9])

    hiddenLayerSizes = [3]

    previousLayer = input
    previousLayerSize = x1.shape[1]

    for layerSize in hiddenLayerSizes:
        w = tf.Variable(tf.random_normal([previousLayerSize, layerSize]))
        b = tf.Variable(tf.random_normal([layerSize]))

        hiddenLayer= tf.matmul(previousLayer, w) + b
        # hiddenLayer = tf.nn.relu(hiddenLayer)
        # hiddenLayer = tf.nn.dropout(hiddenLayer, dropoutRate)
        previousLayerSize = layerSize
        previousLayer = hiddenLayer


    weights = tf.Variable(tf.random_normal([previousLayerSize, 9]))
    outputLayer = tf.matmul(previousLayer, weights)

    lossFunction = tf.nn.softmax_cross_entropy_with_logits(logits=outputLayer, labels=label)
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(lossFunction)

    pred = tf.nn.softmax(outputLayer)

    isCorrect = tf.equal(tf.cast(outputLayer, tf.int32), label)
    accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))


    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        batchSize = 13
        for i in range(1, (int(len(x_train) / batchSize)) + 2):
            batch_x = x_train[(i - 1) * batchSize:i * batchSize]
            batch_y = y_train[(i - 1) * batchSize:i * batchSize]

            train_step.run(feed_dict={input: batch_x, label: batch_y})
            xxx = sess.run([accuracy], feed_dict={input: batch_x, label: batch_y})

            print('accuracy %s ' % xxx)

        prediction = pred.eval(feed_dict={input: x2})
        y2Truth = pd.get_dummies(y2)
        print('Score %s ' % np.mean(prediction == y2Truth))

        print('done tf %s ' % prediction)

# result = pipelineText()
#tensorFlow1()
result = xgboost()
# onehot = pd.get_dummies(result)
# submission = pd.DataFrame()
# submission['ID'] = pid
# submission = submission.join(onehot)
# submission.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
# submission.to_csv('submission_xgb.csv', index=False)

# print('Score %s ' % score)
# submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])

print('done')