import pandas as pd
import numpy as np
from sklearn import *
import sklearn
from xgboost import XGBClassifier
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV

train = pd.read_csv('data\\training_variants')
trainingText = pd.read_csv('data\\training_text', sep="\|\|", engine='python', header=None, names=["ID","Text"], skiprows=1)

test = pd.read_csv('data\\test_variants')
testText = pd.read_csv('data\\test_text', sep="\|\|", engine='python', header=None, names=["ID","Text"], skiprows=1)
pid = test['ID'].values

train = train.merge(trainingText, on='ID', how='left')

y = train['Class']
train = train.drop('Class', axis = 1)

x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.2)

print('\n##################\nXGBoost\n##################')

param = {}
# param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
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


score = metrics.log_loss(y2, prediction)
print('Score %s ' % score)


# submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
# submission['ID'] = pid
# submission.to_csv('submission_xgb.csv', index=False)

print('done')