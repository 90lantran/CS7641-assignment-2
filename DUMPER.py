# -*- coding: utf-8 -*-
"""
Run with Python 3.x (not Jython)
"""


import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from loaders import load_adult


df = load_adult(preprocess=True)
X = df[[c for c in df.columns if c != 'target']]
y = df['target']



X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# TODO - wonder why he does all this "Cull" stuff?
# pipe = Pipeline([('Scale',StandardScaler()),
#                  ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                  ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                  ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                  ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),])
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)
Y_train_scaled = np.atleast_2d(y_train).T
X_test_scaled = scaler.transform(X_test)
y_test_scaled = np.atleast_2d(y_test).T
X_train_scaled, X_val_scaled, Y_train_scaled, y_val_scaled = ms.train_test_split(X_train_scaled, Y_train_scaled, test_size=0.2, random_state=1, stratify=Y_train_scaled)
tst = pd.DataFrame(np.hstack((X_test_scaled, y_test_scaled)))
trg = pd.DataFrame(np.hstack((X_train_scaled, Y_train_scaled)))
val = pd.DataFrame(np.hstack((X_val_scaled, y_val_scaled)))
tst.to_csv('m_test.csv',index=False,header=False)
trg.to_csv('m_trg.csv',index=False,header=False)
val.to_csv('m_val.csv',index=False,header=False)