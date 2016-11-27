import numpy as np
import pandas as pd

#regular python imports
import os
import sys
import h5py
import json
import time
from ConfigParser import SafeConfigParser
#local imports

rand_seed = 42
np.random.seed(rand_seed)

#xgboost import
import xgboost as xgb
import scipy.sparse
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
#scikit learn imports

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#training_data = "../Data/subset_training0.csv"
training_data = "../Data/subset_test_paramOpt0.csv"
#training_data = "../Data/conv_training.csv"
#test_data = "../Data/conv_test.csv"
test_data = "../Data/subset_test0.csv"

dataset = np.loadtxt(training_data, delimiter=",",skiprows=1)

#X_train = dataset[:,1:10]
X_train = dataset[:,1:132]
#print X_train[0]
#print X_train
Y_train = dataset[:,132]
#print Y_train[0]
#weights_raw = dataset[:,0]
#print weights_raw
#print weights_raw.min()
#print weights_raw.max()
#weights = (weights_raw - weights_raw.min())/(weights_raw.max()-weights_raw.min())
#print weights



dataset = np.loadtxt(test_data, delimiter=",",skiprows=1)

#X_test = dataset[:,0:9]
X_test = dataset[:,1:132]
#print X_test[0]
#Ab hier bis xgb.DMatrix muesste sinnfrei sein...

#dataset = np.loadtxt(test_verify) #von mir auskommentiert

#Y_test = dataset[:,0]

#max_classes = int(np.max(Y_test)+3)
##X_train = csr.todense()
##X_test = csr.todense()

##print X_train[0,:]

## this is prediction
xg_train = xgb.DMatrix( X_train, label=Y_train)#, weight=weights)
xg_test = xgb.DMatrix(X_test)


X = X_train
y = Y_train
X, X_eval, y, y_eval= train_test_split(X_train, Y_train, test_size=0.3, random_state=3)

xgb3 = XGBRegressor(learning_rate =0.03, reg_alpha = 3e-05, reg_lambda = 3 , n_estimators=300, max_depth=5, objective= 'reg:linear', nthread=2, gamma=0.0, subsample=0.555, colsample_bytree=0.55, scale_pos_weight=3, min_child_weight=5, seed=43)

#xgb3.fit(X, y, eval_set=[(X_eval, y_eval)], early_stopping_rounds=50)
#predict = xgb3.predict(X_test)

#param_test3 = {
 #'max_depth':range(1,6,1),
 #'min_child_weight':range(1,6,1)
#}

#param_test3 = {
 #'gamma':[i/30.0 for i in range(0,5)]
#}

param_test4 = {
 'subsample':[i/30.0 for i in range(5,30)],
 'colsample_bytree':[i/30.0 for i in range(5,30)]
}

#param_test5 = {
 #'subsample':[i/300.0 for i in range(40,80,5)],
 #'colsample_bytree':[i/300.0 for i in range(50,80,5)]
#}
#param_test6 = {
 #'reg_alpha':[3e-5, 3e-3, 0.3, 3, 300],
 #'reg_lambda':[3e-5, 3e-3, 0.3, 3, 300]
#}

print "huuhu"
gsearch3 = GridSearchCV(estimator = xgb3, param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X,y)
print gsearch3.grid_scores_
print gsearch3.best_params_
print gsearch3.best_score_

#.........................................

#param = {
  ##'learning_rate':0.125,
  #'learning_rate':0.05,
  #'reg_alpha':0,
  #'reg_lambda':1,
  #'n_estimators':300,
  #'max_depth':2,
  #'objective':'reg:linear',
  #'nthread':10,
  ##gamma':0.5,
  #'gamma':0.5,
  ##'subsample':0.65,
  #'subsample':0.65,
  ##'colsample_bytree':0.9,
  #'colsample_bytree':0.5,
  #'scale_pos_weight':0,
  #'min_child_weight':0.5,
  ##'seed':BBBB155,
  #'seed':42,
  ##'eval_metric':'rmse',
  #'eval_metric':'mae',
  ##'silent':True
  #'silent':True
  #}

#xgb3 = xgb.train(param, xg_train, 400, verbose_eval=50 )
##xgb3 = xgb.cv(param, xg_train, num_boost_round=800, nfold=3, early_stopping_rounds=80, seed=43)
##print xgb3
##np.savetxt("/local_fat/florian/bundesliga/predictions/tests3.csv", xgb3, delimiter=' ')
#predict = xgb3.predict( xg_test )

##print predict

#rounded_prediction = np.round(predict)
#np.savetxt("../Pred/firstTest.csv", np.column_stack((predict, rounded_prediction)), fmt=["%f"] + ["%i"], delimiter=' ')

#.............................

#np.savetxt("/home/yannic/Documents/AI/bundesliga/predictions/seeds_xg/"+home_or_away+"_"+jbos+"_rounded.csv", rounded_prediction, fmt='%i', delimiter=' ')

#print X_train[0]
#print Y_train[0]
#print X_test[0]














