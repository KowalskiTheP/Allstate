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

training_data = "../Data/subset_training1.csv"

#training_data = "../Data/subset_test_paramOpt0.csv"

#training_data = "../Data/conv_training.csv"
#test_data = "../Data/conv_test.csv"
test_data = "../Data/subset_test1.csv"

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

#xgb3 = XGBRegressor(learning_rate =0.1, reg_alpha = 3e-05, reg_lambda = 3 , n_estimators=300, max_depth=5, objective= 'reg:linear', nthread=2, gamma=0.0, subsample=0.555, colsample_bytree=0.55, scale_pos_weight=3, min_child_weight=5, seed=43)

#xgb3.fit(X, y, eval_set=[(X_eval, y_eval)], early_stopping_rounds=50)
#predict = xgb3.predict(X_test)

#param_test1 = {
 #'max_depth':range(1,2,1),           #1,6,1
 #'min_child_weight':range(1,2,1)     #1,6,1
#}

#param_test2 = {
 #'gamma':[i/30.0 for i in range(0,1)]
#}

#param_test3 = {
 #'subsample':[i/30.0 for i in range(5,30)],
 #'colsample_bytree':[i/30.0 for i in range(5,30)]
#}

##param_test4 = {
 ##'subsample':[i/300.0 for i in range(40,80,5)],
 ##'colsample_bytree':[i/300.0 for i in range(50,80,5)]
##}

#param_test5 = {
 #'reg_alpha':[3e-5, 3e-3, 0.3, 3, 300],
 #'reg_lambda':[3e-5, 3e-3, 0.3, 3, 300]
#}

#paramTuningDict = {
    #'learning_rate' : 0.1, 
    #'reg_alpha' :3e-05, 
    #'reg_lambda' :3 , 
    #'n_estimators' : 300, 
    #'max_depth':5, 
    #'objective' : 'reg:linear', 
    #'nthread' : 2, 
    #'gamma' : 0.1, 
    #'subsample' : 0.555, 
    #'colsample_bytree' : 0.55, 
    #'scale_pos_weight' : 3, 
    #'min_child_weight' : 5, 
    #'seed' : 42
    #}

##xgb3 = XGBRegressor(paramTuningDict)

#print "huuhu"
#for hypParam in range(1, 6):
  #print hypParam
  #if hypParam == 1:
    #xgb1 = XGBRegressor(paramTuningDict)
    #print 'xgb1'
    #gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=2,iid=False, cv=5)
    #print 'gsearch1'
    #gsearch1.fit(X,y)
    #print 'fit1'
    #print gsearch1.best_params_
    #paramTuningDict['max_depth'] = gsearch1.best_params_['max_depth']
    #paramTuningDict['min_child_weight'] = gsearch1.best_params_['min_child_weight']
    #with open('../Pred/hyperParamTuning.txt', 'a') as paramFile:
      #paramFile.write(str(gsearch1.best_params_.items())+"\n")

  #if hypParam == 2:
    #xgb2 = XGBRegressor(paramTuningDict)
    #print 'xgb2'
    #gsearch2 = GridSearchCV(estimator = xgb2, param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=2,iid=False, cv=5)
    #print 'gsearch2'
    #gsearch2.fit(X,y)
    #print 'fit2'
    #print gsearch2.best_params_
    #paramTuningDict['gamma'] = gsearch2.best_params_['gamma']
    #with open('../Pred/hyperParamTuning.txt', 'a') as paramFile:
      #paramFile.write(str(gsearch2.best_params_.items())+"\n")

  #if hypParam == 3:
    #xgb3 = XGBRegressor(paramTuningDict)
    #gsearch3 = GridSearchCV(estimator = xgb3, param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    #gsearch3.fit(X,y)
    #print gsearch3.best_params_
    #paramTuningDict['subsample'] = gsearch3.best_params_['subsample']
    #paramTuningDict['colsample_bytree'] = gsearch3.best_params_['colsample_bytree']
    #with open('../Pred/hyperParamTuning.txt', 'a') as paramFile:
      #paramFile.write(str(gsearch3.best_params_.items())+"\n")

  ##if hypParam == 4:
    ##xgb3 = XGBRegressor(paramTuningDict)
    ##gsearch3 = GridSearchCV(estimator = xgb3, param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    ##gsearch3.fit(X,y)
    ##print gsearch3.best_params_
    ##paramTuningDict['gamma'] = gsearch3.best_params_['gamma']
    ##with open('../Pred/hyperParamTuning.txt', 'a') as paramFile:
      ##paramFile.write(str(gsearch3.best_params_.items())+"\n")

  #if hypParam == 5:
    #xgb5 = XGBRegressor(paramTuningDict)
    #gsearch5 = GridSearchCV(estimator = xgb5, param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    #gsearch5.fit(X,y)
    #print gsearch5.best_params_
    #paramTuningDict['reg_alpha'] = gsearch5.best_params_['reg_alpha']
    #paramTuningDict['reg_lambda'] = gsearch5.best_params_['reg_lambda']
    #with open('../Pred/hyperParamTuning.txt', 'a') as paramFile:
      #paramFile.write(str(gsearch5.best_params_.items())+"\n")
  
  #print hypParam

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
  
param = np.load('../Pred/lrate_0.05_iterStep_3_mae.npy').item()
print param 
param['silent']=True
param['eval_metric']='mae'

xgb3 = xgb.train(param, xg_train, 400, verbose_eval=50 )
#xgb3 = xgb.cv(param, xg_train, num_boost_round=800, nfold=3, early_stopping_rounds=80, seed=43)
#print xgb3
#np.savetxt("/local_fat/florian/bundesliga/predictions/tests3.csv", xgb3, delimiter=' ')
predict = xgb3.predict( xg_test )

#print predict

rounded_prediction = np.round(predict)
np.savetxt("../Pred/firstTest.csv", np.column_stack((predict, rounded_prediction)), fmt=["%f"] + ["%i"], delimiter=' ')

#.............................

#np.savetxt("/home/yannic/Documents/AI/bundesliga/predictions/seeds_xg/"+home_or_away+"_"+jbos+"_rounded.csv", rounded_prediction, fmt='%i', delimiter=' ')

#print X_train[0]
#print Y_train[0]
#print X_test[0]














