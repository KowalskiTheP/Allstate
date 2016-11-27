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

#training_data = "../Data/subset_test_paramOpt0.csv"
#test_data = "../Data/subset_test0.csv"
training_data = "../Data/subset_training_crossV_1.csv"
test_data = "../Data/subset_test_crossV_1.csv"

trainSet_pd = pd.read_csv(training_data,delimiter=",", header=0, index_col=0)

#testSet_pd = pd.read_csv(test_data,delimiter=",", header=0, index_col=0)

trainSet_pd.drop(["id","cat21","cat22","cat27","cat54","cat63","cat70","cat86","cat88","cat92","cat93","cat97","cat107","cat108","cont13","loss"],axis=1, inplace=True,)
#trainSet_pd = trainSet_pd.iloc[:,[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 23, 28, 36, 40, 50, 57, 72, 73, 79, 80, 81, 82, 87, 89, 90, 101, 103, 105, 111, 118, 119, 123]]

#dataset = np.loadtxt(training_data, delimiter=",",skiprows=1)

#X_train = dataset[:,1:132]
X_train = trainSet_pd
print X_train.iloc[0,-1]
dataset = np.loadtxt(training_data, delimiter=",",skiprows=1)
Y_train = dataset[:,132]
print Y_train[0]

#X_test = dataset[:,1:132]

xg_train = xgb.DMatrix( X_train, label=Y_train)#, weight=weights)
#xg_test = xgb.DMatrix(X_test)


X = X_train
y = Y_train
X, X_eval, y, y_eval= train_test_split(X_train, Y_train, test_size=0.3, random_state=3)

#for i in range(1,10,2):
    #print i

param_test1 = {
 'max_depth':range(1,10,1),           #1,6,1
 'min_child_weight':range(1,10,1)     #1,6,1
}

param_test2 = {
 'gamma':[i/30.0 for i in range(0,30,1)]           #0,10
}

param_test3 = {
 'subsample':[i/30.0 for i in range(15,30,1)],           #5,30
 'colsample_bytree':[i/30.0 for i in range(15,30,1)]     #5,30
}

#param_test4 = {
 #'subsample':[i/300.0 for i in range(40,80,5)],
 #'colsample_bytree':[i/300.0 for i in range(50,80,5)]
#}

param_test5 = {                                         
 'reg_alpha':[3e-5, 3e-4, 3e-3, 3e-2, 0.3, 3, 30, 300],                 #3e-5, 3e-3, 0.3, 3, 300
 'reg_lambda':[3e-5, 3e-4, 3e-3, 3e-2, 0.3, 3, 30, 300]                 #3e-5, 3e-3, 0.3, 3, 300
}

paramTuningDict = {
    'learning_rate' : 0.05, 
    'reg_alpha' : 3, 
    'reg_lambda' : 3 , 
    'n_estimators' : 1000, 
    'max_depth':5, 
    'objective' : 'reg:linear', 
    'nthread' : 4, 
    'gamma' : 0.0, 
    'subsample' : 0.8, 
    'colsample_bytree' : 0.8, 
    'scale_pos_weight' : 1, 
    'min_child_weight' : 1, 
    'seed' : 42
    }



#xgb3 = XGBRegressor(paramTuningDict)

rateOfLearning = 0.05

try:
    for iterStep in range(1,4):
      print "Iteration step: ", iterStep
      if iterStep != 1:
        paramTuningDict = np.load('../Pred/lrate_'+str(rateOfLearning)+'_iterStep_'+str(iterStep-1)+'_mae.npy').item()
      paramTuningDict['learning_rate']=rateOfLearning
      for hypParam in range(1, 6):
        if hypParam == 1:
          xgb1 = XGBRegressor(learning_rate=paramTuningDict['learning_rate'],
                              reg_alpha=paramTuningDict['reg_alpha'],
                              reg_lambda=paramTuningDict['reg_lambda'],
                              n_estimators=paramTuningDict['n_estimators'],
                              max_depth=paramTuningDict['max_depth'],
                              objective=paramTuningDict['objective'],
                              nthread=paramTuningDict['nthread'],
                              gamma=paramTuningDict['gamma'],
                              subsample=paramTuningDict['subsample'], 
                              colsample_bytree=paramTuningDict['colsample_bytree'],
                              scale_pos_weight=paramTuningDict['scale_pos_weight'],
                              min_child_weight=paramTuningDict['min_child_weight'],
                              seed=paramTuningDict['seed'])
          #gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=1,iid=False, cv=5)
          gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=1,iid=False, cv=5)
          gsearch1.fit(X,y)
          print gsearch1.best_params_, "\n"
          paramTuningDict['max_depth'] = gsearch1.best_params_['max_depth']
          paramTuningDict['min_child_weight'] = gsearch1.best_params_['min_child_weight']
          #with open('../Pred/iterStep_'+str(iterStep)+'.txt', 'a') as paramFile:
            #paramFile.write(str(gsearch1.best_params_.items())+"\n")

        if hypParam == 2:
          print "Previously 'max_depth' set to: ", paramTuningDict['max_depth']
          print "Previously 'min_child_weight' set to: ", paramTuningDict['min_child_weight']
          xgb2 = XGBRegressor(learning_rate=paramTuningDict['learning_rate'],
                              reg_alpha=paramTuningDict['reg_alpha'],
                              reg_lambda=paramTuningDict['reg_lambda'],
                              n_estimators=paramTuningDict['n_estimators'],
                              max_depth=paramTuningDict['max_depth'],
                              objective=paramTuningDict['objective'],
                              nthread=paramTuningDict['nthread'],
                              gamma=paramTuningDict['gamma'],
                              subsample=paramTuningDict['subsample'], 
                              colsample_bytree=paramTuningDict['colsample_bytree'],
                              scale_pos_weight=paramTuningDict['scale_pos_weight'],
                              min_child_weight=paramTuningDict['min_child_weight'],
                              seed=paramTuningDict['seed'])
          gsearch2 = GridSearchCV(estimator = xgb2, param_grid = param_test2, scoring='neg_mean_absolute_error',n_jobs=1,iid=False, cv=5)
          gsearch2.fit(X,y)
          print gsearch2.best_params_, "\n"
          paramTuningDict['gamma'] = gsearch2.best_params_['gamma']
          #with open('../Pred/iterStep_'+str(iterStep)+'.txt', 'a') as paramFile:
            #paramFile.write(str(gsearch2.best_params_.items())+"\n")

        if hypParam == 3:
          print "Previously 'gamma' set to: ", paramTuningDict['gamma']
          xgb3 = XGBRegressor(learning_rate=paramTuningDict['learning_rate'],
                              reg_alpha=paramTuningDict['reg_alpha'],
                              reg_lambda=paramTuningDict['reg_lambda'],
                              n_estimators=paramTuningDict['n_estimators'],
                              max_depth=paramTuningDict['max_depth'],
                              objective=paramTuningDict['objective'],
                              nthread=paramTuningDict['nthread'],
                              gamma=paramTuningDict['gamma'],
                              subsample=paramTuningDict['subsample'], 
                              colsample_bytree=paramTuningDict['colsample_bytree'],
                              scale_pos_weight=paramTuningDict['scale_pos_weight'],
                              min_child_weight=paramTuningDict['min_child_weight'],
                              seed=paramTuningDict['seed'])
          gsearch3 = GridSearchCV(estimator = xgb3, param_grid = param_test3, scoring='neg_mean_absolute_error',n_jobs=1,iid=False, cv=5)
          gsearch3.fit(X,y)
          print gsearch3.best_params_, "\n"
          paramTuningDict['subsample'] = gsearch3.best_params_['subsample']
          paramTuningDict['colsample_bytree'] = gsearch3.best_params_['colsample_bytree']
          #with open('../Pred/iterStep_'+str(iterStep)+'.txt', 'a') as paramFile:
            #paramFile.write(str(gsearch3.best_params_.items())+"\n")

        #if hypParam == 4:
          #xgb3 = XGBRegressor(paramTuningDict)
          #gsearch3 = GridSearchCV(estimator = xgb3, param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
          #gsearch3.fit(X,y)
          #print gsearch3.best_params_
          #paramTuningDict['gamma'] = gsearch3.best_params_['gamma']
          #with open('../Pred/hyperParamTuning.txt', 'a') as paramFile:
            #paramFile.write(str(gsearch3.best_params_.items())+"\n")

        if hypParam == 5:
          print "Previously 'subsample' set to: ", paramTuningDict['subsample']
          print "Previously 'colsample_bytree' set to: ", paramTuningDict['colsample_bytree']
          xgb5 = XGBRegressor(learning_rate=paramTuningDict['learning_rate'],
                              reg_alpha=paramTuningDict['reg_alpha'],
                              reg_lambda=paramTuningDict['reg_lambda'],
                              n_estimators=paramTuningDict['n_estimators'],
                              max_depth=paramTuningDict['max_depth'],
                              objective=paramTuningDict['objective'],
                              nthread=paramTuningDict['nthread'],
                              gamma=paramTuningDict['gamma'],
                              subsample=paramTuningDict['subsample'], 
                              colsample_bytree=paramTuningDict['colsample_bytree'],
                              scale_pos_weight=paramTuningDict['scale_pos_weight'],
                              min_child_weight=paramTuningDict['min_child_weight'],
                              seed=paramTuningDict['seed'])
          gsearch5 = GridSearchCV(estimator = xgb5, param_grid = param_test5, scoring='neg_mean_absolute_error',n_jobs=1,iid=False, cv=5)
          gsearch5.fit(X,y)
          print gsearch5.best_params_, "\n"
          paramTuningDict['reg_alpha'] = gsearch5.best_params_['reg_alpha']
          paramTuningDict['reg_lambda'] = gsearch5.best_params_['reg_lambda']
          #with open('../Pred/iterStep_'+str(iterStep)+'.txt', 'a') as paramFile:
            #paramFile.write(str(gsearch5.best_params_.items())+"\n")
      
      with open('../Pred/lrate_'+str(rateOfLearning)+'_iterStep_'+str(iterStep)+'_mae_Fredlth0.01.txt', 'w') as paramFile:
        for p in paramTuningDict.items():
            paramFile.write("%s:%s\n" % p)
      np.save('../Pred/lrate_'+str(rateOfLearning)+'_iterStep_'+str(iterStep)+'_mae_Fredlth0.01.npy', paramTuningDict)    
          
except KeyboardInterrupt:
    print 'interrupted!'



