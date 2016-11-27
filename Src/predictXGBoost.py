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

for cvStep in range(1,11):
  print "CV step: ", cvStep
  training_data = "../Data/subset_training_crossV_"+str(cvStep)+".csv"

#training_data = "../Data/subset_test_paramOpt0.csv"

#training_data = "../Data/conv_training.csv"
#test_data = "../Data/conv_test.csv"
  test_data = "../Data/subset_test_crossV_"+str(cvStep)+".csv"

  dataset = np.loadtxt(training_data, delimiter=",",skiprows=1)
  
  trainSet_pd = pd.read_csv(training_data,delimiter=",", header=0, index_col=0)
  #dataset_pd.drop(["id","cat21","cat22","cat27","cat54","cat63","cat70","cat86","cat88","cat92","cat93","cat97","cat107","cat108","cat116","cont13","loss"],axis=1, inplace=True,)
  trainSet_pd = trainSet_pd.iloc[:,[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 23, 28, 36, 40, 50, 57, 72, 73, 79, 80, 81, 82, 87, 89, 90, 101, 103, 105, 111, 118, 119, 123]]

  #X_train = dataset[:,1:132]
  X_train = trainSet_pd
  dataset = np.loadtxt(training_data, delimiter=",",skiprows=1)
  Y_train = dataset[:,132]
  #print X_train[-1]

  #dataset = np.loadtxt(test_data, delimiter=",",skiprows=1)
  #X_test = dataset[:,1:132]
  testset_pd = pd.read_csv(test_data,delimiter=",", header=0, index_col=0)
  #testset_pd.drop(["id","cat21","cat22","cat27","cat54","cat63","cat70","cat86","cat88","cat92","cat93","cat97","cat107","cat108","cat116","cont13","loss"],axis=1, inplace=True,)
  testset_pd = testset_pd.iloc[:,[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 23, 28, 36, 40, 50, 57, 72, 73, 79, 80, 81, 82, 87, 89, 90, 101, 103, 105, 111, 118, 119, 123]]
  X_test = testset_pd


  xg_train = xgb.DMatrix( X_train, label=Y_train)#, weight=weights)
  xg_test = xgb.DMatrix(X_test)
  
  param = np.load('../Pred/lrate_0.05_iterStep_3_mae_Fredgth0.1.npy').item()
  print param 
  param['silent']=True
  param['eval_metric']='mae'

  xgb3 = xgb.train(param, xg_train, 400, verbose_eval=50 )
  predict = xgb3.predict( xg_test )

  rounded_prediction = np.round(predict)
  np.savetxt("../Pred/subset_pred_lr0.05_Fredgth0.1_crossV_"+str(cvStep)+".csv", np.column_stack((predict, rounded_prediction)), fmt=["%f"] + ["%i"], delimiter=' ')















