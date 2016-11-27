#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
#%matplotlib inline
#from IPython import get_ipython
#get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('../Data/subset_test_paramOpt0.csv',delimiter=",", header=0, index_col=0)
target = 'loss'
IDcol = 'id'


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=2, early_stopping_rounds=50):
    
    if useTrainCV:
        print "is in if"
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='mae', early_stopping_rounds=early_stopping_rounds)#, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    print dtrain[predictors]
    alg.fit(dtrain[predictors], dtrain['loss'],eval_metric='mae')
    print "Fitted"
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    print "Predicted"
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    print "Probs"
    
    #Print model report:
    print "\nModel Report"
    print "Explained variance: %.4g" % metrics.explained_variance_score(dtrain['loss'].values, dtrain_predictions)
    print "MAE (Train): %f" % metrics.mean_absolute_error(dtrain['loss'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


    
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]

#print predictors
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=300,
 max_depth=3,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
print "Classifier"
modelfit(xgb1, train, predictors)

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(
    estimator = XGBClassifier( 
        learning_rate =0.1, 
        n_estimators=140, 
        max_depth=5,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
    param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=1,iid=False, cv=3)
print gsearch1.fit(train[predictors],train[target])
print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


