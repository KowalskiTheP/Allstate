import numpy as np
import pandas as pd
import xgboost as xgb



def calc_MAE(Y, Y_hat):
    sumVar = 0
    for i in range(len(Y)):
        sumVar = sumVar + abs(Y.iloc[i]-Y_hat.iloc[i])
    mae = sumVar / len(Y)
    return mae

#totalSum = 0
#for cvStep in range(1,11):
#  print "CV step: ", cvStep
#  Y_hat_data = "../Pred/subset_pred_lr0.05_Fredgth0.1_crossV_"+str(cvStep)+".csv"
#  Y_data = "../Data/subset_test_crossV_"+str(cvStep)+".csv"
#  Y_hat = pd.read_csv(Y_hat_data,delimiter=" ", header=None, index_col=False)
#  Y = pd.read_csv(Y_data,delimiter=",", header=0, index_col=0)
#  Y_hat = Y_hat[0]
#  Y = Y["loss"]
#  #print Y.iloc[0], Y_hat.iloc[0]
#  #print len(Y)
#  totalSum = totalSum + calc_MAE(Y, Y_hat)
#  print calc_MAE(Y, Y_hat)

#print totalSum/10

dataset_train = pd.read_csv('../Data/train_noOutliers.csv',delimiter=",", header=0)
dataset_train = dataset_train.reindex(np.random.permutation(dataset_train.index))
dataset_train = pd.get_dummies(dataset_train, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True)
dataset.drop(['cat22_B', 'cat55_B', 'cat63_B', 'cat74_C', 'cat75_C', 'cat88_B', 'cat89_E', 'cat89_G', 'cat89_I', 'cat90_D', 'cat90_E', 'cat90_F', 'cat90_G', 'cat91_H', 'cat92_C', 'cat92_D', 'cat92_F', 'cat92_I', 'cat94_E', 'cat94_G', 'cat96_B', 'cat96_F', 'cat97_B', 'cat97_F', 'cat99_G', 'cat99_H', 'cat99_I', 'cat99_O', 'cat99_P', 'cat99_R', 'cat99_T', 'cat100_E', 'cat101_B', 'cat101_E', 'cat101_H', 'cat101_K', 'cat101_N', 'cat102_G', 'cat103_K', 'cat103_N', 'cat104_Q', 'cat105_B', 'cat105_P', 'cat105_Q', 'cat105_R', 'cat105_S', 'cat106_B', 'cat106_O', 'cat106_P', 'cat106_R', 'cat107_B', 'cat107_R', 'cat107_S', 'cat107_U', 'cat108_C', 'cat108_F', 'cat108_J', 'cat109_AA', 'cat109_AE', 'cat109_AF', 'cat109_AG', 'cat109_AH', 'cat109_AJ', 'cat109_AK', 'cat109_AN', 'cat109_AO', 'cat109_AP', 'cat109_AQ', 'cat109_AR', 'cat109_AT', 'cat109_AU', 'cat109_AV', 'cat109_AW', 'cat109_AY', 'cat109_B', 'cat109_BA', 'cat109_BC', 'cat109_BE', 'cat109_BF', 'cat109_BG', 'cat109_BK', 'cat109_BM', 'cat109_BN', 'cat109_BP', 'cat109_BR', 'cat109_BS', 'cat109_BT', 'cat109_BV', 'cat109_BY', 'cat109_CB', 'cat109_CC', 'cat109_CE', 'cat109_CF', 'cat109_CG', 'cat109_CH', 'cat109_CI', 'cat109_CJ', 'cat109_CK', 'cat109_CL', 'cat109_H', 'cat109_J', 'cat109_K', 'cat109_O', 'cat109_P', 'cat109_Q', 'cat109_V', 'cat109_ZZ', 'cat111_B', 'cat111_D', 'cat111_F', 'cat111_Q', 'cat111_S', 'cat111_Y', 'cat112_AQ', 'cat112_AW', 'cat112_B', 'cat113_AB', 'cat113_AC', 'cat113_AL', 'cat113_AP', 'cat113_AQ', 'cat113_AR', 'cat113_B', 'cat113_BB', 'cat113_BI', 'cat113_BL', 'cat113_E', 'cat113_G', 'cat113_O', 'cat113_P', 'cat113_T', 'cat113_U', 'cat114_B', 'cat114_D', 'cat114_G', 'cat114_S', 'cat114_W', 'cat114_X', 'cat115_B', 'cat115_C', 'cat115_D', 'cat115_E', 'cat115_W', 'cat115_X'], axis=1, inplace=True)
dataset_length = len(dataset_train.index)
testSize = int(0.2 * dataset_length)

new_ds = {}
for i in range(1,6):
  new_ds[i] = dataset_train[((i-1)*testSize):(i*testSize)]

new_train = {}
new_test = {}
for j in range(1,6):
    new_test[j] = new_ds[j]
    new_train[j] = pd.DataFrame()
    for i in range(1,11):
        if i != j:
            new_train[j] = new_train[j].append(new_ds[i])
#    new_train[j].to_csv("../Data/subset_training_crossV_"+str(j)+".csv")
#    new_test[j].to_csv("../Data/subset_test_crossV_"+str(j)+".csv")

param = {'max_depth':3, 'learning_rate':0.1, 'n_estimators':1000, 'silent':True, 'objective':'reg:linear', 'nthread':4, 'gamma':0, 'min_child_weight':1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'base_score':0.5, 'seed':42, 'missing':None}


for cv in range(1,6):
    y_train = new_train[cv].pop('loss')
    y_test = new_test[cv].pop('loss')
    xgtrain = xgb.DMatrix(new_train[cv], label=y_train)
    xgtest = xgb.DMatrix(new_test[cv], label=y_test)
    
    bst = xgb.train( param, xgtrain, num_round=1000)#, evallist )
    
#estimator= XGBRegressor(max_depth=3, 
                                     learning_rate=0.1, 
                                     n_estimators=1000, 
                                     silent=True, 
                                     objective='reg:linear', 
                                     nthread=4, 
                                     gamma=0, 
                                     min_child_weight=1, 
                                     max_delta_step=0, 
                                     subsample=1, 
                                     colsample_bytree=1, 
                                     colsample_bylevel=1, 
                                     reg_alpha=0, 
                                     reg_lambda=1, 
                                     scale_pos_weight=1, 
                                     base_score=0.5, 
                                     seed=42, 
                                     missing=None)

