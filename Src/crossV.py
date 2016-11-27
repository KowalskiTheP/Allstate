import pandas as pd
import random
import numpy as np

training_data = "../Data/conv_training.csv"
dataset = pd.read_csv(training_data,delimiter=",", header=0, index_col=0)
dataset = dataset.reindex(np.random.permutation(dataset.index))
dataset_length = len(dataset.index)
testSize = int(0.1 * dataset_length)

new_ds = {}
for i in range(1,11):
  new_ds[i] = dataset[((i-1)*testSize):(i*testSize)]

new_train = {}
new_test = {}
for j in range(1,11):
  new_test[j] = new_ds[j]
  new_train[j] = pd.DataFrame()
  for i in range(1,11):
    if i != j:
      new_train[j] = new_train[j].append(new_ds[i])
  new_train[j].to_csv("../Data/subset_training_crossV_"+str(j)+".csv")
  new_test[j].to_csv("../Data/subset_test_crossV_"+str(j)+".csv")
  
