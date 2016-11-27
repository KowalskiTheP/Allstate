import pandas as pd
import random

#training_data = "../Data/conv_training.csv"
#dataset = pd.read_csv(training_data,delimiter=",", header=0, index_col=0)
#dataset_length = len(dataset.index)
#testSize = int(0.001 * dataset_length)

for l in range(1):
  training_data = "../Data/conv_training.csv"
  dataset = pd.read_csv(training_data,delimiter=",", header=0, index_col=0)
  dataset_length = len(dataset.index)
  testSize = int(0.01 * dataset_length)
  random_List = []
  for i in range(testSize):
    k = random.randint(0,dataset_length)
    while k in random_List:
      print k
      k = random.randint(0,dataset_length)
    random_List.append(k)


  subset_test = dataset.iloc[[random_List[0]]]
  for j in range(1,testSize):
    subset_test = subset_test.append(dataset.iloc[[random_List[j]]])
    
  for j in range(testSize):
    dataset.drop([random_List[j]],axis=0,inplace=True)

  dataset.to_csv("../Data/subset_training_paramOpt"+str(l)+".csv")
  subset_test.to_csv("../Data/subset_test_paramOpt"+str(l)+".csv")

