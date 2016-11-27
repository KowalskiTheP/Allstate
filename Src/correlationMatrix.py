import numpy as np 
import pandas as pd 

training_data = "../Data/conv_training.csv"
dataset = pd.read_csv(training_data,delimiter=",", header=0, index_col=0)
print dataset.iloc[0,:]
dataset_npArray = dataset.as_matrix(columns=None)

hightCorr_array = []

covariance_matrix = np.cov(dataset_npArray.T)
correlation_matrix = np.corrcoef(dataset_npArray.T)
corr_array = correlation_matrix[-1]
for i in range(len(corr_array)):
  if abs(corr_array[i]) < 0.01:
    print i,dataset.iloc[0,i], corr_array[i]
print "\n"
for i in range(len(corr_array)):
  if abs(corr_array[i]) > 0.1:
    hightCorr_array.append(i)
    print i,dataset.iloc[0,i] , corr_array[i]

np.savetxt('correlationMatrix.txt',np.corrcoef(dataset_npArray.T), delimiter=',', newline='\n')
#print len(correlation_matrix)
#print correlation_matrix[-1]
print hightCorr_array