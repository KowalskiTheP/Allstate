# May this should be included in the main code... may not...

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
import pandas as pd

def cleanUpString(error):
    e = str(error).split("[")[-1]
    e = e.replace("[", "")
    e = e.replace("]", "")
    e = e.replace(" ", ",")
    e = e.replace("'", "")
    e = e.replace("\n", "")
    e = e.split(",")
    return e
    
training_data = "../Data/train.csv"
test_data = "../Data/test.csv"

dataset_train = pd.read_csv(training_data,delimiter=",", index_col=False, header=0)
dataset_test = pd.read_csv(test_data,delimiter=",", index_col=False, header=0)

for i in range(1,117):
    slist = ['A','B','C']
    try:
            le.fit(slist)
            le.transform(dataset_train.ix[:,i])
    except ValueError as err:
        e = cleanUpString(err)
        for l in range(len(e)):
            slist.append(e[l])
        del e
        try:
            le.fit(slist)
            le.transform(dataset_test.ix[:,i])
        except ValueError as err:
            e = cleanUpString(err)
            for l in range(len(e)):
                slist.append(e[l]) 
            del e
        print len(slist)
        le.fit(slist)
        del slist
 
    dataset_train.ix[:,i] = le.transform(dataset_train.ix[:,i])
    dataset_test.ix[:,i] = le.transform(dataset_test.ix[:,i])

#dataset_train.to_csv("../Data/conv_training.csv", header=False, index=False)
#dataset_test.to_csv("../Data/conv_test.csv", header=False, index=False)
dataset_train.to_csv("../Data/conv_training.csv")
dataset_test.to_csv("../Data/conv_test.csv")
