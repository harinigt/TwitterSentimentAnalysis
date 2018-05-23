import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

def convertToNpArray(train,test):
    """
    Converts the data into numpy arrays
    :param train: training data csv path
    :param test: test data csv path
    :return: training data and labels, test data and labels
    """
    train_data = pd.read_csv(train,delimiter=',', quotechar='"',
                             dtype=None,encoding = "ISO-8859-1",
                             usecols=[0,5])
    train_target = train_data.iloc[:,0]
    train_target_array = np.array(train_target)
    train_target_array = np.reshape(train_target_array,(len(train_target_array),1))
    train_data = train_data.iloc[:,1]
    train_data_array = np.array(train_data)
    train_data_array = np.reshape(train_data_array,(len(train_data_array),1))

    test_data = pd.read_csv(test,delimiter=',', quotechar='"',
                             dtype=None,encoding = "ISO-8859-1",
                            usecols=[0,5], names=['label','tweet'])
    test_data = test_data[test_data.label != 2]
    test_data = test_data.values
    test_target = test_data[:,0]
    test_target_array = np.array(test_target)
    test_target_array = np.reshape(test_target_array, (len(test_target_array), 1))
    test_data = test_data[:,1]
    test_data_array = np.reshape(test_data, (len(test_data), 1))

    return train_data_array,train_target_array,test_data_array,test_target_array

def remove_punc(data_array):
    translator = str.maketrans(string.punctuation, len(string.punctuation)*' ')
    for i in range(len(data_array)):
        data_array[i][0] = data_array[i][0].translate(translator)
    return data_array
#end

def remove_stopwords(data_array):
    print()
#end

if __name__=="__main__":
    train_data_array, train_target_array, test_data_array,test_target_array=convertToNpArray\
        ('data/training.1600000.processed.noemoticon.csv','data/testdata.manual.2009.06.14.csv')
    #Remove punctuations from train and test
    train_data_array = remove_punc(train_data_array)
    test_data_array = remove_punc(test_data_array)

    #Remove stop words
    train_data_array = remove_stopwords(train_data_array)

    print(train_data_array)
    print(np.shape(train_data_array))
    print(np.shape(train_target_array))
    print(np.shape(test_data_array))
    print(np.shape(test_target_array))
