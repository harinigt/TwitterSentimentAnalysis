import numpy as np
import pandas as pd


def convertToNpArray(train,test):
    train_data = pd.read_csv(train,delimiter=',', quotechar='"',
                             dtype=None,encoding = "ISO-8859-1",
                             usecols=['Sentiment','SentimentText'])
    train_target = train_data[['Sentiment']]
    train_target_array = np.array(train_target)
    train_data = train_data[['SentimentText']]

    train_data_array = np.array(train_data)
    test_data = pd.read_csv(test,delimiter=',', quotechar='"',
                             dtype=None,encoding = "ISO-8859-1",
                            usecols=['SentimentText'])

    test_data_array = np.array(test_data)
    return train_data_array,train_target_array,test_data_array


if __name__=="__main__":
    train_data_array, train_target_array, test_data_array=convertToNpArray('data/train.csv','data/test.csv')
