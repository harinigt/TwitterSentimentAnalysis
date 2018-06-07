import numpy as np
import pandas as pd
import string
import operator

global_dict = {}

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
    train_array = create_train_data_subset(train_data)
    np.random.shuffle(train_array)
    # print(np.shape(train_array))
    train_target_array = train_array[:,0]
    train_target_array = np.reshape(train_target_array,(len(train_target_array),1))
    train_data_array = train_array[:,1]
    train_data_array = np.reshape(train_data_array,(len(train_data_array),1))
    test_data = pd.read_csv(test,delimiter=',', quotechar='"',
                             dtype=None,encoding = "ISO-8859-1",
                            usecols=[0,5], names=['label','tweet'])
    test_data = test_data[test_data.label != 2]
    test_data = test_data.values
    test_data = np.append(test_data,create_test_data_subset(train_data),axis=0)
    np.random.shuffle(test_data)
    test_target = test_data[:,0]
    test_target_array = np.array(test_target)
    test_target_array = np.reshape(test_target_array, (len(test_target_array), 1))
    test_data = test_data[:,1]
    test_data_array = np.reshape(test_data, (len(test_data), 1))

    return train_data_array,train_target_array,test_data_array,test_target_array

def create_train_data_subset(train_data):
    train_data_numpy_array = np.array(train_data)
    train_data_final = train_data_numpy_array[750000:850000,:]
    return train_data_final

def create_test_data_subset(train_data):
    train_data_numpy_array = np.array(train_data)
    test_data_final = train_data_numpy_array[0:10000, :]

    test_data_final = np.append(test_data_final,train_data_numpy_array[900000:910000,:],axis=0)
    # print(np.shape(test_data_final))
    return test_data_final


def remove_punc(data_array):
    """

    :param data_array:
    :return:
    """
    translator = str.maketrans(string.punctuation, len(string.punctuation)*' ')
    for i in range(len(data_array)):
        data_array[i][0] = data_array[i][0].translate(translator)
    return data_array
#end

def remove_stopwords(data_array,stopwords_file_path):
    """

    :param data_array:
    :param stopwords_file_path:
    :return:
    """
    stopwords = open(stopwords_file_path,'r')
    stopwords_list = stopwords.read().split('\n')
    for i in range(len(data_array)):
        tweet_tokenized = data_array[i][0].split(' ')
        tweet_tokenized = [word.lower() for word in tweet_tokenized]
        for word in tweet_tokenized:
            if word in stopwords_list:
                tweet_tokenized.remove(word)
        data_array[i][0] = ' '.join(tweet_tokenized)
    return data_array


#end


def build_global_vocab():
    """

    :param train_data_array:
    :return:
    """
    global features
    global train_data_array
    for i in range(len(train_data_array)):
        tweet_tokenized = train_data_array[i][0].split(' ')
        for word in tweet_tokenized:
            if word in global_dict.keys():
                global_dict[word] +=1
            else:
                global_dict[word] = 1
    global_dict.pop('')
    features = dict(sorted(global_dict.items(), key=operator.itemgetter(1),reverse=True)[:2000])

def encodeDataArray():
    global features
    global test_data_array
    global test_encoded_array

    top_2000_word_list = list(features.keys())
    test_encoded_array = np.empty((len(test_data_array),len(top_2000_word_list)))
    for i in range(len(test_data_array)):
        # encoded_array = np.append(encoded_array,(1,1))
        for j in range(len(top_2000_word_list)):
            if top_2000_word_list[j] in test_data_array[i][0]:
                test_encoded_array[i][j] = 1
            else:
                test_encoded_array[i][j] = 0
    return test_encoded_array

def encodeTrainDataArray():
    global features
    global train_data_array
    global train_encoded_array
    training_length = len(train_data_array)
    top_2000_word_list = list(features.keys())
    word_list_length = len(top_2000_word_list)
    train_encoded_array = np.zeros((training_length,word_list_length))
    for i in range(training_length):
        # encoded_array = np.append(encoded_array,(1,1))
        for j in range(word_list_length):
            if top_2000_word_list[j] in train_data_array[i][0]:
                train_encoded_array[i][j] = 1


if __name__=="__main__":
    global features
    global train_data_array
    global test_data_array
    global test_encoded_array
    global train_encoded_array
    # np.set_printoptions(suppress=True)

    train_data_array, train_target_array, test_data_array,test_target_array=convertToNpArray('data/training.1600000.processed.noemoticon.csv','data/testdata.manual.2009.06.14.csv')
    np.save('data/train_target_array_new', train_target_array)
    np.save('data/test_target_array_new', test_target_array)
    #Round 1 - Remove stop words
    train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')
    test_data_array = remove_stopwords(test_data_array, 'stopwords.txt')
    # Remove punctuations from train and test
    train_data_array = remove_punc(train_data_array)
    test_data_array = remove_punc(test_data_array)

    # Round 2 - Remove stop words
    train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')
    test_data_array = remove_stopwords(test_data_array, 'stopwords.txt')

    #Build top 2000 words from training data array
    build_global_vocab()

    #Encode the training and test data
    # train_encoded_array = encodeDataArray(train_data_array)
    # test_encoded_array = encodeDataArray()
    # print(np.sum(test_encoded_array))
    encodeDataArray()
    encodeTrainDataArray()


    # np.save('data/train_encoded_array.npy',train_encoded_array)
    np.save('data/train_encoded_array_new', train_encoded_array)
    np.save('data/test_encoded_array_new', test_encoded_array)



    # print(test_encoded_array)

    # print(global_dict)
    # print(features)
    # test_data_array = remove_stopwords(test_data_array, 'stopwords.txt')
    # print(train_data_array)
    # print(np.shape(train_data_array))
    # print(np.shape(train_target_array))
    # print(np.shape(test_data_array))
    # print(np.shape(test_target_array))
    # remove_punc(train_data_array)
    # print(test_data_array)
