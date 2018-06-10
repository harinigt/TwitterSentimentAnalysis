from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
import datetime

def convertToNpArray(train,test):
    """
    Converts the data into numpy arrays
    :param train: training data csv path
    :param test: test data csv path
    :return: training data and labels, test data and labels
    """
    train_data = pd.read_csv(train, delimiter=',', quotechar='"',
                             dtype=None, encoding="ISO-8859-1",
                             usecols=[0, 5])
    train_array = create_train_data_subset(train_data)
    np.random.shuffle(train_array)
    train_target_array = train_array[:, 0]
    train_target_array = np.reshape(train_target_array, (len(train_target_array), 1))
    train_data_array = train_array[:, 1]
    train_data_array = np.reshape(train_data_array, (len(train_data_array), 1))
    test_data = pd.read_csv(test, delimiter=',', quotechar='"',
                            dtype=None, encoding="ISO-8859-1",
                            usecols=[0, 5], names=['label', 'tweet'])
    test_data = test_data[test_data.label != 2]
    test_data = test_data.values
    test_data = np.append(test_data, create_test_data_subset(train_data), axis=0)
    np.random.shuffle(test_data)
    test_target = test_data[:, 0]
    test_target_array = np.array(test_target)
    test_target_array = np.reshape(test_target_array, (len(test_target_array), 1))
    test_data = test_data[:, 1]
    test_data_array = np.reshape(test_data, (len(test_data), 1))

    return train_data_array,test_data_array,train_target_array,test_target_array

def create_train_data_subset(train_data):
    train_data_numpy_array = np.array(train_data)
    train_data_final = train_data_numpy_array[750000:850000,:]
    return train_data_final

def create_test_data_subset(train_data):
    train_data_numpy_array = np.array(train_data)
    test_data_final = train_data_numpy_array[0:10000, :]
    test_data_final = np.append(test_data_final,train_data_numpy_array[900000:910000,:],axis=0)
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

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_model[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

if __name__=="__main__":
    start_time = datetime.datetime.now()
    
    # define training data
    train_data_array, test_data_array, train_target_array, test_target_array = convertToNpArray('data/training.1600000.processed.noemoticon.csv',
                                                                                                'data/testdata.manual.2009.06.14.csv')
    np.save('data/train_w2v_target_array', train_target_array)
    np.save('data/test_w2v_target_array', test_target_array)

    train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')
    train_data_array = remove_punc(train_data_array)
    train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')

    train_tweets = []
    test_tweets = []

    #Tokenizing training data array
    for row in train_data_array:
        for item in row:
            item = str(item).split(' ')
            train_tweets.append(item)

    #Tokenizing test data array
    for row in test_data_array:
        for item in row:
            item = str(item).split(' ')
            test_tweets.append(item)

    # # Train word2vec model
    # model = Word2Vec(train_tweets, min_count=1)
    # words = list(model.wv.vocab)
    #
    # #Save word2vec model to disk
    # model.save('tweetmodel.bin')

    #Load model from disk
    tweet_model = Word2Vec.load('tweetmodel.bin')

    #Generating Tfidf (term frequencey-inverse document frequency) for the training data matrix
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x for x in train_tweets])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    #Generating the training tweet average array where each row represents a tweet in the training data
    train_tweet_average = np.empty((100000, 100))
    for tweet in train_tweets:
        train_tweet_average = np.append(train_tweet_average, buildWordVector(tweet, 100), axis=0)
    train_tweet_average = scale(train_tweet_average)

    #Save the processed training array
    np.save('data/train_w2v_data_array', train_tweet_average)

    print("Saved Train data array")

    # Generating the test tweet average array where each row represents a tweet in the test data
    test_tweet_average = np.empty((20000, 100))
    for tweet in test_tweets:
        test_tweet_average = np.append(test_tweet_average, buildWordVector(tweet, 100), axis=0)
    test_tweet_average = scale(test_tweet_average)

    #Save the processed test array
    np.save('data/test_w2v_data_array',test_tweet_average)

    print("Saved Test data array")

    end_time = datetime.datetime.now()- start_time
    print("Time taken to generate the word2vec vectors: ", end_time)







    # print(tfidf)
    # print(type(matrix))
    # print(matrix.shape)
    # print(matrix)
    # matrix = matrix.todense()
    # print(matrix)
    # print(new_model)

    # X = model[model.wv.vocab]
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # # create a scatter plot of the projection
    # pyplot.scatter(result[:, 0], result[:, 1])
    # words = list(model.wv.vocab)
    # print(len(words))
    # for i, word in enumerate(words):
    # 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()