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
    """
    Creates a balanced subset of the training data
    :param train_data: Training data array
    :return: Balanced subset of training data
    """
    train_data_numpy_array = np.array(train_data)
    train_data_final = train_data_numpy_array[750000:850000,:]
    return train_data_final

def create_test_data_subset(train_data):
    """
    Creates a balanced subset of the test data
    :param train_data: Test data array
    :return: Balanced subset of test data
    """
    train_data_numpy_array = np.array(train_data)
    test_data_final = train_data_numpy_array[0:10000, :]
    test_data_final = np.append(test_data_final,train_data_numpy_array[900000:910000,:],axis=0)
    return test_data_final

def remove_punc(data_array):
    """
    Removes the punctuation from all the tweets
    :param data_array: Tweet array to be processed
    :return: Tweet array with punctuations removed
    """
    translator = str.maketrans(string.punctuation, len(string.punctuation)*' ')
    for i in range(len(data_array)):
        data_array[i][0] = data_array[i][0].translate(translator)
    return data_array
#end

def remove_stopwords(data_array,stopwords_file_path):
    """
    This method removes stop words like 'the', 'of', 'it'
    :param data_array: Tweet array from which the stop words need to be removed
    :param stopwords_file_path: Stop words file path
    :return: Tweet array with stop words removed
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
    """
    Obtained from the below source:
    https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
    This method builds a vector for each tweet
    :param tokens: Tokens in a tweet
    :param size: Size of the vector to be generated for a tweet
    :return: vector for a tweet
    """
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
    train_data_array, test_data_array, train_target_array, test_target_array = convertToNpArray('data/training.1600000.processed.noemoticon.csv','data/testdata.manual.2009.06.14.csv')

    #Save the target arrays
    np.save('data/train_w2v_target_array_500d', train_target_array)
    np.save('data/test_w2v_target_array', test_target_array)

    #Preprocessing training data arrays
    train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')
    train_data_array = remove_punc(train_data_array)
    train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')

    #Preprocessing test data arrays
    test_data_array = remove_stopwords(test_data_array,'stopwords.txt')
    test_data_array = remove_punc(test_data_array)
    test_data_array = remove_stopwords(test_data_array,'stopwords.txt')

    #Initializing tokenized train and test lists
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


    # Train word2vec model
    # model = Word2Vec(train_tweets, size=500, min_count=1)
    # words = list(model.wv.vocab)
    #
    #Save word2vec model to disk
    # model.save('tweetmodel_500.bin')

    #Load model from disk
    tweet_model = Word2Vec.load('data/tweetmodel.bin')

    #Generating Tfidf (term frequencey-inverse document frequency) for the training data matrix
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x for x in train_tweets])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    # Generating the training tweet average array where each row represents a tweet in the training data
    train_tweet_average = np.empty((0, 500))
    for tweet in train_tweets:
        train_tweet_average = np.append(train_tweet_average, buildWordVector(tweet, 500), axis=0)
    train_tweet_average = scale(train_tweet_average)

    # Save the processed training array
    np.save('data/train_w2v_data_array_500d', train_tweet_average)

    print("Saved Train data array")

    # Generating the test tweet average array where each row represents a tweet in the test data
    test_tweet_average = np.empty((0, 100))
    for tweet in test_tweets:
        test_tweet_average = np.append(test_tweet_average, buildWordVector(tweet, 100), axis=0)
    test_tweet_average = scale(test_tweet_average)

    #Save the processed test array
    np.save('data/test_w2v_data_array',test_tweet_average)

    print("Saved Test data array")

    end_time = datetime.datetime.now()- start_time
    print("Time taken to generate the word2vec vectors: ", end_time)