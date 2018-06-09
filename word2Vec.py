from gensim.models import Word2Vec
# from gensim.models import doc2vec
# LabeledSentence = doc2vec.LabeledSentence
import numpy as np
import pandas as pd
import string
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer



def convertToNpArray(train):
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
    train_data_array = train_array[:,1]
    train_data_array = np.reshape(train_data_array,(len(train_data_array),1))
    return train_data_array

def create_train_data_subset(train_data):
    train_data_numpy_array = np.array(train_data)
    train_data_final = train_data_numpy_array[750000:850000,:]
    return train_data_final

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


	#define training data
	train_data_array = convertToNpArray('data/training.1600000.processed.noemoticon.csv')
	train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')
	train_data_array = remove_punc(train_data_array)
	train_data_array = remove_stopwords(train_data_array, 'stopwords.txt')
	tweets = []
	for row in train_data_array:
		for item in row:
			item = str(item).split(' ')
			tweets.append(item)



	# train model
	model = Word2Vec(tweets, min_count=1)
	# summarize the loaded model
	# print(model)
	# summarize vocabulary
	words = list(model.wv.vocab)
	# print(words)
	# access vector for one word
	# print(model['sunny'])
	# save model
	model.save('tweetmodel.bin')
	# load model
tweet_model = Word2Vec.load('tweetmodel.bin')

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x for x in tweets])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

tweet_average = np.empty((100000,100))
for tweet in tweets:
    tweet_average = np.append(tweet_average,buildWordVector(tweet, 100),axis=0)

print(tweet_average)







#
# print(tfidf)
# print(type(matrix))
# print(matrix.shape)
# print(matrix)
# matrix = matrix.todense()
# print(matrix)





























	# print(new_model)
    #
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