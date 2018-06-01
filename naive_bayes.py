import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.decomposition import PCA, NMF
import datetime

def predict_NB(x,y,z,clf):
    clf.fit(x,y)
    predicted = clf.predict(z)
    return predicted

if __name__ == "__main__":
    startTime = datetime.datetime.now()
    train_data = np.load('data/train_encoded_array.npy')
    #train_data[np.where(train_data==0)] = 0.001
    train_target = np.load('data/train_target_array.npy')
    train_target = train_target.astype('int')
    train_target = train_target.flatten()
                           
    test_data = np.load('data/test_encoded_array.npy')
    #test_data[np.where(test_data==0)] = 0.001
    test_target = np.load('data/test_target_array.npy')
    test_target = test_target.astype('int')
    test_target = test_target.flatten()
                           
    clf = BernoulliNB()
    #pca = PCA(n_components=200).fit(train_data)
    nmf = NMF(n_components=200, init='random', random_state=0)
    x_200d = nmf.fit_transform(train_data)
    #x_200d = pca.transform(train_data)
    print(np.where(x_200d<0), "shape:", np.shape(x_200d))
    z_200d = nmf.transform(test_data)
    predicted = predict_NB(x_200d,train_target,z_200d,clf)
    accuracy = metrics.accuracy_score(test_target, predicted, normalize=False)
    confusion_matrix = metrics.confusion_matrix(test_target, predicted)
      # print(np.shape(predicted))
    print("Accuracy Score: ",accuracy)
    print("Confusion Matrix:\n",confusion_matrix)
    endTime = datetime.datetime.now()- startTime
    print("Total time taken to train: ",endTime)

                       

