from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib as plt
from sklearn.decomposition import PCA
import datetime

def predict_nn(x,y,z,clf):
    clf.fit(x, y)
    predicted = clf.predict(z)
    return predicted


if __name__ == "__main__":
    startTime = datetime.datetime.now()
    x = np.load('data/train_encoded_array.npy')
    x[np.where(x==0)] = 0.001
    y = np.load('data/train_target_array.npy')
    y = y.astype('int')
    y = y.flatten()
    z = np.load('data/test_encoded_array.npy')
    z[np.where(z==0)] = 0.001
    t = np.load('data/test_target_array.npy')
    t = t.astype('int')
    t = t.flatten()
#    clf = svm.SVC(kernel='rbf')

#    pca = PCA(n_components=200).fit(x)
#    x_200d = pca.transform(x)
#    z_200d = pca.transform(z)
    learningRate =[0.1, 0.01, 0.001]
    for lr in learningRate:
        clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(4, ), random_state=1, batch_size='auto', learning_rate='constant', learning_rate_init=lr)
        clf.fit(x, y)
        predicted = predict_nn(x,y,z,clf)
        print("For learning rate: ", lr)
        accuracy = metrics.accuracy_score(t, predicted, normalize=False)
        confusion_matrix = metrics.confusion_matrix(t, predicted)
        # print(np.shape(predicted))
        print("Accuracy Score: ",accuracy)
        print("Confusion Matrix:\n",confusion_matrix)
    endTime = datetime.datetime.now()- startTime
    print("Total time taken to train: ",endTime)


