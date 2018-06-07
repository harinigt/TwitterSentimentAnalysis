import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
from sklearn.decomposition import PCA, NMF
import datetime
import matplotlib.pyplot as plt

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
    train_target[np.where(train_target==4)] = 1
                           
    test_data = np.load('data/test_encoded_array.npy')
    #test_data[np.where(test_data==0)] = 0.001
    test_target = np.load('data/test_target_array.npy')
    test_target = test_target.astype('int')
    test_target = test_target.flatten()
    test_target[np.where(test_target==4)] = 1
                           
    clf = MultinomialNB(alpha=1)
    #pca = PCA(n_components=200).fit(train_data)
    nmf = NMF(n_components=1000, init='random', random_state=0)
    x_200d = nmf.fit_transform(train_data)
    #x_200d = pca.transform(train_data)
    z_200d = nmf.transform(test_data)
    predicted = predict_NB(x_200d,train_target,z_200d,clf)

    #Metrics
    accuracy = metrics.accuracy_score(test_target, predicted, normalize=False)
    confusion_matrix = metrics.confusion_matrix(test_target, predicted)
    average_precision = metrics.average_precision_score(test_target, predicted)
    precision = metrics.precision_score(test_target,predicted)
    recall = metrics.recall_score(test_target,predicted)
    precision_array,recall_array, _ = metrics.precision_recall_curve(test_target,predicted)
    roc_array = metrics.roc_curve(test_target,predicted)

    #Plotting Recall Precision Curve
    plt.step(recall_array, precision_array, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall_array, precision_array, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    lgd = plt.title('2-class Precision-Recall curve: AvgPrecision={0:0.2f}'.format(
        average_precision))

    #Printing Metrics
    print("Accuracy Score: ",accuracy)
    print("Confusion Matrix:\n",confusion_matrix)
    print("Precision:",precision)
    print("Recall:",recall)
    endTime = datetime.datetime.now()- startTime
    print("Total time taken to train: ",endTime)

    #Saving the plot
    plt.savefig('PRCurve_NB_NMF1000_Multinomial.png',bbox_extra_artists=(lgd,), bbox_inches='tight')

                       

