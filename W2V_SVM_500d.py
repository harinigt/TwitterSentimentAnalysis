from sklearn import svm,metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import datetime
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



if __name__ == "__main__":
    startTime = datetime.datetime.now()

    # Load training data
    x = np.load('data/train_w2v_data_array.npy')
    y = np.load('data/train_w2v_target_array.npy')
    y = y.astype('int')
    y = y.flatten()

    # Load test data
    z = np.load('data/test_w2v_data_array.npy')
    t = np.load('data/test_w2v_target_array.npy')
    t = t.astype('int')
    t = t.flatten()

    # Dimensionality reduction to 200 dimensions using PCA
    pca = PCA(n_components=500).fit(x)
    x_500d = pca.transform(x)
    z_500d = pca.transform(z)


    # Predict using SVM
    clf = svm.SVC()
    clf.fit(x_500d, y)
    p = clf.predict(z_500d)

    # Compute training time
    endTime = datetime.datetime.now() - startTime
    print("Total time taken to train: ", endTime)
    print("\n")


    print("W2V SVM with Dimensionality Reduction using PCA down to 500 features")

    # Compute accuracy
    accuracy = metrics.accuracy_score(t, p, normalize=False)
    print("Accuracy: ", (accuracy/len(t)) * 100)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(t, p)
    print("Confusion Matrix:\n", confusion_matrix)


    # Replace 4s with 1s to plot precision-recall curve
    # (only accepts binary values)
    t[np.where(t==4)] = 1
    p[np.where(p==4)] = 1

    # Plot the Precision-Recall curve
    precision, recall, _ = precision_recall_curve(t, p)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(t, p)
    plt.title('W2V SVM Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('data/w2v_svm500d_precisionRecall.png')
    plt.show()

