from sklearn import svm,metrics
import numpy as np
import matplotlib as plt
from sklearn.decomposition import PCA
import datetime

from mlxtend.plotting import plot_decision_regions

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



if __name__ == "__main__":
    startTime = datetime.datetime.now()

    # Load training data
    x = np.load('data/train_encoded_array_new.npy')
    y = np.load('data/train_target_array_new.npy')
    y = y.astype('int')
    y = y.flatten()

    # Load test data
    z = np.load('data/test_encoded_array_new.npy')
    t = np.load('data/test_target_array_new.npy')
    t = t.astype('int')
    t = t.flatten()

    # Dimensionality reduction to 200 dimensions using PCA
    pca = PCA(n_components=200).fit(x)
    x_200d = pca.transform(x)
    z_200d = pca.transform(z)

    # Replace 0s with 0.001s
    t[np.where(t==0)] = 0.001

    # Predict using SVM
    clf = svm.SVC()
    clf.fit(x_200d, y)
    p = clf.predict(z_200d)

    # Compute training time
    endTime = datetime.datetime.now() - startTime
    print("Total time taken to train: ", endTime)
    print("\n")


    print("SVM with Dimensionality Reduction using PCA down to 200 features (classes: 0.001|4)")

    # Compute accuracy
    accuracy = metrics.accuracy_score(t, p, normalize=False)
    print("Accuracy: ", (accuracy/len(t)) * 100)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(t, p)
    print("Confusion Matrix:\n", confusion_matrix)

    # Plotting decision regions
    plot_decision_regions(z, p, clf=clf, legend=2)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('SVM with Dimensionality Reduction using PCA down to 200 features (classes: 0.001|4)')
    plt.show()
    plt.savefig('data/svm200d_decisionRegions')

    # Replace 4s with 1s and 0.001s with 0s to plot precision-recall curve
    # (only accepts binary values)
    t[np.where(t==4)] = 1
    p[np.where(p==4)] = 1
    t[np.where(t==0.001)] = 0
    p[np.where(p==0.001)] = 0

    # Plot the Precision-Recall curve
    precision, recall, _ = precision_recall_curve(t, p)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = average_precision_score(t, p)
    plt.title('SVM (200d, classes: 0.001|4) Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    plt.savefig('data/svm200d_precisionRecall')






