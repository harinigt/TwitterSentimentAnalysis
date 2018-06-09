from sklearn import svm,metrics
import numpy as np
import matplotlib.pyplot as plt
import datetime
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

    # Predict using SVM
    clf = svm.SVC()
    clf.fit(x, y)
    p = clf.predict(z)

    # Compute training time
    endTime = datetime.datetime.now() - startTime
    print("Total time taken to train: ", endTime)
    print("\n")


    print("SVM with 2000 features (classes: 0|4)")

    # Compute accuracy
    accuracy = metrics.accuracy_score(t, p, normalize=False)
    print("Accuracy: ", (accuracy/len(t)) * 100)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(t, p)
    print("Confusion Matrix:\n", confusion_matrix)


    # Replace 4s with 1s to plot precision-recall curve
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
    plt.title('SVM (2000d, classes: 0|4) Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('data/svm2000d_with0s_precisionRecall.png')
    plt.show()





