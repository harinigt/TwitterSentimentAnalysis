import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.decomposition import NMF
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

if __name__ == "__main__":
    startTime = datetime.datetime.now()

    # Load training data
    x = np.load('data/train_w2v_data_array_500d.npy')
    y = np.load('data/train_w2v_target_array_500d.npy')
    y = y.astype('int')
    y = y.flatten()

    # Load test data
    z = np.load('data/test_w2v_data_array_500d.npy')
    t = np.load('data/test_w2v_target_array_500d.npy')
    t = t.astype('int')
    t = t.flatten()

    # Remove -ve values and scale all values by smallest -ve value in array

    xmin = np.amin(x)
    zmin = np.amin(z)
    scale_min = min(xmin,zmin) * -1

    x = np.add(x,scale_min)
    z = np.add(z,scale_min)

    # x = x + 11.573273289802543
    # z = z + 16.698667840828804

    # Predict using Naive Bayes Model
    clf = MultinomialNB(alpha=1)
    clf.fit(x, y)
    p = clf.predict(z)

    # Compute training time
    endTime = datetime.datetime.now() - startTime
    print("Total time taken to train: ", endTime)
    print("\n")

    print("W2V Multinomial Naive Bayes 500d")

    # Compute accuracy
    accuracy = metrics.accuracy_score(t, p, normalize=False)
    print("Accuracy: ", (accuracy / len(t)) * 100)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(t, p)
    print("Confusion Matrix:\n", confusion_matrix)

    # Replace 4s with 1s
    t[np.where(t == 4)] = 1
    p[np.where(p == 4)] = 1

    y_scores = clf.predict_proba(z)

    # Plot the Precision-Recall curve
    precision, recall, _ = metrics.precision_recall_curve(t, y_scores[:, 1])
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    average_precision = metrics.average_precision_score(t, p)
    plt.title('W2V Multinomial NB 500d Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('data/w2v_MultinomialNB_500d_precisionRecall.png')
    plt.show()
