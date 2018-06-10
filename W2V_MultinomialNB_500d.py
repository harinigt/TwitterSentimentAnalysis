import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.decomposition import NMF
import datetime
import matplotlib.pyplot as plt

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

    # Predict using Naive Bayes Model
    clf = MultinomialNB(alpha=1)
    nmf = NMF(n_components=500, init='random', random_state=0)
    x_500d = nmf.fit_transform(x)
    z_500d = nmf.transform(z)
    clf.fit(x_500d, y)
    p = clf.predict(z_500d)

    # Compute training time
    endTime = datetime.datetime.now() - startTime
    print("Total time taken to train: ", endTime)
    print("\n")

    print("W2V Multinomial Naive Bayes with 300 features and alpha = 1")

    # Compute accuracy
    accuracy = metrics.accuracy_score(t, p, normalize=False)
    print("Accuracy: ", (accuracy / len(t)) * 100)

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(t, p)
    print("Confusion Matrix:\n", confusion_matrix)

    # Replace 4s with 1s
    t[np.where(t == 4)] = 1
    p[np.where(p == 4)] = 1

    y_scores = clf.predict_proba(z_500d)

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
    plt.savefig('data/w2v_MultinomialNB500d_alpha1_precisionRecall.png')
    plt.show()
