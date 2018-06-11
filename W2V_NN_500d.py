import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime
from keras import models
from keras.layers import Dense

if __name__ == "__main__":

    startTime = datetime.datetime.now()

    x = np.load('data/train_w2v_data_array_500d.npy')
    y = np.load('data/train_w2v_target_array_500d.npy')
    y = y.astype('int')
    y = y.flatten()
    z = np.load('data/test_w2v_data_array_500d.npy')
    t = np.load('data/test_w2v_target_array_500d.npy')
    t = t.astype('int')
    t = t.flatten()

    learningRate = [0.1]
    for lr in learningRate:
        clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(30,20), batch_size='auto',
                            learning_rate='adaptive', learning_rate_init=lr, early_stopping=True)
        clf.fit(x, y)
        p = clf.predict(z)
        y_scores = clf.predict_proba(z)

        # predicted = predict_nn(x, y, z, clf)
        print("For learning rate: ", lr)
        print("Word2Vec Neural Network with 500 features")

        # Compute accuracy
        accuracy = accuracy_score(t, p, normalize=False)
        print("Accuracy: ", (accuracy / len(t)) * 100)

        # Confusion matrix
        confusion_matrix = confusion_matrix(t, p)
        print("Confusion Matrix:\n", confusion_matrix)

        # Replace 4s with 1s
        t[np.where(t == 4)] = 1
        p[np.where(p == 4)] = 1

        # Plot the Precision-Recall curve
        precision, recall, _ = precision_recall_curve(t, y_scores[:, 1])
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        average_precision = average_precision_score(t, p)
        plt.title('Neural Network Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        filename = "data/w2v_NN_500d_" + str(lr) + "_precisionRecall.png"
        plt.savefig(filename)