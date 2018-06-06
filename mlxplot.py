from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pickle
from sklearn.externals import joblib

svm = SVC(C=0.5, kernel='linear')

x = np.load('data/train_encoded_array.npy')
y = np.load('data/train_target_array.npy')
y = y.astype('int')
y = y.flatten()

z = np.load('data/test_encoded_array.npy')
t = np.load('data/test_target_array.npy')
t = t.astype('int')
t = t.flatten()

pca = PCA(n_components=2).fit(x)
x = pca.transform(x)
z = pca.transform(z)

clf = svm.fit(x, y)
p = clf.predict(z)

# if joblib.dump(clf, 'SVMmodel.pkl'):
#     print("saved model")

# # Plotting decision regions
# plot_decision_regions(z, p, clf=clf, legend=2)
#
# # Adding axes annotations
# plt.xlabel('X0')
# plt.ylabel('X1')
# plt.title('SVM Classifier')
#
# plt.show()



######################################
#
# clf = joblib.load('SVMmodel.pkl')
# p = clf.predict(z)

t[np.where(t==4)] = 1
p[np.where(p==4)] = 1

# Precision-Recall curve
print("t: ", t)
print("p: ", p)

precision, recall, _ = precision_recall_curve(t, p)

# Plot the Precision-Recall curve
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

average_precision = average_precision_score(t, p)

plt.title('SVM Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

plt.show()
