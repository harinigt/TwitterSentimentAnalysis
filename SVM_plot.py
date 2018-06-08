from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA


x = np.load('data/train_encoded_array_new.npy')
y = np.load('data/train_target_array_new.npy')
y = y.astype('int')
y = y.flatten()

z = np.load('data/test_encoded_array_new.npy')
t = np.load('data/test_target_array_new.npy')
t = t.astype('int')
t = t.flatten()

pca = PCA(n_components=200).fit(x)
x = pca.transform(x)
z = pca.transform(z)

pca = PCA(n_components=2).fit(x)
x = pca.transform(x)
z = pca.transform(z)

svm = SVC()
clf = svm.fit(x, y)
p = clf.predict(z)


# Plotting decision regions
plot_decision_regions(x, y, clf=clf, legend=2)
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('SVM Classifier')
plt.show()


