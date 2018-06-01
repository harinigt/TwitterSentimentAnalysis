from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA


# Loading some example data
# iris = datasets.load_iris()
# X = iris.data[:, 2]
# X = X[:, None]
# y = iris.target

# np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# y = np.array([0] * 20 + [1] * 20)
# Z = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# t = np.array([0] * 20 + [1] * 20)


# Training a classifier
svm = SVC(C=0.5, kernel='linear')


####
x = np.load('data/train_encoded_array.npy')
y = np.load('data/train_target_array.npy')
y = y.astype('int')
y = y.flatten()
z = np.load('data/test_encoded_array.npy')
t = np.load('data/test_target_array.npy')
t = t.astype('int')
t = t.flatten()
# clf = svm.SVC()
pca = PCA(n_components=2).fit(x)
x = pca.transform(x)
z = pca.transform(z)
####

svm.fit(x, y)
p = svm.predict(z)

# Plotting decision regions
# plot_decision_regions(X, y, clf=svm, legend=2)
plot_decision_regions(z, p, clf=svm, legend=2)

# Plot

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.title('test')

plt.show()
