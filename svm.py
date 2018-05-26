from sklearn import svm,metrics
import numpy as np
import matplotlib as plt
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

def predict_svm(x,y,z,clf):
    clf.fit(x, y)
    predicted = clf.predict(z)
    return predicted

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

if __name__ == "__main__":
    x = np.load('data/train_encoded_array.npy')
    y = np.load('data/train_target_array.npy')
    y = y.astype('int')
    y = y.flatten()
    z = np.load('data/test_encoded_array.npy')
    t = np.load('data/test_target_array.npy')
    t = t.astype('int')
    t = t.flatten()
    clf = svm.SVC()
    predicted = predict_svm(x,y,z,clf)
    accuracy = metrics.accuracy_score(t, predicted, normalize=False)
    confusion_matrix = metrics.confusion_matrix(t, predicted)
    print(np.shape(predicted))
    print(accuracy)

    ########################################


    # title for the plots
    titles = ('SVC with linear kernel')

    # Set-up 2x2 grid for plotting.
    fig,sub = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)


    pca = PCA(n_components=2).fit(x)
    x_2d = pca.transform(x)

    X0, X1 = x_2d[:, 0], x_2d[:, 1]
    # xx, yy = make_meshgrid(X0, X1)

    # Plot Decision Region using mlxtend's awesome plotting function
    plot_decision_regions(X=x_2d,
                          y=y,
                          clf=clf,
                          legend=2)

    # Update plot object with X/Y axis labels and Figure Title
    plt.xlabel(X.columns[0], size=14)
    plt.ylabel(X.columns[1], size=14)
    plt.title('SVM Decision Region Boundary', size=16)

    # for clf, title, ax in (predicted,titles,sub):
    #     plot_contours(ax, clf, xx, yy,
    #                   cmap=plt.cm.coolwarm, alpha=0.8)
    #     ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel('Sepal length')
    #     ax.set_ylabel('Sepal width')
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title)
    #
    # plt.show()
