#7.7.2019: Addition of axis ticks and labels and legends.
#7.7.2019: classifier comparison alpha colors adjusted to make testing points more transparent, removed QDA and neural net
##7.7.2019: runlog60. Emphasis on knn and svc.
##7.8.2019: n_neighbors reduced to 10; too slow
#7.13.2019: module importing condensed to top of script. running runLog017.
#7.18.2019: runlog60.
print(__doc__)

import pandas as pd
import csv
from numpy import array
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestCentroid
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

aa1 = (pd.read_csv(r"C:\\Users\\r\\Desktop\\opD\\feb_2021\\2.18\\greenhouse_project\\feb18.csv",
                   
                                 names=['classs','hum1.0', 'temp1.0', 'hum2.0', 'temp2.0', #4

                                        'locA', 'statusA', 'statusB', 'hum1.1', 'temp1.1', #9

                                        'hum2.1', 'temp2.1', 'status', 'hum1Dif', 'temp1Dif', #14

                                        'hum2dif', 'temp2Dif', 'startTime', 'endTime', 'elapsedTime'])) #19
print(aa1)
aa1 = array(aa1)
print(aa1[:,36])

def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(8, 14, 1000000), np.linspace(y_min, y_max, 1000000))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the line, the points, and the nearest vectors to the plane
    plot.contourf(xx, yy, Z, alpha=0.65, cmap=plt.cm.plasma)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=32 * sample_weight, alpha=0.55,
                 cmap=plt.cm.plasma, edgecolors='black')
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
##    axis.axis('on')
    axis.set_title(title)

    jfk = ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
    plt.colorbar(jfk, ax=ax)

plt.scatter(x=(aa1[:,12]),y=(aa1[:,24]))
plt.title("phase")
plt.show()
plt.hist(x=(aa1[:,12]),bins=100)

plt.show()

y31=((aa1[:,24])/(aa1[:,36]))
y32=((aa1[:,25])/(aa1[:,36]))
y33=((aa1[:,26])/(aa1[:,36]))
y34=((aa1[:,27])/(aa1[:,36]))
y35=((aa1[:,28])/(aa1[:,36]))
y36=((aa1[:,29])/(aa1[:,36]))
y37=((aa1[:,30])/(aa1[:,36]))
y38=((aa1[:,31])/(aa1[:,36]))
y39=((aa1[:,32])/(aa1[:,36]))                    




def rocFxn(r,s,u,v):
    plt.title(v)
    plt.title("change / time elapsed")
    plt.plasma
    plt.scatter(x=(r), y=(s))
    plt.show()

    plt.title(v)
    plt.scatter(x=(r), y=(u))
    plt.show()
    
    plt.title(v)
    plt.hist(x=(s), bins=75)
    plt.show()

    plt.title(v)
    plt.hist(x=(s), bins=55)
    plt.show()
    


rocFxn((aa1[:,10]),(y32), (aa1[:,26]), ("hum1"))
##
##rocFxn((aa1[:,10]),(y33), (aa1[:,27]), ("temp1"))

rocFxn((aa1[:,10]),(y34), (aa1[:,28]), ("hum2"))

##rocFxn((aa1[:,10]),(y35), (aa1[:,29]), ("temp2"))
##
rocFxn((aa1[:,10]),(y36), (aa1[:,30]), ("hum4"))
##
##rocFxn((aa1[:,10]),(y37), (aa1[:,31]), ("temp4"))











                    ##plt.scatter(x=(aa1[:,7],y19))
                    ##plt.show()
                    ##
                    ##plt.scatter(x=(aa1[:,7],y20))
                    ##plt.show()
                    ##
                    ##plt.scatter(x=(aa1[:,7],y21))
                    ##plt.show()
                    ##
                    ##plt.scatter(x=(aa1[:,7],y22))
                    ##plt.show()
                    ##
                    ##plt.scatter(x=(aa1[:,7],y23))
                    ##plt.show()
                    ##
##
##
##
##
##
##
##
##
##
##
##X = aa1[:, [10,25]]
##
##y = (aa1[:, 11]
##    .tolist())
##
##sample_weight_last_ten = abs(np.random.randn(len(X)))
##sample_weight_constant = np.ones(len(X))
#### and bigger weights to some outliers
##sample_weight_last_ten[15:] *= 5
##sample_weight_last_ten[9] *= 15
##
###### for reference, first fit without sample weights
##
#### fit the model
##clf_weights = svm.SVC(gamma=10)
##clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)
##
##clf_no_weights = svm.SVC(gamma=1)
##clf_no_weights.fit(X, y)
##fig, axes = plt.subplots(1, 2)#, figsize=(175, 110))
##plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
##                       "H2 Constant weights")
##plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
##                       "H2 Modified weights")
##
##
##
##
##print("f")
##
##
##
##X = aa1[:, [10, 27]]
##y = (aa1[:, 11]
##    .tolist())
##
##
##sample_weight_last_ten = abs(np.random.randn(len(X)))
##sample_weight_constant = np.ones(len(X))
#### and bigger weights to some outliers
##sample_weight_last_ten[15:] *= 5
##sample_weight_last_ten[9] *= 15
##
#### for reference, first fit without sample weights
##
#### fit the model
##clf_weights = svm.SVC(gamma=10)
##clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)
##print("done")
##
##clf_no_weights = svm.SVC(gamma=1)
##clf_no_weights.fit(X, y)
##
##fig, axes = plt.subplots(1, 2)#, figsize=(175, 110))
##plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
##                       "H4 Constant weights")
##plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
##                       "H4 Modified weights")
##
##print("8976")
##
##
##X = aa1[:, [10,29]]
##
##y = (aa1[:, 11]
##    .tolist())
##
##sample_weight_last_ten = abs(np.random.randn(len(X)))
##sample_weight_constant = np.ones(len(X))
######## and bigger weights to some outliers
##sample_weight_last_ten[15:] *= 5
##sample_weight_last_ten[9] *= 15
##
#### for reference, first fit without sample weights
##
#### fit the model
##clf_weights = svm.SVC(gamma=10)
##clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)
##
##clf_no_weights = svm.SVC(gamma=1)
##clf_no_weights.fit(X, y)
##
##fig, axes = plt.subplots(1, 2)#, figsize=(175, 110))
##plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
##                       "H1 Constant weights")
##plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
##                       "H1 Modified weights")
##print(plot_decision_function)
##plt.legend('hum1')
##plt.show()
##
##
##
##
##














print("#1")
##
##
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##############################
##
##
##
##
###7.7.19: # of neighbors increased 8-->18.
##"""
##================================
##Nearest Neighbors Classification
##================================
##
##Sample usage of Nearest Neighbors classification.
##It will plot the decision boundaries for each class.
##"""
##print(__doc__)
##
##n_neighbors = 9 #18 is too slow
##
##X = (aa1[:, [7,11]])
##y = (aa1[:, 8]
##    .tolist())
##print("33333")
##h = .02  # step size in the mesh
### Create color maps
##cmap_light = plt.cm.plasma#ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##cmap_bold = plt.cm.plasma#ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##
##for weights in ['uniform', 'distance']:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
##    clf.fit(X, y)
##
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    print(Z)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=32)
##    plt.xlim(xx.min(), xx.max())
##    plt.ylim(yy.min(), yy.max())
##    plt.title("H1 (k = %i, weights = '%s')"
##              % (n_neighbors, weights))
##    plt.axis('on')
##
##
##
##for weights in ['uniform', 'distance']:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm='ball_tree')
##    clf.fit(X, y)
##
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##    print("ravel")
##    print(np.c_[xx.ravel(), yy.ravel()])
##    print("endravel")
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=32)
##    plt.xlim(xx.min(), xx.max())
##    plt.ylim(yy.min(), yy.max())
##    plt.title("3-Class classification (k = %i, weights = '%s')"
##              % (n_neighbors, weights))
##    plt.axis('on')
##
##
##
##X = aa1[:, [7,13]]
##y = (aa1[:, 8]
##    .tolist())
##
##h = .02  # step size in the mesh
######################### Create color maps
##cmap_light = plt.cm.plasma#ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##cmap_bold = plt.cm.plasma#ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##
##for weights in ['uniform', 'distance']:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
##    clf.fit(X, y)
##
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=32)
##    plt.xlim(xx.min(), xx.max())
##    plt.ylim(yy.min(), yy.max())
##    plt.title("H2 (k = %i, weights = '%s')"
##              % (n_neighbors, weights))
##    plt.axis('on')
##
##X = aa1[:, [7,15]]
##y = (aa1[:, 8]
##    .tolist())
##
##h = .02  # step size in the mesh
##
### Create color maps
##cmap_light = plt.cm.plasma#ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##cmap_bold = plt.cm.plasma#ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##
##for weights in ['uniform', 'distance']:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
##    clf.fit(X, y)
##
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=32)
##    plt.xlim(xx.min(), xx.max())
##    plt.ylim(yy.min(), yy.max())
##    plt.title("H4 (k = %i, weights = '%s')"
##              % (n_neighbors, weights))
##    plt.axis('on')
##
##plt.legend('hum2')
##plt.show()
##
##
##
##print("#2")
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##"""
##==================================================
##Plot different SVM classifiers in the iris dataset
##==================================================
##
##Comparison of different linear SVM classifiers on a 2D projection of the iris
##dataset. We only consider the first 2 features of this dataset:
##
##- Sepal length
##- Sepal width
##
##This example shows how to plot the decision surface for four SVM classifiers
##with different kernels.
##
##The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
##different decision boundaries. This can be a consequence of the following
##differences:
##
##- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
##  regular hinge loss.
##
##- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
##  reduction while ``SVC`` uses the One-vs-One multiclass reduction.
##
##Both linear models have linear decision boundaries (intersecting hyperplanes)
##while the non-linear kernel models (polynomial or Gaussian RBF) have more
##flexible non-linear decision boundaries with shapes that depend on the kind of
##kernel and its parameters.
##
##.. NOTE:: while plotting the decision function of classifiers for toy 2D
##   datasets can help get an intuitive understanding of their respective
##   expressive power, be aware that those intuitions don't always generalize to
##   more realistic high-dimensional problems.
##
##"""
##print(__doc__)
##
#########################################################################################################################################################################################
##
##
##X = aa1[:, [7,11]]
##
##y = (aa1[:, 8]
##    .tolist())
##
##def make_meshgrid(x, y, h=.02):
##    """Create a mesh of points to plot in
##
##    Parameters
##    ----------
##    x: data to base x-axis meshgrid on
##    y: data to base y-axis meshgrid on
##    h: stepsize for meshgrid, optional
##
##    Returns
##    -------
##    xx, yy : ndarray
##    """
##    x_min, x_max = x.min() - 1, x.max() + 1
##    y_min, y_max = y.min() - 1, y.max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    return xx, yy
##
##
##def plot_contours(ax, clf, xx, yy, **params):
##    """Plot the decision boundaries for a classifier.
##
##    Parameters
##    ----------
##    ax: matplotlib axes object
##    clf: a classifier
##    xx: meshgrid ndarray
##    yy: meshgrid ndarray
##    params: dictionary of params to pass to contourf, optional
##    """
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##    Z = Z.reshape(xx.shape)
##    out = ax.contourf(xx, yy, Z, **params)
##    return out
##
### we create an instance of SVM and fit out data. We do not scale our
### data since we want to plot the support vectors
##C = 80.0  # SVM regularization parameter
##models = (svm.SVC(kernel='linear', C=C),
##          svm.LinearSVC(C=C, max_iter=9275000),
##          svm.SVC(kernel='rbf', gamma=0.7, C=C),
##          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
##models = (clf.fit(X, y) for clf in models)
##
### title for the plots
##titles = ('SVC with linear kernel',
##          'LinearSVC (linear kernel)',
##          'SVC with RBF kernel',
##          'SVC with polynomial (degree 3) kernel')
##
### Set-up 2x2 grid for plotting.
##fig, sub = plt.subplots(2, 2)
##plt.subplots_adjust(wspace=0.8, hspace=0.4)
##X0, X1 = X[:, 0], X[:, 1]
##xx, yy = make_meshgrid(X0, X1)
##
##for clf, title, ax in zip(models, titles, sub.flatten()):
##    plot_contours(ax, clf, xx, yy,
##                  cmap=plt.cm.plasma, alpha=0.8)
##    ax.scatter(X0, X1, c=y, cmap=plt.cm.plasma, s=30, edgecolors='k')
##    ax.set_xlabel('FAN ZONE')
##    ax.set_ylabel('HUMIDITY (%RH)')
##    ax.set_title(title)
##    ax.axis('on')
##print("red")
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##X = aa1[:, [7,13]]
##
##y = (aa1[:, 8]
##    .tolist())
##
##def make_meshgrid(x, y, h=.02):
##    """Create a mesh of points to plot in
##
##    Parameters
##    ----------
##    x: data to base x-axis meshgrid on
##    y: data to base y-axis meshgrid on
##    h: stepsize for meshgrid, optional
##
##    Returns
##    -------
##    xx, yy : ndarray
##    """
##    x_min, x_max = x.min() - 1, x.max() + 1
##    y_min, y_max = y.min() - 1, y.max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    return xx, yy
##
##
##def plot_contours(ax, clf, xx, yy, **params):
##    """Plot the decision boundaries for a classifier.
##
##    Parameters
##    ----------
##    ax: matplotlib axes object
##    clf: a classifier
##    xx: meshgrid ndarray
##    yy: meshgrid ndarray
##    params: dictionary of params to pass to contourf, optional
##    """
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##    Z = Z.reshape(xx.shape)
##    out = ax.contourf(xx, yy, Z, **params)
##    return out
##
##
##
##
##
##
### we create an instance of SVM and fit out data. We do not scale our
### data since we want to plot the support vectors
##C = 80.0  # SVM regularization parameter
##models = (svm.SVC(kernel='linear', C=C),
##          svm.LinearSVC(C=C, max_iter=9275000),
##          svm.SVC(kernel='rbf', gamma=0.7, C=C),
##          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
##models = (clf.fit(X, y) for clf in models)
##
### title for the plots
##titles = ('H2 SVC with linear kernel',
##          'H2 LinearSVC (linear kernel)',
##          'H2 SVC with RBF kernel',
##          'H2 SVC with polynomial (degree 3) kernel')
##
### Set-up 2x2 grid for plotting.
##fig, sub = plt.subplots(2, 2)
##plt.subplots_adjust(wspace=0.8, hspace=0.4)
##X0, X1 = X[:, 0], X[:, 1]
##xx, yy = make_meshgrid(X0, X1)
##
##for clf, title, ax in zip(models, titles, sub.flatten()):
##    plot_contours(ax, clf, xx, yy,
##                  cmap=plt.cm.plasma, alpha=0.8)
##    ax.scatter(X0, X1, c=y, cmap=plt.cm.plasma, s=30, edgecolors='k')
##    ax.set_xlabel('FAN ZONE')
##    ax.set_ylabel('HUMIDITY (%RH)')
##    ax.set_title(title)
##    ax.axis('on')
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##X = aa1[:, [7,15]]
##
##y = (aa1[:, 8]
##    .tolist())
##
##def make_meshgrid(x, y, h=.02):
##    """Create a mesh of points to plot in
##
##    Parameters
##    ----------
##    x: data to base x-axis meshgrid on
##    y: data to base y-axis meshgrid on
##    h: stepsize for meshgrid, optional
##
##    Returns
##    -------
##    xx, yy : ndarray
##    """
##    x_min, x_max = x.min() - 1, x.max() + 1
##    y_min, y_max = y.min() - 1, y.max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    return xx, yy
##
##
##def plot_contours(ax, clf, xx, yy, **params):
##    """Plot the decision boundaries for a classifier.
##
##    Parameters
##    ----------
##    ax: matplotlib axes object
##    clf: a classifier
##    xx: meshgrid ndarray
##    yy: meshgrid ndarray
##    params: dictionary of params to pass to contourf, optional
##    """
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##    Z = Z.reshape(xx.shape)
##    out = ax.contourf(xx, yy, Z, **params)
##    return out
##
##
##
##
##
##
### we create an instance of SVM and fit out data. We do not scale our
### data since we want to plot the support vectors
##C = 80.0  # SVM regularization parameter
##models = (svm.SVC(kernel='linear', C=C),
##          svm.LinearSVC(C=C, max_iter=9275000),
##          svm.SVC(kernel='rbf', gamma=0.7, C=C),
##          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
##models = (clf.fit(X, y) for clf in models)
##
### title for the plots
##titles = ('SVC with linear kernel',
##          'LinearSVC (linear kernel)',
##          'SVC with RBF kernel',
##          'SVC with polynomial (degree 3) kernel')
##
### Set-up 2x2 grid for plotting.
##fig, sub = plt.subplots(2, 2)
##plt.subplots_adjust(wspace=0.8, hspace=0.4)
##X0, X1 = X[:, 0], X[:, 1]
##xx, yy = make_meshgrid(X0, X1)
##
##for clf, title, ax in zip(models, titles, sub.flatten()):
##    plot_contours(ax, clf, xx, yy,
##                  cmap=plt.cm.plasma, alpha=0.8)
##    ax.scatter(X0, X1, c=y, cmap=plt.cm.plasma, s=30, edgecolors='k')
##    ax.set_xlabel('FAN ZONE')
##    ax.set_ylabel('HUMIDITY (%RH)')
##    ax.set_title(title)
##    ax.axis('on')
##print("red")
##plt.show()
##
####################################
##########################################################
####################################"""
####################################=========================================================
####################################SVM-Kernels
####################################=========================================================
####################################
####################################Three different types of SVM-Kernels are displayed below.
####################################The polynomial and RBF are especially useful when the
####################################data-points are not linearly separable.
####################################
####################################
####################################"""
####################################
##################################### Code source: Gaël Varoquaux
##################################### License: BSD 3 clause
####################################
####################################n_neighbors = 5
####################################
##################################### import some data to play with
####################################pd.read_csv(r"C:\\Users\\r\\Desktop\\logFiles\numPro-0100.csv", names=['class', 'hum1.0', 'temp1.0', 'hum2.0', 'temp2.0',
####################################                                                                          'hum4.0', 'temp4.0', 'locA', 'statusA', 'locB',
####################################                                                                          'statusB', 'hum1.1', 'temp1.1', 'hum2.1', 'temp2.1',
####################################                                                                          'hum4.1', 'temp4.1','status', 'hum1Dif', 'temp1Dif',
####################################                                                                          'hum2dif', 'temp2Dif', 'hum4Dif', 'temp4Dif','time']))
####################################aa1 = array(aa1)
####################################X = aa1[:, [7,11]]
####################################Y = (aa1[:, 8]
####################################    .tolist())
####################################
##################################### figure number
####################################fignum = 1
####################################
##################################### fit the model
####################################for kernel in ('linear', 'poly', 'rbf'):
####################################    clf = svm.SVC(kernel=kernel, gamma=2)
####################################    clf.fit(X, Y)
####################################
####################################    # plot the line, the points, and the nearest vectors to the plane
####################################    plt.figure(fignum, figsize=(4, 3))
####################################    plt.clf()
####################################
####################################    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
####################################                facecolors='none', zorder=10, edgecolors='k')
####################################    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.plasma,
####################################                edgecolors='k')
####################################
####################################    XX, YY = np.mgrid[x_min:x_max:2000j, y_min:y_max:2000j]
####################################    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
####################################    plt.axis('on')
####################################    # Put the result into a color plot
####################################    Z = Z.reshape(XX.shape)
####################################    plt.figure(fignum, figsize=(4, 3))
####################################    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.plasma)
####################################    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
####################################                levels=[-.5, 0, .5])
####################################    plt.axis('on')
####################################    plt.xticks(())
####################################    plt.yticks(())
####################################    fignum = fignum + 1
####################################
####################################plt.show()
####################################
####################################
####################################
####################################
####################################
####################################
####################################
####################################
####################################aa1 = array(aa1)
####################################X = aa1[:, [7,13]]
####################################Y = (aa1[:, 8]
####################################.tolist())
####################################
##################################### figure number
####################################fignum = 1
####################################
##################################### fit the model
####################################for kernel in ('linear', 'poly', 'rbf'):
####################################    clf = svm.SVC(kernel=kernel, gamma=2)
####################################    clf.fit(X, Y)
####################################
####################################    # plot the line, the points, and the nearest vectors to the plane
####################################    plt.figure(fignum, figsize=(4, 3))
####################################    plt.clf()
####################################
####################################    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
####################################                facecolors='none', zorder=10, edgecolors='k')
####################################    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.plasma,
####################################                edgecolors='k')
####################################
####################################    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
####################################    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
####################################    plt.axis('on')
####################################    # Put the result into a color plot
####################################    Z = Z.reshape(XX.shape)
####################################    plt.figure(fignum, figsize=(4, 3))
####################################    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.plasma)
####################################    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
####################################                levels=[-.5, 0, .5])
####################################    plt.axis('on')
####################################    plt.xticks(())
####################################    plt.yticks(())
####################################    fignum = fignum + 1
####################################
####################################plt.show()
####################################
####################################
####################################
####################################
####################################
####################################aa1 = array(aa1)
####################################X = aa1[:, [7,15]]
####################################Y = (aa1[:, 8]
####################################.tolist())
####################################
##################################### figure number
####################################fignum = 1
####################################
##################################### fit the model
####################################for kernel in ('linear', 'poly', 'rbf'):
####################################    clf = svm.SVC(kernel=kernel, gamma=2)
####################################    clf.fit(X, Y)
####################################
####################################    # plot the line, the points, and the nearest vectors to the plane
####################################    plt.figure(fignum, figsize=(4, 3))
####################################    plt.clf()
####################################
####################################    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
####################################                facecolors='none', zorder=10, edgecolors='k')
####################################    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.plasma,
####################################                edgecolors='k')
####################################
####################################    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
####################################    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
####################################    plt.axis('on')
####################################    # Put the result into a color plot
####################################    Z = Z.reshape(XX.shape)
####################################    plt.figure(fignum, figsize=(4, 3))
####################################    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.plasma)
####################################    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
####################################                levels=[-.5, 0, .5])
####################################    plt.axis('on')
####################################    plt.xticks(())
####################################    plt.yticks(())
####################################    fignum = fignum + 1
####################################
####################################plt.show()
####################################
####################################
####################################
####################################
##
##
##
##
##
##
##
##
##"""
##===============================
##Nearest Centroid Classification
##===============================
##
##Sample usage of Nearest Centroid classification.
##It will plot the decision boundaries for each class.
##"""
##print(__doc__)
##
##n_neighbors = 17
##
### import some data to play with
##pd.read_csv(r"C:\\Users\\r\\Desktop\\logFiles\numPro-0100.csv", names=['class', 'hum1.0', 'temp1.0', 'hum2.0', 'temp2.0',
##                                                                          'hum4.0', 'temp4.0', 'locA', 'statusA', 'locB',
##                                                                          'statusB', 'hum1.1', 'temp1.1', 'hum2.1', 'temp2.1',
##                                                                          'hum4.1', 'temp4.1','status', 'hum1Dif', 'temp1Dif',
##                                                                          'hum2dif', 'temp2Dif', 'hum4Dif', 'temp4Dif','time']))
##aa1 = array(aa1)
##X = aa1[:, [7,11]]
##y = (aa1[:, 8]
##    .tolist())
##
##
##h = .02  # step size in the mesh
##
### Create color maps
##cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##
##for shrinkage in [None, .2]:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = NearestCentroid(shrink_threshold=shrinkage)
##    clf.fit(X, y)
##    y_pred = clf.predict(X)
##    print(shrinkage, np.mean(y == y_pred))
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##    plt.axis('on')
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=20)
##    plt.title("2-Class classification (shrink_threshold=%r)"
##              % shrinkage)
##    plt.axis('tight')
##    plt.axis('on')
##
##
##aa1 = array(aa1)
##X = aa1[:, [7,13]]
##y = (aa1[:, 8]
##    .tolist())
##
##
##h = .02  # step size in the mesh
##
### Create color maps
##cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##
##for shrinkage in [None, .2]:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = NearestCentroid(shrink_threshold=shrinkage)
##    clf.fit(X, y)
##    y_pred = clf.predict(X)
##    print(shrinkage, np.mean(y == y_pred))
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##    plt.axis('on')
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=20)
##    plt.title("2-Class classification (shrink_threshold=%r)"
##              % shrinkage)
##    plt.axis('tight')
##    plt.axis('on')
##
##
##
##
##
##
##
##
##
##
##
##aa1 = array(aa1)
##X = aa1[:, [7,15]]
##y = (aa1[:, 8]
##    .tolist())
##
##
##h = .02  # step size in the mesh
##
### Create color maps
##cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
##cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
##
##for shrinkage in [None, .2]:
##    # we create an instance of Neighbours Classifier and fit the data.
##    clf = NearestCentroid(shrink_threshold=shrinkage)
##    clf.fit(X, y)
##    y_pred = clf.predict(X)
##    print(shrinkage, np.mean(y == y_pred))
##    # Plot the decision boundary. For that, we will assign a color to each
##    # point in the mesh [x_min, x_max]x[y_min, y_max].
##    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
##    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                         np.arange(y_min, y_max, h))
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##
##    # Put the result into a color plot
##    Z = Z.reshape(xx.shape)
##    plt.figure()
##    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
##    plt.axis('on')
##    # Plot also the training points
##    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
##                edgecolor='k', s=20)
##    plt.title("2-Class classification (shrink_threshold=%r)"
##              % shrinkage)
##    plt.axis('tight')
##    plt.axis('on')
##
##
##
##
##
##
##
##






























"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


##Code source: Gaël Varoquaux
##              Andreas Müller
## Modified for documentation by Jaques Grobler
## License: BSD 3 clause
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process"]
##,
##         "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes"]#"Neural Net", "AdaBoost",
         #"Naive Bayes"], "QDA"]

classifiers = [
    KNeighborsClassifier(4),
    SVC(kernel="linear", C=1),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]#,


#################################################################################################################################
X = aa1[:, [10,24]]
y = (aa1[:, 11]
    .tolist())

x_min, x_max = X[:, 0].min()-1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10

datasets = X, y
figure = plt.figure()
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
##    X, y = ds
##    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    

    # just plot the dataset first
    cm = plt.cm.plasma
    cm_bright = plt.cm.plasma#ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(2,5,i)#len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
        
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')

    ax.axis('on')
    
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(2,5,i)#len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]


            
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        print(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
        print(ax.contourf)
        plt.axis('on')

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.36)

        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=8, horizontalalignment='right')
        i += 1
        ax.axis('on')
jfk = ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
plt.colorbar(jfk, ax=ax)
plt.legend()



#############################################################################
X = aa1[:, [10,26]]
y = (aa1[:, 11]
    .tolist())
datasets = X, y
figure = plt.figure()
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
##    X, y = ds
##    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.plasma
    cm_bright = plt.cm.plasma#ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(2,5,i)#len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')

    ax.axis('on')
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(2,5,i)#len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
        plt.axis('on')
        
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.36)

        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=8, horizontalalignment='right')
        i += 1
        ax.axis('on')

jfk = ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
plt.colorbar(jfk, ax=ax)
plt.legend()
##print(time.perfcounter())


#######################################################################################################################################################################################
#######################################################################################################################################################################################
X = aa1[:, [10,28]]
y = (aa1[:, 11]
    .tolist())
datasets = X, y
figure = plt.figure()#figsize=(7, 13))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
##    X, y = ds
##    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)


    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.plasma
    cm_bright = plt.cm.plasma #ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(2,5,i)#len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')

    ax.axis('on')
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(2,5,i)#len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        print(xx)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
        plt.axis('on')
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.36)

        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=8, horizontalalignment='right')
        i += 1
        ax.axis('on')
        jfk = ax.contourf(xx, yy, Z, cmap=cm, alpha=.75)
        plt.colorbar(jfk, ax=ax)
plt.legend()

plt.show()






























"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
np.random.seed(5)
aa1 = (pd.read_csv(r"C:\\Users\\r\\Desktop\\logFiles\numPro-0130.csv", names=['classs','hum1.0', 'temp1.0', 'hum2.0', 'temp2.0', 'hum3.0', #5

                                                                            'temp3.0', 'hum4.0', 'temp4.0', 'hum5.0', 'temp5.0', #10

                                                                            'locA',  'statusA',  'locB',  'statusB', 'hum1.1', #15

                                                                            'temp1.1', 'hum2.1', 'temp2.1', 'hum3.1', 'temp3.1', #20

                                                                            'hum4.1', 'temp4.1', 'hum5.1', 'temp5.1', 'status', #25

                                                                            'hum1Dif', 'temp1Dif', 'hum2dif', 'temp2Dif', 'hum3dif', #30

                                                                            'temp3Dif' 'hum4Dif', 'temp4Dif', 'hum5dif', 'temp5Dif', #35

                                                                            'startTime', 'endTime', 'elapsedTime'])) #38


aa1 = array(aa1)
#######################################################################################################################################################################################
#######################################################################################################################################################################################
X = aa1

y = (aa1[:, 8].astype(np.int64))#.tolist())


estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_5', KMeans(n_clusters=5))]

fignum = 1
titles = ['8 clusters', '3 clusters', '5 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 7], X[:, 13], X[:, 15],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('responseLocation')
    ax.set_ylabel('hum2')
    ax.set_zlabel('hum3')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('off', 0),
                    ('on', 1)]:
##    print(X[y == label, 11].mean())
    ax.text3D(X[y == label, 11].mean(),
              X[y == label, 13].mean(),
              X[y == label, 15].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [0,1]).astype(np.float)
ax.scatter(X[:, 7], X[:, 13], X[:, 15], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('responseLoc')
ax.set_ylabel('h2')
ax.set_zlabel('h3')
ax.set_title('Ground Truth')
ax.dist = 12
ax.axis('on')



plt.show()



aa1 = array(aa1)
#######################################################################################################################################################################################
#######################################################################################################################################################################################
X = aa1
print(X)
y = (aa1[:, 8].astype(np.int64))#.tolist())
print(y)

estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 7], X[:, 13], X[:, 15],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('responseLocation')
    ax.set_ylabel('hum2')
    ax.set_zlabel('hum3')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('off', 0),
                    ('on', 1)]:
##    print(X[y == label, 11].mean())
    ax.text3D(X[y == label, 11].mean(),
              X[y == label, 13].mean(),
              X[y == label, 15].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [0,1]).astype(np.float)
ax.scatter(X[:, 7], X[:, 13], X[:, 15], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('responseLoc')
ax.set_ylabel('h2')
ax.set_zlabel('h3')
ax.set_title('Ground Truth')
ax.dist = 12
ax.axis('on')

plt.show()














aa1 = array(aa1)
#######################################################################################################################################################################################
#######################################################################################################################################################################################
X = aa1
print(X)
y = (aa1[:, 8].astype(np.int64))#.tolist())
print(y)

estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_5', KMeans(n_clusters=5))]

fignum = 1
titles = ['8 clusters', '3 clusters', '5 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 7], X[:, 11], X[:, 15],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('responseLocation')
    ax.set_ylabel('hum1')
    ax.set_zlabel('hum3')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('off', 0),
                    ('on', 1)]:
##    print(X[y == label, 11].mean())
    ax.text3D(X[y == label, 11].mean(),
              X[y == label, 13].mean(),
              X[y == label, 15].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [0,1]).astype(np.float)
ax.scatter(X[:, 7], X[:, 11], X[:, 15], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('responseLoc')
ax.set_ylabel('h1')
ax.set_zlabel('h3')
ax.set_title('Ground Truth')
ax.dist = 12
ax.axis('on')
plt.show()
