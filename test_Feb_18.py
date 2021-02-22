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

aa1 = (pd.read_csv(r"C:\\.csv", names=['classs','hum1.0', 'temp1.0', 'hum2.0', 'temp2.0', 'hum3.0', #4

                                                                            'temp3.0', 'hum4.0', 'temp4.0', 'hum5.0', 'temp5.0', #9

                                                                            'locA',  'statusA', 'statusB', 'hum1.1', #13

                                                                            'temp1.1', 'hum2.1', 'temp2.1', 'hum3.1', 'temp3.1', #18

                                                                            'hum4.1', 'temp4.1', 'hum5.1', 'temp5.1', 'status', #23

                                                                            'hum1Dif', 'temp1Dif', 'hum2dif', 'temp2Dif', 'hum3dif', #28

                                                                            'temp3Dif' 'hum4Dif', 'temp4Dif', 'hum5dif', 'temp5Dif', #33

                                                                            'startTime', 'endTime', 'elapsedTime'])) #36
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
    ax = plt.subplot(111)#2,5,i)#len(datasets), len(classifiers) + 1, i)
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
aa1 = (pd.read_csv(r"C:.csv", names=['classs','hum1.0', 'temp1.0', 'hum2.0', 'temp2.0', 'hum3.0', #5

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
