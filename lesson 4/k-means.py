import numpy as np
import sklearn.datasets as Datasets
import seaborn as sn
import matplotlib.pyplot as plt

X, y = Datasets.make_friedman2(200, 0.3)
row, column = X.shape
X = np.c_[X, np.zeros(row)]


def k_means(X, class_num):
    centers, X = set_center(X, class_num)  # set initial centers

    for i in range(100):
        # print(X[:,-1])
        sn.scatterplot(X[:, 0], X[:, 1], hue=X[:, -1].astype(int))  # visualize the data in updating
        plt.show()
        last = np.copy(X)  # very important: because I just directly manipulate the
        # X array so the new array is actually update on X

        new_X = findclusters(centers, X)  # assign examples to their cluster according to the centers

        if np.array_equal(new_X[:, -1], last[:, -1]):  # once np change of existing clusters, end of clustering
            break

        centers = update_centers(new_X, centers)  # update the center of a cluster according to the new clusters

        # if new_X[:,-1].all() == last[:,-1].all():
        #     break

    return {'X': X, 'centers': centers}


def findclusters(centers, X):
    '''

    :param centers:
    :param X:
    :return: new X with all examples assigned to clusters
    '''
    for i in X:
        distance = np.sqrt(np.sum((centers - i) ** 2, axis=1))# calculate the distance of each example to each cluster center
        # print(distance)
        i[-1] = np.argmin(distance)# assign the min distance cluster number to this point meaning it is belong to this cluster
        # print(i[-1])

    return X


def set_center(X, n_class):
    '''

    :param X:
    :param n_class:
    :return: inital centers and X with center assigned cluster
    '''
    pick_index = np.random.choice(len(X), n_class) # choose indexs
    centers = X[pick_index] # use random index to select random centers for initialization

    for i in range(n_class):
        X[pick_index[i], -1] = i # assign the example to its cluster
    # print(X[:,-1])
    return centers, X


def update_centers(X, centers):
    """

    :param X:
    :param centers:
    :return: a list of new centers
    """

    for i in range(len(centers)):
        p_set = X[np.where(X[:, -1] == i)]
        new_center = np.sum(p_set, axis=0) / len(p_set)  # sum over all the points in the same cluster and do average
        centers[i] = new_center  # return the new_center of one cluster

    return centers


def test(X, num_class):
    result = k_means(X, num_class)
    return result['X'][:, -1], result['centers']


test(X, 3)

