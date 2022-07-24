import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

def kmeans_display(X, label, K):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], marker='o', ms = 5, mfc = 'green', mec = 'green')
    plt.plot(X1[:, 0], X1[:, 1], marker='o', ms = 5, mfc = 'red', mec = 'red')
    plt.plot(X2[:, 0], X2[:, 1], marker='o', ms = 5, mfc = 'blue', mec = 'blue')

    plt.axis('equal')
    plt.plot()
    plt.show()

def init_data():
    np.random.seed(11)
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 3
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)

    X = np.concatenate((X0, X1, X2), axis = 0)
    K = 3
    original_label = (np.asarray([0]*N + [1]*N + [2]*N)).T
    
    return X, K, original_label

def visualize(X, original_label, K):
    kmeans_display(X, original_label, K)

def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis=1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

def train(X, K, original_label):
    centers = np.asarray(kmeans_init_centers(X, K))

    while True:
        label = kmeans_assign_labels(X, centers)
        kmeans_update_centers(X, label, K)
        

def execute():
    X, K, original_label = init_data()
    #visualize(X, K, original_label)
    train(X, K, original_label)
    return 0

execute()