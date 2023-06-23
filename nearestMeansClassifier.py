# Name: Tejas Acharya
# Class: EE-559 
# Date: 15-06-2023
# Assignment: Homework 2

import numpy as np
from scipy.spatial.distance import cdist
from plotDecBoundaries import plotDecBoundaries


class NearestMeansClassifier():
    def __init__(self):
        self.C = 0
        self.means = None
        self.means_prime = None
        self.classes = None
        self.features_idx = None


    def fit(self, X, y, features_idx):
        self.classes = np.unique(y)
        self.C = len(self.classes)
        self.features_idx = features_idx
        D = len(features_idx)
        self.means = np.empty((self.C, D))
        self.means_prime = np.empty((self.C, D))
        total_sum = np.zeros((self.C, D))
        N = np.zeros((self.C, ))

        for i in range(len(y)):
            total_sum[y[i] - 1, :] += X[i, features_idx]
            N[y[i] - 1] += 1
        
        for j in range(self.C):
            self.means[j, :] = total_sum[j,:] / N[j]

        self.set_means_prime()

        return
    

    def set_means_prime(self):
        self.means_prime = np.empty_like(self.means)

        for i in range(self.C):
            total_sum = np.zeros((1, self.means.shape[1]))
            N = 0
            for j in range(self.C):
                if i != j:
                    total_sum += self.means[j, :]
                    N += 1
            self.means_prime[i, :] = total_sum / N
        
        return;


    def predict(self, X):
        N = len(X)

        y_hat = np.empty((N,))

        for i in range(N):
            y_hat[i] = self.get_nearest_class(X[i, self.features_idx])
        
        y_hat = y_hat.astype('int32')

        return y_hat
    

    def get_nearest_class(self, x):
        x = np.reshape(x, (1, 2))
        l2_distances = cdist(self.means, x)
        l2_distances_prime = cdist(self.means_prime, x)

        # for i in range(self.C):
        #     l2_distances[i] = self.get_l2_norm(self.means[i, :], x)
        #     l2_distances_prime[i] = self.get_l2_norm(self.means_prime[i, :], x)
        
        condition = l2_distances < l2_distances_prime
        condition = condition.flatten()
        if (sum(condition) == 1):
            return self.classes[condition][0]
        else:
            return 4
    

    def get_l2_norm(self, a, b):
        return cdist(a, b)
    

    def get_error_rate(self, y, y_hat):
        return (sum(y != y_hat) / len(y)) * 100
    

    def get_class_means(self):
        return self.means
    

    def plot_ovr_boundary(self, X, y):
        plotDecBoundaries(X[:, self.features_idx], y, self.means, self.means_prime)
        return
    
    def plot_binary_boundary(self, X, y):
        plotDecBoundaries(X[:, self.features_idx], y, self.means, self.means_prime)
        return