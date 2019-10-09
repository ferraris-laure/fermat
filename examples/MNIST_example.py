import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from fermat import Fermat


def plot_figure(data, s, s_labels, xlabel):
    plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel, fontsize='large')
    for digit in range(10):
        xs = [data[i, 0] for i in range(s) if s_labels[i] == digit]
        ys = [data[i, 1] for i in range(s) if s_labels[i] == digit]
        plt.plot(xs, ys, 'o', label=str(digit))
    plt.legend(numpoints=1)


def main():
    # 1- Preprocessing data
    # Reading mnist data from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_train.csv

    all_data = np.loadtxt('../data/mnist_train.csv', skiprows=1, delimiter=',')
    labels = all_data[:, 0]
    data = all_data[:, 1:]/255.0

    # 2- Computing Fermat distance
    # we only consider a sample of data points
    test_size = 1000

    # compute euclidean distances between data points
    distances = distance_matrix(data[:test_size], data[:test_size])

    # Initialize the fermat model
    # The distances would be computed used the aprox method and the euclidean distances as input
    f = Fermat(alpha=4, path_method='L', k=30, landmarks=30)
    # fit the Fermat model
    f.fit(distances)

    # 3- Visualization
    tsne = TSNE(n_components=2, perplexity=60, n_iter=1000)
    tsne_euclidean = tsne.fit_transform(distances)
    plot_figure(tsne_euclidean, test_size, labels[:test_size], "TSNE on euclidean distance")
    plt.show()

    tsne_fermat = tsne.fit_transform(f.get_distances())
    plot_figure(tsne_fermat, test_size, labels[:test_size], "TSNE on Fermat distance")
    plt.show()


if __name__ == '__main__':
    main()
