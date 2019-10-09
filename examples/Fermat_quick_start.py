import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE

from examples.generate_data import generate_swiss_roll
from fermat import Fermat


def main():
    data, labels = generate_swiss_roll(oscilations=15, a=3, n=250)
    print('Data dimension:{}'.format(data.shape))

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, 80)
    ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], c=labels, s=4)
    plt.title('Swiss Roll Normals dataset \\n N=%s' % (data.shape[0]))

    plt.show()

    distances = distance_matrix(data, data)

    alpha = 3
    k = 100
    landmarks = 30

    # Initialize the model
    f_exact = Fermat(alpha=alpha, path_method='FW')
    f_exact.fit(distances)
    fermat_dist_exact = f_exact.get_distances()

    f_aprox_d = Fermat(alpha, path_method='D', k=k)
    f_aprox_d.fit(distances)
    fermat_dist_aprox_d = f_aprox_d.get_distances()

    f_aprox_d = Fermat(alpha, path_method='L', k=k, landmarks=landmarks)
    f_aprox_d.fit(distances)
    fermat_dist_aprox_l = f_aprox_d.get_distances()

    tsne_model = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=500)

    for dist in [fermat_dist_exact, fermat_dist_aprox_l, fermat_dist_aprox_d]:
        tsnes = tsne_model.fit_transform(dist)
        plt.scatter(tsnes[:, 0], tsnes[:, 1], c=labels, s=5)
        plt.show()


if __name__ == '__main__':
    main()
