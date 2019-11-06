import unittest
from collections import defaultdict, Counter
from pprint import pprint

import numpy as np
from sklearn.datasets.samples_generator import make_blobs

from scipy.spatial.distance import pdist, squareform

from fermat import FermatKMeans, Fermat


class ClusteringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        centers = np.array([
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 1.0],
        ])
        n_samples = 100
        cls.n_clusters, cls.n_features = centers.shape
        cls.X, cls.true_labels = make_blobs(n_samples=n_samples, centers=centers,
                                            cluster_std=1., random_state=42)
        cls.methods = [
            dict(path_method='FW', ),
            dict(path_method='D', k=5),
            dict(path_method='L', k=5, landmarks=10, estimator='up'),
            dict(path_method='L', k=5, landmarks=10, estimator='down'),
            dict(path_method='L', k=5, landmarks=10, estimator='mean'),
            dict(path_method='L', k=5, landmarks=10, estimator='no_lca'),
        ]

    def misses(self, labels):
        cs = [Counter() for _ in range(self.n_clusters)]
        for tlb, lb in zip(self.true_labels, labels):
            cs[tlb][lb] += 1
        return sum(sum(c.values()) - max(c.values()) for c in cs)

    def test_works_with_all_methods_on_euclidean(self):
        for i, extra_args in enumerate(self.methods):
            with self.subTest(i=i, msg="Error with args={}".format(extra_args)):
                fkm = FermatKMeans(
                    distance='euclidean',
                    cluster_qty=self.n_clusters,
                    alpha=4,
                    seed=42,
                    **extra_args
                )
                clusters = fkm.fit_predict(self.X)
                self.assertLess(self.misses(clusters), 10)

    def test_works_with_all_methods_on_matrix(self):

        distance = squareform(pdist(self.X, metric='cityblock'))
        for i, extra_args in enumerate(self.methods):
            with self.subTest(extra_args=extra_args, msg="Error with args={}".format(extra_args)):
                fkm = FermatKMeans(
                    distance='matrix',
                    cluster_qty=self.n_clusters,
                    alpha=4,
                    seed=42,
                    **extra_args
                )
                clusters = fkm.fit_predict(distance)
                self.assertLess(self.misses(clusters), 10)

    def test_invalid_parameters(self):

        fkm = FermatKMeans(
            distance='euclidean',
            cluster_qty=self.n_clusters,
            alpha=4, seed=42,
            path_method='NOT_VALID'
        )
        with self.assertRaises(ValueError):
            fkm.fit_predict(self.X)

        fkm = FermatKMeans(
            distance='not_valid',
            cluster_qty=self.n_clusters,
            alpha=4, seed=42,
            path_method='FW'
        )
        with self.assertRaises(ValueError):
            fkm.fit_predict(self.X)


    def test_distance_vs_distances(self):
        distance = squareform(pdist(self.X, metric='cityblock'))

        for i, extra_args in enumerate(self.methods):
            with self.subTest(extra_args=extra_args, msg="Error with args={}".format(extra_args)):

                f = Fermat(alpha=4, **extra_args)
                f.fit(distance)
                d = f.get_distances()
                for a in range(len(distance)):
                    for b in range(len(distance)):
                        with self.subTest(a=a, b=b, msg="Error with indices={}, {}".format(a, b)):
                            np.testing.assert_almost_equal(d[a, b], f.get_distance(a, b))
