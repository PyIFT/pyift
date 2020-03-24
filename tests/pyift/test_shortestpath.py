import numpy as np
from scipy.sparse import csgraph
from pyift import shortestpath as sp


class TestSeedCompetition:

    def test_seed_competition_grid(self):
        seeds = np.array([[1, 0, 0],
                          [0, 0, 0],
                          [0, 2, 0]])
        image = np.empty((3, 3, 2))
        image[:, :, 0] = np.array([[1, 2, 3],
                                   [2, 3, 4],
                                   [2, 2, 3]])
        image[:, :, 1] = np.array([[5, 6, 8],
                                   [6, 8, 9],
                                   [8, 9, 9]])
        costs, roots, preds, labels = sp.seed_competition(seeds, image=image)

        sqrt2 = np.sqrt(2.0)
        np.testing.assert_equal(costs, np.array([[0,     sqrt2, sqrt2],
                                                 [sqrt2, sqrt2,     1],
                                                 [1,         0,     1]]))
        np.testing.assert_equal(roots, np.array([[0, 0, 7],
                                                 [0, 7, 7],
                                                 [7, 7, 7]]))
        np.testing.assert_equal(preds, np.array([[-1, 0, 5],
                                                 [0,  7, 8],
                                                 [7, -1, 7]]))
        np.testing.assert_equal(labels, np.array([[1, 1, 2],
                                                  [1, 2, 2],
                                                  [2, 2, 2]]))

    def test_seed_competition_sparse(self):
        seeds = np.array([1, 0, 0, 0, 2])
        graph = csgraph.csgraph_from_dense([[0, 3, 2, 0, 0],
                                            [3, 0, 0, 3, 1],
                                            [2, 0, 0, 3, 0],
                                            [0, 3, 3, 0, 2],
                                            [0, 1, 0, 2, 0]])
        costs, roots, preds, labels = sp.seed_competition(seeds, graph=graph)

        np.testing.assert_equal(costs, np.array([0, 1, 2, 2, 0]))
        np.testing.assert_equal(roots, np.array([0, 4, 0, 4, 4]))
        np.testing.assert_equal(preds, np.array([-1, 4, 0, 4, -1]))
        np.testing.assert_equal(labels, np.array([1, 2, 1, 2, 2]))
