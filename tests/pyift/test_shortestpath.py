import numpy as np
from scipy.sparse import csgraph
from pyift import shortestpath as sp


class TestSeedCompetition:
    seeds = np.array([[1, 0, 0],
                      [0, 0, 0],
                      [0, 2, 0]])

    image = np.array([[[1, 5], [2, 6], [3, 8]],
                      [[2, 6], [3, 8], [4, 9]],
                      [[2, 8], [2, 9], [3, 9]]])

    def test_seed_competition_grid(self):
        costs, roots, preds, labels = sp.seed_competition(self.seeds, image=self.image)

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

    def test_error_handling(self):
        with np.testing.assert_raises(TypeError):
            sp.seed_competition(self.seeds, image=0)

        with np.testing.assert_raises(TypeError):
            sp.seed_competition(self.seeds.flatten(), graph=0)

        with np.testing.assert_raises(TypeError):
            sp.dynamic_arc_weight(self.seeds, image=0)

        with np.testing.assert_raises(ValueError):
            sp.seed_competition(self.seeds, np.ones(self.seeds.size))

        with np.testing.assert_raises(ValueError):
            sp.dynamic_arc_weight(self.seeds, np.ones(self.seeds.size))

        with np.testing.assert_raises(ValueError):
            sp.seed_competition(self.seeds)

        with np.testing.assert_raises(ValueError):
            sp.seed_competition(self.seeds, image=self.image,
                                graph=csgraph.csgraph_from_dense(self.image))

    def test_dynamic_arc_weight_grid(self):
        costs, roots, preds, labels, trees = sp.dynamic_arc_weight(self.seeds, self.image)

        sqrt2 = np.sqrt(2.0)
        np.testing.assert_equal(costs, np.array([[0,     sqrt2, 1.5],
                                                 [sqrt2, sqrt2, 1.5],
                                                 [1,         0,   1]]))
        np.testing.assert_equal(roots, np.array([[0, 0, 7],
                                                 [0, 7, 7],
                                                 [7, 7, 7]]))
        np.testing.assert_equal(preds, np.array([[-1, 0, 5],
                                                 [0,  7, 8],
                                                 [7, -1, 7]]))
        np.testing.assert_equal(labels, np.array([[1, 1, 2],
                                                  [1, 2, 2],
                                                  [2, 2, 2]]))

        expected_trees = {(0, 0): (3, np.array([5/3, 17/3])),
                          (2, 1): (6, np.array([17/6, 8.5]))}
        np.testing.assert_(expected_trees.keys() == trees.keys())

        for e_tree, tree in zip(expected_trees.values(), trees.values()):
            np.testing.assert_(e_tree[0] == tree[0])
            np.testing.assert_allclose(e_tree[1], tree[1])
