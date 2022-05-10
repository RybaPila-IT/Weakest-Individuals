import unittest
import numpy as np

from evolutionary.algorithm import EvolutionaryAlgorithm


class TestEvolutionaryAlgorithm(unittest.TestCase):
    def test_mutation(self):
        mut_strength = 2
        algorithm = EvolutionaryAlgorithm(None, mut_strength, None, None, None, None)
        population = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        # noinspection PyUnresolvedReferences
        result_population = algorithm._EvolutionaryAlgorithm__mutate_population(population)
        for original, result in zip(population, result_population):
            for o_item, r_item in zip(original, result):
                self.assertAlmostEqual(o_item, r_item, delta=mut_strength//2)

    def test_crossover(self):
        mut_probability = 1
        algorithm = EvolutionaryAlgorithm(None, None, mut_probability, None, None, None)
        population = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        # noinspection PyUnresolvedReferences
        result_population = algorithm._EvolutionaryAlgorithm__crossover_population(population)
        for result in result_population:
            for r_item in result:
                self.assertGreaterEqual(r_item, 0)
                self.assertLessEqual(r_item, 3)

    def test_succession(self):
        elite_size = 2
        algorithm = EvolutionaryAlgorithm(None, None, None, elite_size, None, None)
        old_eval_population = [
            ([0], 12),
            ([1], 36),
            ([2], -12),
            ([3], -36),
            ([4], 48),
            ([5], 0),
        ]
        new_eval_population = [
            ([6], 100),
            ([7], 0),
            ([8], 0),
            ([9], 12),
            ([10], 32),
            ([11], 42),
        ]
        expected_result = [
            ([4], 48),
            ([1], 36),
            ([6], 100),
            ([11], 42),
            ([10], 32),
            ([9], 12),
        ]
        # noinspection PyUnresolvedReferences
        result_population = algorithm._EvolutionaryAlgorithm__make_succession(old_eval_population, new_eval_population)
        self.assertEqual(expected_result, result_population)


if __name__ == '__main__':
    unittest.main()
