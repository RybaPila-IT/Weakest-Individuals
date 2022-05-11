import unittest
import numpy as np

from evolutionary.algorithm import EvolutionaryAlgorithm


def stub_obj_func(_):
    return 0.0


class TestEvolutionaryAlgorithm(unittest.TestCase):
    def test_mutation(self):
        mut_strength = 2
        algorithm = EvolutionaryAlgorithm(stub_obj_func, mutation_strength=mut_strength)
        population = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        # noinspection PyUnresolvedReferences
        result_population = algorithm._EvolutionaryAlgorithm__mutate_population(population)
        for original, result in zip(population, result_population):
            for o_item, r_item in zip(original, result):
                self.assertAlmostEqual(o_item, r_item, delta=mut_strength//2)

    def test_crossover(self):
        crossover_probability = 1
        algorithm = EvolutionaryAlgorithm(stub_obj_func, crossover_probability=crossover_probability)
        population = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        # noinspection PyUnresolvedReferences
        result_population = algorithm._EvolutionaryAlgorithm__crossover_population(population)
        for result in result_population:
            for r_item in result:
                self.assertGreaterEqual(r_item, 0)
                self.assertLessEqual(r_item, 3)

    def test_succession(self):
        elite_size = 2
        algorithm = EvolutionaryAlgorithm(stub_obj_func, elite_size=elite_size)
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
