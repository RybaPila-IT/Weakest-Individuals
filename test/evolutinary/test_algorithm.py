import unittest

from evolutionary.algorithm import EvolutionaryAlgorithm


class TestEvolutionaryAlgorithm(unittest.TestCase):
    def test_mutation(self):
        mut_strength = 2
        algorithm = EvolutionaryAlgorithm(None, mut_strength, None, None, None, None)
        population = [[0, 0], [1, 1], [2, 2], [3, 3]]
        # noinspection PyUnresolvedReferences
        result_population = algorithm._EvolutionaryAlgorithm__mutate_population(population)
        for original, result in zip(population, result_population):
            for o_item, r_item in zip(original, result):
                self.assertAlmostEqual(o_item, r_item, delta=mut_strength//2)



if __name__ == '__main__':
    unittest.main()