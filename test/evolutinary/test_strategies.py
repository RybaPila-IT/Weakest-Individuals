import unittest
import numpy as np
from evolutionary.strategies import MutationStrategy


def stub_obj_func(_):
    return 42


class TestMutationStrategy(unittest.TestCase):
    def test_strategy(self):
        eval_population = [
            (np.array([1]), 1),
            (np.array([2]), 2),
            (np.array([3]), 3),
            (np.array([4]), 4),
        ]
        mut_strength = 2.0
        threshold = 2
        strategy = MutationStrategy(mut_strength, threshold)
        result_eval_population = strategy.modify_evaluated_population(eval_population, stub_obj_func)
        (first, first_val), (second, second_val), (third, third_val), (fourth, fourth_val) = result_eval_population
        # First 2 elements should be mutated.
        self.assertAlmostEqual(first[0], 1, delta=mut_strength/2)
        self.assertNotEqual(first[0], 1)
        self.assertEqual(first_val, 42)
        self.assertAlmostEqual(second[0], 2, delta=mut_strength/2)
        self.assertNotEqual(second[0], 2)
        self.assertEqual(second_val, 42)
        # Last 2 elements should not be changed.
        self.assertEqual(third[0], 3)
        self.assertEqual(third_val, 3)
        self.assertEqual(fourth[0], 4)
        self.assertEqual(fourth_val, 4)


if __name__ == '__main__':
    unittest.main()
