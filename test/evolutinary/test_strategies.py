import unittest
import numpy as np
from evolutionary.strategies import MutationStrategy, AverageMirroringStrategy, DifferentialEvolutionStrategy


class TestMutationStrategy(unittest.TestCase):
    def test_strategy(self):
        def stub_obj_func(_):
            return 42
        eval_population = [
            (np.array([1]), 1),
            (np.array([2]), 2),
            (np.array([3]), 3),
            (np.array([4]), 4),
        ]
        mut_strength = 1.0
        threshold = 2
        strategy = MutationStrategy(mut_strength, threshold)
        strategy.set_objective_function(stub_obj_func)
        result_eval_population = strategy.modify_evaluated_population(eval_population)
        (first, first_val), (second, second_val), (third, third_val), (fourth, fourth_val) = result_eval_population
        # First 2 elements should be mutated.
        # It values should be within range 1 +- [-4, 4] with almost 100% chances.
        self.assertAlmostEqual(first[0], 1, delta=4)
        self.assertNotEqual(first[0], 1)
        self.assertEqual(first_val, 42)
        self.assertAlmostEqual(second[0], 2, delta=4)
        self.assertNotEqual(second[0], 2)
        self.assertEqual(second_val, 42)
        # Last 2 elements should not be changed.
        self.assertEqual(third[0], 3)
        self.assertEqual(third_val, 3)
        self.assertEqual(fourth[0], 4)
        self.assertEqual(fourth_val, 4)

    def test_strategy_no_objective_function(self):
        eval_population = [
            (np.array([1]), 1),
            (np.array([2]), 2),
            (np.array([3]), 3),
            (np.array([4]), 4),
        ]
        strategy = MutationStrategy(threshold=2)

        with self.assertRaises(RuntimeError):
            strategy.modify_evaluated_population(eval_population)


class TestAverageMirroringStrategy(unittest.TestCase):
    def test_strategy(self):
        def stub_obj_func(x):
            if x[0] < 1 and x[1] < 1:
                return 0.5
            if 1 < x[0] < 2 and 1 < x[1] < 2:
                return 1.5
            if 2 < x[0] < 3 and 2 < x[1] < 3:
                return 2.5
            return 3.5

        eval_population = [
            (np.array([1, 1]), 1),
            (np.array([2, 2]), 2),
            (np.array([3, 3]), 3),
            (np.array([4, 4]), 4)
        ]
        strategy = AverageMirroringStrategy(mirroring_strength=0.98, threshold=2)
        strategy.set_objective_function(stub_obj_func)
        result_eval_population = strategy.modify_evaluated_population(eval_population)
        (first, first_val), (second, second_val), (third, third_val), (fourth, fourth_val) = result_eval_population
        # First element should be mirrored to average side.
        self.assertLess(first[0], 2)
        self.assertGreater(first[0], 1)
        self.assertLess(first[1], 2)
        self.assertGreater(first[1], 1)
        self.assertEqual(first_val, 1.5)
        # Second element should also be mirrored opposite to average side.
        self.assertGreater(second[0], 2)
        self.assertLess(second[0], 3)
        self.assertGreater(second[1], 2)
        self.assertLess(second[1], 3)
        self.assertEqual(second_val, 2.5)
        # Last 2 elements should not be changed.
        self.assertTrue((third == np.array([3, 3])).all())
        self.assertEqual(third_val, 3)
        self.assertTrue((fourth == np.array([4, 4])).all())
        self.assertEqual(fourth_val, 4)

    def test_strategy_no_objective_function(self):
        eval_population = [
            (np.array([1]), 1),
            (np.array([2]), 2),
            (np.array([3]), 3),
            (np.array([4]), 4),
        ]
        strategy = AverageMirroringStrategy(threshold=2)

        with self.assertRaises(RuntimeError):
            strategy.modify_evaluated_population(eval_population)


class TestDifferentialEvolutionStrategy(unittest.TestCase):

    def test_strategy(self):
        def stub_obj_func(_):
            return 42

        eval_population = [
            (np.array([0, 0]), 1),
            (np.array([1, 0]), 2),
            (np.array([0, 1]), 3)
        ]
        strategy = DifferentialEvolutionStrategy(best_strength=0.5, other_strength=0.5, threshold=2)
        strategy.set_objective_function(stub_obj_func)
        result_eval_population = strategy.modify_evaluated_population(eval_population)
        (first, first_val), (second, second_val), (third, third_val) = result_eval_population
        # First point should be moved towards point (0,1) and possibly towards point (1, 0)
        self.assertLess(first[0], 1)
        self.assertGreaterEqual(first[0], 0)
        self.assertGreater(first[1], 0)
        self.assertLess(first[1], 1)
        self.assertEqual(first_val, 42)
        # Second point should be moved towards point (0, 1) and possibly towards point (0, 0)
        self.assertLessEqual(second[0], 1)
        self.assertGreater(second[0], 0)
        self.assertGreater(second[1], 0)
        self.assertLess(second[1], 1)
        self.assertEqual(second_val, 42)
        # Point (0, 1) should be unchanged
        self.assertTrue((third == np.array([0, 1])).all())
        self.assertEqual(third_val, 3)

    def test_strategy_no_objective_function(self):
        eval_population = [
            (np.array([1]), 1),
            (np.array([2]), 2),
            (np.array([3]), 3),
            (np.array([4]), 4),
        ]
        strategy = DifferentialEvolutionStrategy(threshold=2)

        with self.assertRaises(RuntimeError):
            strategy.modify_evaluated_population(eval_population)


if __name__ == '__main__':
    unittest.main()
