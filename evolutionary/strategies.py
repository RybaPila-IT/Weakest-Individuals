import numpy as np
from abc import ABC, abstractmethod
from hints.aliases import EvaluatedPopulation, ObjectiveFunction, Population


class Strategy(ABC):
    @abstractmethod
    def modify_evaluated_population(self, eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        pass

    @abstractmethod
    def set_objective_function(self, obj_func: ObjectiveFunction) -> None:
        pass


class MutationStrategy(Strategy):

    def __init__(self, mutation_strength: float = 5, threshold: int = 20):
        self.__mutation_strength = mutation_strength
        self.__threshold = threshold
        self.__obj_func = None

    def modify_evaluated_population(self, eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        eval_population.sort(key=lambda i: i[1])
        weakest_individuals = self.__select_weakest_individuals(eval_population)
        mutated_weakest_individuals = self.__mutate(weakest_individuals)
        eval_mut_weakest_individuals = self.__evaluate_population(mutated_weakest_individuals)
        # Returning population with modified versions of the weakest individuals.
        return eval_mut_weakest_individuals + eval_population[self.__threshold:]

    def set_objective_function(self, obj_func: ObjectiveFunction) -> None:
        self.__obj_func = obj_func

    def __mutate(self, weakest_individuals: Population) -> Population:
        return [i + (np.random.random_sample(len(i)) - 0.5) * self.__mutation_strength for i in weakest_individuals]

    def __select_weakest_individuals(self, eval_population: EvaluatedPopulation) -> Population:
        return [i[0] for i in eval_population[:self.__threshold]]

    def __evaluate_population(self, population: Population) -> EvaluatedPopulation:
        if self.__obj_func is None:
            raise RuntimeError('objective function is not set for MutationStrategy')
        return [(i, self.__obj_func(i)) for i in population]
