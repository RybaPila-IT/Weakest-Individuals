import numpy as np
from abc import ABC, abstractmethod
from hints.aliases import EvaluatedPopulation, ObjectiveFunction, Population


class Strategy(ABC):
    @abstractmethod
    def modify_evaluated_population(self, eval_population: EvaluatedPopulation,
                                    obj_func: ObjectiveFunction) -> EvaluatedPopulation:
        pass


class MutationStrategy(Strategy):

    def __init__(self, mutation_strength: float = 5, threshold: int = 20):
        self.__mutation_strength = mutation_strength
        self.__threshold = threshold

    def modify_evaluated_population(self, eval_population: EvaluatedPopulation,
                                    obj_func: ObjectiveFunction) -> EvaluatedPopulation:
        eval_population.sort(key=lambda i: i[1])
        weakest_individuals = self.__select_weakest_individuals(eval_population)
        mutated_weakest_individuals = self.__mutate(weakest_individuals)
        eval_mut_weakest_individuals = self.__evaluate_population(mutated_weakest_individuals, obj_func)
        # Returning population with modified versions of the weakest individuals.
        return eval_mut_weakest_individuals + eval_population[self.__threshold:]

    def __mutate(self, weakest_individuals: Population) -> Population:
        return [i + (np.random.random_sample(len(i)) - 0.5) * self.__mutation_strength for i in weakest_individuals]

    def __select_weakest_individuals(self, eval_population: EvaluatedPopulation) -> Population:
        return [i[0] for i in eval_population[:self.__threshold]]

    @staticmethod
    def __evaluate_population(population: Population, obj_func: ObjectiveFunction) -> EvaluatedPopulation:
        return [(i, obj_func(i)) for i in population]
