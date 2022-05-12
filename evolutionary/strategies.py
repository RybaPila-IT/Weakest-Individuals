import random

import numpy as np
from abc import ABC, abstractmethod
from hints.aliases import *


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
        self.__ensure_objective_function()
        # Continue if the objective function is set.
        weakest_individuals = self.__select_weakest_individuals(eval_population)
        mutated_weakest_individuals = self.__mutate(weakest_individuals)
        # Returning population with modified versions of the weakest individuals.
        return self.__evaluate_population(mutated_weakest_individuals) + eval_population[self.__threshold:]

    def set_objective_function(self, obj_func: ObjectiveFunction) -> None:
        self.__obj_func = obj_func

    def __mutate(self, weakest_individuals: Population) -> Population:
        return [i + (np.random.random_sample(len(i)) - 0.5) * self.__mutation_strength for i in weakest_individuals]

    def __select_weakest_individuals(self, eval_population: EvaluatedPopulation) -> Population:
        self.__sort_population(eval_population)
        # Selecting the threshold amount of the weakest individuals.
        return [i[0] for i in eval_population[:self.__threshold]]

    def __evaluate_population(self, population: Population) -> EvaluatedPopulation:
        return [(i, self.__obj_func(i)) for i in population]

    def __ensure_objective_function(self):
        if self.__obj_func is None:
            raise RuntimeError('objective function is not set for MutationStrategy')

    @staticmethod
    def __sort_population(eval_population: EvaluatedPopulation) -> None:
        eval_population.sort(key=lambda i: i[1])


class AverageMirroringStrategy(Strategy):

    def __init__(self, mirroring_strength: float = 5.0, threshold: int = 20):
        self.__mirroring_strength = mirroring_strength
        self.__threshold = threshold
        self.__obj_func = None

    def modify_evaluated_population(self, eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        self.__ensure_objective_function()
        # Continue if the objective function is set.
        weakest_eval_individuals = self.__select_evaluated_weakest_individuals(eval_population)
        avg_individual = self.__average_individual(weakest_eval_individuals)
        mirrored_eval_individuals = self.__mirror_individuals_and_evaluate(weakest_eval_individuals, avg_individual)
        # Returning population with modified versions of the weakest individuals.
        return mirrored_eval_individuals + eval_population[self.__threshold:]

    def set_objective_function(self, obj_func: ObjectiveFunction) -> None:
        self.__obj_func = obj_func

    def __select_evaluated_weakest_individuals(self, eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        self.__sort_population(eval_population)
        # Selecting the threshold amount of the weakest individuals.
        return eval_population[:self.__threshold]

    def __mirror_individuals_and_evaluate(self,
                                          eval_population: EvaluatedPopulation,
                                          avg_individual: Individual) -> EvaluatedPopulation:
        return [
            self.__mirror_single_individual_and_evaluate(i, avg_individual) for i in eval_population
        ]

    @staticmethod
    def __average_individual(eval_population: EvaluatedPopulation) -> Individual:
        return np.average([i[0] for i in eval_population], axis=0)

    def __mirror_single_individual_and_evaluate(self,
                                                eval_individual: EvaluatedIndividual,
                                                avg_individual: Individual) -> EvaluatedIndividual:
        individual, individual_val = eval_individual
        mirroring_vector = avg_individual - individual
        # Initial mirroring performed towards average individual.
        eta = random.uniform(0, 1) * self.__mirroring_strength
        first_mirroring = individual + eta * mirroring_vector
        first_mirror_val = self.__obj_func(first_mirroring)
        if first_mirror_val > individual_val:
            return first_mirroring, first_mirror_val
        # Attempt to find better individual by switching mirroring direction.
        eta = random.uniform(0, 1) * self.__mirroring_strength
        second_mirroring = individual - eta * mirroring_vector
        second_mirroring_val = self.__obj_func(second_mirroring)
        if second_mirroring_val > individual_val:
            return second_mirroring, second_mirroring_val
        # If we failed with mirroring, return original individual.
        return eval_individual

    @staticmethod
    def __sort_population(eval_population: EvaluatedPopulation) -> None:
        eval_population.sort(key=lambda i: i[1])

    def __ensure_objective_function(self):
        if self.__obj_func is None:
            raise RuntimeError('objective function is not set for MirroringStrategy')
