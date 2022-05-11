import random

import numpy as np
from abc import ABC, abstractmethod
from hints.aliases import EvaluatedPopulation, ObjectiveFunction, Population, Individual


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
        self.__sort_population(eval_population)
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
        self.__sort_population(eval_population)
        weakest_individuals = self.__select_weakest_individuals(eval_population)
        mirrored_individuals = self.__mirror_individuals(weakest_individuals)
        eval_mirrored_individuals = self.__evaluate_population(mirrored_individuals)
        # Returning population with modified versions of the weakest individuals.
        return eval_mirrored_individuals + eval_population[self.__threshold:]

    def set_objective_function(self, obj_func: ObjectiveFunction) -> None:
        self.__obj_func = obj_func

    def __select_weakest_individuals(self, eval_population: EvaluatedPopulation) -> Population:
        return [i[0] for i in eval_population[:self.__threshold]]

    def __mirror_individuals(self, population: Population) -> Population:
        avg_individual = self.__average_individual(population)
        return [self.__mirror_single_individual(i, avg_individual) for i in population]

    @staticmethod
    def __average_individual(population: Population) -> Individual:
        return np.average(population, axis=0)

    def __mirror_single_individual(self, individual: Individual, avg_individual: Individual) -> Individual:
        individual_val = self.__obj_func(individual)
        mirroring_vector = avg_individual - individual
        # Initial mirroring performed towards average individual.
        eta = random.uniform(0, 1) * self.__mirroring_strength
        first_mirroring = individual + eta * mirroring_vector
        if self.__obj_func(first_mirroring) > individual_val:
            return first_mirroring
        # Attempt to find better individual by switching mirroring direction.
        eta = random.uniform(0, 1) * self.__mirroring_strength
        second_mirroring = individual + eta * mirroring_vector
        if self.__obj_func(second_mirroring) > individual_val:
            return second_mirroring
        # If we failed with mirroring, return original individual.
        return individual

    def __evaluate_population(self, population: Population) -> EvaluatedPopulation:
        return [(i, self.__obj_func(i)) for i in population]

    @staticmethod
    def __sort_population(eval_population: EvaluatedPopulation) -> None:
        eval_population.sort(key=lambda i: i[1])

    def __ensure_objective_function(self):
        if self.__obj_func is None:
            raise RuntimeError('objective function is not set for MirroringStrategy')
