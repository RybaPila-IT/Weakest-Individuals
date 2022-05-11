import random
import numpy as np
from logger import Logger
from custom_types import *


class EvolutionaryAlgorithm:
    def __init__(self,
                 objective_function: ObjectiveFunction,
                 mutation_strength: float = 2.0,
                 crossover_probability: float = 0.3,
                 elite_size: int = 2,
                 iterations: int = 500,
                 logger: Logger | None = None):
        self.__obj_fun = objective_function
        self.__mutation_strength = mutation_strength
        self.__crossover_probability = crossover_probability
        self.__elite_size = elite_size
        self.__iterations = iterations
        self.__logger = logger
        self.__best_individual_with_score = None

    def run(self, init_population: Population) -> EvaluatedIndividual:
        self.__clean_up()
        # Initial evaluation for algorithm start-up.
        old_eval_population = self.__evaluate_population(init_population)

        for _ in range(self.__iterations):
            selected_individuals = self.__tournament_selection(old_eval_population)
            crossed_individuals = self.__crossover_population(selected_individuals)
            mutated_population = self.__mutate_population(crossed_individuals)
            new_eval_population = self.__evaluate_population(mutated_population)
            old_eval_population = self.__make_succession(old_eval_population, new_eval_population)
            # Since we use elite succession we can select best individual
            # from the population after succession.
            self.__pick_best_individual(old_eval_population)

        return self.__best_individual_with_score

    def __clean_up(self) -> None:
        self.__best_individual_with_score = None

    def __evaluate_population(self, population: Population) -> EvaluatedPopulation:
        return [(i, self.__obj_fun(i)) for i in population]

    @staticmethod
    def __tournament_selection(evaluated_population: EvaluatedPopulation) -> Population:
        reproduced_individuals = []
        for _ in range(len(evaluated_population)):
            tournament_members = random.choices(evaluated_population, k=2)
            tournament_winner = tournament_members[0][0] \
                if tournament_members[0][1] > tournament_members[1][1] \
                else tournament_members[1][0]
            reproduced_individuals.append(tournament_winner)

        return reproduced_individuals

    def __crossover_population(self, population: Population) -> Population:
        result_individuals = []
        for _ in range(len(population)):
            eta = random.uniform(0, 1)
            weight = random.uniform(0, 1)
            fathers = random.choices(population, k=2)
            if eta < self.__crossover_probability:
                result_individuals.append(fathers[0] * weight + fathers[1] * (1 - weight))
            else:
                result_individuals.append(fathers[0])

        return result_individuals

    def __mutate_population(self, population: Population) -> Population:
        return [i + (np.random.random_sample(len(i)) - 0.5) * self.__mutation_strength for i in population]

    def __make_succession(self,
                          old_eval_population: EvaluatedPopulation,
                          new_eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        size = len(old_eval_population)
        old_eval_population.sort(reverse=True, key=lambda i: i[1])
        new_eval_population.sort(reverse=True, key=lambda i: i[1])
        return old_eval_population[:self.__elite_size] + new_eval_population[:size - self.__elite_size]

    def __pick_best_individual(self, evaluated_population: EvaluatedPopulation) -> None:
        self.__best_individual_with_score = max(evaluated_population, key=lambda i: i[1])
