import random
import numpy as np
from logger import Logger
from evolutionary.strategies import Strategy
from hints.aliases import *


class EvolutionaryAlgorithm:
    """
    Class implementing used evolutionary algorithm for experiments.

    This particular class implements genetic algorithm. For more
    information visit this Wikipedia page [1].

    [1]: https://en.wikipedia.org/wiki/Genetic_algorithm
    """

    def __init__(self,
                 objective_function: ObjectiveFunction,
                 strategy: type(Strategy) | None = None,
                 mutation_strength: float = 2.0,
                 crossover_probability: float = 0.3,
                 elite_size: int = 2,
                 iterations: int = 500,
                 population_size: int = 100,
                 logger: type(Logger) | None = None,
                 verbose: bool = False):
        # Default values should be changed after algorithm tuning.
        self.__obj_fun = objective_function
        self.__strategy = strategy
        self.__mutation_strength = mutation_strength
        self.__crossover_probability = crossover_probability
        self.__elite_size = elite_size
        self.__iterations = iterations
        self.__population_size = population_size
        self.__logger = logger
        self.__verbose = verbose
        self.__best_individual_with_score = None

    def run(self, init_population: Population) -> EvaluatedIndividual:
        """
        Perform the EvolutionaryAlgorithm execution.

        IMPORTANT: Initial population size must match the evolutionary algorithm
        population size parameter.

        :param init_population: initial population for the start-up
        :return: the best achieved individual with evaluation
        """
        self.__clean_up()
        self.__prepare_strategy()
        self.__ensure_legit_size(init_population)
        # Initial evaluation for algorithm start-up.
        old_eval_population = self.__evaluate_population(init_population)

        for i in range(self.__iterations):
            # Strategy is applied as the first step of the algorithm.
            if self.__strategy is not None:
                old_eval_population = self.__strategy.modify_evaluated_population(old_eval_population)
            # Regular genetic algorithm steps follow.
            new_eval_population = self.__generate_new_evaluated_population(old_eval_population)
            # Since we use elite succession we can select best individual this way.
            self.__pick_best_individual(new_eval_population)
            # Logs storing for further algorithm analysis.
            if self.__logger is not None:
                self.__logger.generate_new_log_entry(new_eval_population)
            if self.__verbose:
                print(f'Iteration {i + 1} finished')
            # New population becomes the old one for next iteration.
            old_eval_population = new_eval_population

        return self.__best_individual_with_score

    def __ensure_legit_size(self, population):
        if len(population) != self.__population_size:
            raise RuntimeError('invalid initial population size')

    def __clean_up(self) -> None:
        self.__best_individual_with_score = None
        self.__logger.clean_up()

    def __prepare_strategy(self):
        if self.__strategy is not None:
            self.__strategy.set_objective_function(self.__obj_fun)

    def __generate_new_evaluated_population(self, old_eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        # Basically genetic algorithm steps.
        selected_individuals = self.__tournament_selection(old_eval_population)
        crossed_individuals = self.__crossover_population(selected_individuals)
        mutated_population = self.__mutate_population(crossed_individuals)
        new_eval_population = self.__evaluate_population(mutated_population)
        return self.__make_succession(old_eval_population, new_eval_population)

    def __evaluate_population(self, population: Population) -> EvaluatedPopulation:
        return [(i, self.__obj_fun(i)) for i in population]

    def __tournament_selection(self, evaluated_population: EvaluatedPopulation) -> Population:
        reproduced_individuals = []
        for _ in range(self.__population_size):
            tournament_members = random.choices(evaluated_population, k=2)
            tournament_winner = tournament_members[0][0] \
                if tournament_members[0][1] > tournament_members[1][1] \
                else tournament_members[1][0]
            reproduced_individuals.append(tournament_winner)

        return reproduced_individuals

    def __crossover_population(self, population: Population) -> Population:
        result_individuals = []
        for _ in range(self.__population_size):
            eta = random.uniform(0, 1)
            weight = random.uniform(0, 1)
            fathers = random.choices(population, k=2)
            if eta < self.__crossover_probability:
                result_individuals.append(fathers[0] * weight + fathers[1] * (1 - weight))
            else:
                result_individuals.append(fathers[0])

        return result_individuals

    def __mutate_population(self, population: Population) -> Population:
        return [i + (np.random.standard_normal(len(i))) * self.__mutation_strength for i in population]

    def __make_succession(self,
                          old_eval_population: EvaluatedPopulation,
                          new_eval_population: EvaluatedPopulation) -> EvaluatedPopulation:
        old_eval_population.sort(reverse=True, key=lambda i: i[1])
        new_eval_population.sort(reverse=True, key=lambda i: i[1])
        return old_eval_population[:self.__elite_size] + \
            new_eval_population[:self.__population_size - self.__elite_size]

    def __pick_best_individual(self, evaluated_population: EvaluatedPopulation) -> None:
        self.__best_individual_with_score = max(evaluated_population, key=lambda i: i[1])
