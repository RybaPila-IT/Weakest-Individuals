#!/usr/bin/env python
# coding: utf-8

import math
import random
import numpy as np
import cec2017
from cec2017.functions import f1
from cec2017.functions import f2
from cec2017.functions import f4
from cec2017.functions import f5
from tabulate import tabulate


def construct_individual_obj_fun_value_pairs(population, obj_function, population_size):
    population_with_obj_fun_value = []
    for i in range(0, population_size):
        population_with_obj_fun_value.append((population[i], obj_function(population[i])))

    return population_with_obj_fun_value


def tournament_selection(population_with_obj_fun_value):
    population_size = len(population_with_obj_fun_value)

    reproduced_individuals = []
    for i in range(0, population_size):
        tournament_members = random.choices(population_with_obj_fun_value, k=2)
        if tournament_members[0][1] < tournament_members[1][1]:
            reproduced_individuals.append(tournament_members[0])
        else:
            reproduced_individuals.append(tournament_members[1])

    return reproduced_individuals


def mutate_population(population_with_obj_fun_value, mutation_strength, upper_bound=100, lower_bound=-100):
    population_size = len(population_with_obj_fun_value)
    dimensionality = len(population_with_obj_fun_value[0][0])

    mutated_individuals = []
    for i in range(0, population_size):
        # print("Before mutation: ", population_with_obj_fun_value[i])
        mutated_individual = population_with_obj_fun_value[i][0] + mutation_strength * np.random.normal(
            size=dimensionality)
        mutated_individual[mutated_individual < lower_bound] = lower_bound
        mutated_individual[mutated_individual > upper_bound] = upper_bound
        mutated_individuals.append(mutated_individual)
        # print("After mutation: ", mutated_individual)

    return mutated_individuals


def perform_succession(new_population_with_val, old_population_with_val, elite_size):
    new_population_with_val.sort(key=lambda x: x[1])
    old_population_with_val.sort(key=lambda x: x[1])
    population_size = len(new_population_with_val)

    merged_population = []
    merged_population.extend(old_population_with_val[:elite_size])
    merged_population.extend(new_population_with_val[:population_size - elite_size])

    return merged_population


def evolutionary_algorithm(population_size, mutation_strength, elite_size, obj_function, initial_population):
    objective_function_budget = 10000
    max_iter = math.floor(objective_function_budget / population_size)

    population_with_obj_fun_value = construct_individual_obj_fun_value_pairs(initial_population, obj_function,
                                                                             population_size)
    best_individual = min(population_with_obj_fun_value, key=lambda x: x[1])

    for i in range(0, max_iter):
        reproduced_individuals = tournament_selection(population_with_obj_fun_value)

        mutated_individuals = mutate_population(reproduced_individuals, mutation_strength)
        mutated_individuals_with_obj_fun_value = construct_individual_obj_fun_value_pairs(mutated_individuals,
                                                                                          obj_function, population_size)

        best_from_mutated = min(mutated_individuals_with_obj_fun_value, key=lambda x: x[1])
        if (best_from_mutated[1] <= best_individual[1]):
            best_individual = best_from_mutated

        population_with_obj_fun_value = perform_succession(mutated_individuals_with_obj_fun_value,
                                                           population_with_obj_fun_value, elite_size)
    #         print(best_individual[1])

    return best_individual


# Test działania
def generate_algorithm_results(pop_size, mutation_strength, elite_size, obj_function):
    upper_bound = 100
    lower_bound = -100
    headers = ["Populacja", "Siła mutacji", "Elita", "Wartość min", "Wartość śr", "Odchylenie standardowe",
               "Wartość max"]
    data = []

    for psize in pop_size:
        for mstr in mutation_strength:
            for esize in elite_size:

                results = []
                for i in range(0, 50):
                    initial_population = []
                    for j in range(0, psize):
                        initial_population.append(np.random.uniform(lower_bound, upper_bound, 10))
                    result = evolutionary_algorithm(psize, mstr, esize, obj_function, initial_population)
                    results.append(result[1])
                v_min = min(results)
                v_max = max(results)
                v_std = np.std(results)
                v_avg = np.average(results)

                data.append([psize, mstr, esize, v_min, v_avg, v_std, v_max])

    return headers, data
