import numpy as np
from hints.aliases import Population


class PopulationGenerator:
    @staticmethod
    def generate_population_normal_distribution(loc: float | list[float],
                                                scale: float | list[float],
                                                individual_size: int,
                                                population_size: int) -> Population:
        """
        Generates population with normal distribution.

        :param loc: mean (“centre”) of the distribution
        :param scale: standard deviation (spread or “width”) of the distribution
        :param individual_size: size of the single individual which will be generated
        :param population_size: size of the whole population which will be generated

        :return: Generated population
        """
        return [np.random.normal(loc, scale, individual_size) for _ in range(population_size)]

    @staticmethod
    def generate_population_uniform_distribution(low: float | list[float],
                                                 high: float | list[float],
                                                 individual_size: int,
                                                 population_size: int) -> Population:
        """
        Generates population with uniform distribution.

        :param low: lower bound of the uniform distribution
        :param high: upper bound of the uniform distribution
        :param individual_size: size of the single individual which will be generated
        :param population_size: size of the whole population which will be generated

        :return: Generated population
        """
        return [np.random.uniform(low, high, individual_size) for _ in range(population_size)]
