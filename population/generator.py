import numpy as np
from custom_types import Population


class PopulationGenerator:
    @staticmethod
    def generate_population(loc: float | list[float],
                            scale: float | list[float],
                            individual_size: int,
                            population_size: int) -> Population:
        """
        Generates population with normal distribution.

        :param loc: mean (“centre”) of the distribution
        :param scale: standard deviation (spread or “width”) of the distribution
        :param individual_size: size of the single individual which will be generated
        :param population_size: size of the whole population which will be generated

        :return: generated population.
        """
        return [np.random.normal(loc, scale, individual_size) for _ in range(population_size)]
