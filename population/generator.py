import numpy as np
from custom_types import Population


class PopulationGenerator:
    @staticmethod
    def generate_population(loc: float | list[float],
                            scale: float | list[float],
                            individual_size: int,
                            population_size: int) -> Population:
        return [np.random.normal(loc, scale, individual_size) for _ in range(population_size)]
