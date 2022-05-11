from population.generator import PopulationGenerator
from evolutionary.algorithm import EvolutionaryAlgorithm
from logger import Logger

from cec2017.functions import f9


if __name__ == '__main__':
    logger = Logger()
    algorithm = EvolutionaryAlgorithm(f9, logger=logger, verbose=True)
    init_population = PopulationGenerator.generate_population(0, 10, 2, 100)
    _ = algorithm.run(init_population)

    logger.show_log_plots()
    logger.store_log('log.txt')
