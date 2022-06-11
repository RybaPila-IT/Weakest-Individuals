from population.generator import PopulationGenerator
from evolutionary.algorithm import EvolutionaryAlgorithm
from evolutionary.strategies import MutationStrategy
from logger import Logger
from cec2017.functions import f4
from cec2017.negate import negate


if __name__ == '__main__':
    logger = Logger()
    strategy = MutationStrategy()
    algorithm = EvolutionaryAlgorithm(objective_function=negate(f4), strategy=strategy, logger=logger, verbose=True)
    init_population = PopulationGenerator.generate_population_uniform_distribution(0, 10, 2, 100)
    _ = algorithm.run(init_population)

    logger.show_log_plots()
    logger.store_log('log.txt')
