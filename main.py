from population.generator import PopulationGenerator
from evolutionary.algorithm import EvolutionaryAlgorithm
from evolutionary.strategies import MutationStrategy
from experiment import Experiment
from cec2017.functions import f9
from cec2017.negate import negate


def generator():
    return PopulationGenerator.generate_population_uniform_distribution(0, 10, 2, 100)


if __name__ == '__main__':
    strategy = MutationStrategy()
    algorithm = EvolutionaryAlgorithm(objective_function=negate(f9), strategy=strategy, verbose=False)
    experiment = Experiment(algorithm, generator, duration=10, log_file_path='log.txt')
    experiment.conduct()
