from population.generator import PopulationGenerator
from evolutionary.algorithm import EvolutionaryAlgorithm
from evolutionary.strategies import *
from experiment.comparison import ExperimentComparison
from cec2017.functions import f10
from cec2017.negate import negate


def generator():
    return PopulationGenerator.generate_population_uniform_distribution(0, 10, 2, 100)


if __name__ == '__main__':
    # Example of the usage of ExperimentComparison class.
    strategy0 = None
    strategy1 = MutationStrategy()
    strategy2 = AverageMirroringStrategy()
    strategy3 = DifferentialEvolutionStrategy()

    algorithm0 = EvolutionaryAlgorithm(strategy=strategy0, verbose=False)
    algorithm1 = EvolutionaryAlgorithm(strategy=strategy1, verbose=False)
    algorithm2 = EvolutionaryAlgorithm(strategy=strategy2, verbose=False)
    algorithm3 = EvolutionaryAlgorithm(strategy=strategy3, verbose=False)

    algorithms = [
        algorithm0,
        algorithm1,
        algorithm2,
        algorithm3
    ]
    algorithm_names = [
        'No strategy',
        'Mutation',
        'Mirroring',
        'Differential'
    ]
    obj_func = negate(f10)
    duration = 10

    experiment_comparison = ExperimentComparison(
        algorithms,
        algorithm_names,
        generator,
        obj_func,
        duration
    )

    experiment_comparison.conduct()

