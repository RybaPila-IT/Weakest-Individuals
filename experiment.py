from logger.averaging import AveragingLogger
from evolutionary.algorithm import *


class Experiment:
    """
    Class representing the Experiment object.

    Experiment consist of multiple EvolutionaryAlgorithm execution
    on provided objective function.
    Eventually the averaged results of all runs will be presented.
    """

    def __init__(self,
                 algorithm: EvolutionaryAlgorithm,
                 obj_func: ObjectiveFunction,
                 generator: GeneratePopulationFunction,
                 duration: int = 20):
        """
        Initializes the Experiment object.

        :param algorithm: Algorithm which wille be used during the experiment
        :param generator: Generator of the initial population for the algorithm
        :param duration: Duration of the experiment (how many times' algorithm will be run)
        """
        self.__algorithm = algorithm
        self.__obj_func = obj_func
        self.__population_generator = generator
        self.__duration = duration
        self.__avg_logger = AveragingLogger()

    def conduct(self) -> None:
        """
        Conducts the experiment.
        """

        self.__avg_logger.clean_up()

        for _ in range(self.__duration):
            self.__avg_logger.logging_for_new_run()
            self.__algorithm.set_logger(
                self.__avg_logger.get_logging_logger()
            )
            self.__algorithm.run(
                self.__population_generator()
            )

        self.__avg_logger.show_log_plots()
