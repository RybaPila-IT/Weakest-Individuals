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
                 generator: GeneratePopulationFunction,
                 duration: int = 20,
                 verbose: bool = True,
                 log_file_path: str | None = None):
        """
        Initializes the Experiment object.

        :param algorithm: Algorithm which wille be used during the experiment.
        :param generator: Generator of the initial population for the algorithm.
        :param duration: Duration of the experiment (how many times' algorithm will be run).
        :param verbose: If provide verbose feedback during experiment conduction.
        :param log_file_path: Path to the file where logs will be stored. If None no log will be stored.
        """
        self.__algorithm = algorithm
        self.__population_generator = generator
        self.__duration = duration
        self.__verbose = verbose
        self.__log_file_path = log_file_path
        self.__avg_logger = AveragingLogger()

    def conduct(self) -> None:
        """
        Conducts the experiment.
        """

        self.__avg_logger.clean_up()

        for i in range(self.__duration):

            if self.__verbose:
                print(f'Experiment iteration {i + 1} started')

            self.__avg_logger.logging_for_new_run()
            self.__algorithm.set_logger(
                self.__avg_logger.get_logging_logger()
            )
            self.__algorithm.run(
                self.__population_generator()
            )

        self.__avg_logger.show_log_plots()
        # Optionally store log to the file.
        if self.__log_file_path:
            self.__avg_logger.store_log(self.__log_file_path)
