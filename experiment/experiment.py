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
                 objective_function: ObjectiveFunction,
                 duration: int = 20,
                 verbose: bool = True,
                 show_plots: bool = True,
                 log_file_path: str | None = None):
        """
        Initializes the Experiment object.

        :param algorithm: Algorithm which wille be used during the experiment.
        :param generator: Generator of the initial population for the algorithm.
        :param objective_function: Objective function for the algorithm.
        :param duration: Duration of the experiment (how many times' algorithm will be run).
        :param show_plots: Whether to show the plots or not after finish.
        :param verbose: If provide verbose feedback during experiment conduction.
        :param log_file_path: Path to the file where logs will be stored. If None no log will be stored.
        """
        self.__algorithm = algorithm
        self.__population_generator = generator
        self.__objective_function = objective_function
        self.__duration = duration
        self.__verbose = verbose
        self.__show_plots = show_plots
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
            self.__algorithm.set_objective_function(
                self.__objective_function
            )
            self.__algorithm.run(
                self.__population_generator()
            )

        if self.__show_plots:
            self.__avg_logger.show_log_plots()
        if self.__log_file_path:
            self.__avg_logger.store_log(self.__log_file_path)

    def results(self) -> dict:
        """
        Returns averaged results from the experiment gathered by the averaging logger.
        """
        return self.__avg_logger.get_logger_data()
