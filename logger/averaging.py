import numpy as np
import matplotlib.pyplot as plt
from logger.regular import Logger
from hints.aliases import EvaluatedPopulation


class AveragingLogger:
    """
    Class storing information about multiple EvolutionaryAlgorithm run.

    This class enables a convenient way to store and manage log data
    generated as the result of multiple EvolutionaryAlgorithm runs.

    It is possible to show data in graphical form as plots or
    dump it directly into file for further analysis.
    """
    default_options = {
        'v_max': True,
        'v_min': True,
        'v_avg': True,
    }

    def __init__(self, options: dict | None = None):
        """
        Constructor of the AveragingLogger object.

        Construction is based on the passed dictionary of options.
        This options will be applied to every single run stored in
        the AveragingLogger.

        Available options are:
            - v_max: whether to store current best individual (True/False)
            - v_min: whether to store current worst individual (True/False)
            - v_avg: whether to store current average of individuals (True/False)

        :param options: dictionary containing options used for data collection.
        """
        self.__options = {**AveragingLogger.default_options, **options} \
            if options is not None \
            else AveragingLogger.default_options
        self.__loggers = [
            Logger(self.__options)
        ]

    def clean_up(self) -> None:
        """
        Cleans the AveragingLogger internals after previous runs.
        """
        self.__loggers = []

    def logging_for_new_run(self) -> None:
        """
        Sets AveragingLogger to collect data for the new run.
        """
        self.__loggers.append(
            Logger(self.__options)
        )

    def generate_new_log_entry(self, eval_population: EvaluatedPopulation) -> None:
        """
        Generates new entry for each of collecting logging information.

        This method should be used only by the Experiment class.

        :param eval_population: EvaluatedPopulation of current iteration for which the logs will be generated
        """
        self.__loggers[-1].generate_new_log_entry(eval_population)

    def show_log_plots(self, i: int = -1) -> None:
        """
        Shows the plots generated as the result of the runs.

        Plots are shown accordingly to options passed in the constructor.
        Plots may be shown from some run or from all runs with the averaging
        option.

        :param i: index of the run which plots will be show. When passed -1 the averaging option will be used.
        """
        if i < -1:
            raise RuntimeError(f'invalid i value: {i}')
        if i > -1:
            self.__loggers[i].show_log_plots()
            return
        # Start to present the averaging version of the plots
        plt.figure(figsize=(14, 7), layout='constrained')

        if self.__options['v_max']:
            avg_max = np.mean(
                a=[logger.get_max_run_values() for logger in self.__loggers],
                axis=0
            )
            plt.plot([i for i in range(1, len(avg_max) + 1)], avg_max, label='max')
        if self.__options['v_min']:
            avg_min = np.mean(
                a=[logger.get_min_run_values() for logger in self.__loggers],
                axis=0
            )
            plt.plot([i for i in range(1, len(avg_min) + 1)], avg_min, label='min')
        if self.__options['v_avg']:
            avg_avg = np.mean(
                a=[logger.get_avg_run_values() for logger in self.__loggers],
                axis=0
            )
            plt.plot([i for i in range(1, len(avg_avg) + 1)], avg_avg, label='avg')

        plt.xlabel('Iteration')
        plt.ylabel('Objective function value')
        plt.title(f'Average algorithm results of {len(self.__loggers)} runs')
        plt.legend()
        plt.show()

