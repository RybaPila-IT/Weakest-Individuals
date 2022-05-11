import numpy as np
import matplotlib.pyplot as plt
from hints.aliases import EvaluatedPopulation


class Logger:
    """
    Class storing information about EvolutionaryAlgorithm run.

    This class enables a convenient way to store and manage log data
    generated as the result of EvolutionaryAlgorithm run.

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
        Constructor of the Logger object.

        Construction is based on the passed dictionary of options.

        Available options are:
            - v_max: whether to store current best individual (True/False)
            - v_min: whether to store current worst individual (True/False)
            - v_avg: whether to store current average of individuals (True/False)

        :param options: dictionary containing options used for data collection.
        """
        self.__max = []
        self.__min = []
        self.__avg = []
        self.__options = {**Logger.default_options, **options} \
            if options is not None \
            else Logger.default_options

    def clean_up(self) -> None:
        """
        Cleans the Logger internals after previous run.
        """
        self.__max = []
        self.__min = []
        self.__avg = []

    def generate_new_log_entry(self, eval_population: EvaluatedPopulation) -> None:
        """
        Generates new entry for each of collecting logging information.

        This method should be used only by the EvolutionaryAlgorithm class.

        :param eval_population: EvaluatedPopulation of current iteration for which the logs will be generated
        """
        if self.__options['v_max']:
            self.__max.append(max(eval_population, key=lambda i: i[1])[1])
        if self.__options['v_min']:
            self.__min.append(min(eval_population, key=lambda i: i[1])[1])
        if self.__options['v_avg']:
            self.__avg.append(np.mean([i[1] for i in eval_population]))

    def show_log_plots(self) -> None:
        """
        Shows the plots generated as the result of the EvolutionaryAlgorithm run.

        Plots are shown accordingly to options passed in the constructor.
        """
        plt.figure(figsize=(14, 7), layout='constrained')

        if self.__options['v_max']:
            plt.plot([i for i in range(1, len(self.__max) + 1)], self.__max, label='max')
        if self.__options['v_min']:
            plt.plot([i for i in range(1, len(self.__min) + 1)], self.__min, label='min')
        if self.__options['v_avg']:
            plt.plot([i for i in range(1, len(self.__avg) + 1)], self.__avg, label='avg')

        plt.xlabel('Iteration')
        plt.ylabel('Objective function value')
        plt.title('Algorithm results')
        plt.legend()
        plt.show()

    def store_log(self, file_path: str) -> None:
        """
        Stores generated logs into file.

        :param file_path: path to the file where logs will be stored.
        """
        with open(file_path, 'w') as file:
            if self.__options['v_max']:
                file.write(f'v_max: {self.__max}\n')
            if self.__options['v_min']:
                file.write(f'v_min: {self.__min}\n')
            if self.__options['v_avg']:
                file.write(f'v_avg: {self.__avg}\n')
