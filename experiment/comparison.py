import matplotlib.pyplot as plt
from experiment.experiment import Experiment
from evolutionary.algorithm import EvolutionaryAlgorithm
from hints.aliases import *


class ExperimentComparison:
    """
    Class enabling to conduct multiple experiments and compare them directly.

    Comparison will be conducted with the usage of plots showing different
    experiments results. Plots will be grouped into:
        - Average max value comparison
        - Average min value comparison
        - Average avg value comparison
    and showed for every experiment accordingly.
    """

    data_plots = ['avg_v_max', 'avg_v_min', 'avg_v_avg']
    plot_names = {
        'avg_v_max': 'Average max value comparison',
        'avg_v_min': 'Average min value comparison',
        'avg_v_avg': 'Average avg value comparison'
    }

    def __init__(self,
                 algorithms: list[type(EvolutionaryAlgorithm)],
                 algorithm_names: list[str],
                 generator: GeneratePopulationFunction,
                 obj_function: ObjectiveFunction,
                 duration: int = 20,
                 verbose: bool = True):
        """
        Constructs the Experiment Comparison object.

        :param algorithms: Algorithms which will be compared during experiments.
        :param algorithm_names: Names of the algorithms for better plotting.
        :param generator: Generator of the initial population for Experiment.
        :param obj_function: Objective function for the Experiment.
        :param duration: Duration of the single Experiment.
        :param verbose: Whether to give verbose feedback.
        """
        self.__algorithms = algorithms
        self.__algorithm_names = algorithm_names
        self.__generator = generator
        self.__obj_fun = obj_function
        self.__duration = duration
        self.__verbose = verbose

    def conduct(self) -> None:
        """
        Conducts the series of Experiments and shows their comparison.
        """
        experiment_results = []
        # Perform the experiments and collect the data.
        for algo, name in zip(self.__algorithms, self.__algorithm_names):
            if self.__verbose:
                print(f'Running experiment for algorithm {name}')
            # Start of the experiment.
            experiment = Experiment(
                algorithm=algo,
                objective_function=self.__obj_fun,
                generator=self.__generator,
                duration=self.__duration,
                verbose=False,
                show_plots=False,
                log_file_path=None
            )
            experiment.conduct()
            experiment_results.append(
                (name, experiment.results())
            )
        # Show plots accordingly to options.
        for data_plot in self.data_plots:
            # Extract the only information we need for this plot.
            results_to_show = [(name, results[data_plot]) for name, results in experiment_results]
            # Extract the title of the plot.
            plot_title = self.plot_names[data_plot]
            # Show the plot eventually.
            self.__show_plot(results_to_show, plot_title)
            self.__last_result_info(results_to_show, plot_title)

    @staticmethod
    def __show_plot(results_to_show: list[tuple[str, list[float]]], plot_title: str) -> None:
        plt.figure(figsize=(14, 7), layout='constrained')
        # Actual plotting part.
        for algo_name, result in results_to_show:
            plt.plot([i+1 for i in range(len(result))], result, label=algo_name)
        # Setting the labels for better visual context.
        plt.xlabel('Iteration')
        plt.ylabel('Objective function value')
        plt.title(plot_title)
        plt.legend()
        plt.show()

    @staticmethod
    def __last_result_info(results_to_show: list[tuple[str, list[float]]], plot_title: str) -> None:
        print('\n')
        for algo_name, result in results_to_show:
            print(f'{plot_title}: {algo_name}: last result: {result[-1]}')
        print('\n')
