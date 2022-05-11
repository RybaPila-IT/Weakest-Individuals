import numpy as np
import matplotlib.pyplot as plt
from custom_types import EvaluatedPopulation


class Logger:
    default_options = {
        'v_max': True,
        'v_min': True,
        'v_avg': True,
    }

    def __init__(self, options: dict | None = None):
        self.__max = []
        self.__min = []
        self.__avg = []
        self.__options = {**Logger.default_options, **options} \
            if options is not None \
            else Logger.default_options

    def clean_up(self) -> None:
        self.__max = []
        self.__min = []
        self.__avg = []

    def generate_new_log_entry(self, eval_population: EvaluatedPopulation) -> None:
        if self.__options['v_max']:
            self.__max.append(max(eval_population, key=lambda i: i[1])[1])
        if self.__options['v_min']:
            self.__min.append(min(eval_population, key=lambda i: i[1])[1])
        if self.__options['v_avg']:
            self.__avg.append(np.mean([i[1] for i in eval_population]))

    def show_log_plots(self) -> None:
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
        with open(file_path, 'w') as file:
            if self.__options['v_max']:
                file.write(f'v_max: {self.__max}\n')
            if self.__options['v_min']:
                file.write(f'v_min: {self.__min}\n')
            if self.__options['v_avg']:
                file.write(f'v_avg: {self.__avg}\n')
