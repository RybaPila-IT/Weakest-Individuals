from nptyping import NDArray, Shape, Float
from collections.abc import Callable


Individual = NDArray[Shape['1, *'], Float]
EvaluatedIndividual = tuple[Individual, float]
Population = list[Individual]
EvaluatedPopulation = list[EvaluatedIndividual]
ObjectiveFunction = Callable[[Individual], float]
