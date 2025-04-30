from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np


opt_float_t = np.float64
Solution = np.typing.NDArray[opt_float_t]


class SearchSpace(NamedTuple):
    """
    Represents the information about a search space;
    the number of dimensions, and their bounds.
    * Note: assumes that the bounds of each dimensions are independent
    """

    n_dim: int

    dim_lower_bound: np.typing.NDArray
    """ Lower bounds of the dimensions. """
    dim_upper_bound: np.typing.NDArray
    """ Upper bounds of the dimensions. """

    dim_is_integer: np.typing.NDArray[np.bool]
    """
    Dimensions that should be integer-valued instead of float-valued.
    If discrete but not only integer values, apply a mapping to the dimension
    when evaluating the fitness.
    """

    eval_fitness: Callable[[Solution], float]

    minimisation: bool = True
    """ If this a minimisation (default) or maximisation problem. """

    def update_eval_function(self, eval: Callable[[Solution], float]) -> "SearchSpace":
        return self._replace(eval_fitness=eval)


class Optimiser(ABC):
    search_space: SearchSpace
    rng_seed: int
    _rng: np.random.Generator

    def __init__(self, *, search_space: SearchSpace, rng_seed: int):
        self.search_space = search_space
        self.rng_seed = rng_seed

        self._rng = np.random.default_rng(rng_seed)

    @abstractmethod
    def init(self):
        ...

    @abstractmethod
    def step(self):
        ...

    def step_n(self, n_iter: int):
        for _ in range(n_iter):
            self.step()
    
    def _eval_fitness(self, pos: Solution) -> float:
        return self.search_space.eval_fitness(pos)
    
    def _comp_fitness(self, lhs, rhs) -> bool:
        """ Returns `True` iff lhs is more optimal than rhs. """
        if self.search_space.minimisation:
            return lhs < rhs
        else:
            return lhs > rhs
