import math
from typing import cast
import numpy as np

from optimisers.base import Optimiser, SearchSpace, Solution, opt_float_t


class ArtificialBeeColony(Optimiser):
    """
    Implementation of ABC algorithm based on the original Karaboga and Basturk 2007 paper.
    Assumes population size, employed bee count, and onlooker bee count are equal.
    Upon employed bees abandoning a food source, as the scouting operation they are immediately
    allocated a random new one from a uniform distribution over the entire search space.

    It is not clear in the original paper if the onlooker bees explore towards existing locations
    of other onlooker bees or employee bees - this implementation assumes onlooker bees.
    """

    population_size: int
    employed_count: int
    onlooker_count: int
    pos_age_limit: int

    employed_pos: np.typing.NDArray
    """ arr of positions """
    employed_pos_fitness: np.typing.NDArray[opt_float_t]
    """ arr of fitnesses (floats) """
    employed_pos_age: np.typing.NDArray[np.int32]
    """ arr of current pos ages (ints) """

    onlooker_pos: np.typing.NDArray
    """ arr of positions """
    onlooker_pos_fitness: np.typing.NDArray[opt_float_t]
    """ arr of fitnesses (floats) """

    best_pos: Solution
    best_fitness: float

    def __init__(self, *, search_space: SearchSpace | None = None, rng_seed: int, population_size: int, pos_age_limit: int):
        """
        * `population_size`: the number of food source positions, employed bees, and onlooker bees
        * `pos_age_limit`: the max number of cycles for a candidate with no improvements before being abandoned
        """
        super().__init__(search_space=search_space, rng_seed=rng_seed)

        self.population_size = population_size

        self.employed_count = population_size
        self.onlooker_count = population_size

        self.pos_age_limit = pos_age_limit

    def init(self):
        # init employed bees
        self.employed_pos_age = np.zeros((self.employed_count), dtype=np.int32)
        # see `._generate_new_pos` for notes on using `np.where`
        employed_pos_shape = (self.employed_count, self.search_space.n_dim)
        employed_rand_ints = self._rng.integers(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound, size=employed_pos_shape, endpoint=True).astype(opt_float_t)
        employed_rand_floats = self._rng.uniform(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound, size=employed_pos_shape).astype(opt_float_t)
        employed_dim_integer_flags = np.repeat(
            self.search_space.dim_is_integer[np.newaxis, :], self.employed_count, axis=0)
        self.employed_pos = np.where(
            employed_dim_integer_flags, employed_rand_ints, employed_rand_floats)
        self.employed_pos_fitness = np.fromiter(
            (self._eval_fitness(pos) for pos in self.employed_pos), opt_float_t)

        # init onlooker bees
        onlooker_pos_shape = (self.onlooker_count, self.search_space.n_dim)
        onlooker_rand_ints = self._rng.integers(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound, size=onlooker_pos_shape, endpoint=True).astype(opt_float_t)
        onlooker_rand_floats = self._rng.uniform(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound, size=onlooker_pos_shape).astype(opt_float_t)
        onlooker_dim_integer_flags = np.repeat(
            self.search_space.dim_is_integer[np.newaxis, :], self.onlooker_count, axis=0)
        self.onlooker_pos = np.where(
            onlooker_dim_integer_flags, onlooker_rand_ints, onlooker_rand_floats)
        self.onlooker_pos_fitness = np.fromiter(
            (self._eval_fitness(pos) for pos in self.onlooker_pos), opt_float_t)
        
        # init best results at location of first employed bee
        self.best_pos = self.employed_pos[0]
        self.best_fitness = self.employed_pos_fitness[0]

    def step(self):
        # TODO technically the bee pos updates should all happen *after* the new pos calcs

        # employed bees exploration
        for employed_i in range(self.employed_count):
            # current pos
            current_pos = cast(Solution, self.employed_pos[employed_i])
            current_pos_fitness = cast(
                opt_float_t, self.employed_pos_fitness[employed_i])

            # new pos
            new_pos = self._explore(
                current_pos=current_pos, self_index=employed_i, neighbour_pos_arr=self.employed_pos)
            new_pos_fitness = self._eval_fitness(new_pos)

            # learn new pos if better
            if self._comp_fitness(new_pos_fitness, current_pos_fitness):
                self.employed_pos[employed_i] = new_pos
                self.employed_pos_fitness[employed_i] = new_pos_fitness
                self.employed_pos_age[employed_i] = 0
            else:
                self.employed_pos_age[employed_i] += 1

        # onlooker food source selection
        onlooker_choices = self._rng.choice(range(
            self.employed_count), size=self.onlooker_count, p=self.employed_pos_fitness / np.sum(self.employed_pos_fitness))

        # onlooker exploration
        for onlooker_i, employed_i in enumerate(onlooker_choices):
            # current pos
            current_pos = cast(Solution, self.onlooker_pos[onlooker_i])
            current_pos_fitness = cast(
                opt_float_t, self.onlooker_pos_fitness[onlooker_i])

            # ? should neighbour pos be the chosen positions, instead of the prev positions

            # new pos
            new_pos = self._explore(
                current_pos=self.employed_pos[employed_i], self_index=onlooker_i, neighbour_pos_arr=self.onlooker_pos)
            new_pos_fitness = self._eval_fitness(new_pos)

            # learn new pos if better
            if self._comp_fitness(new_pos_fitness, current_pos_fitness):
                self.onlooker_pos[onlooker_i] = new_pos
                self.onlooker_pos_fitness[onlooker_i] = new_pos_fitness

        # abandon food sources (and find new ones)
        for employed_i in range(self.employed_count):
            if self.employed_pos_age[employed_i] < self.pos_age_limit:
                continue

            # allocate new food source
            new_pos = self._generate_new_pos()
            self.employed_pos[employed_i] = new_pos
            self.employed_pos_fitness[employed_i] = self._eval_fitness(new_pos)
            self.employed_pos_age[employed_i] = 0

        # update best found food source
        iter_employed_best_pos_and_fit = self.employed_pos[i := np.argmax(
            self.employed_pos_fitness)], self.employed_pos_fitness[i]
        iter_onlooker_best_pos_and_fit = self.onlooker_pos[i := np.argmax(
            self.onlooker_pos_fitness)], self.onlooker_pos_fitness[i]

        iter_best_pos, iter_best_fitness = max(
            [iter_employed_best_pos_and_fit, iter_onlooker_best_pos_and_fit], key=lambda v: v[1])

        if self._comp_fitness(iter_best_fitness, self.best_fitness):
            self.best_pos = np.copy(iter_best_pos)
            self.best_fitness = iter_best_fitness

    def _generate_new_pos(self) -> Solution:
        # from testing `np.where` vs looping over the dim integer mask and doing `uniform` or `integers` for each,
        # `np.where` is slower for < ~5 dims, comparable for ~10 dims and faster for > ~15 dims
        # 10 seems like a reasonable number of dimensions, so `np.where` has been used

        rand_ints = self._rng.integers(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound + 1)
        rand_floats = self._rng.uniform(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound)

        return np.where(self.search_space.dim_is_integer, rand_ints, rand_floats)

    def _explore(self, *, current_pos: Solution, self_index: int, neighbour_pos_arr: np.typing.NDArray) -> Solution:
        """
        Explore by selecting a random neighbour, and stepping towards
        it a random amount in a random dimension.
        """

        # ... random neighbour
        neighbour_index = self._rng.integers(0, neighbour_pos_arr.shape[0] - 1)
        if neighbour_index >= self_index:
            neighbour_index += 1  # can't be the current index
        neighbour_pos_arr = cast(Solution, neighbour_pos_arr[neighbour_index])

        # ... in a random dimension
        step_dim = self._rng.integers(0, self.search_space.n_dim)

        # ... a random amount
        step_scalar = self._rng.uniform(-1, 1)
        step_size = step_scalar * \
            cast(opt_float_t, current_pos[step_dim] -
                 neighbour_pos_arr[step_dim])
        if self.search_space.dim_is_integer[step_dim]:
            if abs(step_size) < 1:
                # minimum step magnitude of 1
                step_size = math.copysign(1, step_size)
            else:
                step_size = round(step_size)

        # perform the step
        new_pos = current_pos.copy()
        new_pos[step_dim] += step_size

        # check bounds
        new_pos[step_dim] = np.clip(
            new_pos[step_dim], self.search_space.dim_lower_bound[step_dim], self.search_space.dim_upper_bound[step_dim])

        return new_pos
    
    def current_best(self) -> tuple[Solution, float]:
        return self.best_pos, self.best_fitness
