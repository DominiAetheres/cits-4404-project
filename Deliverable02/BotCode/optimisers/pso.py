from optimisers.base import Optimiser, SearchSpace, Solution, opt_float_t

import numpy as np

class PSO(Optimiser):

    population_size: int
    
    p_increment: float
    """ controls effect of p_best """
    g_increment: float
    """ controls effect of g_best """

    swarm_vel: np.typing.NDArray
    """ array of velocities for swarm """
    swarm_pos: np.typing.NDArray
    """ array of positions for swarm """
    swarm_fitness: np.typing.NDArray
    """ array of current fitnesses of swarm """

    swarm_p_best_pos: np.typing.NDArray
    """ record of p_best positions for each swarm member """
    swarm_p_best_fitness: np.typing.NDArray
    """ record of p_best fitnesses for each swarm member """
    
    g_best_pos: np.typing.NDArray
    """ best position globally so far """
    g_best_fitness: float
    """ fitness of g_best """

    best_pos: Solution
    best_fitness: float


    def __init__(self, *, search_space: SearchSpace | None = None, rng_seed: int, population_size: int, p_increment: float, g_increment: float):
        super().__init__(search_space=search_space, rng_seed=rng_seed)

        self.population_size = population_size
        
        self.p_increment = p_increment
        self.g_increment = g_increment


    def init(self):
        # init swarm positions (thanks nathan for the code)
        pos_shape = (self.population_size, self.search_space.n_dim)
        pos_rand_ints = self._rng.integers(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound, size=pos_shape, endpoint=True).astype(opt_float_t)
        pos_rand_floats = self._rng.uniform(
            low=self.search_space.dim_lower_bound, high=self.search_space.dim_upper_bound, size=pos_shape).astype(opt_float_t)
        pos_dim_integer_flags = np.repeat(
            self.search_space.dim_is_integer[np.newaxis, :], self.population_size, axis=0)
        self.swarm_pos = np.where(
            pos_dim_integer_flags, pos_rand_ints, pos_rand_floats)
        self.swarm_fitness = np.fromiter(
            (self._eval_fitness(pos) for pos in self.swarm_pos), opt_float_t)
        
        # find initial best pos
        idx = np.argmax(self.swarm_fitness)
        self.g_best_pos = self.swarm_pos[idx].copy()
        self.g_best_fitness = self.swarm_fitness[idx].copy()
        
        # p_best at init is simply their init positions
        self.swarm_p_best_pos = self.swarm_pos.copy()
        self.swarm_p_best_fitness = self.swarm_fitness.copy()

        # init swarm velocities to U(-1, 1)
        self.swarm_vel = np.random.uniform(
            low=-1.0,
            high= 1.0,
            size=self.swarm_pos.shape
        )

    def step(self):
        # IMPORTANT NOTE: this version of PSO is based on the original 1995 Kennedy & Eberhart paper,
        #       this means that there is no weighting variable in contrast to the 1998 PSO paper by
        #       Shi and Eberhart.

        # first we calculate random coefficients to apply to p_best and g_best
        p_rand_coef = self._rng.random(self.swarm_pos.shape)
        g_rand_coef = self._rng.random(self.swarm_pos.shape)

        # then we calculate changes in velocity due to p and g factors
        p_dv = self.p_increment * p_rand_coef * (self.swarm_p_best_pos - self.swarm_pos)
        g_dv = self.g_increment * g_rand_coef * (self.g_best_pos - self.swarm_pos)

        # update velocity then position
        self.swarm_vel += p_dv + g_dv
        self.swarm_pos += self.swarm_vel

        # clip within search bounds
        self.swarm_pos = np.clip(self.swarm_pos, self.search_space.dim_lower_bound, self.search_space.dim_upper_bound)

        # enforce int features
        if np.any(self.search_space.dim_is_integer):
            self.swarm_pos[:, self.search_space.dim_is_integer] = np.round(
                self.swarm_pos[:, self.search_space.dim_is_integer]
            ).astype(int)

        # evaluate fitness
        self.swarm_fitness = np.fromiter(
            (self._eval_fitness(pos) for pos in self.swarm_pos), opt_float_t)
        
        # update p_bests
        better_mask = self.swarm_fitness > self.swarm_p_best_fitness
        self.swarm_p_best_pos[better_mask] = self.swarm_pos[better_mask]
        self.swarm_p_best_fitness[better_mask] = self.swarm_fitness[better_mask]

        # update g_best
        idx = np.argmax(self.swarm_p_best_fitness)
        if self.swarm_p_best_fitness[idx] > self.g_best_fitness:
            self.g_best_fitness = self.swarm_p_best_fitness[idx].copy()
            self.g_best_pos = self.swarm_p_best_pos[idx]
    
    def current_best(self) -> tuple[Solution, float]:
        return self.g_best_pos, self.g_best_fitness