from optimisers.base import Optimiser, SearchSpace, Solution, opt_float_t

import numpy as np

class PSOSA(Optimiser):

    population_size: int
    
    p_increment: float
    """ controls effect of p_best """
    g_increment: float
    """ controls effect of g_best """

    w_max: float
    """ maximum weighting """
    w_min: float
    """ minimum weighting """

    v_max: float
    """ maximum particle velocity """

    num_iters: int
    """ number of total iterations """
    curr_iter: int
    """ current number of iterations """

    init_temp: float
    """ initial temperature parameter """
    cool_rate: float
    """ alpha parameter for SA cooling """

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


    def __init__(self, 
                search_space: SearchSpace, rng_seed: int, population_size: int, 
                p_increment: float, g_increment: float, iters: int,
                w_max: float=1, w_min: float=0.2, v_max: float=1,
                init_temp: float=100, cool_rate:float=0.9):

        super().__init__(search_space=search_space, rng_seed=rng_seed)

        self.population_size = population_size
        
        self.p_increment = p_increment
        self.g_increment = g_increment

        self.w_max = w_max
        self.w_min = w_min

        self.v_max = v_max

        self.num_iters = iters
        self.curr_iter = 0

        self.init_temp = init_temp
        self.cool_rate = cool_rate


    def _find_g_best(self):
        idx = np.argmax(self.swarm_fitness)
        self.g_best_pos = self.swarm_pos[idx].copy()
        self.g_best_fitness = self.swarm_fitness[idx].copy()

    def calculate_weight(self):
        w = self.w_max - ((self.w_max - self.w_min)/self.num_iters) * self.curr_iter
        return w

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
        self._find_g_best()
        
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
        # update current iter
        self.curr_iter += 1

        # first we calculate random coefficients to apply to p_best and g_best
        p_rand_coef = self._rng.random(self.swarm_pos.shape)
        g_rand_coef = self._rng.random(self.swarm_pos.shape)

        # calculate weighting for current step
        w = self.calculate_weight()

        # then we calculate changes in velocity due to p and g factors
        p_dv = self.p_increment * p_rand_coef * (self.swarm_p_best_pos - self.swarm_pos)
        g_dv = self.g_increment * g_rand_coef * (self.g_best_pos - self.swarm_pos)

        # update velocity then position
        # use temp variables as positions/velocities may be rejected
        temp_swarm_vel = w * self.swarm_vel + p_dv + g_dv
        temp_swarm_pos = self.swarm_pos + temp_swarm_vel

        # loop metropolis acceptance rule
        while True:
            # delta fitness between current and next positions
            d_fitness = np.fromiter((self._eval_fitness(pos) for pos in temp_swarm_pos), opt_float_t) - self.swarm_fitness
            
            # if delta fitness > 0, accept immediately
            # particles which are not accepted are masked as 1
            # they will accept new changes when recalculating
            # particles accepted will be masked as 0 to prevent modification of parameters
            #mask = (d_fitness < 0).astype(int)

            # random acceptance hurdle value
            R = np.random.uniform(0., 1.)

            metropolis = np.e ** (d_fitness / (self.init_temp * (self.cool_rate ** self.curr_iter)))
            mask = (metropolis > R).astype(int).reshape(-1, 1)
   
            if np.sum(mask) == self.population_size:
                self.swarm_vel = temp_swarm_vel.copy()
                self.swarm_pos += self.swarm_vel
                break

            # recalculate positions for particles not accepted
            p_rand_coef = self._rng.random(self.swarm_pos.shape)
            g_rand_coef = self._rng.random(self.swarm_pos.shape)

            p_dv = self.p_increment * p_rand_coef * (self.swarm_p_best_pos - self.swarm_pos)
            g_dv = self.g_increment * g_rand_coef * (self.g_best_pos - self.swarm_pos)

            # perform masking on velocity to preserve accepted velocities
            temp_swarm_vel = (mask * temp_swarm_vel) + (mask * (w * self.swarm_vel + p_dv + g_dv))
            temp_swarm_pos = self.swarm_pos + temp_swarm_vel

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