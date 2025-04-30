import numpy as np

from optimisers import PSOSA, SearchSpace, Solution


if __name__ == "__main__":
    # flat plane with global minimum (1, 10, 100)
    lower_bounds = np.array((1, 10, 100))
    upper_bounds = lower_bounds + 20
    n_dims = lower_bounds.size
    int_dims = np.array((False,) * n_dims)

    def fit_func(pos: Solution) -> float:
        return np.sum(pos)

    # setup
    space = SearchSpace(n_dim=n_dims, dim_lower_bound=lower_bounds, dim_upper_bound=upper_bounds, dim_is_integer=int_dims, eval_fitness=fit_func)
    pso = PSOSA(search_space=space, rng_seed=42, population_size=5, p_increment=0.1, g_increment=0.1, iters=100)

    pso.init()
    
    for i in range(100):
        pso.step()
        print(f"i: {i:04d}, score: {pso.g_best_fitness:.8g}, pos: {pso.g_best_pos}")
    
    print(f"Final best: pos ({pso.g_best_pos}) with score {pso.g_best_fitness}")