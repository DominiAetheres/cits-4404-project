import numpy as np

from optimisers import ArtificialBeeColony, SearchSpace, Solution


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
    abc_opt = ArtificialBeeColony(search_space=space, rng_seed=42, population_size=5, pos_age_limit=5)
    
    # run the optimiser
    abc_opt.init()
    for iter in range(100):
        abc_opt.step()
        print(f"i: {iter:04d}, score: {abc_opt.best_fitness:.8g}, pos: {abc_opt.best_pos}")
    
    print(f"Final best: pos ({abc_opt.best_pos}) with score {abc_opt.best_fitness}")
