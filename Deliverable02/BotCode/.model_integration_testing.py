from trading_bot import TradingBotOptimiser
from optimisers import PSO, PSOSA, ArtificialBeeColony


filepath = './Deliverable02/Data/testDaily.csv'

rng_seed = 42
population_size = 5
n_iter = 100

pso_optim = TradingBotOptimiser(filepath, PSO(rng_seed=rng_seed, population_size=population_size, p_increment=0.1, g_increment=0.1))
psosa_optim = TradingBotOptimiser(filepath, PSOSA(rng_seed=rng_seed, population_size=population_size, p_increment=0.1, g_increment=0.1, iters=100))
abc_optim = TradingBotOptimiser(filepath, ArtificialBeeColony(rng_seed=rng_seed, population_size=population_size, pos_age_limit=5))

print("PSO optimisation")
pso_best_pos, pso_best_fitness = pso_optim.optimise(n_iter)
print("PSOSA optimisation")
psosa_best_pos, psosa_best_fitness = psosa_optim.optimise(n_iter)
print("ABC optimisation")
abc_best_pos, abc_best_fitness = abc_optim.optimise(n_iter)

print(f"  PSO: score {pso_best_fitness:.8g} at pos ({pso_best_pos})")
print(f"PSOSA: score {psosa_best_fitness:.8g} at pos ({psosa_best_pos})")
print(f"  ABC: score {abc_best_fitness:.8g} at pos ({abc_best_pos})")


"""
bot = TradingBotInstance(filepath, 2, 3, 4, 0.9, 1, 1, 1, 1, 1, 1)

print(bot.simulate_run())
"""



"""
loader = DataLoader(filepath, 3, 3, 3, 0.2)

print(loader.step())
print(loader.step())

print(loader.get_price())
"""

"""
## initial testing
data_processor = TradingDataProcessor(filepath)

print(len(data_processor.get_sma(5)))
print(len(data_processor.get_ema(5, 0.4)))

# idea is to use windows as params for optim
# add momentum as features
# add weights w1 and w2 for crossovers

w1 = 0.9
w2 = 1.1
threshold = 1

def trading_signal(sma, ema, lma, w1, w2, threshold):

    d1 = w1 * (sma - ema)
    d2 = w2 * (ema - lma)

    score = d1 + d2
    return np.where(score > threshold,  1, np.where(score < -threshold, -1, 0))


print(trading_signal(data_processor.get_sma(5)[0], data_processor.get_ema(N=4, alpha=0.4)[0], data_processor.get_lma(6)[0], w1, w2, threshold))

"""