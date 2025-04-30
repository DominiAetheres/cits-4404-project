import numpy as np

from data_processors import DataLoader
from trading_bot import TradingBotInstance, TradingBotOptimiser
from optimisers import PSO, PSOSA, ArtificialBeeColony


filepath = './Deliverable02/Data/testDaily.csv'

optim = TradingBotOptimiser(PSO, filepath, population_size=5, p_increment=0.1, g_increment=0.1)

optim.optimise(100)


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