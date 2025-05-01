from typing import Type

import numpy as np

from data_processors import DataLoader
from optimisers import Optimiser, SearchSpace, Solution


class TradingBotInstance():
    """ 
    TradingBot integrates data loaders and optimisers to form a functional trading algorithm and
    simulates a single instance of a trading run with the given parameters. When simulating swarms,
    and instance of this class should be initialised for each member of the population.
    """

    data_loader: DataLoader

    w1: float
    """ weighting for SMA-EMA crossover """
    w2: float
    """ weighting for EMA-LMA crossover """
    w3: float
    """ weighting for sma momentum """
    w4: float
    """ weighting for ema momentum """
    w5: float
    """ weighting for lma momentum """

    threshold: float
    """ threshold for executing trades """

    cash: float
    """ current $ amount if holding $ else 0 """
    bitcoin: float
    """ current btc amount if holding btc else 0 """


    def __init__(self, dataset_path, window_sma, window_ema, window_lma, ema_alpha, threshold, w1, w2, w3, w4, w5):
        self.data_loader = DataLoader(dataset_path, window_sma, window_ema, window_lma, ema_alpha)
        self.threshold = threshold

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        self.current_signal = 0

        # starting with 1000
        self.cash = 1000
        self.btc = 0

    # private function for calculating the trading signal -1 means sell, +1 means buy, between thresholds is 0 meaning no change
    def _trading_signal(self, sma, ema, lma, sma_momentum, ema_momentum, lma_momentum):
        # each feature having a weight assigned allows the fine-tuning of the effect of each feature on the signal
        d1 = self.w1 * (sma - ema)
        d2 = self.w2 * (ema - lma)
        d3 = self.w3 * sma_momentum
        d4 = self.w4 * ema_momentum
        d5 = self.w5 * lma_momentum

        signal_strength = d1 + d2 + d3 + d4 + d5
        return np.where(signal_strength > self.threshold,  1, np.where(signal_strength < -self.threshold, -1, 0))
    
    # private method to sell
    def _sell_btc(self):
        price = self.data_loader.get_price()
        self.cash = self.btc * price * 0.97
        self.btc = 0

    # private method to buy
    def _buy_btc(self):
        price = self.data_loader.get_price()
        self.btc = self.cash / price * 0.97
        self.cash = 0

    # simulates a run over data
    def simulate_run(self):
        # iterate through all the data in the data loader
        data = self.data_loader.step()
        self.current_signal = -1

        while data is not None:
            signal = self._trading_signal(*data)

            if self.current_signal == signal or signal == 0: 
                data = self.data_loader.step()
                continue
            else:
                if signal == -1:
                    self._sell_btc()
                elif signal == 1:
                    self._buy_btc()
                data = self.data_loader.step()
                self.current_signal = signal
                continue

        if self.cash == 0:
            self._sell_btc()

        return self.cash


class TradingBotOptimiser():
    """
    TradingBotOptimiser instantiates the TradingBotInstance and gathers the score for each simulation.
    Since parameters are different for each optimiser, **kwargs are taken at construction.
    """

    def __init__(self, dataset_path, optimiser: Optimiser):
        self.dataset_path = dataset_path
        
        # the following are the optimisation space dimensions
        # [window_sma, window_ema, window_lma, ema_alpha, w1, w2, w3, w4, w5, threshold]
        lower_bounds = np.array((1, 1, 1, 1e-10, -100, -100, -100, -100, -100, 0))
        upper_bounds = np.array((100, 100, 100, 1, 10, 100, 100, 100, 100, 500))
    
        n_dims = lower_bounds.size
        int_dims = np.array((True, True, True, False, False, False, False, False, False, False))
        
        self.optimiser = optimiser
        self.optimiser.search_space = SearchSpace(
            n_dim=n_dims, 
            dim_lower_bound=lower_bounds, 
            dim_upper_bound=upper_bounds,
            dim_is_integer=int_dims,
            eval_fitness=self._fitness,
            maximisation=True)
    
    def _fitness(self, pos) -> float:
        bot = TradingBotInstance(self.dataset_path, *pos.tolist())
        return bot.simulate_run()
    
    def optimise(self, iter: int) -> tuple[Solution, float]:
        self.optimiser.init()

        for i in range(iter):
            self.optimiser.step()
            if i % 5 == 0:
                print(f"iteration: {i}")
        
        return self.optimiser.current_best()
        