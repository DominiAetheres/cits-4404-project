import numpy as np
from data_processors import TradingDataProcessor
    

class DataLoader():
    """
    Loader for data from the trading data processor. Allows the feeding of data into optimisation algorithms
    through a step-based process.
    """

    sma: list
    ema: list
    lma: list

    window_sma: int
    window_ema: int
    window_lma: int

    ema_alpha: float

    current_step: int


    def __init__(self, dataset_path, window_sma, window_ema, window_lma, ema_alpha):
        processor = TradingDataProcessor(dataset_path)
        
        self.window_sma = int(window_sma)
        self.window_ema = int(window_ema)
        self.window_lma = int(window_lma)

        self.ema_alpha = ema_alpha

        self.prices = processor.closing_prices

        self.sma = processor.get_sma(self.window_sma)
        self.ema = processor.get_ema(self.window_ema, alpha=self.ema_alpha)
        self.lma = processor.get_lma(self.window_lma)
        
        self.max_step = len(self.sma)
        self.current_step = 0

    def _get_current_values(self):
        return self.sma[self.current_step], self.ema[self.current_step], self.lma[self.current_step]
    
    # return difference between last time step and current time step
    def _get_momentum(self):
        # for first time step, return zeroed out momentum
        if self.current_step == 0:
            return 0, 0, 0
        
        last_step = self.current_step - 1
        return self.sma[self.current_step] - self.sma[last_step], self.ema[self.current_step] - self.ema[last_step], self.lma[self.current_step] - self.lma[last_step]
    
    # public method for calling next step
    # return format is (sma, ema, lma, sma momentum, ema momentum, lma momentum)
    def step(self):
        # check if we are at last step, return none if that is the case
        if self.current_step == self.max_step:
            return None

        data = self._get_current_values() + self._get_momentum()
        self.current_step += 1
        return data
    
    # get method for price at current step
    def get_price(self):
        if self.current_step == self.max_step:
            return self.prices[self.max_step - 1]
        return self.prices[self.current_step]

    