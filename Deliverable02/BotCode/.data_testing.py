import matplotlib.pyplot as plt

from data_processors import TradingDataProcessor


filepath = './Deliverable02/Data/testDaily.csv'
data_processor = TradingDataProcessor(filepath)

prices = data_processor.closing_prices
ema = data_processor.get_ema(5, 0.4)

plt.plot(prices)
plt.plot(ema)
plt.plot()
plt.show()