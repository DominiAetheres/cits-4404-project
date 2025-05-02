# Running Simulations

## TradingBotOptimiser

This class handles the optimisation for the trading signal parameters. There are ten trading signal parameters which require modification:

1. SMA window (integer)
2. EMA window (integer)
3. LMA window (integer)
4. EMA alpha (float $0 < x < 1$)
5. SMA - EMA weight (float)
6. EMA - LMA weight (float)
7. SMA momentum weight (float)
8. EMA momentum weight (float)
9. LMA momentum weight (float)
10. Threshold (float)

Instantiate this class and pass in an optimiser object (`PSO`, `PSOSA`, `ArtificialBeeColony`). Run the `optimise()` method on the optimiser object, which returns the optimal parameters.

## TradingBotInstance

Use this class to obtain the final results of the optimisation. After obtaining the parameters from `TradingBotOptimiser`, instantiate this class and pass in the learned parameters listed above. Call the `simulate_run()` method to obtain the final result. The final result will be in terms of the final monetary value held by the bot.

## Summary

1. Use the `TradingBotOptimiser` class and an optimiser to obtain an optimal set of parameters.
2. Use the `TradingBotInstance` class to obtain the results of the parameters in terms of final monetary value.