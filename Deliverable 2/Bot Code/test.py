import numpy as np
# Assuming trading_data_processor.py is in the same directory
from TradingDataProcessor import TradingDataProcessor

# --- Main Execution Example ---

if __name__ == "__main__":
  filepath = 'BTC-Daily.csv' # Make sure this file exists in the same directory

  # 1. Create an instance of the processor - this reads the data
  print(f"Creating TradingDataProcessor for {filepath}...")
  processor = TradingDataProcessor(filepath)
  print("-" * 30)

  # 2. Check if data was loaded successfully before proceeding
  if processor.closing_prices is not None and len(processor.closing_prices) > 0:
    print(f"Data loaded. Number of closing prices: {len(processor.closing_prices)}")
    print(f"First 5 prices: {processor.closing_prices[:5]}")
    print(f"Last 5 prices: {processor.closing_prices[-5:]}")
    print("-" * 30)

    # 3. Define parameters for WMAs
    N_short = 10
    N_long = 50
    alpha_val = 0.1

    # 4. Call methods on the processor instance to get WMA series
    print(f"Calculating SMA({N_short})...")
    sma_10 = processor.get_sma(N_short)

    print(f"Calculating SMA({N_long})...")
    sma_50 = processor.get_sma(N_long)

    print(f"Calculating LMA({N_short})...")
    lma_10 = processor.get_lma(N_short)

    print(f"Calculating EMA({N_long}, alpha={alpha_val})...")
    ema_50 = processor.get_ema(N_long, alpha=alpha_val)
    print("-" * 30)


    # 5. Use the results (Example: print first/last 5 values)
    print("Example Results:")
    if sma_10 is not None:
        print(f"\nFirst 5 SMA({N_short}): {sma_10[:5]}")
        print(f"Last 5 SMA({N_short}):  {sma_10[-5:]}")
    else:
        print(f"Could not calculate SMA({N_short}).")

    if sma_50 is not None:
        print(f"\nFirst 5 SMA({N_long}): {sma_50[:5]}")
        print(f"Last 5 SMA({N_long}):  {sma_50[-5:]}")
    else:
        print(f"Could not calculate SMA({N_long}).")

    if lma_10 is not None:
        print(f"\nFirst 5 LMA({N_short}): {lma_10[:5]}")
        print(f"Last 5 LMA({N_short}):  {lma_10[-5:]}")
    else:
        print(f"Could not calculate LMA({N_short}).")

    if ema_50 is not None:
        print(f"\nFirst 5 EMA({N_long}, {alpha_val}): {ema_50[:5]}")
        print(f"Last 5 EMA({N_long}, {alpha_val}):  {ema_50[-5:]}")
    else:
        print(f"Could not calculate EMA({N_long}, {alpha_val}).")

    # Example: Calculate difference if both SMAs were calculated successfully
    if sma_10 is not None and sma_50 is not None:
         # Ensure they are not all NaNs before differencing
         if not np.isnan(sma_10).all() and not np.isnan(sma_50).all():
              difference = sma_10 - sma_50
              print(f"\nFirst 5 SMA({N_short})-SMA({N_long}) diff: {difference[:5]}")
              print(f"Last 5 SMA({N_short})-SMA({N_long}) diff:  {difference[-5:]}")
         else:
              print("\nCannot calculate difference because one or both SMA series contain only NaNs.")

  else:
    print("\nProcessor could not load data. Cannot perform calculations.")
