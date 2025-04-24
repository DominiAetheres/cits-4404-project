import pandas as pd
import numpy as np

class TradingDataProcessor:
    """
    Processes time series data for trading analysis.
    Reads data from a CSV file upon initialization and provides methods
    to calculate various Weighted Moving Averages (WMAs).
    """
    
    def __init__(self, filepath):
        """
        Initializes the processor by reading closing prices from the specified CSV file.

        Args:
            filepath (str): The path to the CSV file containing trading data.
                            Expected to have a 'close' column.
        """
        self.filepath = filepath
        self.closing_prices = self._read_data()
        if self.closing_prices is None:
            print(f"Warning: Failed to load data from {self.filepath}. WMA calculations will not work.")
        elif len(self.closing_prices) == 0:
             print(f"Warning: Loaded data from {self.filepath}, but found no valid closing prices. WMA calculations will not work.")


    def _read_data(self):
        """
        Reads the CSV data and extracts the closing price.
        Internal method called by __init__.

        Returns:
            np.array: A numpy array of closing prices, or None if an error occurs.
        """
        df = None
        print(f"Attempting to read data from: {self.filepath}")
        try:
            # Try reading with standard parameters first
            try:
                df = pd.read_csv(self.filepath)
            except UnicodeDecodeError:
                print("UTF-8 decoding failed, trying latin1 encoding...")
                df = pd.read_csv(self.filepath, encoding='latin1')
            except Exception as e:
                 print(f"Initial read failed: {e}. Trying without header...")
                 # If initial read fails, try without header before giving up
                 try:
                    df = pd.read_csv(self.filepath, header=None, encoding='latin1' if 'UnicodeDecodeError' in locals() else 'utf-8')
                    print("Read successful assuming no header.")
                    # Attempt to assign reasonable column names if no header
                    if len(df.columns) >= 7:
                         df.columns = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close'] + [f'col_{i}' for i in range(7, len(df.columns))]
                         print("Assigned standard column names based on position.")
                    else:
                         print("Warning: Could not confidently assign column names without header.")
                         return None # Cannot proceed without knowing the close column
                 except Exception as e_no_header:
                     print(f"Reading without header also failed: {e_no_header}")
                     return None

            # --- Column Name Handling ---
            close_col_name = None
            for col in df.columns:
                col_str = str(col).strip().lower()
                if col_str == 'close':
                    close_col_name = col
                    break

            if close_col_name is None:
                for col in df.columns:
                     col_str = str(col).strip().lower()
                     if col_str == 'close': # Check again after potential rename
                         close_col_name = col
                         break

            if close_col_name is None and len(df.columns) >= 7:
                 print("Warning: 'close' column not found by name. Assuming 7th column (index 6) is close price.")
                 close_col_name = df.columns[6]
                 # Ensure the assumed column name doesn't clash before renaming
                 if 'close' not in df.columns:
                      df = df.rename(columns={close_col_name: 'close'})
                      close_col_name = 'close'
                 else:
                      # If 'close' exists but wasn't found, something is odd. Use index directly.
                      close_col_name = df.columns[6]


            if close_col_name is None or close_col_name not in df.columns:
                print("Error: Could not identify the 'close' price column after checks.")
                print(f"Available columns: {list(df.columns)}")
                return None

            # --- Data Conversion and Return ---
            close_prices = pd.to_numeric(df[close_col_name], errors='coerce').dropna().values
            if len(close_prices) == 0:
                 print(f"Error: Column '{close_col_name}' found, but contained no valid numeric data after cleaning.")
                 return None

            print(f"Successfully read and processed {len(close_prices)} data points from {self.filepath}")
            return close_prices

        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during file processing: {e}")
            return None

    # --- Static Helper Methods for WMA Calculations ---
    # These don't depend on the instance state directly, only on inputs

    @staticmethod
    def _pad(P, N):
        """Pads the start of the price series P."""
        if N <= 1: return P
        actual_pad_len = min(len(P) - 1, N - 1)
        if actual_pad_len <= 0: return P
        padding = np.flip(P[1 : actual_pad_len + 1])
        return np.append(padding, P)

    @staticmethod
    def _sma_filter(N):
        """Creates a Simple Moving Average filter."""
        if N <= 0: return np.array([])
        return np.ones(N) / N

    @staticmethod
    def _lma_filter(N):
        """Creates a Linear Weighted Moving Average filter."""
        if N <= 0: return np.array([])
        weights = np.arange(N, 0, -1)
        return weights / np.sum(weights)

    @staticmethod
    def _ema_filter(N, alpha):
        """Creates an Exponential Moving Average filter kernel."""
        if not (0 < alpha <= 1) or N <= 0: return np.array([])
        k = np.arange(N)
        weights = (1 - alpha) ** k
        return weights / np.sum(weights)

    @staticmethod
    def _calculate_wma(P, N, kernel):
        """Calculates the Weighted Moving Average using convolution."""
        if P is None or kernel is None or N <= 0 or len(kernel) != N or len(P) == 0:
            return None # Basic input check
        if len(P) < N:
            # print(f"Warning: Price series length ({len(P)}) < window size ({N}). Returning NaNs.")
            return np.full(len(P), np.nan)

        padded_P = TradingDataProcessor._pad(P, N)
        if len(padded_P) < N:
             # print(f"Error: Padding failed for short series. Cannot compute WMA.")
             return np.full(len(P), np.nan)

        result = np.convolve(padded_P, kernel, 'valid')
        if len(result) != len(P):
           # print(f"Warning: WMA output length mismatch. Returning NaNs.")
           return np.full(len(P), np.nan)
        return result

    # --- Public Methods for Calculating Specific WMAs ---

    def get_sma(self, N):
        """
        Calculates the Simple Moving Average (SMA) for the loaded data.

        Args:
            N (int): The window size.

        Returns:
            np.array: The SMA series, or None if data wasn't loaded or N is invalid.
        """
        if self.closing_prices is None:
            print("Error: Price data not loaded. Cannot calculate SMA.")
            return None
        if N <= 0:
            print("Error: Window size N must be positive for SMA.")
            return None

        kernel = self._sma_filter(N)
        if len(kernel) == 0: return None # Filter creation failed
        return self._calculate_wma(self.closing_prices, N, kernel)

    def get_lma(self, N):
        """
        Calculates the Linear Weighted Moving Average (LMA) for the loaded data.

        Args:
            N (int): The window size.

        Returns:
            np.array: The LMA series, or None if data wasn't loaded or N is invalid.
        """
        if self.closing_prices is None:
            print("Error: Price data not loaded. Cannot calculate LMA.")
            return None
        if N <= 0:
            print("Error: Window size N must be positive for LMA.")
            return None

        kernel = self._lma_filter(N)
        if len(kernel) == 0: return None
        return self._calculate_wma(self.closing_prices, N, kernel)

    def get_ema(self, N, alpha):
        """
        Calculates the Exponential Moving Average (EMA) for the loaded data.

        Args:
            N (int): The window size.
            alpha (float): The smoothing factor (0 < alpha <= 1).

        Returns:
            np.array: The EMA series, or None if data wasn't loaded or parameters are invalid.
        """
        if self.closing_prices is None:
            print("Error: Price data not loaded. Cannot calculate EMA.")
            return None
        if N <= 0 or not (0 < alpha <= 1):
             print("Error: Invalid parameters for EMA (N > 0 and 0 < alpha <= 1 required).")
             return None

        kernel = self._ema_filter(N, alpha)
        if len(kernel) == 0: return None
        return self._calculate_wma(self.closing_prices, N, kernel)