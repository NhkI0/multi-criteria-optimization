"""
Simple data loader for portfolio optimization
Reuses data from the original project
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_portfolio_data(data_dir):
    """
    Load stock data from CSV files.

    Returns:
        prices: DataFrame with stock prices
        mean_returns: Series with annualized mean returns (μ)
        cov_matrix: DataFrame with annualized covariance matrix (Σ)
        sector_mapping: Dict mapping ticker -> sector name
    """
    all_dataframes = []
    sector_mapping = {}

    data_path = Path(data_dir)

    for csv_file in data_path.glob("*.csv"):
        sector_name = csv_file.stem.replace('_', ' ')
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        df = df.dropna(how='all')

        # Map each ticker to its sector
        for col in df.columns:
            sector_mapping[col] = sector_name

        all_dataframes.append(df)

    # Combine all stock data
    prices = pd.concat(all_dataframes, axis=1)

    # Remove assets with too much missing data
    prices = prices.dropna(axis=1, thresh=len(prices) * 0.8)

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Annualize (252 trading days)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    print(f"Loaded {len(mean_returns)} assets")
    print(f"Period: {prices.index[0]} to {prices.index[-1]}")

    return prices, mean_returns, cov_matrix, sector_mapping


if __name__ == "__main__":
    # Test the loader
    data_dir = "../data"
    prices, mu, Sigma, sectors = load_portfolio_data(data_dir)
    print(f"\nMean return range: {mu.min():.2%} to {mu.max():.2%}")
    print(f"Mean volatility: {np.sqrt(np.diag(Sigma)).mean():.2%}")
