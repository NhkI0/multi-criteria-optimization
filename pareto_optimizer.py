"""
Multi-objective portfolio optimization with Pareto front generation
Implements scalarization and epsilon-constraint methods
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, List, Dict
import json


def f3_transaction_cost(w, w_current, c_prop=0.001):
    """f3(w) = Σ c_prop |w_i - w_current_i| (transaction costs)"""
    return c_prop * np.sum(np.abs(w - w_current))


class ParetoPortfolioOptimizer:
    """Multi-objective portfolio optimizer"""

    def __init__(self, mean_returns, cov_matrix, sector_mapping=None):
        """
        Args:
            mean_returns: Series with mean returns (μ)
            cov_matrix: DataFrame with covariance matrix (Σ)
            sector_mapping: Dict mapping ticker -> sector
        """
        self.mu = mean_returns.values
        self.Sigma = cov_matrix.values
        self.asset_names = list(mean_returns.index)
        self.n_assets = len(self.asset_names)
        self.sector_mapping = sector_mapping or {}

    # === Objective Functions ===

    def f1_return(self, w):
        """f1(w) = -w^T μ (negative return, to minimize)"""
        return -np.dot(w, self.mu)

    def f2_variance(self, w):
        """f2(w) = w^T Σ w (portfolio variance)"""
        return np.dot(w, np.dot(self.Sigma, w))

    # === LEVEL 1: Bi-objective (f1, f2) ===

    def generate_pareto_front_level1(self, n_points=50):
        """
        Generate Pareto front for Level 1: min{f1, f2} under C_Base

        Method: Weighted sum scalarization
        Minimize: α·f1(w) + (1-α)·f2(w)

        Returns:
            pareto_solutions: List of dicts with weights and objectives
        """
        print(f"\n{'=' * 60}")
        print("LEVEL 1: Generating Pareto Front (Return-Risk)")
        print(f"{'=' * 60}")
        print(f"Method: Weighted Sum Scalarization")
        print(f"Points: {n_points}")

        pareto_solutions = []

        # Normalize objectives for better scaling
        # First, get approximate ranges
        w_equal = np.ones(self.n_assets) / self.n_assets
        f1_base = abs(self.f1_return(w_equal))
        f2_base = self.f2_variance(w_equal)

        alphas = np.linspace(0, 1, n_points)

        for i, alpha in enumerate(alphas):
            # Scalarized objective
            def objective(w):
                f1 = self.f1_return(w) / f1_base
                f2 = self.f2_variance(w) / f2_base
                return alpha * f1 + (1 - alpha) * f2

            # Optimize
            w_init = np.ones(self.n_assets) / self.n_assets
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.n_assets)]

            result = minimize(objective, w_init, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 500, 'disp': False})

            if result.success:
                w = result.x
                pareto_solutions.append({
                    'weights': w.tolist(),
                    'f1_negative_return': float(self.f1_return(w)),
                    'f1_return': float(-self.f1_return(w)),  # Actual return
                    'f2_variance': float(self.f2_variance(w)),
                    'f2_volatility': float(np.sqrt(self.f2_variance(w))),  # More interpretable
                    'alpha': float(alpha),
                    'n_assets': int(np.sum(w > 0.001))
                })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_points} points generated")

        print(f"\n[OK] Generated {len(pareto_solutions)} Pareto-optimal portfolios")

        # Display range
        returns = [-sol['f1_negative_return'] for sol in pareto_solutions]
        vols = [sol['f2_volatility'] for sol in pareto_solutions]
        print(f"  Return range: {min(returns):.2%} to {max(returns):.2%}")
        print(f"  Risk range: {min(vols):.2%} to {max(vols):.2%}")

        return pareto_solutions

    # === LEVEL 2: Tri-objective (f1, f2, f3) with Cardinality ===

    def generate_pareto_front_level2(self, K=20, w_current=None, c_prop=0.001, n_points=30):
        """
        Generate Pareto front for Level 2: min{f1, f2, f3} under C_Base ∩ C_Op

        Method: Weighted sum with cardinality constraint
        Minimize: α·f1(w) + β·f2(w) + γ·f3(w)
        Subject to: Σ I(w_i > δ_tol) = K

        Args:
            K: Number of assets (cardinality)
            w_current: Current portfolio weights (for transaction costs)
            c_prop: Transaction cost rate
            n_points: Number of Pareto points to generate

        Returns:
            pareto_solutions: List of dicts with weights and objectives
        """
        print(f"\n{'=' * 60}")
        print("LEVEL 2: Generating Pareto Front (Return-Risk-Costs)")
        print(f"{'=' * 60}")
        print(f"Method: Weighted Sum with Cardinality K={K}")
        print(f"Points: {n_points}")

        if w_current is None:
            w_current = np.ones(self.n_assets) / self.n_assets

        pareto_solutions = []

        # Select K best assets by Sharpe ratio (greedy heuristic for cardinality)
        sharpe_ratios = self.mu / np.sqrt(np.diag(self.Sigma))
        best_k_indices = np.argsort(sharpe_ratios)[-K:]
        best_k_indices = np.sort(best_k_indices)

        print(f"  Selected {K} assets with best Sharpe ratios")

        # Reduced problem on K assets
        mu_k = self.mu[best_k_indices]
        Sigma_k = self.Sigma[np.ix_(best_k_indices, best_k_indices)]
        w_current_k = w_current[best_k_indices]

        # Normalize objectives
        w_equal_k = np.ones(K) / K
        f1_base = abs(np.dot(-w_equal_k, mu_k))
        f2_base = np.dot(w_equal_k, np.dot(Sigma_k, w_equal_k))
        f3_base = c_prop * np.sum(np.abs(w_equal_k - w_current_k))

        # Generate random weight combinations for (α, β, γ)
        np.random.seed(42)
        weight_combinations = []

        # Systematic sampling
        for alpha in np.linspace(0, 1, 6):
            for beta in np.linspace(0, 1 - alpha, 6):
                gamma = 1 - alpha - beta
                if gamma >= 0:
                    weight_combinations.append((alpha, beta, gamma))

        # Random sampling
        for _ in range(n_points - len(weight_combinations)):
            alpha = np.random.random()
            beta = np.random.random() * (1 - alpha)
            gamma = 1 - alpha - beta
            weight_combinations.append((alpha, beta, gamma))

        weight_combinations = weight_combinations[:n_points]

        for i, (alpha, beta, gamma) in enumerate(weight_combinations):
            # Scalarized objective on K assets
            def objective(w_k):
                f1 = -np.dot(w_k, mu_k) / f1_base if f1_base > 0 else 0
                f2 = np.dot(w_k, np.dot(Sigma_k, w_k)) / f2_base if f2_base > 0 else 0
                f3 = (c_prop * np.sum(np.abs(w_k - w_current_k))) / f3_base if f3_base > 0 else 0
                return alpha * f1 + beta * f2 + gamma * f3

            # Optimize on K assets
            w_init_k = np.ones(K) / K
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0.001, 0.5) for _ in range(K)]  # min weight 0.1% to ensure K assets

            result = minimize(objective, w_init_k, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 500, 'disp': False})

            if result.success:
                w_k = result.x

                # Reconstruct full portfolio
                w_full = np.zeros(self.n_assets)
                w_full[best_k_indices] = w_k

                pareto_solutions.append({
                    'weights': w_full.tolist(),
                    'f1_return': float(-self.f1_return(w_full)),
                    'f2_variance': float(self.f2_variance(w_full)),
                    'f2_volatility': float(np.sqrt(self.f2_variance(w_full))),
                    'f3_transaction_cost': float(f3_transaction_cost(w_full, w_current, c_prop)),
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'gamma': float(gamma),
                    'K': int(K),
                    'n_assets': int(np.sum(w_full > 0.001))
                })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_points} points generated")

        print(f"\n[OK] Generated {len(pareto_solutions)} Pareto-optimal portfolios")

        # Display range
        returns = [sol['f1_return'] for sol in pareto_solutions]
        vols = [sol['f2_volatility'] for sol in pareto_solutions]
        costs = [sol['f3_transaction_cost'] for sol in pareto_solutions]
        print(f"  Return range: {min(returns):.2%} to {max(returns):.2%}")
        print(f"  Risk range: {min(vols):.2%} to {max(vols):.2%}")
        print(f"  Cost range: {min(costs):.4f} to {max(costs):.4f}")

        return pareto_solutions

    # === Utility Functions ===

    def save_pareto_front(self, solutions, filename):
        """Save Pareto front solutions to JSON"""
        with open(filename, 'w') as f:
            json.dump(solutions, f, indent=2)
        print(f"\n[OK] Saved Pareto front to {filename}")

    def get_portfolio_details(self, weights):
        """Get detailed allocation for a portfolio"""
        portfolio = []
        for i, w in enumerate(weights):
            if w > 0.001:  # Only include meaningful positions
                portfolio.append({
                    'ticker': self.asset_names[i],
                    'sector': self.sector_mapping.get(self.asset_names[i], 'Unknown'),
                    'weight': w
                })

        # Sort by weight
        portfolio = sorted(portfolio, key=lambda x: x['weight'], reverse=True)

        return portfolio


if __name__ == "__main__":
    # Test the optimizer
    from data_loader import load_portfolio_data

    data_dir = "../multi-criteria-optimization/data"
    _, mu, Sigma, sectors = load_portfolio_data(data_dir)

    optimizer = ParetoPortfolioOptimizer(mu, Sigma, sectors)

    # Test Level 1
    pareto_l1 = optimizer.generate_pareto_front_level1(n_points=50)
    optimizer.save_pareto_front(pareto_l1, 'pareto_level1.json')

    # Test Level 2
    pareto_l2 = optimizer.generate_pareto_front_level2(K=20, n_points=30)
    optimizer.save_pareto_front(pareto_l2, 'pareto_level2.json')
