"""
Multi-objective portfolio optimization with Pareto front generation
Implements scalarization and epsilon-constraint methods
"""
import numpy as np
from scipy.optimize import minimize
import json


def f3_transaction_cost(w, w_current, c_prop=0.001):
    """f3(w) = Σ c_prop |w_i - w_current_i| (transaction costs)"""
    return c_prop * np.sum(np.abs(w - w_current))


def deduplicate_portfolios(portfolios, tolerance=1e-4):
    """
    Remove duplicate portfolios that are too similar

    Args:
        portfolios: List of portfolio dictionaries
        tolerance: Maximum difference in weights to consider portfolios as duplicates

    Returns:
        Deduplicated list of portfolios
    """
    if not portfolios:
        return portfolios

    unique_portfolios = []

    for portfolio in portfolios:
        w = np.array(portfolio['weights'])
        is_duplicate = False

        # Check if this portfolio is too similar to any existing one
        for unique_port in unique_portfolios:
            w_unique = np.array(unique_port['weights'])

            # Calculate maximum absolute difference in weights
            max_diff = np.max(np.abs(w - w_unique))

            if max_diff < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_portfolios.append(portfolio)

    return unique_portfolios


def save_pareto_front(solutions, filename):
    """Save Pareto front solutions to JSON"""
    with open(filename, 'w') as f:
        json.dump(solutions, f, indent=2)
    print(f"\n[OK] Saved Pareto front to {filename}")


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

    def compute_sector_info(self, w, threshold=0.001):
        """
        Compute sector allocation information for a portfolio

        Args:
            w: Portfolio weights
            threshold: Minimum weight to consider an asset

        Returns:
            dict with 'sector_weights' and 'sector_allocation'
        """
        sector_weights = {}
        sector_allocation = {}

        for i, weight in enumerate(w):
            if weight > threshold:
                asset_name = self.asset_names[i]
                sector = self.sector_mapping.get(asset_name, "Unknown")

                # Accumulate weights by sector
                sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

                # Track which assets belong to each sector
                if sector not in sector_allocation:
                    sector_allocation[sector] = []
                sector_allocation[sector].append(i)

        return {
            'sector_weights': sector_weights,
            'sector_allocation': sector_allocation
        }

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

                # Compute sector information
                sector_info = self.compute_sector_info(w)

                pareto_solutions.append({
                    'weights': w.tolist(),
                    'f1_negative_return': float(self.f1_return(w)),
                    'f1_return': float(-self.f1_return(w)),  # Actual return
                    'f2_variance': float(self.f2_variance(w)),
                    'f2_volatility': float(np.sqrt(self.f2_variance(w))),  # More interpretable
                    'alpha': float(alpha),
                    'n_assets': int(np.sum(w > 0.001)),
                    'sector_weights': sector_info['sector_weights'],
                    'sector_allocation': sector_info['sector_allocation']
                })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_points} points generated")

        print(f"\n[OK] Generated {len(pareto_solutions)} Pareto-optimal portfolios")

        # Remove duplicates
        original_count = len(pareto_solutions)
        pareto_solutions = deduplicate_portfolios(pareto_solutions, tolerance=1e-4)
        if len(pareto_solutions) < original_count:
            print(f"  Removed {original_count - len(pareto_solutions)} duplicates, {len(pareto_solutions)} unique portfolios remain")

        # Display range
        returns = [-sol['f1_negative_return'] for sol in pareto_solutions]
        vols = [sol['f2_volatility'] for sol in pareto_solutions]
        print(f"  Return range: {min(returns):.2%} to {max(returns):.2%}")
        print(f"  Risk range: {min(vols):.2%} to {max(vols):.2%}")

        return pareto_solutions

    # === LEVEL 2: Tri-objective (f1, f2, f3) with Cardinality ===

    def select_k_assets_adaptive(self, K, alpha, beta, gamma, w_current):
        """
        Select K assets adaptively based on objective weights (α, β, γ)

        This creates different asset selections for different trade-offs:
        - High α (return): Select K assets with high returns
        - High β (risk): Select K assets with low volatility
        - High γ (cost): Select K assets from current portfolio

        Args:
            K: Number of assets to select
            alpha: Weight for return objective
            beta: Weight for risk objective
            gamma: Weight for transaction cost objective
            w_current: Current portfolio weights

        Returns:
            indices: Array of K selected asset indices
        """
        n = self.n_assets

        # Compute a value score for each asset based on objective weights
        scores = np.zeros(n)

        # Return component: favor high returns
        if alpha > 0:
            # Normalize returns to [0, 1] range for fair comparison
            normalized_returns = (self.mu - self.mu.min()) / (self.mu.max() - self.mu.min() + 1e-10)
            scores += alpha * normalized_returns

        # Risk component: favor low volatility (negate because lower is better)
        if beta > 0:
            vols = np.sqrt(np.diag(self.Sigma))
            # Normalize and invert (lower volatility = higher score)
            normalized_vols = (vols - vols.min()) / (vols.max() - vols.min() + 1e-10)
            scores += beta * (1 - normalized_vols)  # Invert: low vol gets high score

        # Cost component: favor assets in current portfolio
        if gamma > 0:
            # Normalize current weights
            normalized_current = w_current / (w_current.max() + 1e-10)
            scores += gamma * normalized_current

        # Select top K assets by score
        top_k_indices = np.argsort(scores)[-K:]

        return np.sort(top_k_indices)

    def generate_pareto_front_level2(self, K=20, w_current=None, c_prop=0.001, n_points=30):
        """
        Generate Pareto front for Level 2: min{f1, f2, f3} under C_Base ∩ C_Op

        Method: Adaptive K-asset selection + weighted sum scalarization
        Minimize: α·f1(w) + β·f2(w) + γ·f3(w)
        Subject to: Σ I(w_i > δ_tol) = K

        KEY IMPROVEMENT: Each portfolio selects DIFFERENT K assets based on (α, β, γ),
        creating a true 3D pyramid/shell-shaped Pareto front.

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
        print(f"Method: Adaptive K-asset Selection + Weighted Sum")
        print(f"Cardinality: K={K}")
        print(f"Points: {n_points}")
        print(f"Note: Each portfolio selects different K assets based on objectives")

        if w_current is None:
            w_current = np.ones(self.n_assets) / self.n_assets

        pareto_solutions = []

        # Generate weight combinations for (α, β, γ)
        np.random.seed(42)
        weight_combinations = []

        # Systematic sampling to ensure good coverage
        for alpha in np.linspace(0, 1, 6):
            for beta in np.linspace(0, 1 - alpha, 6):
                gamma = 1 - alpha - beta
                if gamma >= 0:
                    weight_combinations.append((alpha, beta, gamma))

        # Random sampling for additional diversity
        for _ in range(n_points - len(weight_combinations)):
            alpha = np.random.random()
            beta = np.random.random() * (1 - alpha)
            gamma = 1 - alpha - beta
            weight_combinations.append((alpha, beta, gamma))

        weight_combinations = weight_combinations[:n_points]

        # Track asset selection diversity
        asset_selections = []

        for i, (alpha, beta, gamma) in enumerate(weight_combinations):
            # ADAPTIVE SELECTION: Choose K assets based on this specific (α, β, γ)
            selected_indices = self.select_k_assets_adaptive(K, alpha, beta, gamma, w_current)
            asset_selections.append(selected_indices)

            # Reduced problem on selected K assets
            mu_k = self.mu[selected_indices]
            Sigma_k = self.Sigma[np.ix_(selected_indices, selected_indices)]
            w_current_k = w_current[selected_indices]

            # Normalize objectives for this specific K-asset subset
            w_equal_k = np.ones(K) / K
            f1_base = abs(np.dot(-w_equal_k, mu_k)) + 1e-10
            f2_base = np.dot(w_equal_k, np.dot(Sigma_k, w_equal_k)) + 1e-10
            f3_base = c_prop * np.sum(np.abs(w_equal_k - w_current_k)) + 1e-10

            # Scalarized objective on selected K assets
            def objective(w_k):
                f1 = -np.dot(w_k, mu_k) / f1_base
                f2 = np.dot(w_k, np.dot(Sigma_k, w_k)) / f2_base
                f3 = (c_prop * np.sum(np.abs(w_k - w_current_k))) / f3_base
                return alpha * f1 + beta * f2 + gamma * f3

            # Optimize weights on selected K assets
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
                w_full[selected_indices] = w_k

                # Compute sector information
                sector_info = self.compute_sector_info(w_full)

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
                    'n_assets': int(np.sum(w_full > 0.001)),
                    'selected_assets': selected_indices.tolist(),
                    'sector_weights': sector_info['sector_weights'],
                    'sector_allocation': sector_info['sector_allocation']
                })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_points} points generated")

        print(f"\n[OK] Generated {len(pareto_solutions)} Pareto-optimal portfolios")

        # Analyze asset selection diversity
        unique_selections = len(set([tuple(sel) for sel in asset_selections]))
        print(f"  Asset selection diversity: {unique_selections} unique K-asset combinations (out of {len(asset_selections)})")

        # Remove duplicates
        original_count = len(pareto_solutions)
        pareto_solutions = deduplicate_portfolios(pareto_solutions, tolerance=1e-4)
        if len(pareto_solutions) < original_count:
            print(f"  Removed {original_count - len(pareto_solutions)} duplicates, {len(pareto_solutions)} unique portfolios remain")

        # Display range
        returns = [sol['f1_return'] for sol in pareto_solutions]
        vols = [sol['f2_volatility'] for sol in pareto_solutions]
        costs = [sol['f3_transaction_cost'] for sol in pareto_solutions]
        print(f"  Return range: {min(returns):.2%} to {max(returns):.2%}")
        print(f"  Risk range: {min(vols):.2%} to {max(vols):.2%}")
        print(f"  Cost range: {min(costs):.4f} to {max(costs):.4f}")

        return pareto_solutions

    # === Utility Functions ===

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
    save_pareto_front(pareto_l1, 'pareto_level1.json')

    # Test Level 2
    pareto_l2 = optimizer.generate_pareto_front_level2(K=20, n_points=30)
    save_pareto_front(pareto_l2, 'pareto_level2.json')
