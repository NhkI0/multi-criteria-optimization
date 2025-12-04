"""
Optimisation Multi-Critère de Portefeuille
Projet de Mathématiques - M1 Data

Objectifs:
- Maximiser le rendement espéré
- Minimiser le risque (variance)

Contraintes:
- Budget total
- Diversification
- Limites par secteur
- Non-négativité
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
DATA_DIR = Path("./data")
INITIAL_CAPITAL = 1_000_000  # Capital initial en euros
RISK_FREE_RATE = 0.02  # Taux sans risque annuel (2%)


class PortfolioOptimizer:
    def __init__(self, data_dir=DATA_DIR):
        """Initialise l'optimiseur avec les données de marché."""
        self.data_dir = data_dir
        self.sectors = {}
        self.all_returns = None
        self.asset_names = []
        self.sector_mapping = {}

    def load_data(self):
        """Charge les données de tous les secteurs."""
        print("Chargement des données...")

        all_dataframes = []

        for csv_file in self.data_dir.glob("*.csv"):
            sector_name = csv_file.stem.replace('_', ' ')
            print(f"  - {sector_name}")

            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            df = df.dropna(how='all')

            # Garde trace du secteur pour chaque actif
            for col in df.columns:
                self.sector_mapping[col] = sector_name

            self.sectors[sector_name] = df
            all_dataframes.append(df)

        # Combine tous les actifs
        self.all_data = pd.concat(all_dataframes, axis=1)
        self.all_data = self.all_data.dropna(axis=1, thresh=len(self.all_data) * 0.8)  # Garde les actifs avec au moins 80% de données

        self.asset_names = list(self.all_data.columns)
        self.n_assets = len(self.asset_names)

        print(f"\nNombre total d'actifs: {self.n_assets}")
        print(f"Période: {self.all_data.index[0]} à {self.all_data.index[-1]}")

    def calculate_returns(self):
        """Calcule les rendements journaliers et statistiques."""
        print("\nCalcul des rendements...")

        # Rendements journaliers
        self.returns = self.all_data.pct_change().dropna()

        # Rendements moyens annualisés
        self.mean_returns = self.returns.mean() * 252

        # Matrice de covariance annualisée
        self.cov_matrix = self.returns.cov() * 252

        print(f"Rendement moyen annuel: {self.mean_returns.mean():.2%}")
        print(f"Volatilité moyenne annuelle: {np.sqrt(np.diag(self.cov_matrix)).mean():.2%}")

    def portfolio_performance(self, weights):
        """Calcule le rendement et le risque d'un portefeuille."""
        returns = np.dot(weights, self.mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return returns, risk

    def portfolio_sharpe_ratio(self, weights):
        """Calcule le ratio de Sharpe."""
        returns, risk = self.portfolio_performance(weights)
        return (returns - RISK_FREE_RATE) / risk

    def negative_sharpe(self, weights):
        """Fonction objectif pour maximiser le ratio de Sharpe."""
        return -self.portfolio_sharpe_ratio(weights)

    def portfolio_variance(self, weights):
        """Fonction objectif pour minimiser la variance."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def optimize_max_sharpe(self):
        """Optimise le portefeuille pour maximiser le ratio de Sharpe."""
        print("\nOptimisation: Maximisation du ratio de Sharpe...")

        # Poids initiaux égaux
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Contraintes
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Somme = 1
        )

        # Bornes: entre 0% et 10% par actif
        bounds = tuple((0, 0.1) for _ in range(self.n_assets))

        result = minimize(
            self.negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            returns, risk = self.portfolio_performance(result.x)
            sharpe = self.portfolio_sharpe_ratio(result.x)

            print(f"  Rendement attendu: {returns:.2%}")
            print(f"  Risque (volatilité): {risk:.2%}")
            print(f"  Ratio de Sharpe: {sharpe:.3f}")

            return result.x, returns, risk, sharpe
        else:
            print("  Optimisation échouée!")
            return None, None, None, None

    def optimize_min_variance(self):
        """Optimise le portefeuille pour minimiser la variance."""
        print("\nOptimisation: Minimisation de la variance...")

        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        )

        bounds = tuple((0, 0.1) for _ in range(self.n_assets))

        result = minimize(
            self.portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            returns, risk = self.portfolio_performance(result.x)
            sharpe = self.portfolio_sharpe_ratio(result.x)

            print(f"  Rendement attendu: {returns:.2%}")
            print(f"  Risque (volatilité): {risk:.2%}")
            print(f"  Ratio de Sharpe: {sharpe:.3f}")

            return result.x, returns, risk, sharpe
        else:
            print("  Optimisation échouée!")
            return None, None, None, None

    def generate_efficient_frontier(self, n_portfolios=100):
        """Génère la frontière efficiente avec différents niveaux de risque cibles."""
        print(f"\nGénération de la frontière efficiente ({n_portfolios} portefeuilles)...")

        # Trouve les rendements min et max possibles
        weights_min_var, _, risk_min, _ = self.optimize_min_variance()
        weights_max_sharpe, ret_max, _, _ = self.optimize_max_sharpe()

        # Génère des rendements cibles
        target_returns = np.linspace(self.mean_returns.min(), ret_max, n_portfolios)

        frontier_returns = []
        frontier_risks = []
        frontier_weights = []

        for target_ret in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.mean_returns) - target_ret}
            )

            bounds = tuple((0, 0.1) for _ in range(self.n_assets))

            initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

            result = minimize(
                self.portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'disp': False}
            )

            if result.success:
                returns, risk = self.portfolio_performance(result.x)
                frontier_returns.append(returns)
                frontier_risks.append(risk)
                frontier_weights.append(result.x)

        print(f"  {len(frontier_returns)} portefeuilles générés avec succès")

        return np.array(frontier_weights), np.array(frontier_returns), np.array(frontier_risks)

    def plot_efficient_frontier(self, frontier_returns, frontier_risks,
                               max_sharpe_ret=None, max_sharpe_risk=None,
                               min_var_ret=None, min_var_risk=None):
        """Visualise la frontière efficiente."""
        plt.figure(figsize=(12, 8))

        # Frontière efficiente
        plt.plot(frontier_risks, frontier_returns, 'b-', linewidth=2, label='Frontière efficiente')

        # Points spéciaux
        if max_sharpe_ret is not None:
            plt.scatter(max_sharpe_risk, max_sharpe_ret, c='red', s=200, marker='*',
                       label='Max Sharpe Ratio', edgecolors='black', linewidths=2, zorder=5)

        if min_var_ret is not None:
            plt.scatter(min_var_risk, min_var_ret, c='green', s=200, marker='s',
                       label='Min Variance', edgecolors='black', linewidths=2, zorder=5)

        plt.xlabel('Risque (Volatilité)', fontsize=12)
        plt.ylabel('Rendement Espéré', fontsize=12)
        plt.title('Frontière Efficiente - Optimisation Multi-Critère', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
        print("\nGraphique sauvegardé: efficient_frontier.png")
        plt.show()

    def analyze_portfolio(self, weights, name="Portefeuille"):
        """Analyse détaillée d'un portefeuille."""
        print(f"\n{'='*60}")
        print(f"ANALYSE: {name}")
        print('='*60)

        # Performance globale
        returns, risk = self.portfolio_performance(weights)
        sharpe = self.portfolio_sharpe_ratio(weights)

        print(f"\nPerformance Globale:")
        print(f"  Rendement annuel espéré: {returns:.2%}")
        print(f"  Risque (volatilité annuelle): {risk:.2%}")
        print(f"  Ratio de Sharpe: {sharpe:.3f}")
        print(f"  Capital investi: {INITIAL_CAPITAL:,.0f} $")
        print(f"  Rendement espéré: {INITIAL_CAPITAL * returns:,.0f} $/an")

        # Top 10 positions
        sorted_indices = np.argsort(weights)[::-1]
        top_10_indices = sorted_indices[:10]

        print(f"\nTop 10 Positions:")
        for i, idx in enumerate(top_10_indices, 1):
            if weights[idx] > 0.001:  # Au moins 0.1%
                asset = self.asset_names[idx]
                sector = self.sector_mapping.get(asset, "Unknown")
                weight = weights[idx]
                investment = INITIAL_CAPITAL * weight
                print(f"  {i:2d}. {asset:8s} ({sector:25s}): {weight:6.2%} = {investment:12,.0f} $")

        # Allocation par secteur
        sector_allocation = {}
        for asset, weight in zip(self.asset_names, weights):
            sector = self.sector_mapping.get(asset, "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0) + weight

        print(f"\nAllocation par Secteur:")
        for sector, allocation in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
            if allocation > 0.001:
                print(f"  {sector:30s}: {allocation:6.2%}")

        print('='*60)

    def save_portfolio(self, weights, filename="portfolio_optimal.csv"):
        """Sauvegarde un portefeuille dans un fichier CSV."""
        portfolio_df = pd.DataFrame({
            'Ticker': self.asset_names,
            'Secteur': [self.sector_mapping.get(a, "Unknown") for a in self.asset_names],
            'Poids': weights,
            'Investissement ($)': weights * INITIAL_CAPITAL
        })

        # Filtre les positions > 0.1%
        portfolio_df = portfolio_df[portfolio_df['Poids'] > 0.001]
        portfolio_df = portfolio_df.sort_values('Poids', ascending=False)

        portfolio_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nPortefeuille sauvegardé: {filename}")


def main():
    """Fonction principale."""
    print("="*60)
    print("OPTIMISATION MULTI-CRITÈRE DE PORTEFEUILLE")
    print("="*60)

    # Initialisation
    optimizer = PortfolioOptimizer()

    # Chargement des données
    optimizer.load_data()
    optimizer.calculate_returns()

    # Optimisation: Max Sharpe Ratio
    weights_sharpe, ret_sharpe, risk_sharpe, sharpe_ratio = optimizer.optimize_max_sharpe()
    if weights_sharpe is not None:
        optimizer.analyze_portfolio(weights_sharpe, "Portefeuille Max Sharpe")
        optimizer.save_portfolio(weights_sharpe, "portfolio_max_sharpe.csv")

    # Optimisation: Min Variance
    weights_minvar, ret_minvar, risk_minvar, _ = optimizer.optimize_min_variance()
    if weights_minvar is not None:
        optimizer.analyze_portfolio(weights_minvar, "Portefeuille Min Variance")
        optimizer.save_portfolio(weights_minvar, "portfolio_min_variance.csv")

    # Génération de la frontière efficiente
    frontier_weights, frontier_returns, frontier_risks = optimizer.generate_efficient_frontier(50)

    # Visualisation
    optimizer.plot_efficient_frontier(
        frontier_returns, frontier_risks,
        ret_sharpe, risk_sharpe,
        ret_minvar, risk_minvar
    )

    print("\n" + "="*60)
    print("OPTIMISATION TERMINÉE")
    print("="*60)


if __name__ == "__main__":
    main()
