"""
Application Streamlit - Optimisation Multi-Critère de Portefeuille
Interface interactive pour l'optimisation et l'analyse de portefeuille
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# === Configuration ===
DATA_DIR = Path("./data")

class PortfolioOptimizer:
    def __init__(self, data_dir=DATA_DIR, initial_capital=1_000_000, risk_free_rate=0.02):
        """Initialise l'optimiseur avec les données de marché."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.sectors = {}
        self.all_returns = None
        self.asset_names = []
        self.sector_mapping = {}

    def load_data(self):
        """Charge les données de tous les secteurs."""
        all_dataframes = []

        for csv_file in self.data_dir.glob("*.csv"):
            sector_name = csv_file.stem.replace('_', ' ')
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            df = df.dropna(how='all')

            for col in df.columns:
                self.sector_mapping[col] = sector_name

            self.sectors[sector_name] = df
            all_dataframes.append(df)

        self.all_data = pd.concat(all_dataframes, axis=1)
        self.all_data = self.all_data.dropna(axis=1, thresh=len(self.all_data) * 0.8)
        self.asset_names = list(self.all_data.columns)
        self.n_assets = len(self.asset_names)

    def calculate_returns(self):
        """Calcule les rendements journaliers et statistiques."""
        self.returns = self.all_data.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252

    def portfolio_performance(self, weights):
        """Calcule le rendement et le risque d'un portefeuille."""
        returns = np.dot(weights, self.mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return returns, risk

    def portfolio_sharpe_ratio(self, weights):
        """Calcule le ratio de Sharpe."""
        returns, risk = self.portfolio_performance(weights)
        return (returns - self.risk_free_rate) / risk

    def negative_sharpe(self, weights):
        """Fonction objectif pour maximiser le ratio de Sharpe."""
        return -self.portfolio_sharpe_ratio(weights)

    def portfolio_variance(self, weights):
        """Fonction objectif pour minimiser la variance."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def optimize(self, objective='max_sharpe', max_weight=0.1, min_weight=0.0):
        """Optimise le portefeuille selon l'objectif spécifié."""
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        )

        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))

        if objective == 'max_sharpe':
            obj_func = self.negative_sharpe
        else:  # min_variance
            obj_func = self.portfolio_variance

        result = minimize(
            obj_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': False}
        )

        if result.success:
            returns, risk = self.portfolio_performance(result.x)
            sharpe = self.portfolio_sharpe_ratio(result.x)
            return result.x, returns, risk, sharpe
        else:
            return None, None, None, None

    def generate_efficient_frontier(self, n_portfolios=50, max_weight=0.1):
        """Génère la frontière efficiente."""
        weights_min_var, _, risk_min, _ = self.optimize('min_variance', max_weight)
        weights_max_sharpe, ret_max, _, _ = self.optimize('max_sharpe', max_weight)

        target_returns = np.linspace(self.mean_returns.min(), ret_max, n_portfolios)

        frontier_returns = []
        frontier_risks = []
        frontier_weights = []

        for target_ret in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.mean_returns) - target_ret}
            )

            bounds = tuple((0, max_weight) for _ in range(self.n_assets))
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

        return np.array(frontier_weights), np.array(frontier_returns), np.array(frontier_risks)

    def get_portfolio_allocation(self, weights):
        """Retourne l'allocation détaillée du portefeuille."""
        portfolio_df = pd.DataFrame({
            'Ticker': self.asset_names,
            'Secteur': [self.sector_mapping.get(a, "Unknown") for a in self.asset_names],
            'Poids': weights,
            'Investissement ($)': weights * self.initial_capital
        })

        portfolio_df = portfolio_df[portfolio_df['Poids'] > 0.001]
        portfolio_df = portfolio_df.sort_values('Poids', ascending=False)

        return portfolio_df

    def get_sector_allocation(self, weights):
        """Retourne l'allocation par secteur."""
        sector_allocation = {}
        for asset, weight in zip(self.asset_names, weights):
            sector = self.sector_mapping.get(asset, "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0) + weight

        return pd.DataFrame(list(sector_allocation.items()), columns=['Secteur', 'Allocation']).sort_values('Allocation', ascending=False)


# === Initialisation de la session ===
@st.cache_resource
def load_optimizer():
    """Charge l'optimiseur une seule fois."""
    optimizer = PortfolioOptimizer()
    optimizer.load_data()
    optimizer.calculate_returns()
    return optimizer


# === Application principale ===
def main():
    # En-tête
    st.markdown('<div class="main-header">📈 Optimisation Multi-Critère de Portefeuille</div>', unsafe_allow_html=True)

    # Chargement des données
    with st.spinner('Chargement des données...'):
        optimizer = load_optimizer()

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # Paramètres de capital
    initial_capital = st.sidebar.number_input(
        "Capital Initial ($)",
        min_value=10_000,
        max_value=100_000_000,
        value=1_000_000,
        step=100_000,
        format="%d"
    )
    optimizer.initial_capital = initial_capital

    # Paramètres d'optimisation
    st.sidebar.subheader("Contraintes de Portefeuille")
    max_weight = st.sidebar.slider(
        "Poids maximum par actif (%)",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    ) / 100

    risk_free_rate = st.sidebar.slider(
        "Taux sans risque (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    ) / 100
    optimizer.risk_free_rate = risk_free_rate

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "🎯 Optimisation", "📈 Frontière Efficiente", "📋 Comparaison"])

    # === TAB 1: Vue d'ensemble ===
    with tab1:
        st.header("Vue d'ensemble des données")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nombre d'actifs", optimizer.n_assets)

        with col2:
            st.metric("Nombre de secteurs", len(optimizer.sectors))

        with col3:
            st.metric("Rendement moyen", f"{optimizer.mean_returns.mean():.2%}")

        with col4:
            st.metric("Volatilité moyenne", f"{np.sqrt(np.diag(optimizer.cov_matrix)).mean():.2%}")

        st.subheader("Distribution des actifs par secteur")
        sector_counts = pd.Series([optimizer.sector_mapping[a] for a in optimizer.asset_names]).value_counts()

        fig = px.bar(
            x=sector_counts.index,
            y=sector_counts.values,
            labels={'x': 'Secteur', 'y': 'Nombre d\'actifs'},
            title="Nombre d'actifs par secteur"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistiques détaillées
        st.subheader("Top 10 des actifs par rendement")
        top_returns = optimizer.mean_returns.nlargest(10)
        top_df = pd.DataFrame({
            'Ticker': top_returns.index,
            'Secteur': [optimizer.sector_mapping.get(t, "Unknown") for t in top_returns.index],
            'Rendement Annuel': [f"{r:.2%}" for r in top_returns.values]
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    # === TAB 2: Optimisation ===
    with tab2:
        st.header("Optimisation de Portefeuille")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Maximisation du Ratio de Sharpe")
            st.write("Objectif: Meilleur rendement ajusté au risque")

            if st.button("Optimiser (Max Sharpe)", use_container_width=True):
                with st.spinner('Optimisation en cours...'):
                    weights, returns, risk, sharpe = optimizer.optimize('max_sharpe', max_weight)

                    if weights is not None:
                        st.success("Optimisation réussie!")

                        # Métriques
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric("Rendement", f"{returns:.2%}")
                        metric_col2.metric("Risque", f"{risk:.2%}")
                        metric_col3.metric("Sharpe", f"{sharpe:.3f}")

                        # Allocation
                        st.subheader("Top 10 Positions")
                        portfolio_df = optimizer.get_portfolio_allocation(weights).head(10)
                        portfolio_df['Poids'] = portfolio_df['Poids'].apply(lambda x: f"{x:.2%}")
                        portfolio_df['Investissement ($)'] = portfolio_df['Investissement ($)'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

                        # Allocation par secteur
                        st.subheader("Allocation par Secteur")
                        sector_df = optimizer.get_sector_allocation(weights)
                        fig = px.pie(
                            sector_df,
                            values='Allocation',
                            names='Secteur',
                            title='Répartition par secteur'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Bouton de téléchargement
                        csv = optimizer.get_portfolio_allocation(weights).to_csv(index=False)
                        st.download_button(
                            "📥 Télécharger le portefeuille",
                            csv,
                            "portfolio_max_sharpe.csv",
                            "text/csv"
                        )

        with col2:
            st.subheader("🛡️ Minimisation de la Variance")
            st.write("Objectif: Risque le plus faible possible")

            if st.button("Optimiser (Min Variance)", use_container_width=True):
                with st.spinner('Optimisation en cours...'):
                    weights, returns, risk, sharpe = optimizer.optimize('min_variance', max_weight)

                    if weights is not None:
                        st.success("Optimisation réussie!")

                        # Métriques
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric("Rendement", f"{returns:.2%}")
                        metric_col2.metric("Risque", f"{risk:.2%}")
                        metric_col3.metric("Sharpe", f"{sharpe:.3f}")

                        # Allocation
                        st.subheader("Top 10 Positions")
                        portfolio_df = optimizer.get_portfolio_allocation(weights).head(10)
                        portfolio_df['Poids'] = portfolio_df['Poids'].apply(lambda x: f"{x:.2%}")
                        portfolio_df['Investissement ($)'] = portfolio_df['Investissement ($)'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

                        # Allocation par secteur
                        st.subheader("Allocation par Secteur")
                        sector_df = optimizer.get_sector_allocation(weights)
                        fig = px.pie(
                            sector_df,
                            values='Allocation',
                            names='Secteur',
                            title='Répartition par secteur'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Bouton de téléchargement
                        csv = optimizer.get_portfolio_allocation(weights).to_csv(index=False)
                        st.download_button(
                            "📥 Télécharger le portefeuille",
                            csv,
                            "portfolio_min_variance.csv",
                            "text/csv"
                        )

    # === TAB 3: Frontière Efficiente ===
    with tab3:
        st.header("Frontière Efficiente")
        st.write("Visualisation de l'ensemble des portefeuilles optimaux")

        n_portfolios = st.slider(
            "Nombre de portefeuilles à générer",
            min_value=20,
            max_value=100,
            value=50,
            step=10
        )

        if st.button("Générer la Frontière Efficiente", use_container_width=True):
            with st.spinner('Génération en cours...'):
                # Génère la frontière
                frontier_weights, frontier_returns, frontier_risks = optimizer.generate_efficient_frontier(n_portfolios, max_weight)

                # Calcule les points spéciaux
                weights_sharpe, ret_sharpe, risk_sharpe, sharpe_ratio = optimizer.optimize('max_sharpe', max_weight)
                weights_minvar, ret_minvar, risk_minvar, _ = optimizer.optimize('min_variance', max_weight)

                # Crée le graphique interactif
                fig = go.Figure()

                # Frontière efficiente
                fig.add_trace(go.Scatter(
                    x=frontier_risks,
                    y=frontier_returns,
                    mode='lines',
                    name='Frontière Efficiente',
                    line=dict(color='blue', width=3),
                    hovertemplate='<b>Risque:</b> %{x:.2%}<br><b>Rendement:</b> %{y:.2%}<extra></extra>'
                ))

                # Point Max Sharpe
                fig.add_trace(go.Scatter(
                    x=[risk_sharpe],
                    y=[ret_sharpe],
                    mode='markers',
                    name='Max Sharpe Ratio',
                    marker=dict(color='red', size=15, symbol='star'),
                    hovertemplate=f'<b>Max Sharpe</b><br>Risque: {risk_sharpe:.2%}<br>Rendement: {ret_sharpe:.2%}<br>Sharpe: {sharpe_ratio:.3f}<extra></extra>'
                ))

                # Point Min Variance
                fig.add_trace(go.Scatter(
                    x=[risk_minvar],
                    y=[ret_minvar],
                    mode='markers',
                    name='Min Variance',
                    marker=dict(color='green', size=15, symbol='square'),
                    hovertemplate=f'<b>Min Variance</b><br>Risque: {risk_minvar:.2%}<br>Rendement: {ret_minvar:.2%}<extra></extra>'
                ))

                fig.update_layout(
                    title='Frontière Efficiente - Optimisation Multi-Critère',
                    xaxis_title='Risque (Volatilité)',
                    yaxis_title='Rendement Espéré',
                    hovermode='closest',
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Statistiques
                st.subheader("Statistiques de la Frontière")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Portefeuilles générés", len(frontier_returns))

                with col2:
                    st.metric("Risque min", f"{frontier_risks.min():.2%}")
                    st.metric("Risque max", f"{frontier_risks.max():.2%}")

                with col3:
                    st.metric("Rendement min", f"{frontier_returns.min():.2%}")
                    st.metric("Rendement max", f"{frontier_returns.max():.2%}")

    # === TAB 4: Comparaison ===
    with tab4:
        st.header("Comparaison des Stratégies")
        st.write("Comparaison côte à côte des différentes stratégies d'optimisation")

        if st.button("Générer la Comparaison", use_container_width=True):
            with st.spinner('Calcul en cours...'):
                # Optimise les deux stratégies
                weights_sharpe, ret_sharpe, risk_sharpe, sharpe_sharpe = optimizer.optimize('max_sharpe', max_weight)
                weights_minvar, ret_minvar, risk_minvar, sharpe_minvar = optimizer.optimize('min_variance', max_weight)

                # Comparaison des métriques
                st.subheader("Métriques de Performance")

                comparison_df = pd.DataFrame({
                    'Métrique': ['Rendement Annuel', 'Risque (Volatilité)', 'Ratio de Sharpe', 'Rendement Espéré ($)'],
                    'Max Sharpe': [
                        f"{ret_sharpe:.2%}",
                        f"{risk_sharpe:.2%}",
                        f"{sharpe_sharpe:.3f}",
                        f"${initial_capital * ret_sharpe:,.0f}"
                    ],
                    'Min Variance': [
                        f"{ret_minvar:.2%}",
                        f"{risk_minvar:.2%}",
                        f"{sharpe_minvar:.3f}",
                        f"${initial_capital * ret_minvar:,.0f}"
                    ]
                })

                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Comparaison des allocations sectorielles
                st.subheader("Comparaison des Allocations Sectorielles")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Max Sharpe Ratio**")
                    sector_sharpe = optimizer.get_sector_allocation(weights_sharpe)
                    fig1 = px.bar(
                        sector_sharpe,
                        x='Secteur',
                        y='Allocation',
                        title='Allocation Max Sharpe'
                    )
                    fig1.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.write("**Min Variance**")
                    sector_minvar = optimizer.get_sector_allocation(weights_minvar)
                    fig2 = px.bar(
                        sector_minvar,
                        x='Secteur',
                        y='Allocation',
                        title='Allocation Min Variance'
                    )
                    fig2.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig2, use_container_width=True)

                # Comparaison des top positions
                st.subheader("Top 5 Positions - Comparaison")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Max Sharpe Ratio**")
                    portfolio_sharpe = optimizer.get_portfolio_allocation(weights_sharpe).head(5)
                    portfolio_sharpe['Poids'] = portfolio_sharpe['Poids'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(portfolio_sharpe[['Ticker', 'Secteur', 'Poids']], use_container_width=True, hide_index=True)

                with col2:
                    st.write("**Min Variance**")
                    portfolio_minvar = optimizer.get_portfolio_allocation(weights_minvar).head(5)
                    portfolio_minvar['Poids'] = portfolio_minvar['Poids'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(portfolio_minvar[['Ticker', 'Secteur', 'Poids']], use_container_width=True, hide_index=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **À propos**

        Application d'optimisation multi-critère de portefeuille.

        Projet M1 Data - Mathématiques
        """
    )


if __name__ == "__main__":
    main()
