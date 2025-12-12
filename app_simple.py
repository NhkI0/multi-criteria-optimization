"""
Improved Streamlit App for Pareto Front Visualization
- On-demand portfolio generation based on slider
- Click on portfolios in graphs
- Meaningful portfolio descriptions
- Detailed holdings with ticker names
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_portfolio_data
from pareto_optimizer import ParetoPortfolioOptimizer
from comparison_methods import EpsilonConstraintOptimizer, NSGA2Optimizer

st.set_page_config(
    page_title="Portfolio Pareto Optimizer",
    page_icon="📊",
    layout="wide"
)


# === Load Data and Initialize Optimizers ===
@st.cache_data
def load_data():
    """Load portfolio data once and cache it"""
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("Data directory not found!")
        st.stop()

    prices, mu, Sigma, sectors = load_portfolio_data(str(data_dir))
    return prices, mu, Sigma, sectors


@st.cache_data
def load_asset_names():
    """Load asset ticker names from data files"""
    data_dir = Path("data")
    if not data_dir.exists():
        return None

    all_tickers = []
    for csv_file in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file, index_col=0, nrows=0)
        all_tickers.extend(df.columns.tolist())

    return all_tickers


# Load data
with st.spinner("Loading portfolio data...", show_time=True):
    prices, mu, Sigma, sectors = load_data()
    asset_names = load_asset_names()

# Initialize optimizers (cached through data)
optimizer = ParetoPortfolioOptimizer(mu, Sigma, sectors)
epsilon_opt = EpsilonConstraintOptimizer(mu, Sigma, sectors)
nsga2_opt = NSGA2Optimizer(mu, Sigma, sectors)


# === On-Demand Portfolio Generation ===


@st.cache_data(show_spinner="Generating Weighted Sum portfolios...", show_time=True)
def generate_weighted_sum_portfolios(n_points, _mu, _Sigma, _sectors):
    """Generate Pareto front using weighted sum method"""
    opt = ParetoPortfolioOptimizer(_mu, _Sigma, _sectors)
    return opt.generate_pareto_front_level1(n_points=n_points)


@st.cache_data(show_spinner="Generating Epsilon-Constraint portfolios...", show_time=True)
def generate_epsilon_portfolios(n_points, _mu, _Sigma, _sectors):
    """Generate Pareto front using epsilon-constraint method"""
    opt = EpsilonConstraintOptimizer(_mu, _Sigma, _sectors)
    return opt.generate_pareto_front_epsilon_level1(n_points=n_points)


@st.cache_data(show_spinner="Generating NSGA-2 portfolios...", show_time=True)
def generate_nsga2_portfolios(pop_size, _mu, _Sigma, _sectors):
    """Generate Pareto front using NSGA-2 method"""
    opt = NSGA2Optimizer(_mu, _Sigma, _sectors)
    # Use more generations for better convergence and exploration
    n_gen = max(150, pop_size * 2)  # At least 150 generations, scales with population
    return opt.generate_pareto_front_nsga2_level1(pop_size=pop_size, n_gen=n_gen, verbose=False)


@st.cache_data(show_spinner="Generating Level 2 portfolios...", show_time=True)
def generate_level2_portfolios(K, n_points, _mu, _Sigma, _sectors):
    """Generate Level 2 tri-objective Pareto front"""
    opt = ParetoPortfolioOptimizer(_mu, _Sigma, _sectors)
    return opt.generate_pareto_front_level2(K=K, w_current=None, c_prop=0.001, n_points=n_points)


# === Helper Functions ===


def get_portfolio_label(portfolio, index=None):
    """Generate a meaningful label for a portfolio"""
    ret = portfolio['f1_return']
    risk = portfolio['f2_volatility']

    # Classify portfolio by risk profile
    if risk < 0.15:
        risk_profile = "Conservative"
    elif risk < 0.25:
        risk_profile = "Balanced"
    else:
        risk_profile = "Aggressive"

    label = f"{risk_profile}: {ret:.1%} return, {risk:.1%} risk"
    return label


def display_portfolio_details(portfolio, asset_names=None, portfolio_value=1_000_000):
    """Display detailed portfolio allocation"""

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{portfolio['f1_return']:.2%}")
    col2.metric("Risk (Volatility)", f"{portfolio['f2_volatility']:.2%}")

    if 'f3_transaction_cost' in portfolio:
        col3.metric("Transaction Cost", f"{portfolio['f3_transaction_cost']:.4f}")
    else:
        sharpe = (portfolio['f1_return'] - 0.02) / portfolio['f2_volatility']
        col3.metric("Sharpe Ratio", f"{sharpe:.3f}")

    col4, col5 = st.columns(2)
    col4.metric("Number of Assets", portfolio['n_assets'])

    # Calculate Sharpe ratio
    sharpe = (portfolio['f1_return'] - 0.02) / portfolio['f2_volatility']
    col5.metric("Sharpe Ratio", f"{sharpe:.3f}")

    st.markdown("---")

    # Holdings
    weights = np.array(portfolio['weights'])
    non_zero_indices = np.where(weights > 0.001)[0]
    sorted_indices = non_zero_indices[np.argsort(weights[non_zero_indices])[::-1]]

    st.subheader(f"Portfolio Holdings ({len(sorted_indices)} assets) - Total Value: ${portfolio_value:,.0f}")

    # Create detailed holdings table with SECTOR column
    holdings_data = []
    for i in sorted_indices:
        ticker = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
        weight = weights[i]
        value = weight * portfolio_value

        # Find sector for this asset
        sector = "Unknown"
        if 'sector_allocation' in portfolio:
            for sec, assets in portfolio['sector_allocation'].items():
                if i in assets:
                    sector = sec
                    break

        holdings_data.append({
            'Ticker': ticker,
            'Sector': sector,
            'Weight': weight,
            'Value ($)': value
        })

    holdings_df = pd.DataFrame(holdings_data)

    # Format for display
    holdings_df_display = holdings_df.copy()
    holdings_df_display['Weight'] = holdings_df_display['Weight'].apply(lambda x: f"{x:.2%}")
    holdings_df_display['Value ($)'] = holdings_df_display['Value ($)'].apply(lambda x: f"${x:,.0f}")

    # Show all holdings
    st.dataframe(holdings_df_display, width='stretch', hide_index=True)

    # Download button with original numeric values for Weight and Value
    csv = holdings_df.to_csv(index=False)
    st.download_button(
        "📥 Download Portfolio CSV",
        csv,
        "portfolio_allocation.csv",
        "text/csv",
        width='stretch'
    )

    st.markdown("---")

    # === Sector Breakdown Visualization ===
    if 'sector_weights' in portfolio and portfolio['sector_weights']:
        st.subheader("📈 Sector Allocation (Macro-Economic Structure)")

        # Calculate sector counts (number of assets per sector)
        sector_counts = {}
        sector_data = portfolio['sector_weights']

        # Get sector for each holding and count
        for i in sorted_indices:
            ticker = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
            # Get sector from portfolio's sector_allocation if available
            if 'sector_allocation' in portfolio:
                for sector, assets in portfolio['sector_allocation'].items():
                    if i in assets:
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                        break

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Pie chart for sector distribution by WEIGHT
            sectors_list = list(sector_data.keys())
            sector_weights_list = list(sector_data.values())

            fig_pie = go.Figure(data=[go.Pie(
                labels=sectors_list,
                values=sector_weights_list,
                hole=0.3,
                textposition='inside',
                textinfo='label+percent',
                marker=dict(colors=None)  # Use default colors
            )])

            fig_pie.update_layout(
                title='Sector Distribution by Weight (%)',
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            # Pie chart for sector distribution by NUMBER OF ASSETS
            if sector_counts:
                sectors_count_list = list(sector_counts.keys())
                count_values = list(sector_counts.values())

                fig_pie_count = go.Figure(data=[go.Pie(
                    labels=sectors_count_list,
                    values=count_values,
                    hole=0.3,
                    textposition='inside',
                    textinfo='label+value',
                    marker=dict(colors=None)  # Use default colors
                )])

                fig_pie_count.update_layout(
                    title='Sector Distribution by Number of Assets',
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig_pie_count, use_container_width=True)

        # Bar charts for detailed view
        col_bar1, col_bar2 = st.columns(2)

        with col_bar1:
            # Bar chart for sector weights
            sector_df = pd.DataFrame({
                'Sector': sectors_list,
                'Weight': sector_weights_list
            }).sort_values('Weight', ascending=True)

            fig_bar = go.Figure(data=[go.Bar(
                x=sector_df['Weight'],
                y=sector_df['Sector'],
                orientation='h',
                marker=dict(color=sector_df['Weight'], colorscale='Viridis'),
                text=[f"{w:.1%}" for w in sector_df['Weight']],
                textposition='auto'
            )])

            fig_bar.update_layout(
                title='Sector Weights (%)',
                xaxis_title='Portfolio Weight',
                yaxis_title='Sector',
                height=400,
                xaxis=dict(tickformat='.1%')
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        with col_bar2:
            # Bar chart for sector counts
            if sector_counts:
                sector_count_df = pd.DataFrame({
                    'Sector': list(sector_counts.keys()),
                    'Count': list(sector_counts.values())
                }).sort_values('Count', ascending=True)

                fig_bar_count = go.Figure(data=[go.Bar(
                    x=sector_count_df['Count'],
                    y=sector_count_df['Sector'],
                    orientation='h',
                    marker=dict(color=sector_count_df['Count'], colorscale='Plasma'),
                    text=sector_count_df['Count'],
                    textposition='auto'
                )])

                fig_bar_count.update_layout(
                    title='Number of Assets per Sector',
                    xaxis_title='Number of Assets',
                    yaxis_title='Sector',
                    height=400
                )

                st.plotly_chart(fig_bar_count, use_container_width=True)

        # Concentration analysis
        max_sector = max(sector_data.items(), key=lambda x: x[1])
        top_3_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)[:3]

        st.info(f"""
        **Sector Concentration Analysis:**
        - **Dominant Sector**: {max_sector[0]} ({max_sector[1]:.1%} of portfolio weight)
        - **Top 3 Sectors**: {', '.join([f"{s[0]} ({s[1]:.1%})" for s in top_3_sectors])}
        - **Diversification**: Portfolio is spread across {len([v for v in sector_data.values() if v > 0.01])} significant sectors
        - **Total Assets**: {len(sorted_indices)} assets across {len(sector_counts)} sectors
        """)




def create_clickable_scatter(portfolios, title, show_3d=False):
    """Create interactive scatter plot with clickable points"""

    if show_3d:
        # 3D plot
        risks = [p['f2_volatility'] for p in portfolios]
        returns = [p['f1_return'] for p in portfolios]
        costs = [p['f3_transaction_cost'] for p in portfolios]
        labels = [get_portfolio_label(p, i) for i, p in enumerate(portfolios)]

        fig = go.Figure()

        # Convert to percentage for display
        risks_pct = [r * 100 for r in risks]
        returns_pct = [r * 100 for r in returns]

        # Create hover text manually to avoid sprintf issues
        hover_texts = [f"<b>{label}</b><br>Risk: {risk:.2f}%<br>Return: {ret:.2f}%<br>Cost: {cost:.4f}"
                       for label, risk, ret, cost in zip(labels, risks_pct, returns_pct, costs)]

        fig.add_trace(go.Scatter3d(
            x=risks_pct,
            y=returns_pct,
            z=costs,
            mode='markers',
            marker=dict(
                size=6,
                color=returns_pct,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Return (%)"),
                line=dict(width=1, color='white')
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            customdata=list(range(len(portfolios)))  # Store index for click detection
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Risk (Volatility) %',
                yaxis_title='Expected Return %',
                zaxis_title='Transaction Cost'
            ),
            height=700,
            hovermode='closest'
        )

    else:
        # 2D plot
        risks = [p['f2_volatility'] for p in portfolios]
        returns = [p['f1_return'] for p in portfolios]
        labels = [get_portfolio_label(p, i) for i, p in enumerate(portfolios)]

        # Convert to percentage for display
        risks_pct = [r * 100 for r in risks]
        returns_pct = [r * 100 for r in returns]

        # Create hover text manually to avoid sprintf issues
        hover_texts = [f"<b>{label}</b><br>Risk: {risk:.2f}%<br>Return: {ret:.2f}%"
                       for label, risk, ret in zip(labels, risks_pct, returns_pct)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=risks_pct,
            y=returns_pct,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=returns_pct,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Return (%)"),
                line=dict(width=1, color='white')
            ),
            line=dict(width=2, color='rgba(100, 149, 237, 0.3)'),
            hovertext=hover_texts,
            hoverinfo='text',
            customdata=list(range(len(portfolios)))
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Risk (Volatility) %',
            yaxis_title='Expected Return %',
            height=600,
            hovermode='closest',
            clickmode='event+select'
        )

    return fig


# === Main App ===

st.title("📊 Portfolio Multi-Objective Optimization")
st.markdown("**Click on any portfolio in the graph to view details**")

# === Sidebar ===
st.sidebar.header("⚙️ Configuration")

# Portfolio value selector
st.sidebar.subheader("💰 Portfolio Settings")
portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)",
    min_value=1_000,
    max_value=100_000_000,
    value=1_000_000,
    step=100_000,
    format="%d",
    help="Enter the total amount you want to invest"
)

# Number of portfolios to generate
st.sidebar.subheader("📊 Optimization Settings")
num_portfolios_display = st.sidebar.slider(
    "Number of Portfolios to Generate",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="Generate this many portfolios using the selected optimization method. More portfolios = "
         "denser Pareto front but slower generation."
)

st.sidebar.markdown("---")

level = st.sidebar.radio(
    "Select Optimization Level",
    ["Level 1: Bi-objective (Return-Risk)",
     "Level 2: Tri-objective (Return-Risk-Costs)"]
)

# Initialize session state for selected portfolio
if 'selected_portfolio_idx' not in st.session_state:
    st.session_state.selected_portfolio_idx = 0

# === LEVEL 1 ===
if "Level 1" in level:
    st.header("Level 1: Bi-objective Optimization")
    st.markdown("**Objectives**: Maximize Return, Minimize Risk")
    st.markdown("**Constraints**: C_Base (budget + no short-selling)")

    # Method selection
    method = st.sidebar.radio(
        "Optimization Method",
        ["Weighted Sum Scalarization", "Epsilon-Constraint Method", "NSGA-2 (Evolutionary)", "Compare All Methods"]
    )

    if method == "Weighted Sum Scalarization":
        portfolios = generate_weighted_sum_portfolios(num_portfolios_display, mu, Sigma, sectors)
        st.subheader("Method: Weighted Sum Scalarization")
        st.latex(r"\min_w \alpha \cdot f_1(w) + (1-\alpha) \cdot f_2(w)")
        st.caption(f"Generated {len(portfolios)} portfolios")

    elif method == "Epsilon-Constraint Method":
        portfolios = generate_epsilon_portfolios(num_portfolios_display, mu, Sigma, sectors)
        st.subheader("Method: Epsilon-Constraint")
        st.latex(r"\min_w f_1(w) \text{ s.t. } f_2(w) \leq \epsilon")
        st.caption(f"Generated {len(portfolios)} portfolios")

    elif method == "NSGA-2 (Evolutionary)":
        portfolios = generate_nsga2_portfolios(num_portfolios_display, mu, Sigma, sectors)
        st.subheader("Method: NSGA-2 (Non-dominated Sorting Genetic Algorithm)")
        st.markdown("""
        **Evolutionary algorithm** that uses:
        - Non-dominated sorting to rank solutions
        - Crowding distance to maintain diversity
        - Can find non-convex Pareto fronts
        """)
        st.caption(f"Generated {len(portfolios)} Pareto-optimal portfolios")

    else:  # Compare All Methods
        st.subheader("Method Comparison: All Three Algorithms")

        # Generate portfolios for comparison
        portfolios_w = generate_weighted_sum_portfolios(num_portfolios_display, mu, Sigma, sectors)
        portfolios_e = generate_epsilon_portfolios(num_portfolios_display, mu, Sigma, sectors)
        portfolios_n = generate_nsga2_portfolios(num_portfolios_display, mu, Sigma, sectors)

        fig = go.Figure()

        # Weighted sum
        risks_w = [p['f2_volatility'] * 100 for p in portfolios_w]
        returns_w = [p['f1_return'] * 100 for p in portfolios_w]
        hover_w = [f"<b>Weighted Sum</b><br>Risk: {r:.2f}%<br>Return: {ret:.2f}%"
                   for r, ret in zip(risks_w, returns_w)]

        fig.add_trace(go.Scatter(
            x=risks_w,
            y=returns_w,
            mode='markers',
            name='Weighted Sum',
            marker=dict(size=10, color='blue', line=dict(width=1, color='white')),
            hovertext=hover_w,
            hoverinfo='text'
        ))

        # Epsilon-constraint
        risks_e = [p['f2_volatility'] * 100 for p in portfolios_e]
        returns_e = [p['f1_return'] * 100 for p in portfolios_e]
        hover_e = [f"<b>Epsilon-Constraint</b><br>Risk: {r:.2f}%<br>Return: {ret:.2f}%"
                   for r, ret in zip(risks_e, returns_e)]

        fig.add_trace(go.Scatter(
            x=risks_e,
            y=returns_e,
            mode='markers',
            name='Epsilon-Constraint',
            marker=dict(size=10, color='red', symbol='x', line=dict(width=1, color='white')),
            hovertext=hover_e,
            hoverinfo='text'
        ))

        # NSGA-2
        risks_n = [p['f2_volatility'] * 100 for p in portfolios_n]
        returns_n = [p['f1_return'] * 100 for p in portfolios_n]
        hover_n = [f"<b>NSGA-2</b><br>Risk: {r:.2f}%<br>Return: {ret:.2f}%"
                   for r, ret in zip(risks_n, returns_n)]

        fig.add_trace(go.Scatter(
            x=risks_n,
            y=returns_n,
            mode='markers',
            name='NSGA-2',
            marker=dict(size=8, color='green', symbol='diamond', line=dict(width=1, color='white')),
            hovertext=hover_n,
            hoverinfo='text'
        ))

        fig.update_layout(
            title='Pareto Front Comparison: Three Optimization Methods',
            xaxis_title='Risk (Volatility) %',
            yaxis_title='Expected Return %',
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig, width='stretch')

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Weighted Sum", f"{len(portfolios_w)} portfolios")
            st.caption("✓ Fast convergence<br>✓ Smooth front<br>⚠ May miss non-convex regions", unsafe_allow_html=True)

        with col2:
            st.metric("Epsilon-Constraint", f"{len(portfolios_e)} portfolios")
            st.caption("✓ Robust<br>✓ Good for non-convex<br>⚠ Slower", unsafe_allow_html=True)

        with col3:
            st.metric("NSGA-2", f"{len(portfolios_n)} portfolios")
            st.caption("✓ Population-based<br>✓ Diverse solutions<br>✓ Handles complexity well", unsafe_allow_html=True)

        st.info("""
        **Key Observations:**
        - **Weighted Sum** and **Epsilon-Constraint** produce similar results (gradient-based methods)
        - **NSGA-2** generates a denser, more diverse set of solutions
        - All three methods converge to the same Pareto front, validating the results
        - NSGA-2 is particularly good at exploring the entire solution space
        """)

        # Don't show portfolio selection for comparison mode
        st.stop()

    # Portfolio selection section - Filter FIRST
    st.subheader("📋 Portfolio Selection & Filtering")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Filter by return constraint
        returns = [p['f1_return'] for p in portfolios]
        min_return = st.slider(
            "Minimum Required Return (%)",
            min_value=float(min(returns)) * 100,
            max_value=float(max(returns)) * 100,
            value=float(min(returns)) * 100,
            step=1.0
        ) / 100

        filtered = [p for p in portfolios if p['f1_return'] >= min_return]
        st.write(f"**{len(filtered)}** portfolios meet this constraint")

    # Plot filtered portfolios
    st.markdown("---")
    fig = create_clickable_scatter(filtered, f'Pareto Front: {method}', show_3d=False)

    # Capture click events
    selected_points = st.plotly_chart(fig, width='stretch', on_select="rerun", key="level1_plot")

    st.markdown("---")

    # Portfolio selector with meaningful labels
    selected = None
    if filtered:
        with col2:
            # Create unique labels with portfolio numbers
            portfolio_labels = [f"#{i+1}: {get_portfolio_label(p)}" for i, p in enumerate(filtered)]

            # Determine default index from clicked point
            default_idx = 0
            if selected_points and len(selected_points.selection.points) > 0:
                clicked_idx = selected_points.selection.points[0]['point_index']
                if clicked_idx < len(filtered):
                    default_idx = clicked_idx
                    clicked_portfolio = filtered[clicked_idx]
                    st.info(f"📍 Selected from graph: {get_portfolio_label(clicked_portfolio)}")

            selected_label = st.selectbox(
                "Choose a portfolio:",
                options=portfolio_labels,
                index=default_idx,
                key="level1_portfolio_select"
            )

            # Extract index from selected label
            selected_idx = portfolio_labels.index(selected_label)
            selected = filtered[selected_idx]

    # Display selected portfolio details
    if selected is not None:
        st.markdown("---")
        st.subheader("📊 Selected Portfolio Details")
        display_portfolio_details(selected, asset_names, portfolio_value)

# === LEVEL 2 ===
else:
    st.header("Level 2: Tri-objective Optimization with Cardinality")
    st.markdown("**Objectives**: Maximize Return, Minimize Risk, Minimize Transaction Costs")
    st.markdown("**Constraints**: C_Base ∩ C_Op (budget + no short + cardinality K)")

    K_choice = st.sidebar.radio(
        "Cardinality (K)",
        [5, 10, 20, 30],
        help="Number of assets in portfolio. Lower K = more concentrated (e.g., top performers like NVDA)"
    )

    portfolios = generate_level2_portfolios(K_choice, num_portfolios_display, mu, Sigma, sectors)

    st.subheader(f"Tri-objective Pareto Front (K={K_choice} assets)")

    # Add explanation based on K value
    if K_choice <= 10:
        st.info(f"""
        **Concentrated Portfolio (K={K_choice})**:
        - Focus on top performers (highest Sharpe ratios like NVDA, AVGO, AMD)
        - Higher potential returns but also higher risk
        - Less diversification
        - Suitable for aggressive investors
        """)
    else:
        st.info(f"""
        **Diversified Portfolio (K={K_choice})**:
        - Spread across more sectors and assets
        - Lower risk through diversification
        - More stable returns
        - Suitable for balanced/conservative investors
        """)

    st.caption(f"Generated {len(portfolios)} portfolios")

    # Portfolio selection & Filtering - Filter FIRST
    st.subheader("📋 Portfolio Selection & Filtering")

    col1, col2 = st.columns([1, 2])

    with col1:
        returns = [p['f1_return'] for p in portfolios]
        min_return = st.slider(
            "Minimum Required Return (%)",
            min_value=float(min(returns)) * 100,
            max_value=float(max(returns)) * 100,
            value=float(min(returns)) * 100,
            step=1.0,
            key="level2_return_slider"
        ) / 100

        filtered = [p for p in portfolios if p['f1_return'] >= min_return]
        st.write(f"**{len(filtered)}** portfolios meet this constraint")

    # 3D plot with filtered portfolios
    st.markdown("---")
    fig = create_clickable_scatter(filtered, f'3D Pareto Front (K={K_choice})', show_3d=True)
    selected_points = st.plotly_chart(fig, width='stretch', on_select="rerun", key="level2_plot")

    # Portfolio selector with meaningful labels
    selected = None
    if filtered:
        with col2:
            # Create unique labels with portfolio numbers
            portfolio_labels = [f"#{i+1}: {get_portfolio_label(p)}" for i, p in enumerate(filtered)]

            # Determine default index from clicked point
            default_idx = 0
            if selected_points and len(selected_points.selection.points) > 0:
                clicked_idx = selected_points.selection.points[0]['point_index']
                if clicked_idx < len(filtered):
                    default_idx = clicked_idx
                    clicked_portfolio = filtered[clicked_idx]
                    st.info(f"📍 Selected from graph: {get_portfolio_label(clicked_portfolio)}")

            selected_label = st.selectbox(
                "Choose a portfolio:",
                options=portfolio_labels,
                index=default_idx,
                key="level2_portfolio_select"
            )

            # Extract index from selected label
            selected_idx = portfolio_labels.index(selected_label)
            selected = filtered[selected_idx]

    # Display selected portfolio
    if selected is not None:
        st.markdown("---")
        st.subheader("📊 Selected Portfolio Details")
        display_portfolio_details(selected, asset_names, portfolio_value)

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.info("""
Made by:
-   BÉCHU Thomas
-   GUENGANT Noé
-   KERAUTRET Malo
""")
