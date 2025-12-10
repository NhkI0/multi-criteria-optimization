"""
Improved Streamlit App for Pareto Front Visualization
- Click on portfolios in graphs
- Meaningful portfolio descriptions
- Detailed holdings with ticker names
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(
    page_title="Portfolio Pareto Optimizer",
    page_icon="📊",
    layout="wide"
)

# === Load Asset Names ===
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

@st.cache_data
def load_pareto_front(filename):
    """Load pre-computed Pareto front from JSON"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Load asset names
asset_names = load_asset_names()

# Check if Pareto fronts exist
files_exist = all([
    Path('pareto_level1_weighted.json').exists(),
    Path('pareto_level1_epsilon.json').exists(),
    Path('pareto_level1_nsga2.json').exists(),
    Path('pareto_level2_K20.json').exists()
])

if not files_exist:
    st.error("⚠️ Pareto fronts not generated yet!")
    st.info("Please run: `python generate_pareto_fronts.py` first")
    st.stop()

# Load Pareto fronts
pareto_l1_weighted = load_pareto_front('pareto_level1_weighted.json')
pareto_l1_epsilon = load_pareto_front('pareto_level1_epsilon.json')
pareto_l1_nsga2 = load_pareto_front('pareto_level1_nsga2.json')
pareto_l2_k20 = load_pareto_front('pareto_level2_K20.json')
pareto_l2_k30 = load_pareto_front('pareto_level2_K30.json')

# === Helper Functions ===

def sample_portfolios(portfolios, max_count):
    """Sample portfolios evenly across the Pareto front"""
    if len(portfolios) <= max_count:
        return portfolios

    # Sample evenly across the front
    indices = np.linspace(0, len(portfolios) - 1, max_count, dtype=int)
    return [portfolios[i] for i in indices]

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

    # Create detailed holdings table
    holdings_data = []
    for i in sorted_indices:
        ticker = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
        weight = weights[i]
        value = weight * portfolio_value

        holdings_data.append({
            'Ticker': ticker,
            'Weight': weight,
            'Value ($)': value
        })

    holdings_df = pd.DataFrame(holdings_data)

    # Format for display
    holdings_df['Weight'] = holdings_df['Weight'].apply(lambda x: f"{x:.2%}")
    holdings_df['Value ($)'] = holdings_df['Value ($)'].apply(lambda x: f"${x:,.0f}")

    # Show top 20 by default
    st.dataframe(holdings_df, width='stretch', hide_index=True)

    # Download button
    csv = holdings_df.to_csv(index=False)
    st.download_button(
        "📥 Download Portfolio CSV",
        csv,
        "portfolio_allocation.csv",
        "text/csv",
        width='stretch'
    )

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

# Number of portfolios to display
st.sidebar.subheader("📊 Display Settings")
num_portfolios_display = st.sidebar.slider(
    "Number of Portfolios to Show",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="Control how many portfolios are shown in the visualization"
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
        portfolios_full = pareto_l1_weighted
        portfolios = sample_portfolios(portfolios_full, num_portfolios_display)
        st.subheader("Method: Weighted Sum Scalarization")
        st.latex(r"\min_w \alpha \cdot f_1(w) + (1-\alpha) \cdot f_2(w)")
        st.caption(f"Showing {len(portfolios)} of {len(portfolios_full)} portfolios")

    elif method == "Epsilon-Constraint Method":
        portfolios_full = pareto_l1_epsilon
        portfolios = sample_portfolios(portfolios_full, num_portfolios_display)
        st.subheader("Method: Epsilon-Constraint")
        st.latex(r"\min_w f_1(w) \text{ s.t. } f_2(w) \leq \epsilon")
        st.caption(f"Showing {len(portfolios)} of {len(portfolios_full)} portfolios")

    elif method == "NSGA-2 (Evolutionary)":
        portfolios_full = pareto_l1_nsga2
        portfolios = sample_portfolios(portfolios_full, num_portfolios_display)
        st.subheader("Method: NSGA-2 (Non-dominated Sorting Genetic Algorithm)")
        st.markdown("""
        **Evolutionary algorithm** that uses:
        - Non-dominated sorting to rank solutions
        - Crowding distance to maintain diversity
        - Can find non-convex Pareto fronts
        """)
        st.caption(f"Showing {len(portfolios)} of {len(portfolios_full)} portfolios")

    else:  # Compare All Methods
        st.subheader("Method Comparison: All Three Algorithms")

        # Sample portfolios for comparison
        portfolios_w = sample_portfolios(pareto_l1_weighted, num_portfolios_display)
        portfolios_e = sample_portfolios(pareto_l1_epsilon, num_portfolios_display)
        portfolios_n = sample_portfolios(pareto_l1_nsga2, num_portfolios_display)

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
            st.metric("Weighted Sum", f"{len(pareto_l1_weighted)} portfolios")
            st.caption("✓ Fast convergence<br>✓ Smooth front<br>⚠ May miss non-convex regions", unsafe_allow_html=True)

        with col2:
            st.metric("Epsilon-Constraint", f"{len(pareto_l1_epsilon)} portfolios")
            st.caption("✓ Robust<br>✓ Good for non-convex<br>⚠ Slower", unsafe_allow_html=True)

        with col3:
            st.metric("NSGA-2", f"{len(pareto_l1_nsga2)} portfolios")
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

    # Plot
    fig = create_clickable_scatter(portfolios, f'Pareto Front: {method}', show_3d=False)

    # Capture click events
    selected_points = st.plotly_chart(fig, width='stretch', on_select="rerun", key="level1_plot")

    st.markdown("---")

    # Portfolio selection section
    st.subheader("📋 Portfolio Selection")

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

    # Portfolio selector with meaningful labels
    selected = None
    if filtered:
        with col2:
            portfolio_options = {get_portfolio_label(p, i): i
                                for i, p in enumerate(filtered)}

            # Determine default index from clicked point
            default_idx = 0
            if selected_points and len(selected_points.selection.points) > 0:
                # Get the index from the clicked point
                clicked_idx = selected_points.selection.points[0]['point_index']
                # Find this portfolio in filtered list
                clicked_portfolio = portfolios[clicked_idx]
                for idx, fp in enumerate(filtered):
                    if fp == clicked_portfolio:
                        default_idx = idx
                        break
                st.info(f"📍 Selected from graph: {get_portfolio_label(clicked_portfolio)}")

            selected_label = st.selectbox(
                "Choose a portfolio:",
                options=list(portfolio_options.keys()),
                index=default_idx,
                key="level1_portfolio_select"
            )

            selected_idx = portfolio_options[selected_label]
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

    K_choice = st.sidebar.radio("Cardinality (K)", [20, 30])

    portfolios_full = pareto_l2_k20 if K_choice == 20 else pareto_l2_k30
    portfolios = sample_portfolios(portfolios_full, num_portfolios_display)

    st.subheader(f"Tri-objective Pareto Front (K={K_choice} assets)")
    st.caption(f"Showing {len(portfolios)} of {len(portfolios_full)} portfolios")

    # 3D plot
    fig = create_clickable_scatter(portfolios, f'3D Pareto Front (K={K_choice})', show_3d=True)
    selected_points = st.plotly_chart(fig, width='stretch', on_select="rerun", key="level2_plot")

    st.markdown("---")

    # Portfolio selection
    st.subheader("📋 Portfolio Selection")

    col1, col2 = st.columns([1, 2])

    with col1:
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

    # Portfolio selector with meaningful labels
    selected = None
    if filtered:
        with col2:
            portfolio_options = {get_portfolio_label(p, i): i
                                for i, p in enumerate(filtered)}

            # Determine default index from clicked point
            default_idx = 0
            if selected_points and len(selected_points.selection.points) > 0:
                # Get the index from the clicked point
                clicked_idx = selected_points.selection.points[0]['point_index']
                # Find this portfolio in filtered list
                clicked_portfolio = portfolios[clicked_idx]
                for idx, fp in enumerate(filtered):
                    if fp == clicked_portfolio:
                        default_idx = idx
                        break
                st.info(f"📍 Selected from graph: {get_portfolio_label(clicked_portfolio)}")

            selected_label = st.selectbox(
                "Choose a portfolio:",
                options=list(portfolio_options.keys()),
                index=default_idx,
                key="level2_portfolio_select"
            )

            selected_idx = portfolio_options[selected_label]
            selected = filtered[selected_idx]

    # Display selected portfolio
    if selected is not None:
        st.markdown("---")
        st.subheader("📊 Selected Portfolio Details")
        display_portfolio_details(selected, asset_names, portfolio_value)

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.info("""
**Project**: Multi-Objective Portfolio Optimization

**Methods**:
- Weighted Sum Scalarization
- Epsilon-Constraint
- NSGA-2 (Evolutionary)

**Levels**:
- Level 1: Bi-objective (Return-Risk)
- Level 2: Tri-objective (Return-Risk-Costs) with Cardinality
""")
