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
    data_dir = Path("../data")
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
    Path('pareto_level2_K20.json').exists()
])

if not files_exist:
    st.error("⚠️ Pareto fronts not generated yet!")
    st.info("Please run: `python generate_pareto_fronts.py` first")
    st.stop()

# Load Pareto fronts
pareto_l1_weighted = load_pareto_front('pareto_level1_weighted.json')
pareto_l1_epsilon = load_pareto_front('pareto_level1_epsilon.json')
pareto_l2_k20 = load_pareto_front('pareto_level2_K20.json')
pareto_l2_k30 = load_pareto_front('pareto_level2_K30.json')

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

def display_portfolio_details(portfolio, asset_names=None):
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

    st.subheader(f"Portfolio Holdings ({len(sorted_indices)} assets)")

    # Create detailed holdings table
    holdings_data = []
    for i in sorted_indices:
        ticker = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
        weight = weights[i]
        value = weight * 1_000_000  # Assuming $1M portfolio

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
    st.dataframe(holdings_df.head(20), use_container_width=True, hide_index=True)

    if len(holdings_df) > 20:
        with st.expander(f"Show all {len(holdings_df)} holdings"):
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    # Download button
    csv = holdings_df.to_csv(index=False)
    st.download_button(
        "📥 Download Portfolio CSV",
        csv,
        "portfolio_allocation.csv",
        "text/csv",
        use_container_width=True
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
        ["Weighted Sum Scalarization", "Epsilon-Constraint Method", "Compare Both"]
    )

    if method == "Weighted Sum Scalarization":
        portfolios = pareto_l1_weighted
        st.subheader("Method: Weighted Sum Scalarization")
        st.latex(r"\min_w \alpha \cdot f_1(w) + (1-\alpha) \cdot f_2(w)")

    elif method == "Epsilon-Constraint Method":
        portfolios = pareto_l1_epsilon
        st.subheader("Method: Epsilon-Constraint")
        st.latex(r"\min_w f_1(w) \text{ s.t. } f_2(w) \leq \epsilon")

    else:  # Compare Both
        st.subheader("Method Comparison")

        fig = go.Figure()

        # Weighted sum
        risks_w = [p['f2_volatility'] * 100 for p in pareto_l1_weighted]
        returns_w = [p['f1_return'] * 100 for p in pareto_l1_weighted]
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
        risks_e = [p['f2_volatility'] * 100 for p in pareto_l1_epsilon]
        returns_e = [p['f1_return'] * 100 for p in pareto_l1_epsilon]
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

        fig.update_layout(
            title='Method Comparison',
            xaxis_title='Risk (Volatility) %',
            yaxis_title='Expected Return %',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Observations:**
        - Both methods generate similar Pareto fronts
        - Weighted sum: Faster, smoother curve
        - Epsilon-constraint: More robust for non-convex problems
        """)

        # Don't show portfolio selection for comparison mode
        st.stop()

    # Plot
    fig = create_clickable_scatter(portfolios, f'Pareto Front: {method}', show_3d=False)

    # Capture click events
    selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="level1_plot")

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
        display_portfolio_details(selected, asset_names)

# === LEVEL 2 ===
else:
    st.header("Level 2: Tri-objective Optimization with Cardinality")
    st.markdown("**Objectives**: Maximize Return, Minimize Risk, Minimize Transaction Costs")
    st.markdown("**Constraints**: C_Base ∩ C_Op (budget + no short + cardinality K)")

    K_choice = st.sidebar.radio("Cardinality (K)", [20, 30])

    portfolios = pareto_l2_k20 if K_choice == 20 else pareto_l2_k30

    st.subheader(f"Tri-objective Pareto Front (K={K_choice} assets)")

    # 3D plot
    fig = create_clickable_scatter(portfolios, f'3D Pareto Front (K={K_choice})', show_3d=True)
    selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="level2_plot")

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
        display_portfolio_details(selected, asset_names)

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.info("""
**Project**: Multi-Objective Portfolio Optimization

**Methods**:
- Weighted Sum Scalarization
- Epsilon-Constraint

**Levels**:
- Level 1: Bi-objective (Return-Risk)
- Level 2: Tri-objective (Return-Risk-Costs) with Cardinality
""")
