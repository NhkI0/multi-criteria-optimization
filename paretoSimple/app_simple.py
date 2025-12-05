"""
Simple Streamlit App for Pareto Front Visualization
Loads pre-computed Pareto fronts and allows portfolio selection
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

st.title("📊 Portfolio Multi-Objective Optimization")
st.markdown("**Pareto Front Visualization and Portfolio Selection**")

# === Load Pareto Fronts ===

@st.cache_data
def load_pareto_front(filename):
    """Load pre-computed Pareto front from JSON"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Check if files exist
files_exist = all([
    Path('pareto_level1_weighted.json').exists(),
    Path('pareto_level1_epsilon.json').exists(),
    Path('pareto_level2_K20.json').exists()
])

if not files_exist:
    st.error("⚠️ Pareto fronts not generated yet!")
    st.info("Please run: `python generate_pareto_fronts.py` first")
    st.stop()

# Load data
pareto_l1_weighted = load_pareto_front('pareto_level1_weighted.json')
pareto_l1_epsilon = load_pareto_front('pareto_level1_epsilon.json')
pareto_l2_k20 = load_pareto_front('pareto_level2_K20.json')
pareto_l2_k30 = load_pareto_front('pareto_level2_K30.json')

# === Sidebar ===
st.sidebar.header("⚙️ Configuration")

level = st.sidebar.radio(
    "Select Optimization Level",
    ["Level 1: Bi-objective (Return-Risk)",
     "Level 2: Tri-objective (Return-Risk-Costs)"]
)

# === LEVEL 1 ===
if "Level 1" in level:
    st.header("Level 1: Bi-objective Optimization")
    st.markdown("**Objectives**: Maximize Return, Minimize Risk")
    st.markdown("**Constraints**: C_Base (budget + no short-selling)")

    # Method comparison
    method = st.sidebar.radio(
        "Optimization Method",
        ["Weighted Sum Scalarization", "Epsilon-Constraint Method", "Compare Both"]
    )

    if method == "Weighted Sum Scalarization":
        st.subheader("Method: Weighted Sum Scalarization")
        st.latex(r"\min_w \alpha \cdot f_1(w) + (1-\alpha) \cdot f_2(w)")

        # Plot Pareto front
        fig = go.Figure()

        risks = [sol['f2_volatility'] for sol in pareto_l1_weighted]
        returns = [sol['f1_return'] for sol in pareto_l1_weighted]

        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+lines',
            name='Pareto Front',
            marker=dict(size=8, color=returns, colorscale='Viridis', showscale=True,
                       colorbar=dict(title="Return")),
            hovertemplate='<b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        ))

        fig.update_layout(
            title='Pareto Front: Risk vs Return',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            hovermode='closest',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Portfolio selection
        st.subheader("Portfolio Selection")

        min_return = st.slider(
            "Minimum Required Return (r_min)",
            min_value=0.0,
            max_value=max(returns),
            value=min(returns),
            step=0.01,
            format="%.2f"
        )

        # Filter portfolios
        filtered = [sol for sol in pareto_l1_weighted if sol['f1_return'] >= min_return]

        st.write(f"**Portfolios meeting constraint**: {len(filtered)}/{len(pareto_l1_weighted)}")

        if filtered:
            # Show table
            df = pd.DataFrame([{
                'Return': f"{sol['f1_return']:.2%}",
                'Risk': f"{sol['f2_volatility']:.2%}",
                'N Assets': sol['n_assets']
            } for sol in filtered])

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Select specific portfolio
            idx = st.selectbox("Select portfolio to view details", range(len(filtered)))

            selected = filtered[idx]

            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{selected['f1_return']:.2%}")
            col2.metric("Risk (Volatility)", f"{selected['f2_volatility']:.2%}")
            col3.metric("Number of Assets", selected['n_assets'])

            # Show weights
            weights = np.array(selected['weights'])
            top_indices = np.argsort(weights)[::-1][:20]

            st.write("**Top 20 Holdings:**")
            holdings_data = []
            for i in top_indices:
                if weights[i] > 0.001:
                    holdings_data.append({
                        'Asset Index': i,
                        'Weight': f"{weights[i]:.2%}"
                    })

            if holdings_data:
                st.dataframe(pd.DataFrame(holdings_data), hide_index=True)

    elif method == "Epsilon-Constraint Method":
        st.subheader("Method: Epsilon-Constraint")
        st.latex(r"\min_w f_1(w) \text{ s.t. } f_2(w) \leq \epsilon")

        # Plot
        fig = go.Figure()

        risks = [sol['f2_volatility'] for sol in pareto_l1_epsilon]
        returns = [sol['f1_return'] for sol in pareto_l1_epsilon]

        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+lines',
            name='Pareto Front (ε-constraint)',
            marker=dict(size=8, color='red'),
            hovertemplate='<b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        ))

        fig.update_layout(
            title='Pareto Front: Risk vs Return (Epsilon-Constraint)',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    else:  # Compare Both
        st.subheader("Method Comparison")

        fig = go.Figure()

        # Weighted sum
        risks_w = [sol['f2_volatility'] for sol in pareto_l1_weighted]
        returns_w = [sol['f1_return'] for sol in pareto_l1_weighted]

        fig.add_trace(go.Scatter(
            x=risks_w,
            y=returns_w,
            mode='markers',
            name='Weighted Sum',
            marker=dict(size=8, color='blue'),
            hovertemplate='<b>Method:</b> Weighted Sum<br><b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        ))

        # Epsilon-constraint
        risks_e = [sol['f2_volatility'] for sol in pareto_l1_epsilon]
        returns_e = [sol['f1_return'] for sol in pareto_l1_epsilon]

        fig.add_trace(go.Scatter(
            x=risks_e,
            y=returns_e,
            mode='markers',
            name='Epsilon-Constraint',
            marker=dict(size=8, color='red', symbol='x'),
            hovertemplate='<b>Method:</b> Epsilon-Constraint<br><b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<extra></extra>'
        ))

        fig.update_layout(
            title='Method Comparison: Weighted Sum vs Epsilon-Constraint',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Observations:**
        - Both methods generate similar Pareto fronts
        - Weighted sum: Faster, but may miss non-convex regions
        - Epsilon-constraint: More robust, slower
        """)

# === LEVEL 2 ===
else:
    st.header("Level 2: Tri-objective Optimization with Cardinality")
    st.markdown("**Objectives**: Maximize Return, Minimize Risk, Minimize Transaction Costs")
    st.markdown("**Constraints**: C_Base ∩ C_Op (budget + no short + cardinality K)")

    K_choice = st.sidebar.radio("Cardinality (K)", [20, 30])

    pareto_l2 = pareto_l2_k20 if K_choice == 20 else pareto_l2_k30

    st.subheader(f"Tri-objective Pareto Front (K={K_choice} assets)")

    # 3D scatter plot
    fig = go.Figure()

    returns = [sol['f1_return'] for sol in pareto_l2]
    risks = [sol['f2_volatility'] for sol in pareto_l2]
    costs = [sol['f3_transaction_cost'] for sol in pareto_l2]

    fig.add_trace(go.Scatter3d(
        x=risks,
        y=returns,
        z=costs,
        mode='markers',
        marker=dict(
            size=6,
            color=returns,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Return")
        ),
        hovertemplate='<b>Risk:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<br><b>Cost:</b> %{z:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'3D Pareto Front (K={K_choice})',
        scene=dict(
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            zaxis_title='Transaction Cost'
        ),
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    # Portfolio selection with r_min constraint
    st.subheader("Portfolio Selection")

    min_return = st.slider(
        "Minimum Required Return (r_min)",
        min_value=0.0,
        max_value=max(returns),
        value=min(returns),
        step=0.01,
        format="%.2f"
    )

    # Filter
    filtered = [sol for sol in pareto_l2 if sol['f1_return'] >= min_return]

    st.write(f"**Portfolios meeting constraint**: {len(filtered)}/{len(pareto_l2)}")

    if filtered:
        # Show table
        df = pd.DataFrame([{
            'Return': f"{sol['f1_return']:.2%}",
            'Risk': f"{sol['f2_volatility']:.2%}",
            'Cost': f"{sol['f3_transaction_cost']:.4f}",
            'N Assets': sol['n_assets']
        } for sol in filtered])

        st.dataframe(df, use_container_width=True, hide_index=True)

# === Footer ===
st.sidebar.markdown("---")
st.sidebar.info("""
**Project**: Multi-Objective Portfolio Optimization
**Methods**: Scalarization, Epsilon-Constraint
**Levels**: 1 (Bi-obj) + 2 (Tri-obj with cardinality)
""")
