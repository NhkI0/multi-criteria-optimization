# Portfolio Multi-Objective Optimization - Simplified Implementation

**Correct implementation following project requirements**

## Overview

This project implements multi-objective portfolio optimization with **Pareto front generation** as required by the assignment.

### Key Requirements Implemented:

✅ **Level 1**: Bi-objective optimization (f1: return, f2: risk)
✅ **Level 2**: Tri-objective optimization (f1, f2, f3: transaction costs) with cardinality
✅ **Two comparison methods**: Weighted Sum Scalarization + Epsilon-Constraint
✅ **Pareto fronts**: Pre-computed sets of optimal portfolios
✅ **Streamlit app**: Visualize Pareto fronts and select portfolios with r_min constraint
✅ **Sector analysis**: Portfolio breakdown by industry

## Mathematical Formulation

### Objective Functions

```
f1(w) = -w^T μ        (maximize return → minimize -return)
f2(w) = w^T Σ w       (minimize variance)
f3(w) = Σ c_prop |w_i - w_{t,i}|  (minimize transaction costs)
```

### Constraints

**C_Base** (Level 1):
- Σw_i = 1 (budget constraint)
- w_i ≥ 0 (no short-selling)

**C_Op** (Level 2):
- Σ I(w_i > δ_tol) = K (cardinality: exactly K assets)

## Project Structure

```
portfolio-pareto-simple/
├── data_loader.py              # Load stock data
├── pareto_optimizer.py         # Main optimizer (weighted sum)
├── comparison_methods.py       # Epsilon-constraint method
├── generate_pareto_fronts.py   # Generate all Pareto fronts
├── app_simple.py               # Streamlit visualization app
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Pareto Fronts

**IMPORTANT**: Run this first to generate pre-computed Pareto fronts:

```bash
python generate_pareto_fronts.py
```

This will create:
- `pareto_level1_weighted.json` (50 portfolios, weighted sum)
- `pareto_level1_epsilon.json` (30 portfolios, epsilon-constraint)
- `pareto_level2_K20.json` (40 portfolios, K=20 assets)
- `pareto_level2_K30.json` (40 portfolios, K=30 assets)

**Expected output:**
```
[1/5] Loading data...
Loaded 192 assets
Period: 2015-01-02 to 2025-10-31

[2/5] Level 1 - Weighted Sum Scalarization...
✓ Generated 50 Pareto-optimal portfolios
  Return range: 12.45% to 32.44%
  Risk range: 12.91% to 19.35%
  Time: 15.2s

[3/5] Level 1 - Epsilon-Constraint Method...
✓ Generated 30 solutions
  Time: 12.8s

[4/5] Level 2 - Tri-objective with K=20...
✓ Generated 40 Pareto-optimal portfolios
  Return range: 25.12% to 41.54%
  Risk range: 18.23% to 26.97%
  Cost range: 0.0012 to 0.0045
  Time: 8.3s

[5/5] Level 2 - Tri-objective with K=30...
✓ Generated 40 Pareto-optimal portfolios
  Time: 10.1s

✓ ALL PARETO FRONTS GENERATED SUCCESSFULLY
```

### Step 2: Launch Streamlit App

```bash
streamlit run app_simple.py
```

The app will open at `http://localhost:8501`

## Features

### Level 1: Bi-objective Optimization

**Visualization**:
- 2D scatter plot (Risk vs Return)
- Pareto front curve
- Color-coded by return

**Methods**:
1. **Weighted Sum Scalarization**:
   - min α·f1(w) + (1-α)·f2(w)
   - Vary α from 0 to 1

2. **Epsilon-Constraint**:
   - min f1(w) subject to f2(w) ≤ ε
   - Vary ε (risk constraint)

3. **Comparison**: Overlay both methods to compare quality

**Portfolio Selection**:
- Slider to set minimum return constraint (r_min)
- Filter portfolios: return ≥ r_min
- View top 20 holdings for selected portfolio

### Level 2: Tri-objective with Cardinality

**Visualization**:
- Interactive 3D scatter plot
- Axes: Risk, Return, Transaction Costs
- Color-coded by return

**Configuration**:
- Choose K=20 or K=30 (number of assets)
- Set minimum return constraint
- Select specific portfolio from Pareto front

**Portfolio Details**:
- Expected return, risk, transaction cost
- Number of assets (should be ≈K)
- Top holdings with weights

## Methodology

### Pareto Front Generation

A **Pareto front** is a set of solutions where:
- No objective can be improved without worsening another
- Represents optimal trade-offs between objectives

### Weighted Sum Scalarization

```
min α·f1(w) + β·f2(w) + γ·f3(w)
subject to: α + β + γ = 1, α,β,γ ≥ 0
```

**Process**:
1. Systematically vary weights (α, β, γ)
2. For each weight combination, solve single-objective problem
3. Collect all solutions → Pareto front

**Advantages**:
- Simple to implement
- Fast convergence
- Works well for convex problems

**Limitations**:
- May miss non-convex regions of Pareto front
- Uniform weight distribution doesn't guarantee uniform Pareto coverage

### Epsilon-Constraint Method

```
min f1(w)
subject to: f2(w) ≤ ε2, f3(w) ≤ ε3
```

**Process**:
1. Vary constraint bounds (ε2, ε3)
2. Optimize primary objective with constraints
3. Collect solutions → Pareto front

**Advantages**:
- Can find non-convex Pareto regions
- More robust for non-convex problems

**Limitations**:
- Slower (each point requires constrained optimization)
- Need to know appropriate ε ranges

### Cardinality Constraint Implementation

Since Σ I(w_i > δ_tol) = K is a **mixed-integer constraint** (NP-hard), we use a **greedy heuristic**:

1. **Select K best assets**: Use individual Sharpe ratios
2. **Optimize on subset**: Solve continuous problem on K assets only
3. **Reconstruct**: Fill full weight vector (others = 0)

**Trade-off**:
- Not globally optimal (may miss better K-combinations)
- Very fast (O(K³) instead of O(N³))
- Good solutions in practice

## Results Interpretation

### Pareto Front Analysis

**Shape**:
- Convex curve → classic risk-return trade-off
- Each point is **Pareto-optimal** (can't improve both simultaneously)
- Endpoints:
  - Left: Minimum risk (conservative)
  - Right: Maximum return (aggressive)

**Selection**:
- Risk-averse investor → left side of front
- Risk-tolerant investor → right side
- Middle ground → Max Sharpe ratio (best risk-adjusted return)

### Level 1 vs Level 2

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Objectives | 2 (return, risk) | 3 (return, risk, costs) |
| Constraint | C_Base | C_Base ∩ C_Op (cardinality) |
| Visualization | 2D curve | 3D surface |
| Portfolios | ~50 assets | Exactly K assets |
| Practical | Theoretical | Realistic |

**Key Insight**: Level 2 is more realistic because:
- Limits number of assets (easier management)
- Considers transaction costs (real-world friction)
- Still achieves good returns with fewer assets

## Comparison: This Implementation vs Original

| Feature | Original | This (Simplified) |
|---------|----------|-------------------|
| **Core Approach** | Single optimal solutions | **Pareto fronts** ✅ |
| **Objectives** | Max Sharpe, Min Var only | **f1, f2, f3** ✅ |
| **Methods** | SLSQP only | **Scalarization + ε-constraint** ✅ |
| **Transaction Costs** | ❌ Missing | **✅ Implemented** |
| **Streamlit** | On-demand optimization | **Pre-computed fronts** ✅ |
| **Requirements** | ❌ Didn't match assignment | **✅ Matches PDF** |

## Limitations

1. **Data assumptions**: Historical returns ≠ future returns
2. **Normal distribution**: Real returns have fat tails
3. **Static optimization**: Doesn't account for market changes
4. **Cardinality heuristic**: Not guaranteed optimal
5. **Transaction costs**: Linear model (reality may be non-linear)

## Future Improvements

1. **NSGA-II**: Implement genetic algorithm for comparison
2. **Robustness (Level 3)**: Add resampling procedure
3. **Better heuristics**: Clustering-based asset selection
4. **Dynamic optimization**: Rebalancing over time
5. **Risk measures**: VaR, CVaR instead of variance

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
- Deb, K. et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
- Chankong, V. & Haimes, Y. (1983). Multiobjective Decision Making.

---

**Project**: M1 Data - Mathématiques
**Due**: 13 December 2025, 23:59
**Deliverables**: Report (5-8 pages) + Streamlit App + GitHub repository
