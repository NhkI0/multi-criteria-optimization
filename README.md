# Portfolio Multi-Objective Optimization - Simplified Implementation

**Correct implementation following project requirements**

## Overview

This project implements multi-objective portfolio optimization with **Pareto front generation** as required by the assignment.

### Key Requirements Implemented:

✅ **Level 1**: Bi-objective optimization (f1: return, f2: risk)
✅ **Level 2**: Tri-objective optimization (f1, f2, f3: transaction costs) with cardinality
✅ **Three comparison methods**: Weighted Sum + Epsilon-Constraint + NSGA-2 (Evolutionary)
✅ **Pareto fronts**: Pre-computed sets of optimal portfolios
✅ **Interactive Streamlit app**: Click-to-select portfolios, configurable investment amount
✅ **Sector analysis**: Portfolio breakdown by industry
✅ **Comprehensive documentation**: Detailed implementation guide (see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md))

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
paretoSimple/
├── data_loader.py              # Load stock data, calculate returns & covariance
├── pareto_optimizer.py         # Weighted sum scalarization optimizer
├── comparison_methods.py       # Epsilon-constraint + NSGA-2 methods
├── generate_pareto_fronts.py   # Generate all Pareto fronts (run first!)
├── app_simple.py               # Interactive Streamlit visualization app
├── IMPLEMENTATION_GUIDE.md     # Detailed technical documentation (NEW!)
├── README.md                   # This quick-start guide
└── *.json                      # Pre-computed Pareto fronts (generated)
```

## Installation

```bash
pip install numpy pandas scipy streamlit plotly deap
```

**Required packages**:
- `numpy`, `pandas`: Data manipulation
- `scipy`: Optimization (SLSQP)
- `streamlit`, `plotly`: Interactive visualization
- `deap`: Evolutionary algorithms (NSGA-2)

## Usage

### Step 1: Generate Pareto Fronts

**IMPORTANT**: Run this first to generate pre-computed Pareto fronts:

```bash
python generate_pareto_fronts.py
```

This will create:
- `pareto_level1_weighted.json` (50 portfolios, weighted sum)
- `pareto_level1_epsilon.json` (30 portfolios, epsilon-constraint)
- `pareto_level1_nsga2.json` (~100 portfolios, NSGA-2 evolutionary)
- `pareto_level2_K20.json` (40 portfolios, K=20 assets)
- `pareto_level2_K30.json` (40 portfolios, K=30 assets)

**Expected output:**
```
======================================================================
PORTFOLIO OPTIMIZATION - PARETO FRONT GENERATION
======================================================================

[1/6] Loading data...
Loaded 192 assets
Period: 2015-01-02 00:00:00 to 2025-10-31 00:00:00

[2/6] Level 1 - Weighted Sum Scalarization...
[OK] Generated 50 Pareto-optimal portfolios
  Return range: 12.41% to 70.71%
  Risk range: 12.91% to 49.58%
  Time: 21.5s

[3/6] Level 1 - Epsilon-Constraint Method...
[OK] Generated 30 solutions
  Return range: 12.68% to 70.71%
  Risk range: 12.91% to 49.58%
  Time: 15.2s

[4/6] Level 1 - NSGA-2 Method...
[OK] Generated 100 Pareto-optimal solutions
  Return range: 15.02% to 57.05%
  Risk range: 14.04% to 38.24%
  Time: 5.8s

[5/6] Level 2 - Tri-objective with K=20...
[OK] Generated 40 Pareto-optimal portfolios
  Return range: 23.49% to 65.94%
  Risk range: 17.12% to 47.93%
  Cost range: 0.0018 to 0.0019
  Time: 1.2s

[6/6] Level 2 - Tri-objective with K=30...
[OK] Generated 40 Pareto-optimal portfolios
  Return range: 22.71% to 65.59%
  Risk range: 16.27% to 47.53%
  Cost range: 0.0017 to 0.0019
  Time: 3.6s

======================================================================
[OK] ALL PARETO FRONTS GENERATED SUCCESSFULLY
======================================================================

Generated files:
  - pareto_level1_weighted.json (50 portfolios)
  - pareto_level1_epsilon.json (30 portfolios)
  - pareto_level1_nsga2.json (~100 portfolios)
  - pareto_level2_K20.json (40 portfolios)
  - pareto_level2_K30.json (40 portfolios)

Now run: streamlit run app_simple.py
```

**Total runtime**: ~47 seconds

### Step 2: Launch Streamlit App

```bash
streamlit run app_simple.py
```

The app will open at `http://localhost:8501`

## Features

### Configuration Options (Sidebar)

**💰 Portfolio Settings**:
- Set investment amount ($1,000 - $100,000,000)
- Default: $1,000,000
- All portfolio values update dynamically

**📊 Display Settings**:
- Number of portfolios to show (10-100)
- Controls visualization density
- Faster performance with fewer portfolios

### Level 1: Bi-objective Optimization

**Visualization**:
- Interactive 2D scatter plot (Risk vs Return)
- Click on any point to select that portfolio
- Hover for portfolio details
- Color-coded by return level

**Methods**:
1. **Weighted Sum Scalarization** (50 portfolios):
   - Formula: min α·f1(w) + (1-α)·f2(w)
   - Fast, smooth Pareto front
   - ~21 seconds to generate

2. **Epsilon-Constraint Method** (30 portfolios):
   - Formula: min f1(w) subject to f2(w) ≤ ε
   - Robust for non-convex problems
   - ~15 seconds to generate

3. **NSGA-2 Evolutionary** (~100 portfolios):
   - Population-based genetic algorithm
   - Dense, diverse solution set
   - ~6 seconds to generate
   - Green diamonds on comparison chart

4. **Compare All Methods**: Overlay all three to validate results

**Portfolio Selection**:
- Click portfolio on graph OR choose from dropdown
- Dropdown shows meaningful labels (e.g., "Conservative: 12.4% return, 12.9% risk")
- Filter by minimum return constraint
- View all holdings with actual ticker names

### Level 2: Tri-objective with Cardinality

**Visualization**:
- Interactive 3D scatter plot
- Axes: Risk (%), Return (%), Transaction Costs
- Rotate, zoom, pan to explore
- Click points to select portfolios
- Color-coded by return level

**Configuration**:
- Choose K=20 or K=30 (cardinality: number of assets)
- Set minimum return constraint
- Select portfolio by clicking OR dropdown
- Shows "Showing X of Y portfolios" based on display settings

**Portfolio Details**:
- Expected return, risk (volatility), transaction cost
- Sharpe ratio calculation
- Number of assets (exactly K)
- Full holdings table with dollar values
- Download CSV button

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

## Implementation Summary

| Feature | Status | Details |
|---------|--------|---------|
| **Core Approach** | ✅ Complete | Pareto front generation (not single solutions) |
| **Objectives** | ✅ Complete | f1 (return), f2 (risk), f3 (transaction costs) |
| **Optimization Methods** | ✅ Three methods | Weighted Sum + Epsilon-Constraint + NSGA-2 |
| **Transaction Costs** | ✅ Implemented | Level 2 with c_prop = 0.001 |
| **Cardinality Constraint** | ✅ Implemented | K ∈ {20, 30} with greedy heuristic |
| **Interactive UI** | ✅ Enhanced | Click-to-select, configurable investment amount |
| **Visualization** | ✅ Advanced | 2D/3D plots, method comparison, portfolio details |
| **Documentation** | ✅ Comprehensive | README + IMPLEMENTATION_GUIDE.md (30+ pages) |

## Limitations

1. **Data assumptions**: Historical returns ≠ future returns
2. **Normal distribution**: Real returns have fat tails
3. **Static optimization**: Doesn't account for market changes
4. **Cardinality heuristic**: Not guaranteed optimal
5. **Transaction costs**: Linear model (reality may be non-linear)

## Future Improvements

1. ~~**NSGA-II**: Implement genetic algorithm for comparison~~ ✅ **DONE**
2. **Robustness (Level 3)**: Add resampling procedure for parameter uncertainty
3. **Better heuristics**: Clustering-based asset selection for cardinality
4. **Dynamic optimization**: Multi-period rebalancing strategy
5. **Risk measures**: VaR (Value at Risk), CVaR (Conditional VaR)
6. **Backtesting**: Out-of-sample performance validation
7. **Real-time data**: Integration with financial APIs (Yahoo Finance, Alpha Vantage)
8. **Advanced constraints**: Sector limits, turnover constraints, ESG scores

## References

### Academic Papers

- **Markowitz, H. (1952)**. "Portfolio Selection". *Journal of Finance*, 7(1), 77-91.
  - Foundation of modern portfolio theory (MPT)

- **Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002)**. "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II". *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
  - NSGA-2 algorithm used in this implementation

- **Chankong, V. & Haimes, Y. Y. (1983)**. *Multiobjective Decision Making: Theory and Methodology*. North-Holland.
  - Epsilon-constraint method

### Libraries

- **NumPy & SciPy**: Scientific computing and optimization
- **Pandas**: Data manipulation and analysis
- **DEAP** (Distributed Evolutionary Algorithms in Python): Evolutionary computation framework
- **Streamlit**: Interactive web applications for data science
- **Plotly**: Interactive visualization library

### Additional Resources

- For detailed implementation: See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- For method comparisons: See "Compare All Methods" in the Streamlit app
- Source code: Available in this repository

---

**Project**: Portfolio Multi-Objective Optimization
**Documentation**: README.md (quick start) + IMPLEMENTATION_GUIDE.md (detailed)
**Deliverables**: ✅ Interactive Streamlit App + ✅ Pre-computed Pareto Fronts + ✅ Comprehensive Documentation
