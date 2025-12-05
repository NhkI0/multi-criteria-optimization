"""
Main script to generate Pareto fronts for all levels
Run this BEFORE launching the Streamlit app
"""
from data_loader import load_portfolio_data
from pareto_optimizer import ParetoPortfolioOptimizer
from comparison_methods import EpsilonConstraintOptimizer
import time

def main():
    print("="*70)
    print("PORTFOLIO OPTIMIZATION - PARETO FRONT GENERATION")
    print("="*70)

    # Load data
    print("\n[1/5] Loading data...")
    data_dir = "../data"
    prices, mu, Sigma, sectors = load_portfolio_data(data_dir)

    # Initialize optimizers
    optimizer = ParetoPortfolioOptimizer(mu, Sigma, sectors)
    epsilon_opt = EpsilonConstraintOptimizer(mu, Sigma)

    # === LEVEL 1: Bi-objective (Return-Risk) ===

    # Method 1: Weighted Sum
    print("\n[2/5] Level 1 - Weighted Sum Scalarization...")
    start = time.time()
    pareto_l1_weighted = optimizer.generate_pareto_front_level1(n_points=50)
    optimizer.save_pareto_front(pareto_l1_weighted, 'pareto_level1_weighted.json')
    print(f"  Time: {time.time() - start:.1f}s")

    # Method 2: Epsilon-Constraint (for comparison)
    print("\n[3/5] Level 1 - Epsilon-Constraint Method...")
    start = time.time()
    pareto_l1_epsilon = epsilon_opt.generate_pareto_front_epsilon_level1(n_points=30)
    optimizer.save_pareto_front(pareto_l1_epsilon, 'pareto_level1_epsilon.json')
    print(f"  Time: {time.time() - start:.1f}s")

    # === LEVEL 2: Tri-objective (Return-Risk-Costs) with Cardinality ===

    print("\n[4/5] Level 2 - Tri-objective with K=20...")
    start = time.time()
    pareto_l2 = optimizer.generate_pareto_front_level2(
        K=20,
        w_current=None,  # Assume starting from scratch
        c_prop=0.001,
        n_points=40
    )
    optimizer.save_pareto_front(pareto_l2, 'pareto_level2_K20.json')
    print(f"  Time: {time.time() - start:.1f}s")

    # Optional: Generate with different K values
    print("\n[5/5] Level 2 - Tri-objective with K=30...")
    start = time.time()
    pareto_l2_k30 = optimizer.generate_pareto_front_level2(
        K=30,
        w_current=None,
        c_prop=0.001,
        n_points=40
    )
    optimizer.save_pareto_front(pareto_l2_k30, 'pareto_level2_K30.json')
    print(f"  Time: {time.time() - start:.1f}s")

    print("\n" + "="*70)
    print("[OK] ALL PARETO FRONTS GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - pareto_level1_weighted.json (50 portfolios)")
    print("  - pareto_level1_epsilon.json (30 portfolios)")
    print("  - pareto_level2_K20.json (40 portfolios)")
    print("  - pareto_level2_K30.json (40 portfolios)")
    print("\nNow run: streamlit run app_simple.py")


if __name__ == "__main__":
    main()
