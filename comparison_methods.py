"""
Alternative optimization methods for comparison
Implements epsilon-constraint method and NSGA-2
"""
import numpy as np
from scipy.optimize import minimize
from deap import base, creator, tools, algorithms
import random


class EpsilonConstraintOptimizer:
    """
    Epsilon-constraint method for multi-objective optimization

    Idea: Optimize one objective while constraining others
    min f1(w)
    subject to: f2(w) ≤ ε2, f3(w) ≤ ε3, C_Base
    """

    def __init__(self, mean_returns, cov_matrix):
        self.mu = mean_returns.values
        self.Sigma = cov_matrix.values
        self.n_assets = len(self.mu)

    def f1_return(self, w):
        return -np.dot(w, self.mu)

    def f2_variance(self, w):
        return np.dot(w, np.dot(self.Sigma, w))

    def f3_transaction_cost(self, w, w_current, c_prop):
        return c_prop * np.sum(np.abs(w - w_current))

    def generate_pareto_front_epsilon_level1(self, n_points=30):
        """
        Generate Pareto front using epsilon-constraint on Level 1

        Approach:
        - min f1(w) subject to f2(w) ≤ ε
        - Vary ε from min to max risk
        """
        print(f"\n{'='*60}")
        print("LEVEL 1: Epsilon-Constraint Method")
        print(f"{'='*60}")
        print(f"Approach: min f1 subject to f2 <= epsilon")
        print(f"Points: {n_points}")

        pareto_solutions = []

        # Find risk range
        # Min risk: optimize f2 alone
        w_init = np.ones(self.n_assets) / self.n_assets
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_assets)]

        result_min_risk = minimize(self.f2_variance, w_init, method='SLSQP',
                                   bounds=bounds, constraints=constraints,
                                   options={'maxiter': 500})

        if result_min_risk.success:
            risk_min = self.f2_variance(result_min_risk.x)
        else:
            risk_min = 0.01

        # Max return portfolio (tends to have high risk)
        result_max_return = minimize(self.f1_return, w_init, method='SLSQP',
                                     bounds=bounds, constraints=constraints,
                                     options={'maxiter': 500})

        if result_max_return.success:
            risk_max = self.f2_variance(result_max_return.x)
        else:
            risk_max = 0.5

        print(f"  Risk (variance) range: {risk_min:.4f} to {risk_max:.4f}")

        # Generate epsilon values
        epsilons = np.linspace(risk_min, risk_max, n_points)

        for i, epsilon in enumerate(epsilons):
            # Optimize return with risk constraint
            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w, eps=epsilon: eps - self.f2_variance(w)}  # f2 ≤ ε
            ]

            result = minimize(self.f1_return, w_init, method='SLSQP',
                            bounds=bounds, constraints=constraints_list,
                            options={'maxiter': 500, 'disp': False})

            if result.success:
                w = result.x
                pareto_solutions.append({
                    'weights': w.tolist(),
                    'f1_return': float(-self.f1_return(w)),
                    'f2_variance': float(self.f2_variance(w)),
                    'f2_volatility': float(np.sqrt(self.f2_variance(w))),
                    'epsilon': float(epsilon),
                    'n_assets': int(np.sum(w > 0.001))
                })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_points} points generated")

        print(f"\n[OK] Generated {len(pareto_solutions)} solutions")

        # Display range
        returns = [sol['f1_return'] for sol in pareto_solutions]
        vols = [sol['f2_volatility'] for sol in pareto_solutions]
        print(f"  Return range: {min(returns):.2%} to {max(returns):.2%}")
        print(f"  Risk range: {min(vols):.2%} to {max(vols):.2%}")

        return pareto_solutions


class NSGA2Optimizer:
    """
    NSGA-2 (Non-dominated Sorting Genetic Algorithm II)

    A multi-objective evolutionary algorithm that:
    - Uses non-dominated sorting to rank solutions
    - Maintains diversity through crowding distance
    - Can find non-convex Pareto fronts
    """

    def __init__(self, mean_returns, cov_matrix):
        self.mu = mean_returns.values
        self.Sigma = cov_matrix.values
        self.n_assets = len(self.mu)

    def f1_return(self, w):
        """Maximize return (minimize negative return)"""
        return -np.dot(w, self.mu)

    def f2_variance(self, w):
        """Minimize variance"""
        return np.dot(w, np.dot(self.Sigma, w))

    def evaluate_portfolio(self, individual):
        """Evaluate both objectives for NSGA-2"""
        w = np.array(individual)
        # Return tuple of objectives to minimize: (negative return, variance)
        return (self.f1_return(w), self.f2_variance(w))

    def repair_portfolio(self, individual):
        """Repair portfolio to satisfy constraints"""
        w = np.array(individual)
        # Clip negative weights
        w = np.maximum(w, 0)
        # Normalize to sum to 1
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(self.n_assets) / self.n_assets
        return w.tolist()

    def generate_pareto_front_nsga2_level1(self, pop_size=100, n_gen=150, verbose=True):
        """
        Generate Pareto front using NSGA-2 for Level 1

        Parameters:
        - pop_size: Population size
        - n_gen: Number of generations
        - verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print("LEVEL 1: NSGA-2 Method")
            print(f"{'='*60}")
            print(f"Population size: {pop_size}")
            print(f"Generations: {n_gen}")

        # Setup DEAP framework
        # Clear any existing definitions
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize both
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Individual: random weights that sum to 1
        def create_individual():
            w = np.random.dirichlet(np.ones(self.n_assets))
            return creator.Individual(w.tolist())

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_portfolio)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("repair", self.repair_portfolio)

        # Create initial population
        pop = toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Evolution
        for gen in range(n_gen):
            # Select offspring
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:  # Crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.3:  # Mutation probability
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Repair constraints
            offspring = [creator.Individual(toolbox.repair(ind)) for ind in offspring]

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select next generation
            pop = toolbox.select(pop + offspring, pop_size)

            if verbose and (gen + 1) % 30 == 0:
                print(f"  Generation {gen+1}/{n_gen} completed")

        # Extract Pareto front
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        if verbose:
            print(f"\n[OK] Generated {len(pareto_front)} Pareto-optimal solutions")

        # Convert to standard format
        pareto_solutions = []
        for ind in pareto_front:
            w = np.array(ind)
            pareto_solutions.append({
                'weights': w.tolist(),
                'f1_return': float(-self.f1_return(w)),
                'f2_variance': float(self.f2_variance(w)),
                'f2_volatility': float(np.sqrt(self.f2_variance(w))),
                'n_assets': int(np.sum(w > 0.001))
            })

        # Sort by risk for consistent visualization
        pareto_solutions.sort(key=lambda x: x['f2_volatility'])

        if verbose:
            returns = [sol['f1_return'] for sol in pareto_solutions]
            vols = [sol['f2_volatility'] for sol in pareto_solutions]
            print(f"  Return range: {min(returns):.2%} to {max(returns):.2%}")
            print(f"  Risk range: {min(vols):.2%} to {max(vols):.2%}")

        return pareto_solutions


if __name__ == "__main__":
    from data_loader import load_portfolio_data

    data_dir = "../multi-criteria-optimization/data"
    _, mu, Sigma, _ = load_portfolio_data(data_dir)

    epsilon_opt = EpsilonConstraintOptimizer(mu, Sigma)
    pareto_epsilon = epsilon_opt.generate_pareto_front_epsilon_level1(n_points=30)

    # Save
    import json
    with open('pareto_epsilon_level1.json', 'w') as f:
        json.dump(pareto_epsilon, f, indent=2)
    print("\n[OK] Saved to pareto_epsilon_level1.json")
