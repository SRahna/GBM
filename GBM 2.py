import numpy as np
import cvxpy as cp
from collections import defaultdict
from functools import partial

# ----------------- Parameters -----------------
P0 = 100.0
mu, sigma = 0.05, 0.3
T = 2
dt = 1.0
N_outer = 500
N_inner = 40
gamma = 0.95
strike_price = 100.0
barrier_price = 120.0
n_actions = 2
np.random.seed(42)

# ----------------- Utilities -----------------
def simulate_gbm_paths(P0, mu, sigma, T, N_paths, dt):
    n_steps = int(T / dt)
    paths = np.zeros((N_paths, n_steps + 1))
    paths[:, 0] = P0
    for t in range(1, n_steps + 1):
        Z = np.random.randn(N_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

def simulate_one_step_from_price(P_start, mu, sigma, dt, N_inner):
    Z = np.random.randn(N_inner)
    return P_start * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

def payoff(price): return max(price - strike_price, 0)
def is_knocked_out(price): return price >= barrier_price

def make_rich_basis(state_prices, n_basis, width_factor=2.0):
    all_prices = sorted(set(price for prices in state_prices.values() for price in prices))
    min_p, max_p = min(all_prices), max(all_prices)

    full_basis = []
    n_rbf = 25
    centers = np.linspace(min_p, max_p, n_rbf)
    width = (centers[1] - centers[0]) * width_factor
    def rbf(p, c, w): return np.exp(-((p - c)**2) / (2 * w**2))
    for c in centers:
        full_basis.append(partial(rbf, c=c, w=width))

    def poly(p, d): return (p / 100.0) ** d
    for deg in range(1, 6):
        full_basis.append(partial(poly, d=deg))

    def trig(p, f): return np.sin(f * p / 100.0)
    def cos_trig(p, f): return np.cos(f * p / 100.0)
    for freq in [np.pi, 2*np.pi, 3*np.pi]:
        full_basis.append(partial(trig, f=freq))
        full_basis.append(partial(cos_trig, f=freq))

    def payoff_feat(p): return max(p - strike_price, 0) / 100.0
    def barrier_feat(p): return float(p < barrier_price)
    def payoff_sqrt(p): return np.sqrt(max(p - strike_price, 0)) / 10.0
    def inverse_price(p): return 1.0 / max(p, 1.0)

    full_basis.extend([payoff_feat, barrier_feat, payoff_sqrt, inverse_price])
    basis_list = full_basis[:n_basis]

    def phi(price, n):
        return basis_list[n](price) if n < len(basis_list) else 0.0

    return phi, len(basis_list)

# ----------------- MDP Construction -----------------
outer_paths = simulate_gbm_paths(P0, mu, sigma, T, N_outer, dt)

state_prices = {
    t: sorted(set(outer_paths[:, t].tolist()))
    for t in range(T + 1)
}

# Build transition probabilities using inner simulation
transition_probs = {t: defaultdict(float) for t in range(1, T + 1)}
for t in range(1, T + 1):
    for s_prev in state_prices[t - 1]:
        inner_next = simulate_one_step_from_price(s_prev, mu, sigma, dt, N_inner)
        for s in inner_next:
            nearest = min(state_prices[t], key=lambda x: abs(x - s))
            transition_probs[t][(s_prev, nearest)] += 1
        total = sum(transition_probs[t][(s_prev, s_next)] for s_next in state_prices[t])
        for s_next in state_prices[t]:
            key = (s_prev, s_next)
            if transition_probs[t][key] > 0:
                transition_probs[t][key] /= total

# Reward function
reward = {}
for t in range(T + 1):
    r = np.zeros((len(state_prices[t]), n_actions))
    for s, price in enumerate(state_prices[t]):
        r[s, 0] = payoff(price) * (1 - is_knocked_out(price))
        r[s, 1] = 0.0
    reward[t] = r

# ----------------- Approximate LP with Optimization -----------------
def solve_approx_lp(n_basis):
    phi, n_basis = make_rich_basis(state_prices, n_basis=n_basis, width_factor=2.0)

    # Precompute phi values
    phi_vals = {
        t: np.array([[phi(price, n) for n in range(n_basis)]
                     for price in state_prices[t]])
        for t in range(T + 1)
    }

    beta = {t: cp.Variable((n_basis, n_actions)) for t in range(T + 1)}
    obj = 0
    for t in range(T + 1):
        phi_mat = phi_vals[t]
        r = reward[t]
        obj += cp.sum(cp.multiply(phi_mat @ beta[t], r))

    constraints = []
    s0 = state_prices[0][0]
    init_phi = phi_vals[0][0]
    constraints.append(cp.sum(cp.multiply(init_phi, beta[0][:, 1])) == 1)

    for t in range(1, T + 1):
        for n in range(n_basis):
            lhs = 0
            for s, price in enumerate(state_prices[t]):
                phi_sn = phi_vals[t][s, n]
                for a in range(n_actions):
                    phi_s = phi_vals[t][s]
                    lhs += phi_sn * cp.sum(cp.multiply(phi_s, beta[t][:, a]))

            rhs_terms = []
            for s_prev, price_prev in enumerate(state_prices[t - 1]):
                if is_knocked_out(price_prev): continue
                phi_prev = phi_vals[t - 1][s_prev]
                for a_prev in range(n_actions):
                    if a_prev == 1:
                        for s_next, price_next in enumerate(state_prices[t]):
                            key = (price_prev, price_next)
                            if key in transition_probs[t]:
                                prob = transition_probs[t][key]
                                phi_next = phi_vals[t][s_next, n]
                                rhs_terms.append(
                                    gamma * prob * phi_next * cp.sum(cp.multiply(phi_prev, beta[t - 1][:, a_prev]))
                                )
            rhs = cp.sum(rhs_terms)
            constraints.append(lhs == rhs)

    for t in range(T + 1):
        for a in range(n_actions):
            constraints.append(phi_vals[t] @ beta[t][:, a] >= 0)

    problem = cp.Problem(cp.Maximize(obj), constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    return problem.value

# Run evaluation for selected basis sizes
import pandas as pd
exact_placeholder = None  # Placeholder if exact LP not required for now
results = []

for n_basis in [20, 25]:
    approx_val = solve_approx_lp(n_basis)
    results.append((n_basis, approx_val))
    print(approx_val)
#import ace_tools as tools; tools.display_dataframe_to_user(name="Approximate LP Results", dataframe=pd.DataFrame(results, columns=["n_basis", "Approximate Value"]))
