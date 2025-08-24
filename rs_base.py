# rs_mcc_only.py
# McCormick (McC) DRO procurement optimization — CVXPY MILP only
# No Mosek, no CPP, no SAA.

import numpy as np
import pandas as pd
import math
import cvxpy as cp
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from hmmlearn import hmm
from itertools import product
import time

# External OOS evaluator (your module)
from oos_analys_RS import OOS_analys

warnings.filterwarnings("ignore")


# ============================ HMM helper ============================

def train_hmm_with_sorted_states(data, n_components=2, random_state=42, covariance_type='full', n_iter=100):
    """
    Train a Gaussian HMM on 1-D labels; sort states by mean so indices are stable/interpretable.
    Returns:
        model (fitted and re-ordered),
        remapped_states (np.array),
        sorted_transmat,
        sorted_means,
        state_counts (np.array length n_components)
    """
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(data)
    hidden_states = model.predict(data)

    means = model.means_.flatten()
    sort_idx = np.argsort(means)
    mapping = {old: new for new, old in enumerate(sort_idx)}
    remapped = np.array([mapping[s] for s in hidden_states])

    state_counts = np.bincount(remapped, minlength=n_components)
    sorted_transmat = model.transmat_[sort_idx][:, sort_idx]
    sorted_means = model.means_[sort_idx]
    sorted_covars = model.covars_[sort_idx]

    model.transmat_ = sorted_transmat
    model.means_ = sorted_means
    model.covars_ = sorted_covars
    return model, remapped, sorted_transmat, sorted_means, state_counts


# ============================ demand generators ============================

from scipy.special import gamma as sp_gamma
from scipy.optimize import fsolve

def get_gaussian_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean_vector = np.full((time_horizon,), mean)
    covariance_matrix = np.diag(np.full((time_horizon,), std_dev**2))
    cov_value = rho * std_dev**2
    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value
    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    samples = np.round(samples).astype(int).T
    return pd.DataFrame(samples)

def get_gamma_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)
    k = mean**2 / std_dev**2
    theta = std_dev**2 / mean
    m = time_horizon * M
    samples = np.random.gamma(shape=k, scale=theta, size=m)
    samples = np.round(samples).astype(int).reshape((time_horizon, M))
    return pd.DataFrame(samples)

def get_lognormal_demand(time_horizon, mean, std_dev, rho, M, seed=None):
    if seed is not None:
        np.random.seed(seed)
    normal_mean = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    normal_std_dev = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))
    mean_vector = np.full((time_horizon,), normal_mean)
    covariance_matrix = np.diag(np.full((time_horizon,), normal_std_dev**2))
    cov_value = rho * normal_std_dev**2
    for j in range(time_horizon - 1):
        covariance_matrix[j, j+1] = covariance_matrix[j+1, j] = cov_value
    warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite")
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, M)
    samples = np.round(np.exp(samples)).astype(int).T
    return pd.DataFrame(samples)

def get_weibull_demand(time_horizon, mean, std_dev, M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def weibull_params(mean_, std_):
        def equations(vars_):
            k_, lam_ = vars_
            mean_eq = lam_ * sp_gamma(1 + 1/k_) - mean_
            std_eq = lam_ * np.sqrt(sp_gamma(1 + 2/k_) - sp_gamma(1 + 1/k_)**2) - std_
            return [mean_eq, std_eq]
        return fsolve(equations, (0.5, mean_))

    k, lam = weibull_params(mean, std_dev)
    samples = np.random.weibull(k, size=(time_horizon, M)) * lam
    samples = np.round(samples).astype(int)
    return pd.DataFrame(samples)

def get_mixture_demand(distribution_list, time_horizon, M, seed=None, M_max=20000):
    if seed is not None:
        np.random.seed(seed)

    all_samples = []
    all_labels = []
    for i, dist_info in enumerate(distribution_list):
        dist_type = dist_info['type'].lower()
        weight = dist_info['weight']
        params = dist_info['params']
        sample_size_max = int(round(weight * M_max))
        local_seed = (seed or 0) + i * 99991

        if dist_type == 'gaussian':
            samples = get_gaussian_demand(time_horizon, params['mean'], params['std_dev'], params.get('rho', 0), sample_size_max, local_seed)
        elif dist_type == 'gamma':
            samples = get_gamma_demand(time_horizon, params['mean'], params['std_dev'], sample_size_max, local_seed)
        elif dist_type == 'lognormal':
            samples = get_lognormal_demand(time_horizon, params['mean'], params['std_dev'], params.get('rho', 0), sample_size_max, local_seed)
        elif dist_type == 'weibull':
            samples = get_weibull_demand(time_horizon, params['mean'], params['std_dev'], sample_size_max, local_seed)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        sample_size = int(round(weight * M))
        samples = samples.iloc[:, :sample_size]
        all_samples.append(samples)
        all_labels.extend([i]*sample_size)

    combined = pd.concat(all_samples, axis=1).T
    combined['label'] = all_labels
    shuffled = combined.sample(frac=1, random_state=seed).reset_index(drop=True).iloc[:M, :]
    labels = shuffled['label']
    samples_df = shuffled.drop(columns='label').T
    samples_df.columns = range(samples_df.shape[1])
    return samples_df, labels


# ============================ Model class (McC only) ============================

class Models:
    def __init__(self, h, b, I_0, B_0, R, input_parameters_file, dist, input_demand, N):
        data_price    = pd.read_excel(input_parameters_file, sheet_name='price')
        data_supplier = pd.read_excel(input_parameters_file, sheet_name='supplier')
        data_capacity = pd.read_excel(input_parameters_file, sheet_name='capacity')

        self.h = h
        self.b = b
        self.I_0 = I_0
        self.B_0 = B_0
        self.N = N
        self.Nlist = list(range(N))
        self.R = R
        self.dist = dist
        self.demand = input_demand  # dict: state -> DataFrame (time x Nk[state])
        self.price_df = data_price

        self.time = range(next(iter(input_demand.values())).shape[0])  # rows of any state's DF
        self.supplier, self.order_cost, self.lead_time, self.quality_level = self.get_suppliers(data_supplier)
        self.prices, self.capacities = self.get_time_suppliers(data_price, data_capacity)

        self.t_supplier = [(t, s) for t in self.time for s in self.supplier]
        self.t_supplier_n = [(t, s, n) for t in self.time for s in self.supplier for n in self.Nlist]

    @staticmethod
    def get_structure(*args):
        if len(args) == 2:
            return [(a, b) for a in args[0] for b in args[1]]
        if len(args) == 3:
            return [(a, b, c) for a in args[0] for b in args[1] for c in args[2]]
        if len(args) == 4:
            return [(a, b, c, d) for a in args[0] for b in args[1] for c in args[2] for d in args[3]]
        raise ValueError("get_structure supports up to 4 iterable args")

    @staticmethod
    def _multidict_like(multi_temp):
        """
        Mimic gurobipy.multidict: input {key: [v1, v2, v3]}
        return (keys_list, dict1, dict2, dict3)
        """
        keys = list(multi_temp.keys())
        d1, d2, d3 = {}, {}, {}
        for k, vals in multi_temp.items():
            d1[k], d2[k], d3[k] = vals
        return keys, d1, d2, d3

    def get_suppliers(self, data_supplier):
        supplier = data_supplier['supplier'].values
        order_cost = data_supplier['order_cost'].values
        lead_time = data_supplier['lead_time'].values
        quality_level = data_supplier['quality_level'].values
        multi_temp = {}
        for i in range(len(supplier)):
            multi_temp[supplier[i]] = [float(order_cost[i]), float(lead_time[i]), float(quality_level[i])]
        return self._multidict_like(multi_temp)

    def get_time_suppliers(self, data_price, data_capacity):
        price_sn, capacity_sn = [0], [0]
        for i in range(1, len(self.supplier)+1):
            price_sn.append(data_price['s'+str(i)].values)
            capacity_sn.append(data_capacity['s'+str(i)].values)
        prices = {}
        capacities = {}
        for t in self.time:
            for s in self.supplier:
                n = int(s[1:])
                prices[(t, s)] = float(price_sn[n][t])
                capacities[(t, s)] = float(capacity_sn[n][t])
        return prices, capacities

    # ================= McC MILP in CVXPY (exact translation of your Gurobi model) =================
    def optimize_McC_rs(self, K, epsilon, r, Nk, solver_preference=None, verbose=False):
        """
        CVXPY MILP version of your McCormick (McC) DRO model.
        Args:
            K: number of regimes
            epsilon: dict {state -> epsilon_k}
            r: array-like of length K (weights/probabilities)
            Nk: array-like of length K (sample counts for each regime)
        Returns:
            (objective_value, df_solution)
        """
        T_idx = list(self.time)
        S = list(self.supplier)
        T = T_idx[-1]

        P_I = self.I_0 - self.B_0
        b = self.b
        h = self.h
        Mbig = 999_999.0

        # demand bounds per (k,n)
        dU, dL = {}, {}
        for k in range(K):
            dU[k], dL[k] = {}, {}
            for n in range(Nk[k]):
                col = self.demand[k].iloc[:, n]
                dU[k][n] = float(col.max())
                dL[k][n] = float(col.min())

        # decision variables
        Q     = {(t, s): cp.Variable(nonneg=True,  name=f"order_quantity[{t},{s}]")   for t in T_idx for s in S}
        theta = {(t, s): cp.Variable(nonneg=True,  name=f"arrive_quantity[{t},{s}]")  for t in T_idx for s in S}
        Y     = {(t, s): cp.Variable(boolean=True, name=f"if_make_order_arrive[{t},{s}]") for t in T_idx for s in S}

        alpha = {(k, n): cp.Variable(name=f"alpha[{k},{n}]") for k in range(K) for n in range(Nk[k])}
        beta  = {k: cp.Variable(nonneg=True, name=f"beta[{k}]") for k in range(K)}

        delta = {(k,n,t): cp.Variable(nonneg=True, name=f"delta[{k},{n},{t}]") for k in range(K) for n in range(Nk[k]) for t in T_idx}
        sigma = {(k,n,t): cp.Variable(nonneg=True, name=f"sigma[{k},{n},{t}]") for k in range(K) for n in range(Nk[k]) for t in T_idx}
        gamma = {(k,n,t): cp.Variable(nonneg=True, name=f"gamma[{k},{n},{t}]") for k in range(K) for n in range(Nk[k]) for t in T_idx}
        tau   = {(k,n,t): cp.Variable(nonneg=True, name=f"tau[{k},{n},{t}]")   for k in range(K) for n in range(Nk[k]) for t in T_idx}
        phi   = {(k,n,t): cp.Variable(nonneg=True, name=f"phi[{k},{n},{t}]")   for k in range(K) for n in range(Nk[k]) for t in T_idx}
        xi    = {(k,n,t): cp.Variable(nonneg=True, name=f"xi[{k},{n},{t}]")    for k in range(K) for n in range(Nk[k]) for t in T_idx}
        zeta  = {(k,n,t): cp.Variable(nonneg=True, name=f"zeta[{k},{n},{t}]")  for k in range(K) for n in range(Nk[k]) for t in T_idx}
        varphi= {(k,n,t): cp.Variable(nonneg=True, name=f"varphi[{k},{n},{t}]")for k in range(K) for n in range(Nk[k]) for t in T_idx}
        eta   = {(k,n,t): cp.Variable(nonneg=True, name=f"eta[{k},{n},{t}]")   for k in range(K) for n in range(Nk[k]) for t in T_idx}
        ta    = {(k,n)  : cp.Variable(nonneg=True, name=f"ta[{k},{n}]")        for k in range(K) for n in range(Nk[k])}

        cons = []

        # arrival linkage: theta[tp,s] = sum_t Q[t,s] where t + L_s == tp
        for s in S:
            Lt = int(round(self.lead_time[s]))
            for tp in T_idx:
                cons.append(theta[tp, s] == cp.sum([Q[t, s] for t in T_idx if t + Lt == tp]))

        # big-M and capacity
        for t in T_idx:
            for s in S:
                cons += [
                    Q[t, s] <= Mbig * Y[t, s],
                    Q[t, s] <= self.capacities[(t, s)],
                    Y[t, s] >= 0,
                    Y[t, s] <= 1
                ]

        # McC DRO constraints
        for k in range(K):
            for n in range(Nk[k]):
                cons.append(ta[k, n] <= beta[k])

            for n in range(Nk[k]):
                for t in T_idx:
                    cons.append(delta[k, n, t] + sigma[k, n, t] - ta[k, n] <= 0)

            for n in range(Nk[k]):
                # master inequality bounding alpha[k,n]
                terms = []
                terms.append((b + h) * cp.sum(cp.hstack([tau[k, n, t] for t in T_idx if t < T])))
                terms.append((T + 1) * h * P_I)
                for t in T_idx:
                    dnt = float(self.demand[k].iloc[t, n])
                    terms.append((delta[k, n, t] - sigma[k, n, t]) * dnt)
                    terms.append((T - t + 1) * h * cp.sum(cp.hstack([theta[t, s] for s in S])))
                    terms.append(xi[k, n, t]    * (T - t + 1) * (b + h) * dU[k][n])
                    terms.append(- zeta[k, n, t]* (T - t + 1) * (b + h) * dL[k][n])
                    terms.append(eta[k, n, t]   * (T - t + 1) * (b + h))
                cons.append(cp.sum(cp.hstack(terms)) <= alpha[k, n])

                for t in T_idx:
                    cons.append(1 + phi[k, n, t] + xi[k, n, t] - zeta[k, n, t] - varphi[k, n, t] <= 0)

                for t in T_idx:
                    cons.append(
                        sigma[k, n, t] - delta[k, n, t]
                        + (zeta[k, n, t] - xi[k, n, t]) * (T - t + 1) * (h + b)
                        - (T - t + 1) * h
                        <= 0
                    )

                for t in T_idx:
                    sum_theta_t = cp.sum(cp.hstack([theta[t, s] for s in S]))
                    if t == 0:
                        cons.append(
                            gamma[k, n, t] - tau[k, n, t]
                            - phi[k, n, t] * dL[k][n] - xi[k, n, t] * dU[k][n]
                            + zeta[k, n, t] * dL[k][n] + varphi[k, n, t] * dU[k][n]
                            - eta[k, n, t] - sum_theta_t - P_I
                            <= 0
                        )
                    elif t == T:
                        cons.append(
                            tau[k, n, t - 1] - gamma[k, n, t - 1]
                            - phi[k, n, t] * dL[k][n] - xi[k, n, t] * dU[k][n]
                            + zeta[k, n, t] * dL[k][n] + varphi[k, n, t] * dU[k][n]
                            - eta[k, n, t] - sum_theta_t
                            <= 0
                        )
                    else:
                        cons.append(
                            gamma[k, n, t] - tau[k, n, t] + tau[k, n, t - 1] - gamma[k, n, t - 1]
                            - phi[k, n, t] * dL[k][n] - xi[k, n, t] * dU[k][n]
                            + zeta[k, n, t] * dL[k][n] + varphi[k, n, t] * dU[k][n]
                            - eta[k, n, t] - sum_theta_t
                            <= 0
                        )

        # objective
        fixed_order = cp.sum([ self.order_cost[s] * Y[t, s] for t in T_idx for s in S ])
        purchase    = cp.sum([ self.prices[(t, s)] * Q[t, s] for t in T_idx for s in S ])
        dro_term    = cp.sum([ (r[k] * cp.sum(cp.hstack([alpha[k, n] for n in range(Nk[k])])) / Nk[k]
                                + r[k] * beta[k] * epsilon[k]) for k in range(K) ])
        obj = fixed_order + purchase + dro_term

        prob = cp.Problem(cp.Minimize(obj), cons)

        # solver order (MIP-capable)
        order = []
        if solver_preference is not None:
            order.append(solver_preference)
        order += [cp.SCIPY, cp.GUROBI, cp.CBC, cp.GLPK_MI, cp.SCIP, cp.ECOS_BB]

        status = None
        for s in order:
            if s not in cp.installed_solvers():
                continue
            try:
                prob.solve(solver=s, verbose=verbose)
                status = prob.status
                if status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed; last status: {status}; installed={cp.installed_solvers()}")

        # export solution in a simple DataFrame
        rows = []
        for (t, s), v in Q.items():
            rows.append({"variable_name": f"order_quantity[{t},{s}]", "value": float(v.value)})
        for (t, s), v in theta.items():
            rows.append({"variable_name": f"arrive_quantity[{t},{s}]", "value": float(v.value)})
        for (t, s), v in Y.items():
            rows.append({"variable_name": f"if_make_order_arrive[{t},{s}]", "value": float(v.value)})
        df_result = pd.DataFrame(rows)
        return float(prob.value), df_result


# ============================ experiment driver (McC only) ============================

if __name__ == "__main__":
    input_parameters_file = 'input_parameters.xlsx'
    h = 5
    b = 20
    I_0 = 1800
    B_0 = 0
    R = 0

    k_fold = 5
    oos_size = 5000

    planning_horizon = 8
    rho = 0

    seed = 25
    random_state = seed

    input_sample_no = [10, 20, 40]  # adjust as you like
    oos_analys = OOS_analys(h, b, I_0, B_0, input_parameters_file)
    all_res = {}
    all_df = []
    num_regime = 2

    # generate input samples (mixture)
    Demand_samples, Label_samples = get_mixture_demand(
        distribution_list=[
            {'type': 'gaussian', 'weight': 0.5, 'params': {'mean': 1800, 'std_dev': 500, 'rho': 0}},
            {'type': 'gamma',    'weight': 0.5, 'params': {'mean': 2000, 'std_dev': 500}}
        ],
        time_horizon=planning_horizon, M=200, seed=seed
    )

    for input_dist in ['mix']:
        all_res['input_dist'] = input_dist
        for out_sample_dist in ['mix']:
            for N in input_sample_no:
                start = time.time()
                all_res['N'] = N

                input_demand = Demand_samples.iloc[:, :N]
                labels = Label_samples[:N]

                n_components = num_regime
                state_samples_dict = {i: [] for i in range(n_components)}

                data = np.array(input_demand).T
                if num_regime > 1:
                    Labels = labels.values.reshape(len(labels), 1)
                    model, _, _, _, num = train_hmm_with_sorted_states(Labels, n_components, random_state, 'full', 100)
                    print(num)
                    for i, state in enumerate(labels):
                        state_samples_dict[state].append(data[i])

                    state_samples_df = {}
                    state_counts = []
                    for state, samples in state_samples_dict.items():
                        samples_array = np.array(samples).T if len(samples) > 0 else np.zeros((planning_horizon, 0))
                        state_samples_df[state] = pd.DataFrame(samples_array, columns=[f"{j}" for j in range(samples_array.shape[1])])
                        state_counts.append(state_samples_df[state].shape[1])
                    print(f'state_counts is {state_counts}')
                    for state, df in state_samples_df.items():
                        df.to_csv(f"state_{state}_samples.csv", index=False)
                        print(f"sample with state {state} saved: state_{state}_samples.csv")
                else:
                    state_samples_df = {0: pd.DataFrame(data.T)}
                    state_counts = [N]

                # --------- Cross-validate epsilon for McC only ----------
                cross_res = {'McC_rs': {}}
                min_epsilons = [ {0: 0, 1: 0} for _ in range(k_fold-1) ]  # init

                for k in range(1, k_fold):
                    num_fold = N // k_fold
                    train_size = N - num_fold
                    all_cols = list(range(input_demand.shape[1]))
                    selected_cols = [i for i in all_cols if i < k*num_fold or i >= (k+1)*num_fold]

                    train_demand = input_demand.iloc[:, selected_cols]
                    train_demand.columns = range(train_demand.shape[1])
                    CV_input_demand = input_demand.iloc[:, k*num_fold:(k+1)*num_fold]
                    CV_input_demand.columns = range(num_fold)

                    train_labels = labels[selected_cols].values.reshape(-1, 1)
                    model_cv, cv_states, cv_transmat, _, train_num = train_hmm_with_sorted_states(
                        train_labels, n_components, random_state, 'full', 100
                    )

                    # split train samples by latest known state label for indexing
                    train_input_demand = {}
                    uniques = pd.unique(labels)
                    for l in uniques:
                        mask = (train_labels.flatten() == l)
                        train_input_demand[int(l)] = train_demand.loc[:, mask]
                        train_input_demand[int(l)].columns = range(train_input_demand[int(l)].shape[1])

                    # set solver
                    solve = Models(h, b, I_0, B_0, R, input_parameters_file, input_dist, train_input_demand, train_size)

                    # we use the row of transition matrix corresponding to the last observed label in the train split
                    idx_row = int(labels[k*num_fold-1]) if k*num_fold-1 >= 0 else 0
                    tran_matrix = model_cv.transmat_[idx_row]
                    min_cost = 1e18

                    list_epsilon = [0, 30, 60]  # search grid
                    combos = [ {0: e0, 1: e1} for (e0, e1) in product(list_epsilon, list_epsilon) ]

                    for eps in combos:
                        fold_costs = []
                        for j in range(num_fold):
                            r_vec = model_cv.transmat_[ int(labels[min(N-1, k*num_fold + j - 1)]) ] if (k*num_fold + j - 1) >= 0 else tran_matrix
                            # Nk per state for train
                            Nk_train = [train_input_demand.get(st, pd.DataFrame()).shape[1] for st in range(n_components)]
                            # guard zero-sample state
                            Nk_sanitized = [max(1, x) for x in Nk_train]

                            obj, solution = solve.optimize_McC_rs(n_components, eps, r_vec, Nk_sanitized,
                                                                  solver_preference=None, verbose=False)
                            # OOS on validation column j
                            cost, _ = oos_analys.cal_out_of_sample(solution, CV_input_demand[j])
                            fold_costs.append(cost.total_cost.mean())
                        avg_cost = float(np.mean(fold_costs))
                        if avg_cost < min_cost:
                            min_cost = avg_cost
                            min_epsilons[k-1] = eps

                # average best eps per state
                for state in state_samples_df.keys():
                    cross_res['McC_rs'][state] = np.mean([min_epsilons[i][state] for i in range(k_fold-1)])

                # --------- Final solve on full (by regime) ----------
                input_demand_regime = {state: pd.read_csv(f'state_{state}_samples.csv') for state in state_samples_df.keys()}
                state_frequencies = model.transmat_[labels[len(labels)-1:]][0]  # last row (prob of next state)
                print("Chosen eps:", cross_res['McC_rs'], "state_frequencies:", state_frequencies, "state_counts:", state_counts)

                solve_full = Models(h, b, I_0, B_0, R, input_parameters_file, input_dist, input_demand_regime, N)
                eps_final = cross_res['McC_rs']
                # sanitize Nk for final (avoid zero)
                Nk_final = [max(1, input_demand_regime.get(k, pd.DataFrame()).shape[1]) for k in range(n_components)]

                obj, solution = solve_full.optimize_McC_rs(num_regime, eps_final, state_frequencies, Nk_final,
                                                           solver_preference=None, verbose=False)

                # --------- Evaluate out-of-sample for different mixture weights ----------
                for weight in [0.3, 0.5, 0.7]:
                    oos_demands_mix, labels_oos = get_mixture_demand(
                        distribution_list=[
                            {'type': 'gaussian', 'weight': weight,   'params': {'mean': 1800, 'std_dev': 500, 'rho': 0}},
                            {'type': 'gamma',    'weight': 1-weight, 'params': {'mean': 2000, 'std_dev': 500}}
                        ],
                        time_horizon=planning_horizon, M=oos_size, seed=25
                    )
                    oos_demands = oos_demands_mix
                    oos_cost, oos_details = oos_analys.cal_out_of_sample(solution, oos_demands)

                    print(f'[McC] N={N}, OoS weight={weight}: mean={oos_cost.total_cost.mean():.3f}')
                    end = time.time()

                    all_res['model_name'] = 'McC_rs'
                    all_res['oos_weight'] = weight
                    all_res['obj'] = obj
                    all_res['epsilon'] = eps_final
                    all_res['oos_mean'] = oos_cost.total_cost.mean()
                    all_res['order'] = oos_cost.fixed_order_cost.mean()
                    all_res['purchase'] = oos_cost.purchase_cost.mean()
                    all_res['oos_inv'] = oos_cost.inv_cost.mean()
                    all_res['oos_backlog'] = oos_cost.backlog_cost.mean()
                    all_res['oos_std'] = oos_cost.total_cost.std()
                    all_res['seed'] = seed
                    all_res['time_min'] = (end - start) / 60
                    all_df.append(pd.DataFrame.from_dict(all_res, orient='index'))

    # Save summary
    if all_df:
        res_df = pd.concat(all_df, axis=1).T
        res_df.to_excel('McC_only_results.xlsx', index=False)
        pd.options.display.float_format = '{:.2f}'.format
        print(res_df)
    else:
        print("No results to save.")
