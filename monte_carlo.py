#%%
# Import libraries
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import time
from scipy.stats import norm
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool

# %%
# Define functions
def geo_bm_paths(S0, T, r, sigma, M, N, seed=None):
    """
    Function:
        Generate geometric Brownian motion paths.
    Parameters:
        S0 (float): Initial stock price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        sigma (float): Volatility
        M (int): Number of time steps
        N (int): Number of paths to simulate
        seed (int, optional): Random seed for reproducibility
    Returns:
        np.ndarray: Simulated paths
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / M
    path = np.zeros(M + 1)
    path[0] = S0
    for t in range(1, M + 1):
        Z = np.random.normal()
        path[t] = path[t - 1] + r * path[t - 1] * dt + sigma * path[t - 1] * np.sqrt(dt) * Z
    return path

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Function:
        Calculate Black-Scholes call option price.
    Parameters:
        S (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        sigma (float): Volatility
    Returns:
       float: Black-Scholes call option price
    """
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def simulate_option_price_non_parallel(S0, K, T, r, sigma, M, N, seed=None):
    """
    Function:
        Simulate call option price using Monte Carlo method (non-parallel).
    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        sigma (float): Volatility
        M (int): Number of time steps
        N (int): Number of paths to simulate
        seed (int, optional): Random seed for reproducibility
    Returns:
        float: Simulated call option price
        float: Runtime of the simulation
    """
    start_time = time.time()
    payoffs = np.zeros(N)
    for i in range(N):
        path = geo_bm_paths(S0, T, r, sigma, M, N, seed[i])
        payoffs[i] = max(path[-1] - K, 0)
    option_price = np.mean(payoffs) * np.exp(-r * T)
    print(option_price)
    runtime = time.time() - start_time
    return option_price, runtime


def simulate_option_price_parallel(S0, K, T, r, sigma, M, N, seed=None):
    """
    Function:
        Simulate call option price using Monte Carlo method (parallel).
    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free interest rate
        sigma (float): Volatility
        M (int): Number of time steps
        N (int): Number of paths to simulate
        seed (int, optional): Random seed for reproducibility
    Returns:
        float: Simulated call option price
        float: Runtime of the simulation
    """
    start_time = time.time()
    num_jobs = multiprocessing.cpu_count()
    chunk_size = N // num_jobs

    def parallel_simulate(chunk_size, seed):
        payoffs = np.zeros(chunk_size)
        # print(seed)
        for i in range(chunk_size):
            path = geo_bm_paths(S0, T, r, sigma, M, N, seed[i])
            payoffs[i] = max(path[-1] - K, 0)
        return np.mean(payoffs) * np.exp(-r * T)

    seeds = np.array_split(seed, num_jobs)
    results = Parallel(n_jobs=num_jobs)(delayed(parallel_simulate)(chunk_size, s) for s in seeds)
    option_price = np.mean(results)
    runtime = time.time() - start_time

    return option_price, runtime


# create a mp version of simulate_option_price_parallel
def simulate_option_price_parallel_MP(S0, K, T, r, sigma, M, N, seed=None):
    start_time = time.time()

    num_jobs = multiprocessing.cpu_count()
    chunk_size = N // num_jobs
    chunk_size_list = int(chunk_size) * np.ones(num_jobs)
    chunk_size_list = chunk_size_list.astype(np.int64)
    seeds = np.array_split(seed, num_jobs)
    pool = Pool(ncpus=num_jobs)

    def parallel_simulate_MP(chunk_size, seed):
        payoffs = np.zeros(chunk_size)
        for i in range(chunk_size):
            path = geo_bm_paths(S0, T, r, sigma, M, N, seed[i])
            payoffs[i] = max(path[-1] - K, 0)
        temp = np.mean(payoffs) * np.exp(-r * T)
        return temp

    results = pool.map(parallel_simulate_MP, chunk_size_list, seeds)
    option_price = np.mean(results)
    runtime = time.time() - start_time
    return option_price, runtime

# %%
# Main code
if __name__ == "__main__":
    S0 = 100  # Initial stock price
    K = 100  # Strike price
    T = 1.0  # Time to maturity
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility
    M = 1000  # Number of time steps
    # Ns = [1000, 5000, 10000]
    Ns = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]  # Number of paths
    # Ns = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]  # Number of paths
    results = []

    for N in Ns:
        # Monte Carlo Simulation (Non-Parallel)
        seed = np.arange(42, N + 42)  # Random seed
        monte_carlo_price, non_parallel_runtime = simulate_option_price_non_parallel(S0, K, T, r, sigma, M, N, seed)
        # Monte Carlo Simulation (Parallel)
        parallel_price, parallel_runtime = simulate_option_price_parallel_MP(S0, K, T, r, sigma, M, N, seed)
        # Black-Scholes Price
        black_scholes_price = black_scholes_call_price(S0, K, T, r, sigma)
        # Error Calculation
        error = abs(monte_carlo_price - black_scholes_price)
        # Append results
        results.append([N, monte_carlo_price, black_scholes_price, error, non_parallel_runtime, parallel_runtime])

    # Convert results to DataFrame
    results_df = pd.DataFrame(results,
                              columns=['N', 'MC_Price', 'BS_Price', 'Error', 'NonParallel_Time', 'Parallel_Time'])
    print(results_df)

    # Save results to CSV
    results_df.to_csv('monte_carlo.csv', index=False)

