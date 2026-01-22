import numpy as np

# Generate X
def generate_X(n, seed=None):
    """
    Generate explanatory variable X ~ Uniform(0, 10)

    Parameters
    ----------
    n : int
        Sample size
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray
        Array of shape (n,)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(low=0.0, high=10.0, size=n)
    return X

# Error: Baseline (Normal, Homoskedastic, Independent)
def generate_error_baseline(n, sigma=1.0, seed=None):
    """
    ε_i ~ N(0, sigma^2)
    """
    if seed is not None:
        np.random.seed(seed)

    eps = np.random.normal(loc=0.0, scale=sigma, size=n)
    return eps


# Error: Non-normal (Heavy-tailed)
def generate_error_non_normal(n, sigma=1.0, df=3, seed=None):
    """
    Heavy-tailed errors using Student-t distribution.
    Errors are centered and scaled to have mean 0 and variance sigma^2.
    """
    if df <= 2:
        raise ValueError("Degrees of freedom must be > 2 for finite variance.")

    if seed is not None:
        np.random.seed(seed)

    eps_raw = np.random.standard_t(df=df, size=n)
    eps_centered = eps_raw - np.mean(eps_raw)

    current_std = np.std(eps_centered, ddof=1)
    eps = eps_centered * (sigma / current_std)

    return eps

# Error: Heteroskedastic
def generate_error_heteroskedastic(X, sigma=1.0, gamma=0.5, seed=None):
    """
    Var(ε_i | X_i) = sigma^2 * (1 + gamma * X_i)
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    if seed is not None:
        np.random.seed(seed)

    n = len(X)
    Z = np.random.normal(loc=0.0, scale=1.0, size=n)

    variance = sigma**2 * (1.0 + gamma * X)
    eps = np.sqrt(variance) * Z

    return eps

# Error: Autocorrelated (AR(1))
def generate_error_autocorrelated(n, sigma=1.0, rho=0.5, seed=None):
    """
    ε_t = rho * ε_{t-1} + u_t
    u_t ~ N(0, sigma^2)
    """
    if abs(rho) >= 1:
        raise ValueError("rho must satisfy |rho| < 1 for stationarity.")

    if seed is not None:
        np.random.seed(seed)

    u = np.random.normal(loc=0.0, scale=sigma, size=n)
    eps = np.zeros(n)

    eps[0] = u[0]
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + u[t]

    return eps

# Generate Y
def generate_Y(X, eps, beta0=2.0, beta1=3.0):
    """
    Generate response variable Y from:
    Y = beta0 + beta1 * X + eps
    """
    Y = beta0 + beta1 * X + eps
    return Y
