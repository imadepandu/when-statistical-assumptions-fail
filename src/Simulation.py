import numpy as np
import pandas as pd
import statsmodels.api as sm

from Data_Generation import (
    generate_X,
    generate_error_baseline,
    generate_error_non_normal,
    generate_error_heteroskedastic,
    generate_error_autocorrelated,
    generate_Y
)

# Single Monte Carlo Replication
def run_single_simulation(
    n,
    scenario,
    beta0=2.0,
    beta1=3.0,
    sigma=1.0,
    df=3,
    gamma=0.5,
    rho=0.5,
    alpha=0.05,
    seed=None
):
    """
    Run a single Monte Carlo replication for a given scenario.
    """

    # Generate X (fixed within this replication)
    X = generate_X(n, seed=seed)

    # Generate error based on scenario
    if scenario == "baseline":
        eps = generate_error_baseline(n, sigma=sigma, seed=seed)

    elif scenario == "non_normal":
        eps = generate_error_non_normal(n, sigma=sigma, df=df, seed=seed)

    elif scenario == "heteroskedastic":
        eps = generate_error_heteroskedastic(X, sigma=sigma, gamma=gamma, seed=seed)

    elif scenario == "autocorrelated":
        eps = generate_error_autocorrelated(n, sigma=sigma, rho=rho, seed=seed)

    else:
        raise ValueError("Unknown scenario.")

    # Generate Y
    Y = generate_Y(X, eps, beta0=beta0, beta1=beta1)

    # OLS estimation
    X_ols = sm.add_constant(X)
    model = sm.OLS(Y, X_ols)
    results = model.fit()

    beta_hat = results.params[1]
    se_classic = results.bse[1]
    p_value = results.pvalues[1]

    # Classical confidence interval
    ci_low, ci_high = results.conf_int(alpha=alpha)[1]

    # Robust standard errors
    if scenario == "autocorrelated":
        se_robust = results.get_robustcov_results(
            cov_type="HAC",
            maxlags=1
        ).bse[1]
    else:
        se_robust = results.get_robustcov_results(
            cov_type="HC1"
        ).bse[1]

    reject = int(p_value < alpha)

    return {
        "beta_hat": beta_hat,
        "se_classic": se_classic,
        "se_robust": se_robust,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "reject": reject
    }

# Monte Carlo Loop
def run_monte_carlo_simulation(
    n,
    scenario,
    n_sim,
    beta0=2.0,
    beta1=3.0,
    sigma=1.0,
    df=3,
    gamma=0.5,
    rho=0.5,
    alpha=0.05,
    seed=None
):
    """
    Run Monte Carlo simulation for a given scenario and sample size.
    """

    results = []

    for r in range(n_sim):
        res = run_single_simulation(
            n=n,
            scenario=scenario,
            beta0=beta0,
            beta1=beta1,
            sigma=sigma,
            df=df,
            gamma=gamma,
            rho=rho,
            alpha=alpha,
            seed=None if seed is None else seed + r
        )
        results.append(res)

    df_results = pd.DataFrame(results)
    df_results["n"] = n
    df_results["scenario"] = scenario

    return df_results
