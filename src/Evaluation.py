import numpy as np
import pandas as pd
from scipy import stats

# Core Estimator Properties
def compute_bias(beta_hat, beta_true):
    """
    Bias = E(beta_hat) - beta_true
    """
    return np.mean(beta_hat) - beta_true

def compute_variance(beta_hat):
    """
    Variance of the estimator.
    """
    return np.var(beta_hat, ddof=1)

def compute_coverage(ci_low, ci_high, beta_true):
    """
    Coverage probability of confidence intervals.
    """
    covered = (ci_low <= beta_true) & (beta_true <= ci_high)
    return np.mean(covered)

def compute_type1_error(rejects):
    """
    Type I error rate.
    """
    return np.mean(rejects)

# Robust Inference Utilities
def compute_robust_ci(beta_hat, se, alpha=0.05):
    """
    Confidence interval using normal approximation
    (appropriate for robust standard errors).
    """
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low = beta_hat - z * se
    ci_high = beta_hat + z * se
    return ci_low, ci_high

def compute_robust_rejection(beta_hat, se, beta_true, alpha=0.05):
    """
    Hypothesis test using robust standard errors.
    """
    z_stat = (beta_hat - beta_true) / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    return (p_values < alpha).astype(int)

# Monte Carlo Summary
def summarize_results(
    df,
    beta_true,
    alpha=0.05
):
    """
    Aggregate Monte Carlo results and compute evaluation metrics.

    Expected columns in df:
    - beta_hat
    - se_classic
    - se_robust
    - ci_low
    - ci_high
    - reject
    - n
    - scenario
    """

    summaries = []

    grouped = df.groupby(["scenario", "n"])

    for (scenario, n), g in grouped:
        beta_hat = g["beta_hat"].values

        # Classical inference
        ci_low = g["ci_low"].values
        ci_high = g["ci_high"].values
        reject_classic = g["reject"].values

        # Robust inference
        se_robust = g["se_robust"].values
        ci_low_rob, ci_high_rob = compute_robust_ci(
            beta_hat, se_robust, alpha=alpha
        )
        reject_robust = compute_robust_rejection(
            beta_hat, se_robust, beta_true, alpha=alpha
        )

        summary = {
            "scenario": scenario,
            "n": n,
            "bias": compute_bias(beta_hat, beta_true),
            "variance": compute_variance(beta_hat),
            "coverage_classic": compute_coverage(ci_low, ci_high, beta_true),
            "coverage_robust": compute_coverage(ci_low_rob, ci_high_rob, beta_true),
            "type1_error_classic": compute_type1_error(reject_classic),
            "type1_error_robust": compute_type1_error(reject_robust)
        }

        summaries.append(summary)

    return pd.DataFrame(summaries)
