import numpy as np
import pandas as pd
from pathlib import Path

from Simulation import run_monte_carlo_simulation
from Evaluation import summarize_results

# GLOBAL CONFIGURATION
SEED = 123
np.random.seed(SEED)

SAMPLE_SIZES = [30, 100, 500]
SCENARIOS = [
    "baseline",
    "non_normal",
    "heteroskedastic",
    "autocorrelated"
]

N_SIMULATIONS = 1000
ALPHA = 0.05

TRUE_BETA_0 = 1.0
TRUE_BETA_1 = 0.0   # H0 true → Type I error focus

# OUTPUT DIRECTORIES
RAW_PATH = Path("results/raw")
SUMMARY_PATH = Path("results/summary")

RAW_PATH.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.mkdir(parents=True, exist_ok=True)

# RUN MONTE CARLO SIMULATIONS
all_results = []

for n in SAMPLE_SIZES:
    for scenario in SCENARIOS:
        print(f"Running: n={n}, scenario={scenario}")

        sim_df = run_monte_carlo_simulation(
            n=n,
            scenario=scenario,
            n_sim=N_SIMULATIONS,
            beta0=TRUE_BETA_0,
            beta1=TRUE_BETA_1,
            alpha=ALPHA,
            seed=SEED
        )

        sim_df.to_csv(
            RAW_PATH / f"sim_n{n}_{scenario}.csv",
            index=False
        )

        all_results.append(sim_df)

# EVALUATION
results_df = pd.concat(all_results, ignore_index=True)

summary_df = summarize_results(
    results_df,
    beta_true=TRUE_BETA_1,
    alpha=ALPHA
)

summary_df.to_csv(
    SUMMARY_PATH / "summary_results.csv",
    index=False
)

print("\nSimulation completed successfully.")
print(summary_df)
