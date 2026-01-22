import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

# CONFIGURATION
SUMMARY_PATH = Path("results/summary/summary_results.csv")
FIGURE_PATH = Path("results/figures")

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
TYPE1_TARGET = ALPHA
COVERAGE_TARGET = 1 - ALPHA

# LOAD DATA
df = pd.read_csv(SUMMARY_PATH)

# Rename for clarity (optional but cleaner)
df = df.rename(columns={
    "error_type": "scenario"
})

# HELPER FUNCTION
def plot_metric(
    df,
    metric,
    ylabel,
    filename,
    reference_line=None
):
    """
    Generic plotting function for Monte Carlo summary metrics.
    """
    plt.figure(figsize=(8, 5))

    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario]
        plt.plot(
            subset["n"],
            subset[metric],
            marker="o",
            label=scenario
        )

    if reference_line is not None:
        plt.axhline(
            reference_line,
            linestyle="--",
            linewidth=1
        )

    plt.xlabel("Sample Size (n)")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Sample Size")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / filename, dpi=300)
    plt.close()

# 1. TYPE I ERROR
plot_metric(
    df,
    metric="type1_error_classic",
    ylabel="Type I Error (Classic SE)",
    filename="type1_error_classic.png",
    reference_line=TYPE1_TARGET
)

plot_metric(
    df,
    metric="type1_error_robust",
    ylabel="Type I Error (Robust SE)",
    filename="type1_error_robust.png",
    reference_line=TYPE1_TARGET
)

# 2. COVERAGE PROBABILITY
plot_metric(
    df,
    metric="coverage_classic",
    ylabel="Coverage Probability (Classic CI)",
    filename="coverage_classic.png",
    reference_line=COVERAGE_TARGET
)

plot_metric(
    df,
    metric="coverage_robust",
    ylabel="Coverage Probability (Robust CI)",
    filename="coverage_robust.png",
    reference_line=COVERAGE_TARGET
)

# 3. BIAS
plot_metric(
    df,
    metric="bias",
    ylabel="Bias of β̂₁",
    filename="bias.png",
    reference_line=0.0
)

# 4. VARIANCE
plot_metric(
    df,
    metric="variance",
    ylabel="Variance of β̂₁",
    filename="variance.png"
)

print("Visualization completed successfully.")
print(f"Figures saved in: {FIGURE_PATH.resolve()}")
