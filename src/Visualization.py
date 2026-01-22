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

# GIF

# CONFIGURATION 
SUMMARY_FILE = Path("results/summary/summary_results.csv")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "baseline",
    "non_normal",
    "heteroskedastic",
    "autocorrelated"
]

METRICS = {
    "bias": {
        "column": "bias",
        "ylabel": "Bias",
        "title": "Bias of OLS Estimator"
    },
    "variance": {
        "column": "variance",
        "ylabel": "Variance",
        "title": "Variance of OLS Estimator"
    },
    "type1_error_classic": {
        "column": "type1_error_classic",
        "ylabel": "Type I Error Rate",
        "title": "Type I Error (Classical SE)",
        "hline": 0.05
    },
    "type1_error_robust": {
        "column": "type1_error_robust",
        "ylabel": "Type I Error Rate",
        "title": "Type I Error (Robust SE)",
        "hline": 0.05
    },
    "coverage_classic": {
        "column": "coverage_classic",
        "ylabel": "Coverage Probability",
        "title": "Coverage Probability (Classical CI)",
        "hline": 0.95
    },
    "coverage_robust": {
        "column": "coverage_robust",
        "ylabel": "Coverage Probability",
        "title": "Coverage Probability (Robust CI)",
        "hline": 0.95
    }
}

# LOAD DATA
df = pd.read_csv(SUMMARY_FILE)

# GIF CREATION FUNCTION
def create_metric_gif(metric_name, metric_info):
    frames = []

    for scenario in SCENARIOS:
        data = df[df["scenario"] == scenario].sort_values("n")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            data["n"],
            data[metric_info["column"]],
            marker="o",
            linewidth=2
        )

        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel(metric_info["ylabel"])
        ax.set_title(f"{metric_info['title']}\nScenario: {scenario}")

        if "hline" in metric_info:
            ax.axhline(
                metric_info["hline"],
                linestyle="--",
                linewidth=1
            )

        ax.grid(True, alpha=0.3)

        frame_path = OUTPUT_DIR / f"_temp_{metric_name}_{scenario}.png"
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close()

        frames.append(imageio.imread(frame_path))
        frame_path.unlink()  # clean temporary file

    gif_path = OUTPUT_DIR / f"{metric_name}.gif"
    imageio.mimsave(gif_path, frames, duration=1.5)

    print(f"Saved: {gif_path}")

# MAIN LOOP
for metric_name, metric_info in METRICS.items():
    create_metric_gif(metric_name, metric_info)

print("\nAll GIF visualizations created successfully.")
