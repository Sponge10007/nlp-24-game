import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics_path: str, output_file: str) -> None:
    try:
        df = pd.read_csv(metrics_path)
    except FileNotFoundError:
        print(f"Missing metrics file: {metrics_path}")
        return

    if df.empty:
        print("Metrics file is empty.")
        return

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    has_reward = "mean_correctness_reward" in df.columns
    has_format = "format_rate" in df.columns
    error_cols = [
        col
        for col in (
            "missing_answer_count",
            "false_unsolvable_claim_count",
            "number_mismatch_count",
            "illegal_character_count",
            "syntax_error_count",
            "division_by_zero_count",
            "wrong_value_count",
            "fabricated_unsolvable_count",
        )
        if col in df.columns
    ]

    subplot_count = 1 + int(has_reward or has_format) + int(bool(error_cols))
    fig, axes = plt.subplots(subplot_count, 1, figsize=(12, 4 * subplot_count), sharex=True)
    if subplot_count == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(df["step"], df["batch_accuracy"], label="Batch Accuracy", color="lightgray", linewidth=1)
    ax.plot(df["step"], df["smoothed_accuracy"], label="Smoothed Accuracy (100 reward calls)", color="red", linewidth=2)
    ax.set_title("GRPO Training Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    axis_index = 1
    if has_reward or has_format:
        ax = axes[axis_index]
        if has_reward:
            ax.plot(df["step"], df["mean_correctness_reward"], label="Mean Correctness Reward", color="#2563eb", linewidth=2)
            ax.set_ylabel("Reward")
        if has_format:
            ax2 = ax.twinx()
            ax2.plot(df["step"], df["format_rate"], label="Format Rate", color="#16a34a", linewidth=1.5, alpha=0.8)
            ax2.set_ylabel("Format Rate (%)")
            ax2.set_ylim(-5, 105)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(loc="best")
        ax.set_title("Reward and Format Compliance")
        ax.grid(True, linestyle="--", alpha=0.5)
        axis_index += 1

    if error_cols:
        ax = axes[axis_index]
        for col in error_cols:
            ax.plot(df["step"], df[col], label=col.replace("_count", ""), linewidth=1.4)
        ax.set_title("Error Type Counts per Reward Call")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", ncol=2)

    axes[-1].set_xlabel("Reward Calls")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved curve to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRPO training metrics.")
    parser.add_argument("--metrics", default="training_metrics.csv")
    parser.add_argument("--output", default="accuracy_curve.png")
    args = parser.parse_args()
    plot_metrics(args.metrics, args.output)


if __name__ == "__main__":
    main()
