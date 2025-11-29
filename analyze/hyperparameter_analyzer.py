import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


class HyperparameterAnalyzer:
    def __init__(self, results_dir: str = "training_data/logs/hyperparameter_search"):
        self.results_dir = Path(results_dir)

    def load_search_results(self, filename: str) -> pd.DataFrame:
        filepath = self.results_dir / filename
        return pd.read_csv(filepath)

    def plot_hyperparameter_impact(
        self,
        results_df: pd.DataFrame,
        param_name: str,
        metric: str = "mean_reward",
        save_path: Optional[str] = None,
    ):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.scatter(
            results_df[param_name],
            results_df[metric],
            s=100,
            alpha=0.6,
            c=results_df[metric],
            cmap="viridis",
        )
        ax1.set_xlabel(param_name)
        ax1.set_ylabel(metric)
        ax1.set_title(f"{metric} vs {param_name}")
        ax1.grid(True, alpha=0.3)

        if results_df[param_name].nunique() < 10:
            results_df.boxplot(column=metric, by=param_name, ax=ax2)
            ax2.set_title(f"{metric} distribution by {param_name}")
            ax2.set_xlabel(param_name)
            ax2.set_ylabel(metric)
        else:
            sorted_df = results_df.sort_values(param_name)
            ax2.plot(sorted_df[param_name], sorted_df[metric], "o-", markersize=8)
            ax2.set_xlabel(param_name)
            ax2.set_ylabel(metric)
            ax2.set_title(f"{metric} trend by {param_name}")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def create_heatmap(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = "mean_reward",
        save_path: Optional[str] = None,
    ):

        pivot_table = results_df.pivot_table(
            values=metric, index=param1, columns=param2, aggfunc="mean"
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            cbar_kws={"label": metric},
        )
        plt.title(f"{metric} Heatmap: {param1} vs {param2}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
