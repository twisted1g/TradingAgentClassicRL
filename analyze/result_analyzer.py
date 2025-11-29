import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional


class ResultsAnalyzer:

    def __init__(self, log_base_dir: str = "training_data/logs"):
        self.log_base_dir = Path(log_base_dir)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (15, 10)

    def load_experiment(self, experiment_name: str) -> Dict:
        exp_dir = self.log_base_dir / experiment_name

        if not exp_dir.exists():
            raise ValueError(f"Эксперимент {experiment_name} не найден")

        with open(exp_dir / "config.json", "r") as f:
            config = json.load(f)

        with open(exp_dir / "training_summary.json", "r") as f:
            summary = json.load(f)

        episodes_df = pd.read_csv(exp_dir / "episodes.csv")

        eval_csv = exp_dir / "evaluations.csv"
        eval_df = pd.read_csv(eval_csv) if eval_csv.exists() else None

        return {
            "name": experiment_name,
            "config": config,
            "summary": summary,
            "episodes": episodes_df,
            "evaluations": eval_df,
            "dir": exp_dir,
        }

    def compare_experiments(
        self, experiment_names: List[str], save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Сравнение экспериментов", fontsize=16, fontweight="bold")

        colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_names)))

        for idx, exp_name in enumerate(experiment_names):
            exp_data = self.load_experiment(exp_name)
            episodes_df = exp_data["episodes"]

            color = colors[idx]
            label = exp_data["config"].get("agent_name", exp_name)

            axes[0, 0].plot(
                episodes_df["episode"], episodes_df["reward"], alpha=0.3, color=color
            )
            window = min(100, len(episodes_df) // 10)
            if len(episodes_df) >= window:
                rolling_mean = episodes_df["reward"].rolling(window=window).mean()
                axes[0, 0].plot(
                    episodes_df["episode"],
                    rolling_mean,
                    label=label,
                    color=color,
                    linewidth=2,
                )

            axes[0, 1].plot(
                episodes_df["episode"],
                episodes_df["portfolio_value"],
                label=label,
                color=color,
                alpha=0.7,
            )

            axes[0, 2].plot(
                episodes_df["episode"], episodes_df["epsilon"], label=label, color=color
            )

            if "win_rate" in episodes_df.columns:
                rolling_wr = episodes_df["win_rate"].rolling(window=window).mean()
                axes[1, 0].plot(
                    episodes_df["episode"], rolling_wr, label=label, color=color
                )

            if "n_trades" in episodes_df.columns:
                axes[1, 1].plot(
                    episodes_df["episode"],
                    episodes_df["n_trades"],
                    label=label,
                    color=color,
                    alpha=0.7,
                )

            if "max_drawdown" in episodes_df.columns:
                rolling_dd = episodes_df["max_drawdown"].rolling(window=window).mean()
                axes[1, 2].plot(
                    episodes_df["episode"], rolling_dd, label=label, color=color
                )

        axes[0, 0].set_title("Episode Reward (с скользящим средним)")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Portfolio Value")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Value ($)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].set_title("Epsilon Decay")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Epsilon")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].set_title("Win Rate (скользящее среднее)")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Win Rate (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Number of Trades")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Trades")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].set_title("Max Drawdown (скользящее среднее)")
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylabel("Drawdown")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"График сохранен: {save_path}")

        plt.show()

    def create_summary_table(self, experiment_names: List[str]) -> pd.DataFrame:
        results = []

        for exp_name in experiment_names:
            exp_data = self.load_experiment(exp_name)
            config = exp_data["config"]
            episodes_df = exp_data["episodes"]
            summary = exp_data["summary"]

            last_episodes = episodes_df.tail(100)

            result = {
                "Experiment": exp_name,
                "Agent": config.get("agent_name", "Unknown"),
                "Episodes": len(episodes_df),
                "Learning Rate": config.get("learning_rate", "-"),
                "Discount Factor": config.get("discount_factor", "-"),
                "Final Reward": episodes_df["reward"].iloc[-1],
                "Mean Reward (last 100)": last_episodes["reward"].mean(),
                "Std Reward (last 100)": last_episodes["reward"].std(),
                "Final Portfolio": episodes_df["portfolio_value"].iloc[-1],
                "Max Portfolio": episodes_df["portfolio_value"].max(),
                "Total Trades": (
                    last_episodes["n_trades"].sum()
                    if "n_trades" in last_episodes
                    else 0
                ),
                "Win Rate (last 100)": (
                    last_episodes["win_rate"].mean()
                    if "win_rate" in last_episodes
                    else 0
                ),
                "Training Time (min)": summary["training_time"] / 60,
            }

            results.append(result)

        df = pd.DataFrame(results)

        df["Final Reward"] = df["Final Reward"].round(2)
        df["Mean Reward (last 100)"] = df["Mean Reward (last 100)"].round(2)
        df["Final Portfolio"] = df["Final Portfolio"].round(2)
        df["Win Rate (last 100)"] = df["Win Rate (last 100)"].round(1)
        df["Training Time (min)"] = df["Training Time (min)"].round(2)

        return df

    def plot_single_experiment(
        self, experiment_name: str, save_path: Optional[str] = None
    ):
        exp_data = self.load_experiment(experiment_name)
        episodes_df = exp_data["episodes"]
        config = exp_data["config"]

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        agent_name = config.get("agent_name", experiment_name)
        fig.suptitle(f"Детальный анализ: {agent_name}", fontsize=16, fontweight="bold")

        ax1 = fig.add_subplot(gs[0, :])
        window = 100
        rolling_mean = episodes_df["reward"].rolling(window=window).mean()
        rolling_std = episodes_df["reward"].rolling(window=window).std()

        ax1.fill_between(
            episodes_df["episode"],
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.3,
            color="blue",
        )
        ax1.plot(
            episodes_df["episode"],
            rolling_mean,
            color="blue",
            linewidth=2,
            label=f"Mean (window={window})",
        )
        ax1.plot(
            episodes_df["episode"],
            episodes_df["reward"],
            alpha=0.2,
            color="gray",
            label="Raw reward",
        )
        ax1.set_title("Episode Reward")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            episodes_df["episode"],
            episodes_df["portfolio_value"],
            color="green",
            linewidth=2,
        )
        ax2.axhline(
            y=config.get("initial_balance", 1000),
            color="red",
            linestyle="--",
            label="Initial",
        )
        ax2.set_title("Portfolio Value")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Value ($)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        if "win_rate" in episodes_df.columns:
            rolling_wr = episodes_df["win_rate"].rolling(window=50).mean()
            ax3.plot(episodes_df["episode"], rolling_wr, color="purple", linewidth=2)
            ax3.set_title("Win Rate (rolling mean)")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Win Rate (%)")
            ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 2])
        if "n_trades" in episodes_df.columns:
            ax4.plot(
                episodes_df["episode"],
                episodes_df["n_trades"],
                color="orange",
                alpha=0.7,
            )
            ax4.set_title("Number of Trades per Episode")
            ax4.set_xlabel("Episode")
            ax4.set_ylabel("Trades")
            ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(
            episodes_df["episode"], episodes_df["epsilon"], color="red", linewidth=2
        )
        ax5.set_title("Epsilon Decay")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Epsilon")
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        last_rewards = episodes_df["reward"].tail(500)
        ax6.hist(last_rewards, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax6.axvline(
            last_rewards.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {last_rewards.mean():.2f}",
        )
        ax6.set_title("Reward Distribution (last 500 episodes)")
        ax6.set_xlabel("Reward")
        ax6.set_ylabel("Frequency")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 2])
        if "max_drawdown" in episodes_df.columns:
            rolling_dd = episodes_df["max_drawdown"].rolling(window=50).mean()
            ax7.plot(episodes_df["episode"], rolling_dd, color="darkred", linewidth=2)
            ax7.set_title("Max Drawdown (rolling mean)")
            ax7.set_xlabel("Episode")
            ax7.set_ylabel("Drawdown")
            ax7.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"График сохранен: {save_path}")

        plt.show()

    def list_experiments(self) -> List[str]:
        if not self.log_base_dir.exists():
            return []

        experiments = [
            d.name
            for d in self.log_base_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]

        return sorted(experiments)

    def print_experiment_info(self, experiment_name: str):
        exp_data = self.load_experiment(experiment_name)
        config = exp_data["config"]
        summary = exp_data["summary"]
        episodes_df = exp_data["episodes"]

        print(f"\n{'='*70}")
        print(f"Эксперимент: {experiment_name}")
        print(f"{'='*70}")
        print(f"\nКонфигурация:")
        print(f"  Agent: {config.get('agent_name')}")
        print(f"  Type: {config.get('agent_type')}")
        print(f"  Episodes: {config.get('n_episodes')}")
        print(f"  Learning Rate: {config.get('learning_rate')}")
        print(f"  Discount Factor: {config.get('discount_factor')}")
        print(f"  Epsilon: {config.get('epsilon_start')} → {config.get('epsilon_end')}")

        print(f"\nРезультаты:")
        print(f"  Время обучения: {summary['training_time']/60:.2f} минут")
        print(f"  Финальная награда: {episodes_df['reward'].iloc[-1]:.2f}")
        print(
            f"  Средняя награда (last 100): {episodes_df['reward'].tail(100).mean():.2f}"
        )
        print(f"  Финальный портфель: ${episodes_df['portfolio_value'].iloc[-1]:.2f}")
        print(f"  Максимальный портфель: ${episodes_df['portfolio_value'].max():.2f}")

        if "n_trades" in episodes_df:
            print(f"  Всего сделок: {episodes_df['n_trades'].sum()}")
            print(
                f"  Win Rate (last 100): {episodes_df['win_rate'].tail(100).mean():.1f}%"
            )

        print(f"{'='*70}\n")
