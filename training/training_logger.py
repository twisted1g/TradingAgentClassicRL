import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
from .dataclasses import TrainingConfig, EpisodeMetrics


class TrainingLogger:
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_metrics: List[EpisodeMetrics] = []
        self.eval_results: List[Dict] = []
        self.checkpoint_info: List[Dict] = []

        self.episodes_csv = self.log_dir / "episodes.csv"
        self.eval_csv = self.log_dir / "evaluations.csv"

        self._init_csv_files()

    def _init_csv_files(self):
        if not self.episodes_csv.exists():
            pd.DataFrame(
                columns=[
                    "episode",
                    "reward",
                    "steps",
                    "epsilon",
                    "portfolio_value",
                    "n_trades",
                    "win_rate",
                    "avg_pnl",
                    "max_drawdown",
                    "timestamp",
                ]
            ).to_csv(self.episodes_csv, index=False)

        if not self.eval_csv.exists():
            pd.DataFrame(
                columns=[
                    "episode",
                    "mean_reward",
                    "std_reward",
                    "min_reward",
                    "max_reward",
                    "mean_length",
                    "timestamp",
                ]
            ).to_csv(self.eval_csv, index=False)

    def log_episode(self, metrics: EpisodeMetrics):
        self.episode_metrics.append(metrics)

        df = pd.DataFrame([metrics.to_dict()])
        df.to_csv(self.episodes_csv, mode="a", header=False, index=False)

    def log_evaluation(self, episode: int, eval_results: Dict):
        eval_data = {"episode": episode, "timestamp": time.time(), **eval_results}
        self.eval_results.append(eval_data)

        df = pd.DataFrame([eval_data])
        df.to_csv(self.eval_csv, mode="a", header=False, index=False)

    def log_checkpoint(self, episode: int, checkpoint_path: str):
        self.checkpoint_info.append(
            {"episode": episode, "path": checkpoint_path, "timestamp": time.time()}
        )

    def save_summary(self, config: TrainingConfig, training_time: float):
        summary = {
            "config": config.to_dict(),
            "training_time": training_time,
            "total_episodes": len(self.episode_metrics),
            "final_metrics": (
                self.episode_metrics[-1].to_dict() if self.episode_metrics else {}
            ),
            "checkpoints": self.checkpoint_info,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.log_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Логи сохранены в: {self.log_dir}")
