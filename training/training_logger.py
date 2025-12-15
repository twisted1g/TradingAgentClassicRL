import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from .dataclasses import TrainingConfig, EpisodeMetrics


class TrainingLogger:
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_metrics = []
        self.eval_results = []
        self.checkpoint_info = []

        self.episodes_csv = self.log_dir / "episodes.csv"
        self.eval_csv = self.log_dir / "evaluations.csv"
        self._init_csv_files()

    def _init_csv_files(self):
        if not self.episodes_csv.exists():
            pd.DataFrame(columns=list(EpisodeMetrics(0,0,0,0,0,0,0,0,0).to_dict().keys())
                        ).to_csv(self.episodes_csv, index=False)
        if not self.eval_csv.exists():
            pd.DataFrame(columns=["episode","mean_reward","std_reward",
                                  "min_reward","max_reward","mean_portfolio",
                                  "mean_trades","mean_steps","timestamp"]).to_csv(self.eval_csv, index=False)

    def log_episode(self, metrics: EpisodeMetrics):
        self.episode_metrics.append(metrics)
        pd.DataFrame([metrics.to_dict()]).to_csv(self.episodes_csv, mode="a", header=False, index=False)

    def log_evaluation(self, episode: int, eval_results: dict):
        eval_data = {"episode": episode, **eval_results, "timestamp": time.time()}
        self.eval_results.append(eval_data)
        pd.DataFrame([eval_data])[list(eval_data.keys())].to_csv(self.eval_csv, mode="a", header=False, index=False)

    def log_checkpoint(self, episode: int, path: str):
        self.checkpoint_info.append({"episode": episode, "path": path, "timestamp": time.time()})

    def save_summary(self, config: TrainingConfig, training_time: float):
        summary = {
            "config": config.to_dict(),
            "training_time": training_time,
            "training_time_minutes": training_time / 60,
            "total_episodes": len(self.episode_metrics),
            "final_metrics": self.episode_metrics[-1].to_dict() if self.episode_metrics else {},
            "checkpoints": self.checkpoint_info,
            "total_evaluations": len(self.eval_results),
            "timestamp": datetime.now().isoformat(),
        }
        
        summary_path = self.log_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        summary_text = f"""
{'='*80}
СВОДКА ОБУЧЕНИЯ
{'='*80}
Агент: {config.agent_name}
Эксперимент: {self.log_dir.name}
Время обучения: {training_time/60:.2f} минут
Всего эпизодов: {len(self.episode_metrics)}
Всего оценок: {len(self.eval_results)}
{'='*80}
"""
        with open(self.log_dir / "summary.txt", "w") as f:
            f.write(summary_text)