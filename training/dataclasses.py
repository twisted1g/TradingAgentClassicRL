from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:

    agent_name: str
    agent_type: str
    n_episodes: int
    max_steps: int

    learning_rate: float
    discount_factor: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    lambda_param: Optional[float] = None
    replace_traces: Optional[bool] = None

    initial_balance: float = 1000.0
    commission: float = 0.0001
    slippage: float = 0.0005

    eval_frequency: int = 100
    save_frequency: int = 500

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EpisodeMetrics:

    episode: int
    reward: float
    steps: int
    epsilon: float
    portfolio_value: float
    n_trades: int
    win_rate: float
    avg_pnl: float
    max_drawdown: float
    timestamp: float

    def to_dict(self) -> Dict:
        return asdict(self)
