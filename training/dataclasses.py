from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import time


@dataclass
class TrainingConfig:
    agent_name: str = "QLearning"
    agent_type: str = "QLearning"
    n_episodes: int = 5000
    n_episodes_start: int = 0
    max_steps: int = 1000
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9998
    
    initial_balance: float = 1000.0
    window_size: int = 10
    commission: float = 0.0001
    slippage: float = 0.0001
    max_holding_time: int = 72
    max_drawdown_threshold: float = 0.08
    
    eval_frequency: int = 200
    save_frequency: int = 1000
    patience: int = 10
    seed: int = 42
    
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }


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
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

