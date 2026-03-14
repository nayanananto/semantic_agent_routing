# bandit.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

class LinUCB:
    """
    Contextual bandit for routing optimization.
    Actions = agent names.
    Context = prompt embedding vector.
    """
    def __init__(self, actions: List[str], dim: int, alpha: float = 1.0):
        self.actions = actions
        self.dim = dim
        self.alpha = alpha
        self.A: Dict[str, np.ndarray] = {a: np.eye(dim, dtype=np.float32) for a in actions}
        self.b: Dict[str, np.ndarray] = {a: np.zeros((dim,), dtype=np.float32) for a in actions}

    def score(self, action: str, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A[action])
        theta = A_inv @ self.b[action]
        exploit = float(x @ theta)
        explore = float(self.alpha * np.sqrt(x @ A_inv @ x))
        return exploit + explore

    def choose(self, x: np.ndarray, candidate_actions: Optional[List[str]] = None) -> str:
        pool = candidate_actions if candidate_actions else self.actions
        best_a = None
        best_score = -1e18
        for a in pool:
            s = self.score(a, x)
            if s > best_score:
                best_score = s
                best_a = a
        return best_a if best_a is not None else pool[0]

    def update(self, action: str, x: np.ndarray, reward: float) -> None:
        # A += x x^T ; b += reward x
        x = x.reshape(-1)
        self.A[action] += np.outer(x, x).astype(np.float32)
        self.b[action] += (reward * x).astype(np.float32)
