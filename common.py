from typing import Callable
from functools import cache

class Team:
    def __init__(self, seed):
        self.seed = seed

class WinProb:
    def __init__(self, prob_func: Callable[[Team, Team], float]):
        self.prob_func = prob_func

    @cache
    def __getitem__(self, x: tuple[Team]) -> float:
        x1, x2 = x
        return self.prob_func(x1, x2)

def seed_prob(x1: Team, x2: Team) -> float:
    return 1. - (x1.seed)/(x1.seed + x2.seed)