from typing import Callable
from functools import cache
from math import prod

class Team:
    def __init__(self, seed):
        self.seed = seed

class WinMatrix:
    def __init__(self, prob_func: Callable[[Team, Team], float]):
        self.prob_func = prob_func

    @cache
    def __getitem__(self, x: tuple[Team]) -> float:
        x1, x2 = x
        return self.prob_func(x1, x2)

def seed_prob(x1: Team, x2: Team) -> float:
    return 1. - (x1.seed)/(x1.seed + x2.seed)

class Bracket:
    def __init__(self, depth: int, teams: list[Team], win_matrix: WinMatrix):
        self.depth = depth
        assert len(teams) == 2 ** depth, "type `Bracket` must recieve 2 ** depth ({}) teams at index 0 of arg `teams` but received {} instead".format(2 ** depth, len(teams))
        self.teams = teams[0]
        if self.depth >= 1:
            self.next_level = Bracket(depth - 1, teams[1:])
        self.W = win_matrix
    
    def score(self) -> float:
        if self.depth == 0:
            return 1.
        return prod([self.W[winner, (t := self.teams[n*2: n*2 + 1])[not t.index(winner)]] for n, winner in enumerate(self.next_level.teams)]) * self.next_level.score()