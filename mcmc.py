from typing import Callable
from copy import copy

import numpy as np
from tqdm import tqdm

from common import Team, Bracket, WinMatrix

class MetropolisHastingsBracket:
    def __init__(self, teams: list[Team], prob_func: Callable[[Team, Team], float]):
        self.teams = teams
        self.prob_func = prob_func
        self.W = WinMatrix(prob_func)
        self.seed: Bracket = Bracket.RandomBracket(self.teams, self.W)
        self.X: list[Bracket] = [self.seed]
    
    def _run_iter(self):
        b = self.X[-1]
        self.X.append(MetropolisHastingsBracket.accept(copy(b), copy(b).random_transpose()))
    
    def run(self, iters: int = 1000, verbose: bool = True):
        if verbose:
            for _ in (pbar := tqdm(range(iters))):
                self._run_iter()
                pbar.set_description_str("score: {}".format(self.X[-1].score()))
        else:
            for _ in range(iters):
                self._run_iter()
        return self.X
    
    def compute_mode(self) -> Bracket:
        mp = {}
        _c = {}
        for x in self.X:
            h = hash(x)
            if h in mp:
                _c[h] += 1
            else:
                mp[h] = x
                _c[h] = 1
        return mp[list(_c.keys())[np.argmax(list(_c.values()))]]
    
    @classmethod
    def accept(cls, i: Bracket, j: Bracket):
        p = j.score()/i.score()
        if p >= 1:
            return j
        return np.random.choice((j, i), p=(p, 1-p))