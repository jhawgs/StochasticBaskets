from typing import Callable
from copy import copy

import numpy as np
from tqdm import tqdm
import math

from common import Team, Bracket, WinMatrix

class MetropolisHastingsBracket:
    def __init__(self, teams: list[Team], prob_func: Callable[[Team, Team], float], simulate_anneal: bool):
        self.teams = teams
        self.prob_func = prob_func
        self.W = WinMatrix(prob_func)
        self.seed: Bracket = Bracket.RandomBracket(self.teams, self.W)
        self.X: list[Bracket] = [self.seed]
        self.simulate_anneal = simulate_anneal
        if self.simulate_anneal:
            self.T = 100000
            self.alpha = 0.9
            self.T_min = 1
    
    def _run_iter(self):
        b = self.X[-1]
        if not self.simulate_anneal:
            self.X.append(MetropolisHastingsBracket.accept(copy(b), copy(b).random_transpose()))
        else: #Simulated Annealing 
            self.X.append(MetropolisHastingsBracket.anneal_accept(copy(b), copy(b).random_transpose(), self.T))
    
    def run(self, iters: int = 1000, verbose: bool = True):
        if not self.simulate_anneal:
            if verbose:
                for _ in (pbar := tqdm(range(iters))):
                    self._run_iter()
                    pbar.set_description_str("score: {}".format(self.X[-1].score()))
            else:
                for _ in range(iters):
                    self._run_iter()
        else:
            if verbose:
                while self.T > self.T_min:
                    self._run_iter()
                    self.T = self.alpha * self.T
                    print(self.X[-1].score())
            else:
                while self.T > self.T_min:
                    self._run_iter()
                    self.T = self.alpha * self.T
        return self.X
    
    def compute_mode(self, burnin: int = 0) -> Bracket:
        mp = {}
        _c = {}
        for x in self.X[burnin:]:
            h = hash(x)
            if h in mp:
                _c[h] += 1
            else:
                mp[h] = x
                _c[h] = 1
        return mp[list(_c.keys())[np.argmax(list(_c.values()))]]
    
    @classmethod
    def accept(cls, i: Bracket, j: Bracket, extremity: float = 1):
        p = (j.score()/i.score()) ** extremity
        if p >= 1:
            return j
        return np.random.choice((j, i), p=(p, 1-p))

    @classmethod
    def anneal_accept(cls, i: Bracket, j: Bracket, T: float):
        delta = (j.score() - i.score())*1e20
        if delta > 0: #If j is better just send it
            return j
        else: #Otherwise send j with high temps and don't with low temps
            u = np.random.uniform(0, 1, 1)[0]
            print(f"Move to worse prob: {math.exp(delta/T)}")
            if u < math.exp(delta/T):
                return j
            else:
                return i