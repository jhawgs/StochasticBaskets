from typing import Callable, Optional
from copy import copy

import numpy as np
from tqdm import tqdm
import math

from common import Team, Bracket, WinMatrix

class MetropolisHastingsBracket:
    def __init__(self, teams: list[Team], prob_func: Optional[Callable[[Team, Team], float]] = None, win_matrix: Optional[WinMatrix] = None, simulate_anneal: bool = False):
        self.teams = teams
        self.prob_func = prob_func
        assert (prob_func is not None or win_matrix is not None) and not (prob_func is not None and win_matrix is not None)
        self.W = win_matrix or WinMatrix(prob_func)
        self.seed: Bracket = Bracket.RandomBracket(self.teams, self.W)
        self.X: list[Bracket] = [self.seed]
        self.simulate_anneal = simulate_anneal
        if self.simulate_anneal:
            self.T = 500000
            self.T_list = [self.T]
            self.alpha = 0.99
            self.T_min = 1
            self.T_dict = {}
    
    def _run_iter(self):
        b = self.X[-1]
        if not self.simulate_anneal:
            self.X.append(MetropolisHastingsBracket.accept(copy(b), copy(b).random_transpose()))
        else: #Simulated Annealing 
            self.X.append(MetropolisHastingsBracket.anneal_accept(copy(b), copy(b).random_transpose(), self.T))
    
    def run(self, iters: int = 1500, verbose: bool = True):
        if not self.simulate_anneal:
            if verbose:
                for _ in (pbar := tqdm(range(iters))):
                    self._run_iter()
                    pbar.set_description_str("score: {}".format(self.X[-1].score()))
            else:
                for _ in range(iters):
                    self._run_iter()
        else:
            T_dict = {
                int(self.T_list[0]/2): -1,
                int(self.T_list[0]/10): -1,
                int(self.T_list[0]/1000): -1,
                int(self.T_list[0]/100000): -1, 
            }
            while self.T > self.T_min:
                print(self.T)
                self._run_iter()
                self.T = self.alpha * self.T
                self.T_list.append(self.T)

                if self.T > int(self.T_list[0]/2):
                    T_dict[int(self.T_list[0]/2)] = len(self.T_list)
                if self.T > int(self.T_list[0]/10):
                    T_dict[int(self.T_list[0]/10)] = len(self.T_list)
                if self.T > int(self.T_list[0]/1000):
                    T_dict[int(self.T_list[0]/1000)] = len(self.T_list)
                if self.T > int(self.T_list[0]/100000):
                    T_dict[int(self.T_list[0]/100000)] = len(self.T_list)

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
    def accept(cls, i: Bracket, j: Bracket, extremity: float = 1) -> Bracket:
        p = (j.score()/i.score()) ** extremity
        if p >= 1:
            return j
        return np.random.choice((j, i), p=(p, 1-p))

    @classmethod
    def anneal_accept(cls, i: Bracket, j: Bracket, T: float) -> Bracket:
        delta = (j.score() - i.score())*1e20
        if delta > 0: #If j is better just send it
            return j
        else: #Otherwise send j with high temps and don't with low temps
            u = np.random.uniform(0, 1, 1)[0]
            #print(f"Move to worse prob: {math.exp(delta/T)}")
            if u < math.exp(delta/T):
                return j
            else:
                return i