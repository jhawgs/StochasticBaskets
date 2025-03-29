from random import sample
from copy import copy

import numpy as np
from tqdm import tqdm

from common import Team, WinMatrix, Bracket
from mcmc import MetropolisHastingsBracket
from utils import bracket_idx_to_overall

class Seeding:
    def __init__(self, teams: list[Team], win_matrix: WinMatrix):
        assert len(teams) == 64
        self.teams = teams
        self.seed: dict[Team, int] = {t: n for n, t in enumerate(self.teams)}
        self.win_matrix = win_matrix
        self.mlb = None
        self._score = None
    
    def __str__(self) -> str:
        return "\n".join(map(lambda x: "(" + str(1 + x[0]//4) + ") " + x[1].name, enumerate(self.teams)))
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.teams = copy(result.teams)
        result._score = None
        return result
    
    def __hash__(self):
        return hash(tuple(self.teams))

    def find_maximimum_likelihood_bracket(self, iters: int = 1000, verbose: bool = True) -> Bracket:
        mhb = MetropolisHastingsBracket(Seeding.arrange(self.teams), win_matrix=self.win_matrix, simulate_anneal=True)
        self.mlb = mhb.run(iters=iters, verbose=verbose)[-1]
        return self.mlb
    
    def score(self, iters: int = 1000, verbose: bool = True) -> float:
        if self._score is not None:
            return self._score
        matchups = self.find_maximimum_likelihood_bracket(iters=iters, verbose=verbose).build_matchups()
        expected_outcomes = len(list(filter(lambda x: min(self.seed[x[0]], self.seed[x[1]]) == self.seed[x[2]], matchups)))
        self._score = expected_outcomes
        return expected_outcomes
    
    def random_transpose(self):
        idx1 = sample(range(len(self.teams)), 1)[0]
        idx2 = sample(range(len(self.teams)), 1)[0]
        t1 = self.teams[idx1]
        t2 = self.teams[idx2]
        self.teams[idx1] = t2
        self.teams[idx2] = t1
        return self
    
    @classmethod
    def arrange(cls, teams: list[Team]) -> list[Team]:
        return [teams[bracket_idx_to_overall[i]] for i in range(64)]
    
    @classmethod
    def inverse_arrange(cls, teams: list[Team]) -> list[Team]:
        return [teams[{v: k for k, v in bracket_idx_to_overall.items()}[i]] for i in range(64)]
    
    @classmethod
    def RandomSeeding(cls, teams: list[Team], win_matrix: WinMatrix):
        return cls(sample(teams, len(teams)), win_matrix)

class MetropolisHastingsSeedings:
    def __init__(self, teams: list[Team], win_matrix: WinMatrix):
        self.teams = teams
        self.W = win_matrix
        self.x0: Seeding = Seeding.RandomSeeding(self.teams, self.W)
        self.X: list[Seeding] = [self.x0]
    
    def _run_iter(self):
        b = self.X[-1]
        self.X.append(MetropolisHastingsSeedings.accept(copy(b), copy(b).random_transpose()))
    
    def run(self, iters: int = 1000, verbose: bool = True):
        if verbose:
            for _ in (pbar := tqdm(range(iters))):
                self._run_iter()
                pbar.set_description_str("score: {}".format(self.X[-1].score()))
        else:
            for _ in range(iters):
                self._run_iter()
        return self.X
    
    def compute_mode(self, burnin: int = 0) -> Seeding:
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
    def accept(cls, i: Seeding, j: Seeding, extremity: float = 1) -> Seeding:
        p = (j.score()/i.score()) ** extremity
        if p >= 1:
            return j
        return np.random.choice((j, i), p=(p, 1-p))