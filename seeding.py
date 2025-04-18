from random import sample
from copy import copy
from math import exp

import numpy as np
from tqdm import tqdm

from common import Team, WinMatrix, Bracket
from mcmc import MetropolisHastingsBracket
from utils import bracket_idx_to_overall, expected_depth, bracket_0

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
        #result._score = None
        return result
    
    def __hash__(self):
        return hash(tuple(self.teams))

    def find_maximimum_likelihood_bracket(self, iters: int = 1500, verbose: bool = True) -> Bracket:
        mhb = MetropolisHastingsBracket(Seeding.arrange(self.teams), win_matrix=self.win_matrix, simulate_anneal=True)
        self.mlb = mhb.run(iters=iters, verbose=verbose)[-1]
        return self.mlb
    
    def score(self, iters: int = 1500, reps: int = 10, verbose: bool = False, exponential_score: bool = True) -> float:
        if self._score is not None:
            return self._score
        s = 0
        for _ in range(reps):
            self.find_maximimum_likelihood_bracket(iters=iters, verbose=verbose)
            depths = np.array([self.mlb.find_depth(i) for i in self.teams])
            error = np.sum(np.square(expected_depth - depths))
            s -= error
        s /= reps
        if exponential_score:
            s = exp(s)
        self._score = s
        #expected outcomes fails
        #matchups = self.find_maximimum_likelihood_bracket(iters=iters, verbose=verbose).build_matchups()
        #expected_outcomes = len(list(filter(lambda x: min(self.seed[x[0]], self.seed[x[1]]) == self.seed[x[2]], matchups)))
        #if exponential_score:
        #    expected_outcomes = exp(expected_outcomes)
        #self._score = expected_outcomes
        return self._score
    
    def random_transpose(self):
        idx1 = sample(range(len(self.teams)), 1)[0]
        idx2 = sample(range(len(self.teams)), 1)[0]
        t1 = self.teams[idx1]
        t2 = self.teams[idx2]
        self.teams[idx1] = t2
        self.teams[idx2] = t1
        self._score = None
        return self
    
    def prepare_pickle(self):
        if hasattr(self.win_matrix, "prob_func"):
            del self.win_matrix.prob_func
        if self.mlb is not None:
            self.mlb.prepare_pickle()
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
    def __init__(self, teams: list[Team], win_matrix: WinMatrix, seed_real: bool = False):
        self.teams = teams
        self.W = win_matrix
        self.x0: Seeding = Seeding(Seeding.inverse_arrange(bracket_0()), self.W) if seed_real else Seeding.RandomSeeding(self.teams, self.W)
        self.X: list[Seeding] = [self.x0]
        self.T = 1000
        self.alpha = 0.999#.999
        self.T_min = 1
    
    def _run_iter(self, anneal: bool = False):
        b = self.X[-1]
        if anneal:
            self.X.append(self.anneal_accept(copy(b), copy(b).random_transpose()))
        else:
            self.X.append(MetropolisHastingsSeedings.accept(copy(b), copy(b).random_transpose()))
    
    def run(self, iters: int = 1000, verbose: bool = True, anneal: bool = False):
        if verbose:
            for _ in (pbar := tqdm(range(iters))):
                self._run_iter(anneal=anneal)
                pbar.set_description_str("score: {}".format(self.X[-1].score()))
        else:
            for _ in range(iters):
                self._run_iter(anneal=anneal)
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
    
    def anneal_accept(self, i: Seeding, j: Seeding, extremity: float = 1) -> Seeding:
        self.T = self.alpha * self.T
        delta = (j.score(exponential_score=False) - i.score(exponential_score=False))#p = (j.score()/i.score()) ** extremity
        if delta > 0: #If j is better just send it
            #print(j._score, i._score)
            #print("send it")
            return j
        else: #Otherwise send j with high temps and don't with low temps
            u = np.random.uniform(0, 1, 1)[0]
            #print(f"Move to worse prob: {math.exp(delta/T)}")
            #print(exp(delta/self.T))
            if u < exp(delta/self.T):
                #print("accept")
                return j
            else:
                #print("reject")
                return i

    @classmethod
    def accept(cls, i: Seeding, j: Seeding, extremity: float = 1) -> Seeding:
        p = (j.score()/i.score()) ** extremity
        if p >= 1:
            return j
        return np.random.choice((j, i), p=(p, 1-p))