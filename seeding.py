from random import sample
from copy import copy
from math import exp
from functools import partial

import numpy as np
from tqdm import tqdm
from joblib.parallel import delayed, Parallel

from common import Team, WinMatrix, Bracket
from mcmc import MetropolisHastingsBracket
from utils import bracket_idx_to_overall, expected_depth, bracket_0, naive_bracket

class Seeding:
    def __init__(self, teams: list[Team], win_matrix: WinMatrix):
        assert len(teams) == 64
        self.teams = teams
        self.seed: dict[Team, int] = {t: n for n, t in enumerate(self.teams)}
        self.win_matrix = win_matrix
        self.mlb = None
        self._score = None
        self.dist = None
    
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
    
    def mean_variance(self, iters: int = 100000, verbose: bool = True) -> float:
        mhb = MetropolisHastingsBracket(Seeding.arrange(self.teams), win_matrix=self.win_matrix, simulate_anneal=False)
        self.dist = [-i.depth_error() for i in mhb.run(iters=iters, verbose=verbose)]
        #print(np.mean(self.dist), np.var(self.dist)/12)
        return np.mean(self.dist) - np.std(self.dist)#return np.mean(self.dist) - np.var(self.dist)/12
    
    def score(self, iters: int = 6000, reps: int = 9, verbose: bool = False, exponential_score: bool = False) -> float:
        if self._score is not None:
            return self._score
        self._score = 0
        scores = Parallel(n_jobs=-1)(delayed(partial(self.mean_variance, iters=iters, verbose=verbose))() for _ in range(reps))
        self._score = sum(scores)
        #for _ in range(reps):
        #    self._score += self.mean_variance(iters=iters, verbose=verbose)
            #print(self._score/(_ + 1))
        self._score /= reps
        return self._score
    
    def random_transpose(self, limit: bool = True):
        idx1 = sample(range(len(self.teams)), 1)[0]
        idx2 = sample(range(max(0, idx1 - 5), min(idx1 + 5, len(self.teams))), 1)[0] if limit else sample(range(len(self.teams)), 1)[0]
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
        self.x0: Seeding = Seeding(Seeding.inverse_arrange(naive_bracket()), self.W)#Seeding(Seeding.inverse_arrange(bracket_0()), self.W) if seed_real else Seeding.RandomSeeding(self.teams, self.W)
        self.X: list[Seeding] = [self.x0]
        self.T = .2#.1#5#10#00
        self.alpha = 1.#0.999#.999
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
        self.T = max(self.alpha * self.T, 1e-50)
        if len(self.X) > 500:
            if sample([True, False], counts=[1, 500], k=1)[0]:
                if sample([True, False], counts=[20, 80], k=1)[0]:
                    return sample(self.X, k=1)[0]
                else:
                    for _ in range(sample(range(10), k=1)[0]):
                        i.random_transpose()
                    return i.random_transpose()
        delta = (j.score(exponential_score=False) - i.score(exponential_score=False))#p = (j.score()/i.score()) ** extremity
        print(exp(delta/self.T), j.score(exponential_score=False), i.score(exponential_score=False))
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