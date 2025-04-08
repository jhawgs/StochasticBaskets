from typing import Callable, Optional
from copy import copy
from math import prod, log2
from random import choice
from pickle import load, dump
import os
import numpy as np
bracket_idx_to_overall = {0: 0, 32: 1, 48: 2, 16: 3, 62: 4, 46: 5, 14: 6, 30: 7, 10: 8, 58: 9, 26: 10, 42: 11, 54: 12, 22: 13, 38: 14, 6: 15, 4: 16, 20: 17, 52: 18, 36: 19, 40: 20, 8: 21, 24: 22, 56: 23, 44: 24, 28: 25, 12: 26, 60: 27, 18: 28, 50: 29, 2: 30, 34: 31, 19: 32, 3: 33, 35: 34, 51: 35, 61: 36, 45: 37, 29: 38, 13: 39, 9: 40, 25: 41, 41: 42, 57: 43, 5: 44, 21: 45, 37: 46, 53: 47, 39: 48, 55: 49, 7: 50, 23: 51, 43: 52, 11: 53, 59: 54, 27: 55, 31: 56, 47: 57, 15: 58, 63: 59, 17: 60, 49: 61, 33: 62, 1: 63}
expected_depth = np.array([6 - 0] + [6 - 1] + [6 - 2, 6 - 2] + [6 - 3, 6 - 3, 6 - 3, 6 - 3] + [6 - 4] * 8 + [6 - 5] * 16 + [6 - 6] * 32)

CACHE = "./cache.pkl"

class Team:
    def __init__(self, name: str, seed: int, id: Optional[str] = None):
        self.name = name
        self.seed = seed
        self.id = id or name
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self):
        return hash(self.id)

def inverse_arrange(teams: list[Team]) -> list[Team]:
        return [teams[{v: k for k, v in bracket_idx_to_overall.items()}[i]] for i in range(64)]

class WinMatrix:
    def __init__(self, prob_func: Callable[[Team, Team], float]):
        self.prob_func = prob_func
        self.cache = {}
        self.load()

    def __getitem__(self, x: tuple[Team, Team]) -> float:
        x1, x2 = x
        if (i := (x1.id, x2.id)) not in self.cache:
            self.cache[i] = result = self.prob_func(x1, x2)
            return result
        else:
            return self.cache[i]
    
    def load(self):
        if os.path.isfile(CACHE):
            with open(CACHE, "rb") as doc:
                self.cache = load(doc)
    
    def save(self):
        with open(CACHE, "wb") as doc:
            dump(self.cache, doc)

class Bracket:
    def __init__(self, depth: int, teams: list[list[Team]], win_matrix: WinMatrix):
        self.depth = depth
        self.W = win_matrix
        assert len(teams[0]) == 2 ** depth, "type `Bracket` must recieve 2 ** depth ({}) teams at index 0 of arg `teams` but received {} instead".format(2 ** depth, len(teams[0]))
        self.teams = teams[0]
        if self.depth >= 1:
            self._next_level = Bracket(depth - 1, teams[1:], self.W)
        else:
            self._next_level = None
        self.games = [g for i in range(depth) for g in [{"depth": i + 1, "idx": n} for n in range(int(2 ** i))]]
        if self.depth == 6:
            self.seed: dict[Team, int] = {t: n for n, t in enumerate(inverse_arrange(self.teams))}
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._next_level = copy(result._next_level)
        result.teams = copy(result.teams)
        return result
    
    def __eq__(self, x) -> bool:
        if type(x) != Bracket:
            return False
        return all([i1 == x1 for i1, x1 in zip(self.teams, x.teams)]) and self._next_level == x._next_level

    def __str__(self) -> str:
        return " ".join([str(i) for i in self.teams]) + ("\n" + str(self._next_level) if self._next_level is not None else "")
    
    def __hash__(self):
        return hash(tuple(self.recursive_teams()))
    
    def prepare_pickle(self):
        if hasattr(self.W, "prob_func"):
            del self.W.prob_func
        if self._next_level is not None:
            self._next_level.prepare_pickle()
        return self
    
    def recursive_teams(self) -> list[Team]:
        if self._next_level is None:
            return self.teams
        return self.teams + self._next_level.recursive_teams()
    
    def score(self) -> float:
        if self.depth == 0:
            return 1.
        return prod([self.W[winner, (t := self.teams[n*2: n*2 + 2])[not t.index(winner)]] for n, winner in enumerate(self._next_level.teams)]) * self._next_level.score()
    
    def _recursive_apply_transpose(self, old_winner: Team, new_winner: Team):# needs to just paste in winner for all loser spots
        if old_winner in self.teams:
            idx = self.teams.index(old_winner)
            self.teams.remove(old_winner)
            self.teams.insert(idx, new_winner)
            if self._next_level is not None:
                self._next_level._recursive_apply_transpose(old_winner, new_winner)
    
    def transpose_game(self, idx: int):
        game = self.teams[idx*2: idx*2 + 2]
        old_winner = self._next_level.teams[idx]
        new_winner = game[not game.index(old_winner)]
        self._next_level._recursive_apply_transpose(old_winner, new_winner)
    
    def random_transpose(self):
        game = choice(self.games)
        level = self
        while level.depth > game["depth"]:
            level = level._next_level
        assert level.depth == game["depth"], "failed to find level with depth {}, `Bracket` object is malformed".format(game["depth"])
        level.transpose_game(game["idx"])
        return self
    
    def build_matchups(self) -> tuple[tuple[Team, Team, Team]]:
        if self.depth == 0:
            return tuple()
        return tuple(zip(self.teams[::2], self.teams[1::2], self._next_level.teams)) + self._next_level.build_matchups()
    
    def depth_error(self) -> int:
        depths = np.array([self.find_depth(i) for i in inverse_arrange(self.teams)])
        error = np.sum(np.square(expected_depth - depths))
        return error

    def count_expected(self) -> int:
        matchups = self.build_matchups()
        expected_outcomes = len(list(filter(lambda x: min(self.seed[x[0]], self.seed[x[1]]) == self.seed[x[2]], matchups)))
        return expected_outcomes
    
    def find_depth(self, team: Team) -> int:
        if self.depth == 0 or team not in self.teams:
            return 0
        else:
            return 1 + self._next_level.find_depth(team)
    
    @classmethod
    def NaiveBracket(cls, teams: list[Team], win_matrix: WinMatrix):
        depth = int(log2(len(teams)))
        assert 2 ** depth == len(teams), "arg `teams` must have a length of a power 2 but has length {}".format(len(teams))
        _teams = [teams]
        while len(_teams[-1]) > 1:
            _teams.append([_teams[-1][n*2:n*2+2][win_matrix[_teams[-1][n*2], _teams[-1][n*2+1]] < .5] for n in range(int(len(_teams[-1])/2))])
        return cls(depth, _teams, win_matrix)
    
    @classmethod
    def RandomBracket(cls, teams: list[Team], win_matrix: WinMatrix):
        depth = int(log2(len(teams)))
        assert 2 ** depth == len(teams), "arg `teams` must have a length of a power 2 but has length {}".format(len(teams))
        _teams = [teams]
        while len(_teams[-1]) > 1:
            _teams.append([choice(_teams[-1][n*2:n*2+2]) for n in range(int(len(_teams[-1])/2))])
        return cls(depth, _teams, win_matrix)