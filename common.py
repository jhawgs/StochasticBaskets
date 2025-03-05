from typing import Callable
from functools import cache
from copy import copy
from math import prod, log2
from random import choice

class Team:
    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
    
    def __str__(self) -> str:
        return self.name

class WinMatrix:
    def __init__(self, prob_func: Callable[[Team, Team], float]):
        self.prob_func = prob_func

    @cache
    def __getitem__(self, x: tuple[Team]) -> float:
        x1, x2 = x
        return self.prob_func(x1, x2)

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
    
    def score(self) -> float:
        if self.depth == 0:
            return 1.
        return prod([self.W[winner, (t := self.teams[n*2: n*2 + 2])[not t.index(winner)]] for n, winner in enumerate(self._next_level.teams)]) * self._next_level.score()
    
    def _recursive_apply_transpose(self, old_winner, candidate1, candidate2):
        new_winner = choice((candidate1, candidate2))
        if old_winner in self.teams:
            idx = self.teams.index(old_winner)
            self.teams.remove(old_winner)
            self.teams.insert(idx, new_winner)
            if self._next_level is not None:
                gs = idx//2
                self._next_level._recursive_apply_transpose(old_winner, *self.teams[2*gs: 2*gs + 2])
    
    def transpose_game(self, idx: int):
        game = self.teams[idx*2: idx*2 + 2]
        old_winner = self._next_level.teams[idx]
        new_winner = game[not game.index(old_winner)]
        self._next_level._recursive_apply_transpose(old_winner, new_winner, new_winner)

    
    def random_transpose(self):
        game = choice(self.games)
        level = self
        while level.depth > game["depth"]:
            level = level._next_level
        assert level.depth == game["depth"], "failed to find level with depth {}, `Bracket` object is malformed".format(game["depth"])
        level.transpose_game(game["idx"])
        return self
    
    @classmethod
    def RandomBracket(cls, teams: list[Team], win_matrix: WinMatrix):
        depth = int(log2(len(teams)))
        assert 2 ** depth == len(teams), "arg `teams` must have a length of a power 2 but has length {}".format(len(teams))
        _teams = [teams]
        while len(_teams[-1]) > 1:
            _teams.append([choice(_teams[-1][n*2:n*2+2]) for n in range(int(len(_teams[-1])/2))])
        return cls(depth, _teams, win_matrix) 