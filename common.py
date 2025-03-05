from typing import Callable
from functools import cache
from math import prod, log2
from random import choice

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
    
    def score(self) -> float:
        if self.depth == 0:
            return 1.
        return prod([self.W[winner, (t := self.teams[n*2: n*2 + 2])[not t.index(winner)]] for n, winner in enumerate(self._next_level.teams)]) * self._next_level.score()
    
    def transpose_game(self, idx: int):
        game = self.teams[idx*2: idx*2 + 2]
        self._next_level.teams[idx] = game[not game.index(self._next_level.teams[idx])]
    
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