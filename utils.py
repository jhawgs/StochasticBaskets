from common import Team, WinMatrix
from prob import make_prob_func

def four_team_set() -> list[Team]:
    return [
        Team("Team 1", 1),
        Team("Team 4", 4),
        Team("Team 2", 2),
        Team("Team 3", 3),
    ]

def eight_team_set() -> list[Team]:
    return [
        Team("Team 1", 1),
        Team("Team 8", 8),
        Team("Team 4", 4),
        Team("Team 5", 5),
        Team("Team 3", 3),
        Team("Team 6", 6),
        Team("Team 2", 2),
        Team("Team 7", 7),
    ]

def sixteen_team_set(suffix="") -> list[Team]:
    return [
        Team("Team 1" + suffix, 1),
        Team("Team 16" + suffix, 16),
        Team("Team 8" + suffix, 8),
        Team("Team 9" + suffix, 9),
        Team("Team 4" + suffix, 4),
        Team("Team 13" + suffix, 13),
        Team("Team 12" + suffix, 12),
        Team("Team 5" + suffix, 5),
        Team("Team 6" + suffix, 6),
        Team("Team 11" + suffix, 11),
        Team("Team 3" + suffix, 3),
        Team("Team 14" + suffix, 14),
        Team("Team 7" + suffix, 7),
        Team("Team 10" + suffix, 10),
        Team("Team 2" + suffix, 2),
        Team("Team 15" + suffix, 7),
    ]

def sixtyfour_team_set() -> list[Team]:
    return sum([sixteen_team_set(" ({})".format(i + 1)) for i in range(4)], start=[])

def seed_based_prob(x1: Team, x2: Team) -> float:
    return 1. - (x1.seed)/(x1.seed + x2.seed)

def seed_based_W() -> WinMatrix:
    return WinMatrix(seed_based_prob)

def rfc_W() -> WinMatrix:
    return WinMatrix(make_prob_func())