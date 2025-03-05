from common import Team, WinMatrix

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

def seed_based_prob(x1: Team, x2: Team) -> float:
    return 1. - (x1.seed)/(x1.seed + x2.seed)

def seed_based_W() -> WinMatrix:
    return WinMatrix(seed_based_prob)