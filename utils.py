import numpy as np

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

def bracket_0() -> list[Team]:
    return [
    Team("Alabama", 1), Team("Texas A&M-Corpus Christi", 16), Team("Maryland", 8), Team("West Virginia", 9),
    Team("San Diego State", 5), Team("College of Charleston", 12), Team("Virginia", 4), Team("Furman", 13),
    Team("Creighton", 6), Team("NC State", 11), Team("Baylor", 3), Team("UC Santa Barbara", 14),
    Team("Missouri", 7), Team("Utah State", 10), Team("Arizona", 2), Team("Princeton", 15),
    Team("Purdue", 1), Team("Fairleigh Dickinson", 16), Team("Memphis", 8), Team("Florida Atlantic", 9),
    Team("Duke", 5), Team("Oral Roberts", 12), Team("Tennessee", 4), Team("Louisiana", 13),
    Team("Kentucky", 6), Team("Providence", 11), Team("Kansas State", 3), Team("Montana State", 14),
    Team("Michigan State", 7), Team("Southern California", 10), Team("Marquette", 2), Team("Vermont", 15),
    Team("Houston", 1), Team("Northern Kentucky", 16), Team("Iowa", 8), Team("Auburn", 9),
    Team("Miami (FL)", 5), Team("Drake", 12), Team("Indiana", 4), Team("Kent State", 13),
    Team("Iowa State", 6), Team("Pittsburgh", 11), Team("Xavier", 3), Team("Kennesaw State", 14),
    Team("Texas A&M", 7), Team("Penn State", 10), Team("Texas", 2), Team("Colgate", 15),
    Team("Kansas", 1), Team("Howard", 16), Team("Arkansas", 8), Team("Illinois", 9),
    Team("Saint Mary's (CA)", 5), Team("Virginia Commonwealth", 12), Team("Connecticut", 4), Team("Iona", 13),
    Team("TCU", 6), Team("Arizona State", 11), Team("Gonzaga", 3), Team("Grand Canyon", 14),
    Team("Northwestern", 7), Team("Boise State", 10), Team("UCLA", 2), Team("UNC Asheville", 15)
]

def naive_bracket() -> list[Team]:
    return [Team("Houston", 1), Team("Fairleigh Dickinson", 16), Team("Miami (FL)", 8), Team("Iowa State", 9), Team("Virginia", 5), Team("Iowa", 12), Team("Indiana", 4), Team("Iona", 13), Team("Kentucky", 6), Team("Furman", 11), Team("Texas", 3), Team("UNC Asheville", 14), Team("Texas A&M", 7), Team("Memphis", 10), Team("College of Charleston", 2), Team("Howard", 15), Team("Marquette", 1), Team("Princeton", 16), Team("Utah State", 8), Team("Duke", 9), Team("Oral Roberts", 5), Team("TCU", 12), Team("Alabama", 4), Team("Auburn", 13), Team("Arizona", 6), Team("Kent State", 11), Team("San Diego State", 3), Team("Missouri", 14), Team("West Virginia", 7), Team("Michigan State", 10), Team("Kansas State", 2), Team("Texas A&M-Corpus Christi", 15), Team("Saint Mary's (CA)", 1), Team("Kennesaw State", 16), Team("UC Santa Barbara", 8), Team("Xavier", 9), Team("Boise State", 5), Team("Louisiana", 12), Team("Kansas", 4), Team("Penn State", 13), Team("Florida Atlantic", 6), Team("Illinois", 11), Team("Tennessee", 3), Team("Pittsburgh", 14), Team("Virginia Commonwealth", 7), Team("Southern California", 10), Team("Connecticut", 2), Team("Grand Canyon", 15), Team("UCLA", 1), Team("Montana State", 16), Team("Northwestern", 8), Team("NC State", 9), Team("Maryland", 5), Team("Drake", 12), Team("Gonzaga", 4), Team("Vermont", 13), Team("Creighton", 6), Team("Arizona State", 11), Team("Baylor", 3), Team("Northern Kentucky", 14), Team("Arkansas", 7), Team("Providence", 10), Team("Purdue", 2), Team("Colgate", 15)]

def tyler_bracket() -> list[Team]:
    return [
        Team("Team 1", 0),
        Team("Team 2", 1),
        Team("Team 3", 2),
        Team("Team 4", 3)
    ]

def tyler_prob_func(x1: Team, x2: Team) -> float:
    return [[0.50, 0.49, 0.99, 0.98],
     [0.51, 0.50, 0.48, 0.47],
     [0.01, 0.52, 0.50, 0.46],
     [0.02, 0.53, 0.54, 0.50]][x1.seed][x2.seed]

bracket_idx_to_overall = {0: 0, 32: 1, 48: 2, 16: 3, 62: 4, 46: 5, 14: 6, 30: 7, 10: 8, 58: 9, 26: 10, 42: 11, 54: 12, 22: 13, 38: 14, 6: 15, 4: 16, 20: 17, 52: 18, 36: 19, 40: 20, 8: 21, 24: 22, 56: 23, 44: 24, 28: 25, 12: 26, 60: 27, 18: 28, 50: 29, 2: 30, 34: 31, 19: 32, 3: 33, 35: 34, 51: 35, 61: 36, 45: 37, 29: 38, 13: 39, 9: 40, 25: 41, 41: 42, 57: 43, 5: 44, 21: 45, 37: 46, 53: 47, 39: 48, 55: 49, 7: 50, 23: 51, 43: 52, 11: 53, 59: 54, 27: 55, 31: 56, 47: 57, 15: 58, 63: 59, 17: 60, 49: 61, 33: 62, 1: 63}
expected_depth = np.array([6 - 0] + [6 - 1] + [6 - 2, 6 - 2] + [6 - 3, 6 - 3, 6 - 3, 6 - 3] + [6 - 4] * 8 + [6 - 5] * 16 + [6 - 6] * 32)