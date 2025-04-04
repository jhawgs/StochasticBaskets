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
    Team("Purdue", 1), Team("Texas Southern", 16), Team("Memphis", 8), Team("Florida Atlantic", 9),
    Team("Duke", 5), Team("Oral Roberts", 12), Team("Tennessee", 4), Team("Louisiana", 13),
    Team("Kentucky", 6), Team("Providence", 11), Team("Kansas State", 3), Team("Montana State", 14),
    Team("Michigan State", 7), Team("Southern California", 10), Team("Marquette", 2), Team("Vermont", 15),
    Team("Houston", 1), Team("Northern Kentucky", 16), Team("Iowa", 8), Team("Auburn", 9),
    Team("Miami (FL)", 5), Team("Drake", 12), Team("Indiana", 4), Team("Kent State", 13),
    Team("Iowa State", 6), Team("Mississippi State", 11), Team("Xavier", 3), Team("Kennesaw State", 14),
    Team("Texas A&M", 7), Team("Penn State", 10), Team("Texas", 2), Team("Colgate", 15),
    Team("Kansas", 1), Team("Howard", 16), Team("Arkansas", 8), Team("Illinois", 9),
    Team("Saint Mary's (CA)", 5), Team("Virginia Commonwealth", 12), Team("Connecticut", 4), Team("Iona", 13),
    Team("TCU", 6), Team("Nevada", 11), Team("Gonzaga", 3), Team("Grand Canyon", 14),
    Team("Northwestern", 7), Team("Boise State", 10), Team("UCLA", 2), Team("UNC Asheville", 15)
]

bracket_idx_to_overall = {0: 0, 1: 63, 2: 31, 3: 32, 4: 15, 5: 48, 6: 16, 7: 47, 8: 7, 9: 56, 10: 24, 11: 39, 12: 8, 13: 55, 14: 23, 15: 40, 16: 3, 17: 60, 18: 28, 19: 35, 20: 12, 21: 51, 22: 19, 23: 44, 24: 4, 25: 59, 26: 27, 27: 36, 28: 11, 29: 52, 30: 20, 31: 43, 32: 1, 33: 62, 34: 30, 35: 33, 36: 14, 37: 49, 38: 17, 39: 46, 40: 6, 41: 57, 42: 25, 43: 38, 44: 9, 45: 54, 46: 22, 47: 41, 48: 2, 49: 61, 50: 29, 51: 34, 52: 13, 53: 50, 54: 18, 55: 45, 56: 5, 57: 58, 58: 26, 59: 37, 60: 10, 61: 53, 62: 21, 63: 42}
expected_depth = np.array([6 - 0] + [6 - 1] + [6 - 2, 6 - 2] + [6 - 3, 6 - 3, 6 - 3, 6 - 3] + [6 - 4] * 8 + [6 - 5] * 16 + [6 - 6] * 32)