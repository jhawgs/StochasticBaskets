import matplotlib.pyplot as plt

from utils import seed_based_prob, sixtyfour_team_set
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = sixtyfour_team_set()
    mh = MetropolisHastingsBracket(teams, seed_based_prob)
    X = mh.run(1000)
    plt.plot([x.score() for x in X])
    plt.show()