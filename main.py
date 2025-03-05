from utils import seed_based_prob, four_team_set
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = four_team_set()
    mh = MetropolisHastingsBracket(teams, seed_based_prob)
    X = mh.run()
    print(*X[-10:], sep="\n")