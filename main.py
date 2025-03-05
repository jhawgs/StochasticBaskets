from utils import seed_based_prob, eight_team_set
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = eight_team_set()
    mh = MetropolisHastingsBracket(teams, seed_based_prob)
    X = mh.run()
    print(mh.compute_mode())