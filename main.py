from utils import seed_based_prob, sixteen_team_set
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = sixteen_team_set()
    mh = MetropolisHastingsBracket(teams, seed_based_prob)
    X = mh.run(10000)
    print(mh.compute_mode())