from utils import seed_based_prob, sixtyfour_team_set
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = sixtyfour_team_set()
    mh = MetropolisHastingsBracket(teams, seed_based_prob)
    X = mh.run(750000)
    print(mh.compute_mode(burnin=50000))