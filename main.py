from utils import make_prob_func, bracket_0
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = bracket_0()
    mh = MetropolisHastingsBracket(teams, make_prob_func(), False)
    X = mh.run(1000)
    print(mh.compute_mode())