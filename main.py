from utils import make_prob_func, bracket_0
from mcmc import MetropolisHastingsBracket

if __name__ == "__main__":
    teams = bracket_0()
    mh = MetropolisHastingsBracket(teams, prob_func=make_prob_func(True), simulate_anneal=True)
    X = mh.run(1000)
    print(mh.X[-1])