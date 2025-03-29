import sys

if __name__ == "__main__":
    if sys.argv[1] == "bracket":
        from utils import make_prob_func, bracket_0
        from mcmc import MetropolisHastingsBracket
        teams = bracket_0()
        mh = MetropolisHastingsBracket(teams, prob_func=make_prob_func(True), simulate_anneal=True)
        X = mh.run(1000)
        print(X[-1])
    else:
        from seeding import MetropolisHastingsSeedings
        mh = MetropolisHastingsSeedings()
        X = mh.run(1000)
        print(mh.compute_mode())