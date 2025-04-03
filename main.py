import sys
import pickle

if __name__ == "__main__":
    if sys.argv[1] == "bracket":
        from utils import make_prob_func, bracket_0
        from mcmc import MetropolisHastingsBracket
        teams = bracket_0()
        mh = MetropolisHastingsBracket(teams, prob_func=make_prob_func(True), simulate_anneal=True)
        X = mh.run(1500)
        print(X[-1])
    else:
        from utils import make_prob_func, bracket_0
        from common import WinMatrix
        from seeding import MetropolisHastingsSeedings
        mh = MetropolisHastingsSeedings(bracket_0(), win_matrix=WinMatrix(make_prob_func()))
        X = mh.run(3000, anneal=True)#mh.run(100000)
        with open("./seeding_results.pkl", "wb") as doc:
            pickle.dump([i.prepare_pickle() for i in X], doc)
        print(mh.compute_mode())