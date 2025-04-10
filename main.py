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
        mh.W.save()
    elif sys.argv[1] == "seed-optimize":
        from utils import make_prob_func, bracket_0
        from common import WinMatrix
        from seeding import MetropolisHastingsSeedings
        try:
            mh = MetropolisHastingsSeedings(bracket_0(), win_matrix=WinMatrix(make_prob_func()), seed_real=True)
            X = mh.run(20000, anneal=True)#mh.run(100000)
        except KeyboardInterrupt:
            pass
        with open("./seeding_optim.pkl", "wb") as doc:
            pickle.dump([i.prepare_pickle() for i in mh.X], doc)
        print(mh.X[-1])#(mh.compute_mode())
        print(mh.X[-1].mlb)
        mh.W.save()
    elif sys.argv[1] == "seed-optimize-resume":
        from utils import make_prob_func, bracket_0
        from common import WinMatrix
        from seeding import MetropolisHastingsSeedings
        W = WinMatrix(make_prob_func())
        try:
            mh = MetropolisHastingsSeedings(bracket_0(), win_matrix=W, seed_real=True)
            with open("./seeding_optim.pkl", "wb") as doc:
                mh.X = pickle.load(doc)
            for i in mh.X:
                i.win_matrix = W
            X = mh.run(20000, anneal=True)#mh.run(100000)
        except KeyboardInterrupt:
            pass
        with open("./seeding_optim.pkl", "wb") as doc:
            pickle.dump([i.prepare_pickle() for i in mh.X], doc)
        print(mh.X[-1])#(mh.compute_mode())
        print(mh.X[-1].mlb)
        mh.W.save()
    elif sys.argv[1] == "seed-sample":
        from utils import make_prob_func, bracket_0
        from common import WinMatrix
        from seeding import MetropolisHastingsSeedings
        try:
            mh = MetropolisHastingsSeedings(bracket_0(), win_matrix=WinMatrix(make_prob_func()))
            X = mh.run(30000)#mh.run(100000)
        except KeyboardInterrupt:
            pass
        with open("./seeding_sample.pkl", "wb") as doc:
            pickle.dump([i.prepare_pickle() for i in mh.X], doc)
        print(mh.X[-1])#(mh.compute_mode())
        print(mh.X[-1].mlb)
        mh.W.save()
    elif sys.argv[1] == "tyler-bracket":
        pass