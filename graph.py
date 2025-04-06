import matplotlib.pyplot as plt

from utils import seed_based_prob, sixtyfour_team_set, eight_team_set, sixteen_team_set
from mcmc import MetropolisHastingsBracket
import math

if __name__ == "__main__":
    #teams = sixtyfour_team_set()
    anneal = False
    if anneal:
        teams = sixtyfour_team_set()
        mh = MetropolisHastingsBracket(teams, seed_based_prob, simulate_anneal = True)
        res = mh.run(verbose=False)
        X, T, T_dict = res["X"], res["T"], res["T_dict"]
        print(X[-1].score())
        plt.plot([math.log(x.score()) for x in X])
        plt.text(plt.xlim()[1]*0.6, plt.ylim()[0]*0.95, f'Final Score: {X[-1].score():.2E}')
        plt.axvline(x=T_dict[int(T[0]/10)], color='r', linestyle='--', label=f'T={int(T[0]/10)}')
        plt.text(T_dict[int(T[0]/10)], plt.ylim()[1]*1.2, f'T={int(T[0]/10)}', rotation=90, verticalalignment='top', horizontalalignment='right')
        plt.axvline(x=T_dict[int(T[0]/1000)], color='r', linestyle='--', label=f'T={int(T[0]/1000)}')
        plt.text(T_dict[int(T[0]/1000)], plt.ylim()[1]*1.2, f'T={int(T[0]/1000)}', rotation=90, verticalalignment='top', horizontalalignment='right')
        plt.axvline(x=T_dict[int(T[0]/100000)], color='r', linestyle='--', label=f'T={int(T[0]/100000)}')
        plt.text(T_dict[int(T[0]/100000)], plt.ylim()[1]*1.2, f'T={int(T[0]/100000)}', rotation=90, verticalalignment='top', horizontalalignment='right')
        
        plt.ylabel("Score (Log Scaled)")
        plt.xlabel("Iterations")
        plt.title("Simulated Annealing of 64-Bracket Scores")
        plt.show()

        team_dict = [0] *16
        for _ in range(30):
            teams = sixtyfour_team_set()
            mh = MetropolisHastingsBracket(teams, seed_based_prob, simulate_anneal = True)
            res = mh.run(verbose=False)
            X, T, T_dict = res["X"], res["T"], res["T_dict"]
            for team in X[-1]._next_level._next_level._next_level._next_level.teams:
                team_dict[team.seed-1] += 1
            
        plt.bar(list(range(1,17)), [t/sum(team_dict) for t in team_dict], tick_label = list(range(1,17)))
        plt.ylabel("Proportion of time in Top-4")
        plt.xlabel("Seeds (1-16)")
        plt.title("Proportion of time seeds spend in Top-4 (30 Trials)")
        plt.show()
    else:
        iters = 400000
        teams = sixtyfour_team_set()
        mh = MetropolisHastingsBracket(teams, seed_based_prob)
        X = mh.run(iters=iters,verbose=False)
        best_score = sum([x.score() for x in X[-1000:]])/len(X[-1000:])
        def make_Xmean(X):
            res = []
            sumX = 0e0
            for i in range(len(X)):
                if i % 1000 == 0 and sumX != 0:
                    res.append(math.log(sumX) - math.log(1000))
                    sumX = 0
                else:
                    sumX += X[i].score()
                    print(sumX)
            return res
        X = make_Xmean(X)
        #X, T, T_dict = res["X"], res["T"], res["T_dict"]
        #print(X[-1].score())
        print(X[-1])
        #plt.plot([math.log(x.score()) for x in X])
        print(X)
        plt.plot(X)
        
        #plt.text(plt.xlim()[1]*0.6, plt.ylim()[0]*0.95, f'Final Score: {X[-1].score():.2E}')
        plt.text(plt.xlim()[1]*0.5, plt.ylim()[1]*1.05, f'Avg Score: {best_score:.2E}')
        plt.axvline(x=100, color='r', linestyle='--', label=f'100000 iterations')
        plt.text(100, plt.ylim()[0]*0.985, f'100000 iterations', rotation=0, verticalalignment='top', horizontalalignment='right')
        plt.axvline(x=220, color='r', linestyle='--', label=f'220000 iterations')
        plt.text(220, plt.ylim()[0]*0.985, f'220000 iterations', rotation=0, verticalalignment='top', horizontalalignment='right')
        plt.axvline(x=350, color='r', linestyle='--', label=f'350000 iterations')
        plt.text(350, plt.ylim()[0]*0.985, f'350000 iterations', rotation=00, verticalalignment='top', horizontalalignment='right')
        
        plt.ylabel("Score (Log Scaled)")
        plt.xlabel("Iterations (Mean of 1000 iter per point)")
        plt.title("MCMC of 64-Bracket Scores")
        plt.show()

        

