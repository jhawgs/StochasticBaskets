import pandas as pd
from sportsipy.ncaab.teams import Teams
from os.path import isfile
from numpy import array
from pickle import load
import warnings
warnings.filterwarnings('ignore')

with open("./teams.pkl", "rb") as doc:
    TEAMS = load(doc)

#@title Config { run: "auto" }
method = ["tournament", "matchups", "sgd_gravity", "save"][-1]
plot = False #@param {type:"boolean"}
equal01 = False #@param {type:"boolean"}
addpg = False #@param {type:"boolean"}
replicate = False #@param {type:"boolean"}
secondrd = False #@param {type:"boolean"}
configyear = 2023 #@param {type:"integer"}
samplesize = 900 #@param {type:"integer"}
use_bank = False #@param {type:"boolean"}
team_bank = [
  "Gonzaga",
  "Georgia State",
  "Boise State",
  "Memphis"
]
fav2und2result = {}
SGD_NATIVE = False

#game of form (perspective name, game object)

def none_replace(l, val=0):
  if len(l) == 0:
    return l
  else:
    try:
      int(l[0])
      return [i if i is not None else val for i in l] if type(l) is list else array([i if i is not None else val for i in l])
    except:
      return [[i if i is not None else val for i in list(l[0])]] if type(l[0]) is list else [array([i if i is not None else val for i in list(l[0])])]

def get_team_data(team, year, teams=None):
  team = [i for i in filter(lambda e: e.name == team, teams)][0]
  return (team.points/team.games_played, team.opp_points/team.games_played, team.strength_of_schedule)

def make_row(team1, team2, year, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(year)
  team1 = [i for i in filter(lambda e: e.name == team1, teams)][0]
  team2 = [i for i in filter(lambda e: e.name == team2, teams)][0]
  ref1 = find_other_perspective((team1.name, team1.schedule[-1]))
  ref2 = find_other_perspective((team2.name, team2.schedule[-1]))
  if (ref1.opponent_rank if not ref1.opponent_rank is None else ((get_team_data(ref1.opponent_name, 0, teams=teams)[2] * -1) if ref2.opponent_rank is None else 100)) > (ref2.opponent_rank if not ref2.opponent_rank is None else ((get_team_data(ref2.opponent_name, 0, teams=teams)[2] * -1)) if ref1.opponent_rank is None else 100):
    fav = team2
    und = team1
  else:
    fav = team1
    und = team2
  fav = get_team_data(fav.name, year, teams=teams)
  und = get_team_data(und.name, year, teams=teams)
  return [fav[0], und[0], fav[1], und[1], fav[2], und[2]]

def bracket_parse(path):
  bracket = pd.read_excel(path)
  bracketlist = []
  for i in bracket:
    bracketlist.append([i for i in filter(lambda e: type(e) == type(""), bracket[i])])
  return bracketlist

def game_in(game, all):
  for i in all:
    if game[0] == i[1].opponent_name and i[0] == game[1].opponent_name:
      return True
  return False

def build_tourney(year, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(year)
  tourney = []
  for i in teams:
    for n in filter(lambda e: e.type == "NCAA", i.schedule):
      if not game_in((i.name, n), tourney):
        tourney.append((i.name, n))
  return tourney

def build_row(game, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(game[1].datetime.year)
  other = find_other_perspective(game, teams=teams)
  if (game[1].opponent_rank if not game[1].opponent_rank is None else ((get_team_data(game[1].opponent_name, 0, teams=teams)[2] * -1) if other.opponent_rank is None else 100)) < (other.opponent_rank if not other.opponent_rank is None else ((get_team_data(other.opponent_name, 0, teams=teams)[2] * -1) if game[1].opponent_rank is None else 100)):
    fav = game[1].opponent_name
    favwin = 0 if game[1].result == "Win" else 1
    und = game[0]
  else:
    fav = game[0]
    favwin = 1 if game[1].result == "Win" else 0
    und = game[1].opponent_name
  row = [favwin]
  row.append([i for i in filter(lambda e: e.name == fav, teams)][0].points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == und, teams)][0].points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == fav, teams)][0].opp_points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == und, teams)][0].opp_points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
  row.append([i for i in filter(lambda e: e.name == fav, teams)][0].strength_of_schedule)
  row.append([i for i in filter(lambda e: e.name == und, teams)][0].strength_of_schedule)
  return row

def build_row_known(fav, und, teams=None, again=False):
  
  row = []
  try:
    row.append([i for i in filter(lambda e: e.name == fav, teams)][0].points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == und, teams)][0].points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == fav, teams)][0].opp_points/[i for i in filter(lambda e: e.name == fav, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == und, teams)][0].opp_points/[i for i in filter(lambda e: e.name == und, teams)][0].games_played)
    row.append([i for i in filter(lambda e: e.name == fav, teams)][0].strength_of_schedule)
    row.append([i for i in filter(lambda e: e.name == und, teams)][0].strength_of_schedule)
  except Exception as err:
    if again:
      print("FAVUND",fav, und)
      print("TEAMS", [i.name for i in teams])
      raise err
    else:
      return build_row_known(fav.replace("-", " "), und.replace("-", " "), teams=teams, again=True)
  return row

"""
def build_tourney_data(year):
  teams = Teams(year)
  tourney_data = []
  for i in build_tourney(year, teams=teams):
    tourney_data.append(build_row(i, teams=teams))
  df = pd.DataFrame()
  df["favwin01"] = [i[0] for i in tourney_data]
  df["ppgfav"] = [i[1] for i in tourney_data]
  df["ppgund"] = [i[2] for i in tourney_data]
  df["papgfav"] = [i[3] for i in tourney_data]
  df["papgund"] = [i[4] for i in tourney_data]
  df["sosfav"] = [i[5] for i in tourney_data]
  df["sosund"] = [i[6] for i in tourney_data]
  return df
"""

def find_other_perspective(game, teams=None): #tournament only
  for i in teams:
    if i.name == game[1].opponent_name:
      for n in filter(lambda e: e.type == "NCAA", i.schedule):
        if game_in((i.name, n), [game]):
          return n

"""
def build_tourneys_data(years):
  all = pd.DataFrame()
  for i in years:
    all = pd.concat([all, build_tourney_data(i)])
  return all
"""
  
def equalize(data):
  while True:
    num_zeros = (data["favwin01"].values == 0).sum()
    num_ones = (data["favwin01"].values == 1).sum()
    lowest = min(num_ones, num_zeros)
    counts = [0, 0]
    i = 0
    favin = int([i for i in data.columns].index("favwin01"))
    new = pd.DataFrame()
    while i < len(data["favwin01"].values) - 1:
      i += 1
      try:
        counts[int(data.loc[i][favin])] += 1
        if counts[int(data.loc[i][favin])] <= lowest:
          new = pd.concat([new, pd.DataFrame(data.loc[i])], axis=1)
      except KeyError:
        pass
    new = new.transpose()
    num_zeros = (new["favwin01"].values == 0).sum()
    num_ones = (new["favwin01"].values == 1).sum()
    try:
      assert abs(num_zeros - num_ones) <= 2, str(num_zeros) + " " + str(num_ones)
      break
    except AssertionError:
      data = new
  return new

def add_pg(df):
  df["points_per_game0"] = [df["points0"].values[i]/df["games_played0"].values[i] for i in range(len(df["points0"].values))]
  df["points_per_game1"] = [df["points1"].values[i]/df["games_played1"].values[i] for i in range(len(df["points0"].values))]
  df["points_allowed_per_game0"] = [df["opp_points0"].values[i]/df["games_played0"].values[i] for i in range(len(df["points0"].values))]
  df["points_allowed_per_game1"] = [df["opp_points1"].values[i]/df["games_played1"].values[i] for i in range(len(df["points0"].values))]
  return df

def get_fields(year, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(year)
  fields = []
  r = 0
  for n in teams:
    r = n
    break
  for i in r.dataframe:
    try:
      int(r.dataframe[i][0])
      fields.append(i)
    except:
      pass
  return fields

def get_col_data(cols, suf, team, year, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(year)
  df = None
  for i in filter(lambda e: e.name == team, teams):
    df = i.dataframe[cols]
    df.columns = [str(n) + str(suf) for n in df.columns]
    return df

def splice(df1, df2):
  n = 0
  for _ in df1.iterrows():
    n += 1
  df1.index = [str(i) for i in range(n)]
  df2.index = [str(i) for i in range(n)]
  return pd.concat([df1, df2], axis=1)

def build_full_row(game, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(game[1].datetime.year)
  other = find_other_perspective(game, teams=teams)
  if (game[1].opponent_rank if not game[1].opponent_rank is None else ((get_team_data(game[1].opponent_name, 0, teams=teams)[2] * -1) if other.opponent_rank is None else 100)) < (other.opponent_rank if not other.opponent_rank is None else ((get_team_data(other.opponent_name, 0, teams=teams)[2] * -1) if game[1].opponent_rank is None else 100)):
    fav = game[1].opponent_name
    favwin = 0 if game[1].result == "Win" else 1
    und = game[0]
  else:
    fav = game[0]
    favwin = 1 if game[1].result == "Win" else 0
    und = game[1].opponent_name
  row = splice(get_col_data(get_fields(configyear, teams=teams), "0", und, configyear, teams=teams), get_col_data(get_fields(configyear, teams=teams), "1", fav, configyear, teams=teams))
  row["favwin01"] = favwin
  return row

def build_full_tourney_data(year):
  teams = Teams(year)
  tourney_data = pd.DataFrame()
  n = 0
  for i in build_tourney(year, teams=teams):
    new = build_full_row(i, teams=teams)
    new.index = [n]
    n += 1
    tourney_data = pd.concat([tourney_data, new], axis=0)
  return tourney_data

def build_full_tourneys_data(years):
  all = pd.DataFrame()
  for i in years:
    all = pd.concat([all, build_full_tourney_data(i)], axis=0)
  return all

if isfile("./fulltenyears.csv"):
  fulldata = pd.read_csv("./fulltenyears.csv")
  if addpg:
    try:
      fulldata["points_per_game0"]
    except:
      fulldata = add_pg(fulldata)
  else:
    try:
      fulldata["points_per_game0"]
      fulldata = fulldata.drop("points_per_game0", axis=1)
      fulldata = fulldata.drop("points_per_game1", axis=1)
      fulldata = fulldata.drop("points_allowed_per_game0", axis=1)
      fulldata = fulldata.drop("points_allowed_per_game1", axis=1)
    except:
      pass
  savefull = fulldata
  #fulldata = fulldata.drop("")
else:
  fulldata = build_full_tourneys_data(range(2010, 2020))
  if addpg:
    fulldata = add_pg(fulldata)
  savefull = fulldata
fulldata["_constant"] = [1 for i in range(len(fulldata["favwin01"]))]
if equal01:
  fulldata = equalize(fulldata)
#fulldata = sm.add_constant(fulldata)
fulldata.head()

if not isfile("./fulltenyears.csv"):
  fulldata.to_csv("./fulltenyears.csv")

fullXtrain = fulldata.drop("favwin01", axis=1)
try:
  fullXtrain = fullXtrain.drop("Unnamed: 0", axis=1)
except Exception as err:
  print(err)
while True:
  try:
    fullXtrain = fullXtrain.drop("_constant")
  except:
    break
fullYtrain = fulldata[["favwin01"]]
fullXtrain.head()

if isfile("./keys.txt"):
  keys = eval(open("keys.txt", "r").read())

def build_matchup(fav, und, teams=None, keys=keys): #could cause problems
  fav, und = get_fav(fav, und, teams=teams)
  row = splice(get_col_data(get_fields(configyear, teams=teams), "0", und[1], configyear, teams=teams), get_col_data(get_fields(configyear, teams=teams), "1", fav[1], configyear, teams=teams))
  row["_constant"] = [1]
  row = add_pg(row)
  row = row[keys].values
  return row

def get_fav(fav, und, teams=None):
  if fav[0] > und[0]:
    return und, fav
  else:
    return fav, und

def populate_bracket(r1, predict, data, year=configyear, teams=None):
  if teams is None:
    print("TEAMS NONE")
    teams = Teams(year)
  rounds = [r1]
  remaining = r1
  while len(remaining) > 1:
    victors = []
    for i in range(int(len(remaining)/2)):
      try:
        result = fav2und2result[data][predict][remaining[i * 2]][remaining[(i * 2) + 1]]
        print("found", end="")
      except:
        result = remaining[i * 2] if round(predict(none_replace(data(remaining[i * 2], remaining[(i * 2) + 1], teams=teams)))) == 1 else remaining[(i * 2) + 1]
        if data not in fav2und2result:
          fav2und2result[data] = {}
          fav2und2result[data][predict] = {}
          fav2und2result[data][predict][remaining[i * 2]] = {}
          fav2und2result[data][predict][remaining[i * 2]][remaining[(i * 2) + 1]] = result
        elif predict not in fav2und2result[data]:
          fav2und2result[data][predict] = {}
          fav2und2result[data][predict][remaining[i * 2]] = {}
          fav2und2result[data][predict][remaining[i * 2]][remaining[(i * 2) + 1]] = result
        elif remaining[i * 2] not in fav2und2result[data][predict]:
          fav2und2result[data][predict][remaining[i * 2]] = {}
          fav2und2result[data][predict][remaining[i * 2]][remaining[(i * 2) + 1]] = result
        else:
          fav2und2result[data][predict][remaining[i * 2]][remaining[(i * 2) + 1]] = result
      victors.append(result)
    remaining = victors
    rounds.append(victors)
  return rounds

def wrap_build(fav, und, teams=None):
  return build_row_known(fav[1], und[1], teams=teams)

def load_bracket(path):
  with open(path, "r") as doc:
    l = eval(doc.read())
  return l

def build_combo_row(fav, und, teams=None):
  fulldata = build_matchup(fav, und, teams=teams)
  return fulldata
 
def wrap_build(fav, und, teams=TEAMS):
  return build_combo_row((1, fav), (2, und), teams)[0]

if __name__ == "__main__":
    with open("./lr.pkl", "rb") as doc:
        lr = load(doc)
    print(lr.predict_proba([wrap_build("Gonzaga", "Kansas")]))
    print(lr.predict_proba([wrap_build("Kansas", "Gonzaga")]))
    print(lr.predict_proba([wrap_build("Kansas", "Abilene Christian")]))
    with open("./rfc.pkl", "rb") as doc:
        rfc = load(doc)
    print(rfc.predict_proba([wrap_build("Gonzaga", "Kansas")]))
    print(rfc.predict_proba([wrap_build("Kansas", "Gonzaga")]))
    print(rfc.predict_proba([wrap_build("Kansas", "Abilene Christian")]))