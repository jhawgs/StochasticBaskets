from os.path import isfile
from pickle import load, dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from data import wrap_build

data = pd.read_csv("./fulltenyears.csv")
data = pd.concat((data, data.rename({c: c.replace(c[-1], "1" if c[-1] == "0" else "0") for c in data.columns}, axis=1))).drop(["Unnamed: 0", "Unnamed: 1"], axis=1)
data["favwin01"] = pd.concat((data["favwin01"].dropna(), (~data["favwin00"].dropna().astype(bool)).astype("float")))
data["_constant"] = 1

if isfile("./keys.txt"):
  keys = eval(open("keys.txt", "r").read())

def fit_models():
    rfc = RandomForestClassifier().fit(data.drop("favwin01", axis=1)[keys], data["favwin01"])
    with open("./rfc.pkl", "wb") as doc:
        dump(rfc, doc)

    lr = LogisticRegression(max_iter=100000).fit(data.drop("favwin01", axis=1)[keys], data["favwin01"])
    with open("./lr.pkl", "wb") as doc:
        dump(lr, doc)

def in_memory_rfc(data):
    return RandomForestClassifier().fit(data.drop("favwin01", axis=1)[keys], data["favwin01"])

def load_model(rfc=True):
    if rfc:
        with open("./rfc.pkl", "rb") as doc:
            return load(doc)
    else:
        with open("./lr.pkl", "rb") as doc:
            return load(doc)

def make_prob_func(rfc=True):
    m = load_model(rfc=rfc)
    def f(x1, x2):
        return m.predict_proba([wrap_build(x1.name, x2.name)])[0][1]
    return f

if __name__ == "__main__":
    fit_models()