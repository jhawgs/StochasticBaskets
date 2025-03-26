from os.path import isfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from pickle import load, dump

data = pd.read_csv("./fulltenyears.csv")
if isfile("./keys.txt"):
  keys = eval(open("keys.txt", "r").read())

def fit_models():
    rfc = RandomForestClassifier().fit(data.drop("favwin01", axis=1)[keys], data["favwin01"])
    with open("./rfc.pkl", "wb") as doc:
        dump(rfc, doc)

    lr = LogisticRegression(max_iter=100000).fit(data.drop("favwin01", axis=1)[keys], data["favwin01"])
    with open("./lr.pkl", "wb") as doc:
        dump(lr, doc)

def load_model(rfc=True):
    if rfc:
        with open("./rfc.pkl", "rb") as doc:
            return load(doc)
    else:
        with open("./lr.pkl", "rb") as doc:
            return load(doc)

if __name__ == "__main__":
    fit_models()