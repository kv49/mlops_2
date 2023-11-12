import pickle
import pandas as pd
import os

#initial_path = os.getcwd()
src_path = os.path.dirname(os.path.abspath(__file__))
#os.chdir(working_path) # set directory of the script as current

#print(initial_path)
#print(src_path)

df = pd.read_csv(src_path + "/../../data/stage4/test.csv")
X = df.iloc[:,[1,2,3]]
y = df.iloc[:,0]

def test1():
    with open(src_path + "/../../models/model.pkl", "rb") as fd:
        clf = pickle.load(fd)
    score = clf.score(X, y)
    assert score > 0.75