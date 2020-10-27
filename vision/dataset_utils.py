import json

from sklearn.model_selection import train_test_split


path = "/home/kevin/bin/scraping_engine/data/export-2020-10-27T14_14_30.622Z.json"
with open(path, "r") as f:
    paths = json.load(f)
train, test = train_test_split(paths, shuffle=True, train_size=.8)
print(train)
print(test)