import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier as DT

from controlled_bagging import ControlledBagging
import settings
import utils

features = settings.features_header[:-3]  # last columns are Start, End, Origin, Activity
train_dir = os.path.join("data", "Train")
test_dir = os.path.join("data", "Test")
train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
test_files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]
files = train_files + test_files

training_df = pd.DataFrame(data=[], columns=features + settings.labels)
eval_df = pd.DataFrame(data=[], columns=features + settings.labels)
users = [os.path.basename(f).split(".")[0] for f in files]

holdout = 0.1
for f in files:
    _df = utils.read_features(f)
    _df["origin"] = os.path.basename(f).split(".")[0]
    for act in settings.activities:
        subset = _df.loc[_df["Activity"] == act]
        perm = np.random.permutation(len(subset))
        cutoff = int(len(subset) * (1 - holdout))
        training_df = pd.concat([training_df, subset.iloc[perm[:cutoff]]], ignore_index=True)
        eval_df = pd.concat([eval_df, subset.iloc[perm[cutoff:]]], ignore_index=True)
    training_df = training_df.sort_values(by=["Start"])
    eval_df = eval_df.sort_values(by=["End"])

out_cols = settings.labels + settings.activities
out_cols += ["b_pred_" + str(j) for j in range(settings.cv)]
for act in settings.activities:
    out_cols += ["b_" + act + "_" + str(j) for j in range(settings.cv)]
out_cols += ["s_pred_" + str(j) for j in range(settings.cv)]
for act in settings.activities:
    out_cols += ["s_" + act + "_" + str(j) for j in range(settings.cv)]

folder = "tiebreaker-bagging"
params = "cv10_dt_skew19_k100"
pred_dir = os.path.join(folder, params)
train_pred_files = [os.path.join(pred_dir, x) for x in os.listdir(pred_dir) if "train" in x]
test_pred_files = [os.path.join(pred_dir, x) for x in os.listdir(pred_dir) if "test" in x]

for i, user in enumerate(users):
    print(user)
    train_f = [x for x in train_pred_files if user in x][0]
    test_f = [x for x in test_pred_files if user in x][0]
    train_df = pd.read_csv(train_f)
    test_df = pd.read_csv(test_f)

    train_other = train_df.loc[train_df["origin"] != user]
    train_same = train_df.loc[train_df["origin"] == user]
    test_same = test_df.loc[test_df["origin"] == user]

    for act in settings.activities:
        train_df[act] = train_df[["b_" + act + "_" + str(j) for j in range(settings.cv)]].mean(axis=1)
    train_df["pred"] = train_df[settings.activities].idxmax(axis=1)
    print("All training data (just baseline)")
    utils.print_results(train_df)

    for act in settings.activities:
        train_other[act] = train_other[["b_" + act + "_" + str(j) for j in range(settings.cv)]].mean(axis=1)
    train_other["pred"] = train_other[settings.activities].idxmax(axis=1)
    print("Non-specific training data (just baseline)")
    utils.print_results(train_other)

    for act in settings.activities:
        train_same[act] = train_same[["s_" + act + "_" + str(j) for j in range(settings.cv)]].mean(axis=1)
    train_same["pred"] = train_same[settings.activities].idxmax(axis=1)
    print("User-specific data (just specific)")
    utils.print_results(train_same)

    print("Eval: User-specific data (all model)")
    print_results(test_same)
