import os
import numpy as np
import pandas as pd

import utils
import settings
from training import training
from secondary_model import secondary_model

features = settings.features_header[:-3]  # last columns are Start, End, Activity
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
    _df["Origin"] = os.path.basename(f).split(".")[0]
    for act in settings.activities:
        subset = _df.loc[_df["Activity"] == act]
        perm = np.random.permutation(len(subset))
        cutoff = int(len(subset) * (1 - holdout))
        training_df = pd.concat([training_df, subset.iloc[perm[:cutoff]]], ignore_index=True)
        eval_df = pd.concat([eval_df, subset.iloc[perm[cutoff:]]], ignore_index=True)
    training_df = training_df.sort_values(by=["Start"])
    eval_df = eval_df.sort_values(by=["End"])


def special_eval(_df, _user, _predict):
    if not _predict:
        subset = _df.loc[_df["Origin"] != _user]
    else:
        subset = _df.loc[_df["Origin"] == _user]
    utils.print_results(subset)


out_cols = settings.labels + settings.activities
out_cols += ["b_pred_" + str(j) for j in range(settings.cv)]

for act in settings.activities:
    out_cols += ["b_" + act + "_" + str(j) for j in range(settings.cv)]
out_cols += ["s_pred_" + str(j) for j in range(settings.cv)]

for act in settings.activities:
    out_cols += ["s_" + act + "_" + str(j) for j in range(settings.cv)]

folder = "tiebreaker-bagging"
filebase = "cv10_dt_skew19_k100.csv"
folder_dir = os.path.join(folder, filebase.split(".")[0])

if not os.path.exists(folder_dir):
    os.makedirs(folder_dir)

for user in users:
    print(user)
    baseline_df = training_df.copy(deep=True)
    specific_df = training_df.loc[training_df["Origin"] == user].copy(deep=True)

    training_df, eval_df = training(baseline_df, training_df, eval_df, prefix='b_')
    training_df, eval_df = training(specific_df, training_df, eval_df, prefix='s_')

    utils.evaluate_model(training_df, folder_dir, user + "_train_" + filebase, out_cols=out_cols, adaptive=True)
    utils.evaluate_model(eval_df, folder_dir, user + "_test_" + filebase, out_cols=out_cols, adaptive=True)

    print("TRAIN")
    special_eval(training_df, user, _predict=False)
    print("TEST")
    special_eval(eval_df, user, _predict=True)

print('')
print('Secondary Model')
print('')
secondary_model(users, folder_dir, 'SVM', out_cols=out_cols, prefix="s_")
