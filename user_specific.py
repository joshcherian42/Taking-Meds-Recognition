import os
import numpy as np
import pandas as pd

import utils
import settings
from training import training
from secondary_model import secondary_model

features = settings.features_header[:-3]  # last columns are Start, End, Origin, Activity
train_dir = os.path.join("data", "Train")
test_dir = os.path.join("data", "Test")
train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
test_files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]
files = train_files + test_files

train_users = []  # train set for each user
test_users = []   # test set for each user

users = [os.path.basename(f).split(".")[0] for f in files]

for f in files:
    _df = utils.read_features(f)
    _df["Origin"] = os.path.basename(f).split(".")[0]
    train_df = pd.DataFrame(data=[], columns=list(_df))
    test_df = pd.DataFrame(data=[], columns=list(_df))
    for act in settings.activities:
        subset = _df.loc[_df["Activity"] == act]
        perm = np.random.permutation(len(subset))
        cutoff = int(len(subset) * 0.9)
        train_df = pd.concat([train_df, subset.iloc[perm[:cutoff]]], ignore_index=True)
        test_df = pd.concat([test_df, subset.iloc[perm[cutoff:]]], ignore_index=True)
    train_df = train_df.sort_values(by=["Start"])
    test_df = test_df.sort_values(by=["End"])
    train_users.append(train_df)
    test_users.append(test_df)

folder = "bagging_results"
filebase = "cv10_dt_skew19_k100.csv"
for i, user in enumerate(users):
    print(user)
    training_df = train_users[i]
    eval_df = test_users[i]
    training_df, eval_df = training(training_df, training_df, eval_df)

    utils.evaluate_model(training_df, folder, user + "_train_" + filebase)
    utils.evaluate_model(eval_df, folder, user + "_test_" + filebase)

# Print out results
pred_files = [os.path.join(folder, x) for x in os.listdir(folder)]
for f in pred_files:
    print(os.path.basename(f).split("_")[0])
    pred_df = pd.read_csv(f)
    utils.print_results(pred_df)

print('')
print('Secondary Model')
print('')
secondary_model(users, folder, 'SVM')
