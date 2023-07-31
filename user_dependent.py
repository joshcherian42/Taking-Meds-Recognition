import os
import numpy as np
import pandas as pd

import utils
import settings
from training import training
from controlled_bagging import ControlledBagging
from secondary_model import secondary_model


features = settings.features_header[:-4]  # last columns are Start, End, Origin, Activity
train_dir = os.path.join("data", "Train")
test_dir = os.path.join("data", "Test")
train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
test_files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]
files = train_files + test_files

training_df = pd.DataFrame(data=[], columns=list(utils.read_features(files[0])) + ["origin"])
eval_df = pd.DataFrame(data=[], columns=list(utils.read_features(files[0])) + ["origin"])
users = [os.path.basename(f).split(".")[0] for f in files]

for f in files:
    _df = utils.read_features(f)
    _df["Origin"] = os.path.basename(f).split(".")[0]
    for act in settings.activities:
        subset = _df.loc[_df["Activity"] == act]
        perm = np.random.permutation(len(subset))
        cutoff = int(len(subset) * 0.9)
        training_df = pd.concat([training_df, subset.iloc[perm[:cutoff]]], ignore_index=True)
        eval_df = pd.concat([eval_df, subset.iloc[perm[cutoff:]]], ignore_index=True)
    training_df = training_df.sort_values(by=["Start"])
    eval_df = eval_df.sort_values(by=["End"])

training_df, eval_df = training(training_df, training_df, eval_df)

folder = "bagging_results_dependent"
filebase = "cv10_dt_skew19_k100_v2.csv"

train_file = os.path.join(folder, "dependent_train_" + filebase)
test_file = os.path.join(folder, "dependent_test_" + filebase)

utils.evaluate_model(training_df, folder, "dependent_train_" + filebase)
utils.evaluate_model(eval_df, folder, "dependent_test_" + filebase)

# Print out results
pred_df = pd.read_csv(train_file)
_pred_df = pd.read_csv(test_file)

for user in users:
    print(user)
    subset = _pred_df.loc[_pred_df["Origin"] == user]
    utils.print_results(subset)

print('')
print('Secondary Model')
print('')
secondary_model(users, folder, 'SVM')
