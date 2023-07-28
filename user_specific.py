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
models = {u: [] for u in users}  # list of all clfs made for each user
for i, user in enumerate(users):

    skf = StratifiedKFold(n_splits=settings.cv)
    avg_f1 = []
    training_df = train_users[i]
    eval_df = test_users[i]

    fold = 0
    for train_index, test_index in tqdm(skf.split(training_df[features], training_df["Activity"])):
        train_df = training_df.iloc[train_index, :].copy(deep=True)
        test_df = training_df.iloc[test_index, :].copy(deep=True)

        clf = ControlledBagging(_clf=DT(min_samples_leaf=2, criterion="entropy", random_state=0), _skew=19, _k=100, _train_size=1.0, _feats=features)

        train_df = clf.train(train_df, verbose=False)
        test_df = clf.predict(test_df, verbose=False)

        # Update results
        models[user].append(clf)
        training_results = clf.predict(training_df, verbose=False)
        eval_results = clf.predict(eval_df, verbose=False)
        training_df["pred_" + str(fold)] = training_results["pred"].values
        eval_df["pred_" + str(fold)] = eval_results["pred"].values
        for act in settings.activities:
            training_df[act + "_" + str(fold)] = training_results[act].values
            eval_df[act + "_" + str(fold)] = eval_results[act].values

        avg_f1.append(metrics.f1_score(y_true=test_df["Activity"], y_pred=test_df["pred"], average="macro"))
        fold += 1
    print(user, ":", np.mean(avg_f1), avg_f1)

    utils.evaluate_model(models[user], training_df, folder, user + "_train_" + filebase)
    utils.evaluate_model(models[user], eval_df, folder, user + "_test_" + filebase)

# Print out results
pred_files = [os.path.join(folder, x) for x in os.listdir(folder)]
for f in pred_files:
    print(os.path.basename(f).split("_")[0])
    pred_df = pd.read_csv(f)
    utils.print_results(pred_df)
