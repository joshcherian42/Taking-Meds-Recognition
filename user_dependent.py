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


models = []

skf = StratifiedKFold(n_splits=settings.cv)
avg_f1 = []
fold = 0
for train_index, test_index in tqdm(skf.split(training_df[features], training_df["Activity"])):
    train_df = training_df.iloc[train_index, :].copy(deep=True)
    test_df = training_df.iloc[test_index, :].copy(deep=True)

    clf = ControlledBagging(_clf=DT(min_samples_leaf=2, criterion="entropy", random_state=0), _skew=19, _k=100, _train_size=1.0, _feats=features)

    train_df = clf.train(train_df, verbose=False)
    test_df = clf.predict(test_df, verbose=False)

    # Update results
    models.append(clf)
    training_results = clf.predict(training_df, verbose=False)
    eval_results = clf.predict(eval_df, verbose=False)
    training_df["pred_" + str(fold)] = training_results["pred"].values
    eval_df["pred_" + str(fold)] = eval_results["pred"].values
    for act in settings.activities:
        training_df[act + "_" + str(fold)] = training_results[act].values
        eval_df[act + "_" + str(fold)] = eval_results[act].values

    avg_f1.append(metrics.f1_score(y_true=test_df["Activity"], y_pred=test_df["pred"], average="macro"))
    fold += 1
print("Avg. F1:", np.mean(avg_f1), avg_f1)

folder = "bagging_results_dependent"
filebase = "cv10_dt_skew19_k100_v2.csv"

train_file = os.path.join(folder, "dependent_train_" + filebase)
test_file = os.path.join(folder, "dependent_test_" + filebase)

utils.evaluate_model(models, training_df, folder, train_file)
utils.evaluate_model(models, eval_df, folder, test_file)

# Print out results
pred_df = pd.read_csv(train_file)
_pred_df = pd.read_csv(test_file)

for user in users:
    print(user)
    subset = _pred_df.loc[_pred_df["origin"] == user]
    utils.print_results(subset)
