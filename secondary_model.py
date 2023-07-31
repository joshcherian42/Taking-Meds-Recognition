import os
import numpy as np
import pandas as pd
import statistics
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF


import settings
import utils


def classification_wrapper(model_type, train_df, features, train_y):
    if model_type == 'DT':
        return DT().fit(train_df[features].values, train_y)
    elif model_type == 'SVM':
        return SVC().fit(train_df[features].values, train_y)
    elif model_type == 'MLP':
        return MLP().fit(train_df[features].values, train_y)
    elif model_type == 'RF':
        return RF(random_state=0).fit(train_df[features].values, train_y)


def portfolio_wrapper(model_type, a, b):
    if model_type == 'cosine':
        sim = np.dot(a, b) / np.sqrt(np.dot(a, a)) / np.sqrt(np.dot(b, b))
        if sim < -1:  # in case zero votes
            sim = -1
        return sim
    elif model_type == 'manhattan':
        sim = np.sum(np.abs(np.subtract(a, b)))
        if sim != 0:
            sim = 1 / sim
        else:
            sim = 100000
        return sim


def ranking(probs):
    indices = list(zip(settings.activities, probs))  # name, prob
    indices.sort(key=lambda x: x[1], reverse=True)
    indices = [x[0] for x in indices]
    return indices


def get_stats(train_df, feats):
    means = {}  # activity -> list of means for each label
    stdevs = {}
    ranks = {}

    for i, ele in enumerate(settings.activities):
        subset = train_df.loc[train_df["Activity"] == ele]
        means[ele] = subset[feats].mean().values
        stdevs[ele] = subset[feats].std().values

        indices = ranking(subset[settings.activities].mean().values)
        ranks[ele] = indices

    return means, stdevs, ranks


def secondary_model(users, folder, model_type):
    feats = [x for x in settings.out_cols if x not in settings.labels]
    feats = [x for x in feats if "pred" not in x and '_' in x]
    train_pred_files = [os.path.join(folder, x) for x in os.listdir(folder) if "train" in x]
    test_pred_files = [os.path.join(folder, x) for x in os.listdir(folder) if "test" in x]

    classification_methods = ['DT', 'RF', 'MLP', 'SVM']
    portfolio_methods = ['cosine', 'manhattan']
    ranking_methods = ['weighted jaccard', 'thresholded jaccard']

    for i, user in enumerate(users):

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(user)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        if len(train_pred_files) == 1:
            train_f = train_pred_files[0]
            test_f = test_pred_files[0]
        else:
            train_f = [x for x in train_pred_files if user in x][0]
            test_f = [x for x in test_pred_files if user in x][0]
        train_df = pd.read_csv(train_f)
        test_df = pd.read_csv(test_f)
        train_df = train_df.loc[train_df["Origin"] == user].reset_index(drop=True)
        test_df = test_df.loc[test_df["Origin"] == user].reset_index(drop=True)

        lbs = [n for n in settings.labels if n != 'Origin']
        pred_df = train_df[settings.activities + lbs]
        _pred_df = test_df[settings.activities + lbs]

        if model_type in classification_methods:
            train_y = [1 if x == "taking meds" else 0 for x in train_df["Activity"].values]
            # test_y = [1 if x == "taking meds" else 0 for x in test_df["Activity"].values]

            _clf = classification_wrapper(model_type, train_df, feats, train_y)
            train_df["pred"] = _clf.predict(train_df[feats].values)
            test_df["pred"] = _clf.predict(test_df[feats].values)
            train_df.loc[train_df["pred"] == 1, "pred"] = "taking meds"
            train_df.loc[train_df["pred"] == 0, "pred"] = "nothing"
            test_df.loc[test_df["pred"] == 1, "pred"] = "taking meds"
            test_df.loc[test_df["pred"] == 0, "pred"] = "nothing"

        elif model_type in portfolio_methods:
            means, stdevs, _ = get_stats(train_df, feats)

            vals = test_df[feats].values
            data = {ele: [] for ele in settings.activities}
            for ind, row in tqdm(enumerate(vals)):
                for i, ele in enumerate(settings.activities):
                    sim = portfolio_wrapper(model_type, vals[ind], means[ele])
                    data[ele].append(sim)
            data = pd.DataFrame.from_dict(data)
            test_df["pred"] = data[settings.activities].idxmax(axis=1)

        elif model_type in ranking_methods:
            ranks = {}
            means, stdevs, ranks = get_stats(train_df, feats)

            data = {ele: [] for ele in settings.activities}
            t = 2
            for ind, row in test_df.iterrows():
                r = ranking(row[settings.activities])
                for i, ele in enumerate(settings.activities):
                    activity_mean = statistics.fmean(means[ele][settings.cv * i: settings.cv * (i + 1)])
                    activity_stdev = statistics.stdev(stdevs[ele][settings.cv * i: settings.cv * (i + 1)])
                    if model_type == 'thresholded jaccard' and row[ele] <= activity_mean - (activity_stdev / 2):
                        val = 0
                    else:
                        val = len(set(ranks[ele][:t]).intersection(set(r[:t]))) / len(set(ranks[ele][:t]).union(set(r[:t])))
                        if model_type == 'weighted jaccard':
                            val *= row[ele]

                    data[ele].append(val)

            df = pd.DataFrame.from_dict(data)
            df["pred"] = df[settings.activities].idxmax(axis=1)
            df["Activity"] = _pred_df["Activity"]

        utils.print_results(test_df)
        print('')
