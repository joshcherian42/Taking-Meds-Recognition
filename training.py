import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier as DT

import settings
from controlled_bagging import ControlledBagging


def training(subset_df, training_df, eval_df, prefix=''):
    features = settings.features_header[:-3]
    skf = StratifiedKFold(n_splits=settings.cv)
    avg_f1 = []
    fold = 0

    for train_index, test_index in tqdm(skf.split(subset_df[features], subset_df["Activity"])):
        train_df = subset_df.iloc[train_index, :].copy(deep=True)
        test_df = subset_df.iloc[test_index, :].copy(deep=True)

        clf = ControlledBagging(_clf=DT(min_samples_leaf=2, criterion="entropy", random_state=0), _skew=19, _k=100, _train_size=1.0, _feats=features)

        train_df = clf.train(train_df, verbose=False)
        test_df = clf.predict(test_df, verbose=False)

        training_results = clf.predict(training_df, verbose=False)
        eval_results = clf.predict(eval_df, verbose=False)
        training_df[prefix + "pred_" + str(fold)] = training_results["pred"].values
        eval_df[prefix + "pred_" + str(fold)] = eval_results["pred"].values

        for act in settings.activities:
            training_df[prefix + act + "_" + str(fold)] = training_results[act].values
            eval_df[prefix + act + "_" + str(fold)] = eval_results[act].values

        avg_f1.append(metrics.f1_score(y_true=test_df["Activity"], y_pred=test_df["pred"], average="macro"))
        fold += 1
    print("Avg. F1:", np.mean(avg_f1), avg_f1)

    return training_df, eval_df
