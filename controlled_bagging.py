import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, SelectFromModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import clone

import utils
import settings


class ControlledBagging:
    def __init__(self,
                 _clf=RF(n_estimators=30, max_depth=30, min_samples_leaf=2, criterion="entropy"),
                 _scaler=MinMaxScaler(),
                 _subset=SelectKBest(chi2, k=50),
                 _skew=20, _k=10, _train_size=0.8, _feats=[]):

        self.LE = LabelEncoder()  # converts nominal labels into numeric for model
        self.clf = _clf           # primary classifier
        self.clfs = []
        self.scaler = _scaler
        self.subset = _subset
        self.skew = _skew
        self.k = _k
        self.train_size = _train_size
        self._feats = _feats

    def subset_selection(self, _df, _predict=False):

        if not _predict:
            y = self.LE.fit_transform(_df["Activity"].values)
            x = self.scaler.fit_transform(_df[self._feats].values)
            x = self.subset.fit_transform(x, y)
        else:
            y = self.LE.transform(_df["Activity"].values)
            x = self.subset.transform(self.scaler.transform(_df[self._feats].values))

        return x, y

    # Pass data through classifier (_predict=False trains model, _predict=True tests model)
    def run_model(self, _df, _predict, verbose):
        all_x, all_y = self.subset_selection(_df, _predict=_predict)  # subset selection on entire train set
        same_df = _df.loc[_df["Activity"] != "nothing"]
        diff_df = _df.loc[_df["Activity"] == "nothing"]

        preds = np.zeros((len(_df), len(settings.activities)))
        for ind in range(self.k):
            same_subset = same_df.sample(n=int(self.train_size * len(same_df)))
            diff_subset = diff_df.sample(n=min(int(self.skew * len(same_subset)), len(diff_df)))
            subset = pd.concat([same_subset, diff_subset], ignore_index=True)
            subset = shuffle(subset)
            x, y = self.subset_selection(subset, _predict=True)  # don't retrain the subset selection

            if not _predict:
                time0 = time.time()
                clf = clone(self.clf)
                clf.fit(x, y)
                self.clfs.append(clf)
                time1 = time.time()
                if verbose:
                    print("Train time:", int((time1 - time0)))
            else:
                clf = self.clfs[ind]

            probs = clf.predict_proba(all_x)
            for i, act in enumerate(self.LE.classes_):
                preds[:, i] += probs[:, i]
        for i, act in enumerate(self.LE.classes_):
            _df.loc[:, act] = preds[:, i] / self.k
        _df.loc[:, "pred"] = _df[settings.activities].idxmax(axis=1).values
        return _df

    def train(self, _df, verbose=True):
        # Train classifier
        _df = self.run_model(_df, _predict=False, verbose=verbose)
        if verbose:
            print("TRAIN")
            utils.print_results(_df)
            print()
        return _df

    def predict(self, _df, verbose=True):
        # Pass through primary classifier
        _df = self.run_model(_df, _predict=True, verbose=verbose)
        if verbose:
            print("PREDICT")
            utils.print_results(_df)
            print()
        return _df
