import os
import pandas as pd
import numpy as np
from sklearn import metrics

import settings


def print_results(pred_df):
    print("F1 Score(All Labels):", metrics.f1_score(y_true=pred_df["Activity"], y_pred=pred_df["pred"], average="macro"))
    print(metrics.classification_report(y_true=pred_df["Activity"], y_pred=pred_df["pred"]))
    cm = metrics.confusion_matrix(y_true=pred_df["Activity"], y_pred=pred_df["pred"])

    print("Raw Confusion Matrix")
    print(cm)

    dem = cm.sum(axis=1, keepdims=True)
    for i, ele in enumerate(dem):
        if ele == 0:
            dem[i] = 1
    cm = cm / dem
    cm = (cm * 100).astype(int)
    print("Normalized Confusion Matrix")
    print(cm)
    print('')


def read_features(file):
    df = pd.read_csv(file)
    feats = list(df)[:-3]  # last three are Activity, Start and End
    df = df.set_index(np.arange(len(df)))

    df["Activity"] = df["Activity"].str.lower()  # make labels lowercase
    # Special Cases
    df.loc[df["Activity"].str.contains("eating"), "Activity"] = "eating"
    df.loc[~df["Activity"].isin(settings._acts), "Activity"] = "nothing"

    df = df[feats + ["Start", "End", "Activity"]]
    return df


def evaluate_model(_df, _folder, _filename, out_cols=settings.out_cols, adaptive=False):
    # Determine final result
    for act in settings.activities:
        if not adaptive:
            _df[act] = _df[[act + "_" + str(j) for j in range(settings.cv)]].mean(axis=1)
        else:
            _df[act] = _df[["b_" + act + "_" + str(j) for j in range(settings.cv)] + ["s_" + act + "_" + str(j) for j in range(settings.cv)]].mean(axis=1)
    _df["pred"] = _df[settings.activities].idxmax(axis=1).values

    if not os.path.exists(_folder):
        os.makedirs(_folder)

    _df[out_cols].to_csv(os.path.join(_folder, _filename), index=False)
