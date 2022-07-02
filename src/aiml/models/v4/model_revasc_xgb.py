from path_utils import model_root
import os
import pandas as pd
import pickle
import numpy as np
import argparse
from aiml.models.v4.protocol import revasc_xgb_feature_orders

import xgboost as xgb
import numpy as np

def classify_model(df, model, args):
    m1, t1 = model
    if args.label_name == 'both':
        df1 = df.drop(columns=['cabg', 'intervention', 'set0'])
    else:
        df1 = df.drop(columns=[args.label_name, 'set0'])
    x1 = df1.values
    feature_names1 = df1.columns
    if m1.booster == 'gbtree':
        s1 = m1.predict(xgb.DMatrix(x1, feature_names=feature_names1), iteration_range=(0, m1.best_iteration + 1)) #ntree_limit=m1.best_ntree_limit
    else:
        s1 = m1.predict(xgb.DMatrix(x1, feature_names=feature_names1))

    if args.use_derived_threshold:
        y1_pred = s1 >= t1
    else:
        y1_pred = s1 >= 0.5
    return 1 * y1_pred, s1


def convert_prob(p, s):
    return 1 - s if p == 0 else s


def inference(model, df_outbag, args):
    # calculate out-of-bag properties
    y1_pred, s1 = classify_model(df_outbag, model, args)

    return y1_pred, s1


# def label(df_outbag):
#     # real out-of-bag labels
#     y1 = 1 * df_outbag['out5'].isin(['T1MI', 'T2MI', 'Acute']).values
#     y2 = 1 * df_outbag['out5'].isin(['T1MI']).values
#
#     return y1, y2


class RevascXGBoostModel:
    def __init__(self, label_name: str=None, dump_path: str=None):
        if label_name is not None:
            self.label_name = label_name

        if dump_path is not None:
            package = self.load_model(dump_path)
            self.MAX_NUM_SUB_MODELS = 1
            self.models = package['models']

            parser = argparse.ArgumentParser()
            self.args = parser.parse_args([])
            # self.args.tpr1 = package['tpr1']
            self.args.use_derived_threshold = False
            self.args.label_name = self.label_name

        self.training_info = {'event_dmi30d': {'auc': 0.849, 'tpr': 0.921, 'fpr': 0.565},
                              'event_dead': {'auc': 0.922, 'tpr': 0.981, 'fpr': 0.675}
                              }

        self.thresholds = {'cabg': {'def': 0.5, 'roc': 0.003, 'pr': 0.246},
                           'intv': {'def': 0.5, 'roc': 0.024, 'pr': 0.324},
                           '(cabg|intv)': {'def': 0.5, 'roc': 0.013, 'pr': 0.144}}

    def __len__(self):
        return len(self.models)

    def load_model(self, dump_path):
        #dump_path = os.path.join(model_root, 'v4', 'revasc_models_{}_xgb.pickle'.format(self.label_name))

        with open(dump_path, 'rb') as handle:
            models = pickle.load(handle)
        return models

    def inference_single(self, idx=0, features=None):
        features = features.copy()
        features['logtrop1'] = features['logtrop0']
        features.pop('logtrop0')
        features = {k: [features[k]] for k in revasc_xgb_feature_orders}
        features['cabg'] = 'nan'
        features['intervention'] = 'nan'
        features['set0'] = 'nan'
        df = pd.DataFrame.from_dict(features)

        revasc_keys = ['cabg', 'intv']
        probs = {k: list() for k in revasc_keys}

        if idx == -1:
            for m_idx, m in enumerate(self.models):
                if m_idx < self.MAX_NUM_SUB_MODELS:
                    for k_idx, k in enumerate(revasc_keys):
                        _, prob = inference(m[k_idx], df, self.args)
                        probs[k].append(prob)
            # TODO: find a better way to estimate the ensemble model scores
            for k in revasc_keys:
                probs[k] = np.stack(probs[k], axis=0).mean(axis=0, keepdims=True)
        else:
            for k_idx, k in enumerate(revasc_keys):
                _, prob = inference(self.models[idx][k_idx], df, self.args)
                probs[k] = prob

        probs['(cabg|intv)'] = (probs['cabg'] + probs['intv']) / 2.

        results = dict()
        for k in self.thresholds:
            results['{}_prob_xgb'.format(k)] = probs[k].squeeze().tolist()
            for t in self.thresholds[k]:
                results['{}_pred_{}_xgb'.format(k, t)] = int(probs[k] >= self.thresholds[k][t])
                results['{}_thld_{}_xgb'.format(k, t)] = self.thresholds[k][t]

        return results
