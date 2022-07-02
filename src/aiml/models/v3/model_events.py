from path_utils import model_root
import os
import pandas as pd
import pickle
import numpy as np
from aiml.models.v3.protocol import xgb_feature_orders as events_feature_orders
import argparse
import xgboost as xgb
import numpy as np


def classify_model(df, model, args):
    m1, t1 = model
    df1 = df.drop(columns=[args.event_name, 'angiogram', 'out5'])
    x1 = df1.values
    feature_names1 = df1.columns
    if m1.booster == 'gbtree':
        s1 = m1.predict(xgb.DMatrix(x1, feature_names=feature_names1), iteration_range=(0, m1.best_iteration + 1)) #ntree_limit=m1.best_ntree_limit
    else:
        s1 = m1.predict(xgb.DMatrix(x1, feature_names=feature_names1))

    if not args.use_derived_threshold:
        t1 = 0.5
    y1_pred = s1 >= t1

    return 1 * y1_pred, s1, t1


def convert_prob(p, s):
    return 1 - s if p == 0 else s


def inference(model, df_outbag, args):
    # calculate out-of-bag properties
    y1_pred, s1, t1 = classify_model(df_outbag, model, args)

    return y1_pred, s1, np.array([t1])


def label(df_outbag):
    # real out-of-bag labels
    y1 = 1 * df_outbag['out5'].isin(['T1MI', 'T2MI', 'Acute']).values
    y2 = 1 * df_outbag['out5'].isin(['T1MI']).values

    return y1, y2


class XGBoostEventModel:
    def __init__(self, event_name=None, dump_path: str=None):
        self.version = 2
        self.MAX_NUM_SUB_MODELS = 9
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args([])
        if event_name is not None:
            self.event_name = event_name
            self.args.event_name = self.event_name
        if dump_path is not None:
            package = self.load_model(dump_path)
            self.models = package['models']
            assert package['version'] == self.version
            self.args.tpr1 = package['tpr1']
            self.args.use_derived_threshold = package['use_derived_threshold']

        self.training_info = {'event_dmi30d': {'auc': 0.849, 'tpr': 0.921, 'fpr': 0.565},
                            'event_dead': {'auc': 0.922, 'tpr': 0.981, 'fpr': 0.675}
                            }

    def __len__(self):
        return len(self.models)

    def load_model(self, dump_path):
        #dump_path = os.path.join(model_root, 'v3', 'models_{}.pickle'.format(self.event_name))
        with open(dump_path, 'rb') as handle:
            models = pickle.load(handle)
        return models

    def inference_single(self, idx=0, features=None):
        features = features.copy()
        features['logtrop1'] = features['logtrop0']
        features.pop('logtrop0')
        features = {k: [features[k]] for k in events_feature_orders}
        features[self.event_name] = 'nan'
        features['out5'] = 'nan'
        df = pd.DataFrame.from_dict(features)

        if idx == -1:
            preds = list()
            for m_idx, m in enumerate(self.models):
                if m_idx < self.MAX_NUM_SUB_MODELS:
                    preds.append(inference(m, df, self.args))
            # TODO: find a better way to estimate the ensemble model scores
            pred_ensemble = np.concatenate(preds, axis=1)
            values, counts = np.unique(pred_ensemble[0], return_counts=True)
            pred = np.array(values[np.argmax(counts)], dtype=np.int)
            prob = np.array(pred_ensemble[1])
            thld = np.array(pred_ensemble[2])
        else:
            pred, prob, thld = inference(self.models[idx], df, self.args)

        results = {'{}_pred_xgb'.format(self.event_name): pred.squeeze().tolist(),
                   '{}_prob_xgb'.format(self.event_name): prob.squeeze().tolist(),
                   '{}_thld_xgb'.format(self.event_name): thld.squeeze().tolist(),
                   '{}_training_info_xgb'.format(self.event_name): self.training_info[self.event_name]
                   }

        return results
