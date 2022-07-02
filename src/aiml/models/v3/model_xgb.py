from path_utils import model_root
import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import numpy as np
from aiml.models.v3.protocol import xgb_feature_orders


def classify_model(df, model, version, use_derived_threshold_l1=True, use_derived_threshold_l2=True):
    (m1, t1), (m2, t2) = model
    if version == 0 or version == 1:
        df1 = df.drop(columns=['out5', 'hst_angio', 'angiogram'])
    elif version == 2:
        df1 = df.drop(columns=['out5', 'angiogram'])
    df2 = df.drop(columns=['out5'])
    x1 = df1.values
    x2 = df2.values
    feature_names1 = df1.columns
    feature_names2 = df2.columns
    if m1.booster == 'gbtree':
        s1 = m1.predict(xgb.DMatrix(x1, feature_names=feature_names1),
                        iteration_range=(0, m1.best_iteration + 1)) #ntree_limit=m1.best_ntree_limit
    else:
        s1 = m1.predict(xgb.DMatrix(x1, feature_names=feature_names1))
    if m2.booster == 'gbtree':
        s2 = m2.predict(xgb.DMatrix(x2, feature_names=feature_names2),
                        iteration_range=(0, m2.best_iteration + 1)) #ntree_limit=m2.best_ntree_limit
    else:
        s2 = m2.predict(xgb.DMatrix(x2, feature_names=feature_names2))

    if not use_derived_threshold_l1:
        t1 = 0.5
    y1_pred = s1 >= t1

    if not use_derived_threshold_l2:
        t2 = 0.5
    y2_pred = s2 >= t2

    return 1 * y1_pred, 1 * y2_pred, s1, s2, t1, t2


def convert_prob(p, s):
    return 1 - s if p == 0 else s


def inference(model, df_outbag, version=2, use_derived_threshold_l1=True, use_derived_threshold_l2=True):
    # calculate out-of-bag properties
    y1_pred, y2_pred, s1, s2, t1, t2 = classify_model(df_outbag, model,
                                                      version=version,
                                                      use_derived_threshold_l1=use_derived_threshold_l1,
                                                      use_derived_threshold_l2=use_derived_threshold_l2)

    return y1_pred, y2_pred, s1, s2, np.array([t1]), np.array([t2])


def label(df_outbag):
    # real out-of-bag labels
    y1 = 1 * df_outbag['out5'].isin(['T1MI', 'T2MI', 'Acute']).values
    y2 = 1 * df_outbag['out5'].isin(['T1MI']).values

    return y1, y2


class XGBoostModel:
    def __init__(self, dump_path: str=None):
        if dump_path is not None:
            self.version = 2
            package = self.load_model(dump_path)
            self.MAX_NUM_SUB_MODELS = 19
            self.models = package['models']
            self.tpr1 = package['tpr1']
            self.tpr2 = package['tpr2']
            assert package['version'] == self.version
            self.use_derived_threshold_l1 = package['use_derived_threshold_l1']
            self.use_derived_threshold_l2 = package['use_derived_threshold_l2']

            self.training_info = {'l1': {'auc': 0.988, 'tpr': 0.972, 'fpr': 0.115},
                                'l2': {'auc': 0.877, 'tpr': 0.880, 'fpr': 0.258}
                                }

    def __len__(self):
        return len(self.models)

    def load_model(self, dump_path):
        #dump_path = os.path.join(model_root, 'v3', 'models_xgb.pickle')

        with open(dump_path, 'rb') as handle:
            models = pickle.load(handle)
        return models

    def inference_single(self, idx=0, features=None):
        features = features.copy()
        features['logtrop1'] = features['logtrop0']
        features.pop('logtrop0')
        features = {k: [features[k]] for k in xgb_feature_orders}
        features['out5'] = 'nan'
        df = pd.DataFrame.from_dict(features)

        if idx == -1:
            preds = list()
            for m_idx, m in enumerate(self.models):
                if m_idx < self.MAX_NUM_SUB_MODELS:
                    preds.append(inference(m, df,
                                           version=self.version,
                                           use_derived_threshold_l1=self.use_derived_threshold_l1,
                                           use_derived_threshold_l2=self.use_derived_threshold_l2))

            # TODO: find a better way to estimate the ensemble model scores
            pred_ensemble = np.concatenate(preds, axis=1)
            values, counts = np.unique(pred_ensemble[0], return_counts=True)
            pred_l1 = np.array(values[np.argmax(counts)], dtype=np.int)
            values, counts = np.unique(pred_ensemble[1], return_counts=True)
            pred_l2 = np.array(values[np.argmax(counts)], dtype=np.int)
            prob_l1 = np.array(pred_ensemble[2])
            prob_l2 = np.array(pred_ensemble[3])
            thld_l1 = np.array(pred_ensemble[4])
            thld_l2 = np.array(pred_ensemble[5])
        else:
            pred_l1, pred_l2, prob_l1, prob_l2, thld_l1, thld_l2 = \
                inference(self.models[idx], df,
                          version=self.version,
                          use_derived_threshold_l1=self.use_derived_threshold_l1,
                          use_derived_threshold_l2=self.use_derived_threshold_l2)

        if pred_l1 == 0:
            p3_pred = 0
        elif pred_l1 == 1:
            if pred_l2 == 0:
                p3_pred = 1
            else:
                p3_pred = 2

        p3_pred = np.array([p3_pred])

        results = {
            'l1_pred_xgb': pred_l1.squeeze().tolist(),
            'l1_prob_xgb': prob_l1.squeeze().tolist(),
            'l1_thld_xgb': thld_l1.squeeze().tolist(),
            'l2_pred_xgb': pred_l2.squeeze().tolist(),
            'l2_prob_xgb': prob_l2.squeeze().tolist(),
            'l2_thld_xgb': thld_l2.squeeze().tolist(),
            'p3_pred_xgb': p3_pred.squeeze().tolist(),
            'training_info_xgb': self.training_info,
        }

        return results
