import torch
from path_utils import model_root
import os
import pandas as pd
import pickle
import numpy as np
import re
import warnings


class RevascDLModel:
    def __init__(self, dump_path: str=None):
        if dump_path is not None:
            self.models, self.data_loader = self.load_model(dump_path)
            self.trop_feats = {k: [self.data_loader.ignore_value] for k in self.data_loader.trop_keys +
                                ['time_{}'.format(t) for t in self.data_loader.trop_keys]}
            self.mode = 'cpu'
            self.thresholds = {'cabg': {'def': 0.5, 'roc': 0.363, 'pr': 0.934},
                            'intv': {'def': 0.5, 'roc': 0.607, 'pr': 0.962},
                            '(cabg|intv)': {'def': 0.5, 'roc': 0.690/2., 'pr': 1.484/2.}}

    def __len__(self):
        return len(self.models)

    def change_mode(self, mode='cpu'):
        self.mode = mode

        if self.mode == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn('No GPU found, keep using CPU!')
                device = 'cpu'

        for m_idx, m in enumerate(self.models):
            self.models[m_idx] = m.to(self.mode)

    def load_model(self, dump_path):
        #dump_path = os.path.join(model_root, 'v4', 'revasc_models.pickle')

        with open(dump_path, 'rb') as handle:
            package = pickle.load(handle)

        for m in package['models']:
            m.eval()
        return package['models'], package['data_loader'].dataset

    def process_feature(self, features):

        message = dict()

        labels = ['out5', 'event_mi', 'event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi',
                  'event_dead', 'event_dmi30d', 'cabg', 'intervention']
        labels = {k: 'nan' for k in labels}
        features = {**features, **labels}
        features = {k: [features[k]] if features[k] !=
                    'nan' else self.data_loader.ignore_value for k in features}
        features = {**features, **self.trop_feats}
        df = pd.DataFrame.from_dict(features, dtype='float')
        df = self.data_loader.class2index(df)

        return df, message

    def inference_single(self, idx=0, features=None):

        df, message = self.process_feature(features)
        # print(df)
        # print(message)
        features, _ = self.data_loader.get_item(df.iloc[0])
        features = features.unsqueeze(0)
        features = features.to(self.mode)

        revasc_keys = ['cabg', 'intv']
        probs = {k: list() for k in revasc_keys}
        if idx == -1:
            for m in self.models:
                with torch.no_grad():
                    regression_logits, binary_cls_logits, cls_logits = m(
                        features)
                for k_idx, k in enumerate(revasc_keys):
                    prob = torch.sigmoid(
                        binary_cls_logits[0, k_idx]).cpu().numpy()
                    probs[k].append(prob)
            # TODO: find a better way to estimate the ensemble model scores
            for k in revasc_keys:
                probs[k] = np.stack(probs[k], axis=0).mean(
                    axis=0, keepdims=True)
        else:
            with torch.no_grad():
                regression_logits, binary_cls_logits, cls_logits = self.models[idx](
                    features)
            for k_idx, k in enumerate(revasc_keys):
                probs[k] = torch.sigmoid(
                    binary_cls_logits[0, k_idx]).cpu().numpy()

        probs['(cabg|intv)'] = (probs['cabg'] + probs['intv']) / 2.

        results = dict()
        for k in self.thresholds:
            results['{}_prob_dl'.format(k)] = probs[k].squeeze().tolist()
            for t in self.thresholds[k]:
                results['{}_pred_{}_dl'.format(k, t)] = int(
                    probs[k] >= self.thresholds[k][t])
                results['{}_thld_{}_dl'.format(k, t)] = self.thresholds[k][t]

        return {**results, **message}
