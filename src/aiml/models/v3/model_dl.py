import torch
from path_utils import model_root
import os
import pandas as pd
import pickle
import numpy as np
import re
from aiml.models.v3.protocol import trop_regex, time_trop_regex, phys_feature_log
import warnings


class DLModel:
    def __init__(self, dump_path: str=None):
        if dump_path is not None:
            self.models, self.data_loader = self.load_model(dump_path)
            self.default_threshold = True
            self.quantized_trop_feats = {k: [self.data_loader.ignore_value] for k in self.data_loader.quantized_trop_keys +
                                        self.data_loader.trop_keys + self.data_loader.fake_trop_keys +
                                        ['time_{}'.format(t) for t in self.data_loader.trop_keys +
                                        self.data_loader.fake_trop_keys]}
            self.mode = 'cpu'

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
        #dump_path = os.path.join(model_root, 'v3', 'models.pickle')

        with open(dump_path, 'rb') as handle:
            package = pickle.load(handle)

        for m in package['models']:
            m.eval()
        return package['models'], package['data_loader'].dataset

    def process_feature(self, features):

        message = dict()

        labels = ['out5', 'event_mi', 'event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi',
                  'event_dead', 'event_dmi30d']
        labels = {k: 'nan' for k in labels}
        features = {**features, **labels}
        features = {k: [features[k]] if features[k] != 'nan' else self.data_loader.ignore_value for k in features}
        features = {**features, **self.quantized_trop_feats}
        df = pd.DataFrame.from_dict(features, dtype='float')
        df = self.data_loader.class2index(df)

        return df, message

    def inference_single(self, idx=0, features=None):

        df, message = self.process_feature(features)
        features, _ = self.data_loader.get_item(df.iloc[0])
        features = features.unsqueeze(0)
        features = features.to(self.mode)

        if idx == -1:
            probs = list()
            for m in self.models:
                with torch.no_grad():
                    regression_logits, binary_cls_logits, cls_logits, mu_sigma, curve_params = m(features)
                prob_out5 = torch.softmax(cls_logits[0], dim=1).cpu().numpy()
                probs.append(prob_out5)
            # TODO: find a better way to estimate the ensemble model scores
            probs_ensemble = np.concatenate(probs, axis=0).mean(axis=0, keepdims=True)
            prob_out5 = probs_ensemble
        else:
            with torch.no_grad():
                regression_logits, binary_cls_logits, cls_logits, mu_sigma, curve_params = self.models[idx](features)
            prob_out5 = torch.softmax(cls_logits[0], dim=1).cpu().numpy()

        prob_out5 = self.reorder(prob_out5)
        pred_out5 = np.expand_dims(np.argmax(prob_out5, axis=1), axis=1)
        pred_3c, prob_3c = self.prob_converter(prob_out5)
        prob_l1 = 1 - prob_3c[:, 0:1]
        pred_l1 = (prob_l1 > 0.5).astype(np.int)
        prob_l2 = 1 - (prob_3c[:, 0:1] + prob_3c[:, 1:2])
        pred_l2 = (prob_l2 > 0.5).astype(np.int)

        curve_params = curve_params.squeeze().cpu().numpy()
        class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}
        curve_params = {k: curve_params[:, v].tolist() for k, v in class_dict.items()}
        results = {'p5_pred_dl': pred_out5.squeeze().tolist(), 'p5_prob_dl': prob_out5.squeeze().tolist(),
                   'p3_pred_dl': pred_3c.squeeze().tolist(), 'p3_prob_dl': prob_3c.squeeze().tolist(),
                   'l1_pred_dl': pred_l1.squeeze().tolist(), 'l1_prob_dl': prob_l1.squeeze().tolist(),
                   'l2_pred_dl': pred_l2.squeeze().tolist(), 'l2_prob_dl': prob_l2.squeeze().tolist(),
                   'curve_params_dl': curve_params
                   }
        return {**results, **message}

    # def label_single(self, features=None):
    #     k = 'out5'
    #     features = {k: [features[k]]}
    #     df = pd.DataFrame.from_dict(features)
    #     return label(df)

    def prob_converter(self, prob):

        class_dict = {'Normal': 0, 'Chronic': 1, 'Acute': 2, 'T2MI': 3, 'T1MI': 4}

        prob_3c = - np.ones([prob.shape[0], 3], dtype=np.float)
        prob_3c[:, 0] = prob[:, class_dict['Chronic']] + prob[:, class_dict['Normal']]
        prob_3c[:, 1] = prob[:, class_dict['Acute']] + prob[:, class_dict['T2MI']]
        prob_3c[:, 2] = prob[:, class_dict['T1MI']]

        pred_3c = np.expand_dims(np.argmax(prob_3c, axis=1), axis=1)

        return pred_3c, prob_3c

    def reorder(self, prob):
        class_dict = {'Acute': 0, 'Chronic': 1, 'Normal': 2, 'T1MI': 3, 'T2MI': 4}
        new_order = ['Normal', 'Chronic', 'Acute', 'T2MI', 'T1MI']
        prob_new = np.ones(prob.shape) * -1
        for k, v in class_dict.items():
            prob_new[:, new_order.index(k)] = prob[:, v]
        assert np.sum(prob_new == -1) == 0

        return prob_new
