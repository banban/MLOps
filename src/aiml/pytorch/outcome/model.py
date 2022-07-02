import torch
import torch.nn as nn
import numpy as np
import math
# from utils import plot, draw
from aiml.pytorch.outcome import protocol

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unsequeeze(nn.Module):

    def __init__(self, dim=-1):
        super(Unsequeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.unsqueeze(dim=self.dim)

        return x


class Flatten2(nn.Module):

    def __init__(self):
        super(Flatten2, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], -1)
        return x


class Scale(nn.Module):

    def __init__(self, scale=30):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class TroponinNet(nn.Module):

    def loss_func(self, logits, targets):
        return - (targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def network_troponin_profiles(self):
        return nn.Linear(4, 5)

    def network_troponin_profiles2(self, in_features):
        return nn.Linear(in_features, 20)

    def network_trops(self):

        num_units = 512
        modules = list()
        modules.append(Flatten())
        modules.append(nn.Linear(self.num_time_steps, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        modules.append(nn.Linear(num_units, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        modules.append(nn.Linear(num_units, self.num_time_steps))
        modules.append(nn.Tanh())

        net = nn.Sequential(*modules)

        return net

    def network_rnn(self):

        modules = list()
        # modules.append(Unsequeeze(dim=2))
        rnn = nn.LSTM(256, 128, 2)
        modules.append(rnn)

        net = nn.Sequential(*modules)

        return net

    def network_l1(self):

        num_units = 512
        modules = list()
        modules.append(Flatten())
        modules.append(nn.Linear(10, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        in_features = num_units

        net = nn.Sequential(*modules)

        return net, in_features

    def network_l2(self):

        num_units = 512
        modules = list()
        modules.append(Flatten())
        modules.append(nn.Linear(10 + 61, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_units, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        # modules.append(nn.Dropout())

        in_features = num_units

        net = nn.Sequential(*modules)

        return net, in_features

    def network_out5(self, use_luke):

        if use_luke:
            input_len = self.feature_len['luke'] * self.luke_multiplier + self.feature_len['phys'] + \
                        self.feature_len['bio'] + self.feature_len['onehot'] + \
                        + self.feature_len['angio'] * self.luke_multiplier
        else:
            input_len = 20 + self.feature_len['phys'] + \
                        self.feature_len['bio'] + self.feature_len['onehot'] + self.feature_len['angio']

        if self.data_version == 1:
            input_len += self.feature_len['onset']

        num_units = 512
        modules = list()
        modules.append(Flatten())
        modules.append(nn.Linear(input_len, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout())
        modules.append(nn.Linear(num_units, num_units))
        modules.append(nn.BatchNorm1d(num_units))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout())

        in_features = num_units

        net = nn.Sequential(*modules)

        return net, in_features

    def __init__(self, target_info):
        super(TroponinNet, self).__init__()

        self.num_time_steps = 12
        self.luke_multiplier = 100
        self.data_version = target_info['data_version']
        self.feature_len = protocol.get_feature_len(self.data_version)
        feature_start = np.cumsum([0] + list(self.feature_len.values())[:-1])
        feature_ends = np.cumsum(list(self.feature_len.values()))

        self.feature_arrangement = {n: [s, e] for n, s, e in zip(self.feature_len, feature_start, feature_ends)}

        self.target_info = target_info

        self.net_l1, in_features = self.network_l1()
        self.net_l2, in_features = self.network_l2()
        self.net_out5, in_features = self.network_out5(self.target_info['use_luke'])
        self.net_onset, in_features = self.network_out5(True)
        self.net_trop, in_features = self.network_out5(True)
        # self.net_rnn = self.network_rnn()
        # self.net_trops = self.network_trops()
        self.net_trop_prof = self.network_troponin_profiles()
        self.net_trop_prof2 = self.network_troponin_profiles2(in_features)

        # regression module
        modules = list()
        modules.append(nn.Linear(in_features, len(self.target_info['regression_cols'])))
        self.regressor = nn.Sequential(*modules)

        # binary classification module
        modules = list()
        modules.append(nn.Linear(in_features, len(self.target_info['binary_cls_cols'])))
        self.binary_classifier = nn.Sequential(*modules)

        # classification module
        self.classifiers = dict()
        for t_name in self.target_info['cls_cols_dict']:
            modules = list()
            if t_name == 'onset':
                modules.append(nn.Linear(in_features, 2))
            else:
                modules.append(nn.Linear(in_features, self.target_info['cls_cols_dict'][t_name]))
            classifier = nn.Sequential(*modules)
            self.add_module(t_name, classifier)
            self.classifiers[t_name] = classifier

        if 'onset' in target_info['cls_cols_dict']:
            from cdf_layer import CDFLayer
            # class 0: < 1, 1: 1-3, 2:3:6, 3:6-12, 4:12:24
            low = np.arange(start=0, stop=48, step=1)
            up = low + 1
            up = up / 48.
            low = low / 48.
            self.cdflayer = CDFLayer(num_input=in_features, up=up, low=low)

    def forward(self, input):

        input_dict = {k: input[:, v[0]:v[1]] for k, v in self.feature_arrangement.items()}

        if 'outl1l' in self.target_info['cls_cols_dict']:
            input_l1 = torch.cat([input_dict['luke']], dim=1)
            features_l1 = self.net_l1(input_l1)

        if 'out2l' in self.target_info['cls_cols_dict']:
            input_l2 = torch.cat([input_dict['luke'], input_dict['phys'],
                                  input_dict['bio'], input_dict['onehot'], input_dict['angio']], dim=1)
            features_l2 = self.net_l2(input_l2)

        if 'onset' in self.target_info['cls_cols_dict']:
            feature_names = ['luke'] * self.luke_multiplier + ['phys', 'bio', 'onehot']
            if self.data_version == 1:
                feature_names += ['onset']
            feature_names += ['angio'] * self.luke_multiplier

            input_onset = torch.cat([input_dict[k] for k in feature_names], dim=1)
            features_onset = self.net_onset(input_onset)

        if 'trop0' in self.target_info['regression_cols']:
            feature_names = ['luke'] * self.luke_multiplier + ['phys', 'bio', 'onehot']
            if self.data_version == 1:
                feature_names += ['onset']
            feature_names += ['angio'] * self.luke_multiplier
            input_trop = torch.cat([input_dict[k] for k in feature_names], dim=1)
            features_trop = self.net_trop(input_trop)

        cls_logits = dict()
        for c_name in self.target_info['cls_cols_dict']:
            classifier = self.classifiers[c_name]
            if c_name == 'outl1':
                cls_logits[c_name] = classifier(features_l1)
            elif c_name == 'outl2':
                cls_logits[c_name] = classifier(features_l2)
            elif c_name == 'out3c':
                probl1 = torch.softmax(cls_logits['outl1'], dim=1)
                probl2 = torch.softmax(cls_logits['outl2'], dim=1)
                probl3 = torch.cat([probl1[:, 0:1] * probl2[:, 0:1] + probl1[:, 0:1] * probl2[:, 1:],
                                    probl1[:, 1:] * probl2[:, 0:1],
                                    probl1[:, 1:] * probl2[:, 1:]], dim=1)

                cls_logits['out3c'] = probl3 / probl3.sum(dim=1, keepdim=True)
            elif c_name == 'out5':
                cls_logits['out5'] = None
            elif c_name == 'onset':
                mu_sigma = classifier(features_onset)
                mu_sigma = torch.sigmoid(mu_sigma)
                cdf_probs = self.cdflayer(mu_sigma)
                # class 0: < 1, 1: 1-3, 2:3:6, 3:6-12, 4:12:24
                onset_0 = cdf_probs[:, 0:1].mean(dim=1, keepdim=True)
                onset_1 = cdf_probs[:, 1:3].mean(dim=1, keepdim=True)
                onset_2 = cdf_probs[:, 3:6].mean(dim=1, keepdim=True)
                onset_3 = cdf_probs[:, 6:12].mean(dim=1, keepdim=True)
                onset_4 = cdf_probs[:, 12:24].mean(dim=1, keepdim=True)
                onset_5 = cdf_probs[:, 24:48].mean(dim=1, keepdim=True)
                cdf_probs = torch.cat([onset_0, onset_1, onset_2, onset_3, onset_4, onset_5], dim=1)
                cdf_probs = cdf_probs / cdf_probs.sum(dim=1, keepdim=True)
                cls_logits['onset'] = cdf_probs

        if 'trop0' in self.target_info['regression_cols']:
            # raw_trop = input_dict['raw_trop'].squeeze(dim=2)
            # prob_out5 = torch.softmax(cls_logits['out5'], dim=1)
            feature_names = ['time_trop', 'time_fake_trop']
            time_trop = torch.cat([input_dict[k] for k in feature_names], dim=1).squeeze(dim=2)

            selector = time_trop != -1e10
            if 'onset' in self.target_info['cls_cols_dict']:
                mu = - mu_sigma[:, 0:1].detach() * 2.
                time_trop = time_trop - mu.unsqueeze(dim=1)

            curve_params_flat = self.net_trop_prof2(features_trop)
            curve_params = torch.abs(curve_params_flat.view(-1, 4, 5))
            A, B, alpha, beta = curve_params[:, 0:1], curve_params[:, 1:2], curve_params[:, 2:3], curve_params[:, 3:4]

            # curve_params = self.net_trop_prof.weight
            # A, B, alpha, beta = curve_params[:, 0].view(1, 1, -1), \
            #                     curve_params[:, 1].view(1, 1, -1), \
            #                     curve_params[:, 2].view(1, 1, -1), \
            #                     curve_params[:, 3].view(1, 1, -1)

            # A = torch.abs(A)
            # B = torch.abs(B)
            # alpha = torch.abs(alpha)
            # beta = torch.abs(beta)
            time_trop = selector * time_trop
            # prob_out5 = prob_out5.unsqueeze(1).detach()
            trop = - A * torch.exp(-time_trop * alpha) + B * torch.exp(-time_trop * beta) + math.log(3)
            # trop = (prob_out5 * trop).sum(dim=2)
            regression_logits = trop

        if 'out5' in self.target_info['cls_cols_dict']:
            if self.target_info['use_luke']:
                feature_names = ['luke'] * self.luke_multiplier + ['phys', 'bio', 'onehot']
                if self.data_version == 1:
                    feature_names += ['onset']
                feature_names += ['angio'] * self.luke_multiplier
                input_5 = torch.cat([input_dict[k] for k in feature_names], dim=1)
            else:
                feature_names = ['phys', 'bio', 'onehot']
                if self.data_version == 1:
                    feature_names += ['onset']
                feature_names += ['angio']
                curve_params_as_features = curve_params_flat.view(list(curve_params_flat.shape) + [1, 1])
                input_5 = torch.cat([input_dict[k] for k in feature_names] + [curve_params_as_features], dim=1)

            features_out5 = self.net_out5(input_5)
            cls_logits['out5'] = classifier(features_out5)

        if 'onset' not in self.target_info['cls_cols_dict']:
            mu_sigma = None

        features = features_out5
        # regression_logits = self.regressor(features)
        binary_cls_logits = self.binary_classifier(features)

        return regression_logits, binary_cls_logits, [cls_logits[k] for k in cls_logits], mu_sigma, curve_params

    def loss(self, logits, targets):
        return self.loss_func(logits, targets)

    def prob(self, logits):
        return self.pred_func(logits)

    def pred(self, logits):
        return logits.detach().max(dim=1)[1]

    def correct(self, logits, targets):
        pred = self.pred(logits)
        return (pred == targets).sum().item()

    def get_cam(self, features):
        cls_weights = self.classifier[-1].weight
        cls_bias = self.classifier[-1].bias

        act_maps = list()
        for i in range(cls_weights.shape[0]):
            act_maps.append((features * cls_weights[i].view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
                            + cls_bias[i].view(1, -1, 1, 1))
        return torch.cat(act_maps, dim=1)

    def get_cam_fast(self, features, classifier):

        cls_weights = classifier[-1].weight
        cls_bias = classifier[-1].bias

        cls_weights = cls_weights.permute(1, 0)
        cls_weights = cls_weights.view(1, cls_weights.shape[0], 1, 1, cls_weights.shape[1])
        act_maps = (features.view(list(features.shape) + [1]) * cls_weights).sum(dim=1)
        act_maps = act_maps.permute(0, 3, 1, 2) + cls_bias.view(1, -1, 1, 1)

        return act_maps


def get_network(target_info):
    return TroponinNet(target_info)
