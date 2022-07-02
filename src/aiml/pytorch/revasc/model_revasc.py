import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import numpy as np
import math
# from utils import plot, draw
import aiml.pytorch.revasc.protocol as protocol


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

    def network(self):

        input_len = self.feature_len['luke'] * self.luke_multiplier + self.feature_len['phys'] + self.feature_len['bio']

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

        self.feature_len = protocol.get_feature_len()
        feature_start = np.cumsum([0] + list(self.feature_len.values())[:-1])
        feature_ends = np.cumsum(list(self.feature_len.values()))

        self.feature_arrangement = {n: [s, e] for n, s, e in zip(self.feature_len, feature_start, feature_ends)}

        self.target_info = target_info
        self.luke_multiplier = self.target_info['luke_multiplier']

        self.net, in_features = self.network()
        # print(self.net)

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
            modules.append(nn.Linear(in_features, self.target_info['cls_cols_dict'][t_name]))
            classifier = nn.Sequential(*modules)
            self.add_module(t_name, classifier)
            self.classifiers[t_name] = classifier

    def forward(self, input):

        input_dict = {k: input[:, v[0]:v[1]] for k, v in self.feature_arrangement.items()}

        feature_names = ['luke'] * self.luke_multiplier + ['phys', 'bio']
        input = torch.cat([input_dict[k] for k in feature_names], dim=1)
        features = self.net(input)

        cls_logits = dict()
        for c_name in self.target_info['cls_cols_dict']:
            classifier = self.classifiers[c_name]
            cls_logits[c_name] = classifier(features)

        regression_logits = self.regressor(features)
        binary_cls_logits = self.binary_classifier(features)

        return regression_logits, binary_cls_logits, [cls_logits[k] for k in cls_logits]

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
