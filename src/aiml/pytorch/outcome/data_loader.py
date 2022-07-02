import torch
import torch.utils.data as data
import pandas as pd
from torchvision import transforms as T
import numpy as np
# from utils import plot
from aiml.pytorch.outcome import protocol


class TroponinDataset(data.Dataset):

    def __init__(self, csv_path,
                 target_info, target_translator=None,
                 set_key=None, set_scope=None,
                 use_random_crop=None, transform=None):
        super(TroponinDataset, self).__init__()

        self.use_random_crop = use_random_crop
        self.transform = transform
        self.post_transform = T.Compose([])

        self.df = pd.read_csv(csv_path)
        strparts = csv_path.split('.')[:-1]
        self.stats = pd.read_csv('{}_stats.csv'.format('.'.join(strparts)), index_col=0)
        self.target_info = target_info

        for k in self.df:
            if k in self.target_info['binary_cls_cols'] or k in self.target_info['cls_cols_dict']:
                variations = list(self.df[k].loc[self.df[k].notna()].unique())

                num_variations = len(variations)

                if num_variations < 100:
                    print('[{}] [{} variations]: {}'.format(k, num_variations, variations))
                else:
                    print('[{}] [{} variations]'.format(k, num_variations))

        self.ignore_value = -1e10
        self.df.fillna(self.ignore_value, inplace=True)
        self._class2index(target_translator)

        selected = None
        for s in set_scope:
            subset = s
            subset_selection = self.df[set_key] == subset
            if selected is None:
                selected = subset_selection
            else:
                selected = selected | subset_selection

        self.df = self.df.loc[selected]
        # self.df = self.df.drop(columns=['set{}'.format(b) for b in range(50)])

        self.trop_keys = protocol.get_trop_keys()
        self.fake_trop_keys = protocol.get_fake_trop_keys()

        self.luke_trop_keys = protocol.get_luke_trop_keys()
        self.luke_mean = torch.tensor(np.array(self.stats[self.luke_trop_keys].loc['mean']), dtype=torch.float).reshape(-1, 1,
                                                                                                                 1)
        self.luke_std = torch.tensor(np.array(self.stats[self.luke_trop_keys].loc['std']), dtype=torch.float).reshape(-1, 1, 1)

        self.onehot_keys = protocol.get_onehot_keys(target_info['data_version'])
        self.onehot_choices = np.load('{}_onehot_encoding.npy'.format('.'.join(csv_path.split('.')[:-1])),
                                      allow_pickle=True).item()

        # self.phys_keys = [k for k in self.df.keys() if 'phys_' in k]
        self.phys_keys = protocol.get_phys_keys()
        self.phys_mean = torch.tensor(np.array(self.stats[self.phys_keys].loc['mean']), dtype=torch.float).reshape(-1, 1, 1)
        self.phys_std = torch.tensor(np.array(self.stats[self.phys_keys].loc['std']), dtype=torch.float).reshape(-1, 1, 1)

        # self.quantized_trop_keys = [k for k in self.df.keys() if 'quantized_trop' in k]
        self.quantized_trop_keys = protocol.get_quantized_trop_keys()

        self.bio_keys = protocol.get_bio_keys(target_info['data_version'])
        self.bio_mean = torch.tensor(np.array(self.stats[self.bio_keys].loc['mean']), dtype=torch.float).reshape(-1, 1, 1)
        self.bio_std = torch.tensor(np.array(self.stats[self.bio_keys].loc['std']), dtype=torch.float).reshape(-1, 1, 1)

        if self.target_info['model_version'] == 'imp':
            self.regression_keys = [k for k in self.target_info['regression_cols']]
            self.regression_mean = torch.tensor(np.array(self.stats[self.regression_keys].loc['mean']), dtype=torch.float)
            self.regression_std = torch.tensor(np.array(self.stats[self.regression_keys].loc['std']), dtype=torch.float)

        print('Dataset size: {}'.format(len(self.df)))

        self.df.reset_index(drop=True)

    def _class2index(self, target_translator):

        compute_translator = False
        if target_translator is not None:
            self.target_translator = target_translator
        else:
            self.target_translator = {'binary_cls_cols': dict(),
                                      'cls_cols_dict': dict()}
            compute_translator = True

        for t in self.target_info['binary_cls_cols']:
            if compute_translator:
                choices = [c for c in self.df[t].unique() if c != self.ignore_value]
                choices = np.sort(choices)
                indices = np.arange(len(choices))
                translator = dict(zip(choices, indices))
                self.target_translator[t] = translator

            for c in self.target_translator[t]:
                self.df.loc[self.df[t] == c, t] = self.target_translator[t][c]

        for t in self.target_info['cls_cols_dict']:
            if compute_translator:
                choices = [c for c in self.df[t].unique() if c != self.ignore_value]
                choices = np.sort(choices)
                indices = np.arange(len(choices))
                translator = dict(zip(choices, indices))
                self.target_translator[t] = translator

            for c in self.target_translator[t]:
                self.df.loc[self.df[t] == c, t] = self.target_translator[t][c]

    def class2index(self, df):
        df = df.copy()
        for t in self.target_info['binary_cls_cols']:
            for c in self.target_translator[t]:
                df.loc[df[t] == c, t] = self.target_translator[t][c]

        for t in self.target_info['cls_cols_dict']:
            for c in self.target_translator[t]:
                df.loc[df[t] == c, t] = self.target_translator[t][c]

        return df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        info = self.df.iloc[index]
        return self.get_item(info)

    def get_item(self, info):

        # quantized troponin features, np.log-ed
        feature = np.array([info[k] for k in self.quantized_trop_keys])
        feature[feature == self.ignore_value] = 1
        feature = np.log(feature)
        feature[np.isnan(feature)] = 0
        feature = torch.tensor(feature, dtype=torch.float).view(-1, 1, 1)

        # first six raw troponin features, # np.log-ed
        feature_raw_trop = np.array([info[k] for k in self.trop_keys])
        # selector = feature_raw_trop == self.ignore_value
        # feature2[feature2 == self.ignore_value] = 1
        # feature2 = np.log(feature2)
        # feature2[np.isnan(feature2)] = 0
        # feature_raw_trop[selector] = 0
        feature_raw_trop = torch.tensor(feature_raw_trop, dtype=torch.float).view(-1, 1, 1)

        # first six raw troponin time
        time_raw_trop = np.array([info['time_{}'.format(k)] for k in self.trop_keys])
        time_raw_trop = torch.tensor(time_raw_trop, dtype=torch.float).view(-1, 1, 1)

        feature_fake_trop = np.array([info[k] for k in self.fake_trop_keys])
        feature_fake_trop = torch.tensor(feature_fake_trop, dtype=torch.float).view(-1, 1, 1)

        time_fake_trop = np.array([info['time_{}'.format(k)] for k in self.fake_trop_keys])
        time_fake_trop = torch.tensor(time_fake_trop, dtype=torch.float).view(-1, 1, 1)

        # luke's troponin features
        feature_luke = torch.tensor(np.array(np.array([info[k] for k in self.luke_trop_keys])).reshape(-1, 1, 1),
                                    dtype=torch.float)
        selector = feature_luke == self.ignore_value
        feature_luke = (feature_luke - self.luke_mean) / self.luke_std
        feature_luke[selector] = 0

        features_bio = torch.tensor(np.array(np.array([info[k] for k in self.bio_keys])).reshape(-1, 1, 1),
                                    dtype=torch.float)
        selector = features_bio == self.ignore_value
        features_bio = (features_bio - self.bio_mean) / self.bio_std
        features_bio[selector] = 0

        features_phys = torch.tensor(np.array(np.array([info[k] for k in self.phys_keys])).reshape(-1, 1, 1),
                                     dtype=torch.float)
        selector = features_phys == self.ignore_value
        features_phys = (features_phys - self.phys_mean) / self.phys_std
        features_phys[selector] = 0

        # normalizes if no value exists in all onehot options for each variable, i.e., [0.5, 0.5]
        # onehots = [self.onehot_choices[k] == info[k]
        #            if info[k] != self.ignore_value
        #            else np.ones(self.onehot_choices[k].shape) / self.onehot_choices[k].shape[0]
        #            for k in self.onehot_keys]
        # or simply keep no value as emtpy onehot, i.e., [0, 0]
        onehots = [self.onehot_choices[k] == info[k] for k in self.onehot_keys]
        features_1hot = np.concatenate(onehots, axis=0)
        features_1hot = torch.tensor(features_1hot.reshape(-1, 1, 1), dtype=torch.float)

        # quantized trop: 12, raw trop: 6, time trop: 6, phys: 21, bio: 5, onehot: 35, luke: 10
        feature = torch.cat([feature,
                             feature_raw_trop, time_raw_trop,
                             feature_fake_trop, time_fake_trop,
                             features_phys, features_bio, features_1hot, feature_luke], dim=0)

        targets = list()
        regression_targets = [torch.tensor([info[c]], dtype=torch.float) for c in self.target_info['regression_cols']]
        regression_targets = torch.cat(regression_targets, 0)

        if self.target_info['model_version'] == 'imp':
            selector = regression_targets == self.ignore_value
            regression_targets = (regression_targets - self.regression_mean) / self.regression_std
            regression_targets[selector] = self.ignore_value

        targets.extend([regression_targets])
        targets.extend([torch.tensor([info[c]], dtype=torch.float) for c in self.target_info['binary_cls_cols']])
        targets.extend([self.onehot(c, info[c]) for c in self.target_info['cls_cols_dict']])
        targets = torch.cat(targets, 0)

        return feature, targets

    def onehot(self, c, v):
        size = self.target_info['cls_cols_dict'][c]
        output = torch.zeros(size, dtype=torch.float)
        v = int(v)
        if v == self.ignore_value:
            output += self.ignore_value
        else:
            output[v] = 1
        return output


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    features, targets = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    features = torch.stack(features, 0)
    targets = torch.stack(targets, 0)
    return features, targets


def get_loader_from_dataset(csv_path, target_info, target_translator, batch_size, transform, set_key, set_scope,
                            use_random_crop, shuffle, num_workers, drop_last):
    dataset = TroponinDataset(
                              csv_path=csv_path,
                              target_info=target_info, target_translator=target_translator,
                              set_key=set_key, set_scope=set_scope,
                              use_random_crop=use_random_crop, transform=transform,
                              )

    # Data loader for wsi dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length)
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              collate_fn=collate_fn)

    return data_loader
