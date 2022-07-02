import os.path

import numpy as np
from path_utils import cache_root_dr


def get_trop_keys():
    trop_keys = ['trop0', 'trop1', 'trop2', 'trop3', 'trop4', 'trop5', 'trop6']
    return trop_keys


def get_fake_trop_keys():
    fake_trop_keys = ['trop7', 'trop8']
    return fake_trop_keys


def get_luke_trop_keys():
    luke_trop_keys = ['avgtrop', 'avgspd',
                      'maxtrop', 'mintrop',
                      'maxvel', 'minvel', 'divtrop',
                      'difftrop', 'diffvel', 'logtrop0']
    return luke_trop_keys


def get_onehot_keys():
    onehot_keys = []
    return onehot_keys


def get_onehot_codes():

    csv_path = os.path.join(cache_root_dr, 'data_raw_trop8_phys_master_onehot_encoding.npy')
    onehot_choices = np.load(csv_path, allow_pickle=True).item()

    onehot_keys = get_onehot_keys()
    onehot_codes = {k: onehot_choices[k] for k in onehot_keys}

    return onehot_codes


def get_bio_keys():
    bio_keys = ['age', 'mdrd_gfr']
    return bio_keys


def get_phys_keys():
    phys_keys = ['phys_fibrin', 'phys_urea', 'phys_urate', 'phys_albumin',
                 'phys_haeglob', 'phys_ph', 'phys_crp', 'phys_creat',
                 'phys_pco2', 'phys_bnp', 'phys_platev',
                 'phys_dimer', 'phys_platec', 'phys_hba1c', 'phys_wbc',
                 'phys_tsh', 'phys_ferritin', 'phys_lacta',
                 'phys_lactv']

    return phys_keys


def get_feature_len():
    num_raw_trops = len(get_trop_keys())
    num_phys = len(get_phys_keys())
    num_bios = len(get_bio_keys())
    num_luke_trops = len(get_luke_trop_keys())

    feature_len = {'raw_trop': num_raw_trops, 'time_trop': num_raw_trops,
                   'phys': num_phys, 'bio': num_bios, 'luke': num_luke_trops}

    return feature_len
