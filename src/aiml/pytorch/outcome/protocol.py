import os.path

import numpy as np
from path_utils import cache_root, cache_root_d2


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


def get_onehot_keys(data_version):
    if data_version == 1:
        onehot_keys = ['smoking', 'gender', 'hst_priormi', 'hst_dm', 'hst_htn', 'hst_std', 'ischaemia_ecg',
                       'cad', 'dyslipid',
                       'fhx', 'hst_familyhx',
                       'hst_onsetrest', 'onset',
                       'hst_angio', 'angiogram']

    elif data_version == 2:
        event_priors = ['priorami', 'prioracs', 'priorangina', 'priorvt', 'priorcva', 'priorrenal', 'priorsmoke',
                        'priorcopd', 'priorpci', 'priorcabg', 'priordiab', 'priorhtn', 'priorhf', 'priorarrhythmia',
                        'priorhyperlipid']
        onehot_keys = event_priors + ['gender', 'angiogram']
    elif data_version == 3:
        onehot_keys = []
    return onehot_keys


def get_onehot_codes(data_version):
    if data_version == 1:
        csv_path = os.path.join(cache_root, 'data_raw_trop8_phys_onehot_encoding.npy')
        onehot_choices = np.load(csv_path, allow_pickle=True).item()

    elif data_version == 2:
        csv_path = os.path.join(cache_root_d2, 'data_raw_trop8_phys_onehot_encoding.npy')
        onehot_choices = np.load(csv_path, allow_pickle=True).item()

    onehot_keys = get_onehot_keys(data_version)
    onehot_codes = {k: onehot_choices[k] for k in onehot_keys}

    return onehot_codes


def get_bio_keys(data_version):
    if data_version == 1:
        bio_keys = ['age', 'hr', 'sbp', 'dbp', 'mdrd_gfr']
    elif data_version == 2:
        bio_keys = ['age', 'mdrd_gfr']
    return bio_keys


def get_phys_keys():
    phys_keys = ['phys_fibrin', 'phys_urea', 'phys_urate', 'phys_albumin',
                 'phys_haeglob', 'phys_ph', 'phys_crp', 'phys_creat',
                 'phys_pco2', 'phys_bnp', 'phys_ckmb', 'phys_platev',
                 'phys_dimer', 'phys_platec', 'phys_hba1c', 'phys_wbc',
                 'phys_tsh', 'phys_po2', 'phys_ferritin', 'phys_lacta',
                 'phys_lactv']
    return phys_keys


def get_quantized_trop_keys():
    quantized_trop_keys = ['quantized_trop_0-2',
                           'quantized_trop_2-4',
                           'quantized_trop_4-6',
                           'quantized_trop_6-8',
                           'quantized_trop_8-10',
                           'quantized_trop_10-12',
                           'quantized_trop_12-14',
                           'quantized_trop_14-16',
                           'quantized_trop_16-18',
                           'quantized_trop_18-20',
                           'quantized_trop_20-22',
                           'quantized_trop_22-24']
    return quantized_trop_keys


def get_feature_len(data_version):
    num_time_steps = len(get_quantized_trop_keys())
    num_raw_trops = len(get_trop_keys())
    num_fake_trops = len(get_fake_trop_keys())
    num_phys = len(get_phys_keys())
    num_bios = len(get_bio_keys(data_version))
    num_luke_trops = len(get_luke_trop_keys())
    onehot_codes = get_onehot_codes(data_version)
    num_onehot_codes = len(np.concatenate(list(onehot_codes.values()), axis=0))
    # feature arrangement
    if data_version == 1:
        feature_len = {'quantized_trop': num_time_steps,
                       'raw_trop': num_raw_trops, 'time_trop': num_raw_trops,
                       'fake_trop': num_fake_trops, 'time_fake_trop': num_fake_trops,
                       'phys': num_phys, 'bio': num_bios,
                       'onehot': num_onehot_codes - 4 - 6, 'onset': 6, 'angio': 4, 'luke': num_luke_trops}
    elif data_version == 2:
        feature_len = {'quantized_trop': num_time_steps,
                       'raw_trop': num_raw_trops, 'time_trop': num_raw_trops,
                       'fake_trop': num_fake_trops, 'time_fake_trop': num_fake_trops,
                       'phys': num_phys, 'bio': num_bios,
                       'onehot': num_onehot_codes - 2, 'angio': 2, 'luke': num_luke_trops}

    return feature_len
