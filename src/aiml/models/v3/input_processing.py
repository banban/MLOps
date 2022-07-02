import numpy as np
from aiml.utils import get_luke_trop_features
from aiml.models.v3.protocol import features_names, trop_regex, time_trop_regex, luke_trop_features, phys_feature_log
import re


luke_trop_feats = {k: 'nan' for k in luke_trop_features}


def fill_missing_variables(query_dict):

    query_keys = list(query_dict.keys())
    matched_query_keys = list()
    for fn in features_names:
        matched_keys = [q for q in query_keys if re.match(fn, q)]
        if len(matched_keys) != 0 and fn != trop_regex and fn != time_trop_regex:
            matched_keys = [fn]
        matched_query_keys.extend(matched_keys)

    unmatched_query_keys = [f for f in features_names if f not in matched_query_keys + [trop_regex, time_trop_regex]]
    unused_query_keys = [f for f in query_keys if f not in matched_query_keys + [trop_regex, time_trop_regex]]
    matched_query_dict = {fn: query_dict[fn] for fn in matched_query_keys if fn in query_dict}
    unmatched_query_dict = {fn: 'nan' for fn in unmatched_query_keys}

    features = {**matched_query_dict, **unmatched_query_dict}
    return features, matched_query_dict, unmatched_query_keys, unused_query_keys


def process_variables(features):
    message = dict()

    trop_keys = [k for k in features.keys() if re.match(trop_regex, k)]
    trop_keys.sort()
    time_trop_keys = [k for k in features.keys() if re.match(time_trop_regex, k)]
    time_trop_keys.sort()

    if len(trop_keys) == 0:
        features_luke = luke_trop_feats
    else:
        # check if all trops are accompanied by time_ prefix
        if not all([re.match(r'^time_{}$'.format(k), tk) for k, tk in zip(trop_keys, time_trop_keys)]):
            raise ValueError('Not all trop variables are accompanied by time_ prefix')

        # recording troponin tests by the time, ignore the sorted order
        time_trops = np.array([features[k] for k in time_trop_keys]).astype(np.float)
        time_trops = time_trops / (1000. * 60. * 60. * 24.)
        trops = np.array([features[k] for k in trop_keys]).astype(np.float)

        if any(time_trops < 0.):
            raise ValueError('Time of troponin (time_trop) contain value(s) below 0 (prior to admission).')

        if any(time_trops > 1.):
            selector = time_trops <= 1.
            time_trops = time_trops[selector]
            trops = trops[selector]

            if len(time_trops) == 0:
                raise ValueError('Time of troponin (time_trop) contain value(s) larger than 24 hours since '
                                 'admission, which is allowed. However removing those result in nil troponin tests')
            else:
                message['warning_message'] = 'Time of troponin (time_trop) contain value(s) larger than 24 hours ' \
                                             'since admission. These troponin tests have been removed from analysis.'

        # check if trop has value less than 3
        if np.any(trops < 0):
            raise ValueError('Troponin values should not be less than 0')
        elif np.any((trops < 3) & (trops >= 0.)):
            trops = np.clip(trops, a_min=3., a_max=None)
            message['warning_message'] = 'Troponin values between 0 to 3 are elevated to 3.'

        if len(set(time_trops)) != len(time_trops):
            sel = np.ones(time_trops.shape)
            for t_idx, t in enumerate(time_trops):
                if not np.isnan(t):
                    if trops[t_idx] != np.max(trops[time_trops == t]):
                        sel[t_idx] = 0
            time_trops = time_trops[sel == 1]
            trops = trops[sel == 1]
            message['warning_message'] = 'Multiple troponin values are associated with the same timestamp. The one ' \
                                         'with the largest troponin value is retained.'

        time_trop_index = time_trops.argsort()
        time_trops = np.expand_dims(time_trops[time_trop_index], 0)
        trops = np.expand_dims(trops[time_trop_index], 0)

        # if only one troponin test is done, append a nan in both trops and
        # time_trops to make get_luke_trop_features working
        if trops.shape[1] == 1:
            trops = np.concatenate([trops, np.array([[np.nan]])], axis=1)
            time_trops = np.concatenate([time_trops, np.array([[np.nan]])], axis=1)

        luke_variables, luke_variable_names = get_luke_trop_features(trops, time_trops)
        features = {k: v for k, v in features.items() if k not in trop_keys + time_trop_keys}
        features_luke = {luke_variable_names[i]: luke_variables[0, i] if not np.isnan(luke_variables[0, i]) else 'nan'
                         for i in range(luke_variables.shape[1])}

    features = {**features, **features_luke}

    for k in phys_feature_log:
        if features[k] != 'nan':
            features[k] = np.log(max(float(features[k]), 0.01))

    return features, message
