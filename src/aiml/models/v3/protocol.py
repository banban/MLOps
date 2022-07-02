trop_regex = r'trop\d{1,2}'
time_trop_regex = r'time_trop\d{1,2}'

phys_feature_names = ['phys_pco2', 'phys_lacta', 'phys_po2', 'phys_ph', 'phys_bnp', 'phys_ckmb', 'phys_fibrin',
                      'phys_urea', 'phys_creat', 'phys_urate', 'phys_albumin', 'phys_haeglob', 'phys_wbc',
                      'phys_platec', 'phys_platev', 'phys_hba1c', 'phys_tsh', 'phys_crp', 'phys_ferritin',
                      'phys_dimer', 'phys_lactv']

# a subset of features in phys_features that does not need to be log-transformed
phys_feature_no_log = ['phys_albumin', 'phys_haeglob', 'phys_platev', 'phys_ph', 'phys_urate']

phys_feature_log = [p for p in phys_feature_names if p not in phys_feature_no_log]

event_feature_names = ['priorami', 'prioracs', 'priorangina', 'priorvt', 'priorcva', 'priorrenal', 'priorsmoke',
                       'priorcopd', 'priorpci', 'priorcabg', 'priordiab', 'priorhtn', 'priorhf', 'priorarrhythmia',
                       'priorhyperlipid']

features_names = \
  [trop_regex, time_trop_regex] + \
  phys_feature_names + \
  event_feature_names + \
  ['gender', 'age', 'angiogram', 'mdrd_gfr']

luke_trop_features = ['avgtrop', 'avgspd', 'maxtrop', 'mintrop',
                      'maxvel', 'minvel', 'divtrop', 'difftrop',
                      'diffvel', 'logtrop0']

xgb_feature_orders = ['avgtrop', 'avgspd', 'maxtrop', 'mintrop',
                      'maxvel', 'minvel', 'divtrop', 'difftrop',
                      'diffvel', 'logtrop1',
                      'phys_albumin', 'phys_bnp', 'phys_ckmb', 'phys_creat',
                      'phys_crp', 'phys_dimer', 'phys_ferritin', 'phys_fibrin',
                      'phys_haeglob', 'phys_hba1c', 'phys_lacta', 'phys_lactv',
                      'phys_pco2', 'phys_ph', 'phys_platec', 'phys_platev',
                      'phys_po2', 'phys_tsh', 'phys_urate', 'phys_urea',
                      'phys_wbc', 'priorami', 'prioracs', 'priorangina',
                      'priorvt', 'priorcva', 'priorrenal', 'priorsmoke',
                      'priorcopd', 'priorpci', 'priorcabg', 'priordiab',
                      'priorhtn', 'priorhf', 'priorarrhythmia', 'priorhyperlipid',
                      'gender', 'age', 'angiogram', 'mdrd_gfr'
                      ]

imputation_regression_cols = ['phys_fibrin',
 'phys_urea',
 'phys_urate',
 'phys_albumin',
 'phys_haeglob',
 'phys_ph',
 'phys_crp',
 'phys_creat',
 'phys_pco2',
 'phys_bnp',
 'phys_ckmb',
 'phys_platev',
 'phys_dimer',
 'phys_platec',
 'phys_hba1c',
 'phys_wbc',
 'phys_tsh',
 'phys_po2',
 'phys_ferritin',
 'phys_lacta',
 'phys_lactv',
 'age',
 'mdrd_gfr',
 'avgtrop',
 'avgspd',
 'maxtrop',
 'mintrop',
 'maxvel',
 'minvel',
 'divtrop',
 'difftrop',
 'diffvel',
 'logtrop0']


def feature_name_convertion(outbag_df):
    """
    Converting the feature names from what is defined by the endpoint service to what's used in the rapid-x algorithm
    This is for the revasularization data
    """

    phys_keys = {k: k.lower().replace('phys', 'phys_') for k in outbag_df.keys() if 'phys' in k}
    prior_keys = {k: k.lower().replace('hx', 'prior') for k in outbag_df.keys() if 'hx' in k}
    trop_keys = {k: k.lower().replace('tropvalue', 'trop') for k in outbag_df.keys() if 'tropValue' in k}
    time_trop_keys = {k: 'time_' + k.lower().replace('troptimems', 'trop') for k in outbag_df.keys() if
                      'tropTimeMS' in k}
    other_keys = {'age': 'age', 'angiogramIndicated': 'angiogram', 'genderMale': 'gender'}
    prior_keys['hxSmoking'] = 'priorsmoke'
    prior_keys['hxHyperlipidemia'] = 'priorhyperlipid'
    prior_keys['physEGFR'] = 'mdrd_gfr'
    prior_keys['hxDM'] = 'priordiab'
    converter = {**phys_keys, **prior_keys, **trop_keys, **time_trop_keys, **other_keys}
    new_keys = list(converter.values())
    # print([k for k, v in converter.items() if v not in features_names if 'trop' not in k])
    # print([k for k in features_names if k not in new_keys if 'trop' not in k])
    non = [k for k in outbag_df.keys() if k not in converter]
    outbag_df = outbag_df.rename(columns=converter)
    for k in list(prior_keys.values()) + ['gender', 'angiogram']:
        outbag_df[k] = outbag_df[k].astype(float)

    return outbag_df

