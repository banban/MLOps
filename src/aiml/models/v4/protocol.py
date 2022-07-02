from aiml.models.v3.protocol import *

revasc_exclude_feature_names = ['phys_ckmb', 'phys_po2', 'angiogram']
revasc_xgb_feature_orders = [k for k in xgb_feature_orders if k not in revasc_exclude_feature_names]