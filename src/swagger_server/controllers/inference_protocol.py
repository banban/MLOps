import logging
from flask import current_app

# old models dependencies
from aiml.models.v3.model_dl import DLModel
from aiml.models.v3.model_xgb import XGBoostModel
from aiml.models.v3.model_events import XGBoostEventModel
from aiml.models.v3.input_processing import process_variables, fill_missing_variables

# new models dependencies
from aiml.models.v4.model_revasc_dl import RevascDLModel
from aiml.models.v4.model_revasc_xgb import RevascXGBoostModel
# from aiml.models.v4.protocol import revasc_xgb_feature_orders
#from aiml.pytorch.revasc.model_revasc import TroponinNet
#from xgboost import Booster
#from swagger_server.models.model import Model  # noqa: E501


def predict(type: str, category: str, record: list[any], version=None, idx: int = -1):
    '''Generalized method of prediction combining logic from different types and versions models
    '''
    if type == "all":
        response_dict_dl = predict(
            type="outcome", category="dl", record=record.copy(), version=3, idx=idx)
        response_dict_xgb = predict(
            type="outcome", category="xgb", record=record.copy(), version=3, idx=idx)
        response_dict_events = predict(
            type="outcome", category="event", record=record.copy(), version=3, idx=idx)
        response_dict_revasc_dl = predict(
            type="revasc", category="dl", record=record.copy(), version=4, idx=idx)
        response_dict_revasc_xgb = predict(
            type="revasc", category="xgb", record=record.copy(), version=4, idx=idx)

        if isinstance(response_dict_revasc_xgb, dict):  # merge single records
            return {**response_dict_dl, **response_dict_xgb, **response_dict_events,
                    **response_dict_revasc_dl, **response_dict_revasc_xgb}
        if isinstance(response_dict_revasc_xgb, list):  # merge arrays
            return (response_dict_dl + response_dict_xgb + response_dict_events
                    + response_dict_revasc_dl + response_dict_revasc_xgb)

    files = dict()
    features = dict()
    with current_app.app_context():
        models_config = current_app.models
        files = current_app.files
        # features_config = current_app.features
        predictor_version = current_app.VERSION

    # extract idx from payload if defined
    if 'idx' in record:
        idx = record["idx"]
        # record.pop("idx") #failed test_send_unused_features expects that key
    # transform troponine array into linear features
    if 'troponine' in record:
        index = 0
        for item in record["troponine"]:
            record[f'trop{index}'] = item["value"]
            record[f'time_trop{index}'] = item["time"]
            index = index + 1
        record.pop("troponine")

    # #fill gaps for non existing features from config if required
    # feature_config = [f for f in features_config if f["version"] == version]
    # if (feature_config.__len__()>0):
    #     features_empty = dict.fromkeys(feature_config[0]["labels"], np.nan)
    #     features = {**features_empty, **features}

    response_dict = {}
    features, matched_query_dict, unmatched_query_keys, unused_query_keys = fill_missing_variables(
        record)
    response_dict['predictor_version'] = predictor_version
    response_dict['matched_query_dict'] = matched_query_dict
    response_dict['unmatched_query_keys'] = unmatched_query_keys
    response_dict['unused_query_keys'] = unused_query_keys
    try:
        features, message = process_variables(features)
        response_dict = {**response_dict, **message}
    except ValueError as inst:
        # logging.error(f'error_message: {inst.args[0]}; type: {type}, category: {category}, version: {version}, idx: {idx}')
        # logging.debug(f'record: {record}')
        response_dict[
            'error_message'] = f'Process failed due to the following reason: {inst.args[0]}'
        return response_dict

    if (type == "parse_variables"):
        return response_dict

    model_config = []
    if version is not None:
        model_config = [m for m in models_config if m["category"] == category
                        and m["version"] == version]
    else:
        model_config = [m for m in models_config if m["category"] == category]

    filename = None
    if model_config.__len__() > 0:
        filename = model_config[0]["filename"]

    model = None  # PredictionModel() <-Py310
    if filename is not None and filename not in files:
        logging.debug(f'Loading file into cache: {filename} ...')
        if (type == "outcome" and category == "xgb"):  # v3 models
            model = XGBoostModel(dump_path=filename)
        elif (type == "outcome" and category == "dl"):
            model = DLModel(dump_path=filename)
        elif (type == "event" and category == "dmi30d"):
            model = XGBoostEventModel(
                event_name=f'{type}_{category}', dump_path=filename)
        elif (type == "revasc" and category == "xgb"):  # v4 models
            model = RevascXGBoostModel(label_name='both', dump_path=filename)
        elif (type == "revasc" and category == "dl"):
            model = RevascDLModel(dump_path=filename)
            model.change_mode('cpu')

        # update context
        files[filename] = model
        with current_app.app_context():
            current_app.files = files

    if filename in files:
        model = files[filename]
        # logging.debug(f'using filename: {filename} in model class: {model.__class__.__name__}')
        results = model.inference_single(idx=idx, features=features)
        response_dict = {**results, **response_dict}
    return response_dict


### some future improvements...###
# from typing import Protocol, List, Dict, Any  # noqa: F401
# # Alternative syntax for unions requires Python 3.10 or newer Pylance
# class PredictionService(DLModel | XGBoostModel | XGBoostEventModel | RevascDLModel | RevascXGBoostModel):
#     pass
#     def inference_single(self, idx=0, features=None) -> InferenceResult:

# #Protocols cannot be instantiated
# class PredictionModel(Protocol):
#     def inference_single(self, idx=0, features=None) -> InferenceResult:
#         ...

# class PredictionService():
#     def predict(self, model: PredictionModel, idx=0, features=None) -> list[InferenceResult]:
#         result = list[InferenceResult]()
#         return result
