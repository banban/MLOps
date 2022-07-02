#from ast import IsNot
import connexion
from flask import current_app
from swagger_server.controllers import inference_protocol  # noqa: E501


### AIML interfaces ###
def inference_health_check():
    """health check
    used by rapidx_ai endpoint zl/health_check # noqa: E501
    :rtype: None
    """
    with current_app.app_context():
        result = current_app.models
    return result


def inference_predict(body, type="all", category=None, version=None, idx=-1):  # noqa: E501
    """Predict list of inferences with given input array
    used by rapidx_ai endpoints: zl/predict/... # noqa: E501

    :param body: List of patient records
    :type body: list | bytes | str
    :param type: Type of prediction
    :type type: str
    :param category: Category for prediction
    :type category: str
    :param version: Model version for prediction
    :type version: int
    :param idx: number of models for prediction, -1 use all
    :type idx: int

    :rtype: InferenceResult | List[InferenceResult]
    """

    records = []
    if connexion.request.is_json:
        # does not work for complex objects with nested arrays
        #records = [PatientRecord.from_dict(d) for d in connexion.request.get_json()]  # noqa: E501
        #records = [PatientRecord(config=d) for d in connexion.request.get_json()]  # noqa: E501
        json_obj = connexion.request.get_json()  # noqa: E501
        if isinstance(json_obj, list):
            # throttling or paging ?
            if json_obj.__len__() > 500:
                json_obj = json_obj[0:500]
            records = [d for d in json_obj]
        elif isinstance(json_obj, dict):
            records.append(json_obj)

    results = []
    for record in records:
        #logging.debug(f'record: {record}, category: {category}, record: {record} ')
        if isinstance(record, list):  # array
            for listRecord in record:
                result = inference_protocol.predict(
                    type, category, listRecord, version, idx)
                results.append(result)
        elif isinstance(record, dict):  # single record
            result = inference_protocol.predict(
                type, category, record, version, idx)
            results.append(result)

    # return 1st object if input is single record
    if results.__len__() == 1 and isinstance(record, dict):
        return results[0]
    # return array if input is array
    return results


### HeartAI API interfaces for compartibility ###
def inference_ping():
    return {"msg": "Hello world!"}


def inference_pingByPost(body):
    """body is ignored"""
    return inference_ping()


def inference_predict_parse_variables(body):  # noqa: E501
    return inference_predict(body, type="parse_variables", category=None, version=None)


def inference_predict_outcome_dl(body):  # noqa: E501
    return inference_predict(body, type="outcome", category="dl", version=3)


def inference_predict_outcome_xgb(body):  # noqa: E501
    return inference_predict(body, type="outcome", category="xgb", version=3)


def inference_predict_cardiac_diagnosis_dl(body):  # noqa: E501
    return inference_predict(body, type="outcome", category="dl", version=3)


def inference_predict_cardiac_diagnosis_xgb(body):  # noqa: E501
    return inference_predict(body, type="outcome", category="xgb", version=3)


def inference_predict_event_30day_xgb(body):  # noqa: E501
    return inference_predict(body, type="event", category="dmi30d", version=3)


def inference_predict_event_xgb(body):  # noqa: E501
    return inference_predict(body, type="event", category="dmi30d", version=3)


def inference_predict_revasc_xgb(body):  # noqa: E501
    return inference_predict(body, type="revasc", category="xgb", version=4)


def inference_predict_revasc_dl(body):  # noqa: E501
    return inference_predict(body, type="revasc", category="dl", version=4)
