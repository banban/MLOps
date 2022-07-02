import connexion
import six

from swagger_server.models.model import Model  # noqa: E501
from swagger_server import util


def add_model(body):  # noqa: E501
    """Add a new model to the storage

     # noqa: E501

    :param body: Model object that needs to be added to the storage
    :type body: dict | bytes

    :rtype: None
    """
    if connexion.request.is_json:
        body = Model.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic3!'


def delete_model(modelId, api_key=None):  # noqa: E501
    """Deletes a model

     # noqa: E501

    :param modelId: Model id to delete
    :type modelId: int
    :param api_key: 
    :type api_key: str

    :rtype: None
    """
    return 'do some magic4!'


def find_models_by_tags(tags):  # noqa: E501
    """Finds Models by tags

    Muliple tags can be provided with comma separated strings. Use tag1, tag2, tag3 for testing. # noqa: E501

    :param tags: Tags to filter by
    :type tags: List[str]

    :rtype: List[Model]
    """
    return 'do some magic8!'


def find_models_by_version(version):  # noqa: E501
    """Finds Models by version

    Multiple version values can be provided with comma separated strings # noqa: E501

    :param version: Model&#39;s version values that need to be considered for inference
    :type version: 

    :rtype: List[Model]
    """
    return 'do some magic5!'


def get_model_by_id(modelId):  # noqa: E501
    """Find model by ID

    Returns a single model # noqa: E501

    :param modelId: ID of model to return
    :type modelId: int

    :rtype: Model
    """
    return 'do some magic9!'


def update_model(body):  # noqa: E501
    """Update an existing model

     # noqa: E501

    :param body: Model object that needs to be added to the storage
    :type body: dict | bytes

    :rtype: None
    """
    if connexion.request.is_json:
        body = Model.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic6!'


def update_model_with_form(modelId, filename=None, version=None):  # noqa: E501
    """Updates a model in the storage with form data

     # noqa: E501

    :param modelId: ID of model that needs to be updated
    :type modelId: int
    :param filename: Updated filename of the model
    :type filename: str
    :param version: Updated version of the model
    :type version: str

    :rtype: None
    """
    return 'do some magic10!'


def upload_file(modelId, additionalMetadata=None, file=None):  # noqa: E501
    """uploads a pickle file

     # noqa: E501

    :param modelId: ID of model to update
    :type modelId: int
    :param additionalMetadata: Additional data to pass to server
    :type additionalMetadata: str
    :param file: file to upload
    :type file: werkzeug.datastructures.FileStorage

    :rtype: None
    """
    return 'do some magic7!'
