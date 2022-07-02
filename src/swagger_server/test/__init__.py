import logging
from unittest import TestCase
from flask_testing import TestCase
import connexion
from flask import current_app
from swagger_server.encoder import JSONEncoder
from swagger_server.util import get_config_data


class BaseTestCase(TestCase):
    host = 'http://localhost:5000/rapidx_ai/api/public/analytics/zl'

    def setup_log(self):  # noqa: E501
        # print("setup logging...")
        logging.basicConfig(filename='./test.log',
                            encoding='utf-8', level=logging.INFO)
        logging.getLogger('connexion.operation').setLevel('ERROR')

    def tearDown(self):
        # print("disable logging...")
        logging.disable(logging.NOTSET)

    def create_app(self):
        """
        Simulate API here, with configuration you need.
        """
        # print(__name__)
        app = connexion.App(__name__, specification_dir='../swagger/')
        app.app.json_encoder = JSONEncoder
        app.add_api('swagger.yaml')

        models, features = get_config_data("config.json")

        with app.app.app_context():
            current_app.models = models
            current_app.features = features
            current_app.files = {}  # lazy load when used
            current_app.VERSION = "4.0.2"

        return app.app
