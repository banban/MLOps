# coding: utf-8
from __future__ import absolute_import
from swagger_server.test import BaseTestCase
from flask import json
#from six import BytesIO
#from swagger_server.models.inference_result import InferenceResult  # noqa: E501
#from swagger_server.models.patient_record import PatientRecord  # noqa: E501


class TestInferenceController(BaseTestCase):
    """InferenceController integration test stubs"""

    def test_inference_health_check(self):
        """Test case for inference_health_check
        """
        response = self.client.open(
            f'{BaseTestCase.host}/health_check',
            method='GET')
        print(response.data.decode('utf-8'))
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_inference_predict(self):
        """Test case for inference_predict
        Predict inference result with given input array
        """
        body = []
        query_string = [('type', 'all'),
                        ('age', 70),
                        ('gender', 1),
                        ('angiogram', 1),
                        ('trop1', 30),
                        ('time_trop1', 3600000),
                        ('trop2', 200),
                        ('time_trop2', 7200000)
                        ]
        response = self.client.open(
            f'{BaseTestCase.host}/predict',
            method='POST',
            data=json.dumps(body),
            content_type='application/json',
            query_string=query_string)
        print(response.data.decode('utf-8'))
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
