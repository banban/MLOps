#from __future__ import absolute_import
from random import seed, random, randint
import unittest
import json
import pandas as pd
import numpy as np
import torch
import requests
import sys
sys.path.insert(0, '.')
sys.path.insert(1, './aiml')
from aiml.models.v4.protocol import features_names
from swagger_server.test import BaseTestCase

DAY_MILLISECONDS = 24 * 60 * 60 * 1000


class TestPosts(BaseTestCase):
    #port = 5000

    def setUp(self):
        if not sys.warnoptions:
            import warnings
            warnings.simplefilter("ignore")
        self.setup_log()
        host = BaseTestCase.host
        self.send_post = lambda data: requests.post(
            f'{host}/predict', json=data)  # test all predictions
        self.send_post_variable = lambda data: requests.post(
            f'{host}/tools/parse_variables', json=data)
        self.send_post_dl = lambda data: requests.post(
            f'{host}/predict/cardiac_diagnosis/dl', json=data)
        self.send_post_xgb = lambda data: requests.post(
            f'{host}/predict/cardiac_diagnosis/xgb', json=data)
        self.send_post_revasc_dl = lambda data: requests.post(
            f'{host}/predict/revasc/dl', json=data)
        self.send_post_revasc_xgb = lambda data: requests.post(
            f'{host}/predict/revasc/xgb', json=data)
        self.send_post_event = lambda data: requests.post(
            f'{host}/predict/event_30day/xgb', json=data)
        self.send_ping = lambda data: requests.get(
            f'{host}/ping', data=data)
        self.send_check = lambda data: requests.get(
            f'{host}/health_check', data=data)

        self.test_cases = pd.read_csv(
            './swagger_server/test/test_cases.csv').fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.test_cases = [dict(c) for _, c in self.test_cases.iterrows()]
        #self.test_case = self.test_cases[0]

    def test_connection(self):
        data = {'test': 'GET'}
        result = self.send_ping(data)
        self.assertEqual(result.ok, True)

        result = self.send_check(data)
        self.assertEqual(result.ok, True)
        config = result.json()
        self.assertEqual(config.__len__() > 0, True,
                         'model(s) are cached by service /')

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_cuda(self):
        """https://github.com/facebookresearch/metaseq/blob/main/gpu_tests/test_fp16_optimizer.py"""
        self.assertEqual(torch.cuda.is_initialized(),
                         True, 'CUDA is initialized /')
        pass

    def test_post_allcases(self):
        '''Test all cases and check sizes
        '''
        data = self.test_cases.copy()
        result = self.send_post_variable(data)
        self.assertEqual(result.ok, True)
        output = result.json()
        # print(f'input: {data.__len__()}, output: {output.__len__()}')
        self.assertEqual(data.__len__() == output.__len__(), True,
                         'input and output sizes matches /')

        data = self.test_cases.copy()
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        output = result.json()
        #print(f'input: {data.__len__()}, output: {output.__len__()}')
        self.assertEqual(data.__len__() == output.__len__(), True,
                         'input and output sizes matches /')

    # def test_compare_outputs(self):

    def test_post_ddos(self):
        '''Test threshoilding and check sizes
        '''
        big_data = []
        seed(1)
        for index in range(0, 1000):
            #seed, random, randint
            data = {"age": randint(1, 100), "gender": bool(randint(0, 1)), "angiogram": 0, "logtrop0": random(), "trop0": randint(
                1, 50), "time_trop0": randint(1, 17280000), "trop1": randint(1, 50), "time_trop1": randint(1, 17280000)}
            big_data.append(data)
        result = self.send_post_revasc_dl(big_data)
        self.assertEqual(result.ok, True)
        output = result.json()
        # print(f'input: {big_data.__len__()}, output: {output.__len__()}')
        self.assertEqual(big_data.__len__() > output.__len__(), True,
                         'big input shrinked output sizes /')

    def test_post_case1(self):
        # self.assertEqual(np.all([fn in test_case for fn in features_names]), True)

        data = self.test_cases[0].copy()
        data['idx'] = 0

        result = self.send_post_variable(data)
        self.assertEqual(result.ok, True)

        # print(data)
        result = self.send_post_dl(data)
        self.assertEqual(result.ok, True)
        #logging.info(f'test_post_case1.1 status:{result.status_code}, duration:{result.elapsed}')

        result = self.send_post_xgb(data)
        self.assertEqual(result.ok, True)
        #logging.info(f'test_post_case1.2 status:{result.status_code}, duration:{result.elapsed}')

        result = self.send_post_event(data)
        self.assertEqual(result.ok, True)
        #logging.info(f'test_post_case1.3 status:{result.status_code}, duration:{result.elapsed}')

        # response_dict = json.loads(result.text.replace("'", '"'))
        # for k in known_response:
        #     self.assertEqual(known_response[k], response_dict[k])

    def test_send_unused_features(self):
        data = self.test_cases[0].copy()
        data['idx'] = 0

        result = self.send_post_variable(data)
        self.assertEqual(result.ok, True)
        # print(result.text)
        response_dict = json.loads(result.text.replace("'", '"'))
        unused_keys = ['Unnamed: 0', 'cohort_id', 'ds', 'supercell_id', 'subjectid',
                       'avgtrop', 'avgspd', 'maxtrop', 'mintrop', 'maxvel', 'minvel', 'divtrop', 'difftrop', 'diffvel', 'logtrop0',
                       'out5', 'out3c', 'outl1', 'outl2', 'event_mi', 'event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi',
                       'event_dead', 'event_dmi30d',
                       'quantized_trop_0-2', 'quantized_trop_2-4', 'quantized_trop_4-6', 'quantized_trop_6-8', 'quantized_trop_8-10',
                       'quantized_trop_10-12', 'quantized_trop_12-14', 'quantized_trop_14-16', 'quantized_trop_16-18',
                       'quantized_trop_18-20', 'quantized_trop_20-22', 'quantized_trop_22-24', 'set', 'idx']
        # print(f'expected unused_keys: {unused_keys}')
        # print(f'response unused_keys: {response_dict["unused_query_keys"]}')
        # print(f'returned: {response_dict["unused_query_keys"].__len__()}, expected: {unused_keys.__len__()}')
        self.assertEqual(
            response_dict["unused_query_keys"].__len__() == unused_keys.__len__(), True)

    def test_randomly_omitted_post_case1(self):
        # self.assertEqual(np.all([fn in test_case for fn in features_names]), True)

        # with 50% features
        ratio = 0.5
        random_keys = [features_names[i] for i in np.random.permutation(
            len(features_names))][:int(len(features_names)*ratio)]

        data = {k: self.test_cases[0][k]
                for k in self.test_cases[0] if k in random_keys}
        data['idx'] = 0
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual(
            np.all([k not in data for k in response_dict['unmatched_query_keys']]), True)
        self.assertEqual(
            np.all([k in data for k in response_dict['matched_query_dict'].keys()]), True)

        # with one random feature
        random_keys = [features_names[i]
                       for i in np.random.permutation(len(features_names))][0]

        data = {k: self.test_cases[0][k]
                for k in self.test_cases[0] if k in random_keys}
        data['idx'] = 0
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual(
            np.all([k not in data for k in response_dict['unmatched_query_keys']]), True)
        self.assertEqual(
            np.all([k in data for k in response_dict['matched_query_dict'].keys()]), True)

    def test_no_idx_post_case1(self):
        # self.assertEqual(np.all([fn in test_case for fn in features_names]), True)
        data = self.test_cases[0].copy()
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        # response_dict = json.loads(result.text.replace("'", '"'))
        # for k in known_response:
        #     self.assertEqual(known_response[k], response_dict[k])

    def test_null_post(self):
        test_case = {fn: 'nan' for fn in features_names}

        data = self.test_cases[0].copy()
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        # response_dict = json.loads(result.text.replace("'", '"'))
        # for k in known_response:
        #     self.assertEqual(known_response[k], response_dict[k])

    def test_age_only_varied_post(self):

        data = dict()
        for age in range(10, 90, 10):
            data['age'] = age
            # print(data)
            result = self.send_post(data)
            self.assertEqual(result.ok, True)
            response_dict = json.loads(result.text.replace("'", '"'))
            # print(response_dict)
            # for k in known_response:
            #     self.assertEqual(known_response[k], response_dict[k])
            self.assertEqual('error_message' not in response_dict, True)

    def test_single_trop_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 20,
            "time_trop1": 0.20 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_nil_trop_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_two_digits_trop_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 20,
            "time_trop1": 0.20 * DAY_MILLISECONDS,
            "trop11": 20,
            "time_trop11": 0.20 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_trop_time_large_out_of_bound_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop11": 20,
            "time_trop11": 2 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' in response_dict, True)  # Failed!

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 20,
            "time_trop1": 0.5 * DAY_MILLISECONDS,
            "trop11": 20,
            "time_trop11": 2 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('warning_message' in response_dict, True)

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop11": 20,
            "time_trop11": -0.1 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' in response_dict, True)

    def test_trop_time_naming_mismatch_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 20,
            "time_trop11": 0.2 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' in response_dict, True)  # Fails!

    def test_trop_order_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 3,
            "trop2": 3,
            "trop11": 3,
            "trop21": 4,
            "time_trop1": 0.1 * DAY_MILLISECONDS,
            "time_trop2": 0.2 * DAY_MILLISECONDS,
            "time_trop11": 0.3 * DAY_MILLISECONDS,
            "time_trop21": 0.4 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_trop_lower_bound_post(self):

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 2.99,
            "time_trop1": 0.4 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('warning_message' in response_dict, True)  # fails

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": -0.01,
            "time_trop1": 0.4 * DAY_MILLISECONDS,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual('error_message' in response_dict, True)

    def test_lukah_t1mi_case(self):
        data = {
            "age": 70,
            "gender": 1,
            "angiogram": 1,
            "trop1": 30,
            "time_trop1": 3600000,
            "trop2": 200,
            "time_trop2": 7200000,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual(response_dict['l1_pred_xgb'], 1)  # fails!
        self.assertEqual(response_dict['l2_pred_xgb'], 1)
        self.assertEqual(response_dict['p3_pred_xgb'], 2)

        data = {
            "age": 60,
            "gender": 0,
            "angiogram": 0,
            "trop1": 5,
            "time_trop1": 3600000,
            "trop2": 10,
            "time_trop2": 7200000,
        }

        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertEqual(response_dict['l1_pred_xgb'], 0)
        self.assertEqual(response_dict['l2_pred_xgb'], 0)
        self.assertEqual(response_dict['p3_pred_xgb'], 0)

    def test_joey_wbc_case(self):
        data = {"idPatient": [40876], "idEvent": [4527856], "age": [67.95131278538813], "gender": [0.0], "angiogram": [0.0],
                "time_trop1": [0.0], "trop1": [69.0], "time_trop2": [31500000.0], "trop2": [329.0], "phys_albumin": [19.0],
                "phys_bnp": [1753.0], "phys_creat": [112.0], "phys_crp": [286.0], "phys_haeglob": [147.0], "phys_lacta": [1.8],
                "phys_lactv": [2.1], "phys_pco2": [30.0], "phys_ph": [7.38], "phys_platec": [17.0], "phys_platev": [9.3],
                "phys_po2": [87.0], "phys_tsh": [0.72], "phys_urea": [8.4], "phys_urate": [0.28], "phys_wbc": [0.0],
                "mdrd_gfr": [44.0], "prioracs": [0.0], "priorami": [0.0], "priorcabg": [0.0], "priorcopd": [1.0],
                "priorcva": [0.0], "priordiab": [0.0], "priorhf": [0.0], "priorhtn": [0.0], "priorhyperlipid": [0.0],
                "priorpci": [0.0], "priorrenal": [0.0], "priorsmoke": [1.0]}

        for k in data:
            data[k] = data[k][0]
        data['phys_wbc'] = 0.01
        # print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # print(response_dict)
        self.assertNotEqual(str(response_dict['l1_prob_dl']), 'nan')  # fails!
        self.assertNotEqual(str(response_dict['l2_prob_dl']), 'nan')


if __name__ == '__main__':
    # if len(sys.argv) == 2:
    #     TestPosts.port = int(argv[1])
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    unittest.main()
