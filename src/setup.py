# coding: utf-8

import sys
from setuptools import setup, find_packages

NAME = "swagger_server"
VERSION = "4.0.2"

# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = ["connexion"]

setup(
    name=NAME,
    version=VERSION,
    description="Containerization of machine learning",
    author_email="zhibin.liao@adelaide.edu.au",
    url="",
    keywords=["Swagger", "Containerization of machine learning"],
    install_requires=REQUIRES,
    packages=find_packages(),
    package_data={'': ['swagger/swagger.yaml']},
    include_package_data=True,
    entry_points={
        'console_scripts': ['swagger_server=swagger_server.__main__:main']},
    long_description="""\
    Myocardial infarction risk predicting ML models released by Docker container as a web service. Additional requirements - such as 30-day death prediction and CABG/Intervention prediction at this point. It is believed that the requirements will keep growing. This project aims to adjust or create a new docker container every time a new ML model is built and be able to bulk process capacity if a user chooses to send package containing an arbitrary number of patient records.
    """
)

