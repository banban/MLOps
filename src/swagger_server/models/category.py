# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class Category(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, code: str=None, name: str=None):  # noqa: E501
        """Category - a model defined in Swagger

        :param code: The code of this Category.  # noqa: E501
        :type code: str
        :param name: The name of this Category.  # noqa: E501
        :type name: str
        """
        self.swagger_types = {
            'code': str,
            'name': str
        }

        self.attribute_map = {
            'code': 'code',
            'name': 'name'
        }

        self._code = code
        self._name = name

    @classmethod
    def from_dict(cls, dikt) -> 'Category':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Category of this Category.  # noqa: E501
        :rtype: Category
        """
        return util.deserialize_model(dikt, cls)

    @property
    def code(self) -> str:
        """Gets the code of this Category.


        :return: The code of this Category.
        :rtype: str
        """
        return self._code

    @code.setter
    def code(self, code: str):
        """Sets the code of this Category.


        :param code: The code of this Category.
        :type code: str
        """

        self._code = code

    @property
    def name(self) -> str:
        """Gets the name of this Category.


        :return: The name of this Category.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this Category.


        :param name: The name of this Category.
        :type name: str
        """

        self._name = name
