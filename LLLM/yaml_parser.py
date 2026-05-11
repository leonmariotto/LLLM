"""
Module YamlParser
"""

import logging
from typing import Dict
import strictyaml
from collections.abc import Mapping

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


class YamlParserError(Exception):
    """
    Custom class for yaml parsing error
    """


class YamlParser:
    """
    Class YamlParser
    """

    def __init__(self):
        """
        Init YamlParser
        """
        self.logger = logging.getLogger(__name__)
        self.data: Dict = {}

    def parse(self, path: str):
        self.logger.debug("Parse YAML file [%s]", path)

        # Read file content into string
        try:
            with open(path, "r", encoding="utf-8") as yaml_file:
                yaml_text = yaml_file.read()
        except OSError as err:
            self.logger.error("Error opening the file [%s]", path)
            raise YamlParserError from err
        # Validate YAML and parse it into a dict
        try:
            yaml_data = strictyaml.load(yaml_text).data
        except strictyaml.YAMLError as err:
            self.logger.error("Error parsing the yaml [%s]", path)
            raise YamlParserError from err
        if isinstance(yaml_data, Mapping):
            self.data.update(yaml_data)
