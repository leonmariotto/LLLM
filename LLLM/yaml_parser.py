"""
Module YamlParser
"""

import logging
from typing import Any, cast
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

    def __init__(self) -> None:
        """
        Init YamlParser
        """
        self.logger = logging.getLogger(__name__)
        self.data: dict[str, Any] = {}

    def parse(self, path: str) -> None:
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
            strict_yaml = cast(Any, strictyaml)
            load_yaml = strict_yaml.load
            yaml_data = load_yaml(yaml_text).data
        except strictyaml.YAMLError as err:
            self.logger.error("Error parsing the yaml [%s]", path)
            raise YamlParserError from err
        if isinstance(yaml_data, Mapping):
            self.data.update(cast(Mapping[str, Any], yaml_data))
