from rec2vec import logger

import yaml


def load_config(path: str = './rec2vec/configs/graph_config.yaml') -> dict:
    """
    Loads a yaml config and returns dictionary containing the configuration.

    :param path:    path to config file
    :return:        dictionary with contents of config file
    """

    logger.debug(f'load_config({path})')

    with open(path) as config:
        return yaml.safe_load(config)
