from rec2vec import logger
from chardet import detect


def get_encoding(file: str) -> str:
    """
    Returns the (automatically detected) encoding of a file.

    :param file:    path to a file
    :return:        encoding of that file
    """

    logger.trace(f'get_encoding({file})')

    with open(file, 'rb') as f:
        return detect(f.read())['encoding']
