import logging
import logging.config

from yaml import safe_load
from rec2vec.util.logger import addLoggingLevel


with open('./rec2vec/configs/logger.yaml') as f:
    config = safe_load(f)

logging.config.dictConfig(config=config)
addLoggingLevel(level_name='TRACE', level_num=logging.DEBUG - 5)
logger = logging.getLogger('rec2vec:logger')
logger.setLevel("TRACE")
