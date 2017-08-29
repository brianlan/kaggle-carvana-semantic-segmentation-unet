import os
import logging
import datetime
from pytz import timezone as tz


def get_cur_ts():
    return datetime.datetime.now(tz('Asia/Shanghai'))


LOG_DIR = './log'
logger = logging.getLogger('tf-unet')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.sep.join([
    LOG_DIR,
    datetime.datetime.strftime(get_cur_ts(), '%Y%m%d')
]))
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)