#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-17 16:33:13
# @Author  : skyhiter

import logging
import logging.config
import time
from pathlib import Path


def log_path():
    """获取log文件路径，形如'./logs/20190417_16h57m08s.log'
    """
    curr_time = time.strftime('%Y%m%d_%Hh%Mm%Ss', time.localtime(time.time()))
    log_file = Path('./data/logs/')  # log file root path
    log_file.mkdir(exist_ok=True, parents=True)  # if not exists, mkdir
    log_file = log_file / '{}.log'.format(curr_time)

    return log_file


# 配置参考Django
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console_formatter': {
            'format': '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        },
        'file_formatter': {
            'format': '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        }
    },
    'handlers': {
        'console_handler': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'console_formatter',
            'stream': 'ext://sys.stdout',  # 默认是sys.stderr, 重定向的时候会重定向不到stdout
        },
        'file_handler': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'file_formatter',
            'filename': log_path(),
            'mode': 'w',  # 也可以是a
        }
    },
    'loggers': {
        'main': {
            'handlers': ['console_handler', 'file_handler'],  # 该logger同时写控制台和文件
            'level': 'DEBUG',
            'propagate': True,  # 传播选项
        },
    }
}

logger = logging.getLogger('main')  # name 是 logger的唯一标识符。即name一样，说明是同一个logger，这在跨文件使用同一个logger很有用
logging.config.dictConfig(DEFAULT_LOGGING)  # 读入json形式的配置文件

if __name__ == '__main__':
    logger.info('this is a main logger')
