# utils.py
import os
import sys
import json
import logging
from datetime import datetime

from config import RECEIVED_DIR


def setup_logger(name, log_file=None):
    """配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出（可选）
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_file(file_data, filename):
    """保存接收的文件到 received_files 目录"""
    filepath = os.path.join(RECEIVED_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(file_data)
    return filepath


def generate_job_id():
    """生成任务ID"""
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')