import logging

def setup_logger(log_filename):
    logger = logging.getLogger("aad")
    logger.setLevel(logging.DEBUG)
    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger