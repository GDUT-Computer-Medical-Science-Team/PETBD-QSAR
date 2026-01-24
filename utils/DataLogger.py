import logging
import colorlog

"""
"""
log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

default_formats = {
    'color_format': '%(log_color)s[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d\t--\t%(message)s',
    'log_format': '[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d\t--\t%(message)s'
}


class DataLogger(object):
    def __init__(self, log_file=None, logger_level=logging.INFO, file_level=logging.DEBUG, console_level=logging.INFO):
        """
        """
        self.log_file = log_file
        self.logger_level = logger_level
        self.console_level = console_level
        self.file_level = file_level
        self.formatter = logging.Formatter(default_formats.get('log_format'))

    def getlog(self, logger_name=None, log_file=None, disable_console_output=False):
        """
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.logger_level)
        if not disable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(colorlog.ColoredFormatter(fmt=default_formats.get('color_format'),
                                                                   datefmt='%Y-%m-%d %H:%M:%S',
                                                                   log_colors=log_colors_config))
            logger.addHandler(console_handler)
            console_handler.close()

        if log_file is not None:
            self.log_file = log_file
        if self.log_file is not None:
            file_handler = logging.FileHandler(self.log_file, 'a', encoding='utf-8')
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)
            file_handler.close()

        return logger
