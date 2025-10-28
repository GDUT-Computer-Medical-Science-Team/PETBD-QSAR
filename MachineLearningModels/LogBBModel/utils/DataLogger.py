import logging
import colorlog

"""
[Chinese text removed]
"""
log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

default_formats = {
    # [Chinese text removed]
    'color_format': '%(log_color)s[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d\t--\t%(message)s',
    # [Chinese text removed]
    'log_format': '[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d\t--\t%(message)s'
}


class DataLogger(object):
    def __init__(self, log_file=None, logger_level=logging.INFO, file_level=logging.DEBUG, console_level=logging.INFO):
        """
        [Chinese text removed]
        :param log_file: [Chinese text removed]，[Chinese text removed]
        :param logger_level: [Chinese text removed]
        :param file_level: [Chinese text removed]
        :param console_level: [Chinese text removed]
        """
        self.log_file = log_file
        self.logger_level = logger_level
        self.console_level = console_level
        self.file_level = file_level
        self.formatter = logging.Formatter(default_formats.get('log_format'))

    def getlog(self, logger_name=None, log_file=None, disable_console_output=False):
        """
        [Chinese text removed]log[Chinese text removed]
        :param logger_name: [Chinese text removed]logger[Chinese text removed]，[Chinese text removed]root
        :param log_file: [Chinese text removed]，[Chinese text removed]，[Chinese text removed]
        :return: log[Chinese text removed]
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.logger_level)
        # [Chinese text removed]
        if not disable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(colorlog.ColoredFormatter(fmt=default_formats.get('color_format'),
                                                                   datefmt='%Y-%m-%d %H:%M:%S',
                                                                   log_colors=log_colors_config))
            logger.addHandler(console_handler)
            console_handler.close()

        # [Chinese text removed]
        if log_file is not None:
            self.log_file = log_file
        # [Chinese text removed]
        if self.log_file is not None:
            file_handler = logging.FileHandler(self.log_file, 'a', encoding='utf-8')
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)
            file_handler.close()

        return logger
