import logging
import colorlog

"""
终端输出日志颜色配置
"""
log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

default_formats = {
    # 终端输出格式
    'color_format': '%(log_color)s[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d\t--\t%(message)s',
    # 日志输出格式
    'log_format': '[%(levelname)s][%(asctime)s] %(filename)s:%(lineno)d\t--\t%(message)s'
}


class DataLogger(object):
    def __init__(self, log_file=None, logger_level=logging.INFO, file_level=logging.DEBUG, console_level=logging.INFO):
        """
        日志工具类
        :param log_file: 保存日志的文件路径，不指定则不输出文件日志
        :param logger_level: 日志等级
        :param file_level: 输出文件日志等级
        :param console_level: 控制台日志等级
        """
        self.log_file = log_file
        self.logger_level = logger_level
        self.console_level = console_level
        self.file_level = file_level
        self.formatter = logging.Formatter(default_formats.get('log_format'))

    def getlog(self, logger_name=None, log_file=None, disable_console_output=False):
        """
        获取log对象
        :param logger_name: 指定logger的名字，不指定则默认为root
        :param log_file: 保存日志的文件路径，默认使用类初始化所指定的文件路径，在本函数中重新指定则重定向文件路径
        :return: log对象
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.logger_level)
        # 设置控制台日志输出格式
        if not disable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(colorlog.ColoredFormatter(fmt=default_formats.get('color_format'),
                                                                   datefmt='%Y-%m-%d %H:%M:%S',
                                                                   log_colors=log_colors_config))
            logger.addHandler(console_handler)
            console_handler.close()

        # 重定向日志文件路径
        if log_file is not None:
            self.log_file = log_file
        # 设置文件日志输出格式
        if self.log_file is not None:
            file_handler = logging.FileHandler(self.log_file, 'a', encoding='utf-8')
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)
            file_handler.close()

        return logger
