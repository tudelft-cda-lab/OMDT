import logging
import sys


class RicherDomainLogger:
    def __init__(self, loggername, debug):
        """

        :param loggername: Name of logger (String)
        :param debug: {True, Default:False} If set to True, it creates an additional logger file where every single logger message gets
        stored.

        """

        # root/ Main logger
        self.root_logger = logging.getLogger(loggername)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        self.root_logger.setLevel(logging.DEBUG)

        # Console output handling. Prints every logger message of critical level only
        console_output_handler = logging.StreamHandler(sys.stdout)
        console_output_handler.setLevel(logging.CRITICAL)
        console_output_handler.setFormatter(formatter)
        self.root_logger.addHandler(console_output_handler)

        # logger file handler
        if debug:
            file_handler = logging.FileHandler('RicherDomainLogger.log')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            self.root_logger.addHandler(file_handler)

        self.root_logger.propagate = False
