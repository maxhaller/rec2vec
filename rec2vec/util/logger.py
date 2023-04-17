import logging


def addLoggingLevel(level_name: str, level_num: int, method_name: str = None) -> None:
    """
    Utility function to allow for logging on trace level.

    :param level_name:  name of level
    :param level_num:   log level
    :param method_name: method name (if None, the level name is used in lower case)
    """
    if not method_name:
        method_name = level_name.lower()

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)
