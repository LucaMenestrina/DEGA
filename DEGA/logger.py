import logging

# Thanks to https://stackoverflow.com/questions/14844970/modifying-logging-message-format-based-on-message-logging-level-in-python3


class Formatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            self._style._fmt = "DEBUG:%(module)s: line %(lineno)d: %(msg)s"
        elif record.levelno == logging.INFO:
            self._style._fmt = "%(msg)s"
        elif record.levelno == logging.WARNING:
            self._style._fmt = "WARNING:%(module)s: %(msg)s"
        elif record.levelno == logging.ERROR:
            self._style._fmt = "ERROR:%(module)s: %(msg)s"
        elif record.levelno == logging.CRITICAL:
            self._style._fmt = "CRITICAL:%(module)s: line %(lineno)d: %(msg)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


log = logging.getLogger()

handler = logging.StreamHandler()
handler.setFormatter(Formatter())
log.addHandler(handler)
log.setLevel(logging.INFO)
