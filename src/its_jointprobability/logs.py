import logging
import logging.config

import fastapi_cli.utils.cli

class MyFormatter(fastapi_cli.utils.cli.CustomFormatter):
    """ Like the formatter from fastapi, but supporting other fields like `module`. """
    def formatMessage(self, record: logging.LogRecord) -> str:
        msg = logging.Formatter.formatMessage(self, record)
        return self.toolkit.print_as_string(msg, tag=record.levelname)

def setup_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": MyFormatter,
                "fmt": "%(module)s: %(message)s",
                # "fmt": "%(name)s %(module)s: %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn.error": {"level": "INFO", "propagate": False},
            "watchfiles.main": {"level": "INFO", "propagate": False},
            "": {"handlers": ["default"], "level": "DEBUG", "propagate": False}
        }
    }
    logging.config.dictConfig(logging_config)

