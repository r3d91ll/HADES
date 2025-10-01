"""
Structured Logging
==================

Centralized logging configuration using structlog for structured JSON logging.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog

# Global flag to track if logging has been setup
_logging_initialized = False


class LogManager:
    """
    Centralized logging configuration using structlog.

    Features:
    - Structured JSON logging
    - Automatic log rotation
    - Context preservation (run_id, processor)
    - Multiple output targets
    - No database logging
    """

    @staticmethod
    def setup(config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Setup logging configuration.

        Args:
            config_path: Path to logging configuration JSON (optional)
            log_level: Default log level
        """
        global _logging_initialized

        if _logging_initialized:
            return

        # Setup log directory
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # Configure Python logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(message)s'
        )

        # Setup handlers
        handlers = []

        # Main log file with rotation (10MB, keep 5 backups)
        main_handler = RotatingFileHandler(
            log_dir / "processors.log",
            maxBytes=10_485_760,
            backupCount=5
        )
        main_handler.setLevel(getattr(logging, log_level))
        handlers.append(main_handler)

        # Error log file (10MB, keep 3 backups)
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10_485_760,
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

        # Add handlers to root logger
        root_logger = logging.getLogger()
        for handler in handlers:
            root_logger.addHandler(handler)

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        _logging_initialized = True

        # Log initialization
        logger = structlog.get_logger()
        logger.info("logging_initialized", log_dir=str(log_dir), level=log_level)

    @staticmethod
    def get_logger(processor_name: str, run_id: str):
        """
        Get a logger instance with processor context.

        Args:
            processor_name: Name of the processor
            run_id: Run ID for this processing session

        Returns:
            Configured logger instance
        """
        # Ensure logging is setup
        if not _logging_initialized:
            LogManager.setup()

        return structlog.get_logger().bind(
            processor=processor_name,
            run_id=run_id
        )
