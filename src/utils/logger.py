"""
Logging utility.
Provides consistent logging to both console and file across the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


_loggers = {}


def get_logger(name, log_dir=None, level="INFO", console=True, file=True):
    """
    Get or create a named logger with console and file handlers.
    
    Args:
        name: Logger name (usually module name).
        log_dir: Directory for log files. If None, file logging is disabled.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Enable console output.
        file: Enable file output (requires log_dir).
        
    Returns:
        logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers = []  # Clear any existing handlers
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger
