# https://chatgpt.com/c/67f450c5-1424-8013-87a0-8f2f7d996db1
# for the future: re-analyze the files and understand how we could better use a "RAG pipeline with LangChain or LlamaIndex"


import logging
import os


def setup_logging(log_file='app.log'):
    """
    Set up logging configuration for the RAG project.

    This function configures logging to output messages to both the console and a file.
    The file handler is set to capture DEBUG and above levels while the console displays INFO and above.

    Parameters:
        log_file (str): The file path where log messages will be written.

    Returns:
        logging.Logger: Configured logger instance for the project.
    """
    logger = logging.getLogger('RAG_Project')
    logger.setLevel(logging.DEBUG)

    # Create handlers: one for the console and one for a file.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Set a logging format that includes timestamp, logger name, severity level, and message.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # To prevent duplicate handlers if the logger already has them, check before adding.
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.debug("Logger setup complete. Logging to console and file established.")
    return logger


if __name__ == '__main__':
    # Initialize logging to test and verify that our environment is correctly set up.
    logger = setup_logging()
    logger.info("RAG project initialized. Environment setup is complete.")
    print("Environment setup is complete. Check 'app.log' for detailed logging information.")


