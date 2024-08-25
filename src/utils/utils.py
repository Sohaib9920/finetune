import os
from huggingface_hub import login
import logging
import sys


def hf_login():
    """Login to HuggingFace Hub if HF_TOKEN is defined in the environment"""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is not None:
        login(token=hf_token)


def setup_logging(sft_config):
    """
    Setup the baseConfig of root logger. Use `force=True` to make sure we use it and not any other baseConfig defined earlier.
    The transformers and datasets loggers have default StreamHandler to stderr. Just remove these handles and use 
    `propagate=True` so that they use root logger handler. Their child loggers like `transformers.trainer` will
    also use the same handles.
    The log_level for main process and replica can be controlled using `log_level` and `log_level_replica` args of sft_conig resp.
    """

    log_level = sft_config.get_process_log_level() 

    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s", # explicit format
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )

    main_module_logger = logging.getLogger("__main__")
    main_module_logger.setLevel(log_level)

    transformers_handler = logging.getLogger("transformers")
    transformers_handler.handlers = []
    transformers_handler.setLevel(log_level)
    transformers_handler.propagate = True

    datasets_handler = logging.getLogger("datasets")
    datasets_handler.handlers = []
    datasets_handler.setLevel(log_level)
    datasets_handler.propagate = True