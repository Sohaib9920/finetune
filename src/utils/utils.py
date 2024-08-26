import os
from huggingface_hub import login
import logging
import sys


def hf_login(required=False):
    """Login to HuggingFace Hub if HF_TOKEN is defined in the environment"""
    hf_token = os.getenv("HF_TOKEN")
    if required and hf_token is None:
        raise ValueError("HF_TOKEN is required but not found in the environment variables.")
    elif hf_token is not None:
        login(token=hf_token)


def init_wandb_training(wandb_config):
    """
    Setup wandb environment variables so that `wandb.init()` uses them inside `WandbCallback` setup. 
    """
    if wandb_config is not None:
        for k,v in wandb_config.items():
            os.environ[k] = v
            

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


def init_wandb_training(sft_config):
    """
    Helper function for setting up Weights & Biases logging tools.
    Setup wandb environment variables that will automatically be passed to `wandb.init()`
    """
    os.environ["WANDB_PROJECT"] = sft_config.wandb_project
    os.environ["WANDB_RUN_GROUP"] = sft_config.wandb_run_group
    if sft_config.wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = sft_config.wandb_run_id
    if sft_config.wandb_tags is not None:
        os.environ["WANDB_TAGS"] = ",".join(sft_config.wandb_tags)