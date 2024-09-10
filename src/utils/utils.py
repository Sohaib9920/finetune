import os
from huggingface_hub import login


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
