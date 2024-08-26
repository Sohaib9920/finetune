from dataclasses import dataclass, field
from typing import Optional, Dict
import trl


@dataclass
class SFTConfig(trl.SFTConfig):
    sdpa_kernel: Optional[str] = field(default=None, metadata={"help": "kernel to enable for sdpa (math, mem, flash). None for auto"})
    wandb_config: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": (
                """
                    Dict of wandb environment variables that are automatically passed
                    when `wandb.init()` is called.
                    Check https://docs.wandb.ai/guides/track/environment-variables for
                    possible keys. 
                    e.g. As yaml:
                    wandb_config:
                        WANDB_PROJECT: "transformers"
                        WANDB_RUN_GROUP: "group_00"
                        WANDB_TAGS: "sft,full_train"
                """
            )
        }
    )