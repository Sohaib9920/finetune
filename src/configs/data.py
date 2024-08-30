from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class DataConfig:
    dataset_mixer: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = field(
        metadata={"help": "Check `get_datasets` doctring for more info"}
    )

    task: Optional[str] = field(default="sft", metadata={"help": "Task to prepare data for"})

    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})