from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DataConfig:
    dataset_mixer: Dict[str, Union[float, Dict[str, float]]] = field(
        metadata={
            "help": (
                """
                    Datasets and either their proportions to be used for training,
                    or a dict of their proportions and the dataset revision to use.
                    e.g.
                    {
                        'HuggingFaceH4/testing_codealpaca_small': 0.5,
                        'HuggingFaceH4/testing_codealpaca_small': {
                            'fraction': 0.5,
                            'revision': '20-examples'
                        }
                    }

                    As yaml
                    dataset_mixer:
                        HuggingFaceH4/testing_codealpaca_small: 0.5
                        HuggingFaceH4/testing_codealpaca_small:
                            fraction: 0.5
                            revision: 20-examples
                """
            )
        }
    )

    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": "List of train test splits to use in the dataset"}
    )

    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})

    subsample_seed: Optional[int] = field(default=42, metadata={"help": "Random seed before subsampling train data"})