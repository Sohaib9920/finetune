from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from ..configs import DataConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import ModelConfig
from typing import Literal


COLUMNS_TO_KEEP = ["messages", "chosen", "rejected", "prompt", "completion", "label", "score"]
DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

def get_datasets(
    data_config: DataConfig
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.
    
    Args:
        data_config (`DataConfig`):
            Dataset configuration containing information about `dataset_mixer`, `dataset_split` and `shuffle`
    Returns:
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []

    dataset_mixer = data_config.dataset_mixer
    splits = data_config.dataset_splits
    revision = "main"
    subsample_seed = data_config.subsample_seed

    for ds, frac_or_dict in dataset_mixer.items():

        # get fracs of all the datasets
        if isinstance(frac_or_dict, dict):
            revision = frac_or_dict.get("revision", "main")  # default to main if no revision is specified
            frac = frac_or_dict.get("fraction", 1.0)  # default to 1.0 if no fraction is specified
        else:
            frac = frac_or_dict
        fracs.append(frac)

        # get train and test splits of all the datasets
        for split in splits:
            if "train" in split:
                if "data/" in ds:
                    train_ds = load_from_disk(ds)[split]
                else:
                    train_ds = load_dataset(
                        ds,
                        split=split,
                        revision=revision,
                    )
                train_ds = train_ds.remove_columns(
                    [col for col in train_ds.column_names if col not in COLUMNS_TO_KEEP]
                )
                raw_train_datasets.append(train_ds)

            elif "test" in split:
                if "data/" in ds:
                    val_ds = load_from_disk(ds)[split]
                else:
                    val_ds = load_dataset(
                        ds,
                        split=split,
                        revision=revision,
                    )
                val_ds = val_ds.remove_columns(
                    [col for col in val_ds.column_names if col not in COLUMNS_TO_KEEP]
                )
                raw_val_datasets.append(val_ds)
            
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")
        
    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")
    
    # Apply subsampling on datasets, concatenate them and then shuffle. 
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_train_datasets) > 0:
        train_subsets = []
        for frac, train_ds in zip(fracs, raw_train_datasets):
            train_subset = train_ds.shuffle(seed=subsample_seed).select(range(int(frac * len(train_ds))))
            train_subsets.append(train_subset)
        
        raw_datasets["train"] = concatenate_datasets(train_subsets)

    if len(raw_val_datasets) > 0:
        raw_datasets["test"] = concatenate_datasets(raw_val_datasets) 
    
    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )
    
    return raw_datasets


def get_tokenizer(model_args: ModelConfig, data_args: DataConfig, set_pad_token: bool = True) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Hack for Qwen-14b which doesn't have an EOS token defined in the tokenizer
    if "qwen-14b" in model_args.model_name_or_path.lower():
        tokenizer.eos_token_id = 151643  # <|endoftext|>

    if set_pad_token is True and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None: 
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def apply_chat_template(
    example,
    tokenizer: PreTrainedTokenizer,
    task: Literal["sft", "generation"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation']}"
        )
    return example