from datasets import DatasetDict, concatenate_datasets, load_dataset, Dataset
from ..configs import DataConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
from trl import ModelConfig
from typing import Literal, Dict
import logging


logger = logging.getLogger(__name__) # controlled by parent

DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


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


def create_messages(example, query_col=None, response_col=None):
    """
    Create messages from query in `query_col` field, reponse in `response_col` field.
    Output them in `messages` field.
    """
    messages = []
    if query_col is not None:
        messages.append({"role": "user", "content": example[query_col]})
    if response_col is not None:
        messages.append({"role": "assistant", "content": example[response_col]})
    example["messages"] = messages
    return example


def remove_last_assistant(example, messages_col):
    """
    If last message is assistant (during generation) then remove it and add its content to `references` field.
    """
    messages = example[messages_col]
    if messages[-1]["role"] == "assistant":
        assistant_message = messages.pop()
        return {**example, "references": assistant_message["content"]}
    return example


def apply_chat_template(
    example, 
    tokenizer: PreTrainedTokenizer,
    messages_col: str,
    task: Literal["sft", "generation"] = "sft",
    system_msg: str = None
):
    """
    Use `tokenizer.chat_template` to convert messages given in `messages_col` of dataset into text. 
    If `system_msg` is given then add system message in messages or override its content if already present.
    If task is generation then add generation_prompt at end. Output text in `text` field of dataset
    """
    messages = example[messages_col]

    if system_msg is not None:
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_msg
        else:
            messages.insert(0, {"role": "system", "content": system_msg})

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
    )
    return example


def get_datasets(
    dataset_mixer: Dict, 
    tokenizer: PreTrainedTokenizer,
    task: Literal["sft", "generation"]
) -> Dict[str, DatasetDict]:
    """
    Generates datasets based on the provided data configuration and tokenizer.

    Args:
        dataset_mixer (Dict): It should have the following structure in YAML:

                AI-MO/NuminaMath-CoT:
                    split: 
                        train: train[:85]
                        test: test[:50%]
                    messages: messages
                    system_msg: Solve MATH using COT
                meta-math/MetaMathQA: 
                    split:
                        train: train[:1%]+train[-1%:]
                    query: query
                    response: response
                ise-uiuc/Magicoder-Evol-Instruct-110K:
                    split:
                        train: train[:2%]
                        test: train[-2%:]
                    query: instruction
                    response: response
                    system_msg: Solve queries related to python

            The `split` must be a dictionary with keys such as 'train' and 'test', and values that describe the data split 
            (see https://huggingface.co/docs/datasets/v2.21.0/loading#slice-splits).

            Provide either `messages` or both `query` and `response`. The `messages` field is useful when examples contain a 
            complete chat sequence (e.g., query + response + query + ...). In such cases, the `messages` column is directly 
            used. If only `query` and `response` are provided, it creates simple query-response examples.

            If the task is generation, providing a `response` or the final assistant message in `messages` creates a new 
            `references` column representing the expected generation output. This column is not created for other tasks.

            The existing `messages` field in the dataset is modified by either overriding/adding a system prompt (if 
            `system_msg` is specified in `dataset_mixer`) or by removing the last assistant message for generation tasks. 
            The tokenizer's `chat_template` is applied to the resulting messages.

        tokenizer (PreTrainedTokenizer): The tokenizer to use, which must have a defined `chat_template`. The `chat_template`
            controls how messages are converted to text, so ensure `tokenizer.chat_template` is set up before use.
        
        task (sft | generation): task for which dataset is task.

    Returns:
        Dict[str, DatasetDict]: A dictionary with dataset names as keys and `DatasetDict` objects containing train/test 
        `Dataset` splits as values. 

        Each split includes a new field `text`, which is generated by applying the tokenizer's `chat_template` to:
        - For SFT (Supervised Fine-Tuning) tasks: concatenated sequences of `query` and `response` (or `query + response + ... + response` 
          if derived from `messages`).
        - For generation tasks: concatenated sequences of `query` (or `query + response + ... + query` if derived from `messages`).

        Additionally, a `references` field is included in the splits only for generation tasks. This field contains the expected 
        output based on the `response` or the final assistant message in the `messages`.

    """
    
    raw_datasets = {}
    for name, info in dataset_mixer.items():
        
        # Load DatasetDict according to specified split
        split = info.get("split")
        if split is None:
            raise ValueError(f"`split` of dataset {name} is missing!")
        if not isinstance(split, dict):
            raise ValueError(f"`split` of dataset {name} must be a dict with keys in ['train', 'test']")
        
        logger.info(f"Loading Dataset: {name} with split {split}")
        ds = load_dataset(name, split=split)

        query_col = info.get("query")
        response_col = info.get("response")
        messages_col = info.get("messages")
        system_msg = info.get("system_msg")
        
        task = task
        if task == "sft":
            # If messages are not present then create them from query and response and store them in "messages" field
            if messages_col is None:
                if query_col is None or response_col is None:
                    raise ValueError(f"Must provide `messages` or both `query` and `response` for `sft` task")
                logger.info(f"Creating messages from `{query_col}` and `{response_col}` fields of dataset: {name}")
                ds = ds.map(create_messages, fn_kwargs={"query_col": query_col, "response_col": response_col})
                messages_col = "messages"

            # Apply chat template on messages
            logger.info(f"Applying chat template on messages given in `{messages_col}` field of dataset: {name} into `text` field.")
            ds = ds.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "messages_col": messages_col, "task": "sft", "system_msg": system_msg})
        
        elif task == "generation":
            # If messages are not present then create them from query
            if messages_col is None:
                if query_col is None:
                    raise ValueError(f"Must provide `messages` or `query` for `generation` task")
                logger.info(f"Creating messages from `{query_col}` field of dataset: {name}")
                ds = ds.map(create_messages, fn_kwargs={"query_col": query_col, "response_col": None})
                messages_col = "messages"

                # If response_col is given then It will be used as `references` 
                if response_col is not None:
                    logger.info(f"`{query_col}` of dataset {name} is renamed to `references`")
                    ds = ds.rename_columns({response_col: "references"})

            # elif they are present then remove the last assistant message if it exists. Provide last assistant_msg as `references`
            else:
                logger.info(f"Removing last assistant message from messages (if any) to `references` field of dataset: {name}")
                ds = ds.map(remove_last_assistant, fn_kwargs={"messages_col": messages_col})

            # Apply chat template on messages
            logger.info(f"Applying chat template on messages given in `{messages_col}` field of dataset: {name} into `text` field.")
            ds = ds.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "messages_col": messages_col, "task": "generation", "system_msg": system_msg})

        else:
            raise ValueError(f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation']}")
        
        raw_datasets[name] = ds
        
    return raw_datasets


def combine_datasets(
    raw_datasets: Dict[str, DatasetDict],
    task: Literal["sft", "generation"]
) -> DatasetDict:
    """
    Remove unnecassy columns from train and test datasets and combine them. 
    For sft: {train: concat(all train splits), test: concat(all test splits)}
    For generation: {name1: train + test split of name1, name2: train + test split of name2, ...}
    """
    train_datasets = []
    test_datasets = []
    allowed_cols = ["text", "references"] if task == "generation" else ["text"]
    datasets_dict = DatasetDict()

    for name, ds_dict in raw_datasets.items():
        train_test_ds = []
        if "train" in ds_dict:
            train_ds = ds_dict["train"]
            train_test_ds.append(train_ds)
            # only keep common columns between different datasets for proper concat later on
            train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in allowed_cols])
            train_datasets.append(train_ds)
        if "test" in ds_dict:
            test_ds = ds_dict["test"]
            train_test_ds.append(test_ds)
            test_ds = test_ds.remove_columns([col for col in test_ds.column_names if col not in allowed_cols])
            test_datasets.append(test_ds)
        
        if task == "generation":
            datasets_dict[name] = concatenate_datasets(train_test_ds)
        
    if task == "sft":
        datasets_dict["train"] = concatenate_datasets(train_datasets) if len(train_datasets) > 0 else None
        datasets_dict["test"] = concatenate_datasets(test_datasets) if len(test_datasets) > 0 else None
    
    return datasets_dict


def prepare_datasets(dataset_mixer: Dict, tokenizer: PreTrainedTokenizer, task: Literal["sft", "generation"]):
    raw_datasets = get_datasets(dataset_mixer, tokenizer, task=task)
    logger.info("Combining Datasets")
    raw_datasets = combine_datasets(raw_datasets, task=task)
    return raw_datasets