from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM
from trl import ModelConfig, SFTConfig, SFTTrainer, get_quantization_config, get_peft_config, get_kbit_device_map
from src.configs import DataConfig
from src.utils import (
    get_datasets,
    get_tokenizer,
    apply_chat_template
)
import sys
from peft import get_peft_model
import torch

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

def main():

    parser = HfArgumentParser((DataConfig, ModelConfig, SFTConfig))
    data_config, model_config, sft_config = parser.parse_yaml_file(sys.argv[1])

    set_seed(sft_config.seed) 

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_config, data_config, set_pad_token=sft_config.packing) # safe to set pad=eos when packing

    #######################
    # Load pretrained model
    #######################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    peft_config = get_peft_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    if peft_config:

        gradient_checkpointing_kwargs = getattr(sft_config, "gradient_checkpointing_kwargs", None) or {}
        if getattr(sft_config, "gradient_checkpointing", False) and (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        ):
            model.enable_input_require_grads()

        model = get_peft_model(model, peft_config)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset if sft_config.do_train else None,
        eval_dataset=eval_dataset if sft_config.do_eval else None,
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":
    main()