from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM
from trl import ModelConfig, SFTTrainer, get_quantization_config, get_peft_config, get_kbit_device_map
from src.configs import DataConfig, SFTConfig
from src.utils import (
    get_datasets,
    get_tokenizer,
    apply_chat_template,
    hf_login
)
import sys
import torch
import logging
import math

logger = logging.getLogger(__name__) # globaly available logger of this module

def main():

    parser = HfArgumentParser((DataConfig, ModelConfig, SFTConfig))
    data_config, model_config, sft_config = parser.parse_yaml_file(sys.argv[1])

    sdpa_kernel = sft_config.sdpa_kernel
    if sdpa_kernel is not None:
        torch.backends.cuda.enable_mem_efficient_sdp(sdpa_kernel == "mem")
        torch.backends.cuda.enable_flash_sdp(sdpa_kernel == "flash")
        torch.backends.cuda.enable_math_sdp(sdpa_kernel == "math")

    set_seed(sft_config.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = sft_config.get_process_log_level() # can set different log level for main process and replicas
    logger.setLevel(log_level)

    logger.warning(
        f"Process rank: {sft_config.local_rank}, device: {sft_config.device}, n_gpu: {sft_config.n_gpu}"
        + f" distributed training: {bool(sft_config.local_rank != -1)}, 16-bits training: {sft_config.fp16 or sft_config.bf16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Data parameters {data_config}")
    logger.info(f"Training/evaluation parameters {sft_config}")

    # Login to HuggingFace Hub if needed
    hf_login()

    #################
    # Prepare dataset
    #################
    raw_datasets = get_datasets(data_config)
    tokenizer = get_tokenizer(model_config, data_config, set_pad_token=sft_config.packing) # safe to set pad=eos when packing
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})

    train_dataset = raw_datasets.get("train")
    eval_dataset = raw_datasets.get("test")

    logger.info(f"Train dataset size: {len(train_dataset) if train_dataset is not None else 0}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset is not None else 0}")

    if train_dataset is not None:
        for ex in train_dataset.select(range(2)):
            logger.info(f"Training example:\n\n{ex['text']}\n")
    
    if eval_dataset is not None:
        for ex in eval_dataset.select(range(2)):
            logger.info(f"Evaluation example:\n\n{ex['text']}\n")

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

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs) # runs on main_procss, use cache for other

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset if sft_config.do_train else None,
        eval_dataset=eval_dataset if sft_config.do_eval else None,
        tokenizer=tokenizer,
        peft_config=peft_config
    )

    ###############
    # Training loop
    ###############
    if sft_config.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    ##########
    # Evaluate
    ##########
    if sft_config.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    ##################################
    # Save model and create model card
    ##################################
    trainer.model.config.use_cache = True # Restore k,v cache for fast inference
    if trainer.is_world_process_zero():
        trainer.model.config.save_pretrained(sft_config.output_dir)

    if sft_config.push_to_hub is True:
        kwargs = {
            "finetuned_from": model_config.model_name_or_path,
            "dataset": list(data_config.dataset_mixer.keys()),
            "dataset_tags": list(data_config.dataset_mixer.keys()),
        }
        logger.info("Pushing to hub...")
        # do save_model, create_model_card and then push everything inside output_dir (excluding _* and checkpoint-*) to 
        # hub_model_id, all on main process
        trainer.push_to_hub(**kwargs) 


if __name__ == "__main__":
    main()