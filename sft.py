import sys
import torch
import logging
import math
from tabulate import tabulate
import wandb
import torch
from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoConfig
from trl import ModelConfig, SFTTrainer, get_quantization_config, get_peft_config, get_kbit_device_map
from src.configs import DataConfig, SFTConfig
from src.utils import (
    prepare_datasets,
    get_tokenizer,
    hf_login,
    init_wandb_training
)

logger = logging.getLogger(__name__) # globaly available logger of this module

def main():

    parser = HfArgumentParser((DataConfig, ModelConfig, SFTConfig))
    data_config, model_config, sft_config = parser.parse_yaml_file(sys.argv[1])

    is_world_process_zero = sft_config.process_index == 0 # same as trainer do; get process_index of sft_config.distributed_state (PartialState)

    sdpa_kernel = sft_config.sdpa_kernel
    if sdpa_kernel is not None:
        torch.backends.cuda.enable_mem_efficient_sdp(sdpa_kernel == "mem")
        torch.backends.cuda.enable_flash_sdp(sdpa_kernel == "flash")
        torch.backends.cuda.enable_math_sdp(sdpa_kernel == "math")

    set_seed(sft_config.seed)

    ########
    # Setup
    ########
    log_level = sft_config.get_process_log_level() 
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s", # explicit format
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )
    logger.setLevel(log_level)
    logging.getLogger("src").setLevel(log_level)

    # Login to HuggingFace Hub if needed
    hf_login(required=(sft_config.push_to_hub is True))

    # Setup WandB
    log_wandb = ("wandb" in sft_config.report_to) and is_world_process_zero
    if log_wandb:
        init_wandb_training(sft_config.wandb_config) # must be before init
    
    if log_wandb:
        # Initializing wandb eariler than WandbCallback.setup() in order to log stdout to wandb.
        # Ignores trail_name in init. Now we set run_name using env variable and it is not overriden by args.run_name. Other than that its the same
        wandb.init()
        # wandb.config is updated by sft_config, model.config and model.peft_config in WandbCallback.setup() so no need to do manually
        # wandb logging working:
        # trainer uses `log()` method everytime which first add current epoch and step to log and then append the updated log to trainer_state log_history.
        # Then it gives these logs to WandbCallback.on_log(). On main process: If metric key is one of train_result then add it as summary metric. 
        # Otherwise, convert keys "eval_{metric}" to "eval/{metric}", "test_{metric}" to "test/{metric}", "train_{metric}" and "{metric}" to "train/{metric}"
        # Then add "train/global_step" to log which is default x-axis added in `setup()` using `wandb.define_metric("train/global_step")`
        # Then simply `wandb.log()`

    logger.warning(
        f"Process rank: {sft_config.local_rank}, device: {sft_config.device}, n_gpu: {sft_config.n_gpu}"
        + f" distributed training: {bool(sft_config.local_rank != -1)}, 16-bits training: {sft_config.fp16 or sft_config.bf16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Data parameters {data_config}")
    logger.info(f"Training/evaluation parameters {sft_config}")

    #################
    # Prepare dataset
    #################
    tokenizer = get_tokenizer(model_config, data_config, set_pad_token=sft_config.packing) # safe to set pad=eos when packing
    with sft_config.main_process_first(desc="Preparing dataset"):
        raw_datasets = prepare_datasets(data_config, tokenizer=tokenizer)

    train_dataset = raw_datasets.get("train")
    eval_dataset = raw_datasets.get("test")

    raw_train_examples = len(train_dataset) if train_dataset is not None else 0
    raw_eval_examples = len(eval_dataset) if eval_dataset is not None else 0
    
    logger.info(f"Raw Train Examples: {raw_train_examples}")
    logger.info(f"Raw Eval Examples: {raw_eval_examples}")

    if log_wandb:
        wandb.run.summary["raw_train_examples"] = raw_train_examples
        wandb.run.summary["raw_eval_examples"] = raw_eval_examples

    if raw_train_examples > 0:
        for ex in train_dataset.select(range(2)):
            logger.info(f"Training example:\n\n{ex['text']}\n")
    
    if raw_eval_examples > 0:
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
    # bnb_4bit_compute_dtype is same as torch_dtype for consistency with model weights
    # bnb_4bit_quant_storage is same as torch_dtype for fsdp/deepspeed support
    peft_config = get_peft_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    logger.info("Loading Model")
    if sft_config.testing:
        config = AutoConfig.from_pretrained(model_config.model_name_or_path)
        config.num_hidden_layers = 2
        config.hidden_size = 32
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch_dtype, attn_implementation=model_config.attn_implementation)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs) # download weights on main_procss, use cache for other
            
    model = torch.compile(model)

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

    # Trainer uses `prepare_model_for_kbit_training` to upcast all non-int8 params (layernorm, embedding and lm_head) to 
    # float32 (except in fsdp/deepspeed case) just to have layernorm in float32 for stable qlora training. 
    # For mixed precion, we can safely downcast lm_head and embedding to save a lot of memory with same loss (atleast for Llama). 
    # Trainer do this for bf16 int4 qlora. We also do for fp16 int4 and int8:

    if (
        (sft_config.bf16 or sft_config.fp16)
        and (getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))
    ):
        dtype = torch.bfloat16 if sft_config.bf16 else torch.float16
        for name, module in model.named_modules():
            if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
                module = module.to(dtype)

    ##########################################
    # Log processed data and model information
    ##########################################
    train_examples = len(trainer.train_dataset) if trainer.train_dataset is not None else 0
    eval_examples = len(trainer.eval_dataset) if trainer.eval_dataset is not None else 0

    logger.info(f"Train Examples: {train_examples}")
    logger.info(f"Eval Examples: {eval_examples}")

    if log_wandb:
        wandb.run.summary["train_examples"] = train_examples
        wandb.run.summary["eval_examples"] = eval_examples

    ###############
    # Training loop
    ###############
    if sft_config.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["raw_train_examples"] = raw_train_examples
        metrics["train_examples"] = train_examples
        trainer.log_metrics("train", metrics) # only print formatted metrics on main_process. Does not log to wandb
        trainer.save_metrics("train", metrics) # just save metrics to {split}.json and updated combined metrics to all_results.json on main_process
        # trainer has already logged the train_result.metrics to wandb as summary metrics 

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
        # trainer already logged the evaluate result like usual. Extras added manually as summary metrics

        if log_wandb:
            wandb.run.summary["perplexity"] = perplexity
    
    ##################################
    # Save model and create model card
    ##################################
    trainer.save_state() # saving state AFTER all the logging added to log_history of trainer_state
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True # Restore k,v cache for fast inference
    if trainer.is_world_process_zero():
        trainer.model.config.save_pretrained(sft_config.output_dir)

    kwargs = {
        "finetuned_from": model_config.model_name_or_path,
        "dataset": list(data_config.dataset_mixer.keys()),
        "dataset_tags": list(data_config.dataset_mixer.keys()),
    }

    if sft_config.push_to_hub is True:
        logger.info("Pushing to hub...")
        # do `save_model`, `create_model_card` and then push everything inside output_dir (excluding _* and checkpoint-*) to 
        # hub_model_id, all on main process
        trainer.push_to_hub(**kwargs)

    else:
        # Both run on main_process
        trainer.save_model() 
        trainer.create_model_card(**kwargs)
        logger.info(f"Model saved to {sft_config.output_dir}")

if __name__ == "__main__":
    main()