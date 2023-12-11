# Fine-Tune LLM on Code/Text Dataset

This script is designed for fine-tuning a language model on a code/text dataset. It utilizes the Hugging Face `transformers` library and supports various options for customization.

## Usage

```bash
python fine_tune_llm.py --model_path <model_path> --dataset_name <dataset_name> --load_dataset_from_disk <load_dataset_from_disk> --dataset_path <dataset_path> --subset <subset> --split <split> --size_valid_set <size_valid_set> --test_size <test_size> --streaming --shuffle_buffer <shuffle_buffer> --data_column <data_column> --seq_length <seq_length> --max_steps <max_steps> --batch_size <batch_size> --gradient_accumulation_steps <gradient_accumulation_steps> --eos_token_id <eos_token_id> --learning_rate <learning_rate> --lr_scheduler_type <lr_scheduler_type> --num_warmup_steps <num_warmup_steps> --weight_decay <weight_decay> --run_name <run_name> --local_rank <local_rank> --no_fp16 --bf16 --no_gradient_checkpointing --seed <seed> --num_workers <num_workers> --output_dir <output_dir> --log_freq <log_freq> --eval_freq <eval_freq> --save_freq <save_freq> --fim_rate <fim_rate> --fim_spm_rate <fim_spm_rate> --use_peft_lora --lora_r <lora_r> --lora_alpha <lora_alpha> --lora_dropout <lora_dropout> --lora_target_modules <lora_target_modules> --use_flash_attn --use_4bit_qunatization --use_nested_quant --bnb_4bit_quant_type <bnb_4bit_quant_type> --bnb_4bit_compute_dtype <bnb_4bit_compute_dtype> --use_8bit_qunatization --push_to_hub
