CUDA_VISIBLE_DEVICES=7 python train_sentence_rm.py \
--model_name_or_path="allenai/longformer-base-4096" \
--output_dir="ckpt/3v1_param" \
--per_device_train_batch_size=32 \
--num_train_epochs=20 \
--gradient_accumulation_steps=8 \
--gradient_checkpointing=True \
--learning_rate=1.41e-5 \
--report_to="wandb" \
--remove_unused_columns=False \
--optim="adamw_torch" \
--logging_steps=5 \
--evaluation_strategy="steps" \
--max_length=2048 \
--torch_dtype=float32 \
--save_steps=20 \
--lora_task_type=SEQ_CLS \
--tokenizer_name="allenai/longformer-base-4096" \
--train_dataset_path="datasets/rm/train_data_3v1.json" \
--test_dataset_path="datasets/rm/test_200_3v1.json" \
--num_class_labels=2 \
--pos_neg_ratio=3.0 \
--class_weight=4.0 \
--lora_yc_r=32 \
--lora_yc_alpha=64 \
--lora_yc_dropout=0.1 \
--seed=42 \
--use_lora=False