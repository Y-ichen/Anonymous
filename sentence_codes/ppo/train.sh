MODEL_NAME="mistral-ckpt/checkpoint-4000"
TOKENIZER_NAME="mistral-ckpt/checkpoint-4000"
REWARD_MODEL_NAME="rm_ckpt/3v1_param/checkpoint-700"
LOG_WITH="wandb"
LEARNING_RATE=5e-6
LR_SCHEDULER_TYPE="linear"
OUTPUT_MAX_LENGTH=400
MINI_BATCH_SIZE=1
BATCH_SIZE=32
PPO_EPOCHS=2
GRADIENT_ACCUMULATION_STEPS=1
ADAFACTOR=false
EARLY_STOPPING=false
TARGET_KL=0.1
REWARD_BASELINE=0.5
BATCHED_GEN=true
SAVE_FREQ=5
OUTPUT_DIR="ckpt/3v1_700"
SEED=42
TRAIN_EPOCHS=4
STEPS=1200
INIT_KL_COEF=0.2
ADAP_KL_CTRL=true
LOCAL_RANK=0
DATA_FILE_PATH="datasets/ppo/train_20k.json"
TRACKER_KWARGS="{\"wandb\": {\"entity\": \"your_entity\", \"name\": \"3v1_700\"}}"
PROJECT_NAME="sentence_ppo"

accelerate launch --main_process_port 27777 --multi_gpu --num_machines 1  --num_processes 8 train_sentence_ppo.py \
    --model_name "$MODEL_NAME" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --reward_model_name "$REWARD_MODEL_NAME" \
    --log_with "$LOG_WITH" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --output_max_length "$OUTPUT_MAX_LENGTH" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --ppo_epochs "$PPO_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --adafactor "$ADAFACTOR" \
    --early_stopping "$EARLY_STOPPING" \
    --target_kl "$TARGET_KL" \
    --reward_baseline "$REWARD_BASELINE" \
    --batched_gen "$BATCHED_GEN" \
    --save_freq "$SAVE_FREQ" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --train_epochs "$TRAIN_EPOCHS" \
    --steps "$STEPS" \
    --init_kl_coef "$INIT_KL_COEF" \
    --adap_kl_ctrl "$ADAP_KL_CTRL" \
    --local_rank "$LOCAL_RANK" \
    --data_file_path "$DATA_FILE_PATH" \
    --tracker_kwargs "$TRACKER_KWARGS" \
    --project_name "$PROJECT_NAME" 