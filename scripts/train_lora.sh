#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH

LOAD="/xxx/mplug-owl2-llama2-7b"

deepspeed --include localhost:$1 --master_port 6688 src/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path $LOAD \
    --version v1 \
    --dataset_type single \
    --weight_next_token 0.05 \
    --continuous_rating_loss True \
    --closeset_rating_loss True \
    --use_fix_std True \
    --detach_pred_std True \
    --data_paths /xxx/koniq/metas/train_koniq_7k2.json \
    --data_weights 1 \
    --image_folder /xxx/ \
    --output_dir ./checkpoints/Qscorer_lora_t1 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
