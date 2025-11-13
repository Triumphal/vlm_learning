# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 train.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path models/llava_clip-L-14-336_Qwen1.5-1.8B \
    --train_type use_lora \
    --data_path datasets/LLaVA-CC3M-Pretrain-595K \
    --remove_unused_columns false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --bf16 true \
    --fp16 false \
    --dataloader_persistent_workers True \
    --output_dir outputs/output_model_user_lora_251114 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10

# --model_max_length 2048
    # --evaluation_strategy "no" \
# --save_strategy "steps" \
# --save_steps 10 \
# --save_steps 1000 \
# --save_strategy "epoch" \