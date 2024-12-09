# Item and User LLM are initialized by specific pretrain_dir.
exp=1

cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--loss nce \
--epochs 1 \
--dataset Pixel50K \
--train_batch_size 4 \
--MAX_TEXT_LENGTH 128 \
--MAX_ITEM_LIST_LENGTH 5 \
--checkpoint_dir experiments/test/$exp \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir ../models/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
--user_pretrain_dir ../models/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
--text_path /root/autodl-tmp/HLLM/information \
--text_keys '[\"title\",\"tag\",\"description\"]' \
--val_only True 