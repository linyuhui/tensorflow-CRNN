python train.py \
    --batch_size=8 \
    --train_log_dir=train_logs/ \
    --dataset_name=text \
    --data_root=your_data_root \
    --save_interval_secs=180 \
    --split_name=train \
    --model_name=crnn \
    --reset_train_dir=True