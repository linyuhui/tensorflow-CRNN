python eval.py \
    --train_log_dir=your_ckpt_dir \
    --eval_log_dir=your_eval_dir \
    --split_name=validation \
    --dataset_name=text \
    --num_batches= \
    --batch_size= \
    --data_root=your_tf_record_dir \
    --model_name=crnn \
    --eval_type=loop