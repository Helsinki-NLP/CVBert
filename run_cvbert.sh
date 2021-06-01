python run_cvbert.py \
    --train_file data/Twitter/tweets_thunberg_nobel2019.labelled.head1000.csv \
    --validation_file data/Twitter/tweets_thunberg_nobel2019.labelled.second1000.csv \
    --model_name_or_path bert-base-uncased \
    --output_dir tmp \
    --line_by_line True \
    --per_device_train_batch_size 10 \
    --num_train_epochs 20 \
    --max_seq_length 500  \

