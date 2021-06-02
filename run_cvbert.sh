python run_cvbert.py \
    --train_file data/Twitter/head1000.balanced500.shuffled.csv \
    --validation_file data/Twitter/tweets_thunberg_nobel2019.labelled.second1000.csv \
    --tokenizer_name bert-base-uncased \
    --output_dir tmp \
    --line_by_line True \
    --per_device_train_batch_size 20 \
    --num_train_epochs 20 \
    --max_seq_length 500  \
    --learning_rate 1e-10 \

