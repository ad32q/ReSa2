#!/bin/bash

pd_tbs=16
lr=5e-6
epoch=4
save_steps=50000
q_max_len=32
p_max_len=128
training_mode=oq.nll
train_data_name=dr
train_data_dir=./msmarco_passage/process/train_data/${train_data_name}
output_dir=./msmarco_passage/models/${train_data_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}

python build_train.py \
  --tokenizer_name ./model/bert-base-uncased \
  --negative_file ./data/msmarco_passage/raw/train.negatives.tsv \
  --qrels data/msmarco_passage/raw/qrels.train.tsv \
  --queries data/msmarco_passage/raw/train.query.txt \
  --collection data/msmarco_passage/raw/corpus.tsv \
  --save_to ./msmarco_passage/process/train_data/${train_data_name}

if [ $? -ne 0 ]; then
    echo "build_train.py failed, terminating script."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --training_mode ${training_mode} \
  --output_dir ${output_dir} \
  --model_name_or_path ./model/bert-base-uncased \
  --save_steps ${save_steps} \
  --logging_steps 500 \
  --train_dir ${train_data_dir} \
  --fp16 \
  --per_device_train_batch_size ${pd_tbs} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --dataloader_num_workers 2 \
  --train_n_passages 8 \
  --q_max_len ${q_max_len} \
  --p_max_len ${p_max_len}

if [ $? -ne 0 ]; then
    echo "tevatron.driver.train failed, terminating script."
    exit 1
fi

echo "All steps completed successfully."
