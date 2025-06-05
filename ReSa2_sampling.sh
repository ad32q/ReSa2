#!/bin/bash
#!/bin/bash

pd_tbs=16
lr=5e-6
epoch=4
q_max_len=32
p_max_len=128

model_name=dr
model_dir=./msmarco_passage/models/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
save_path=./msmarco_passage/results/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
gpu_id=0


# encode corpus and queries
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                         --model_name_or_path ${model_dir} \
                                                         --fp16 \
                                                         --p_max_len 128 \
                                                         --per_device_eval_batch_size 512 \
                                                         --encode_in_path ./output/corpus_128 \
                                                         --encoded_save_path ${save_path}/corpus_128.pt

query_dir=./output
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                       --model_name_or_path ${model_dir} \
                                                       --fp16 \
                                                       --q_max_len 32 \
                                                       --encode_is_qry \
                                                       --per_device_eval_batch_size 512 \
                                                       --encode_in_path ${query_dir}/train.query.json \
                                                       --encoded_save_path ${save_path}/train.query.pt
                                                       
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                       --model_name_or_path ${model_dir} \
                                                       --fp16 \
                                                       --q_max_len 128 \
                                                       --per_device_eval_batch_size 512 \
                                                       --encode_in_path ${query_dir}/positives.json \
                                                       --encoded_save_path ${save_path}/positives.pt

# index and retrieval
index_type=Flat
index_dir=${save_path}/${index_type}

python -m tevatron.faiss_retriever \
          --query_reps ${save_path}/train.query.pt \
          --passage_reps ${save_path}/corpus_128.pt \
          --index_type ${index_type} \
          --batch_size 16 \
          --depth 1000 \
          --save_ranking_file ${index_dir}/train.query.rank.txt \
          --save_index \
          --save_index_dir ${index_dir}


python firstSampling.py --rank_file ${index_dir}/train.query.rank.txt \
                        --qrels_file ./data/msmarco_passage/raw/qrels.train.tsv \
                        --output_file ./nn/s1_output.txt
                        
python secondRetrieve.py \
  --query_vectors ${save_path}/positives.pt \
  --rank_file ./nn/s1_output.txt \
  --corpus_file ${save_path}/corpus_128.pt \
  --output_file nn/r2_output.txt
