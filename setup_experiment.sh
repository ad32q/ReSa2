#!/bin/bash

TOKENIZER="model/bert-base-uncased" 

QUERY_FILE1="data/msmarco_passage/raw/queries.dev.small.tsv"
QUERY_FILE2="data/msmarco_passage/raw/train.query.txt"
QUERY_FILE3="data/msmarco_passage/raw/positives.tsv"
PASSAGE_FILE="data/msmarco_passage/raw/corpus.tsv"  

SAVE_QUERY1="output/queries.dev.small.json"
SAVE_QUERY2="output/train.query.json"
SAVE_QUERY3="output/positives.json"

SAVE_PASSAGE="output/corpus_128"  
TRUNCATE_QUERY=32                 
TRUNCATE_PASSAGE=128                
NUM_SPLITS=10                      

# Step 1: Process queries
echo "=== Starting query processing ==="
python tokenize_queries.py \
  --tokenizer_name $TOKENIZER \
  --query_file $QUERY_FILE1 \
  --save_to $SAVE_QUERY1

python tokenize_queries.py \
  --tokenizer_name $TOKENIZER \
  --query_file $QUERY_FILE2 \
  --save_to $SAVE_QUERY2

python tokenize_queries.py \
  --tokenizer_name $TOKENIZER \
  --query_file $QUERY_FILE3 \
  --save_to $SAVE_QUERY3

# Check if query processing was successful
if [ $? -ne 0 ]; then
  echo "Error: Query processing failed!"
  exit 1
fi

# Step 2: Process passages
echo "=== Starting passage processing ==="
python tokenize_passages.py \
  --tokenizer_name $TOKENIZER \
  --file $PASSAGE_FILE \
  --save_to $SAVE_PASSAGE \

# Check if passage processing was successful
if [ $? -ne 0 ]; then
  echo "Error: Passage processing failed!"
  exit 1
fi

echo "=== All processing completed ==="
