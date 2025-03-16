from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from tevatron.preprocessor import MarcoPassageTrainPreProcessor as TrainPreProcessor
import math
import random

def load_ranking(rank_file, relevance, n_sample, depth, tau=0.5):
    """Load data from the ranking file and select negative samples based on probability
    
    Args:
        rank_file (str): Path to the ranking file, format: [query, _, paragraph, _, score, _]
        relevance (dict): Dictionary mapping queries to sets of relevant passages {q: {positive_paragraphs}}
        n_sample (int): Number of negative samples to select
        depth (int): Maximum ranking depth to consider for negative samples
        tau (float): Temperature factor in probability computation, default is 1/4
    
    Yields:
        tuple: (query, list of positive samples, list of sampled negative samples)
    """
    with open(rank_file) as rf:
        lines = iter(rf)
        
        # Initialize first query
        try:
            parts = next(lines).strip().split()
            curr_q, curr_p = parts[0], parts[2]
            curr_score = float(parts[4])
            
            pos_scores = [curr_score] if curr_p in relevance.get(curr_q, set()) else []
            negatives = [] if curr_p in relevance.get(curr_q, set()) else [(curr_p, curr_score)]
        except StopIteration:
            return  # Return immediately if the file is empty

        def _sample_negatives(pos_scores, negatives):
            """Generate sampled negatives based on positive and negative sample scores"""
            # If no positive samples, fallback to random selection
            if not pos_scores:
                candidates = [p for p, _ in negatives[:depth]]
                return random.sample(candidates, min(n_sample, len(candidates)))
            
            # Compute probability weights
            pos_max = max(pos_scores)  # Use the highest positive sample score as a reference
            candidates = negatives[:depth]
            if not candidates:
                return []
            
            # Compute exponential weights
            weights = [math.exp(-(s - pos_max)**2 * tau) for _, s in candidates]
            total = sum(weights)
            
            # Handle case where all weights are zero
            if total <= 1e-8:
                return random.sample([p for p, _ in candidates], min(n_sample, len(candidates)))
            
            # Normalize probabilities
            probs = [w / total for w in weights]
            
            # Weighted sampling without replacement
            sampled = []
            remain_indices = list(range(len(candidates)))
            remain_probs = probs.copy()
            
            for _ in range(min(n_sample, len(candidates))):
                # Roulette wheel selection
                r = random.uniform(0, sum(remain_probs))
                accum = 0
                for i in range(len(remain_probs)):
                    accum += remain_probs[i]
                    if accum >= r:
                        chosen_idx = remain_indices[i]
                        sampled.append(candidates[chosen_idx][0])
                        # Remove selected sample
                        del remain_indices[i]
                        del remain_probs[i]
                        # Renormalize remaining probabilities
                        if remain_probs:
                            sum_remain = sum(remain_probs)
                            remain_probs = [p/sum_remain for p in remain_probs]
                        break
            return sampled

        # Process remaining lines
        while True:
            try:
                parts = next(lines).strip().split()
                q, p = parts[0], parts[2]
                score = float(parts[4])
                
                # Generate results when encountering a new query
                if q != curr_q:
                    sampled = _sample_negatives(pos_scores, negatives)
                    yield curr_q, relevance.get(curr_q, []), sampled
                    
                    # Reset state
                    curr_q = q
                    pos_scores = [score] if p in relevance.get(q, set()) else []
                    negatives = [] if p in relevance.get(q, set()) else [(p, score)]
                else:
                    # Accumulate positive and negative sample scores
                    if p in relevance.get(q, set()):
                        pos_scores.append(score)
                    else:
                        negatives.append((p, score))
                        
            except StopIteration:
                # Process last query
                sampled = _sample_negatives(pos_scores, negatives)
                yield curr_q, relevance.get(curr_q, []), sampled
                return

random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--nn_file', required=True, help='TREC run format, returned by DR_OQ model.')
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--query_variation', required=False, default='None')

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=7)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()

qrel = TrainPreProcessor.read_qrel(args.qrels)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    tokenizer=tokenizer,
    max_length=args.truncate,
    query_variation_file=args.query_variation
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

pbar = tqdm(load_ranking(args.nn_file, qrel, args.n_sample, args.depth))
with Pool() as p:
    for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.json'), 'w')
            pbar.set_description(f'split - {shard_id:02d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()
